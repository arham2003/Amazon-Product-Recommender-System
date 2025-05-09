# app.py
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import wordninja
import csv
from io import StringIO
import pickle
import os

app = Flask(__name__)


def format_category(text):
    if not isinstance(text, str):
        return text
    return ' '.join(wordninja.split(text)).title()


app.jinja_env.filters['format_category'] = format_category


def process_reviews(row):
    """Extract multiple reviews from comma-separated entries"""
    try:
        # Get users
        users = [u.strip() for u in str(row['user_name']).split(',') if u.strip()]
        n_users = len(users)
        if n_users == 0:
            return []
            
        # Process review IDs
        review_ids = []
        if pd.notna(row['review_id']):
            review_ids = [r.strip() for r in str(row['review_id']).split(',')]
            
        # Process titles
        titles = []
        if pd.notna(row['review_title']):
            titles = [t.strip() for t in str(row['review_title']).split(',')]
            
        # Process contents
        contents = []
        if pd.notna(row['review_content']):
            # Split by commas but be smarter about it since reviews might contain commas
            parts = []
            part = ""
            in_quote = False
            for char in str(row['review_content']):
                if char == '"':
                    in_quote = not in_quote
                elif char == ',' and not in_quote:
                    parts.append(part.strip())
                    part = ""
                    continue
                part += char
            if part:
                parts.append(part.strip())
            contents = parts

        # Ensure all lists have same length by padding
        max_len = max(n_users, len(review_ids), len(titles), len(contents))
        users = users + ['Unknown'] * (max_len - len(users))
        review_ids = review_ids + [''] * (max_len - len(review_ids))
        titles = titles + [''] * (max_len - len(titles))
        contents = contents + [''] * (max_len - len(contents))

        # Create review objects
        return [{
            'user': users[i] if i < len(users) else 'Unknown',
            'review_id': review_ids[i] if i < len(review_ids) else '',
            'title': titles[i] if i < len(titles) else '',
            'content': contents[i] if i < len(contents) else ''
        } for i in range(max_len)]

    except Exception as e:
        print(f"Error processing reviews: {e}")
        return []


def build_ground_truth(df, cosine_sim, similarity_threshold=0.4):
    """
    Build ground truth dictionary based on product content similarity and category
    """
    # Make sure DataFrame is aligned with similarity matrix
    df_subset = df.iloc[:cosine_sim.shape[0]]
    cosine_matrix_size = cosine_sim.shape[0]
    
    print(f"DataFrame size: {len(df)}, Cosine matrix size: {cosine_matrix_size}")
    print(f"Using {len(df_subset)} products for ground truth generation")
    
    ground_truth = {}
    
    for idx, row in df_subset.iterrows():
        product_id = row['product_id']
        product_category = row['category']
        product_name = row['product_name']
        
        # Get primary content-based similar indices excluding exact match
        similar_indices = [
            i for i, score in enumerate(cosine_sim[idx])
            if i != idx and score > similarity_threshold
        ]
        
        # Filter out products with identical names (exact duplicates)
        similar_product_ids = []
        for i in similar_indices:
            other_name = df_subset.iloc[i]['product_name']
            if other_name.strip().lower() != product_name.strip().lower():
                similar_product_ids.append(df_subset.iloc[i]['product_id'])
        
        # If not enough similar items, use category-based fallback
        if len(similar_product_ids) < 3:
            category = row.get('category', None)
            if category:
                fallback_products = df_subset[
                    (df_subset['category'] == category) &
                    (df_subset['product_id'] != product_id)
                ]['product_id'].tolist()[:10]  # Limit to 10 fallback products
                
                # Add fallback products until at least 5 similar items are reached
                for pid in fallback_products:
                    if pid not in similar_product_ids:
                        similar_product_ids.append(pid)
                    if len(similar_product_ids) >= 5:
                        break
        
        # Assign to ground truth
        ground_truth[product_id] = similar_product_ids
    
    return ground_truth


def average_precision_at_k(recommended, ground_truth, k):
    """Calculate average precision at k"""
    if not ground_truth:
        return 0.0
    recommended_k = recommended[:k]
    score = 0.0
    hits = 0
    for i, item in enumerate(recommended_k):
        if item in ground_truth:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(ground_truth), k)


def load_data():
    df = pd.read_csv("amazon.csv")
    df = df.dropna().drop_duplicates()

    df['discount_percentage'] = df['discount_percentage'].astype(str).str.replace('%', '').astype(float)
    df['rating'] = pd.to_numeric(df['rating'].astype(str).str.replace('|', ''), errors='coerce')
    df['rating_count'] = df['rating_count'].astype(str).str.replace(',', '').astype(int)

    # Clean text data
    stop_words = set(stopwords.words('english'))

    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = ' '.join([word for word in text.split() if word not in stop_words])
        return text

    text_columns = ['product_name', 'category', 'about_product']
    for col in text_columns:
        df[col] = df[col].apply(clean_text)

    # Process reviews
    df['reviews'] = df.apply(process_reviews, axis=1)
    
    # Create combined text for feature extraction
    df['text_corpus'] = (
        df['product_name'].fillna('') + ' ' +
        df['category'].fillna('') + ' ' +
        df['about_product'].fillna('')
    )
    
    # Load precomputed similarity matrix or compute if not available
    cosine_sim_path = "cosine_similarity_matrix.pkl"
    ground_truth_path = "ground_truth.pkl"
    
    # Load or compute similarity matrix
    if os.path.exists(cosine_sim_path):
        print("Loading precomputed similarity matrix...")
        with open(cosine_sim_path, 'rb') as f:
            cosine_sim = pickle.load(f)
    else:
        # Backup: compute similarity matrix if precomputed one isn't available
        print("Precomputed similarity matrix not found. Computing now...")
        vectorizer = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=2)
        tfidf_matrix = vectorizer.fit_transform(df['text_corpus'].fillna(''))
        cosine_sim = cosine_similarity(tfidf_matrix)
        
        # Save for next time
        with open(cosine_sim_path, 'wb') as f:
            pickle.dump(cosine_sim, f)
    
    # Ensure DataFrame is aligned with similarity matrix
    df = df.iloc[:cosine_sim.shape[0]].reset_index(drop=True)
    print(f"DataFrame size after alignment: {len(df)}, Cosine matrix size: {cosine_sim.shape[0]}")
    
    # Load or compute ground truth
    if os.path.exists(ground_truth_path):
        print("Loading precomputed ground truth...")
        with open(ground_truth_path, 'rb') as f:
            ground_truth = pickle.load(f)
    else:
        print("Precomputed ground truth not found. Computing now...")
        ground_truth = build_ground_truth(df, cosine_sim)
        
        # Save for next time
        with open(ground_truth_path, 'wb') as f:
            pickle.dump(ground_truth, f)
    
    return df, cosine_sim, ground_truth


df, cosine_sim, ground_truth = load_data()


def get_recommendations(product_id, num=5):
    try:
        # Get the index of the product
        idx = df.index[df['product_id'] == product_id].tolist()[0]
        
        # Verify idx is within bounds of cosine_sim
        if idx >= cosine_sim.shape[0]:
            print(f"Warning: Product index {idx} is out of bounds for similarity matrix of size {cosine_sim.shape[0]}")
            return pd.DataFrame()
        
        # Get similarity scores for this product with all other products
        sim_scores = list(enumerate(cosine_sim[idx]))
        
        # Sort products by similarity score
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N most similar products (excluding the product itself)
        sim_scores = sim_scores[1:num+1]
        sim_indices = [i[0] for i in sim_scores]
        
        # Get the recommended product IDs
        recommended_product_ids = df.iloc[sim_indices]['product_id'].tolist()
        
        # Calculate MAP using the pre-computed ground truth
        gt_for_product = ground_truth.get(product_id, [])
        map_score = average_precision_at_k(recommended_product_ids, gt_for_product, k=num)
        
        # Get recommendations dataframe
        recommendations = df.iloc[sim_indices].copy()
        
        # Add similarity scores and map value to recommendations
        recommendations['similarity'] = [score for _, score in sim_scores]
        recommendations['map'] = map_score
        recommendations = recommendations[['product_id', 'product_name', 'rating', 'img_link', 'discounted_price', 'actual_price', 'similarity', 'map']]
        
        return recommendations
    except Exception as e:
        print(f"Error getting recommendations: {e}")
        return pd.DataFrame()


@app.route('/')
def home():
    query = request.args.get('query', '').strip()
    featured_products = df.sample(12)

    if query:
        search_results = df[df['product_name'].str.contains(query, case=False)]
    else:
        search_results = pd.DataFrame()

    return render_template('index.html',
                           search_results=search_results,
                           featured_products=featured_products,
                           query=query)


@app.route('/product/<product_id>')
def product_detail(product_id):
    try:
        product = df[df['product_id'] == product_id].iloc[0]
        recommendations = get_recommendations(product_id)
        reviews = product['reviews'][:5]  # Show first 5 reviews on product page
        total_reviews = len(product['reviews'])
        return render_template('product.html',
                            product=product,
                            recommendations=recommendations,
                            reviews=reviews,
                            total_reviews=total_reviews)
    except Exception as e:
        print(f"Error rendering product page: {e}")
        return "Product not found or error loading product details", 404


@app.route('/product/<product_id>/reviews')
def product_reviews(product_id):
    try:
        product = df[df['product_id'] == product_id].iloc[0]
        return render_template('reviews.html',
                            product=product,
                            reviews=product['reviews'])
    except Exception as e:
        print(f"Error rendering reviews page: {e}")
        return "Product not found or error loading reviews", 404


if __name__ == '__main__':
    app.run(debug=True)