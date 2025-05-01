# app.py
from flask import Flask, render_template, request, jsonify
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
import threading

app = Flask(__name__)

# Global variables to store data
df = None
cosine_sim = None
ground_truth = None
loading_completed = False
loading_error = None


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


def load_data_async():
    """Load data in a background thread"""
    global df, cosine_sim, ground_truth, loading_completed, loading_error
    
    try:
        # Load basic data first for quick startup
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
        
        # Try to load pre-computed matrices first
        cosine_sim_path = "cosine_similarity_matrix.pkl"
        ground_truth_path = "ground_truth.pkl"
        
        if os.path.exists(cosine_sim_path) and os.path.exists(ground_truth_path):
            print("Loading precomputed similarity matrix and ground truth...")
            with open(cosine_sim_path, 'rb') as f:
                cosine_sim = pickle.load(f)
            with open(ground_truth_path, 'rb') as f:
                ground_truth = pickle.load(f)
                
            # Ensure DataFrame is aligned with similarity matrix
            df = df.iloc[:cosine_sim.shape[0]].reset_index(drop=True)
            print(f"DataFrame size after alignment: {len(df)}, Cosine matrix size: {cosine_sim.shape[0]}")
        else:
            # We'll compute a smaller similarity matrix for demo purposes
            print("Precomputed files not found. Creating a smaller demo version...")
            # Use only first 1000 items for quick startup
            small_df = df.head(1000).copy()
            df = small_df
            vectorizer = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=2)
            tfidf_matrix = vectorizer.fit_transform(small_df['text_corpus'].fillna(''))
            cosine_sim = cosine_similarity(tfidf_matrix)
            
            # Build a smaller ground truth
            ground_truth = build_ground_truth(small_df, cosine_sim)
            
            # Save for next time
            with open(cosine_sim_path, 'wb') as f:
                pickle.dump(cosine_sim, f)
            with open(ground_truth_path, 'wb') as f:
                pickle.dump(ground_truth, f)
            
        loading_completed = True
    except Exception as e:
        loading_error = str(e)
        print(f"Error loading data: {e}")
        # Still mark as completed to allow the app to function
        loading_completed = True


# Start loading data in background thread
threading.Thread(target=load_data_async, daemon=True).start()


def get_recommendations(product_id, num=5):
    """Get recommendations for a product"""
    global df, cosine_sim, ground_truth
    
    # If data isn't loaded yet, return empty DataFrame
    if not loading_completed or df is None or cosine_sim is None:
        return pd.DataFrame()
    
    try:
        # Get the index of the product
        product_indices = df.index[df['product_id'] == product_id].tolist()
        if not product_indices:
            return pd.DataFrame()
            
        idx = product_indices[0]
        
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
    """Home page and healthcheck endpoint"""
    # For healthcheck when the app is starting up
    if request.headers.get('User-Agent', '').startswith('Railway-Health'):
        return jsonify({"status": "ok"})
    
    # For regular users
    if not loading_completed:
        return "Application is starting up. Please wait a moment and refresh the page."
        
    if loading_error:
        return f"Error loading data: {loading_error}"
        
    query = request.args.get('query', '').strip()
    
    try:
        featured_products = df.sample(min(12, len(df)))
    except Exception as e:
        featured_products = df.head(12) if df is not None and len(df) > 0 else pd.DataFrame()

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
    """Product detail page"""
    if not loading_completed:
        return "Application is starting up. Please wait a moment and refresh the page."
        
    try:
        product_rows = df[df['product_id'] == product_id]
        if product_rows.empty:
            return "Product not found", 404
            
        product = product_rows.iloc[0]
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
        return f"Error loading product: {str(e)}", 500


@app.route('/product/<product_id>/reviews')
def product_reviews(product_id):
    """Product reviews page"""
    if not loading_completed:
        return "Application is starting up. Please wait a moment and refresh the page."
        
    try:
        product_rows = df[df['product_id'] == product_id]
        if product_rows.empty:
            return "Product not found", 404
            
        product = product_rows.iloc[0]
    return render_template('reviews.html',
                           product=product,
                           reviews=product['reviews'])
    except Exception as e:
        print(f"Error rendering reviews page: {e}")
        return f"Error loading reviews: {str(e)}", 500


@app.route('/health')
def health_check():
    """Explicit health check endpoint"""
    return jsonify({"status": "ok"})


if __name__ == '__main__':
    app.run(debug=True)