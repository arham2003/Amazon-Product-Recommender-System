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

app = Flask(__name__)


def format_category(text):
    if not isinstance(text, str):
        return text
    return ' '.join(wordninja.split(text)).title()


app.jinja_env.filters['format_category'] = format_category


def split_columns(text, n):
    """Split text into exactly n parts using n-1 splits"""
    if pd.isna(text):
        return [''] * n
    parts = []
    remaining = str(text).strip()
    for _ in range(n - 1):
        if ',' in remaining:
            idx = remaining.find(',')
            parts.append(remaining[:idx].strip())
            remaining = remaining[idx + 1:].strip()
        else:
            break
    parts.append(remaining.strip())
    return parts[:n] + [''] * (n - len(parts))


def process_reviews(row):
    try:
        # Get base user count
        users = [u.strip() for u in str(row['user_name']).split(',') if u.strip()]
        n_users = len(users)
        if n_users == 0:
            return []

        # Process other columns to match user count
        review_ids = split_columns(row['review_id'], n_users)
        titles = split_columns(row['review_title'], n_users)
        contents = split_columns(row['review_content'], n_users)

        # Create review objects
        return [{
            'user': users[i] if i < len(users) else 'Unknown',
            'review_id': review_ids[i] if i < len(review_ids) else '',
            'title': titles[i] if i < len(titles) else '',
            'content': contents[i] if i < len(contents) else ''
        } for i in range(n_users)]

    except Exception as e:
        print(f"Error processing reviews: {e}")
        return []


def load_data():
    df = pd.read_csv("amazon.csv")
    df = df.dropna().drop_duplicates()

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

    # Prepare recommendation features
    df['combined_features'] = df['product_name'] + ' ' + df['category'] + ' ' + df['about_product']
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    df['cosine_sim'] = list(cosine_similarity(tfidf_matrix))

    return df


df = load_data()


def get_recommendations(product_id, num=5):
    try:
        idx = df.index[df['product_id'] == product_id].tolist()[0]
        sim_scores = list(enumerate(df.iloc[idx]['cosine_sim']))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_indices = [i[0] for i in sim_scores[1:num + 1]]
        return df.iloc[sim_indices]
    except:
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
    product = df[df['product_id'] == product_id].iloc[0]
    recommendations = get_recommendations(product_id)
    reviews = product['reviews'][:5]
    total_reviews = len(product['reviews'])
    return render_template('product.html',
                           product=product,
                           recommendations=recommendations,
                           reviews=reviews,
                           total_reviews=total_reviews)


@app.route('/product/<product_id>/reviews')
def product_reviews(product_id):
    product = df[df['product_id'] == product_id].iloc[0]
    return render_template('reviews.html',
                           product=product,
                           reviews=product['reviews'])


if __name__ == '__main__':
    app.run(debug=True)