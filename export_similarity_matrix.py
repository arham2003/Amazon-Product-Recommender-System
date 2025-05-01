import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

print("Loading data...")
# Load the dataset
df = pd.read_csv("amazon.csv")

print("Cleaning data...")
# Clean the data
df = df.dropna()
df = df.drop_duplicates()

# Convert price columns and other numeric columns
df['discounted_price'] = df['discounted_price'].astype(str).str.replace('₹', '').str.replace(',', '').astype(float)
df['actual_price'] = df['actual_price'].astype(str).str.replace('₹', '').str.replace(',', '').astype(float)
df['discount_percentage'] = df['discount_percentage'].astype(str).str.replace('%', '').astype(float)
df['rating'] = pd.to_numeric(df['rating'].astype(str).str.replace('|', ''), errors='coerce')
df['rating_count'] = df['rating_count'].astype(str).str.replace(',', '').astype(int)

# Cleaning and preprocessing text
def clean_text(text):
    # Convert to lowercase
    text = str(text).lower()
    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    # Split text into words and rejoin without stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Apply the clean_text function to the DataFrame columns
df['product_name'] = df['product_name'].apply(clean_text)
df['about_product'] = df['about_product'].apply(clean_text)
df['review_content'] = df['review_content'].apply(clean_text)
df['category'] = df['category'].apply(clean_text)

print("Generating TF-IDF matrix...")
# Create combined text for feature extraction
df['text_corpus'] = (
    df['product_name'].fillna('') + ' ' +
    df['category'].fillna('') + ' ' +
    df['about_product'].fillna('') + ' ' +
    df['review_content'].fillna('')
)
df['text_corpus'] = df['text_corpus'].fillna('')

# Create TF-IDF matrix
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=2, ngram_range=(1, 1))
tfidf_matrix = vectorizer.fit_transform(df['text_corpus'])

print("Computing cosine similarity matrix...")
# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix)

print("Saving cosine similarity matrix...")
# Save the cosine similarity matrix
with open('cosine_similarity_matrix.pkl', 'wb') as f:
    pickle.dump(cosine_sim, f)

print("Building ground truth...")
# Build ground truth based on content similarity and category
def build_ground_truth(df, cosine_sim, similarity_threshold=0.4):
    # Make sure DataFrame is aligned with the similarity matrix
    df_subset = df.iloc[:cosine_sim.shape[0]]
    cosine_matrix_size = cosine_sim.shape[0]
    
    print(f"DataFrame size: {len(df)}, Cosine matrix size: {cosine_matrix_size}")
    print(f"Using {len(df_subset)} products for ground truth generation")
    
    ground_truth = {}
    
    for idx, row in df_subset.iterrows():
        if idx % 100 == 0:
            print(f"Processing product {idx}/{len(df_subset)}")
            
        if idx >= cosine_matrix_size:
            print(f"Warning: Skipping index {idx} as it's out of bounds for similarity matrix")
            continue
        
        product_id = row['product_id']
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

# Generate ground truth
ground_truth = build_ground_truth(df, cosine_sim)

print("Saving ground truth...")
# Save the ground truth
with open('ground_truth.pkl', 'wb') as f:
    pickle.dump(ground_truth, f)

# Calculate some statistics about the ground truth
gt_lengths = [len(v) for v in ground_truth.values()]
print(f"Ground truth stats:")
print(f"  Average # of similar products per product: {np.mean(gt_lengths):.2f}")
print(f"  Median: {np.median(gt_lengths)}, Max: {np.max(gt_lengths)}, Min: {np.min(gt_lengths)}")

print("Done! Similarity matrix and ground truth have been saved.") 