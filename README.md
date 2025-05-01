# Amazon Product Recommendation System

This is a Flask-based web application that provides content-based product recommendations using Amazon product data.

## Features

- Product browsing with dynamic recommendations
- Product detail pages with customer reviews
- Review pages showing all reviews for a product
- Advanced content-based recommendation system with MAP metrics
- Dynamic extraction of reviews from comma-separated data

## Setup Instructions

1. **Install Dependencies**
   ```
   pip install flask pandas numpy nltk scikit-learn wordninja textblob
   ```

2. **Download NLTK Components**
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

3. **Data Preparation**
   - Place your `amazon.csv` file in the root directory
   - Generate the similarity matrix and ground truth by running:
     ```
     python export_similarity_matrix.py
     ```
     This will create two files:
     - `cosine_similarity_matrix.pkl`: Precomputed similarity matrix
     - `ground_truth.pkl`: Precomputed recommendation ground truth

4. **Run the Application**
   ```
   python app.py
   ```
   The application will be available at `http://127.0.0.1:5000/`

## File Structure

- `app.py`: Main Flask application
- `export_similarity_matrix.py`: Script to generate the cosine similarity matrix and ground truth
- `cosine_similarity_matrix.pkl`: Precomputed similarity matrix (generated)
- `ground_truth.pkl`: Precomputed recommendation ground truth (generated)
- `templates/`: HTML templates for the web interface
  - `index.html`: Home page
  - `product.html`: Product detail page with recommendations and MAP metrics
  - `reviews.html`: Page showing all reviews for a product
- `amazon.csv`: Dataset with product information and reviews

## How It Works

### Recommendation System

The recommendation system uses the following approaches:

1. **Content-Based Filtering**:
   - Uses TF-IDF to convert product information into numerical vectors
   - Calculates cosine similarity between these vectors to find similar products
   - Products with higher similarity scores are recommended first

2. **Ground Truth Generation**:
   - For each product, we identify truly relevant recommendations
   - Uses content similarity with a threshold value
   - Falls back to category-based recommendations when needed

3. **Evaluation Metrics**:
   - Mean Average Precision (MAP) is calculated for each set of recommendations
   - This metric is displayed on the product page to indicate recommendation quality

### Review Extraction

Reviews are extracted dynamically from comma-separated entries in the dataset, properly handling:
- Multiple users per product
- Multiple review titles
- Multiple review contents
- Commas within review content (using a smart parsing strategy) 