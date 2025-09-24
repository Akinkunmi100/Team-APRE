"""
Sample script to analyze the integrated review data
"""

import pandas as pd
import streamlit as st
from utils.data_preprocessing import DataPreprocessor
from models.absa_model import ABSASentimentAnalyzer

# Load the integrated data
df = pd.read_csv('data/processed/reviews_integrated.csv')

print(f"Loaded {len(df)} reviews")
print(f"Columns: {list(df.columns)}")

# Initialize components
preprocessor = DataPreprocessor()
analyzer = ABSASentimentAnalyzer()

# Sample analysis
sample = df.head(10)
for idx, row in sample.iterrows():
    print(f"\nReview: {row['review_text'][:100]}...")
    print(f"Product: {row['product_name']}")
    print(f"Rating: {row['rating']}")
    
    # Analyze sentiment
    result = analyzer.analyze_single(row['review_text'])
    print(f"Sentiment: {result['sentiment']} (confidence: {result.get('confidence', 0):.2f})")
