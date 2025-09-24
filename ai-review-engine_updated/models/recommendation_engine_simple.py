"""
Simplified Recommendation Engine for User-Friendly App
Lightweight version without complex database dependencies
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter
import logging

logger = logging.getLogger(__name__)

class RecommendationEngine:
    """Simplified recommendation engine for user-friendly app"""
    
    def __init__(self):
        """Initialize the recommendation engine with basic functionality"""
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.product_features = {}
        self.similarity_matrix = None
        self.product_ratings = {}
        logger.info("Simplified RecommendationEngine initialized")
    
    def fit(self, df: pd.DataFrame):
        """
        Fit the recommendation engine on data
        
        Args:
            df: DataFrame with product reviews
        """
        if df is None or df.empty:
            logger.warning("Empty dataframe provided to recommendation engine")
            return
        
        try:
            # Calculate average ratings per product
            if 'product' in df.columns and 'rating' in df.columns:
                self.product_ratings = df.groupby('product')['rating'].mean().to_dict()
            
            # Calculate sentiment scores per product
            if 'product' in df.columns and 'sentiment_label' in df.columns:
                sentiment_scores = df.groupby('product')['sentiment_label'].apply(
                    lambda x: (x == 'positive').mean() if len(x) > 0 else 0.5
                ).to_dict()
                
                for product, score in sentiment_scores.items():
                    if product not in self.product_features:
                        self.product_features[product] = {}
                    self.product_features[product]['sentiment_score'] = score
            
            # Create text features if review text available
            if 'product' in df.columns and 'review_text' in df.columns:
                product_texts = df.groupby('product')['review_text'].apply(
                    lambda x: ' '.join(x.astype(str)[:10])  # Use first 10 reviews
                ).to_dict()
                
                if product_texts:
                    texts = list(product_texts.values())
                    products = list(product_texts.keys())
                    
                    # Create TF-IDF features
                    tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
                    self.similarity_matrix = cosine_similarity(tfidf_matrix)
                    
                    # Store product indices
                    self.product_to_idx = {product: idx for idx, product in enumerate(products)}
                    self.idx_to_product = {idx: product for idx, product in enumerate(products)}
            
            logger.info(f"Recommendation engine fitted on {len(df)} reviews")
            
        except Exception as e:
            logger.error(f"Error fitting recommendation engine: {e}")
    
    def get_recommendations(self, user_preferences: Dict[str, Any] = None, 
                          num_recommendations: int = 5) -> List[Dict[str, Any]]:
        """
        Get product recommendations
        
        Args:
            user_preferences: User preference dictionary
            num_recommendations: Number of recommendations to return
            
        Returns:
            List of recommendation dictionaries
        """
        try:
            if not self.product_ratings:
                return self._get_default_recommendations()
            
            # Sort products by rating and sentiment
            product_scores = []
            for product, rating in self.product_ratings.items():
                sentiment_score = self.product_features.get(product, {}).get('sentiment_score', 0.5)
                combined_score = (rating * 0.6) + (sentiment_score * 4.0 * 0.4)  # Normalize sentiment to 0-4 scale
                
                product_scores.append({
                    'product': product,
                    'rating': rating,
                    'sentiment_score': sentiment_score,
                    'combined_score': combined_score
                })
            
            # Sort by combined score
            product_scores.sort(key=lambda x: x['combined_score'], reverse=True)
            
            # Return top recommendations
            recommendations = []
            for item in product_scores[:num_recommendations]:
                recommendations.append({
                    'product_name': item['product'],
                    'score': item['combined_score'],
                    'rating': item['rating'],
                    'sentiment_rate': item['sentiment_score'] * 100,
                    'reason': f"High rating ({item['rating']:.1f}/5) with {item['sentiment_score']*100:.0f}% positive sentiment"
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return self._get_default_recommendations()
    
    def get_similar_products(self, product_name: str, num_similar: int = 5) -> List[Dict[str, Any]]:
        """
        Get products similar to the given product
        
        Args:
            product_name: Name of the reference product
            num_similar: Number of similar products to return
            
        Returns:
            List of similar product dictionaries
        """
        if not hasattr(self, 'product_to_idx') or product_name not in self.product_to_idx:
            return []
        
        try:
            product_idx = self.product_to_idx[product_name]
            similarities = self.similarity_matrix[product_idx]
            
            # Get indices of most similar products (excluding self)
            similar_indices = np.argsort(similarities)[::-1][1:num_similar+1]
            
            similar_products = []
            for idx in similar_indices:
                similar_product = self.idx_to_product[idx]
                similarity_score = similarities[idx]
                rating = self.product_ratings.get(similar_product, 0)
                
                similar_products.append({
                    'product_name': similar_product,
                    'similarity_score': similarity_score,
                    'rating': rating,
                    'reason': f"Similar content and reviews (similarity: {similarity_score:.2f})"
                })
            
            return similar_products
            
        except Exception as e:
            logger.error(f"Error getting similar products: {e}")
            return []
    
    def get_sentiment_based_recommendations(self, min_polarity: float = 0.6, n: int = 5) -> List[Dict[str, Any]]:
        """
        Get recommendations based on sentiment analysis
        
        Args:
            min_polarity: Minimum sentiment polarity
            n: Number of recommendations
            
        Returns:
            List of sentiment-based recommendations
        """
        try:
            recommendations = []
            
            for product, features in self.product_features.items():
                sentiment_score = features.get('sentiment_score', 0)
                if sentiment_score >= min_polarity:
                    rating = self.product_ratings.get(product, 0)
                    
                    recommendations.append({
                        'product_name': product,
                        'score': sentiment_score,
                        'rating': rating,
                        'positive_reviews': f"{sentiment_score*100:.0f}%",
                        'reason': f"High positive sentiment ({sentiment_score*100:.0f}%)"
                    })
            
            # Sort by sentiment score
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            return recommendations[:n]
            
        except Exception as e:
            logger.error(f"Error getting sentiment-based recommendations: {e}")
            return []
    
    def _get_default_recommendations(self) -> List[Dict[str, Any]]:
        """Get default recommendations when no data is available"""
        return [
            {
                'product_name': 'iPhone 15 Pro',
                'score': 4.5,
                'rating': 4.5,
                'sentiment_rate': 85,
                'reason': 'Popular choice with excellent reviews'
            },
            {
                'product_name': 'Samsung Galaxy S24 Ultra',
                'score': 4.4,
                'rating': 4.4,
                'sentiment_rate': 82,
                'reason': 'High-end Android device with great features'
            },
            {
                'product_name': 'Google Pixel 8 Pro',
                'score': 4.3,
                'rating': 4.3,
                'sentiment_rate': 80,
                'reason': 'Excellent camera and pure Android experience'
            }
        ]