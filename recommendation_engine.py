"""
Advanced Recommendation Engine for Product Suggestions
Implements collaborative filtering, content-based filtering, and hybrid approaches
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None
from collections import defaultdict, Counter
import logging
from datetime import datetime, timedelta
import pickle
import redis
import json
import hashlib

from database.models import db_manager, Product, Review, User, UserActivity

logger = logging.getLogger(__name__)

if TORCH_AVAILABLE:
    class CollaborativeFilteringModel(nn.Module):
        """Neural collaborative filtering model"""
        
        def __init__(self, n_users, n_items, embedding_dim=50, hidden_layers=[64, 32]):
            super().__init__()
            self.user_embedding = nn.Embedding(n_users, embedding_dim)
            self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
            # MLP layers
            self.fc_layers = nn.ModuleList()
            input_dim = embedding_dim * 2
            for hidden_dim in hidden_layers:
                self.fc_layers.append(nn.Linear(input_dim, hidden_dim))
                input_dim = hidden_dim
            
            self.output_layer = nn.Linear(hidden_layers[-1], 1)
            self.dropout = nn.Dropout(0.2)
            
        def forward(self, user_ids, item_ids):
            user_embeds = self.user_embedding(user_ids)
            item_embeds = self.item_embedding(item_ids)
        
            # Concatenate embeddings
            x = torch.cat([user_embeds, item_embeds], dim=-1)
            
            # Pass through MLP
            for layer in self.fc_layers:
                x = F.relu(layer(x))
                x = self.dropout(x)
            
            output = torch.sigmoid(self.output_layer(x))
            return output.squeeze()
else:
    # Dummy class when torch is not available
    class CollaborativeFilteringModel:
        def __init__(self, n_users, n_items, embedding_dim=50, hidden_layers=[64, 32]):
            pass
        def forward(self, user_ids, item_ids):
            return None

class RecommendationEngine:
    """Main recommendation engine combining multiple strategies"""
    
    def __init__(self):
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        except:
            self.redis_client = None
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            self.sentence_model = None
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.collab_model = None
        self.product_embeddings = {}
        self.user_profiles = {}
        self.popularity_scores = {}
        self.cache_ttl = 3600  # 1 hour cache
        
        # Load or initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize or load pre-trained models"""
        try:
            # Try to load existing models
            self.load_models()
            logger.info("Loaded existing recommendation models")
        except:
            logger.info("Initializing new recommendation models")
            self.update_models()
    
    def update_models(self):
        """Update all recommendation models with latest data"""
        session = db_manager.get_session()
        
        try:
            # Get all products and reviews
            products = session.query(Product).all()
            reviews = session.query(Review).all()
            
            # Update product embeddings
            self._update_product_embeddings(products, reviews)
            
            # Update popularity scores
            self._calculate_popularity_scores(products, reviews)
            
            # Update collaborative filtering model
            self._train_collaborative_model(reviews)
            
            # Update user profiles
            self._build_user_profiles(reviews)
            
            # Save models
            self.save_models()
            
            logger.info("Successfully updated all recommendation models")
            
        finally:
            session.close()
    
    def _update_product_embeddings(self, products: List[Product], reviews: List[Review]):
        """Create semantic embeddings for products"""
        
        # Combine product info and reviews for each product
        product_texts = {}
        for product in products:
            # Base product text
            text_parts = [
                product.name or "",
                product.brand or "",
                product.description or "",
                f"Price: {product.price}" if product.price else "",
                f"Rating: {product.average_rating}" if product.average_rating else ""
            ]
            
            # Add product specifications
            if product.specifications:
                specs = json.loads(product.specifications) if isinstance(product.specifications, str) else product.specifications
                for key, value in specs.items():
                    text_parts.append(f"{key}: {value}")
            
            product_texts[product.id] = " ".join(filter(None, text_parts))
        
        # Add review content
        review_texts = defaultdict(list)
        for review in reviews:
            if review.review_text:
                review_texts[review.product_id].append(review.review_text)
        
        # Combine and create embeddings
        for product_id, base_text in product_texts.items():
            review_summary = " ".join(review_texts.get(product_id, [])[:10])  # Use top 10 reviews
            full_text = f"{base_text} {review_summary}"
            
            # Create embedding
            if self.sentence_model:
                embedding = self.sentence_model.encode(full_text)
                self.product_embeddings[product_id] = embedding
            else:
                # Fallback to TF-IDF if sentence transformers not available
                self.product_embeddings[product_id] = full_text
    
    def _calculate_popularity_scores(self, products: List[Product], reviews: List[Review]):
        """Calculate popularity scores based on multiple factors"""
        
        # Count reviews per product
        review_counts = Counter(r.product_id for r in reviews)
        
        # Recent review counts (last 30 days)
        recent_date = datetime.now() - timedelta(days=30)
        recent_reviews = Counter(
            r.product_id for r in reviews 
            if r.created_at and r.created_at > recent_date
        )
        
        # Calculate scores
        for product in products:
            pid = product.id
            
            # Factors for popularity
            total_reviews = review_counts.get(pid, 0)
            recent_review_count = recent_reviews.get(pid, 0)
            avg_rating = product.average_rating or 0
            
            # Weighted popularity score
            popularity = (
                total_reviews * 0.3 +
                recent_review_count * 0.4 +
                avg_rating * 20 * 0.3  # Scale rating to similar range
            )
            
            # Apply logarithmic dampening for very popular items
            if popularity > 100:
                popularity = 100 + np.log(popularity - 99)
            
            self.popularity_scores[pid] = popularity
    
    def _train_collaborative_model(self, reviews: List[Review]):
        """Train neural collaborative filtering model"""
        
        # Prepare data
        user_ids = []
        product_ids = []
        ratings = []
        
        user_map = {}
        product_map = {}
        
        for review in reviews:
            if review.user_id and review.rating:
                # Map to continuous IDs
                if review.user_id not in user_map:
                    user_map[review.user_id] = len(user_map)
                if review.product_id not in product_map:
                    product_map[review.product_id] = len(product_map)
                
                user_ids.append(user_map[review.user_id])
                product_ids.append(product_map[review.product_id])
                ratings.append(review.rating / 5.0)  # Normalize to 0-1
        
        if len(user_ids) > 100 and TORCH_AVAILABLE:  # Only train if we have enough data and torch is available
            # Create and train model
            n_users = len(user_map)
            n_items = len(product_map)
            
            self.collab_model = CollaborativeFilteringModel(n_users, n_items)
            self.user_map = user_map
            self.product_map = product_map
            
            # Simple training (in production, use proper training loop)
            optimizer = torch.optim.Adam(self.collab_model.parameters(), lr=0.001)
            
            user_tensor = torch.LongTensor(user_ids)
            item_tensor = torch.LongTensor(product_ids)
            rating_tensor = torch.FloatTensor(ratings)
            
            for epoch in range(10):
                self.collab_model.train()
                predictions = self.collab_model(user_tensor, item_tensor)
                loss = F.mse_loss(predictions, rating_tensor)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            logger.info(f"Trained collaborative model with {n_users} users and {n_items} items")
    
    def _build_user_profiles(self, reviews: List[Review]):
        """Build user preference profiles"""
        
        user_preferences = defaultdict(lambda: {
            'categories': Counter(),
            'brands': Counter(),
            'price_range': [],
            'avg_rating_given': [],
            'sentiment_preference': Counter(),
            'review_count': 0
        })
        
        for review in reviews:
            if review.user_id:
                profile = user_preferences[review.user_id]
                
                # Get product info
                session = db_manager.get_session()
                product = session.query(Product).filter_by(id=review.product_id).first()
                session.close()
                
                if product:
                    profile['categories'][product.category] += 1
                    profile['brands'][product.brand] += 1
                    if product.price:
                        profile['price_range'].append(product.price)
                
                if review.rating:
                    profile['avg_rating_given'].append(review.rating)
                
                if review.sentiment:
                    profile['sentiment_preference'][review.sentiment] += 1
                
                profile['review_count'] += 1
        
        # Process profiles
        for user_id, profile in user_preferences.items():
            if profile['price_range']:
                profile['avg_price'] = np.mean(profile['price_range'])
                profile['price_std'] = np.std(profile['price_range'])
            
            if profile['avg_rating_given']:
                profile['avg_rating_given'] = np.mean(profile['avg_rating_given'])
            
            self.user_profiles[user_id] = profile
    
    def get_recommendations(
        self,
        product_id: Optional[int] = None,
        user_id: Optional[int] = None,
        search_query: Optional[str] = None,
        n_recommendations: int = 10,
        strategy: str = 'hybrid'
    ) -> List[Dict[str, Any]]:
        """
        Get product recommendations based on various strategies
        
        Args:
            product_id: Base product for content-based recommendations
            user_id: User for personalized recommendations
            search_query: Search query for query-based recommendations
            n_recommendations: Number of recommendations to return
            strategy: 'content', 'collaborative', 'popular', 'hybrid'
        
        Returns:
            List of recommended products with scores
        """
        
        # Check cache
        cache_key = f"recommendations:{product_id}:{user_id}:{search_query}:{strategy}"
        if self.redis_client:
            try:
                cached = self.redis_client.get(cache_key)
                if cached:
                    return json.loads(cached)
            except:
                pass
        
        recommendations = []
        
        if strategy == 'content' or strategy == 'hybrid':
            if product_id:
                content_recs = self._get_content_based_recommendations(product_id, n_recommendations)
                recommendations.extend(content_recs)
        
        if strategy == 'collaborative' or strategy == 'hybrid':
            if user_id:
                collab_recs = self._get_collaborative_recommendations(user_id, n_recommendations)
                recommendations.extend(collab_recs)
        
        if strategy == 'popular' or (strategy == 'hybrid' and not recommendations):
            popular_recs = self._get_popular_recommendations(n_recommendations)
            recommendations.extend(popular_recs)
        
        if search_query:
            query_recs = self._get_query_based_recommendations(search_query, n_recommendations)
            recommendations.extend(query_recs)
        
        # Combine and rank recommendations
        final_recommendations = self._combine_recommendations(recommendations, n_recommendations)
        
        # Add diversity
        final_recommendations = self._diversify_recommendations(final_recommendations)
        
        # Cache results
        self.redis_client.setex(
            cache_key,
            self.cache_ttl,
            json.dumps(final_recommendations)
        )
        
        return final_recommendations
    
    def _get_content_based_recommendations(self, product_id: int, n: int) -> List[Dict]:
        """Get similar products based on content similarity"""
        
        if product_id not in self.product_embeddings:
            return []
        
        base_embedding = self.product_embeddings[product_id]
        similarities = {}
        
        for pid, embedding in self.product_embeddings.items():
            if pid != product_id:
                similarity = cosine_similarity(
                    base_embedding.reshape(1, -1),
                    embedding.reshape(1, -1)
                )[0][0]
                similarities[pid] = similarity
        
        # Get top N similar products
        top_products = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:n]
        
        recommendations = []
        session = db_manager.get_session()
        
        for pid, score in top_products:
            product = session.query(Product).filter_by(id=pid).first()
            if product:
                recommendations.append({
                    'product_id': pid,
                    'product_name': product.name,
                    'brand': product.brand,
                    'price': product.price,
                    'rating': product.average_rating,
                    'score': float(score),
                    'reason': 'Similar to your viewed product',
                    'method': 'content'
                })
        
        session.close()
        return recommendations

    def get_aspect_based_recommendations(self, aspect: str, sentiment: str, n: int) -> List[Dict]:
        """
        Get recommendations based on specific aspect and sentiment

        Args:
            aspect: The aspect to focus on (e.g., 'camera', 'battery')
            sentiment: The sentiment type ('positive', 'negative', 'neutral')
            n: Number of recommendations to return

        Returns:
            List of recommended products
        """
        # For now, return popular recommendations as a fallback
        # In a full implementation, this would filter by aspect sentiment
        return self._get_popular_recommendations(n)
    
    def _get_collaborative_recommendations(self, user_id: int, n: int) -> List[Dict]:
        """Get recommendations based on collaborative filtering"""
        
        if not self.collab_model or user_id not in self.user_map:
            return []
        
        user_idx = self.user_map[user_id]
        recommendations = []
        
        # Get predictions for all products
        product_scores = {}
        
        for product_id, product_idx in self.product_map.items():
            with torch.no_grad():
                user_tensor = torch.LongTensor([user_idx])
                item_tensor = torch.LongTensor([product_idx])
                score = self.collab_model(user_tensor, item_tensor).item()
                product_scores[product_id] = score
        
        # Get top N products
        top_products = sorted(product_scores.items(), key=lambda x: x[1], reverse=True)[:n]
        
        session = db_manager.get_session()
        
        for pid, score in top_products:
            product = session.query(Product).filter_by(id=pid).first()
            if product:
                recommendations.append({
                    'product_id': pid,
                    'product_name': product.name,
                    'brand': product.brand,
                    'price': product.price,
                    'rating': product.average_rating,
                    'score': float(score),
                    'reason': 'Based on your preferences',
                    'method': 'collaborative'
                })
        
        session.close()
        return recommendations
    
    def _get_popular_recommendations(self, n: int) -> List[Dict]:
        """Get popular products"""
        
        top_products = sorted(
            self.popularity_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n]
        
        recommendations = []
        session = db_manager.get_session()
        
        for pid, score in top_products:
            product = session.query(Product).filter_by(id=pid).first()
            if product:
                recommendations.append({
                    'product_id': pid,
                    'product_name': product.name,
                    'brand': product.brand,
                    'price': product.price,
                    'rating': product.average_rating,
                    'score': float(score),
                    'reason': 'Trending now',
                    'method': 'popular'
                })
        
        session.close()
        return recommendations
    
    def _get_query_based_recommendations(self, query: str, n: int) -> List[Dict]:
        """Get recommendations based on search query"""
        
        # Create query embedding
        query_embedding = self.sentence_model.encode(query)
        
        similarities = {}
        for pid, embedding in self.product_embeddings.items():
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                embedding.reshape(1, -1)
            )[0][0]
            similarities[pid] = similarity
        
        # Get top N similar products
        top_products = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:n]
        
        recommendations = []
        session = db_manager.get_session()
        
        for pid, score in top_products:
            product = session.query(Product).filter_by(id=pid).first()
            if product:
                recommendations.append({
                    'product_id': pid,
                    'product_name': product.name,
                    'brand': product.brand,
                    'price': product.price,
                    'rating': product.average_rating,
                    'score': float(score),
                    'reason': 'Matches your search',
                    'method': 'query'
                })
        
        session.close()
        return recommendations
    
    def _combine_recommendations(self, recommendations: List[Dict], n: int) -> List[Dict]:
        """Combine recommendations from different sources"""
        
        # Group by product_id and combine scores
        product_scores = defaultdict(lambda: {'scores': [], 'data': None})
        
        for rec in recommendations:
            pid = rec['product_id']
            product_scores[pid]['scores'].append(rec['score'])
            if not product_scores[pid]['data']:
                product_scores[pid]['data'] = rec
        
        # Calculate final scores
        final_recs = []
        for pid, info in product_scores.items():
            # Weighted average of scores
            final_score = np.mean(info['scores']) * (1 + 0.1 * len(info['scores']))
            
            rec = info['data'].copy()
            rec['score'] = final_score
            rec['methods_used'] = len(info['scores'])
            final_recs.append(rec)
        
        # Sort by final score
        final_recs.sort(key=lambda x: x['score'], reverse=True)
        
        return final_recs[:n]
    
    def _diversify_recommendations(self, recommendations: List[Dict]) -> List[Dict]:
        """Add diversity to recommendations"""
        
        if len(recommendations) <= 3:
            return recommendations
        
        diversified = []
        seen_brands = set()
        seen_categories = set()
        
        # First pass: ensure brand diversity
        for rec in recommendations:
            brand = rec.get('brand', '')
            if brand not in seen_brands or len(diversified) < 3:
                diversified.append(rec)
                seen_brands.add(brand)
        
        # Add remaining if needed
        for rec in recommendations:
            if rec not in diversified and len(diversified) < len(recommendations):
                diversified.append(rec)
        
        return diversified
    
    def get_personalized_feed(self, user_id: int, limit: int = 20) -> List[Dict]:
        """Get personalized product feed for user"""
        
        # Get user profile
        profile = self.user_profiles.get(user_id, {})
        
        # Mix different recommendation strategies
        recommendations = []
        
        # Get collaborative recommendations
        collab_recs = self._get_collaborative_recommendations(user_id, limit // 2)
        recommendations.extend(collab_recs)
        
        # Get popular in user's preferred categories
        if profile.get('categories'):
            top_category = profile['categories'].most_common(1)[0][0]
            category_recs = self._get_category_recommendations(top_category, limit // 4)
            recommendations.extend(category_recs)
        
        # Get trending products
        trending_recs = self._get_trending_recommendations(limit // 4)
        recommendations.extend(trending_recs)
        
        # Combine and return
        return self._combine_recommendations(recommendations, limit)
    
    def _get_category_recommendations(self, category: str, n: int) -> List[Dict]:
        """Get recommendations from specific category"""
        
        session = db_manager.get_session()
        products = session.query(Product).filter_by(category=category).limit(n).all()
        
        recommendations = []
        for product in products:
            recommendations.append({
                'product_id': product.id,
                'product_name': product.name,
                'brand': product.brand,
                'price': product.price,
                'rating': product.average_rating,
                'score': self.popularity_scores.get(product.id, 0),
                'reason': f'Popular in {category}',
                'method': 'category'
            })
        
        session.close()
        return recommendations
    
    def _get_trending_recommendations(self, n: int) -> List[Dict]:
        """Get trending products based on recent activity"""
        
        session = db_manager.get_session()
        
        # Get products with recent reviews
        recent_date = datetime.now() - timedelta(days=7)
        recent_reviews = session.query(Review).filter(
            Review.created_at > recent_date
        ).all()
        
        product_activity = Counter(r.product_id for r in recent_reviews)
        top_trending = product_activity.most_common(n)
        
        recommendations = []
        for pid, count in top_trending:
            product = session.query(Product).filter_by(id=pid).first()
            if product:
                recommendations.append({
                    'product_id': pid,
                    'product_name': product.name,
                    'brand': product.brand,
                    'price': product.price,
                    'rating': product.average_rating,
                    'score': float(count),
                    'reason': 'Trending this week',
                    'method': 'trending'
                })
        
        session.close()
        return recommendations
    
    def save_models(self):
        """Save recommendation models to disk"""
        
        models_data = {
            'product_embeddings': self.product_embeddings,
            'user_profiles': self.user_profiles,
            'popularity_scores': self.popularity_scores,
            'user_map': getattr(self, 'user_map', {}),
            'product_map': getattr(self, 'product_map', {})
        }
        
        with open('models/recommendation_models.pkl', 'wb') as f:
            pickle.dump(models_data, f)
        
        if self.collab_model:
            torch.save(self.collab_model.state_dict(), 'models/collab_model.pth')
    
    def load_models(self):
        """Load recommendation models from disk"""
        
        with open('models/recommendation_models.pkl', 'rb') as f:
            models_data = pickle.load(f)
        
        self.product_embeddings = models_data['product_embeddings']
        self.user_profiles = models_data['user_profiles']
        self.popularity_scores = models_data['popularity_scores']
        self.user_map = models_data.get('user_map', {})
        self.product_map = models_data.get('product_map', {})
        
        # Load collaborative model if exists
        import os
        if os.path.exists('models/collab_model.pth') and self.user_map and self.product_map:
            n_users = len(self.user_map)
            n_items = len(self.product_map)
            self.collab_model = CollaborativeFilteringModel(n_users, n_items)
            self.collab_model.load_state_dict(torch.load('models/collab_model.pth'))
            self.collab_model.eval()

# Global instance
recommendation_engine = RecommendationEngine()

# Alias for compatibility
PhoneRecommendationEngine = RecommendationEngine
