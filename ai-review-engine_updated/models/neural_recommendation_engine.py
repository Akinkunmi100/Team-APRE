"""
Enhanced Neural Recommendation Engine
Advanced deep learning-based recommendations with personalization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import logging
from datetime import datetime, timedelta
import pickle
import json
from collections import defaultdict, Counter
from dataclasses import dataclass
import hashlib

# Optional deep learning imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available - using classical ML methods")

logger = logging.getLogger(__name__)

@dataclass
class UserProfile:
    """User profile for personalized recommendations"""
    user_id: str
    preferences: Dict[str, float] = None
    interaction_history: List[Dict] = None
    demographic_info: Dict[str, Any] = None
    behavioral_patterns: Dict[str, float] = None
    last_updated: datetime = None
    
    def __post_init__(self):
        if self.preferences is None:
            self.preferences = {}
        if self.interaction_history is None:
            self.interaction_history = []
        if self.demographic_info is None:
            self.demographic_info = {}
        if self.behavioral_patterns is None:
            self.behavioral_patterns = {}
        if self.last_updated is None:
            self.last_updated = datetime.now()

if TORCH_AVAILABLE:
    class NeuralCollaborativeFiltering(nn.Module):
        """Neural Collaborative Filtering model with embeddings"""
        
        def __init__(self, n_users: int, n_items: int, embedding_dim: int = 64, 
                     hidden_layers: List[int] = [128, 64, 32]):
            super().__init__()
            
            # User and item embeddings
            self.user_embedding = nn.Embedding(n_users, embedding_dim)
            self.item_embedding = nn.Embedding(n_items, embedding_dim)
            self.user_bias = nn.Embedding(n_users, 1)
            self.item_bias = nn.Embedding(n_items, 1)
            
            # MLP layers for interaction learning
            self.mlp_layers = nn.ModuleList()
            input_dim = embedding_dim * 2
            
            for hidden_dim in hidden_layers:
                self.mlp_layers.append(nn.Linear(input_dim, hidden_dim))
                self.mlp_layers.append(nn.BatchNorm1d(hidden_dim))
                self.mlp_layers.append(nn.ReLU())
                self.mlp_layers.append(nn.Dropout(0.2))
                input_dim = hidden_dim
            
            # Output layer
            self.output_layer = nn.Linear(hidden_layers[-1], 1)
            self.global_bias = nn.Parameter(torch.zeros(1))
            
            # Initialize embeddings
            self._init_weights()
        
        def _init_weights(self):
            """Initialize model weights"""
            nn.init.xavier_uniform_(self.user_embedding.weight)
            nn.init.xavier_uniform_(self.item_embedding.weight)
            nn.init.zeros_(self.user_bias.weight)
            nn.init.zeros_(self.item_bias.weight)
        
        def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
            """Forward pass"""
            # Get embeddings
            user_embeds = self.user_embedding(user_ids)
            item_embeds = self.item_embedding(item_ids)
            
            # Get biases
            user_bias = self.user_bias(user_ids).squeeze()
            item_bias = self.item_bias(item_ids).squeeze()
            
            # Concatenate embeddings for MLP
            mlp_input = torch.cat([user_embeds, item_embeds], dim=1)
            
            # Pass through MLP layers
            mlp_output = mlp_input
            for layer in self.mlp_layers:
                mlp_output = layer(mlp_output)
            
            # Final prediction
            mlp_score = self.output_layer(mlp_output).squeeze()
            prediction = self.global_bias + user_bias + item_bias + mlp_score
            
            return prediction
    
    class ContentBasedNN(nn.Module):
        """Neural network for content-based recommendations"""
        
        def __init__(self, n_features: int, hidden_layers: List[int] = [256, 128, 64]):
            super().__init__()
            
            layers = []
            input_dim = n_features
            
            for hidden_dim in hidden_layers:
                layers.extend([
                    nn.Linear(input_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.3)
                ])
                input_dim = hidden_dim
            
            # Output layer for rating prediction
            layers.append(nn.Linear(hidden_layers[-1], 1))
            layers.append(nn.Sigmoid())  # Normalize to 0-1 range
            
            self.network = nn.Sequential(*layers)
        
        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return self.network(features) * 5.0  # Scale to 0-5 rating range

else:
    # Dummy classes when PyTorch is not available
    class NeuralCollaborativeFiltering:
        def __init__(self, *args, **kwargs):
            pass
        def forward(self, *args, **kwargs):
            return None
    
    class ContentBasedNN:
        def __init__(self, *args, **kwargs):
            pass
        def forward(self, *args, **kwargs):
            return None

class NeuralRecommendationEngine:
    """Advanced neural recommendation engine with deep learning"""
    
    def __init__(self):
        """Initialize the neural recommendation engine"""
        
        # Model components
        self.ncf_model = None
        self.content_model = None
        self.feature_scaler = StandardScaler()
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        
        # Data storage
        self.user_profiles = {}
        self.item_features = {}
        self.interaction_matrix = None
        self.content_features = None
        
        # Model parameters
        self.embedding_dim = 64
        self.learning_rate = 0.001
        self.batch_size = 256
        self.epochs = 100
        
        # Classical fallback models
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.svd_model = TruncatedSVD(n_components=50)
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Recommendation strategies
        self.recommendation_strategies = [
            'neural_collaborative',
            'neural_content',
            'hybrid_ensemble',
            'classical_fallback'
        ]
        
        logger.info(f"Neural Recommendation Engine initialized (PyTorch: {TORCH_AVAILABLE})")
    
    def fit(self, df: pd.DataFrame, user_col: str = 'user_id', item_col: str = 'product', 
            rating_col: str = 'rating', text_col: str = 'review_text'):
        """
        Train the recommendation models on data
        
        Args:
            df: DataFrame with user interactions
            user_col: Column name for user IDs
            item_col: Column name for item/product IDs
            rating_col: Column name for ratings
            text_col: Column name for review text
        """
        try:
            logger.info("Training neural recommendation models...")
            
            # Create user IDs if not present
            if user_col not in df.columns:
                df[user_col] = df.index % 1000  # Simulate user IDs
            
            # Prepare data
            self._prepare_data(df, user_col, item_col, rating_col, text_col)
            
            # Train models based on availability
            if TORCH_AVAILABLE and len(df) > 100:
                self._train_neural_models(df, user_col, item_col, rating_col)
            
            # Always train classical models as fallback
            self._train_classical_models(df, item_col, rating_col, text_col)
            
            # Build user profiles
            self._build_user_profiles(df, user_col, item_col, rating_col)
            
            logger.info("Neural recommendation models trained successfully")
            
        except Exception as e:
            logger.error(f"Error training recommendation models: {e}")
            # Ensure we have at least basic functionality
            self._ensure_fallback_functionality(df, item_col, rating_col)
    
    def _prepare_data(self, df: pd.DataFrame, user_col: str, item_col: str, 
                     rating_col: str, text_col: str):
        """Prepare data for training"""
        
        # Encode users and items
        df['user_encoded'] = self.user_encoder.fit_transform(df[user_col].astype(str))
        df['item_encoded'] = self.item_encoder.fit_transform(df[item_col].astype(str))
        
        # Create interaction matrix
        self.interaction_matrix = df.pivot_table(
            index='user_encoded', 
            columns='item_encoded', 
            values=rating_col, 
            fill_value=0
        )
        
        # Extract content features
        if text_col in df.columns:
            # Group reviews by item and combine text
            item_texts = df.groupby(item_col)[text_col].apply(
                lambda x: ' '.join(x.astype(str))
            ).to_dict()
            
            # Create TF-IDF features
            texts = list(item_texts.values())
            if texts:
                tfidf_features = self.tfidf_vectorizer.fit_transform(texts)
                self.content_features = tfidf_features.toarray()
        
        # Store item features
        for item in df[item_col].unique():
            item_data = df[df[item_col] == item]
            self.item_features[item] = {
                'avg_rating': item_data[rating_col].mean(),
                'rating_count': len(item_data),
                'rating_std': item_data[rating_col].std(),
                'sentiment_score': self._estimate_sentiment(item_data, text_col)
            }
    
    def _train_neural_models(self, df: pd.DataFrame, user_col: str, item_col: str, rating_col: str):
        """Train neural models if PyTorch is available"""
        
        n_users = df['user_encoded'].nunique()
        n_items = df['item_encoded'].nunique()
        
        # Initialize neural collaborative filtering model
        self.ncf_model = NeuralCollaborativeFiltering(n_users, n_items, self.embedding_dim)
        
        # Prepare training data
        user_ids = torch.LongTensor(df['user_encoded'].values)
        item_ids = torch.LongTensor(df['item_encoded'].values)
        ratings = torch.FloatTensor(df[rating_col].values)
        
        # Training setup
        optimizer = optim.Adam(self.ncf_model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop (simplified)
        self.ncf_model.train()
        for epoch in range(min(self.epochs, 50)):  # Reduced for demo
            optimizer.zero_grad()
            predictions = self.ncf_model(user_ids, item_ids)
            loss = criterion(predictions, ratings)
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                logger.info(f"NCF Training Epoch {epoch}, Loss: {loss.item():.4f}")
        
        # Train content-based model if features available
        if self.content_features is not None:
            n_features = self.content_features.shape[1]
            self.content_model = ContentBasedNN(n_features)
            
            # Prepare content training data
            item_features_tensor = torch.FloatTensor(self.content_features)
            item_ratings = torch.FloatTensor([
                self.item_features[item]['avg_rating'] for item in df[item_col].unique()
            ])
            
            # Train content model
            optimizer_content = optim.Adam(self.content_model.parameters(), lr=self.learning_rate)
            
            self.content_model.train()
            for epoch in range(30):  # Fewer epochs for content model
                optimizer_content.zero_grad()
                predictions = self.content_model(item_features_tensor).squeeze()
                loss = criterion(predictions, item_ratings)
                loss.backward()
                optimizer_content.step()
    
    def _train_classical_models(self, df: pd.DataFrame, item_col: str, rating_col: str, text_col: str):
        """Train classical ML models as fallback"""
        
        # Train SVD for collaborative filtering
        if self.interaction_matrix.shape[0] > 1 and self.interaction_matrix.shape[1] > 1:
            self.svd_model.fit(self.interaction_matrix)
        
        # Train random forest for content-based recommendations
        if self.content_features is not None:
            item_ratings = [
                self.item_features[item]['avg_rating'] for item in df[item_col].unique()
            ]
            self.rf_model.fit(self.content_features, item_ratings)
    
    def _build_user_profiles(self, df: pd.DataFrame, user_col: str, item_col: str, rating_col: str):
        """Build detailed user profiles"""
        
        for user_id in df[user_col].unique():
            user_data = df[df[user_col] == user_id]
            
            # Calculate preferences
            preferences = {
                'avg_rating_given': user_data[rating_col].mean(),
                'rating_variance': user_data[rating_col].var(),
                'total_reviews': len(user_data),
                'brand_preferences': user_data.groupby('brand')[rating_col].mean().to_dict() if 'brand' in user_data.columns else {},
                'feature_importance': self._extract_feature_importance(user_data)
            }
            
            # Track interaction history
            interactions = []
            for _, row in user_data.iterrows():
                interactions.append({
                    'item': row[item_col],
                    'rating': row[rating_col],
                    'timestamp': row.get('date', datetime.now()),
                    'sentiment': self._estimate_single_sentiment(row.get('review_text', ''))
                })
            
            # Create user profile
            self.user_profiles[user_id] = UserProfile(
                user_id=str(user_id),
                preferences=preferences,
                interaction_history=interactions,
                behavioral_patterns=self._analyze_user_behavior(user_data),
                last_updated=datetime.now()
            )
    
    def get_recommendations(self, user_id: Optional[str] = None, n_recommendations: int = 10,
                          strategy: str = 'hybrid_ensemble', context: Optional[Dict] = None) -> List[Dict]:
        """
        Get personalized recommendations
        
        Args:
            user_id: User identifier (can be None for general recommendations)
            n_recommendations: Number of recommendations to return
            strategy: Recommendation strategy to use
            context: Additional context for recommendations
            
        Returns:
            List of recommendation dictionaries
        """
        try:
            if strategy == 'hybrid_ensemble':
                return self._get_hybrid_recommendations(user_id, n_recommendations, context)
            elif strategy == 'neural_collaborative':
                return self._get_neural_collaborative_recommendations(user_id, n_recommendations)
            elif strategy == 'neural_content':
                return self._get_neural_content_recommendations(user_id, n_recommendations)
            else:
                return self._get_classical_recommendations(user_id, n_recommendations)
                
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return self._get_fallback_recommendations(n_recommendations)
    
    def _get_hybrid_recommendations(self, user_id: Optional[str], n_recommendations: int,
                                   context: Optional[Dict]) -> List[Dict]:
        """Get hybrid recommendations combining multiple approaches"""
        
        recommendations = []
        
        # Get recommendations from different strategies
        if TORCH_AVAILABLE and self.ncf_model is not None:
            neural_recs = self._get_neural_collaborative_recommendations(user_id, n_recommendations)
            recommendations.extend([(rec, 0.4, 'neural_collab') for rec in neural_recs])
        
        if TORCH_AVAILABLE and self.content_model is not None:
            content_recs = self._get_neural_content_recommendations(user_id, n_recommendations)
            recommendations.extend([(rec, 0.3, 'neural_content') for rec in content_recs])
        
        # Classical recommendations
        classical_recs = self._get_classical_recommendations(user_id, n_recommendations)
        recommendations.extend([(rec, 0.3, 'classical') for rec in classical_recs])
        
        # Combine and rank recommendations
        combined_scores = defaultdict(lambda: {'score': 0, 'methods': [], 'details': {}})
        
        for rec, weight, method in recommendations:
            item_name = rec['product_name']
            combined_scores[item_name]['score'] += rec['score'] * weight
            combined_scores[item_name]['methods'].append(method)
            combined_scores[item_name]['details'].update(rec)
        
        # Sort and format final recommendations
        final_recs = []
        for item_name, info in sorted(combined_scores.items(), 
                                     key=lambda x: x[1]['score'], reverse=True)[:n_recommendations]:
            
            final_recs.append({
                'product_name': item_name,
                'score': info['score'],
                'confidence': min(len(info['methods']) * 0.3, 1.0),
                'reasoning': f"Recommended by {len(info['methods'])} AI models",
                'methods_used': info['methods'],
                'rating': info['details'].get('rating', 0),
                'sentiment_score': info['details'].get('sentiment_score', 0.5),
                'recommendation_type': 'hybrid_ai'
            })
        
        return final_recs
    
    def _get_neural_collaborative_recommendations(self, user_id: Optional[str], 
                                                 n_recommendations: int) -> List[Dict]:
        """Get recommendations from neural collaborative filtering"""
        
        if not TORCH_AVAILABLE or self.ncf_model is None:
            return self._get_classical_recommendations(user_id, n_recommendations)
        
        recommendations = []
        
        try:
            # Get user encoding
            if user_id and str(user_id) in self.user_encoder.classes_:
                user_encoded = self.user_encoder.transform([str(user_id)])[0]
            else:
                # Use average user for new users
                user_encoded = len(self.user_encoder.classes_) // 2
            
            self.ncf_model.eval()
            with torch.no_grad():
                user_tensor = torch.LongTensor([user_encoded] * len(self.item_encoder.classes_))
                item_tensor = torch.LongTensor(range(len(self.item_encoder.classes_)))
                
                predictions = self.ncf_model(user_tensor, item_tensor)
                
                # Get top recommendations
                top_indices = torch.argsort(predictions, descending=True)[:n_recommendations]
                
                for idx in top_indices:
                    item_encoded = idx.item()
                    item_name = self.item_encoder.inverse_transform([item_encoded])[0]
                    score = predictions[idx].item()
                    
                    recommendations.append({
                        'product_name': item_name,
                        'score': max(0, min(5, score)),  # Clamp to valid range
                        'rating': self.item_features.get(item_name, {}).get('avg_rating', 0),
                        'sentiment_score': self.item_features.get(item_name, {}).get('sentiment_score', 0.5),
                        'recommendation_type': 'neural_collaborative'
                    })
        
        except Exception as e:
            logger.error(f"Error in neural collaborative recommendations: {e}")
            return self._get_classical_recommendations(user_id, n_recommendations)
        
        return recommendations
    
    def _get_neural_content_recommendations(self, user_id: Optional[str], 
                                          n_recommendations: int) -> List[Dict]:
        """Get recommendations from neural content-based model"""
        
        if not TORCH_AVAILABLE or self.content_model is None or self.content_features is None:
            return self._get_classical_recommendations(user_id, n_recommendations)
        
        recommendations = []
        
        try:
            self.content_model.eval()
            with torch.no_grad():
                features_tensor = torch.FloatTensor(self.content_features)
                predictions = self.content_model(features_tensor).squeeze()
                
                # Get top recommendations
                top_indices = torch.argsort(predictions, descending=True)[:n_recommendations]
                items = list(self.item_features.keys())
                
                for idx in top_indices:
                    if idx < len(items):
                        item_name = items[idx]
                        score = predictions[idx].item()
                        
                        recommendations.append({
                            'product_name': item_name,
                            'score': score,
                            'rating': self.item_features[item_name]['avg_rating'],
                            'sentiment_score': self.item_features[item_name]['sentiment_score'],
                            'recommendation_type': 'neural_content'
                        })
        
        except Exception as e:
            logger.error(f"Error in neural content recommendations: {e}")
            return self._get_classical_recommendations(user_id, n_recommendations)
        
        return recommendations
    
    def _get_classical_recommendations(self, user_id: Optional[str], n_recommendations: int) -> List[Dict]:
        """Get recommendations from classical ML models"""
        
        recommendations = []
        
        # Get user preferences if available
        user_preferences = {}
        if user_id and user_id in self.user_profiles:
            user_preferences = self.user_profiles[user_id].preferences
        
        # Sort items by combined score
        item_scores = []
        for item_name, features in self.item_features.items():
            score = features['avg_rating'] * 0.6 + features['sentiment_score'] * 2.0 * 0.4
            
            # Apply user preferences if available
            if 'brand_preferences' in user_preferences:
                # Extract brand from item name (simplified)
                for brand, brand_rating in user_preferences['brand_preferences'].items():
                    if brand.lower() in item_name.lower():
                        score += (brand_rating - 3.0) * 0.2  # Adjust based on brand preference
                        break
            
            item_scores.append((item_name, score, features))
        
        # Sort and get top recommendations
        item_scores.sort(key=lambda x: x[1], reverse=True)
        
        for item_name, score, features in item_scores[:n_recommendations]:
            recommendations.append({
                'product_name': item_name,
                'score': score,
                'rating': features['avg_rating'],
                'sentiment_score': features['sentiment_score'],
                'review_count': features['rating_count'],
                'recommendation_type': 'classical'
            })
        
        return recommendations
    
    def _get_fallback_recommendations(self, n_recommendations: int) -> List[Dict]:
        """Fallback recommendations when all else fails"""
        
        fallback_phones = [
            {'product_name': 'iPhone 15 Pro', 'score': 4.5, 'rating': 4.5, 'sentiment_score': 0.85},
            {'product_name': 'Samsung Galaxy S24 Ultra', 'score': 4.4, 'rating': 4.4, 'sentiment_score': 0.82},
            {'product_name': 'Google Pixel 8 Pro', 'score': 4.3, 'rating': 4.3, 'sentiment_score': 0.80},
            {'product_name': 'OnePlus 12', 'score': 4.2, 'rating': 4.2, 'sentiment_score': 0.78},
            {'product_name': 'Xiaomi 14 Pro', 'score': 4.1, 'rating': 4.1, 'sentiment_score': 0.75}
        ]
        
        for phone in fallback_phones:
            phone['recommendation_type'] = 'fallback'
            phone['reasoning'] = 'Popular choice with excellent reviews'
        
        return fallback_phones[:n_recommendations]
    
    def _estimate_sentiment(self, item_data: pd.DataFrame, text_col: str) -> float:
        """Estimate sentiment score for an item"""
        if text_col not in item_data.columns:
            return 0.5
        
        # Simple sentiment estimation based on rating
        avg_rating = item_data['rating'].mean() if 'rating' in item_data.columns else 3.0
        sentiment_score = (avg_rating - 1) / 4  # Normalize to 0-1
        
        return max(0, min(1, sentiment_score))
    
    def _estimate_single_sentiment(self, text: str) -> float:
        """Estimate sentiment for a single text"""
        if not text:
            return 0.5
        
        # Simple keyword-based sentiment
        positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'perfect']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'horrible', 'worst']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count + negative_count == 0:
            return 0.5
        
        return positive_count / (positive_count + negative_count)
    
    def _extract_feature_importance(self, user_data: pd.DataFrame) -> Dict[str, float]:
        """Extract feature importance from user behavior"""
        # Simplified feature importance based on rating patterns
        features = {
            'camera_importance': 0.5,
            'battery_importance': 0.5,
            'performance_importance': 0.5,
            'price_sensitivity': 0.5
        }
        
        # This would be enhanced with actual feature extraction from reviews
        return features
    
    def _analyze_user_behavior(self, user_data: pd.DataFrame) -> Dict[str, float]:
        """Analyze user behavioral patterns"""
        patterns = {
            'review_frequency': len(user_data),
            'rating_generosity': user_data['rating'].mean() - 3.0 if 'rating' in user_data.columns else 0,
            'engagement_level': min(len(user_data) / 10, 1.0),
            'brand_loyalty': self._calculate_brand_loyalty(user_data)
        }
        
        return patterns
    
    def _calculate_brand_loyalty(self, user_data: pd.DataFrame) -> float:
        """Calculate user's brand loyalty"""
        if 'brand' not in user_data.columns:
            return 0.5
        
        brand_counts = user_data['brand'].value_counts()
        if len(brand_counts) == 0:
            return 0.5
        
        # Loyalty is higher if user consistently chooses same brands
        max_brand_count = brand_counts.iloc[0]
        total_count = len(user_data)
        
        return max_brand_count / total_count if total_count > 0 else 0.5
    
    def _ensure_fallback_functionality(self, df: pd.DataFrame, item_col: str, rating_col: str):
        """Ensure basic functionality even if training fails"""
        if df is not None and not df.empty:
            # At least store basic item features
            for item in df[item_col].unique():
                item_data = df[df[item_col] == item]
                self.item_features[item] = {
                    'avg_rating': item_data[rating_col].mean(),
                    'rating_count': len(item_data),
                    'sentiment_score': self._estimate_sentiment(item_data, 'review_text')
                }

    def get_explanation(self, user_id: str, item_name: str) -> str:
        """Get explanation for why an item was recommended"""
        explanations = []
        
        if user_id in self.user_profiles:
            profile = self.user_profiles[user_id]
            
            # Rating-based explanation
            if item_name in self.item_features:
                rating = self.item_features[item_name]['avg_rating']
                explanations.append(f"High user rating ({rating:.1f}/5)")
            
            # Brand preference explanation
            if 'brand_preferences' in profile.preferences:
                for brand in profile.preferences['brand_preferences']:
                    if brand.lower() in item_name.lower():
                        explanations.append(f"Matches your {brand} preference")
                        break
        
        return " • ".join(explanations) if explanations else "Popular choice with good reviews"

# Usage example
if __name__ == "__main__":
    # Test the neural recommendation engine
    engine = NeuralRecommendationEngine()
    
    # Create sample data
    sample_data = pd.DataFrame({
        'user_id': np.random.randint(1, 100, 1000),
        'product': np.random.choice(['iPhone 15', 'Galaxy S24', 'Pixel 8'], 1000),
        'rating': np.random.randint(1, 6, 1000),
        'review_text': ['Good phone'] * 1000,
        'brand': np.random.choice(['Apple', 'Samsung', 'Google'], 1000)
    })
    
    # Train the engine
    engine.fit(sample_data)
    
    # Get recommendations
    recommendations = engine.get_recommendations(user_id='1', n_recommendations=5)
    
    print("Neural AI Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['product_name']} (Score: {rec['score']:.2f}, Type: {rec['recommendation_type']})")