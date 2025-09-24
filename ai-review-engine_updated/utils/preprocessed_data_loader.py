"""
Preprocessed Data Loader
Centralized utility for loading and accessing preprocessed features
"""

import pandas as pd
import numpy as np
import json
import ast
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

class PreprocessedDataLoader:
    """
    Loader for preprocessed data with all NLP features
    """
    
    def __init__(self, dataset_path: str = None):
        """
        Initialize the preprocessed data loader
        
        Args:
            dataset_path: Path to preprocessed dataset (uses default if None)
        """
        if dataset_path is None:
            # Try multiple possible dataset locations
            project_root = Path(__file__).parent.parent
            possible_paths = [
                project_root / "data" / "processed" / "final_dataset_hybrid_preprocessed.csv",
                project_root / "final_dataset_streamlined_clean.csv",
                project_root / "data" / "final_dataset_streamlined_clean.csv"
            ]
            
            # Use the first available file
            for path in possible_paths:
                if path.exists():
                    self.dataset_path = path
                    break
            else:
                # Default to the streamlined clean file
                self.dataset_path = project_root / "final_dataset_streamlined_clean.csv"
        else:
            self.dataset_path = Path(dataset_path)
        
        self.data = None
        self._load_data()
    
    def _load_data(self):
        """Load and parse the preprocessed dataset"""
        try:
            logger.info(f"Loading preprocessed data from {self.dataset_path}")
            self.data = pd.read_csv(self.dataset_path)
            
            # Parse JSON/string columns
            self._parse_json_columns()
            
            # Convert date column
            if 'date' in self.data.columns:
                self.data['date'] = pd.to_datetime(self.data['date'], errors='coerce')
            
            logger.info(f"Successfully loaded {len(self.data)} preprocessed reviews")
            
        except Exception as e:
            logger.error(f"Error loading preprocessed data: {e}")
            raise
    
    def _parse_json_columns(self):
        """Parse JSON string columns into Python objects"""
        json_columns = ['aspects', 'aspect_sentiments', 'key_phrases']
        
        for col in json_columns:
            if col in self.data.columns:
                self.data[col] = self.data[col].apply(self._safe_parse_json)
    
    def _safe_parse_json(self, value):
        """Safely parse JSON or string representation"""
        if pd.isna(value) or value == '[]' or value == '{}':
            return []
        
        try:
            # Try JSON parse
            if isinstance(value, str):
                # Handle both JSON and Python literal syntax
                if value.startswith('[') or value.startswith('{'):
                    try:
                        return json.loads(value)
                    except:
                        return ast.literal_eval(value)
            return value
        except:
            return []
    
    @lru_cache(maxsize=1)
    def get_full_dataset(self) -> pd.DataFrame:
        """
        Get the full preprocessed dataset
        
        Returns:
            Complete preprocessed dataframe
        """
        return self.data.copy()
    
    def get_sentiment_features(self, product: str = None) -> pd.DataFrame:
        """
        Get sentiment features for analysis
        
        Args:
            product: Optional product filter
            
        Returns:
            DataFrame with sentiment features
        """
        df = self.data.copy()
        
        if product:
            df = df[df['product'] == product]
        
        sentiment_cols = [
            'sentiment_polarity', 
            'sentiment_subjectivity', 
            'sentiment_label',
            'sentiment_confidence'
        ]
        
        # Include only existing columns
        existing_cols = [col for col in sentiment_cols if col in df.columns]
        base_cols = ['review_id', 'product', 'rating', 'review_text']
        
        return df[base_cols + existing_cols]
    
    def get_spam_features(self) -> pd.DataFrame:
        """
        Get spam detection features
        
        Returns:
            DataFrame with spam-related features
        """
        spam_cols = [
            'is_spam', 
            'spam_score', 
            'is_spam_combined',
            'credibility_score',
            'enhanced_credibility'
        ]
        
        existing_cols = [col for col in spam_cols if col in self.data.columns]
        base_cols = ['review_id', 'product', 'user_id']
        
        return self.data[base_cols + existing_cols]
    
    def get_aspect_sentiments(self, product: str = None) -> List[Dict]:
        """
        Get aspect-based sentiment analysis results
        
        Args:
            product: Optional product filter
            
        Returns:
            List of aspect sentiment dictionaries
        """
        df = self.data.copy()
        
        if product:
            df = df[df['product'] == product]
        
        aspect_data = []
        for _, row in df.iterrows():
            if 'aspect_sentiments' in row and row['aspect_sentiments']:
                aspect_data.append({
                    'review_id': row.get('review_id'),
                    'product': row.get('product'),
                    'aspects': row['aspect_sentiments']
                })
        
        return aspect_data
    
    def get_key_phrases(self, min_frequency: int = 2) -> Dict[str, int]:
        """
        Get aggregated key phrases across all reviews
        
        Args:
            min_frequency: Minimum frequency for inclusion
            
        Returns:
            Dictionary of phrases and their frequencies
        """
        phrase_counts = {}
        
        if 'key_phrases' in self.data.columns:
            for phrases in self.data['key_phrases'].dropna():
                if phrases and isinstance(phrases, list):
                    for phrase in phrases:
                        phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
        
        # Filter by minimum frequency
        return {k: v for k, v in phrase_counts.items() if v >= min_frequency}
    
    def get_credible_reviews(self, min_credibility: float = 0.7) -> pd.DataFrame:
        """
        Get reviews with high credibility scores
        
        Args:
            min_credibility: Minimum credibility threshold
            
        Returns:
            DataFrame of credible reviews
        """
        credibility_col = 'enhanced_credibility' if 'enhanced_credibility' in self.data.columns else 'credibility_score'
        
        if credibility_col in self.data.columns:
            return self.data[self.data[credibility_col] >= min_credibility]
        else:
            logger.warning("No credibility column found, returning all data")
            return self.data
    
    def get_product_sentiment_summary(self, product: str) -> Dict:
        """
        Get sentiment summary for a specific product
        
        Args:
            product: Product name
            
        Returns:
            Sentiment summary dictionary
        """
        product_data = self.data[self.data['product'] == product]
        
        if product_data.empty:
            return {'error': f'No data found for product: {product}'}
        
        summary = {
            'product': product,
            'total_reviews': len(product_data),
            'avg_rating': product_data['rating'].mean() if 'rating' in product_data.columns else None
        }
        
        # Add sentiment distribution if available
        if 'sentiment_label' in product_data.columns:
            sentiment_dist = product_data['sentiment_label'].value_counts(normalize=True).to_dict()
            summary['sentiment_distribution'] = sentiment_dist
        
        if 'sentiment_polarity' in product_data.columns:
            summary['avg_polarity'] = product_data['sentiment_polarity'].mean()
            summary['avg_subjectivity'] = product_data['sentiment_subjectivity'].mean() if 'sentiment_subjectivity' in product_data.columns else None
        
        # Add aspect summary
        if 'aspect_sentiments' in product_data.columns:
            aspect_summary = {}
            for aspects in product_data['aspect_sentiments'].dropna():
                if aspects and isinstance(aspects, list):
                    for aspect_dict in aspects:
                        if isinstance(aspect_dict, dict):
                            aspect = aspect_dict.get('aspect')
                            sentiment = aspect_dict.get('sentiment')
                            if aspect and sentiment:
                                if aspect not in aspect_summary:
                                    aspect_summary[aspect] = {'positive': 0, 'negative': 0, 'neutral': 0}
                                aspect_summary[aspect][sentiment] = aspect_summary[aspect].get(sentiment, 0) + 1
            summary['aspect_summary'] = aspect_summary
        
        # Add spam statistics
        if 'is_spam_combined' in product_data.columns:
            summary['spam_rate'] = product_data['is_spam_combined'].mean()
        
        return summary
    
    def get_reviews_by_sentiment(self, sentiment: str) -> pd.DataFrame:
        """
        Get reviews filtered by sentiment label
        
        Args:
            sentiment: 'positive', 'negative', or 'neutral'
            
        Returns:
            Filtered DataFrame
        """
        if 'sentiment_label' in self.data.columns:
            return self.data[self.data['sentiment_label'] == sentiment]
        else:
            logger.warning("No sentiment_label column found")
            return pd.DataFrame()
    
    def get_aspect_mentions(self, aspect: str) -> pd.DataFrame:
        """
        Get reviews that mention a specific aspect
        
        Args:
            aspect: Aspect to search for
            
        Returns:
            DataFrame of reviews mentioning the aspect
        """
        matching_reviews = []
        
        for idx, row in self.data.iterrows():
            if 'aspect_sentiments' in row and row['aspect_sentiments']:
                aspects_list = row['aspect_sentiments']
                if isinstance(aspects_list, list):
                    for aspect_dict in aspects_list:
                        if isinstance(aspect_dict, dict) and aspect_dict.get('aspect') == aspect:
                            matching_reviews.append(idx)
                            break
        
        return self.data.loc[matching_reviews] if matching_reviews else pd.DataFrame()
    
    def get_temporal_sentiment(self) -> pd.DataFrame:
        """
        Get sentiment trends over time
        
        Returns:
            DataFrame with temporal sentiment data
        """
        if 'date' not in self.data.columns or 'sentiment_polarity' not in self.data.columns:
            logger.warning("Required columns for temporal analysis not found")
            return pd.DataFrame()
        
        temporal_data = self.data[['date', 'sentiment_polarity', 'sentiment_label']].copy()
        temporal_data['year_month'] = pd.to_datetime(temporal_data['date']).dt.to_period('M')
        
        return temporal_data.groupby('year_month').agg({
            'sentiment_polarity': 'mean',
            'sentiment_label': lambda x: x.value_counts().to_dict()
        }).reset_index()

# Singleton instance for easy access
_loader_instance = None

def get_loader() -> PreprocessedDataLoader:
    """
    Get singleton instance of the preprocessed data loader
    
    Returns:
        PreprocessedDataLoader instance
    """
    global _loader_instance
    if _loader_instance is None:
        _loader_instance = PreprocessedDataLoader()
    return _loader_instance

# Convenience functions
def load_preprocessed_data() -> pd.DataFrame:
    """Load the full preprocessed dataset"""
    return get_loader().get_full_dataset()

def get_sentiment_data(product: str = None) -> pd.DataFrame:
    """Get sentiment features"""
    return get_loader().get_sentiment_features(product)

def get_spam_data() -> pd.DataFrame:
    """Get spam detection features"""
    return get_loader().get_spam_features()

def get_aspect_data(product: str = None) -> List[Dict]:
    """Get aspect sentiment data"""
    return get_loader().get_aspect_sentiments(product)

def get_product_summary(product: str) -> Dict:
    """Get product sentiment summary"""
    return get_loader().get_product_sentiment_summary(product)