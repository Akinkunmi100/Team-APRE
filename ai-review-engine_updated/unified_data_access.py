"""
Unified Data Access Module
Ensures all apps use the cleaned dataset consistently
Replaces create_sample_data() and other sample data functions
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.dataset_config import PRIMARY_DATASET, DATASET_CONFIG
from utils.data_loader import get_data_loader

# Cache for the dataset
_cached_dataset = None

def get_primary_dataset():
    """
    Get the primary cleaned dataset.
    This replaces all create_sample_data() functions.
    
    Returns:
        pandas.DataFrame: The cleaned dataset
    """
    global _cached_dataset
    
    if _cached_dataset is None:
        loader = get_data_loader()
        _cached_dataset = loader.load_data()
    
    return _cached_dataset

def create_sample_data(n_samples=100):
    """
    Create sample data from the cleaned dataset.
    This function maintains backward compatibility with existing code.
    
    Args:
        n_samples: Number of samples to return
        
    Returns:
        pandas.DataFrame: Sample from the cleaned dataset
    """
    dataset = get_primary_dataset()
    
    # Return a sample if dataset is larger than requested
    if len(dataset) > n_samples:
        return dataset.sample(n=n_samples, random_state=42).reset_index(drop=True)
    else:
        return dataset.copy()

def load_sample_data():
    """
    Load sample data for demos and testing.
    Uses the cleaned dataset instead of generating fake data.
    
    Returns:
        pandas.DataFrame: Sample data
    """
    return create_sample_data(n_samples=500)

def get_demo_data():
    """
    Get demonstration data for app showcases.
    
    Returns:
        pandas.DataFrame: Demo dataset
    """
    dataset = get_primary_dataset()
    
    # Get a diverse sample with different brands
    demo_data = []
    brands = dataset['brand'].unique()[:5]  # Top 5 brands
    
    for brand in brands:
        brand_data = dataset[dataset['brand'] == brand].head(20)
        demo_data.append(brand_data)
    
    if demo_data:
        return pd.concat(demo_data, ignore_index=True)
    else:
        return dataset.head(100)

def get_test_reviews():
    """
    Get test reviews for analysis functions.
    
    Returns:
        list: List of review dictionaries
    """
    dataset = create_sample_data(50)
    
    reviews = []
    for _, row in dataset.iterrows():
        reviews.append({
            'review_id': row.get('review_id', f"TEST_{_}"),
            'text': row['review_text'],
            'rating': row.get('rating', None),
            'product': row.get('product', 'Unknown'),
            'brand': row.get('brand', 'Unknown'),
            'date': row.get('date', None)
        })
    
    return reviews

def get_products_for_comparison():
    """
    Get a list of products for comparison features.
    
    Returns:
        list: List of unique product names
    """
    dataset = get_primary_dataset()
    
    # Get products with most reviews
    product_counts = dataset['product'].value_counts().head(20)
    return product_counts.index.tolist()

def get_brands_list():
    """
    Get list of brands for filtering and selection.
    
    Returns:
        list: Sorted list of unique brands
    """
    dataset = get_primary_dataset()
    return sorted(dataset['brand'].dropna().unique().tolist())

def get_dataset_for_analysis(product=None, brand=None, limit=None):
    """
    Get dataset filtered for specific analysis.
    
    Args:
        product: Filter by product name
        brand: Filter by brand
        limit: Maximum number of rows
        
    Returns:
        pandas.DataFrame: Filtered dataset
    """
    dataset = get_primary_dataset()
    
    if product:
        dataset = dataset[dataset['product'].str.contains(product, case=False, na=False)]
    
    if brand:
        dataset = dataset[dataset['brand'] == brand]
    
    if limit and len(dataset) > limit:
        dataset = dataset.head(limit)
    
    return dataset

def get_reviews_by_rating(min_rating=None, max_rating=None):
    """
    Get reviews filtered by rating range.
    
    Args:
        min_rating: Minimum rating (inclusive)
        max_rating: Maximum rating (inclusive)
        
    Returns:
        pandas.DataFrame: Filtered reviews
    """
    dataset = get_primary_dataset()
    
    # Only consider reviews with ratings
    rated_reviews = dataset[dataset['has_rating'] == True].copy()
    
    if min_rating is not None:
        rated_reviews = rated_reviews[rated_reviews['rating'] >= min_rating]
    
    if max_rating is not None:
        rated_reviews = rated_reviews[rated_reviews['rating'] <= max_rating]
    
    return rated_reviews

def get_recent_reviews(days=30):
    """
    Get recent reviews from the dataset.
    
    Args:
        days: Number of days to look back
        
    Returns:
        pandas.DataFrame: Recent reviews
    """
    dataset = get_primary_dataset()
    
    # Convert date column to datetime if needed
    if 'date' in dataset.columns:
        dataset['date'] = pd.to_datetime(dataset['date'], errors='coerce')
        
        # Get recent reviews
        cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=days)
        recent = dataset[dataset['date'] >= cutoff_date]
        
        if len(recent) > 0:
            return recent
    
    # If no recent reviews or no dates, return latest reviews by index
    return dataset.tail(min(100, len(dataset)))

def get_recent_activity_data():
    """
    Get recent activity data from actual dataset without fake generation.
    Returns actual recent reviews or activity, not simulated data.
    
    Returns:
        dict: Recent activity information or None if unavailable
    """
    try:
        dataset = get_primary_dataset()
        
        if len(dataset) == 0:
            return {
                'available': False,
                'message': 'No recent activity data available',
                'timestamp': pd.Timestamp.now().isoformat()
            }
        
        # Get the most recent review (by index, since we don't generate timestamps)
        recent_row = dataset.iloc[-1]
        
        # Return actual data without simulation
        activity = {
            'available': True,
            'type': 'recent_review',
            'timestamp': pd.Timestamp.now().isoformat(),
            'product': recent_row['product'],
            'brand': recent_row['brand'],
            'review_text': recent_row['review_text'][:200] if len(recent_row['review_text']) > 200 else recent_row['review_text'],
            'rating': recent_row.get('rating'),
            'sentiment': recent_row.get('sentiment'),
            'user_id': recent_row.get('user_id'),
            'data_source': 'actual_dataset'
        }
        
        return activity
        
    except Exception as e:
        return {
            'available': False,
            'error': str(e),
            'message': 'Unable to load recent activity data',
            'timestamp': pd.Timestamp.now().isoformat()
        }

# Convenience functions for backward compatibility
def get_sample_reviews(n=100):
    """Backward compatibility wrapper"""
    return create_sample_data(n)

def load_test_data():
    """Backward compatibility wrapper"""
    return load_sample_data()

# Dataset statistics
def get_dataset_stats():
    """
    Get comprehensive statistics about the dataset.
    
    Returns:
        dict: Dataset statistics
    """
    dataset = get_primary_dataset()
    
    return {
        'total_reviews': len(dataset),
        'reviews_with_ratings': dataset['has_rating'].sum() if 'has_rating' in dataset.columns else 0,
        'unique_products': dataset['product'].nunique(),
        'unique_brands': dataset['brand'].nunique(),
        'avg_review_length': dataset['review_text'].str.len().mean(),
        'rating_distribution': dataset['rating'].value_counts().to_dict() if 'rating' in dataset.columns else {},
        'brand_distribution': dataset['brand'].value_counts().head(10).to_dict(),
        'source': 'Cleaned Dataset (final_dataset_streamlined_clean.csv)'
    }

# Export all functions
__all__ = [
    'get_primary_dataset',
    'create_sample_data',
    'load_sample_data',
    'get_demo_data',
    'get_test_reviews',
    'get_products_for_comparison',
    'get_brands_list',
    'get_dataset_for_analysis',
    'get_reviews_by_rating',
    'get_recent_reviews',
    'get_recent_activity_data',
    'get_sample_reviews',
    'load_test_data',
    'get_dataset_stats'
]