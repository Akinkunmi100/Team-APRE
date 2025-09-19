"""
Data Loader Module for AI Phone Review Engine
This module handles all dataset loading operations using the cleaned dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.dataset_config import (
    load_primary_dataset,
    DATASET_CONFIG,
    PRIMARY_DATASET,
    validate_dataset
)

class DataLoader:
    """
    Centralized data loader for the AI Phone Review Engine
    Uses the cleaned dataset: final_dataset_streamlined_clean.csv
    """
    
    def __init__(self):
        """Initialize the data loader"""
        self.dataset_path = PRIMARY_DATASET
        self.config = DATASET_CONFIG
        self.data = None
        self.cached = False
        
    def load_data(self, force_reload=False):
        """
        Load the primary dataset
        
        Args:
            force_reload: Force reload even if data is cached
            
        Returns:
            pandas.DataFrame: The loaded dataset
        """
        if self.data is not None and not force_reload and self.cached:
            print("📦 Using cached dataset")
            return self.data
            
        print(f"Loading dataset from: {self.dataset_path}")
        self.data = load_primary_dataset()
        self.cached = True
        
        # Validate the dataset
        validate_dataset(self.data)
        
        return self.data
    
    def get_sample_data(self, n=100, random_state=42):
        """
        Get a sample of the dataset
        
        Args:
            n: Number of samples
            random_state: Random seed for reproducibility
            
        Returns:
            pandas.DataFrame: Sample of the dataset
        """
        if self.data is None:
            self.load_data()
            
        return self.data.sample(n=min(n, len(self.data)), random_state=random_state)
    
    def get_filtered_data(self, **kwargs):
        """
        Get filtered dataset based on criteria
        
        Args:
            brand: Filter by brand name
            product: Filter by product name
            has_rating: Filter by rating availability
            min_date: Minimum date
            max_date: Maximum date
            source: Filter by data source
            
        Returns:
            pandas.DataFrame: Filtered dataset
        """
        if self.data is None:
            self.load_data()
            
        df = self.data.copy()
        
        # Apply filters
        if 'brand' in kwargs and kwargs['brand']:
            df = df[df['brand'] == kwargs['brand']]
            
        if 'product' in kwargs and kwargs['product']:
            df = df[df['product'].str.contains(kwargs['product'], case=False, na=False)]
            
        if 'has_rating' in kwargs and kwargs['has_rating'] is not None:
            df = df[df['has_rating'] == kwargs['has_rating']]
            
        if 'min_date' in kwargs and kwargs['min_date']:
            df = df[df['date'] >= pd.to_datetime(kwargs['min_date'])]
            
        if 'max_date' in kwargs and kwargs['max_date']:
            df = df[df['date'] <= pd.to_datetime(kwargs['max_date'])]
            
        if 'source' in kwargs and kwargs['source']:
            df = df[df['source'] == kwargs['source']]
            
        return df
    
    def get_brands(self):
        """
        Get list of unique brands in the dataset
        
        Returns:
            list: Unique brand names
        """
        if self.data is None:
            self.load_data()
            
        return sorted(self.data['brand'].dropna().unique().tolist())
    
    def get_products(self, brand=None):
        """
        Get list of products, optionally filtered by brand
        
        Args:
            brand: Filter products by brand
            
        Returns:
            list: Product names
        """
        if self.data is None:
            self.load_data()
            
        df = self.data
        if brand:
            df = df[df['brand'] == brand]
            
        return sorted(df['product'].dropna().unique().tolist())
    
    def get_statistics(self):
        """
        Get dataset statistics
        
        Returns:
            dict: Dataset statistics
        """
        if self.data is None:
            self.load_data()
            
        return {
            'total_reviews': len(self.data),
            'reviews_with_ratings': self.data['has_rating'].sum(),
            'unique_brands': self.data['brand'].nunique(),
            'unique_products': self.data['product'].nunique(),
            'date_range': {
                'min': self.data['date'].min(),
                'max': self.data['date'].max()
            },
            'rating_distribution': self.data['rating'].value_counts().to_dict() if 'rating' in self.data else {},
            'source_distribution': self.data['source'].value_counts().to_dict() if 'source' in self.data else {},
            'avg_review_length': self.data['review_text'].str.len().mean() if 'review_text' in self.data else 0
        }
    
    def prepare_for_ml(self, include_columns=None):
        """
        Prepare dataset for machine learning
        
        Args:
            include_columns: Specific columns to include
            
        Returns:
            pandas.DataFrame: ML-ready dataset
        """
        if self.data is None:
            self.load_data()
            
        if include_columns is None:
            include_columns = self.config['ml_features']
            
        df = self.data[include_columns].copy()
        
        # Handle missing values
        if 'rating' in df.columns:
            df['rating'] = df['rating'].fillna(df['rating'].median())
            
        if 'review_text' in df.columns:
            df['review_text'] = df['review_text'].fillna('')
            
        return df
    
    def export_for_analysis(self, output_path=None):
        """
        Export the dataset for external analysis
        
        Args:
            output_path: Path to save the exported data
            
        Returns:
            str: Path where data was saved
        """
        if self.data is None:
            self.load_data()
            
        if output_path is None:
            output_path = Path('exported_dataset.csv')
            
        self.data.to_csv(output_path, index=False)
        print(f"✅ Dataset exported to: {output_path}")
        
        return str(output_path)

# Singleton instance
_data_loader = None

def get_data_loader():
    """
    Get singleton instance of DataLoader
    
    Returns:
        DataLoader: The data loader instance
    """
    global _data_loader
    if _data_loader is None:
        _data_loader = DataLoader()
    return _data_loader

# Convenience functions
def load_dataset():
    """Load the primary dataset"""
    return get_data_loader().load_data()

def get_sample(n=100):
    """Get a sample of the dataset"""
    return get_data_loader().get_sample_data(n)

def get_brands():
    """Get list of brands"""
    return get_data_loader().get_brands()

def get_products(brand=None):
    """Get list of products"""
    return get_data_loader().get_products(brand)

def get_stats():
    """Get dataset statistics"""
    return get_data_loader().get_statistics()

if __name__ == "__main__":
    # Test the data loader
    print("Testing Data Loader...")
    print("=" * 50)
    
    loader = get_data_loader()
    
    # Load data
    df = loader.load_data()
    print(f"\n✅ Loaded {len(df)} reviews")
    
    # Get statistics
    stats = loader.get_statistics()
    print(f"\n📊 Dataset Statistics:")
    print(f"  • Total reviews: {stats['total_reviews']}")
    print(f"  • Reviews with ratings: {stats['reviews_with_ratings']}")
    print(f"  • Unique brands: {stats['unique_brands']}")
    print(f"  • Unique products: {stats['unique_products']}")
    
    # Get brands
    brands = loader.get_brands()
    print(f"\n🏷️ Top 10 Brands:")
    for brand in brands[:10]:
        print(f"  • {brand}")
    
    print("\n✨ Data loader is working correctly!")