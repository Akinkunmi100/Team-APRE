"""
Dataset Configuration for AI Phone Review Engine
This file centralizes all dataset paths and settings
"""

import os
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent  # Fixed: parent.parent is the project root
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# ============================================================================
# MAIN DATASET CONFIGURATION
# ============================================================================

# PRIMARY DATASET - This is the main dataset to use for all operations
# Updated to use hybrid preprocessed dataset with sentiment and aspects
PRIMARY_DATASET = PROCESSED_DIR / "final_dataset_hybrid_preprocessed.csv"

# Alternative datasets (for reference or fallback)
ALTERNATIVE_DATASETS = {
    "original": PROCESSED_DIR / "final_dataset_streamlined.csv",
    "enhanced": PROCESSED_DIR / "final_dataset_enhanced.csv",
    "lightweight": PROCESSED_DIR / "final_dataset_lightweight.csv",
    "sample": PROCESSED_DIR / "final_dataset_sample.csv"
}

# ============================================================================
# DATASET SETTINGS
# ============================================================================

DATASET_CONFIG = {
    "primary_dataset_path": str(PRIMARY_DATASET),
    "encoding": "utf-8",
    "separator": ",",
    "has_header": True,
    "date_column": "date",
    "text_column": "review_text",
    "rating_column": "rating",
    "product_column": "product",
    "brand_column": "brand",
    "user_column": "user_id",
    "source_column": "source",
    
    # Data quality thresholds
    "min_review_length": 20,  # Minimum characters for valid review
    "max_review_length": 5000,  # Maximum characters to process
    "valid_rating_range": (1, 5),  # Valid rating values
    
    # Processing settings
    "batch_size": 100,  # For batch processing
    "cache_enabled": True,
    "cache_dir": str(DATA_DIR / "cache"),
    
    # Feature columns to use
    "feature_columns": [
        "review_id",
        "user_id", 
        "product",
        "brand",
        "review_text",
        "rating",
        "date",
        "source",
        "has_rating",
        "review_year",
        "review_month"
    ],
    
    # Columns for ML models
    "ml_features": [
        "review_text",
        "rating",
        "brand",
        "product"
    ],
    
    # Statistics about the cleaned dataset
    "dataset_stats": {
        "total_reviews": 4647,
        "reviews_with_ratings": 1861,
        "unique_brands": 60,
        "unique_products": 241,
        "avg_review_length": 189,
        "date_range": "2018-2025",
        "primary_source": "GSM Arena",
        "cleaned_date": "2025-09-12"
    }
}

# ============================================================================
# DATASET LOADER FUNCTION
# ============================================================================

def load_primary_dataset():
    """
    Load the primary cleaned dataset for the AI Phone Review Engine
    
    Returns:
        pandas.DataFrame: The loaded and validated dataset
    """
    import pandas as pd
    
    # Check if dataset exists
    if not PRIMARY_DATASET.exists():
        raise FileNotFoundError(
            f"Primary dataset not found at {PRIMARY_DATASET}. "
            f"Please ensure 'final_dataset_streamlined.csv' exists in {PROCESSED_DIR}"
        )
    
    # Load the dataset
    try:
        df = pd.read_csv(
            PRIMARY_DATASET,
            encoding=DATASET_CONFIG["encoding"],
            sep=DATASET_CONFIG["separator"]
        )
        
        # Convert date column to datetime
        if DATASET_CONFIG["date_column"] in df.columns:
            df[DATASET_CONFIG["date_column"]] = pd.to_datetime(
                df[DATASET_CONFIG["date_column"]], 
                errors='coerce'
            )
        
        # Convert rating to numeric
        if DATASET_CONFIG["rating_column"] in df.columns:
            df[DATASET_CONFIG["rating_column"]] = pd.to_numeric(
                df[DATASET_CONFIG["rating_column"]], 
                errors='coerce'
            )
        
        # Convert has_rating to boolean
        if "has_rating" in df.columns:
            df["has_rating"] = df["has_rating"].astype(str).str.lower() == 'true'
        
        print(f"Successfully loaded dataset: {len(df)} reviews")
        return df
        
    except Exception as e:
        raise Exception(f"Error loading dataset: {str(e)}")

def get_dataset_info():
    """
    Get information about the primary dataset
    
    Returns:
        dict: Dataset statistics and metadata
    """
    return {
        "path": str(PRIMARY_DATASET),
        "config": DATASET_CONFIG,
        "stats": DATASET_CONFIG["dataset_stats"]
    }

def validate_dataset(df):
    """
    Validate that a dataframe has all required columns
    
    Args:
        df: pandas.DataFrame to validate
    
    Returns:
        bool: True if valid, raises exception if not
    """
    required_columns = DATASET_CONFIG["feature_columns"]
    missing_columns = set(required_columns) - set(df.columns)
    
    if missing_columns:
        raise ValueError(f"Dataset missing required columns: {missing_columns}")
    
    return True

# ============================================================================
# EXPORT CONFIGURATION
# ============================================================================

__all__ = [
    'PRIMARY_DATASET',
    'DATASET_CONFIG',
    'load_primary_dataset',
    'get_dataset_info',
    'validate_dataset',
    'PROJECT_ROOT',
    'DATA_DIR',
    'PROCESSED_DIR'
]


