#!/usr/bin/env python3
"""
Test script to verify data processing functionality
"""

import pandas as pd
import sys
import os
from pathlib import Path

print("=" * 60)
print("DATA PROCESSING VERIFICATION TEST")
print("=" * 60)

# Test 1: Check if CSV file exists and is readable
print("\n1. Testing CSV file availability...")
csv_path = Path('final_dataset_streamlined_clean.csv')

if csv_path.exists():
    print(f"✅ CSV file exists: {csv_path}")
    print(f"   File size: {csv_path.stat().st_size:,} bytes")
    
    try:
        # Read first few rows to test structure
        df_sample = pd.read_csv(csv_path, nrows=5)
        print(f"✅ CSV is readable")
        print(f"   Columns: {list(df_sample.columns)}")
        print(f"   Sample shape: {df_sample.shape}")
        
        # Read full dataset
        df = pd.read_csv(csv_path)
        print(f"✅ Full dataset loaded successfully")
        print(f"   Total records: {len(df):,}")
        print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        sys.exit(1)
else:
    print(f"❌ CSV file not found: {csv_path}")
    sys.exit(1)

# Test 2: Check system orchestrator
print("\n2. Testing system orchestrator...")
try:
    sys.path.append('utils')
    from utils.system_orchestrator import EnhancedSystemOrchestrator
    
    print("✅ System orchestrator module imported")
    
    # Initialize orchestrator
    orchestrator = EnhancedSystemOrchestrator()
    print("✅ System orchestrator initialized")
    
    # Get system status
    status = orchestrator.get_system_status()
    print("✅ System status retrieved")
    
    # Print key status information
    print(f"\n   System Health: {status.get('system_health', 'unknown')}")
    print(f"   Data Sources Status:")
    for source, info in status.get('data_sources', {}).items():
        if isinstance(info, dict):
            print(f"     - {source}: {info.get('status', 'unknown')}")
    
    print(f"   Data Info:")
    data_info = status.get('data_info', {})
    print(f"     - Primary source: {data_info.get('primary_source', 'unknown')}")
    print(f"     - Total records: {data_info.get('total_records', 0):,}")
    print(f"     - Data quality score: {data_info.get('data_quality_score', 0):.1f}%")
    
except Exception as e:
    print(f"❌ Error with system orchestrator: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Test basic data processing operations
print("\n3. Testing basic data processing operations...")
try:
    # Test filtering (similar to phone search)
    sample_brands = df['brand'].unique()[:5] if 'brand' in df.columns else []
    print(f"✅ Sample brands available: {list(sample_brands)}")
    
    if len(sample_brands) > 0:
        brand_filter = df['brand'] == sample_brands[0]
        filtered_df = df[brand_filter]
        print(f"✅ Brand filtering works: {len(filtered_df)} records for {sample_brands[0]}")
    
    # Test rating analysis
    if 'rating' in df.columns:
        avg_rating = df['rating'].mean()
        rating_dist = df['rating'].value_counts().to_dict()
        print(f"✅ Rating analysis works: avg={avg_rating:.2f}")
    
    # Test text content
    if 'review_text' in df.columns:
        text_sample = df['review_text'].dropna().iloc[0] if len(df['review_text'].dropna()) > 0 else "No text"
        print(f"✅ Review text available: {len(text_sample)} chars in first review")
    
except Exception as e:
    print(f"❌ Error in basic data processing: {e}")

print("\n" + "=" * 60)
print("TEST SUMMARY")
print("=" * 60)

print("\nBased on the tests above:")
print("- If all tests show ✅, your data processing is working correctly")
print("- If any test shows ❌, there may be issues that need addressing")
print("\nThe system should be able to:")
print("- Load the CSV dataset successfully")
print("- Initialize the system orchestrator")
print("- Perform basic data filtering and analysis")
print("- Handle phone search and analytics operations")

print("\n" + "=" * 60)