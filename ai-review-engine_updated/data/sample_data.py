"""
Generate sample datasets without external dependencies
"""

import pandas as pd
import json
import random
from datetime import datetime, timedelta
import os

# Create sample products
products_data = [
    {
        'product_id': 1,
        'name': 'iPhone 15 Pro Max',
        'brand': 'Apple',
        'category': 'Flagship',
        'current_price': 1299.99,
        'original_price': 1399.99,
        'overall_rating': 4.7,
        'total_reviews': 3456,
        'specifications': json.dumps({
            'display': '6.7" OLED',
            'processor': 'A17 Pro',
            'ram': '8GB',
            'storage': '256GB',
            'camera_main': '48MP',
            'battery': '4422mAh'
        })
    },
    {
        'product_id': 2,
        'name': 'Samsung Galaxy S24 Ultra',
        'brand': 'Samsung',
        'category': 'Flagship',
        'current_price': 1199.99,
        'original_price': 1299.99,
        'overall_rating': 4.6,
        'total_reviews': 2890,
        'specifications': json.dumps({
            'display': '6.8" AMOLED',
            'processor': 'Snapdragon 8 Gen 3',
            'ram': '12GB',
            'storage': '256GB',
            'camera_main': '200MP',
            'battery': '5000mAh'
        })
    },
    {
        'product_id': 3,
        'name': 'Google Pixel 8 Pro',
        'brand': 'Google',
        'category': 'Flagship',
        'current_price': 999.99,
        'original_price': 1099.99,
        'overall_rating': 4.5,
        'total_reviews': 1567,
        'specifications': json.dumps({
            'display': '6.7" OLED',
            'processor': 'Tensor G3',
            'ram': '12GB',
            'storage': '128GB',
            'camera_main': '50MP',
            'battery': '5050mAh'
        })
    },
    {
        'product_id': 4,
        'name': 'OnePlus 12',
        'brand': 'OnePlus',
        'category': 'Flagship',
        'current_price': 799.99,
        'original_price': 899.99,
        'overall_rating': 4.4,
        'total_reviews': 892,
        'specifications': json.dumps({
            'display': '6.82" AMOLED',
            'processor': 'Snapdragon 8 Gen 3',
            'ram': '12GB',
            'storage': '256GB',
            'camera_main': '50MP',
            'battery': '5400mAh'
        })
    },
    {
        'product_id': 5,
        'name': 'Xiaomi 14 Pro',
        'brand': 'Xiaomi',
        'category': 'Mid-Range',
        'current_price': 699.99,
        'original_price': 799.99,
        'overall_rating': 4.3,
        'total_reviews': 1234,
        'specifications': json.dumps({
            'display': '6.73" AMOLED',
            'processor': 'Snapdragon 8 Gen 3',
            'ram': '12GB',
            'storage': '256GB',
            'camera_main': '50MP',
            'battery': '4880mAh'
        })
    }
]

# Create sample reviews
reviews_data = [
    {
        'review_id': 1,
        'product_id': 1,
        'user_id': 101,
        'username': 'tech_enthusiast',
        'rating': 5.0,
        'review_title': 'Best iPhone Ever!',
        'review_text': 'The camera quality is absolutely stunning. The Action Button is a game changer. Battery life has improved significantly compared to my old iPhone 13 Pro.',
        'sentiment': 'positive',
        'sentiment_confidence': 0.95,
        'verified_purchase': True,
        'helpful_count': 234,
        'is_fake': False,
        'aspects_mentioned': json.dumps({'camera': 'positive', 'battery_life': 'positive', 'features': 'positive'})
    },
    {
        'review_id': 2,
        'product_id': 1,
        'user_id': 102,
        'username': 'casual_user',
        'rating': 4.0,
        'review_title': 'Great but expensive',
        'review_text': 'Love the phone, amazing display and performance. However, the price is quite steep and no charger in the box is disappointing.',
        'sentiment': 'positive',
        'sentiment_confidence': 0.78,
        'verified_purchase': True,
        'helpful_count': 156,
        'is_fake': False,
        'aspects_mentioned': json.dumps({'display': 'positive', 'performance': 'positive', 'price': 'negative'})
    },
    {
        'review_id': 3,
        'product_id': 2,
        'user_id': 103,
        'username': 'android_fan',
        'rating': 5.0,
        'review_title': 'Samsung nailed it!',
        'review_text': 'The S Pen functionality is unmatched. 200MP camera is insane. One UI 6 is smooth and feature-rich.',
        'sentiment': 'positive',
        'sentiment_confidence': 0.92,
        'verified_purchase': True,
        'helpful_count': 189,
        'is_fake': False,
        'aspects_mentioned': json.dumps({'camera': 'positive', 'features': 'positive', 'software': 'positive'})
    },
    {
        'review_id': 4,
        'product_id': 2,
        'user_id': 104,
        'username': 'photo_pro',
        'rating': 3.0,
        'review_title': 'Good but not perfect',
        'review_text': 'Camera is good but over-processes images. Battery drains quickly with heavy use. Build quality is excellent though.',
        'sentiment': 'neutral',
        'sentiment_confidence': 0.65,
        'verified_purchase': False,
        'helpful_count': 78,
        'is_fake': False,
        'aspects_mentioned': json.dumps({'camera': 'neutral', 'battery_life': 'negative', 'build_quality': 'positive'})
    },
    {
        'review_id': 5,
        'product_id': 3,
        'user_id': 105,
        'username': 'pixel_lover',
        'rating': 4.5,
        'review_title': 'AI features are amazing',
        'review_text': 'Best computational photography. Magic Eraser and Best Take features are revolutionary. Tensor G3 handles AI tasks brilliantly.',
        'sentiment': 'positive',
        'sentiment_confidence': 0.88,
        'verified_purchase': True,
        'helpful_count': 267,
        'is_fake': False,
        'aspects_mentioned': json.dumps({'camera': 'positive', 'ai_features': 'positive', 'processor': 'positive'})
    },
    {
        'review_id': 6,
        'product_id': 3,
        'user_id': 106,
        'username': 'skeptical_buyer',
        'rating': 2.0,
        'review_title': 'Overheating issues',
        'review_text': 'Phone gets very hot during normal use. Battery life is disappointing. Google needs to fix these Tensor issues.',
        'sentiment': 'negative',
        'sentiment_confidence': 0.85,
        'verified_purchase': True,
        'helpful_count': 145,
        'is_fake': False,
        'aspects_mentioned': json.dumps({'heating': 'negative', 'battery_life': 'negative', 'processor': 'negative'})
    },
    {
        'review_id': 7,
        'product_id': 4,
        'user_id': 107,
        'username': 'value_hunter',
        'rating': 5.0,
        'review_title': 'Best value flagship',
        'review_text': 'Incredible performance for the price. 100W charging is insanely fast. OxygenOS is clean and smooth.',
        'sentiment': 'positive',
        'sentiment_confidence': 0.91,
        'verified_purchase': True,
        'helpful_count': 198,
        'is_fake': False,
        'aspects_mentioned': json.dumps({'value': 'positive', 'charging': 'positive', 'software': 'positive'})
    },
    {
        'review_id': 8,
        'product_id': 4,
        'user_id': 108,
        'username': 'camera_critic',
        'rating': 3.5,
        'review_title': 'Good phone, average camera',
        'review_text': 'Performance is flagship level but camera is just okay. Hasselblad branding doesnt add much. Software updates could be better.',
        'sentiment': 'neutral',
        'sentiment_confidence': 0.70,
        'verified_purchase': False,
        'helpful_count': 89,
        'is_fake': False,
        'aspects_mentioned': json.dumps({'performance': 'positive', 'camera': 'neutral', 'software_updates': 'negative'})
    },
    {
        'review_id': 9,
        'product_id': 5,
        'user_id': 109,
        'username': 'budget_conscious',
        'rating': 4.5,
        'review_title': 'Premium features at mid-range price',
        'review_text': 'Leica cameras are fantastic. Build quality feels premium. MIUI has improved a lot. Great value for money.',
        'sentiment': 'positive',
        'sentiment_confidence': 0.87,
        'verified_purchase': True,
        'helpful_count': 156,
        'is_fake': False,
        'aspects_mentioned': json.dumps({'camera': 'positive', 'build_quality': 'positive', 'value': 'positive'})
    },
    {
        'review_id': 10,
        'product_id': 5,
        'user_id': 110,
        'username': 'xiaomi_user',
        'rating': 1.0,
        'review_title': 'Too many ads',
        'review_text': 'MIUI is full of ads. Phone is good but software ruins the experience. Bloatware everywhere.',
        'sentiment': 'negative',
        'sentiment_confidence': 0.89,
        'verified_purchase': True,
        'helpful_count': 234,
        'is_fake': False,
        'aspects_mentioned': json.dumps({'software': 'negative', 'ads': 'negative', 'bloatware': 'negative'})
    }
]

# Create sample users
users_data = [
    {'user_id': 101, 'username': 'tech_enthusiast', 'email': 'tech@example.com', 'total_reviews': 45, 'verified_reviewer': True},
    {'user_id': 102, 'username': 'casual_user', 'email': 'casual@example.com', 'total_reviews': 12, 'verified_reviewer': True},
    {'user_id': 103, 'username': 'android_fan', 'email': 'android@example.com', 'total_reviews': 67, 'verified_reviewer': True},
    {'user_id': 104, 'username': 'photo_pro', 'email': 'photo@example.com', 'total_reviews': 89, 'verified_reviewer': False},
    {'user_id': 105, 'username': 'pixel_lover', 'email': 'pixel@example.com', 'total_reviews': 23, 'verified_reviewer': True},
    {'user_id': 106, 'username': 'skeptical_buyer', 'email': 'skeptic@example.com', 'total_reviews': 8, 'verified_reviewer': True},
    {'user_id': 107, 'username': 'value_hunter', 'email': 'value@example.com', 'total_reviews': 34, 'verified_reviewer': True},
    {'user_id': 108, 'username': 'camera_critic', 'email': 'camera@example.com', 'total_reviews': 56, 'verified_reviewer': False},
    {'user_id': 109, 'username': 'budget_conscious', 'email': 'budget@example.com', 'total_reviews': 19, 'verified_reviewer': True},
    {'user_id': 110, 'username': 'xiaomi_user', 'email': 'xiaomi@example.com', 'total_reviews': 5, 'verified_reviewer': True}
]

# Create sample analysis results
analysis_data = [
    {
        'analysis_id': 1,
        'product_id': 1,
        'total_reviews_analyzed': 3456,
        'positive_percentage': 78.5,
        'neutral_percentage': 15.2,
        'negative_percentage': 6.3,
        'average_rating': 4.7,
        'top_positive_aspects': json.dumps(['camera', 'display', 'performance', 'battery_life']),
        'top_negative_aspects': json.dumps(['price', 'no_charger']),
        'fake_review_percentage': 3.2,
        'summary': 'iPhone 15 Pro Max receives overwhelmingly positive reviews with praise for camera and performance.'
    },
    {
        'analysis_id': 2,
        'product_id': 2,
        'total_reviews_analyzed': 2890,
        'positive_percentage': 75.3,
        'neutral_percentage': 17.8,
        'negative_percentage': 6.9,
        'average_rating': 4.6,
        'top_positive_aspects': json.dumps(['camera', 's_pen', 'display', 'features']),
        'top_negative_aspects': json.dumps(['battery_life', 'price', 'size']),
        'fake_review_percentage': 4.1,
        'summary': 'Galaxy S24 Ultra praised for versatility and S Pen, but some concerns about battery life.'
    },
    {
        'analysis_id': 3,
        'product_id': 3,
        'total_reviews_analyzed': 1567,
        'positive_percentage': 72.1,
        'neutral_percentage': 18.4,
        'negative_percentage': 9.5,
        'average_rating': 4.5,
        'top_positive_aspects': json.dumps(['ai_features', 'camera', 'software', 'updates']),
        'top_negative_aspects': json.dumps(['heating', 'battery_life', 'tensor_chip']),
        'fake_review_percentage': 2.8,
        'summary': 'Pixel 8 Pro excels in AI and photography but faces criticism for Tensor G3 heating issues.'
    },
    {
        'analysis_id': 4,
        'product_id': 4,
        'total_reviews_analyzed': 892,
        'positive_percentage': 74.8,
        'neutral_percentage': 19.1,
        'negative_percentage': 6.1,
        'average_rating': 4.4,
        'top_positive_aspects': json.dumps(['value', 'performance', 'charging_speed', 'display']),
        'top_negative_aspects': json.dumps(['camera', 'software_updates']),
        'fake_review_percentage': 3.5,
        'summary': 'OnePlus 12 offers excellent value with top performance, though camera lags behind competitors.'
    },
    {
        'analysis_id': 5,
        'product_id': 5,
        'total_reviews_analyzed': 1234,
        'positive_percentage': 68.9,
        'neutral_percentage': 21.3,
        'negative_percentage': 9.8,
        'average_rating': 4.3,
        'top_positive_aspects': json.dumps(['camera', 'build_quality', 'value', 'performance']),
        'top_negative_aspects': json.dumps(['ads', 'miui', 'bloatware']),
        'fake_review_percentage': 5.6,
        'summary': 'Xiaomi 14 Pro delivers premium hardware at competitive price but MIUI remains divisive.'
    }
]

# Save datasets
def save_sample_datasets():
    """Save all sample datasets to CSV files"""
    
    # Create data directory if it doesn't exist
    os.makedirs('data/sample_datasets', exist_ok=True)
    
    # Convert to DataFrames
    products_df = pd.DataFrame(products_data)
    reviews_df = pd.DataFrame(reviews_data)
    users_df = pd.DataFrame(users_data)
    analysis_df = pd.DataFrame(analysis_data)
    
    # Save to CSV
    products_df.to_csv('data/sample_datasets/products.csv', index=False)
    reviews_df.to_csv('data/sample_datasets/reviews.csv', index=False)
    users_df.to_csv('data/sample_datasets/users.csv', index=False)
    analysis_df.to_csv('data/sample_datasets/analysis.csv', index=False)
    
    print("✅ Sample datasets created successfully!")
    print("\n📊 Dataset Statistics:")
    print(f"  - Products: {len(products_df)} items")
    print(f"  - Reviews: {len(reviews_df)} items")
    print(f"  - Users: {len(users_df)} items")
    print(f"  - Analysis: {len(analysis_df)} items")
    print("\n📁 Files saved to: data/sample_datasets/")
    
    return products_df, reviews_df, users_df, analysis_df

if __name__ == "__main__":
    save_sample_datasets()
