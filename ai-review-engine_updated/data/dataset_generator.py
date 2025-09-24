"""
Dataset Generator for AI Phone Review Engine
Creates sample datasets showing the complete data structure
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import random
from faker import Faker
import os

fake = Faker()

class DatasetGenerator:
    """Generate comprehensive sample datasets for the phone review engine"""
    
    def __init__(self):
        self.phone_brands = ['Apple', 'Samsung', 'Google', 'OnePlus', 'Xiaomi', 'OPPO', 'Vivo', 'Realme', 'Motorola', 'Nokia']
        self.phone_models = {
            'Apple': ['iPhone 15 Pro Max', 'iPhone 15 Pro', 'iPhone 15', 'iPhone 14 Pro Max', 'iPhone 14'],
            'Samsung': ['Galaxy S24 Ultra', 'Galaxy S24+', 'Galaxy S24', 'Galaxy Z Fold 5', 'Galaxy Z Flip 5'],
            'Google': ['Pixel 8 Pro', 'Pixel 8', 'Pixel 7a', 'Pixel Fold'],
            'OnePlus': ['OnePlus 12', 'OnePlus 11', 'OnePlus Nord 3'],
            'Xiaomi': ['Xiaomi 14 Pro', 'Xiaomi 13T', 'Redmi Note 13 Pro'],
            'OPPO': ['Find X7 Pro', 'Reno 11 Pro', 'A79'],
            'Vivo': ['X100 Pro', 'V29 Pro', 'Y36'],
            'Realme': ['GT 5 Pro', 'Realme 11 Pro+', 'C67'],
            'Motorola': ['Edge 40 Pro', 'Moto G84', 'Moto G54'],
            'Nokia': ['Nokia G60', 'Nokia C32', 'Nokia X30']
        }
        
        self.categories = ['Flagship', 'Mid-Range', 'Budget', 'Foldable', 'Gaming']
        self.colors = ['Black', 'White', 'Blue', 'Silver', 'Gold', 'Purple', 'Green', 'Red']
        self.storage_options = ['128GB', '256GB', '512GB', '1TB']
        
        # Review templates for realistic text generation
        self.positive_templates = [
            "Amazing {feature}! The {aspect} is absolutely incredible. Best phone I've ever owned.",
            "Love this phone! The {feature} exceeds expectations and the {aspect} is top-notch.",
            "Outstanding performance. The {feature} works flawlessly and {aspect} is impressive.",
            "Couldn't be happier with my purchase. {feature} is fantastic and {aspect} is excellent.",
            "Perfect phone for my needs. The {feature} is revolutionary and {aspect} performs great."
        ]
        
        self.negative_templates = [
            "Disappointed with the {feature}. The {aspect} doesn't live up to the hype.",
            "Not worth the price. {feature} is mediocre and {aspect} has issues.",
            "Expected better. The {feature} is underwhelming and {aspect} needs improvement.",
            "Having problems with {feature}. The {aspect} is not as advertised.",
            "Regret buying this. {feature} is poor and {aspect} constantly fails."
        ]
        
        self.neutral_templates = [
            "The {feature} is okay. {aspect} works as expected, nothing special.",
            "Average phone. {feature} is decent but {aspect} could be better.",
            "It's fine for the price. {feature} meets basic needs, {aspect} is acceptable.",
            "Mixed feelings about {feature}. The {aspect} is satisfactory.",
            "Does the job. {feature} is standard and {aspect} is adequate."
        ]
        
        self.features = ['camera', 'battery life', 'display', 'performance', 'build quality', '5G connectivity', 'face unlock']
        self.aspects = ['speed', 'reliability', 'user interface', 'value for money', 'durability', 'design', 'software']
    
    def generate_products_dataset(self, n_products=100):
        """Generate products dataset"""
        products = []
        
        for i in range(n_products):
            brand = random.choice(self.phone_brands)
            model = random.choice(self.phone_models[brand])
            category = random.choice(self.categories)
            
            # Price based on category
            if category == 'Flagship':
                base_price = random.randint(800, 1500)
            elif category == 'Mid-Range':
                base_price = random.randint(400, 799)
            elif category == 'Budget':
                base_price = random.randint(150, 399)
            elif category == 'Foldable':
                base_price = random.randint(1200, 2000)
            else:  # Gaming
                base_price = random.randint(500, 1000)
            
            # Generate specifications
            specs = {
                'display': f"{round(random.uniform(5.5, 7.2), 1)}\" {random.choice(['OLED', 'AMOLED', 'LCD', 'Super AMOLED'])}",
                'processor': random.choice(['Snapdragon 8 Gen 3', 'A17 Pro', 'Dimensity 9300', 'Exynos 2400', 'Tensor G3']),
                'ram': random.choice(['6GB', '8GB', '12GB', '16GB', '24GB']),
                'storage': random.choice(self.storage_options),
                'camera_main': f"{random.choice([48, 50, 64, 108, 200])}MP",
                'camera_front': f"{random.choice([12, 16, 24, 32])}MP",
                'battery': f"{random.choice([3500, 4000, 4500, 5000, 5500])}mAh",
                'charging': f"{random.choice([18, 25, 33, 45, 67, 100, 120])}W Fast Charging",
                '5g': random.choice([True, False]),
                'water_resistant': random.choice(['IP68', 'IP67', 'IP54', 'No']),
                'weight': f"{random.randint(160, 240)}g"
            }
            
            product = {
                'product_id': i + 1,
                'name': model,
                'brand': brand,
                'model': model.split()[-1],
                'category': category,
                'current_price': base_price,
                'original_price': base_price + random.randint(0, 200),
                'discount_percentage': round(random.uniform(0, 25), 1),
                'color_options': ', '.join(random.sample(self.colors, k=random.randint(2, 5))),
                'storage_options': ', '.join(random.sample(self.storage_options, k=random.randint(1, 4))),
                'release_date': fake.date_between(start_date='-2y', end_date='today'),
                'in_stock': random.choice([True, True, True, False]),  # 75% in stock
                'overall_rating': round(random.uniform(3.0, 5.0), 1),
                'total_reviews': random.randint(10, 5000),
                'total_ratings': random.randint(50, 10000),
                'specifications': json.dumps(specs),
                'description': f"The {model} is a {category.lower()} smartphone from {brand} featuring cutting-edge technology and premium design.",
                'image_url': f"https://example.com/images/{model.lower().replace(' ', '_')}.jpg",
                'source_url': f"https://store.example.com/products/{model.lower().replace(' ', '-')}",
                'source_platform': random.choice(['Jumia', 'Amazon', 'Official Store', 'Temu', 'GSMArena']),
                'last_updated': datetime.now() - timedelta(days=random.randint(0, 30)),
                'created_at': datetime.now() - timedelta(days=random.randint(30, 365))
            }
            
            products.append(product)
        
        return pd.DataFrame(products)
    
    def generate_reviews_dataset(self, products_df, avg_reviews_per_product=50):
        """Generate reviews dataset"""
        reviews = []
        review_id = 1
        
        for _, product in products_df.iterrows():
            n_reviews = random.randint(10, avg_reviews_per_product * 2)
            
            for _ in range(n_reviews):
                # Determine sentiment based on product rating
                if product['overall_rating'] >= 4.0:
                    sentiment_weights = [0.7, 0.2, 0.1]  # More positive
                elif product['overall_rating'] >= 3.0:
                    sentiment_weights = [0.3, 0.4, 0.3]  # Balanced
                else:
                    sentiment_weights = [0.2, 0.3, 0.5]  # More negative
                
                sentiment = np.random.choice(['positive', 'neutral', 'negative'], p=sentiment_weights)
                
                # Generate review text based on sentiment
                if sentiment == 'positive':
                    template = random.choice(self.positive_templates)
                    rating = random.uniform(4.0, 5.0)
                    helpful_multiplier = 1.5
                elif sentiment == 'negative':
                    template = random.choice(self.negative_templates)
                    rating = random.uniform(1.0, 3.0)
                    helpful_multiplier = 1.2
                else:
                    template = random.choice(self.neutral_templates)
                    rating = random.uniform(2.5, 4.0)
                    helpful_multiplier = 1.0
                
                # Generate review text
                feature = random.choice(self.features)
                aspect = random.choice(self.aspects)
                review_text = template.format(feature=feature, aspect=aspect)
                
                # Add more details to review
                if random.random() > 0.5:
                    review_text += f" I've been using it for {random.randint(1, 12)} months now."
                if random.random() > 0.7:
                    review_text += f" The {random.choice(self.features)} is also worth mentioning."
                
                review = {
                    'review_id': review_id,
                    'product_id': product['product_id'],
                    'user_id': random.randint(1, 1000),
                    'username': fake.user_name(),
                    'rating': round(rating, 1),
                    'review_title': f"{sentiment.capitalize()} experience with {product['name']}",
                    'review_text': review_text,
                    'sentiment': sentiment,
                    'sentiment_confidence': round(random.uniform(0.7, 0.99), 2),
                    'aspects_mentioned': json.dumps({
                        feature: sentiment,
                        aspect: sentiment
                    }),
                    'verified_purchase': random.choice([True, True, True, False]),  # 75% verified
                    'helpful_count': int(random.randint(0, 100) * helpful_multiplier),
                    'unhelpful_count': random.randint(0, 20),
                    'review_date': fake.date_time_between(start_date='-1y', end_date='now'),
                    'purchase_date': fake.date_time_between(start_date='-2y', end_date='-1M'),
                    'is_fake': random.choice([False] * 19 + [True]),  # 5% fake reviews
                    'fake_confidence': round(random.uniform(0.05, 0.95), 2) if random.random() > 0.95 else 0.05,
                    'language': 'en',
                    'source': product['source_platform'],
                    'pros': ', '.join(random.sample(self.features, k=random.randint(1, 3))),
                    'cons': ', '.join(random.sample(self.aspects, k=random.randint(0, 2))),
                    'would_recommend': sentiment == 'positive',
                    'created_at': datetime.now() - timedelta(days=random.randint(0, 365))
                }
                
                reviews.append(review)
                review_id += 1
        
        return pd.DataFrame(reviews)
    
    def generate_users_dataset(self, n_users=1000):
        """Generate users dataset"""
        users = []
        
        for i in range(n_users):
            user = {
                'user_id': i + 1,
                'username': fake.user_name(),
                'email': fake.email(),
                'full_name': fake.name(),
                'age': random.randint(18, 70),
                'gender': random.choice(['M', 'F', 'Other', None]),
                'location': fake.city(),
                'country': fake.country(),
                'join_date': fake.date_between(start_date='-3y', end_date='today'),
                'total_reviews': random.randint(0, 100),
                'total_ratings': random.randint(0, 200),
                'helpful_votes_received': random.randint(0, 500),
                'verified_reviewer': random.choice([True, False]),
                'reviewer_ranking': random.choice(['Top Reviewer', 'Helpful Reviewer', 'New Reviewer', None]),
                'interests': ', '.join(random.sample(self.categories, k=random.randint(1, 3))),
                'preferred_brands': ', '.join(random.sample(self.phone_brands, k=random.randint(1, 4))),
                'budget_range': random.choice(['$0-300', '$300-600', '$600-1000', '$1000+']),
                'last_active': datetime.now() - timedelta(days=random.randint(0, 30)),
                'created_at': datetime.now() - timedelta(days=random.randint(30, 1000))
            }
            users.append(user)
        
        return pd.DataFrame(users)
    
    def generate_analysis_dataset(self, reviews_df):
        """Generate analysis results dataset"""
        analyses = []
        
        # Group reviews by product
        for product_id in reviews_df['product_id'].unique()[:50]:  # Analyze first 50 products:
            product_reviews = reviews_df[reviews_df['product_id'] == product_id]
            
            # Calculate sentiment distribution
            sentiment_dist = product_reviews['sentiment'].value_counts(normalize=True).to_dict()
            
            # Extract common aspects
            all_aspects = []
            for aspects_str in product_reviews['aspects_mentioned'].dropna():
                try:
                    aspects = json.loads(aspects_str)
                    all_aspects.extend(aspects.keys())
                except:
                    pass
            
            aspect_counts = pd.Series(all_aspects).value_counts().head(5).to_dict()
            
            analysis = {
                'analysis_id': len(analyses) + 1,
                'product_id': product_id,
                'analysis_date': datetime.now(),
                'total_reviews_analyzed': len(product_reviews),
                'sentiment_distribution': json.dumps(sentiment_dist),
                'positive_percentage': sentiment_dist.get('positive', 0) * 100,
                'neutral_percentage': sentiment_dist.get('neutral', 0) * 100,
                'negative_percentage': sentiment_dist.get('negative', 0) * 100,
                'average_rating': product_reviews['rating'].mean(),
                'rating_std': product_reviews['rating'].std(),
                'top_positive_aspects': json.dumps(aspect_counts),
                'top_negative_aspects': json.dumps({k: v for k, v in aspect_counts.items() if v > 5}),
                'fake_review_percentage': (product_reviews['is_fake'].sum() / len(product_reviews)) * 100,
                'verified_purchase_percentage': (product_reviews['verified_purchase'].sum() / len(product_reviews)) * 100,
                'summary': f"Analysis of {len(product_reviews)} reviews shows {sentiment_dist.get('positive', 0)*100:.1f}% positive sentiment",
                'insights': json.dumps([
                    f"Most discussed feature: {list(aspect_counts.keys())[0] if aspect_counts else 'N/A'}",
                    f"Average rating: {product_reviews['rating'].mean():.1f}/5",
                    f"Fake review rate: {(product_reviews['is_fake'].sum() / len(product_reviews)) * 100:.1f}%"
                ]),
                'created_at': datetime.now()
            }
            
            analyses.append(analysis)
        
        return pd.DataFrame(analyses)
    
    def generate_comparison_dataset(self, products_df):
        """Generate product comparison dataset"""
        comparisons = []
        
        # Generate 50 comparisons
        for i in range(50):
            # Select two random products
            products = products_df.sample(n=2)
            product1 = products.iloc[0]
            product2 = products.iloc[1]
            
            comparison = {
                'comparison_id': i + 1,
                'product1_id': product1['product_id'],
                'product2_id': product2['product_id'],
                'product1_name': product1['name'],
                'product2_name': product2['name'],
                'price_difference': abs(product1['current_price'] - product2['current_price']),
                'rating_difference': abs(product1['overall_rating'] - product2['overall_rating']),
                'winner_product_id': product1['product_id'] if product1['overall_rating'] > product2['overall_rating'] else product2['product_id'],
                'comparison_aspects': json.dumps({
                    'price': 'product1' if product1['current_price'] < product2['current_price'] else 'product2',
                    'rating': 'product1' if product1['overall_rating'] > product2['overall_rating'] else 'product2',
                    'reviews': 'product1' if product1['total_reviews'] > product2['total_reviews'] else 'product2'
                }),
                'user_preference': random.choice(['product1', 'product2']),
                'comparison_date': datetime.now() - timedelta(days=random.randint(0, 30)),
                'created_at': datetime.now() - timedelta(days=random.randint(0, 60))
            }
            
            comparisons.append(comparison)
        
        return pd.DataFrame(comparisons)
    
    def generate_scraping_jobs_dataset(self):
        """Generate scraping jobs dataset"""
        jobs = []
        
        for i in range(20):
            status = random.choice(['completed', 'completed', 'completed', 'failed', 'running'])
            
            job = {
                'job_id': fake.uuid4(),
                'platform': random.choice(['Jumia', 'Temu', 'GSMArena', 'Amazon']),
                'query': random.choice(['iPhone', 'Samsung', 'Budget phones', 'Gaming phones', 'Flagship 2024']),
                'status': status,
                'products_found': random.randint(0, 100) if status == 'completed' else 0,
                'reviews_scraped': random.randint(0, 5000) if status == 'completed' else 0,
                'started_at': datetime.now() - timedelta(hours=random.randint(1, 72)),
                'completed_at': datetime.now() - timedelta(minutes=random.randint(1, 60)) if status == 'completed' else None,
                'error_message': fake.sentence() if status == 'failed' else None,
                'created_at': datetime.now() - timedelta(days=random.randint(0, 30))
            }
            
            jobs.append(job)
        
        return pd.DataFrame(jobs)
    
    def generate_user_activity_dataset(self, users_df, products_df):
        """Generate user activity/interaction dataset"""
        activities = []
        
        for _ in range(1000):
            user = users_df.sample(n=1).iloc[0]
            product = products_df.sample(n=1).iloc[0]
            
            activity = {
                'activity_id': len(activities) + 1,
                'user_id': user['user_id'],
                'product_id': product['product_id'],
                'activity_type': random.choice(['view', 'view', 'view', 'search', 'compare', 'review', 'purchase']),
                'session_id': fake.uuid4(),
                'duration_seconds': random.randint(10, 600),
                'clicked_reviews': random.randint(0, 10),
                'clicked_specs': random.choice([True, False]),
                'added_to_compare': random.choice([True, False, False, False]),
                'search_query': fake.word() if random.random() > 0.7 else None,
                'referrer': random.choice(['google', 'direct', 'social', 'email', None]),
                'device_type': random.choice(['mobile', 'mobile', 'desktop', 'tablet']),
                'timestamp': datetime.now() - timedelta(days=random.randint(0, 90), 
                                                       hours=random.randint(0, 23),
                                                       minutes=random.randint(0, 59))
            }
            
            activities.append(activity)
        
        return pd.DataFrame(activities)
    
    def save_datasets(self, output_dir='data/sample_datasets'):
        """Generate and save all datasets"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("🔄 Generating datasets...")
        
        # Generate all datasets
        products_df = self.generate_products_dataset(100)
        print(f"✅ Generated {len(products_df)} products")
        
        reviews_df = self.generate_reviews_dataset(products_df, avg_reviews_per_product=30)
        print(f"✅ Generated {len(reviews_df)} reviews")
        
        users_df = self.generate_users_dataset(1000)
        print(f"✅ Generated {len(users_df)} users")
        
        analysis_df = self.generate_analysis_dataset(reviews_df)
        print(f"✅ Generated {len(analysis_df)} analysis results")
        
        comparison_df = self.generate_comparison_dataset(products_df)
        print(f"✅ Generated {len(comparison_df)} comparisons")
        
        jobs_df = self.generate_scraping_jobs_dataset()
        print(f"✅ Generated {len(jobs_df)} scraping jobs")
        
        activity_df = self.generate_user_activity_dataset(users_df, products_df)
        print(f"✅ Generated {len(activity_df)} user activities")
        
        # Save to CSV files
        products_df.to_csv(f'{output_dir}/products.csv', index=False)
        reviews_df.to_csv(f'{output_dir}/reviews.csv', index=False)
        users_df.to_csv(f'{output_dir}/users.csv', index=False)
        analysis_df.to_csv(f'{output_dir}/analysis.csv', index=False)
        comparison_df.to_csv(f'{output_dir}/comparisons.csv', index=False)
        jobs_df.to_csv(f'{output_dir}/scraping_jobs.csv', index=False)
        activity_df.to_csv(f'{output_dir}/user_activity.csv', index=False)
        
        # Save to Excel with multiple sheets
        with pd.ExcelWriter(f'{output_dir}/complete_dataset.xlsx', engine='openpyxl') as writer:
            products_df.to_excel(writer, sheet_name='Products', index=False)
            reviews_df.head(100).to_excel(writer, sheet_name='Reviews_Sample', index=False)
            users_df.head(100).to_excel(writer, sheet_name='Users_Sample', index=False)
            analysis_df.to_excel(writer, sheet_name='Analysis', index=False)
            comparison_df.to_excel(writer, sheet_name='Comparisons', index=False)
            
        print(f"\n✅ All datasets saved to {output_dir}/")
        
        # Print statistics
        self.print_dataset_statistics(products_df, reviews_df, users_df)
        
        return {
            'products': products_df,
            'reviews': reviews_df,
            'users': users_df,
            'analysis': analysis_df,
            'comparisons': comparison_df,
            'scraping_jobs': jobs_df,
            'user_activity': activity_df
        }
    
    def print_dataset_statistics(self, products_df, reviews_df, users_df):
        """Print dataset statistics"""
        print("\n" + "="*60)
        print("📊 DATASET STATISTICS")
        print("="*60)
        
        print("\n📱 PRODUCTS:")
        print(f"  Total Products: {len(products_df)}")
        print(f"  Brands: {products_df['brand'].nunique()}")
        print(f"  Categories: {products_df['category'].value_counts().to_dict()}")
        print(f"  Price Range: ${products_df['current_price'].min():.2f} - ${products_df['current_price'].max():.2f}")
        print(f"  Avg Rating: {products_df['overall_rating'].mean():.2f}")
        
        print("\n💬 REVIEWS:")
        print(f"  Total Reviews: {len(reviews_df)}")
        print(f"  Sentiment Distribution:")
        for sentiment, count in reviews_df['sentiment'].value_counts().items():
            print(f"    - {sentiment}: {count} ({count/len(reviews_df)*100:.1f}%)")
        print(f"  Verified Purchases: {reviews_df['verified_purchase'].sum()} ({reviews_df['verified_purchase'].sum()/len(reviews_df)*100:.1f}%)")
        print(f"  Fake Reviews: {reviews_df['is_fake'].sum()} ({reviews_df['is_fake'].sum()/len(reviews_df)*100:.1f}%)")
        
        print("\n👥 USERS:")
        print(f"  Total Users: {len(users_df)}")
        print(f"  Verified Reviewers: {users_df['verified_reviewer'].sum()}")
        print(f"  Countries: {users_df['country'].nunique()}")
        
        print("\n" + "="*60)

if __name__ == "__main__":
    generator = DatasetGenerator()
    datasets = generator.save_datasets()
    
    print("\n🎉 Dataset generation complete!")
    print("📁 Files created:")
    print("  - products.csv")
    print("  - reviews.csv")
    print("  - users.csv")
    print("  - analysis.csv")
    print("  - comparisons.csv")
    print("  - scraping_jobs.csv")
    print("  - user_activity.csv")
    print("  - complete_dataset.xlsx")
