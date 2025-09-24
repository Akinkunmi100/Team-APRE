# 📊 Dataset Documentation - AI Phone Review Engine

## Overview
The AI Phone Review Engine uses a comprehensive multi-table relational database structure to store and analyze phone products, customer reviews, user behavior, and AI-generated insights. This document describes the complete dataset structure, relationships, and sample data formats.

## 🗂️ Database Schema

### Entity Relationship Diagram
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   PRODUCTS  │────<│   REVIEWS   │>────│    USERS    │
└─────────────┘     └─────────────┘     └─────────────┘
       │                   │                    │
       │                   │                    │
       ▼                   ▼                    ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  ANALYSIS   │     │ASPECT_SENT. │     │USER_ACTIVITY│
└─────────────┘     └─────────────┘     └─────────────┘
       │                                        │
       ▼                                        ▼
┌─────────────┐                         ┌─────────────┐
│ COMPARISONS │                         │SCRAPING_JOBS│
└─────────────┘                         └─────────────┘
```

## 📁 Dataset Tables

### 1. **PRODUCTS Table** 📱
Stores information about phone products.

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| product_id | INT | Primary key | 1 |
| name | VARCHAR(255) | Full product name | "iPhone 15 Pro Max" |
| brand | VARCHAR(100) | Manufacturer brand | "Apple" |
| model | VARCHAR(100) | Model identifier | "A3106" |
| category | VARCHAR(50) | Product category | "Flagship" |
| current_price | DECIMAL(10,2) | Current selling price | 1299.99 |
| original_price | DECIMAL(10,2) | Original/MSRP price | 1499.99 |
| discount_percentage | FLOAT | Current discount % | 13.3 |
| color_options | TEXT | Available colors (JSON) | "Black, Silver, Gold, Purple" |
| storage_options | TEXT | Storage variants | "128GB, 256GB, 512GB, 1TB" |
| release_date | DATE | Product release date | "2024-09-15" |
| in_stock | BOOLEAN | Stock availability | true |
| overall_rating | FLOAT | Average rating (1-5) | 4.7 |
| total_reviews | INT | Count of reviews | 3456 |
| total_ratings | INT | Count of ratings | 5678 |
| specifications | JSON | Technical specs | See below |
| description | TEXT | Product description | "The ultimate iPhone experience..." |
| image_url | VARCHAR(500) | Product image URL | "https://..." |
| source_url | VARCHAR(500) | Product page URL | "https://..." |
| source_platform | VARCHAR(50) | Data source | "Jumia" |
| last_updated | TIMESTAMP | Last update time | "2024-12-06 10:30:00" |
| created_at | TIMESTAMP | Record creation | "2024-11-01 09:00:00" |

#### Specifications JSON Structure:
```json
{
  "display": "6.7\" OLED",
  "processor": "A17 Pro",
  "ram": "8GB",
  "storage": "256GB",
  "camera_main": "48MP",
  "camera_front": "12MP",
  "battery": "4422mAh",
  "charging": "27W Fast Charging",
  "5g": true,
  "water_resistant": "IP68",
  "weight": "221g"
}
```

### 2. **REVIEWS Table** 💬
Stores customer reviews and ratings.

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| review_id | INT | Primary key | 1001 |
| product_id | INT | Foreign key to products | 1 |
| user_id | INT | Foreign key to users | 456 |
| username | VARCHAR(100) | Reviewer username | "john_doe123" |
| rating | FLOAT | Review rating (1-5) | 4.5 |
| review_title | VARCHAR(255) | Review headline | "Amazing camera quality!" |
| review_text | TEXT | Full review content | "I've been using this phone for..." |
| sentiment | VARCHAR(20) | AI-detected sentiment | "positive" |
| sentiment_confidence | FLOAT | Confidence score (0-1) | 0.92 |
| aspects_mentioned | JSON | Aspects and sentiments | See below |
| verified_purchase | BOOLEAN | Verified buyer | true |
| helpful_count | INT | Helpful votes | 234 |
| unhelpful_count | INT | Not helpful votes | 12 |
| review_date | DATETIME | Review posted date | "2024-12-01 14:30:00" |
| purchase_date | DATETIME | Product purchase date | "2024-11-15 10:00:00" |
| is_fake | BOOLEAN | AI-detected fake review | false |
| fake_confidence | FLOAT | Fake detection confidence | 0.08 |
| language | VARCHAR(10) | Review language | "en" |
| source | VARCHAR(50) | Review source platform | "Amazon" |
| pros | TEXT | Listed pros | "Battery life, Camera, Display" |
| cons | TEXT | Listed cons | "Price, No charger included" |
| would_recommend | BOOLEAN | Recommendation flag | true |
| created_at | TIMESTAMP | Record creation | "2024-12-01 14:35:00" |

#### Aspects JSON Structure:
```json
{
  "camera": "positive",
  "battery_life": "positive",
  "display": "positive",
  "price": "negative",
  "performance": "neutral"
}
```

### 3. **USERS Table** 👥
Stores user/reviewer information.

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| user_id | INT | Primary key | 456 |
| username | VARCHAR(100) | Unique username | "tech_reviewer_pro" |
| email | VARCHAR(255) | User email | "john@example.com" |
| full_name | VARCHAR(255) | Full name | "John Smith" |
| age | INT | User age | 28 |
| gender | VARCHAR(20) | Gender | "M" |
| location | VARCHAR(255) | City/Location | "New York" |
| country | VARCHAR(100) | Country | "United States" |
| join_date | DATE | Registration date | "2022-03-15" |
| total_reviews | INT | Reviews written | 45 |
| total_ratings | INT | Ratings given | 67 |
| helpful_votes_received | INT | Total helpful votes | 892 |
| verified_reviewer | BOOLEAN | Verified status | true |
| reviewer_ranking | VARCHAR(50) | User rank/badge | "Top Reviewer" |
| interests | TEXT | User interests | "Flagship, Gaming, Camera" |
| preferred_brands | TEXT | Brand preferences | "Apple, Samsung, Google" |
| budget_range | VARCHAR(50) | Typical budget | "$600-1000" |
| last_active | TIMESTAMP | Last activity | "2024-12-06 09:00:00" |
| created_at | TIMESTAMP | Account creation | "2022-03-15 10:00:00" |

### 4. **ANALYSIS Table** 📈
Stores AI analysis results for products.

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| analysis_id | INT | Primary key | 201 |
| product_id | INT | Foreign key to products | 1 |
| analysis_date | DATETIME | Analysis timestamp | "2024-12-06 08:00:00" |
| total_reviews_analyzed | INT | Reviews processed | 500 |
| sentiment_distribution | JSON | Sentiment breakdown | See below |
| positive_percentage | FLOAT | % positive reviews | 72.5 |
| neutral_percentage | FLOAT | % neutral reviews | 18.3 |
| negative_percentage | FLOAT | % negative reviews | 9.2 |
| average_rating | FLOAT | Mean rating | 4.3 |
| rating_std | FLOAT | Rating std deviation | 0.8 |
| top_positive_aspects | JSON | Best features | See below |
| top_negative_aspects | JSON | Common complaints | See below |
| fake_review_percentage | FLOAT | % fake reviews | 5.2 |
| verified_purchase_percentage | FLOAT | % verified buyers | 78.4 |
| summary | TEXT | AI-generated summary | "Overall positive reception..." |
| insights | JSON | Key insights | See below |
| created_at | TIMESTAMP | Analysis creation | "2024-12-06 08:05:00" |

#### Sentiment Distribution JSON:
```json
{
  "positive": 0.725,
  "neutral": 0.183,
  "negative": 0.092
}
```

#### Insights JSON:
```json
[
  "Camera quality is the most praised feature",
  "Battery life exceeds expectations for 87% of users",
  "Price point is the main concern for budget-conscious buyers",
  "5G performance shows significant improvement over previous model"
]
```

### 5. **ASPECT_SENTIMENTS Table** 🎯
Stores aspect-based sentiment analysis.

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| aspect_id | INT | Primary key | 301 |
| review_id | INT | Foreign key to reviews | 1001 |
| product_id | INT | Foreign key to products | 1 |
| aspect | VARCHAR(100) | Feature/aspect | "camera" |
| sentiment | VARCHAR(20) | Sentiment for aspect | "positive" |
| confidence | FLOAT | Confidence score | 0.89 |
| mentions_count | INT | Times mentioned | 45 |
| keywords | TEXT | Related keywords | "photo, picture, lens, zoom" |
| created_at | TIMESTAMP | Record creation | "2024-12-06 08:10:00" |

### 6. **COMPARISONS Table** ⚔️
Stores product comparison data.

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| comparison_id | INT | Primary key | 401 |
| product1_id | INT | First product ID | 1 |
| product2_id | INT | Second product ID | 2 |
| product1_name | VARCHAR(255) | First product name | "iPhone 15 Pro" |
| product2_name | VARCHAR(255) | Second product name | "Galaxy S24 Ultra" |
| price_difference | DECIMAL | Price delta | 100.00 |
| rating_difference | FLOAT | Rating delta | 0.2 |
| winner_product_id | INT | Better product ID | 1 |
| comparison_aspects | JSON | Detailed comparison | See below |
| user_preference | VARCHAR(20) | User choice | "product1" |
| comparison_date | DATETIME | Comparison time | "2024-12-05 15:00:00" |
| created_at | TIMESTAMP | Record creation | "2024-12-05 15:05:00" |

### 7. **USER_ACTIVITY Table** 📊
Tracks user interactions and behavior.

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| activity_id | INT | Primary key | 5001 |
| user_id | INT | Foreign key to users | 456 |
| product_id | INT | Foreign key to products | 1 |
| activity_type | VARCHAR(50) | Type of activity | "view" |
| session_id | VARCHAR(100) | Session identifier | "sess_abc123" |
| duration_seconds | INT | Time spent | 180 |
| clicked_reviews | INT | Reviews clicked | 5 |
| clicked_specs | BOOLEAN | Viewed specifications | true |
| added_to_compare | BOOLEAN | Added to comparison | false |
| search_query | VARCHAR(255) | Search terms used | "best camera phone" |
| referrer | VARCHAR(100) | Traffic source | "google" |
| device_type | VARCHAR(50) | Device used | "mobile" |
| timestamp | TIMESTAMP | Activity time | "2024-12-06 10:15:00" |

### 8. **SCRAPING_JOBS Table** 🌐
Tracks web scraping jobs and status.

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| job_id | VARCHAR(100) | Primary key (UUID) | "abc-123-def-456" |
| platform | VARCHAR(50) | Target platform | "Jumia" |
| query | VARCHAR(255) | Search query | "iPhone 15" |
| status | VARCHAR(20) | Job status | "completed" |
| products_found | INT | Products scraped | 45 |
| reviews_scraped | INT | Reviews scraped | 2340 |
| started_at | TIMESTAMP | Job start time | "2024-12-06 08:00:00" |
| completed_at | TIMESTAMP | Job end time | "2024-12-06 08:45:00" |
| error_message | TEXT | Error details if failed | null |
| created_at | TIMESTAMP | Job creation | "2024-12-06 07:59:00" |

## 📊 Sample Data Statistics

### Typical Dataset Size
- **Products**: 100-500 unique phones
- **Reviews**: 10,000-50,000 customer reviews
- **Users**: 1,000-10,000 registered users
- **Daily Activities**: 5,000-20,000 interactions
- **Analysis Records**: 1 per product per week

### Data Distribution
- **Sentiment**: ~60% positive, 25% neutral, 15% negative
- **Verified Purchases**: ~75% of reviews
- **Fake Reviews**: ~5% (detected by AI)
- **Average Reviews per Product**: 30-100
- **Average Rating**: 4.1/5.0

## 🔄 Data Pipeline

### Data Flow
1. **Scraping**: Web scrapers collect product and review data
2. **Preprocessing**: Text cleaning, normalization, spam detection
3. **AI Analysis**: Sentiment analysis, aspect extraction, fake detection
4. **Storage**: Structured storage in relational database
5. **Aggregation**: Statistical analysis and insights generation
6. **Recommendations**: ML models use data for personalization

### Update Frequency
- **Products**: Daily updates for price/stock
- **Reviews**: Real-time as scraped
- **Analysis**: Weekly batch processing
- **User Activity**: Real-time tracking
- **Recommendations**: Hourly model updates

## 🎯 Key Features

### Data Quality
- **Validation**: Schema validation on all inputs
- **Deduplication**: Automatic duplicate detection
- **Normalization**: Consistent formatting across sources
- **Versioning**: Historical tracking of changes

### AI Enhancement
- **Sentiment Analysis**: Multi-model ensemble approach
- **Aspect Extraction**: NER and dependency parsing
- **Fake Detection**: ML-based authenticity scoring
- **Summarization**: Automatic review summaries
- **Recommendations**: Collaborative and content filtering

## 📈 Usage Examples

### Query Examples

1. **Get top-rated phones**:
```sql
SELECT name, brand, overall_rating, total_reviews
FROM products
WHERE overall_rating >= 4.5
ORDER BY total_reviews DESC
LIMIT 10;
```

2. **Sentiment analysis for a product**:
```sql
SELECT 
  sentiment,
  COUNT(*) as count,
  AVG(rating) as avg_rating
FROM reviews
WHERE product_id = 1
GROUP BY sentiment;
```

3. **User engagement metrics**:
```sql
SELECT 
  DATE(timestamp) as date,
  COUNT(DISTINCT user_id) as unique_users,
  COUNT(*) as total_activities
FROM user_activity
WHERE timestamp >= DATE_SUB(NOW(), INTERVAL 7 DAY)
GROUP BY DATE(timestamp);
```

## 🚀 Getting Started

### Generate Sample Data
```python
from data.dataset_generator import DatasetGenerator

# Create generator
generator = DatasetGenerator()

# Generate all datasets
datasets = generator.save_datasets()

# Access individual datasets
products_df = datasets['products']
reviews_df = datasets['reviews']
users_df = datasets['users']
```

### Load Data into Database
```python
from database.models import db_manager, Product, Review

# Load products
for _, row in products_df.iterrows():
    product = Product(**row.to_dict())
    session.add(product)

session.commit()
```

## 📝 Data Dictionary

### Sentiment Values
- `positive`: Score > 0.6
- `neutral`: Score 0.4-0.6
- `negative`: Score < 0.4

### Categories
- `Flagship`: Premium phones ($800+)
- `Mid-Range`: Balance phones ($400-799)
- `Budget`: Entry phones (<$400)
- `Foldable`: Folding screen phones
- `Gaming`: Gaming-optimized phones

### Activity Types
- `view`: Product page view
- `search`: Search query
- `compare`: Added to comparison
- `review`: Posted review
- `purchase`: Completed purchase

## 🔐 Privacy & Security

### Data Protection
- **PII Handling**: User emails and names are hashed
- **GDPR Compliance**: Right to deletion implemented
- **Encryption**: Sensitive data encrypted at rest
- **Access Control**: Role-based permissions

### Data Retention
- **Reviews**: Permanent storage
- **User Activity**: 90 days rolling window
- **Scraping Logs**: 30 days
- **Analysis Results**: 1 year

---

*Last Updated: December 2024*
*Version: 1.0*
