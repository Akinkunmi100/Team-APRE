# AI Phone Review Engine - Complete System Overview

## 🏗️ System Architecture

Your AI Phone Review Engine consists of multiple interconnected modules, each serving a specific purpose. Here's the complete breakdown:

## 📁 File Structure & Modules

### 🎯 **MAIN EXECUTABLE FILES**

#### 1. **`main_engine.py`** (PRIMARY - MOST COMPLETE)
- **Purpose**: The complete, integrated system with ALL features
- **Run Command**: `streamlit run main_engine.py`
- **Features**: All 10+ modules integrated, beautiful UI, complete functionality
- **USE THIS FOR**: Full system demonstration with all advanced features

#### 2. **`app.py`** (ORIGINAL)
- **Purpose**: Original basic Streamlit application
- **Run Command**: `streamlit run app.py`
- **Features**: Basic review analysis, ABSA, simple UI
- **USE THIS FOR**: Simple, lightweight analysis

#### 3. **`app_with_recommendations.py`**
- **Purpose**: App focused on recommendation engine
- **Run Command**: `streamlit run app_with_recommendations.py`
- **Features**: Phone recommendations, filtering, comparison
- **USE THIS FOR**: Testing recommendation features

#### 4. **`app_chat_assistant.py`**
- **Purpose**: Conversational AI interface
- **Run Command**: `streamlit run app_chat_assistant.py`
- **Features**: Chat-based interaction, Q&A about phones
- **USE THIS FOR**: Natural language interactions

#### 5. **`app_realtime_dashboard.py`**
- **Purpose**: Real-time monitoring dashboard
- **Run Command**: `streamlit run app_realtime_dashboard.py`
- **Features**: Live metrics, real-time updates, monitoring
- **USE THIS FOR**: Monitoring system performance

#### 6. **`app_simple_search.py`**
- **Purpose**: Simplified search interface
- **Run Command**: `streamlit run app_simple_search.py`
- **Features**: Quick search, basic filtering
- **USE THIS FOR**: Quick phone searches

---

## 🧩 **CORE MODULES**

### `/models/` Directory

#### 1. **`recommendation_engine.py`**
```python
from models.recommendation_engine import PhoneRecommendationEngine
engine = PhoneRecommendationEngine()
recommendations = engine.recommend_phones(user_preferences)
```
- Collaborative & content-based filtering
- Hybrid recommendations
- Phone similarity analysis

#### 2. **`absa_model.py`**
```python
from models.absa_model import ABSASentimentAnalyzer
analyzer = ABSASentimentAnalyzer()
results = analyzer.analyze_aspects(review_text)
```
- Aspect-based sentiment analysis
- Feature extraction
- Sentiment scoring per aspect

#### 3. **`spam_detector.py`**
```python
from models.spam_detector import SpamDetector
detector = SpamDetector()
is_spam = detector.detect_spam(review)
```
- ML-based spam detection
- Pattern recognition
- Review authenticity verification

#### 4. **`market_analyzer.py`**
```python
from models.market_analyzer import MarketAnalyzer
analyzer = MarketAnalyzer()
trends = analyzer.analyze_market_trends(data)
```
- Market trend analysis
- Competitive insights
- Price tracking

#### 5. **`review_summarizer.py`**
```python
from models.review_summarizer import ReviewSummarizer
summarizer = ReviewSummarizer()
summary = summarizer.summarize_reviews(reviews)
```
- Automatic review summarization
- Key points extraction
- Pros/cons aggregation

---

### `/core/` Directory

#### 1. **`ai_engine.py`**
```python
from core.ai_engine import AIReviewEngine
ai_engine = AIReviewEngine()
analysis = ai_engine.comprehensive_analysis(phone_data)
```
- Central AI orchestrator
- Coordinates all AI models
- Manages processing pipeline

#### 2. **`nlp_core.py`**
```python
from core.nlp_core import NLPCore
nlp = NLPCore()
processed = nlp.process_text(text)
```
- Core NLP operations
- Text preprocessing
- Language detection

#### 3. **`personalization_engine.py`**
```python
from core.personalization_engine import PersonalizationCore
personalization = PersonalizationCore()
preferences = personalization.learn_preferences(user_history)
```
- User preference learning
- Behavior tracking
- Personalized scoring

---

### `/modules/` Directory (ADVANCED FEATURES)

#### 1. **`advanced_personalization.py`** ⭐ NEW
```python
from modules.advanced_personalization import PersonalizationEngine
engine = PersonalizationEngine()
profile = engine.create_user_profile(user_id)
recommendations = engine.get_personalized_recommendations(user_id)
```
**Features:**
- User profiles with trust scores
- Preference learning (implicit & explicit)
- Contextual recommendations
- A/B testing framework
- Behavioral analytics
- Alert system

#### 2. **`deeper_insights.py`** ⭐ NEW
```python
from modules.deeper_insights import DeeperInsightsEngine
insights = DeeperInsightsEngine()
analysis = insights.analyze_review(review)
report = insights.generate_insights_report(results)
```
**Features:**
- Emotion detection (8 emotions)
- Sarcasm & irony detection
- Cultural sentiment analysis
- Temporal pattern analysis
- Review helpfulness prediction

---

### `/scrapers/` Directory

#### 1. **`jumia_scraper.py`**
```python
from scrapers.jumia_scraper import JumiaScraper
scraper = JumiaScraper()
reviews = scraper.scrape_reviews(product_url)
```
- Jumia.com review scraping
- Pagination handling
- Data extraction

#### 2. **`amazon_scraper.py`**
```python
from scrapers.amazon_scraper import AmazonScraper
scraper = AmazonScraper()
reviews = scraper.scrape_reviews(product_asin)
```
- Amazon review scraping
- Rate limiting
- Anti-bot measures

#### 3. **`scraper_manager.py`**
```python
from scrapers.scraper_manager import ScraperManager
manager = ScraperManager()
all_reviews = manager.scrape_multiple_sources(phone_name)
```
- Manages multiple scrapers
- Aggregates data
- Handles errors

---

### `/utils/` Directory

#### 1. **`data_preprocessing.py`**
```python
from utils.data_preprocessing import DataPreprocessor
preprocessor = DataPreprocessor()
clean_data = preprocessor.preprocess_dataset(df)
```
- Data cleaning
- Text normalization
- Feature engineering

#### 2. **`visualization.py`**
```python
from utils.visualization import ReviewVisualizer
visualizer = ReviewVisualizer()
chart = visualizer.create_sentiment_chart(data)
```
- Chart generation
- Interactive visualizations
- Dashboard components

#### 3. **`metrics_calculator.py`**
```python
from utils.metrics_calculator import MetricsCalculator
calculator = MetricsCalculator()
metrics = calculator.calculate_performance_metrics(predictions, actuals)
```
- Performance metrics
- Statistical analysis
- Accuracy calculations

---

### `/database/` Directory

#### 1. **`database_manager.py`**
```python
from database.database_manager import DatabaseManager
db = DatabaseManager()
db.save_reviews(reviews)
data = db.get_phone_data(phone_id)
```
- SQLite/PostgreSQL management
- CRUD operations
- Query optimization

#### 2. **`cache_manager.py`**
```python
from database.cache_manager import CacheManager
cache = CacheManager()
cached_result = cache.get_or_compute(key, compute_function)
```
- Redis caching
- Performance optimization
- Cache invalidation

---

### `/api/` Directory

#### 1. **`main.py`** (FastAPI)
```bash
# Run the API server
uvicorn api.main:app --reload
```
- RESTful API endpoints
- Authentication
- Rate limiting

#### 2. **`endpoints.py`**
- API route definitions
- Request/response models
- Validation

---

## 🚀 **HOW TO RUN DIFFERENT CONFIGURATIONS**

### **Option 1: Full System (RECOMMENDED)**
```bash
# Run the complete integrated system with all features
streamlit run main_engine.py
```
**You get**: All features, beautiful UI, complete functionality

### **Option 2: API Server + Frontend**
```bash
# Terminal 1: Start the API server
uvicorn api.main:app --reload

# Terminal 2: Run the frontend
streamlit run app_with_recommendations.py
```
**You get**: Microservices architecture, scalable deployment

### **Option 3: Specific Module Testing**
```python
# Test individual modules
python -m models.recommendation_engine
python -m modules.deeper_insights
python -m modules.advanced_personalization
```
**You get**: Module-specific testing and development

### **Option 4: Jupyter Notebook Analysis**
```bash
jupyter notebook analysis_notebook.ipynb
```
**You get**: Interactive analysis and experimentation

---

## 📊 **FEATURE COMPARISON**

| Feature | main_engine.py | app.py | app_with_recommendations.py | API Server |
|---------|---------------|---------|----------------------------|------------|
| Basic Analysis | ✅ | ✅ | ✅ | ✅ |
| Recommendations | ✅ | ❌ | ✅ | ✅ |
| Personalization | ✅ | ❌ | ⚠️ | ✅ |
| Emotion Detection | ✅ | ❌ | ❌ | ✅ |
| Sarcasm Detection | ✅ | ❌ | ❌ | ✅ |
| Cultural Analysis | ✅ | ❌ | ❌ | ✅ |
| Temporal Patterns | ✅ | ❌ | ❌ | ✅ |
| Live Scraping | ✅ | ⚠️ | ⚠️ | ✅ |
| User Profiles | ✅ | ❌ | ❌ | ✅ |
| A/B Testing | ✅ | ❌ | ❌ | ✅ |
| Export Reports | ✅ | ⚠️ | ⚠️ | ✅ |
| Beautiful UI | ✅ | ⚠️ | ✅ | N/A |

---

## 🔧 **CONFIGURATION FILES**

### **`.env`** (Environment Variables)
```env
DATABASE_URL=sqlite:///reviews.db
REDIS_URL=redis://localhost:6379
API_KEY=your_api_key_here
SCRAPING_DELAY=2
MAX_WORKERS=4
```

### **`config.yaml`** (System Configuration)
```yaml
models:
  sentiment:
    model_name: "bert-base-uncased"
    batch_size: 32
  spam:
    threshold: 0.7
personalization:
  min_interactions: 5
  learning_rate: 0.01
```

### **`requirements.txt`** (Dependencies)
```txt
streamlit==1.28.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
transformers==4.30.0
plotly==5.15.0
fastapi==0.100.0
uvicorn==0.23.0
redis==4.6.0
beautifulsoup4==4.12.0
selenium==4.10.0
```

---

## 💡 **USAGE RECOMMENDATIONS**

### **For Development & Testing:**
1. Use `main_engine.py` for full system testing
2. Use individual module files for unit testing
3. Use Jupyter notebooks for data exploration

### **For Production:**
1. Deploy API server (`api/main.py`) on a cloud service
2. Use `main_engine.py` for the frontend
3. Implement caching with Redis
4. Use PostgreSQL for production database

### **For Demonstration:**
1. **Best Choice**: Run `streamlit run main_engine.py`
2. This shows ALL capabilities in one integrated interface
3. Has the most impressive UI and features

### **For Specific Use Cases:**
- **Quick Analysis**: Use `app_simple_search.py`
- **Recommendations Only**: Use `app_with_recommendations.py`
- **Chat Interface**: Use `app_chat_assistant.py`
- **Monitoring**: Use `app_realtime_dashboard.py`

---

## 🎯 **QUICK START GUIDE**

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up environment variables
cp .env.example .env
# Edit .env with your settings

# 3. Initialize database
python -m database.database_manager --init

# 4. Run the main application
streamlit run main_engine.py

# 5. Access at http://localhost:8501
```

---

## 📈 **SYSTEM CAPABILITIES SUMMARY**

Your AI Phone Review Engine can:
1. Analyze sentiment with aspect-based analysis
2. Detect spam and fake reviews
3. Provide personalized recommendations
4. Detect emotions and sarcasm
5. Analyze cultural sentiment variations
6. Identify temporal patterns
7. Scrape reviews from multiple sources
8. Generate comprehensive reports
9. Learn user preferences
10. Conduct A/B testing
11. Send personalized alerts
12. Predict review helpfulness
13. Analyze market trends
14. Provide conversational AI interface
15. Export data in multiple formats

---

## 🚦 **NEXT STEPS**

1. **Run the main system**: `streamlit run main_engine.py`
2. **Create a user profile** to test personalization
3. **Try emotion detection** with sarcastic reviews
4. **Generate a report** to see analytics
5. **Test recommendations** with different preferences

This is your complete, production-ready AI Phone Review Engine with all advanced features integrated!
