# 🚀 Setup and Troubleshooting Guide

## Table of Contents
- [System Status Overview](#system-status-overview)
- [Installation Requirements](#installation-requirements)
- [Progressive Setup Guide](#progressive-setup-guide)
- [Common Issues and Solutions](#common-issues-and-solutions)
- [Testing and Validation](#testing-and-validation)
- [Performance Expectations](#performance-expectations)

## 🔍 System Status Overview

### ✅ What's Complete

1. **Comprehensive Architecture**
   - All modules are designed and documented
   - Modular structure for easy maintenance
   - Clear separation of concerns

2. **Core Code Structure**
   - Python files with proper class definitions
   - Type hints and documentation
   - Error handling frameworks

3. **Advanced Features**
   - Multiple AI models (DeBERTa, RoBERTa, ALBERT, ELECTRA)
   - Personalization engine
   - Real-time analytics
   - Multi-agent RAG system

4. **Documentation**
   - Extensive README
   - API documentation
   - Dataset requirements
   - Architecture diagrams

### ⚠️ What Needs Attention

#### 1. **Dependencies & Environment**

The system requires multiple Python packages across different domains:

```bash
# Core ML/AI packages
pip install transformers==4.36.0
pip install torch==2.1.0  # or tensorflow==2.15.0
pip install sentence-transformers==2.2.2
pip install scikit-learn==1.3.0

# NLP packages
pip install spacy==3.7.0
pip install textblob==0.17.1
pip install nltk==3.8.1
pip install vaderSentiment==3.3.2

# Web framework
pip install fastapi==0.105.0
pip install uvicorn==0.25.0
pip install streamlit==1.29.0
pip install gradio==4.10.0

# Database
pip install sqlalchemy==2.0.23
pip install psycopg2-binary==2.9.9
pip install alembic==1.13.0

# Data processing
pip install pandas==2.1.4
pip install numpy==1.24.3
pip install openpyxl==3.1.2  # For Excel files

# Visualization
pip install plotly==5.18.0
pip install matplotlib==3.8.2
pip install seaborn==0.13.0

# Optional: Real-time features
pip install redis==5.0.1
pip install kafka-python==2.0.2
pip install celery==5.3.4

# Optional: Advanced features
pip install optuna==3.5.0  # AutoML
pip install faiss-cpu==1.7.4  # Vector search
pip install chromadb==0.4.18  # Vector database
```

#### 2. **Configuration Files**

##### `.env` File Structure
Create a `.env` file in the root directory:

```env
# Database Configuration
DATABASE_URL=postgresql://username:password@localhost:5432/review_engine
REDIS_URL=redis://localhost:6379/0

# API Keys (OPTIONAL - See docs/API_KEYS_AND_AUTH.md for details)
# OPENAI_API_KEY=only_if_adding_gpt_features
# HUGGINGFACE_TOKEN=only_for_gated_models

# Application Settings
SECRET_KEY=your_secret_key_here_min_32_chars
ENVIRONMENT=development
DEBUG=True

# Model Settings
MODEL_CACHE_DIR=./models
MAX_SEQUENCE_LENGTH=512
BATCH_SIZE=32

# Scraping Settings
USER_AGENT=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36
SCRAPING_DELAY=2

# Feature Flags
ENABLE_REALTIME=False
ENABLE_PERSONALIZATION=True
ENABLE_DEEPER_INSIGHTS=True
```

##### `config/config.yaml` Structure
Create `config/config.yaml`:

```yaml
# Model Configuration
models:
  sentiment:
    primary: "microsoft/deberta-v3-large"
    fallback: "distilbert-base-uncased-finetuned-sst-2-english"
    cache_dir: "./models"
  
  summarization:
    model: "facebook/bart-large-cnn"
    max_length: 150
  
  ner:
    model: "dbmdz/bert-large-cased-finetuned-conll03-english"
  
  embeddings:
    model: "sentence-transformers/all-mpnet-base-v2"

# Database Configuration
database:
  pool_size: 20
  max_overflow: 40
  pool_timeout: 30
  echo: false

# API Configuration
api:
  rate_limit: 100
  rate_limit_period: 60
  max_request_size: 10485760  # 10MB
  cors_origins:
    - "http://localhost:3000"
    - "http://localhost:8501"

# Sentiment Analysis Settings
sentiment_analysis:
  confidence_threshold: 0.7
  aspects:
    - battery
    - camera
    - display
    - performance
    - price
    - build_quality
    - software
  sentiment_labels:
    - positive
    - negative
    - neutral

# Scraping Configuration
scraping:
  max_workers: 5
  timeout: 30
  retry_attempts: 3
  platforms:
    jumia:
      enabled: true
      base_url: "https://www.jumia.com"
    temu:
      enabled: true
      base_url: "https://www.temu.com"
    reddit:
      enabled: false
      subreddits:
        - "r/PickAnAndroidForMe"
        - "r/phones"
```

#### 3. **Database Setup**

##### PostgreSQL Installation

**Windows:**
```bash
# Download PostgreSQL installer from https://www.postgresql.org/download/windows/
# Run installer and remember your password
```

**macOS:**
```bash
brew install postgresql@15
brew services start postgresql@15
```

**Linux:**
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
```

##### Database Creation
```sql
-- Connect to PostgreSQL
psql -U postgres

-- Create database
CREATE DATABASE review_engine;

-- Create user
CREATE USER review_user WITH PASSWORD 'your_password';

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE review_engine TO review_user;

-- Exit
\q
```

##### Initialize Tables
```python
# Run this Python script
from database.models import db_manager

# Create all tables
db_manager.create_tables()
print("Database tables created successfully!")
```

#### 4. **Model Downloads**

Models will be automatically downloaded on first use, but you can pre-download them:

```python
from transformers import AutoModel, AutoTokenizer

models_to_download = [
    "microsoft/deberta-v3-large",
    "roberta-large-mnli",
    "albert-xxlarge-v2",
    "google/electra-large-discriminator",
    "facebook/bart-large-cnn",
    "j-hartmann/emotion-english-distilroberta-base"
]

for model_name in models_to_download:
    print(f"Downloading {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        print(f"✓ {model_name} downloaded successfully")
    except Exception as e:
        print(f"✗ Error downloading {model_name}: {e}")
```

**Storage Requirements:**
- DeBERTa-v3-large: ~1.5 GB
- RoBERTa-large: ~1.4 GB
- ALBERT-xxlarge: ~850 MB
- ELECTRA-large: ~1.3 GB
- Total: ~6-8 GB for all models

#### 5. **Data Preparation**

##### Fix Excel Data Issues
Your `Review Data.xlsx` needs cleaning:

```python
import pandas as pd

# Load the Excel file
df = pd.read_excel("Review Data.xlsx")

# Fix the rating column (currently contains dates)
# Map the actual ratings from StarRating column or create new ones
df['rating_fixed'] = df['StarRating'].fillna(
    df['RATING'].apply(lambda x: 3.0)  # Default to 3 if no StarRating
)

# Ensure review text is in correct column
df['review_text'] = df['REVIEW CONTENT'].fillna(df['Comment'])

# Save cleaned data
df[['review_text', 'rating_fixed', 'Model', 'Brand', 'REVIEW DATE']].to_csv(
    'reviews_cleaned.csv', 
    index=False
)
print("Data cleaned and saved to reviews_cleaned.csv")
```

## 📦 Progressive Setup Guide

### Phase 1: Minimal Setup (30 minutes)

**Goal:** Get basic sentiment analysis working

```bash
# Step 1: Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Step 2: Install minimal dependencies
pip install pandas numpy scikit-learn transformers torch streamlit

# Step 3: Test basic functionality
python -c "from utils.data_preprocessing import DataPreprocessor; print('✓ Basic setup complete')"
```

### Phase 2: Core Features (1 hour)

**Goal:** Enable database and API

```bash
# Step 1: Install additional dependencies
pip install fastapi uvicorn sqlalchemy psycopg2-binary

# Step 2: Setup PostgreSQL (see Database Setup section)

# Step 3: Create .env file (see Configuration section)

# Step 4: Initialize database
python -c "from database.models import db_manager; db_manager.create_tables()"

# Step 5: Start API server
uvicorn api.main:app --reload
```

### Phase 3: Advanced Features (2+ hours)

**Goal:** Enable all AI models and features

```bash
# Step 1: Install all dependencies
pip install -r requirements.txt

# Step 2: Download spaCy model
python -m spacy download en_core_web_sm

# Step 3: Download NLTK data
python -c "import nltk; nltk.download('all')"

# Step 4: Pre-download transformer models (optional, see Model Downloads)

# Step 5: Enable optional features in .env
# ENABLE_REALTIME=True
# ENABLE_PERSONALIZATION=True
# ENABLE_DEEPER_INSIGHTS=True

# Step 6: Start all services
# Terminal 1: API
uvicorn api.main:app --reload

# Terminal 2: Streamlit
streamlit run app.py

# Terminal 3 (Optional): Redis
redis-server

# Terminal 4 (Optional): Celery
celery -A tasks worker --loglevel=info
```

## 🐛 Common Issues and Solutions

### Issue 1: ModuleNotFoundError
```python
ModuleNotFoundError: No module named 'transformers'
```
**Solution:**
```bash
pip install transformers
# If still fails:
pip install --upgrade transformers
```

### Issue 2: CUDA/GPU Errors
```python
RuntimeError: CUDA out of memory
```
**Solution:**
```python
# In your code, force CPU usage:
import torch
device = torch.device('cpu')
# Or reduce batch size in config.yaml
```

### Issue 3: Database Connection Failed
```python
sqlalchemy.exc.OperationalError: could not connect to server
```
**Solution:**
1. Check PostgreSQL is running: `pg_isready`
2. Verify credentials in .env
3. Check firewall settings
4. Try connecting manually: `psql -U username -d database_name`

### Issue 4: Model Download Failures
```python
OSError: Can't load tokenizer for 'model-name'
```
**Solution:**
```bash
# Use HuggingFace CLI
huggingface-cli login
# Or set token in environment
export HUGGINGFACE_TOKEN=your_token
```

### Issue 5: Memory Issues
```python
MemoryError: Unable to allocate array
```
**Solution:**
1. Reduce batch size in processing
2. Use smaller models (distilbert instead of bert)
3. Process data in chunks
4. Increase system swap space

### Issue 6: Excel File Reading Error
```python
Missing optional dependency 'openpyxl'
```
**Solution:**
```bash
pip install openpyxl
```

## 🧪 Testing and Validation

### Basic Functionality Test
```python
# test_basic.py
def test_imports():
    """Test all core imports work"""
    try:
        from utils.data_preprocessing import DataPreprocessor
        from models.advanced_ai_model import AdvancedAIEngine
        from core.smart_search import SmartSearchEngine
        print("✓ All core imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_preprocessing():
    """Test data preprocessing"""
    from utils.data_preprocessing import DataPreprocessor
    
    preprocessor = DataPreprocessor()
    test_text = "This phone is amazing!"
    result = preprocessor.preprocess_review(test_text)
    
    assert 'cleaned' in result
    assert 'tokens' in result
    print("✓ Preprocessing works")
    return True

def test_sentiment():
    """Test basic sentiment analysis"""
    from textblob import TextBlob
    
    text = "This phone is great!"
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    
    assert sentiment > 0
    print(f"✓ Sentiment analysis works (score: {sentiment})")
    return True

if __name__ == "__main__":
    tests = [test_imports, test_preprocessing, test_sentiment]
    results = [test() for test in tests]
    
    if all(results):
        print("\n✅ All tests passed!")
    else:
        print(f"\n⚠️ {sum(results)}/{len(tests)} tests passed")
```

### API Health Check
```bash
# Check API is running
curl http://localhost:8000/health

# Expected response:
# {"status": "healthy", "timestamp": "2024-01-01T00:00:00"}
```

### Database Connection Test
```python
# test_database.py
from sqlalchemy import create_engine
from database.models import db_manager
import os

def test_database_connection():
    """Test database connectivity"""
    try:
        engine = create_engine(os.getenv('DATABASE_URL'))
        with engine.connect() as conn:
            result = conn.execute("SELECT 1")
            print("✓ Database connection successful")
            return True
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return False

if __name__ == "__main__":
    test_database_connection()
```

## 📊 Performance Expectations

### Initial Setup Times

| Component | Time | Storage | RAM |
|-----------|------|---------|-----|
| Basic Dependencies | 10-15 min | 500 MB | 2 GB |
| All Dependencies | 20-30 min | 2 GB | 4 GB |
| Model Downloads | 30-120 min | 6-8 GB | 8 GB |
| Database Setup | 10-15 min | 100 MB | 1 GB |
| First Run (with downloads) | 2-3 hours | 10 GB | 8 GB |

### Runtime Performance

| Operation | Time | CPU Usage | RAM Usage |
|-----------|------|-----------|-----------|
| Single Review Analysis | 0.5-2 sec | 20-40% | 500 MB |
| Batch (100 reviews) | 10-30 sec | 60-80% | 2 GB |
| Model Loading | 10-30 sec | 100% | 4-6 GB |
| API Response | 50-200 ms | 10-20% | 200 MB |
| Database Query | 10-50 ms | 5-10% | 100 MB |

### Optimization Tips

1. **Use GPU if available:**
```python
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

2. **Batch processing:**
```python
# Process in batches instead of one-by-one
batch_size = 32
for i in range(0, len(reviews), batch_size):
    batch = reviews[i:i+batch_size]
    process_batch(batch)
```

3. **Model caching:**
```python
# Load models once and reuse
class ModelCache:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.models = {}
        return cls._instance
```

4. **Use smaller models for development:**
```python
# Development: Use DistilBERT instead of BERT
model_name = "distilbert-base-uncased" if DEBUG else "bert-large-uncased"
```

## 🎯 Realistic Expectations

### What WILL Work Immediately
- ✅ Basic data analysis scripts
- ✅ Simple sentiment analysis with TextBlob
- ✅ Data preprocessing functions
- ✅ Visualization generation with static data
- ✅ Excel file analysis

### What NEEDS Setup
- ⚠️ Database connections (PostgreSQL required)
- ⚠️ API authentication (tokens needed)
- ⚠️ Advanced AI models (download required)
- ⚠️ Real-time features (Redis/Kafka needed)
- ⚠️ WebSocket notifications (Redis required)
- ⚠️ Production deployment (server configuration)

### Development vs Production

| Aspect | Development | Production |
|--------|------------|------------|
| Database | SQLite OK | PostgreSQL required |
| Models | Smaller models | Full models |
| Caching | Optional | Required |
| Workers | Single process | Multiple workers |
| Monitoring | Basic logging | Full monitoring |
| Security | Basic | Full SSL/TLS |

## 📝 Quick Start Commands

```bash
# Absolute minimum to see something working
pip install pandas transformers streamlit
python analyze_excel_data.py
streamlit run app_simple_search.py

# Recommended for development
pip install -r requirements-dev.txt
python setup.py develop
pytest tests/

# Full production setup
pip install -r requirements.txt
alembic upgrade head
gunicorn api.main:app -w 4
```

## 🚨 Emergency Fixes

### Reset Everything
```bash
# Clean install
rm -rf venv/
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Clear Model Cache
```bash
# Remove cached models
rm -rf ~/.cache/huggingface/
rm -rf ./models/
```

### Database Reset
```sql
-- Drop and recreate database
DROP DATABASE IF EXISTS review_engine;
CREATE DATABASE review_engine;
```

## 💡 Final Notes

The system is **architecturally sound** but requires proper setup:

- **Environment setup**: 30-60 minutes
- **Dependency installation**: 20-30 minutes  
- **Model downloads**: 1-2 hours (depends on internet speed)
- **Configuration**: 15-20 minutes
- **Testing & debugging**: Varies based on issues

Think of it like assembling a high-performance computer:
- ✅ All parts are included (code)
- ⚠️ Assembly required (setup)
- ⚠️ OS installation needed (dependencies)
- ⚠️ Drivers required (configurations)
- ⚠️ First boot takes time (model downloads)

**Recommendation:** Start with Phase 1 (Minimal Setup), verify it works, then progressively add features. This approach minimizes frustration and helps identify issues early.

## 📞 Getting Help

If you encounter issues:

1. Check this troubleshooting guide
2. Review error logs in `logs/` directory
3. Search for the error message online
4. Check package documentation
5. Consider using smaller models initially
6. Start with sample data before using your Excel file

Remember: Complex systems require patience during setup, but once configured properly, they provide powerful capabilities!
