# 🚀 AI Phone Review Engine - Quick Reference Guide

## 🎯 WHICH FILE SHOULD I RUN?

### For Complete Experience (ALL FEATURES):
```bash
streamlit run main_engine.py
```
✅ **USE THIS** - It has EVERYTHING integrated with beautiful UI

### For Specific Needs:
- **Basic Analysis**: `streamlit run app.py`
- **Recommendations Focus**: `streamlit run app_with_recommendations.py`
- **Chat Interface**: `streamlit run app_chat_assistant.py`
- **Real-time Dashboard**: `streamlit run app_realtime_dashboard.py`
- **Simple Search**: `streamlit run app_simple_search.py`
- **API Server**: `uvicorn api.main:app --reload`

---

## 📦 Complete File Structure

```
ai-review-engine/
│
├── 🌟 main_engine.py          # MAIN FILE - Complete integrated system
├── app.py                      # Original basic dashboard
├── app_with_recommendations.py # Recommendation-focused interface
├── app_chat_assistant.py       # Conversational AI interface
├── app_realtime_dashboard.py   # Real-time monitoring
├── app_simple_search.py        # Simplified search interface
│
├── /modules/ (NEW ADVANCED FEATURES)
│   ├── advanced_personalization.py  # User profiles, A/B testing, behavioral analytics
│   └── deeper_insights.py          # Emotion, sarcasm, cultural, temporal analysis
│
├── /models/ (CORE AI MODELS)
│   ├── recommendation_engine.py    # Hybrid recommendations
│   ├── absa_model.py               # Aspect-based sentiment
│   ├── spam_detector.py            # Fake review detection
│   ├── market_analyzer.py          # Market trends
│   └── review_summarizer.py        # Summarization
│
├── /core/ (CORE ENGINES)
│   ├── ai_engine.py                # AI orchestrator
│   ├── nlp_core.py                 # NLP operations
│   └── personalization_engine.py   # Basic personalization
│
├── /scrapers/ (WEB SCRAPING)
│   ├── jumia_scraper.py            # Jumia marketplace
│   ├── amazon_scraper.py           # Amazon reviews
│   └── scraper_manager.py          # Multi-source manager
│
├── /utils/ (UTILITIES)
│   ├── data_preprocessing.py       # Data cleaning & prep
│   ├── visualization.py            # Charts & graphs
│   └── metrics_calculator.py       # Performance metrics
│
├── /database/ (DATA LAYER)
│   ├── database_manager.py         # Database operations
│   └── cache_manager.py            # Redis caching
│
└── /api/ (BACKEND API)
    ├── main.py                     # FastAPI server
    └── endpoints.py                # API routes
```

---

## 🎨 Feature Comparison

| Feature | main_engine.py | app.py | Others |
|---------|:-------------:|:------:|:------:|
| **Core Analysis** | ✅ | ✅ | ⚠️ |
| **Recommendations** | ✅ | ❌ | ⚠️ |
| **User Profiles** | ✅ | ❌ | ❌ |
| **Emotion Detection** | ✅ | ❌ | ❌ |
| **Sarcasm Detection** | ✅ | ❌ | ❌ |
| **Cultural Analysis** | ✅ | ❌ | ❌ |
| **Temporal Patterns** | ✅ | ❌ | ❌ |
| **A/B Testing** | ✅ | ❌ | ❌ |
| **Behavioral Analytics** | ✅ | ❌ | ❌ |
| **Alert System** | ✅ | ❌ | ❌ |
| **Beautiful UI** | ✅ | ⚠️ | ⚠️ |
| **Export Reports** | ✅ | ⚠️ | ❌ |

---

## 🚦 Quick Commands

### 1. Install Everything
```bash
pip install -r requirements.txt
```

### 2. Run Complete System
```bash
streamlit run main_engine.py
```

### 3. Run API Server
```bash
uvicorn api.main:app --reload
```

### 4. Run with Docker
```bash
docker-compose up -d
```

### 5. Test Individual Modules
```python
# Test personalization
python -m modules.advanced_personalization

# Test deeper insights
python -m modules.deeper_insights

# Test recommendations
python -m models.recommendation_engine
```

---

## 🔥 New Advanced Features (in main_engine.py)

### 1. Advanced Personalization
- User trust scores
- Preference learning (implicit & explicit)
- A/B testing with statistical significance
- Behavioral pattern recognition
- Smart alert system

### 2. Deeper Insights
- **8 Emotions**: Joy, Trust, Fear, Surprise, Sadness, Disgust, Anger, Anticipation
- **Sarcasm Detection**: Identifies irony and contradictions
- **Cultural Analysis**: North America, Europe, Asia-Pacific variations
- **Temporal Patterns**: Honeymoon effect, declining satisfaction, etc.
- **Review Quality**: Predicts helpfulness scores

### 3. In main_engine.py Interface
- Dashboard with executive metrics
- Smart search with AI recommendations
- Deep analysis tools
- Emotion & sarcasm detection page
- Cultural insights visualization
- Temporal pattern analysis
- Personalized experience page
- Live scraping interface
- Comprehensive reports

---

## 💡 Usage Examples

### Create User Profile (in main_engine.py)
1. Run `streamlit run main_engine.py`
2. Click "Create Profile" in sidebar
3. Set preferences
4. Get personalized recommendations

### Detect Sarcasm
1. Go to "Emotion & Sarcasm Detection" page
2. Enter review: "This phone is absolutely AMAZING!!! Best purchase ever... NOT!"
3. See sarcasm indicators and confidence

### Analyze Cultural Differences
1. Go to "Cultural Insights" page
2. View regional sentiment variations
3. See preference patterns by culture

### Generate Report
1. Go to "Reports" page
2. Select report type
3. Choose date range
4. Export as PDF/Excel/CSV

---

## 📞 Need Help?

1. **Full Documentation**: See `README.md`
2. **System Overview**: See `SYSTEM_OVERVIEW.md`
3. **Setup Issues**: See `docs/SETUP_AND_TROUBLESHOOTING.md`
4. **API Documentation**: Run server and visit `http://localhost:8000/docs`

---

## 🎯 Bottom Line

**Just run this for everything:**
```bash
streamlit run main_engine.py
```

This gives you the complete AI Phone Review Engine with all features integrated and working together!
