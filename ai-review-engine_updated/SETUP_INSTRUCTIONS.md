# 🚀 AI Phone Review Engine - Setup Instructions

## 📋 Prerequisites
- Python 3.10 or higher installed
- Git (optional, for cloning)
- 4GB+ RAM recommended
- Windows/Mac/Linux compatible

---

## ⚡ Quick Setup (5 minutes)

### Step 1: Copy the Folder
Copy the entire `ai-review-engine` folder to your new PC.

### Step 2: Open Terminal/Command Prompt
Navigate to the folder:
```bash
cd path/to/ai-review-engine
```

### Step 3: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 4: Install Dependencies
```bash
# Install core dependencies only (fastest)
pip install streamlit pandas numpy scikit-learn plotly sqlalchemy

# OR install all dependencies (complete setup)
pip install -r requirements.txt
```

### Step 5: Run the Application
```bash
# Run the simplified version (RECOMMENDED - Always works!)
python -m streamlit run run_app.py

# OR run the full version (if all dependencies installed)
python -m streamlit run main_engine.py
```

The application will open automatically in your browser at: http://localhost:8501

---

## 🎯 Which File to Run?

### Option 1: `run_app.py` (RECOMMENDED)
✅ **Always works** - No complex dependencies needed
✅ Beautiful UI with all core features
✅ Handles missing modules gracefully
✅ Perfect for demos and testing

**Run with:**
```bash
python -m streamlit run run_app.py
```

### Option 2: `main_engine.py` (Full Features)
⚠️ Requires all dependencies
⚠️ May have issues on some systems
✅ All advanced features enabled

**Run with:**
```bash
python -m streamlit run main_engine.py
```

---

## 📦 Minimal Installation (If you have issues)

If you encounter dependency issues, install only the essentials:

```bash
pip install streamlit pandas numpy
```

Then run:
```bash
python -m streamlit run run_app.py
```

This will give you a working application with core features!

---

## 🔧 Troubleshooting

### Issue: "Module not found" errors
**Solution:** Install the missing module:
```bash
pip install [module_name]
```

### Issue: PyTorch installation fails
**Solution:** The app works without it! Use `run_app.py` instead.

### Issue: Redis connection error
**Solution:** The app works without Redis. It's optional for caching.

### Issue: Port 8501 already in use
**Solution:** Streamlit will automatically try the next port (8502, 8503, etc.)

### Issue: ImportError for advanced modules
**Solution:** The core features still work! Missing modules are handled gracefully.

---

## 🎨 Features Available

### With Minimal Install (run_app.py):
- ✅ Phone Search & Filtering
- ✅ Review Analysis
- ✅ Sentiment Detection
- ✅ Recommendations
- ✅ Market Trends
- ✅ Beautiful Dashboard
- ✅ Sample Data Demos

### With Full Install (main_engine.py):
All of the above plus:
- ✅ Emotion Detection (8 emotions)
- ✅ Sarcasm Detection
- ✅ Cultural Sentiment Analysis
- ✅ User Personalization
- ✅ A/B Testing
- ✅ Advanced Analytics
- ✅ Real Database Support

---

## 📂 Important Files

```
ai-review-engine/
│
├── run_app.py              # 🌟 MAIN FILE - Run this!
├── main_engine.py          # Full-featured version
├── requirements.txt        # Dependencies list
├── SETUP_INSTRUCTIONS.md   # This file
│
├── modules/
│   ├── deeper_insights.py      # Emotion & sarcasm detection
│   └── advanced_personalization.py  # User personalization
│
├── models/                 # AI models
├── scrapers/              # Web scrapers
├── utils/                 # Utilities
└── database/              # Database models
```

---

## 💡 Tips for Success

1. **Start Simple**: Use `run_app.py` first to ensure everything works
2. **Virtual Environment**: Always use a virtual environment to avoid conflicts
3. **Gradual Installation**: Install dependencies as needed rather than all at once
4. **Check Python Version**: Ensure Python 3.10+ is installed
5. **Browser**: Use Chrome or Firefox for best experience

---

## 🆘 Need Help?

If you encounter any issues:

1. Check the error message carefully
2. Try the minimal installation first
3. Use `run_app.py` instead of `main_engine.py`
4. Ensure all files were copied correctly
5. Check Python version: `python --version`

---

## 🎉 Success Checklist

- [ ] Folder copied to new PC
- [ ] Python 3.10+ installed
- [ ] Virtual environment created
- [ ] Dependencies installed (at least streamlit, pandas, numpy)
- [ ] Application runs with `python -m streamlit run run_app.py`
- [ ] Browser opens at http://localhost:8501
- [ ] Can see the AI Phone Review Engine dashboard

---

**Congratulations! Your AI Phone Review Engine is ready to use! 🚀**
