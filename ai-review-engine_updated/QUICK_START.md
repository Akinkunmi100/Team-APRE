# 🚀 Quick Start Guide - AI Phone Review Engine

## Prerequisites
- Python 3.8 or higher installed
- Internet connection (for first-time package installation)

## Installation & Running

### Option 1: Automatic Setup (Recommended)

#### Windows:
1. Copy the entire `ai-review-engine` folder to your computer
2. Double-click `portable_setup.bat`
3. The app will automatically install dependencies and start

#### Mac/Linux:
1. Copy the entire `ai-review-engine` folder to your computer
2. Open terminal in the folder
3. Run: `chmod +x portable_setup.sh && ./portable_setup.sh`

### Option 2: Manual Setup

1. **Navigate to the folder:**
   ```bash
   cd path/to/ai-review-engine
   ```

2. **Create virtual environment:**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Mac/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install minimal requirements:**
   ```bash
   pip install -r requirements_minimal.txt
   ```

4. **Run the application:**
   ```bash
   python -m streamlit run run_app.py
   ```

## Accessing the Application
Once running, open your web browser and go to:
- **http://localhost:8501**

## Features Available
✅ Home Dashboard with phone analysis
✅ Smart Phone Search
✅ Review Analysis
✅ Recommendations
✅ Market Trends

## Troubleshooting

### Python not found
- Download from: https://www.python.org/downloads/
- Make sure to check "Add Python to PATH" during installation

### Port 8501 already in use
Run with a different port:
```bash
python -m streamlit run run_app.py --server.port 8502
```

### Missing modules error
Install minimal requirements:
```bash
pip install streamlit pandas numpy plotly
```

## Sharing with Others
To share this app with others:
1. Zip the entire `ai-review-engine` folder
2. Share the zip file
3. Tell them to follow this Quick Start guide

## Remote Access (Optional)
To access from another device on the same network:
```bash
python -m streamlit run run_app.py --server.address 0.0.0.0
```
Then access using: `http://[your-computer-ip]:8501`

## Support
For issues, check:
- Python version: `python --version` (should be 3.8+)
- Streamlit installation: `pip show streamlit`
- Error logs in the terminal window
