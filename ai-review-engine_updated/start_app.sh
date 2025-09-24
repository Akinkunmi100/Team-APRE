#!/bin/bash

echo "========================================"
echo "  AI Phone Review Engine - Starting..."
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.10 or higher"
    exit 1
fi

echo "Python found!"
echo ""

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "No virtual environment found. Creating one..."
    python3 -m venv venv
    source venv/bin/activate
    echo ""
    echo "Installing dependencies..."
    pip install streamlit pandas numpy plotly
fi

echo ""
echo "Starting AI Phone Review Engine..."
echo ""
echo "The application will open in your browser automatically."
echo "If not, open: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the application."
echo "========================================"
echo ""

# Run the application
python -m streamlit run run_app.py
