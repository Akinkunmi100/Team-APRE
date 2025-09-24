#!/bin/bash

echo "=========================================="
echo "AI Phone Review Engine - Portable Setup"
echo "=========================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python is not installed!"
    echo "Please install Python 3.8 or higher"
    echo "Ubuntu/Debian: sudo apt-get install python3 python3-pip python3-venv"
    echo "Mac: brew install python3"
    exit 1
fi

echo "[OK] Python is installed"
python3 --version
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "[OK] Virtual environment created"
else
    echo "[OK] Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "[OK] Virtual environment activated"
echo ""

# Install minimal requirements
echo "Installing minimal requirements..."
pip install --quiet --upgrade pip
pip install --quiet streamlit>=1.28.0 pandas>=2.0.0 numpy>=1.24.0 plotly>=5.15.0
echo "[OK] Minimal requirements installed"
echo ""

echo "=========================================="
echo "Setup complete! Starting the application..."
echo "=========================================="
echo ""

# Run the application
python -m streamlit run run_app.py
