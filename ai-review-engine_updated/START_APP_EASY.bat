@echo off
title AI Phone Review Engine - Easy Start
color 0A

echo ================================================
echo    AI PHONE REVIEW ENGINE - EASY LAUNCHER
echo ================================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed on this computer!
    echo.
    echo Please install Python first:
    echo 1. Go to: https://www.python.org/downloads/
    echo 2. Download Python 3.8 or newer
    echo 3. During installation, CHECK "Add Python to PATH"
    echo 4. After installation, run this script again
    echo.
    pause
    exit /b 1
)

echo [OK] Python is installed
echo.

:: Check if virtual environment exists
if not exist "easy_venv" (
    echo Creating virtual environment (first time only)...
    python -m venv easy_venv
    echo [OK] Virtual environment created
) else (
    echo [OK] Virtual environment found
)

:: Activate virtual environment
call easy_venv\Scripts\activate.bat

:: Check if streamlit is installed
pip show streamlit >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo Installing required packages (first time only)...
    echo This will take 2-5 minutes...
    echo.
    python -m pip install --upgrade pip
    python -m pip install streamlit pandas numpy plotly
    echo.
    echo [OK] Packages installed successfully!
)

echo.
echo ================================================
echo    STARTING AI PHONE REVIEW ENGINE
echo ================================================
echo.
echo The app will open in your browser automatically.
echo If not, go to: http://localhost:8501
echo.
echo Press Ctrl+C to stop the application
echo ================================================
echo.

:: Run the application
python -m streamlit run run_app.py

:: If app stops, pause to see any error messages
pause