@echo off
echo ========================================
echo   AI Phone Review Engine - Starting...
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10 or higher from python.org
    pause
    exit /b 1
)

echo Python found!
echo.

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo No virtual environment found. Creating one...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo.
    echo Installing dependencies...
    pip install streamlit pandas numpy plotly
)

echo.
echo Starting AI Phone Review Engine...
echo.
echo The application will open in your browser automatically.
echo If not, open: http://localhost:8501
echo.
echo Press Ctrl+C to stop the application.
echo ========================================
echo.

REM Run the application
python -m streamlit run run_app.py

REM Keep window open if app crashes
if errorlevel 1 (
    echo.
    echo Application encountered an error.
    pause
)
