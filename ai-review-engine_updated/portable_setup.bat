@echo off
echo ==========================================
echo AI Phone Review Engine - Portable Setup
echo ==========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed!
    echo Please download and install Python from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo [OK] Python is installed
python --version
echo.

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    echo [OK] Virtual environment created
) else (
    echo [OK] Virtual environment already exists
)
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo [OK] Virtual environment activated
echo.

REM Install minimal requirements
echo Installing minimal requirements...
pip install --quiet --upgrade pip
pip install --quiet streamlit>=1.28.0 pandas>=2.0.0 numpy>=1.24.0 plotly>=5.15.0
echo [OK] Minimal requirements installed
echo.

echo ==========================================
echo Setup complete! Starting the application...
echo ==========================================
echo.

REM Run the application
python -m streamlit run run_app.py

pause
