# Ultimate AI Phone Review Engine - Quick Start Script
# Run this script in PowerShell to set up the application

Write-Host "🚀 Ultimate AI Phone Review Engine - Quick Setup" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

# Check if Python is installed
Write-Host "1️⃣ Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found! Please install Python 3.8+ first." -ForegroundColor Red
    Write-Host "Download from: https://python.org/downloads/" -ForegroundColor Blue
    exit 1
}

# Create virtual environment
Write-Host "`n2️⃣ Creating virtual environment..." -ForegroundColor Yellow
try {
    python -m venv venv
    Write-Host "✅ Virtual environment created" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to create virtual environment" -ForegroundColor Red
    exit 1
}

# Activate virtual environment
Write-Host "`n3️⃣ Activating virtual environment..." -ForegroundColor Yellow
try {
    & .\venv\Scripts\Activate.ps1
    Write-Host "✅ Virtual environment activated" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to activate virtual environment" -ForegroundColor Red
    Write-Host "Try running: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser" -ForegroundColor Blue
    exit 1
}

# Install requirements
Write-Host "`n4️⃣ Installing Python packages..." -ForegroundColor Yellow
try {
    pip install --upgrade pip
    pip install Flask Flask-SQLAlchemy Flask-Login Werkzeug pandas numpy scikit-learn nltk textblob plotly requests python-dotenv gunicorn
    Write-Host "✅ All packages installed successfully" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to install packages" -ForegroundColor Red
    exit 1
}

# Initialize database
Write-Host "`n5️⃣ Initializing database..." -ForegroundColor Yellow
try {
    python -c "from ultimate_web_app import app, db; app.app_context().push(); db.create_all(); print('Database initialized!')"
    Write-Host "✅ Database initialized" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to initialize database" -ForegroundColor Red
    exit 1
}

# Create demo users
Write-Host "`n6️⃣ Creating demo users..." -ForegroundColor Yellow
try {
    python -c "
from ultimate_web_app import app, db, User, PlanType
app.app_context().push()
users = [
    {'username': 'demo_user', 'email': 'demo@example.com', 'password': 'demo123', 'plan': 'free'},
    {'username': 'business_user', 'email': 'business@example.com', 'password': 'business123', 'plan': 'business'},
    {'username': 'enterprise_user', 'email': 'enterprise@example.com', 'password': 'enterprise123', 'plan': 'enterprise'}
]
for user_data in users:
    if not User.query.filter_by(username=user_data['username']).first():
        user = User(username=user_data['username'], email=user_data['email'], plan_type=user_data['plan'])
        user.set_password(user_data['password'])
        db.session.add(user)
db.session.commit()
print('Demo users created!')
"
    Write-Host "✅ Demo users created" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Demo users might already exist" -ForegroundColor Orange
}

Write-Host "`n🎉 Setup Complete!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "📋 Demo Accounts:" -ForegroundColor Yellow
Write-Host "  🆓 Free Plan: demo_user / demo123" -ForegroundColor White
Write-Host "  🏢 Business Plan: business_user / business123" -ForegroundColor White  
Write-Host "  🚀 Enterprise Plan: enterprise_user / enterprise123" -ForegroundColor White
Write-Host "`n🚀 To start the application:" -ForegroundColor Yellow
Write-Host "  python ultimate_web_app.py" -ForegroundColor White
Write-Host "`n🌐 Then open your browser to:" -ForegroundColor Yellow
Write-Host "  http://localhost:5000" -ForegroundColor Blue
Write-Host "`nEnjoy your Ultimate AI Phone Review Engine! 🎨✨" -ForegroundColor Magenta