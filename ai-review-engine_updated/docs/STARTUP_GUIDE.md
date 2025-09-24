# 🚀 Ultimate AI Review Engine - Step-by-Step Startup Guide

## Table of Contents
1. [Prerequisites Installation](#1-prerequisites-installation)
2. [Initial Setup](#2-initial-setup)
3. [Database Setup](#3-database-setup)
4. [Environment Configuration](#4-environment-configuration)
5. [Service Startup Sequence](#5-service-startup-sequence)
6. [Verification Steps](#6-verification-steps)
7. [Troubleshooting](#7-troubleshooting)

## 1. Prerequisites Installation

### 1.1. Install Chocolatey (Package Manager)
```powershell
# Run as Administrator
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))
```

### 1.2. Install Python
```powershell
choco install python -y
refreshenv
```

### 1.3. Install PostgreSQL
```powershell
choco install postgresql -y
refreshenv
```

### 1.4. Install Redis
```powershell
choco install redis-64 -y
refreshenv
```

## 2. Initial Setup

### 2.1. Clone or Download Project (if not already done)
```powershell
cd C:\Users\OLANREWAJU BDE\Desktop
git clone <repository-url> ai-review-engine_updated
cd ai-review-engine_updated
```

### 2.2. Create Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate
```

### 2.3. Install Dependencies
```powershell
# Core dependencies
pip install -r requirements.txt

# Additional dependencies
pip install fastapi uvicorn celery redis streamlit flask pytest
```

## 3. Database Setup

### 3.1. Start PostgreSQL Service
```powershell
# Start PostgreSQL service
net start postgresql-x64-15

# Verify it's running
Get-Service postgresql*
```

### 3.2. Create Database
```powershell
# Create the database
createdb ai_review_engine

# Verify database creation
psql -l
```

### 3.3. Initialize Database Schema
```powershell
cd C:\Users\OLANREWAJU BDE\Desktop\ai-review-engine_updated
flask db init
flask db migrate -m "Initial migration"
flask db upgrade
```

## 4. Environment Configuration

### 4.1. Configure Environment Variables
Edit `.env` file in the root directory:
```plaintext
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=ai_review_engine
DB_USER=postgres
DB_PASSWORD=your_secure_password

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379

# Service Ports
API_PORT=8000
WEB_PORT=5000
DASHBOARD_PORT=8501
```

### 4.2. Create Required Directories
```powershell
# Create logs directory
mkdir logs

# Create data directory if not exists
mkdir data
```

## 5. Service Startup Sequence

⚠️ IMPORTANT: Services must be started in this exact order to ensure proper functionality.

### 5.1. Start Redis Server (Cache & Message Broker)
```powershell
# Start Redis service
net start redis

# Verify Redis is running
redis-cli ping  # Should return "PONG"
```

### 5.2. Start PostgreSQL (if not already running)
```powershell
net start postgresql-x64-15
```

### 5.3. Start FastAPI Backend (API Service)
```powershell
cd C:\Users\OLANREWAJU BDE\Desktop\ai-review-engine_updated\api
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 5.4. Start Celery Worker (Background Tasks)
Open a new PowerShell window:
```powershell
cd C:\Users\OLANREWAJU BDE\Desktop\ai-review-engine_updated\worker
celery -A tasks worker --loglevel=info -P eventlet
```

### 5.5. Start Flask Web Interface
Open a new PowerShell window:
```powershell
cd C:\Users\OLANREWAJU BDE\Desktop\ai-review-engine_updated\web
python app.py
```

### 5.6. Start Streamlit Dashboard
Open a new PowerShell window:
```powershell
cd C:\Users\OLANREWAJU BDE\Desktop\ai-review-engine_updated\streamlit
streamlit run dashboard.py
```

## 6. Verification Steps

### 6.1. Check Services
Verify all services are running by accessing:

1. FastAPI Backend:
   - URL: http://localhost:8000
   - Documentation: http://localhost:8000/docs
   - Health check: http://localhost:8000/health

2. Web Interface:
   - URL: http://localhost:5000
   - Should see login page

3. Analytics Dashboard:
   - URL: http://localhost:8501
   - Should see Streamlit interface

4. Celery Monitor:
   - URL: http://localhost:5555
   - Should see worker status

### 6.2. Check Logs
Monitor log files in the `logs` directory:
```powershell
Get-Content -Path logs\api.log -Wait
Get-Content -Path logs\web.log -Wait
Get-Content -Path logs\worker.log -Wait
Get-Content -Path logs\streamlit.log -Wait
```

### 6.3. Test Database Connection
```powershell
psql -d ai_review_engine -c "\dt"  # List tables
```

## 7. Troubleshooting

### 7.1. Port Conflicts
If you see "Address already in use" errors:
```powershell
# Find process using a port
netstat -ano | findstr "8000"  # Replace with problematic port

# Kill process
taskkill /PID <process_id> /F
```

### 7.2. Service Issues
```powershell
# Restart Redis
net stop redis
net start redis

# Restart PostgreSQL
net stop postgresql-x64-15
net start postgresql-x64-15
```

### 7.3. Database Connection Issues
```powershell
# Test PostgreSQL connection
psql -h localhost -U postgres -d ai_review_engine

# Reset PostgreSQL password if needed
psql -U postgres -c "ALTER USER postgres PASSWORD 'new_password';"
```

### 7.4. Redis Connection Issues
```powershell
# Test Redis connection
redis-cli ping

# Flush Redis if needed
redis-cli flushall
```

## Quick Start (Using Startup Script)

For convenience, you can use the provided startup script:
```powershell
cd C:\Users\OLANREWAJU BDE\Desktop\ai-review-engine_updated\scripts
.\start_services.ps1
```

## Shutdown Procedure

To properly shut down all services:
```powershell
# Using the shutdown script
cd C:\Users\OLANREWAJU BDE\Desktop\ai-review-engine_updated\scripts
.\stop_services.ps1
```

## Additional Notes

1. Keep all PowerShell windows open while running the services
2. Monitor the logs directory for any errors
3. Use separate terminal windows for each service for better log visibility
4. The startup script handles most of these steps automatically
5. For development, manual startup might be preferred for better control

---
📝 Documentation last updated: 2025-09-21