# 🚀 Ultimate AI Review Engine - Windows Deployment Guide

## Table of Contents
1. [System Requirements](#1-system-requirements)
2. [Installation Methods](#2-installation-methods)
3. [Development Deployment](#3-development-deployment)
4. [Production Deployment](#4-production-deployment)
5. [Docker Deployment](#5-docker-deployment)
6. [Cloud Deployment](#6-cloud-deployment)
7. [Security Configuration](#7-security-configuration)
8. [Monitoring Setup](#8-monitoring-setup)
9. [Backup & Recovery](#9-backup--recovery)
10. [Performance Tuning](#10-performance-tuning)

## 1. System Requirements

### 1.1. Hardware Requirements
```plaintext
Minimum Requirements:
- CPU: 4 cores
- RAM: 8GB
- Storage: 20GB free space
- Network: 100Mbps

Recommended Requirements:
- CPU: 8 cores
- RAM: 16GB
- Storage: 50GB SSD
- Network: 1Gbps
```

### 1.2. Software Requirements
```plaintext
Required Software Versions:
- Windows 10/11 Pro or Windows Server 2019/2022
- Python 3.10+
- PostgreSQL 15+
- Redis 7+
- Git 2.30+
- PowerShell 5.1+
```

### 1.3. Network Requirements
```plaintext
Required Open Ports:
- 5000: Web Interface
- 8000: API Backend
- 8501: Analytics Dashboard
- 5555: Celery Monitor
- 5432: PostgreSQL
- 6379: Redis
```

## 2. Installation Methods

### 2.1. Using Chocolatey (Recommended)
```powershell
# Install Chocolatey
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))

# Install required software
choco install -y `
    python `
    postgresql `
    redis-64 `
    git `
    nssm `
    nginx

# Verify installations
python --version
pg_config --version
redis-cli --version
git --version
```

### 2.2. Manual Installation
Detailed steps for manual installation of each component:

#### Python Setup
```powershell
# Download Python installer
Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.10.0/python-3.10.0-amd64.exe" -OutFile "python-installer.exe"

# Install Python (silent mode)
Start-Process -Wait -FilePath "python-installer.exe" -ArgumentList "/quiet InstallAllUsers=1 PrependPath=1"
```

#### PostgreSQL Setup
```powershell
# Download PostgreSQL installer
Invoke-WebRequest -Uri "https://get.enterprisedb.com/postgresql/postgresql-15.0-1-windows-x64.exe" -OutFile "postgresql-installer.exe"

# Install PostgreSQL (silent mode)
Start-Process -Wait -FilePath "postgresql-installer.exe" -ArgumentList "--unattendedmodeui none --mode unattended --superpassword `"your_password`""
```

## 3. Development Deployment

### 3.1. Project Setup
```powershell
# Clone repository
cd C:\Users\OLANREWAJU BDE\Desktop
git clone <repository-url> ai-review-engine_updated
cd ai-review-engine_updated

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 3.2. Development Configuration
```powershell
# Create development environment file
@"
DEBUG=True
ENVIRONMENT=development
SECRET_KEY=dev_secret_key
DB_HOST=localhost
DB_PORT=5432
DB_NAME=ai_review_engine_dev
DB_USER=postgres
DB_PASSWORD=dev_password
REDIS_URL=redis://localhost:6379/0
"@ | Out-File -FilePath .env.development
```

### 3.3. Database Setup
```powershell
# Initialize development database
createdb ai_review_engine_dev
flask db upgrade
python scripts/seed_dev_data.py
```

## 4. Production Deployment

### 4.1. Production Environment Setup
```powershell
# Create production environment file
@"
DEBUG=False
ENVIRONMENT=production
SECRET_KEY=your_secure_key_here
DB_HOST=localhost
DB_PORT=5432
DB_NAME=ai_review_engine_prod
DB_USER=ai_review_engine
DB_PASSWORD=your_secure_password
REDIS_URL=redis://localhost:6379/0
ALLOWED_HOSTS=your_domain.com
CSRF_TRUSTED_ORIGINS=https://your_domain.com
"@ | Out-File -FilePath .env.production
```

### 4.2. Configure Services as Windows Services
```powershell
# Install NSSM (Non-Sucking Service Manager)
choco install nssm -y

# Create services for each component
# API Service
nssm install AIReviewAPI "C:\Users\OLANREWAJU BDE\Desktop\ai-review-engine_updated\venv\Scripts\python.exe"
nssm set AIReviewAPI AppParameters "-m uvicorn api.main:app --host 0.0.0.0 --port 8000"
nssm set AIReviewAPI AppDirectory "C:\Users\OLANREWAJU BDE\Desktop\ai-review-engine_updated"
nssm set AIReviewAPI AppEnvironmentExtra "PATH=%PATH%;C:\Users\OLANREWAJU BDE\Desktop\ai-review-engine_updated\venv\Scripts"

# Web Service
nssm install AIReviewWeb "C:\Users\OLANREWAJU BDE\Desktop\ai-review-engine_updated\venv\Scripts\python.exe"
nssm set AIReviewWeb AppParameters "web\app.py"
nssm set AIReviewWeb AppDirectory "C:\Users\OLANREWAJU BDE\Desktop\ai-review-engine_updated"

# Worker Service
nssm install AIReviewWorker "C:\Users\OLANREWAJU BDE\Desktop\ai-review-engine_updated\venv\Scripts\celery.exe"
nssm set AIReviewWorker AppParameters "-A tasks worker --loglevel=info -P eventlet"
nssm set AIReviewWorker AppDirectory "C:\Users\OLANREWAJU BDE\Desktop\ai-review-engine_updated\worker"
```

### 4.3. Configure Nginx as Reverse Proxy
```powershell
# Install Nginx
choco install nginx -y

# Create Nginx configuration
@"
worker_processes 1;
events { worker_connections 1024; }

http {
    sendfile on;
    
    upstream web_app {
        server 127.0.0.1:5000;
    }
    
    upstream api_server {
        server 127.0.0.1:8000;
    }
    
    upstream dashboard {
        server 127.0.0.1:8501;
    }

    server {
        listen 80;
        server_name your_domain.com;
        
        location / {
            proxy_pass http://web_app;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
        }
        
        location /api {
            proxy_pass http://api_server;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
        }
        
        location /dashboard {
            proxy_pass http://dashboard;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_http_version 1.1;
            proxy_set_header Upgrade \$http_upgrade;
            proxy_set_header Connection "upgrade";
        }
    }
}
"@ | Out-File -FilePath "C:\nginx\conf\nginx.conf" -Encoding ASCII
```

## 5. Docker Deployment

### 5.1. Install Docker Desktop
```powershell
choco install docker-desktop -y
```

### 5.2. Configure Docker Compose
```powershell
# Create Docker Compose file
@"
version: '3.8'

services:
  api:
    build: 
      context: .
      dockerfile: docker/api/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DB_HOST=db
      - REDIS_HOST=redis
    depends_on:
      - db
      - redis

  web:
    build:
      context: .
      dockerfile: docker/web/Dockerfile
    ports:
      - "5000:5000"
    depends_on:
      - api

  dashboard:
    build:
      context: .
      dockerfile: docker/dashboard/Dockerfile
    ports:
      - "8501:8501"
    depends_on:
      - api

  worker:
    build:
      context: .
      dockerfile: docker/worker/Dockerfile
    depends_on:
      - redis
      - db

  db:
    image: postgres:15
    environment:
      POSTGRES_DB: ai_review_engine
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: your_secure_password
    volumes:
      - pgdata:/var/lib/postgresql/data

  redis:
    image: redis:7
    volumes:
      - redisdata:/data

volumes:
  pgdata:
  redisdata:
"@ | Out-File -FilePath "docker-compose.yml"
```

### 5.3. Build and Run with Docker
```powershell
# Build and start services
docker-compose build
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f
```

## 6. Cloud Deployment

### 6.1. Azure Deployment
```powershell
# Install Azure CLI
choco install azure-cli -y

# Login to Azure
az login

# Create Azure Container App
az containerapp up `
    --name ai-review-engine `
    --resource-group ai-review-engine-rg `
    --location eastus `
    --environment production `
    --source .
```

### 6.2. AWS Deployment
```powershell
# Install AWS CLI
choco install awscli -y

# Configure AWS credentials
aws configure

# Deploy using CloudFormation
aws cloudformation deploy `
    --template-file deploy/aws/cloudformation.yaml `
    --stack-name ai-review-engine `
    --parameter-overrides Environment=production
```

## 7. Security Configuration

### 7.1. SSL/TLS Setup
```powershell
# Install OpenSSL
choco install openssl -y

# Generate self-signed certificate (development)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 `
    -keyout private.key -out certificate.crt
```

### 7.2. Windows Firewall Configuration
```powershell
# Open required ports
New-NetFirewallRule `
    -DisplayName "AI Review Engine - Web" `
    -Direction Inbound `
    -LocalPort 5000 `
    -Protocol TCP `
    -Action Allow

New-NetFirewallRule `
    -DisplayName "AI Review Engine - API" `
    -Direction Inbound `
    -LocalPort 8000 `
    -Protocol TCP `
    -Action Allow
```

## 8. Monitoring Setup

### 8.1. Configure Windows Event Logging
```powershell
# Create new event log source
New-EventLog -LogName Application -Source "AIReviewEngine"

# Test logging
Write-EventLog `
    -LogName Application `
    -Source "AIReviewEngine" `
    -EventID 1001 `
    -EntryType Information `
    -Message "AI Review Engine started successfully"
```

### 8.2. Setup Application Insights
```powershell
# Install Application Insights SDK
pip install opencensus-ext-azure

# Configure connection string
$env:APPLICATIONINSIGHTS_CONNECTION_STRING = "your_connection_string"
```

## 9. Backup & Recovery

### 9.1. Database Backup
```powershell
# Create backup script
@"
\$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
\$backupPath = "C:\\Backups\\ai_review_engine_\$timestamp.backup"

# Backup database
pg_dump -Fc ai_review_engine > \$backupPath

# Compress backup
Compress-Archive -Path \$backupPath -DestinationPath "\$backupPath.zip"

# Clean up old backups (keep last 7 days)
Get-ChildItem "C:\\Backups" -Filter "*.backup" |
    Where-Object { \$_.LastWriteTime -lt (Get-Date).AddDays(-7) } |
    Remove-Item
"@ | Out-File -FilePath "scripts\backup_database.ps1"

# Schedule backup task
$action = New-ScheduledTaskAction -Execute "PowerShell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"C:\Users\OLANREWAJU BDE\Desktop\ai-review-engine_updated\scripts\backup_database.ps1`""
$trigger = New-ScheduledTaskTrigger -Daily -At 3AM
Register-ScheduledTask -Action $action -Trigger $trigger -TaskName "AIReviewEngineBackup" -Description "Daily backup of AI Review Engine database"
```

### 9.2. Application State Backup
```powershell
# Create application state backup script
@"
\$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
\$backupDir = "C:\\Backups\\AppState_\$timestamp"

# Create backup directory
New-Item -ItemType Directory -Path \$backupDir

# Backup configuration files
Copy-Item ".env*" \$backupDir
Copy-Item "config/*" \$backupDir

# Backup logs
Copy-Item "logs/*" "\$backupDir\\logs"

# Compress backup
Compress-Archive -Path \$backupDir -DestinationPath "\$backupDir.zip"

# Clean up temporary files
Remove-Item -Recurse -Force \$backupDir
"@ | Out-File -FilePath "scripts\backup_app_state.ps1"
```

## 10. Performance Tuning

### 10.1. PostgreSQL Optimization
```powershell
# Update PostgreSQL configuration
@"
max_connections = 100
shared_buffers = 2GB
effective_cache_size = 6GB
maintenance_work_mem = 512MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1
effective_io_concurrency = 200
work_mem = 52428kB
min_wal_size = 1GB
max_wal_size = 4GB
"@ | Out-File -Append -FilePath "C:\Program Files\PostgreSQL\15\data\postgresql.conf"
```

### 10.2. Redis Optimization
```powershell
# Update Redis configuration
@"
maxmemory 2gb
maxmemory-policy allkeys-lru
appendonly yes
appendfsync everysec
"@ | Out-File -Append -FilePath "C:\Program Files\Redis\redis.windows.conf"
```

### 10.3. Windows Performance Optimization
```powershell
# Optimize Windows for background services
Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\PriorityControl" `
    -Name "Win32PrioritySeparation" -Value 38

# Configure power settings for high performance
powercfg /setactive SCHEME_MIN

# Disable unnecessary services
$servicesToDisable = @(
    "TabletInputService",
    "WSearch",
    "WerSvc"
)

foreach ($service in $servicesToDisable) {
    Set-Service -Name $service -StartupType Disabled
    Stop-Service -Name $service -Force
}
```

## Deployment Verification Checklist

```plaintext
□ System Requirements Met
  □ Hardware specifications checked
  □ Required software installed
  □ Network ports available

□ Development Environment
  □ Virtual environment created
  □ Dependencies installed
  □ Development database configured
  □ Local services running

□ Production Environment
  □ Environment variables set
  □ Services installed
  □ Nginx configured
  □ SSL/TLS certificates installed

□ Security
  □ Firewall rules configured
  □ SSL/TLS enabled
  □ Database passwords set
  □ File permissions set

□ Monitoring
  □ Logging configured
  □ Application Insights set up
  □ Alert rules created

□ Backup
  □ Database backup scheduled
  □ Application state backup configured
  □ Backup verification process tested

□ Performance
  □ Database optimized
  □ Cache configured
  □ Windows optimized
  □ Load testing completed
```

---
📝 Documentation last updated: 2025-09-21