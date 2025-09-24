# 🚀 Ultimate AI Review Engine - Deployment Guide

## Table of Contents
1. [System Overview](#system-overview)
2. [Prerequisites](#prerequisites)
3. [Local Development Setup](#local-development-setup)
4. [Docker Deployment](#docker-deployment)
5. [Cloud Deployment](#cloud-deployment)
6. [Configuration](#configuration)
7. [Troubleshooting](#troubleshooting)

## System Overview 📋

### Components
- **Web Interface**: Flask-based web application
- **Database**: PostgreSQL (production) / SQLite (development)
- **Cache**: Redis
- **Task Queue**: Celery
- **ML Models**: Sentiment Analysis, Recommendation Engine
- **Search Engine**: Hybrid (Database + Web)
- **Monitoring**: Flower (Celery) + Custom Metrics

### Features
- User Role Management (Free/Business/Enterprise)
- AI-Powered Search
- Real-time Analytics
- REST API Access
- Multi-user Support
- Data Privacy Controls

## Prerequisites ✅

### Required Software
- Python 3.10+
- PostgreSQL 15+ (for production)
- Redis 7+ (for caching/queues)
- Git

### Optional Components
- Docker & Docker Compose (for containerized deployment)
- NVIDIA CUDA (for GPU acceleration)

### Python Dependencies
```plaintext
# Core Dependencies
flask>=2.3.0
flask-sqlalchemy>=3.0.0
flask-login>=0.6.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
streamlit>=1.25.0

# AI/ML Dependencies
torch>=2.0.0
transformers>=4.30.0
spacy>=3.6.0
nltk>=3.8.0
textblob>=0.17.0

# Web & API
aiohttp>=3.8.0
requests>=2.31.0
beautifulsoup4>=4.12.0
uvicorn>=0.23.0
gunicorn>=21.0.0

# Database & Cache
psycopg2-binary>=2.9.0
redis>=4.6.0
celery>=5.3.0
```

## Local Development Setup 💻

### 1. Environment Setup
```powershell
# Clone repository
git clone https://github.com/your-repo/ai-review-engine.git
cd ai-review-engine

# Create virtual environment
python -m venv main_venv

# Activate environment
# Windows:
.\main_venv\Scripts\activate
# Linux/Mac:
source main_venv/bin/activate

# Install dependencies
pip install -r requirements_ultimate.txt
```

### 2. Database Setup
```powershell
# Initialize database
flask db init
flask db migrate
flask db upgrade

# Create initial admin user
python scripts/create_admin.py
```

### 3. Data Preparation
```powershell
# Ensure data file is present
Copy-Item data/final_dataset_streamlined_clean.csv .

# Initialize data models
python scripts/initialize_models.py
```

### 4. Running the Application
```powershell
# Start the web app
python ultimate_web_app.py
```

Access the application at: http://localhost:5000

## Docker Deployment 🐳

### 1. Prerequisites
- Docker Engine
- Docker Compose
- At least 4GB RAM
- 20GB disk space

### 2. Configuration
```bash
# Create environment file
cp .env.example .env

# Edit environment variables
notepad .env
```

Required environment variables:
```plaintext
DB_PASSWORD=your_secure_password
SECRET_KEY=your_secret_key
REDIS_URL=redis://redis:6379
GOOGLE_SEARCH_API_KEY=your_api_key  # Optional
```

### 3. Deployment
```bash
# Build and start containers
docker-compose up -d

# Monitor logs
docker-compose logs -f

# Scale workers (if needed)
docker-compose up -d --scale celery_worker=3
```

Access points:
- Web App: http://localhost:8501
- API: http://localhost:8000
- Flower: http://localhost:5555

## Cloud Deployment ☁️

### AWS Deployment

1. **Prerequisites**:
```bash
# Configure AWS CLI
aws configure
```

2. **Deployment**:
```bash
# Deploy using script
./deploy.sh aws

# Or manually:
aws cloudformation deploy \
    --template-file deploy/aws/cloudformation.yaml \
    --stack-name review-engine \
    --parameter-overrides Environment=production
```

### Google Cloud

1. **Prerequisites**:
```bash
# Configure gcloud
gcloud init
gcloud auth configure-docker
```

2. **Deployment**:
```bash
# Deploy using script
./deploy.sh gcp

# Or using Cloud Run
gcloud run deploy review-engine \
    --image gcr.io/project/review-engine \
    --platform managed
```

### Azure

1. **Prerequisites**:
```bash
# Login to Azure
az login
```

2. **Deployment**:
```bash
# Deploy using script
./deploy.sh azure

# Or using Azure Container Apps
az containerapp up \
    --name review-engine \
    --resource-group review-engine-rg \
    --source .
```

## Configuration ⚙️

### Application Configuration
```python
# config/app_config.py
config = {
    'DEBUG': False,
    'TESTING': False,
    'DATABASE_URL': 'postgresql://user:pass@localhost:5432/db',
    'REDIS_URL': 'redis://localhost:6379',
    'SECRET_KEY': 'your-secret-key',
    'SESSION_TYPE': 'redis',
    'PERMANENT_SESSION_LIFETIME': 3600,
    'RATE_LIMIT_ENABLED': True,
}
```

### Role-Based Access Control
```python
PLAN_CONFIGS = {
    'free': {
        'search_limit': 20,
        'api_calls_limit': 0,
        'features': ['basic_search']
    },
    'business': {
        'search_limit': 200,
        'api_calls_limit': 1000,
        'features': ['all']
    }
}
```

## Troubleshooting 🔧

### Common Issues

1. **Database Connection**:
```powershell
# Check database status
docker-compose exec postgres pg_isready

# Reset database
docker-compose down -v
docker-compose up -d
```

2. **Redis Connection**:
```powershell
# Test Redis
docker-compose exec redis redis-cli ping

# Clear cache
docker-compose exec redis redis-cli FLUSHALL
```

3. **Application Errors**:
```powershell
# Check logs
docker-compose logs api
docker-compose logs streamlit

# Restart services
docker-compose restart api streamlit
```

### Health Checks

```powershell
# API health
curl http://localhost:8000/health

# Database health
python scripts/check_db_health.py

# Full system check
python scripts/system_health_check.py
```

## Security Notes 🔒

1. **Production Deployment**:
   - Change all default passwords
   - Enable HTTPS
   - Set up proper firewalls
   - Configure rate limiting

2. **Data Protection**:
   - Regular backups
   - Encryption at rest
   - Secure API keys

3. **Access Control**:
   - Use strong passwords
   - Implement 2FA for admin
   - Regular audit logs review

## Performance Optimization 🚄

1. **Caching Strategy**:
   - Redis cache for API responses
   - Browser caching for static assets
   - Database query optimization

2. **Scaling**:
   - Horizontal scaling of workers
   - Database replication
   - Load balancing

---
📝 Documentation last updated: 2025-09-21