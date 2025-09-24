# 🔑 API Keys and Authentication Guide

## Table of Contents
- [Overview](#overview)
- [Required vs Optional Keys](#required-vs-optional-keys)
- [Getting Started Without API Keys](#getting-started-without-api-keys)
- [API Key Details](#api-key-details)
- [Security Best Practices](#security-best-practices)
- [Troubleshooting](#troubleshooting)

## 📌 Overview

**Good News!** The AI Phone Review Engine is designed to work primarily with **open-source models that run locally**. This means you don't need expensive API subscriptions to get started. Most "API keys" are either for security (which you generate yourself) or completely optional enhancements.

## 🎯 Required vs Optional Keys

### ✅ ACTUALLY REQUIRED

| Key | Purpose | How to Generate | Cost |
|-----|---------|-----------------|------|
| **SECRET_KEY** | Session encryption, JWT tokens | Auto-generate (see below) | FREE |
| **DATABASE_URL** | Database connection | Your own database | FREE* |

*Free with SQLite or local PostgreSQL

### ⭕ OPTIONAL ENHANCEMENTS

| Service | Purpose | When Needed | Cost |
|---------|---------|-------------|------|
| **HuggingFace Token** | Download gated models | Premium models only | FREE |
| **OpenAI API** | GPT-3.5/4 integration | If adding GPT features | Pay-per-use |
| **Reddit API** | Scrape Reddit reviews | If scraping Reddit | FREE |
| **Google Cloud** | Cloud services | Cloud deployment | Pay-per-use |

## 🚀 Getting Started Without API Keys

### Quick Start (No External APIs Needed!)

```bash
# Step 1: Generate a secure SECRET_KEY
python -c "import secrets; print('SECRET_KEY=' + secrets.token_hex(32))" > .env

# Step 2: Add local database
echo "DATABASE_URL=sqlite:///local_reviews.db" >> .env

# Step 3: Enable debug mode
echo "DEBUG=True" >> .env

# That's it! You can now run the system
python app.py
```

### Minimal `.env` File

```env
# This is ALL you need to start!
SECRET_KEY=your_64_character_hex_string_here_generate_with_python
DATABASE_URL=sqlite:///local_reviews.db
DEBUG=True

# Everything else is OPTIONAL
```

## 📋 API Key Details

### 1. SECRET_KEY (Required for Security)

**What it is:** A random string used to encrypt sessions and tokens  
**Not an API key:** You generate this yourself - it's not from any service

#### How to Generate:

**Option 1: Python (Recommended)**
```python
import secrets
print(secrets.token_hex(32))
# Output: 64 character hex string
```

**Option 2: Command Line**
```bash
# Linux/Mac
openssl rand -hex 32

# Windows PowerShell
-join ((1..64) | ForEach {'{0:X}' -f (Get-Random -Max 16)})
```

**Option 3: Online Generator**
- Visit: https://randomkeygen.com/
- Use: "CodeIgniter Encryption Keys" section
- Copy any 64+ character string

#### Where to Put It:
```env
# In .env file
SECRET_KEY=a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6a7b8c9d0e1f2
```

### 2. DATABASE_URL (Required for Data Storage)

**What it is:** Connection string for your database  
**Not an API key:** Points to your own database

#### Options:

**Option 1: SQLite (Easiest - No Setup)**
```env
DATABASE_URL=sqlite:///local_reviews.db
# Creates a local file, no server needed
```

**Option 2: PostgreSQL (Production)**
```env
DATABASE_URL=postgresql://username:password@localhost:5432/review_engine
# Requires PostgreSQL installation
```

**Option 3: MySQL**
```env
DATABASE_URL=mysql://username:password@localhost:3306/review_engine
# Requires MySQL installation
```

### 3. HUGGINGFACE_TOKEN (Optional)

**What it is:** Access token for HuggingFace Hub  
**When needed:** Only for gated/private models  
**Cost:** FREE

#### How to Get:

1. Create account at https://huggingface.co/join
2. Go to https://huggingface.co/settings/tokens
3. Click "New token"
4. Name it "review-engine"
5. Select "Read" permission
6. Copy the token

#### Where to Put It:
```env
HUGGINGFACE_TOKEN=hf_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij
```

#### When You Actually Need It:
- Downloading gated models (e.g., Llama 2)
- Avoiding rate limits (>100 requests/hour)
- Accessing private models

**Note:** Most models (DeBERTa, RoBERTa, ALBERT) work WITHOUT this token!

### 4. OPENAI_API_KEY (Optional - Not Currently Used)

**What it is:** API key for OpenAI's GPT models  
**When needed:** Only if you add GPT features  
**Cost:** Pay-per-use (~$0.002 per 1000 tokens)

#### How to Get:

1. Create account at https://platform.openai.com/signup
2. Go to https://platform.openai.com/api-keys
3. Click "Create new secret key"
4. Copy immediately (shown only once)

#### Where to Put It:
```env
OPENAI_API_KEY=sk-proj-abcdefghijklmnopqrstuvwxyz123456789
```

#### Usage Example (If Adding GPT):
```python
# In your code (future feature)
import openai
openai.api_key = os.getenv('OPENAI_API_KEY')
response = openai.ChatCompletion.create(...)
```

### 5. REDDIT_API (Optional - For Reddit Scraping)

**What it is:** Credentials for Reddit's API  
**When needed:** Only if scraping Reddit  
**Cost:** FREE

#### How to Get:

1. Log into Reddit
2. Go to https://www.reddit.com/prefs/apps
3. Click "Create App" or "Create Another App"
4. Fill in:
   - Name: "AI Review Engine"
   - Type: Select "script"
   - Description: "Phone review analysis"
   - Redirect URI: http://localhost:8000
5. Click "Create app"
6. Note the ID (under "personal use script") and SECRET

#### Where to Put It:
```env
REDDIT_CLIENT_ID=ABcd1234EFgh5678
REDDIT_CLIENT_SECRET=IJkl9012MNop3456QRst7890
REDDIT_USER_AGENT=AI-Review-Engine/1.0 by YourUsername
```

### 6. Other Optional Services

#### Google Cloud (For Advanced Features)
```env
# For Google Cloud services (Vision API, Cloud Storage, etc.)
GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json
GOOGLE_CLOUD_PROJECT=your-project-id
```

#### AWS (For S3 Storage, etc.)
```env
AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
AWS_DEFAULT_REGION=us-east-1
```

#### Azure (For Cognitive Services)
```env
AZURE_COGNITIVE_KEY=your-cognitive-services-key
AZURE_COGNITIVE_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
```

## 🔒 Security Best Practices

### 1. Never Commit Keys to Git

**.gitignore should include:**
```gitignore
.env
.env.local
.env.*.local
*.key
*.pem
credentials.json
```

### 2. Use Environment-Specific Files

```bash
.env.development    # Development keys
.env.production     # Production keys
.env.test          # Test keys
```

### 3. Rotate Keys Regularly

```python
# Script to rotate SECRET_KEY
import secrets

def rotate_secret_key():
    new_key = secrets.token_hex(32)
    print(f"New SECRET_KEY: {new_key}")
    # Update .env file
    # Restart application
```

### 4. Use Key Management Services (Production)

**AWS Secrets Manager:**
```python
import boto3

def get_secret(secret_name):
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId=secret_name)
    return response['SecretString']
```

**Azure Key Vault:**
```python
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()
client = SecretClient(vault_url="https://vault.vault.azure.net/", credential=credential)
secret = client.get_secret("secret-name")
```

### 5. Validate Keys on Startup

```python
# In your main application file
def validate_environment():
    required = ['SECRET_KEY', 'DATABASE_URL']
    missing = [key for key in required if not os.getenv(key)]
    
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {missing}")
    
    # Validate SECRET_KEY length
    if len(os.getenv('SECRET_KEY', '')) < 32:
        raise ValueError("SECRET_KEY must be at least 32 characters")
    
    print("✅ Environment validation passed")

# Run on startup
validate_environment()
```

## 🔍 Environment Variable Loading

### Using python-dotenv

```python
# Install
pip install python-dotenv

# In your code
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Access variables
secret_key = os.getenv('SECRET_KEY')
database_url = os.getenv('DATABASE_URL')
```

### Priority Order

Environment variables are loaded in this order (highest priority first):
1. System environment variables
2. `.env.local` file
3. `.env` file
4. Default values in code

## 🆘 Troubleshooting

### Issue: "Missing API Key" Error

**For HuggingFace:**
```python
# Error: "You need to pass a valid `token` to download private models"

# Solution 1: Use public model
model = "distilbert-base-uncased"  # Instead of private model

# Solution 2: Set token
os.environ['HUGGINGFACE_TOKEN'] = 'your_token'

# Solution 3: Use in code
from transformers import AutoModel
model = AutoModel.from_pretrained("model-name", use_auth_token="your_token")
```

### Issue: "Invalid SECRET_KEY"

```python
# Check key length
key = os.getenv('SECRET_KEY', '')
if len(key) < 32:
    print(f"SECRET_KEY too short: {len(key)} characters (need 32+)")

# Generate new key
import secrets
new_key = secrets.token_hex(32)
print(f"Use this: SECRET_KEY={new_key}")
```

### Issue: "Database Connection Failed"

```python
# Test connection
from sqlalchemy import create_engine

# Try SQLite first (always works)
engine = create_engine("sqlite:///test.db")
engine.connect()
print("✅ SQLite works")

# Then test your actual database
db_url = os.getenv('DATABASE_URL')
engine = create_engine(db_url)
try:
    engine.connect()
    print("✅ Database connection successful")
except Exception as e:
    print(f"❌ Connection failed: {e}")
```

## 📊 Cost Comparison

| Service | Free Tier | Paid Usage | Monthly Estimate* |
|---------|-----------|------------|-------------------|
| **Local Models** | ✅ Unlimited | N/A | $0 |
| **HuggingFace** | ✅ Unlimited | Pro: $9/month | $0-9 |
| **OpenAI GPT-3.5** | ❌ None | $0.002/1K tokens | $5-50 |
| **OpenAI GPT-4** | ❌ None | $0.03/1K tokens | $50-500 |
| **Reddit API** | ✅ 60 req/min | Enterprise pricing | $0 |
| **PostgreSQL** | ✅ Local | Cloud: $7+/month | $0-25 |

*Estimates based on moderate usage (10K reviews/month)

## 🚦 Quick Decision Guide

### Do I Need This API Key?

```mermaid
flowchart TD
    Start[Starting AI Review Engine] --> Q1{Testing locally?}
    Q1 -->|Yes| Local[Use SQLite + SECRET_KEY only]
    Q1 -->|No| Q2{Production deployment?}
    
    Q2 -->|Yes| Prod[Need: SECRET_KEY + PostgreSQL]
    Q2 -->|No| Q3{Scraping Reddit?}
    
    Q3 -->|Yes| Reddit[Add Reddit API]
    Q3 -->|No| Q4{Want GPT features?}
    
    Q4 -->|Yes| OpenAI[Add OpenAI API]
    Q4 -->|No| Q5{Using private models?}
    
    Q5 -->|Yes| HF[Add HuggingFace Token]
    Q5 -->|No| Done[No additional keys needed!]
    
    Local --> Done
    Prod --> Q3
    Reddit --> Q4
    OpenAI --> Q5
    HF --> Done
```

## 💡 Recommendations

### For Development/Testing:
```env
# Minimal .env file
SECRET_KEY=dev_key_change_in_production_$(date +%s)
DATABASE_URL=sqlite:///dev_reviews.db
DEBUG=True
```

### For Production:
```env
# Production .env file
SECRET_KEY=${SECURE_RANDOM_KEY}  # Generate with secrets.token_hex(32)
DATABASE_URL=postgresql://${DB_USER}:${DB_PASS}@${DB_HOST}:5432/${DB_NAME}
DEBUG=False
HUGGINGFACE_TOKEN=${HF_TOKEN}  # Optional but recommended
```

### For Full Features:
```env
# All features .env file
SECRET_KEY=${SECURE_KEY}
DATABASE_URL=${POSTGRES_URL}
HUGGINGFACE_TOKEN=${HF_TOKEN}
OPENAI_API_KEY=${OPENAI_KEY}
REDDIT_CLIENT_ID=${REDDIT_ID}
REDDIT_CLIENT_SECRET=${REDDIT_SECRET}
DEBUG=False
```

## 📝 Summary

**The Bottom Line:** You can run the entire AI Phone Review Engine with just:
1. A `SECRET_KEY` you generate yourself (free)
2. A `DATABASE_URL` pointing to SQLite (free)

Everything else is optional and only needed for specific features. The system uses open-source models that run locally, so you don't need expensive API subscriptions to get powerful AI capabilities!

## 🔗 Useful Links

- [HuggingFace Tokens](https://huggingface.co/settings/tokens)
- [OpenAI API Keys](https://platform.openai.com/api-keys)
- [Reddit App Creation](https://www.reddit.com/prefs/apps)
- [PostgreSQL Download](https://www.postgresql.org/download/)
- [Random Key Generator](https://randomkeygen.com/)
- [Environment Variable Best Practices](https://12factor.net/config)
