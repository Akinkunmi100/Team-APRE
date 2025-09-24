"""
FastAPI REST API for AI Review Engine
Production-ready API with authentication, rate limiting, and comprehensive endpoints
"""

from fastapi import FastAPI, HTTPException, Depends, Query, Body, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import hashlib
import secrets
import jwt
from passlib.context import CryptContext
import redis
import json
import uuid
import logging
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.models import (
    db_manager, Product, Review, Analysis, 
    User, UserQuery, ScrapingJob, AspectSentiment
)
from models.advanced_ai_model import AdvancedAIEngine
from scrapers.jumia_scraper import JumiaScraper
from scrapers.temu_scraper import TemuScraper
from utils.data_preprocessing import DataPreprocessor

# Initialize FastAPI app
app = FastAPI(
    title="AI-Powered Phone Review Engine API",
    description="Advanced API for phone review analysis with AI",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
ai_engine = AdvancedAIEngine()
preprocessor = DataPreprocessor()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# Redis for caching and rate limiting
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# JWT settings
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

logger = logging.getLogger(__name__)


# Pydantic models for request/response
class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., regex=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    password: str = Field(..., min_length=8)


class UserLogin(BaseModel):
    username: str
    password: str


class ReviewInput(BaseModel):
    text: str = Field(..., min_length=10, max_length=5000)
    rating: Optional[float] = Field(None, ge=1, le=5)
    verified_purchase: bool = False


class ProductSearch(BaseModel):
    query: str = Field(..., min_length=1, max_length=200)
    platform: str = Field("all", regex="^(all|jumia|temu|gsmarena)$")
    max_results: int = Field(20, ge=1, le=100)


class AnalysisRequest(BaseModel):
    product_id: Optional[int] = None
    product_url: Optional[str] = None
    reviews: Optional[List[ReviewInput]] = None
    analysis_type: str = Field("comprehensive", regex="^(sentiment|fake_detection|summary|comprehensive)$")


class ComparisonRequest(BaseModel):
    product_ids: List[int] = Field(..., min_items=2, max_items=5)
    aspects: Optional[List[str]] = None


# Authentication functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    session = db_manager.get_session()
    user = session.query(User).filter_by(username=username).first()
    session.close()
    
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    return user


# Rate limiting decorator
def rate_limit(max_calls: int = 100, window: int = 3600):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            user = kwargs.get('current_user')
            if user:
                key = f"rate_limit:{user.id}"
                try:
                    calls = redis_client.incr(key)
                    if calls == 1:
                        redis_client.expire(key, window)
                    if calls > max_calls:
                        raise HTTPException(status_code=429, detail="Rate limit exceeded")
                except:
                    pass  # Don't fail if Redis is down
            return await func(*args, **kwargs)
        return wrapper
    return decorator


# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "AI-Powered Phone Review Engine API",
        "version": "2.0.0",
        "status": "operational",
        "endpoints": {
            "docs": "/api/docs",
            "health": "/api/health",
            "auth": "/api/auth/*",
            "products": "/api/products/*",
            "reviews": "/api/reviews/*",
            "analysis": "/api/analysis/*"
        }
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "ai_engine": ai_engine.get_model_info(),
        "database": "connected"
    }


@app.post("/api/auth/register")
async def register(user_data: UserCreate):
    """Register a new user"""
    session = db_manager.get_session()
    
    # Check if user exists
    existing_user = session.query(User).filter(
        (User.username == user_data.username) | (User.email == user_data.email)
    ).first()
    
    if existing_user:
        session.close()
        raise HTTPException(status_code=400, detail="Username or email already registered")
    
    # Create new user
    new_user = User(
        username=user_data.username,
        email=user_data.email,
        password_hash=get_password_hash(user_data.password),
        api_key=secrets.token_urlsafe(32)
    )
    
    session.add(new_user)
    session.commit()
    
    # Create access token
    access_token = create_access_token(
        data={"sub": new_user.username},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    
    result = {
        "username": new_user.username,
        "email": new_user.email,
        "api_key": new_user.api_key,
        "access_token": access_token,
        "token_type": "bearer"
    }
    
    session.close()
    return result


@app.post("/api/auth/login")
async def login(user_credentials: UserLogin):
    """Login user"""
    session = db_manager.get_session()
    
    user = session.query(User).filter_by(username=user_credentials.username).first()
    
    if not user or not verify_password(user_credentials.password, user.password_hash):
        session.close()
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Update last login
    user.last_login = datetime.now()
    session.commit()
    
    # Create access token
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    
    result = {
        "access_token": access_token,
        "token_type": "bearer",
        "username": user.username
    }
    
    session.close()
    return result


@app.get("/api/products/search")
@rate_limit(max_calls=100)
async def search_products(
    query: str = Query(..., min_length=1),
    platform: str = Query("all", regex="^(all|jumia|temu|gsmarena)$"),
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user)
):
    """Search for products across platforms"""
    
    # Check cache
    cache_key = f"search:{query}:{platform}:{limit}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return json.loads(cached_result)
    
    session = db_manager.get_session()
    
    # Search in database first
    products = session.query(Product).filter(
        Product.name.ilike(f"%{query}%")
    ).limit(limit).all()
    
    results = []
    for product in products:
        results.append({
            "id": product.id,
            "name": product.name,
            "brand": product.brand,
            "price": product.current_price,
            "rating": product.overall_rating,
            "reviews_count": product.total_reviews,
            "source": product.source_platform
        })
    
    # Cache results
    redis_client.setex(cache_key, 3600, json.dumps(results))
    
    # Log user query
    user_query = UserQuery(
        user_id=current_user.id,
        query_type="search",
        query_text=query,
        parameters={"platform": platform, "limit": limit},
        results={"count": len(results)}
    )
    session.add(user_query)
    session.commit()
    session.close()
    
    return {"results": results, "count": len(results)}


@app.post("/api/analysis/sentiment")
@rate_limit(max_calls=50)
async def analyze_sentiment(
    request: AnalysisRequest,
    current_user: User = Depends(get_current_user)
):
    """Perform advanced sentiment analysis"""
    
    if request.reviews:
        # Analyze provided reviews
        reviews_data = [{"text": r.text, "rating": r.rating} for r in request.reviews]
        
        # Use advanced AI engine
        results = []
        for review in reviews_data:
            analysis = ai_engine.advanced_sentiment_analysis(review['text'])
            aspects = ai_engine.extract_advanced_aspects(review['text'])
            
            results.append({
                "text": review['text'],
                "sentiment": analysis['ensemble_prediction'],
                "confidence": analysis['confidence'],
                "emotions": analysis.get('emotions'),
                "aspects": aspects,
                "entities": analysis.get('entities')
            })
        
        # Generate summary
        texts = [r['text'] for r in reviews_data]
        summary = ai_engine.generate_ai_summary(texts)
        
        return {
            "analyses": results,
            "summary": summary,
            "model_info": ai_engine.get_model_info()
        }
    
    else:
        raise HTTPException(status_code=400, detail="No reviews provided")


@app.post("/api/analysis/fake-detection")
@rate_limit(max_calls=50)
async def detect_fake_reviews(
    reviews: List[ReviewInput],
    current_user: User = Depends(get_current_user)
):
    """Detect fake reviews using advanced AI"""
    
    reviews_data = [{"text": r.text, "rating": r.rating} for r in reviews]
    
    # Use advanced fake detection
    results = ai_engine.detect_fake_reviews_advanced(reviews_data)
    
    # Calculate statistics
    total_reviews = len(results)
    fake_reviews = sum(1 for r in results if r['is_likely_fake'])
    
    return {
        "total_reviews": total_reviews,
        "fake_reviews": fake_reviews,
        "fake_percentage": (fake_reviews / total_reviews * 100) if total_reviews > 0 else 0,
        "detailed_results": results
    }


@app.get("/api/products/{product_id}")
async def get_product(
    product_id: int,
    include_reviews: bool = Query(False),
    current_user: User = Depends(get_current_user)
):
    """Get product details"""
    
    session = db_manager.get_session()
    
    product = session.query(Product).filter_by(id=product_id).first()
    
    if not product:
        session.close()
        raise HTTPException(status_code=404, detail="Product not found")
    
    result = {
        "id": product.id,
        "name": product.name,
        "brand": product.brand,
        "model": product.model,
        "price": product.current_price,
        "original_price": product.original_price,
        "rating": product.overall_rating,
        "total_reviews": product.total_reviews,
        "specifications": product.specifications,
        "source": product.source_platform,
        "last_updated": product.updated_at.isoformat() if product.updated_at else None
    }
    
    if include_reviews:
        reviews = session.query(Review).filter_by(product_id=product_id).limit(50).all()
        result["reviews"] = [
            {
                "id": r.id,
                "text": r.text,
                "rating": r.rating,
                "sentiment": r.sentiment,
                "verified": r.verified_purchase,
                "date": r.review_date.isoformat() if r.review_date else None
            }
            for r in reviews
        ]
    
    session.close()
    return result


@app.post("/api/products/compare")
@rate_limit(max_calls=30)
async def compare_products(
    request: ComparisonRequest,
    current_user: User = Depends(get_current_user)
):
    """Compare multiple products"""
    
    session = db_manager.get_session()
    
    products = session.query(Product).filter(Product.id.in_(request.product_ids)).all()
    
    if len(products) < 2:
        session.close()
        raise HTTPException(status_code=400, detail="At least 2 products required for comparison")
    
    comparison = []
    for product in products:
        # Get sentiment distribution
        sentiments = session.query(Review.sentiment, func.count(Review.id)).filter_by(
            product_id=product.id
        ).group_by(Review.sentiment).all()
        
        sentiment_dist = {s[0]: s[1] for s in sentiments}
        
        comparison.append({
            "id": product.id,
            "name": product.name,
            "brand": product.brand,
            "price": product.current_price,
            "rating": product.overall_rating,
            "total_reviews": product.total_reviews,
            "sentiment_distribution": sentiment_dist,
            "specifications": product.specifications
        })
    
    session.close()
    
    return {
        "products": comparison,
        "comparison_date": datetime.now().isoformat()
    }


@app.post("/api/scraping/start")
@rate_limit(max_calls=10)
async def start_scraping(
    search: ProductSearch,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """Start a scraping job"""
    
    job_id = str(uuid.uuid4())
    
    # Create job record
    session = db_manager.get_session()
    job = ScrapingJob(
        job_id=job_id,
        platform=search.platform,
        query=search.query,
        status="pending",
        started_at=datetime.now()
    )
    session.add(job)
    session.commit()
    session.close()
    
    # Start background scraping
    background_tasks.add_task(run_scraping_job, job_id, search)
    
    return {
        "job_id": job_id,
        "status": "started",
        "message": "Scraping job started in background"
    }


async def run_scraping_job(job_id: str, search: ProductSearch):
    """Background task to run scraping"""
    session = db_manager.get_session()
    job = session.query(ScrapingJob).filter_by(job_id=job_id).first()
    
    try:
        job.status = "running"
        session.commit()
        
        # Initialize scrapers based on platform
        if search.platform == "jumia" or search.platform == "all":
            scraper = JumiaScraper()
            products = scraper.search_phones(search.query, max_pages=2)
            job.products_found += len(products)
        
        if search.platform == "temu" or search.platform == "all":
            scraper = TemuScraper()
            products = scraper.search_products(search.query, max_products=search.max_results)
            job.products_found += len(products)
        
        job.status = "completed"
        job.completed_at = datetime.now()
        
    except Exception as e:
        job.status = "failed"
        job.errors = {"error": str(e)}
        logger.error(f"Scraping job {job_id} failed: {e}")
    
    finally:
        session.commit()
        session.close()


@app.get("/api/scraping/status/{job_id}")
async def get_scraping_status(
    job_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get scraping job status"""
    
    session = db_manager.get_session()
    job = session.query(ScrapingJob).filter_by(job_id=job_id).first()
    
    if not job:
        session.close()
        raise HTTPException(status_code=404, detail="Job not found")
    
    result = {
        "job_id": job.job_id,
        "status": job.status,
        "platform": job.platform,
        "query": job.query,
        "products_found": job.products_found,
        "reviews_scraped": job.reviews_scraped,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "errors": job.errors
    }
    
    session.close()
    return result


@app.post("/api/ai/ask")
@rate_limit(max_calls=50)
async def ask_ai(
    question: str = Body(...),
    context: Optional[str] = Body(None),
    current_user: User = Depends(get_current_user)
):
    """Ask AI a question about products/reviews"""
    
    if not context:
        return {"error": "Context required for question answering"}
    
    answer = ai_engine.answer_question(context, question)
    
    return {
        "question": question,
        "answer": answer.get("answer"),
        "confidence": answer.get("score", 0),
        "context_used": context[:200] + "..." if len(context) > 200 else context
    }


# Error handlers
@app.exception_handler(404)
async def not_found(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Resource not found"}
    )


@app.exception_handler(500)
async def server_error(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
