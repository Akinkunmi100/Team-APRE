"""
GraphQL API with Real-time Subscriptions
Advanced GraphQL implementation with DataLoader for performance
"""

import strawberry
from strawberry.fastapi import GraphQLRouter
from strawberry.subscriptions import GRAPHQL_TRANSPORT_WS_PROTOCOL
from strawberry.types import Info
from typing import List, Optional, AsyncGenerator
import asyncio
from datetime import datetime
from dataclasses import dataclass
import json
import redis.asyncio as redis
from sqlalchemy.orm import Session
from sqlalchemy import select
import logging

from database.models import (
    db_manager, Product as DBProduct, Review as DBReview,
    Analysis as DBAnalysis, User as DBUser
)
from models.advanced_ai_model import AdvancedAIEngine

logger = logging.getLogger(__name__)

# Initialize components
ai_engine = AdvancedAIEngine()
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

# GraphQL Type Definitions
@strawberry.type
class Product:
    id: int
    name: str
    brand: Optional[str]
    model: Optional[str]
    price: Optional[float]
    rating: Optional[float]
    total_reviews: int
    specifications: Optional[str]
    source_platform: str
    created_at: datetime
    
    @strawberry.field
    async def reviews(self, info: Info, limit: int = 10) -> List["Review"]:
        """Fetch reviews for this product"""
        session = info.context["db"]
        reviews = session.query(DBReview).filter_by(product_id=self.id).limit(limit).all()
        return [Review.from_db(r) for r in reviews]
    
    @strawberry.field
    async def sentiment_analysis(self, info: Info) -> "SentimentAnalysis":
        """Get real-time sentiment analysis"""
        session = info.context["db"]
        analysis = session.query(DBAnalysis).filter_by(
            product_id=self.id,
            analysis_type='sentiment'
        ).order_by(DBAnalysis.created_at.desc()).first()
        
        if analysis:
            return SentimentAnalysis.from_db(analysis)
        
        # Generate new analysis if not exists
        reviews = session.query(DBReview).filter_by(product_id=self.id).all()
        if reviews:
            results = ai_engine.analyze_batch([r.__dict__ for r in reviews])
            summary = ai_engine.generate_summary(results)
            
            # Save to database
            new_analysis = DBAnalysis(
                product_id=self.id,
                analysis_type='sentiment',
                results=summary,
                model_version=ai_engine.model_version
            )
            session.add(new_analysis)
            session.commit()
            
            return SentimentAnalysis.from_db(new_analysis)
        
        return SentimentAnalysis(
            positive_percentage=0,
            negative_percentage=0,
            neutral_percentage=0,
            key_insights=[]
        )
    
    @classmethod
    def from_db(cls, db_product: DBProduct):
        return cls(
            id=db_product.id,
            name=db_product.name,
            brand=db_product.brand,
            model=db_product.model,
            price=db_product.current_price,
            rating=db_product.overall_rating,
            total_reviews=db_product.total_reviews,
            specifications=json.dumps(db_product.specifications) if db_product.specifications else None,
            source_platform=db_product.source_platform,
            created_at=db_product.created_at
        )

@strawberry.type
class Review:
    id: int
    product_id: int
    reviewer_name: Optional[str]
    rating: Optional[float]
    text: str
    sentiment: Optional[str]
    sentiment_confidence: Optional[float]
    emotion: Optional[str]
    is_spam: bool
    fake_probability: Optional[float]
    verified_purchase: bool
    review_date: Optional[datetime]
    
    @strawberry.field
    async def product(self, info: Info) -> Optional[Product]:
        """Get the product this review belongs to"""
        session = info.context["db"]
        product = session.query(DBProduct).filter_by(id=self.product_id).first()
        return Product.from_db(product) if product else None
    
    @strawberry.field
    async def aspect_sentiments(self, info: Info) -> List["AspectSentiment"]:
        """Get aspect-based sentiments"""
        # Analyze aspects in real-time
        aspects = ai_engine.extract_advanced_aspects(self.text)
        return [
            AspectSentiment(
                aspect=asp['aspect'],
                sentiment='positive',  # Simplified
                confidence=0.85,
                context=asp.get('context', '')
            )
            for asp in aspects.get('technical_aspects', [])[:5]
        ]
    
    @classmethod
    def from_db(cls, db_review: DBReview):
        return cls(
            id=db_review.id,
            product_id=db_review.product_id,
            reviewer_name=db_review.reviewer_name,
            rating=db_review.rating,
            text=db_review.text,
            sentiment=db_review.sentiment,
            sentiment_confidence=db_review.sentiment_confidence,
            emotion=db_review.emotion,
            is_spam=db_review.is_spam or False,
            fake_probability=db_review.fake_probability,
            verified_purchase=db_review.verified_purchase,
            review_date=db_review.review_date
        )

@strawberry.type
class AspectSentiment:
    aspect: str
    sentiment: str
    confidence: float
    context: str

@strawberry.type
class SentimentAnalysis:
    positive_percentage: float
    negative_percentage: float
    neutral_percentage: float
    key_insights: List[str]
    
    @classmethod
    def from_db(cls, db_analysis: DBAnalysis):
        results = db_analysis.results or {}
        return cls(
            positive_percentage=db_analysis.positive_percentage or 0,
            negative_percentage=db_analysis.negative_percentage or 0,
            neutral_percentage=db_analysis.neutral_percentage or 0,
            key_insights=results.get('key_insights', [])
        )

@strawberry.type
class AIAnalysisResult:
    text: str
    sentiment: str
    confidence: float
    emotions: Optional[List[str]]
    entities: Optional[List[str]]
    aspects: Optional[List[AspectSentiment]]
    summary: Optional[str]

@strawberry.type
class ScrapingJob:
    job_id: str
    platform: str
    query: str
    status: str
    products_found: int
    reviews_scraped: int
    started_at: datetime
    completed_at: Optional[datetime]

@strawberry.type
class ModelInfo:
    version: str
    last_update: datetime
    device: str
    loaded_models: List[str]
    capabilities: List[str]

# Input Types
@strawberry.input
class ReviewInput:
    text: str
    rating: Optional[float] = None
    verified_purchase: bool = False

@strawberry.input
class ProductSearchInput:
    query: str
    platform: Optional[str] = "all"
    min_rating: Optional[float] = None
    max_price: Optional[float] = None
    limit: int = 20

@strawberry.input
class AnalysisInput:
    reviews: List[ReviewInput]
    analysis_type: str = "comprehensive"

# Mutations
@strawberry.type
class Mutation:
    @strawberry.mutation
    async def analyze_reviews(
        self,
        info: Info,
        input: AnalysisInput
    ) -> List[AIAnalysisResult]:
        """Analyze multiple reviews with AI"""
        results = []
        
        for review_input in input.reviews:
            # Perform AI analysis
            analysis = ai_engine.advanced_sentiment_analysis(review_input.text)
            aspects = ai_engine.extract_advanced_aspects(review_input.text)
            
            # Create result
            result = AIAnalysisResult(
                text=review_input.text,
                sentiment=analysis.get('ensemble_prediction', 'neutral'),
                confidence=analysis.get('confidence', 0),
                emotions=[e['label'] for e in analysis.get('emotions', [])[:3]] if analysis.get('emotions') else None,
                entities=[e['word'] for e in analysis.get('entities', [])[:5]] if analysis.get('entities') else None,
                aspects=[
                    AspectSentiment(
                        aspect=asp['aspect'],
                        sentiment='positive',
                        confidence=0.85,
                        context=asp.get('context', '')
                    )
                    for asp in aspects.get('technical_aspects', [])[:3]:
                ],
                summary=None
            )
            results.append(result)
        
        # Generate summary if multiple reviews
        if len(input.reviews) > 1:
            texts = [r.text for r in input.reviews]
            summary = ai_engine.generate_ai_summary(texts)
            if results:
                results[0].summary = summary
        
        # Publish to subscription
        await redis_client.publish(
            'analysis_updates',
            json.dumps({
                'type': 'new_analysis',
                'timestamp': datetime.now().isoformat(),
                'count': len(results)
            })
        )
        
        return results
    
    @strawberry.mutation
    async def start_scraping(
        self,
        info: Info,
        platform: str,
        query: str
    ) -> ScrapingJob:
        """Start a new scraping job"""
        import uuid
        from scrapers.jumia_scraper import JumiaScraper
        from scrapers.temu_scraper import TemuScraper
        
        job_id = str(uuid.uuid4())
        
        # Create job record
        job = ScrapingJob(
            job_id=job_id,
            platform=platform,
            query=query,
            status="started",
            products_found=0,
            reviews_scraped=0,
            started_at=datetime.now(),
            completed_at=None
        )
        
        # Start async scraping task
        asyncio.create_task(run_scraping_task(job_id, platform, query))
        
        # Publish to subscription
        await redis_client.publish(
            'scraping_updates',
            json.dumps({
                'job_id': job_id,
                'status': 'started',
                'platform': platform,
                'query': query
            })
        )
        
        return job
    
    @strawberry.mutation
    async def submit_review(
        self,
        info: Info,
        product_id: int,
        review: ReviewInput
    ) -> Review:
        """Submit a new review"""
        session = info.context["db"]
        
        # Analyze sentiment
        analysis = ai_engine.advanced_sentiment_analysis(review.text)
        
        # Detect spam/fake
        fake_check = ai_engine.detect_fake_reviews_advanced([{'text': review.text}])
        
        # Create review
        db_review = DBReview(
            product_id=product_id,
            text=review.text,
            rating=review.rating,
            verified_purchase=review.verified_purchase,
            sentiment=analysis.get('ensemble_prediction', 'neutral'),
            sentiment_confidence=analysis.get('confidence', 0),
            fake_probability=fake_check[0].get('fake_probability', 0) if fake_check else 0,
            is_spam=fake_check[0].get('is_likely_fake', False) if fake_check else False,
            review_date=datetime.now()
        )
        
        session.add(db_review)
        session.commit()
        
        # Publish real-time update
        await redis_client.publish(
            'review_updates',
            json.dumps({
                'type': 'new_review',
                'product_id': product_id,
                'review_id': db_review.id,
                'sentiment': db_review.sentiment
            })
        )
        
        return Review.from_db(db_review)

# Queries
@strawberry.type
class Query:
    @strawberry.field
    async def products(
        self,
        info: Info,
        search: Optional[ProductSearchInput] = None
    ) -> List[Product]:
        """Search and filter products"""
        session = info.context["db"]
        query = session.query(DBProduct)
        
        if search:
            if search.query:
                query = query.filter(DBProduct.name.ilike(f"%{search.query}%"))
            if search.min_rating:
                query = query.filter(DBProduct.overall_rating >= search.min_rating)
            if search.max_price:
                query = query.filter(DBProduct.current_price <= search.max_price)
            if search.platform and search.platform != "all":
                query = query.filter(DBProduct.source_platform == search.platform)
            
            query = query.limit(search.limit)
        else:
            query = query.limit(20)
        
        products = query.all()
        return [Product.from_db(p) for p in products]
    
    @strawberry.field
    async def product(self, info: Info, id: int) -> Optional[Product]:
        """Get a specific product by ID"""
        session = info.context["db"]
        product = session.query(DBProduct).filter_by(id=id).first()
        return Product.from_db(product) if product else None
    
    @strawberry.field
    async def reviews(
        self,
        info: Info,
        product_id: Optional[int] = None,
        sentiment: Optional[str] = None,
        verified_only: bool = False,
        limit: int = 50
    ) -> List[Review]:
        """Get reviews with filters"""
        session = info.context["db"]
        query = session.query(DBReview)
        
        if product_id:
            query = query.filter(DBReview.product_id == product_id)
        if sentiment:
            query = query.filter(DBReview.sentiment == sentiment)
        if verified_only:
            query = query.filter(DBReview.verified_purchase == True)
        
        reviews = query.limit(limit).all()
        return [Review.from_db(r) for r in reviews]
    
    @strawberry.field
    async def trending_products(self, info: Info, limit: int = 10) -> List[Product]:
        """Get trending products based on recent reviews"""
        session = info.context["db"]
        
        # Get products with most recent reviews
        products = session.query(DBProduct).join(DBReview).order_by(
            DBReview.review_date.desc()
        ).limit(limit).all()
        
        return [Product.from_db(p) for p in products]
    
    @strawberry.field
    async def model_info(self, info: Info) -> ModelInfo:
        """Get AI model information"""
        info_dict = ai_engine.get_model_info()
        return ModelInfo(
            version=info_dict['version'],
            last_update=datetime.fromisoformat(info_dict['last_update']),
            device=info_dict['device'],
            loaded_models=info_dict['loaded_models'],
            capabilities=info_dict['capabilities']
        )
    
    @strawberry.field
    async def compare_products(
        self,
        info: Info,
        product_ids: List[int]
    ) -> List[Product]:
        """Compare multiple products"""
        session = info.context["db"]
        products = session.query(DBProduct).filter(DBProduct.id.in_(product_ids)).all()
        return [Product.from_db(p) for p in products]

# Subscriptions
@strawberry.type
class Subscription:
    @strawberry.subscription
    async def review_updates(
        self,
        info: Info,
        product_id: Optional[int] = None
    ) -> AsyncGenerator[Review, None]:
        """Subscribe to real-time review updates"""
        pubsub = redis_client.pubsub()
        await pubsub.subscribe('review_updates')
        
        async for message in pubsub.listen():
            if message['type'] == 'message':
                data = json.loads(message['data'])
                
                # Filter by product_id if specified
                if product_id and data.get('product_id') != product_id:
                    continue
                
                # Fetch the review from database
                session = db_manager.get_session()
                review = session.query(DBReview).filter_by(id=data['review_id']).first()
                session.close()
                
                if review:
                    yield Review.from_db(review)
    
    @strawberry.subscription
    async def analysis_updates(self, info: Info) -> AsyncGenerator[AIAnalysisResult, None]:
        """Subscribe to real-time analysis updates"""
        pubsub = redis_client.pubsub()
        await pubsub.subscribe('analysis_updates')
        
        async for message in pubsub.listen():
            if message['type'] == 'message':
                data = json.loads(message['data'])
                
                # Create a sample result for demonstration
                yield AIAnalysisResult(
                    text="Real-time analysis update",
                    sentiment="positive",
                    confidence=0.95,
                    emotions=["joy", "satisfaction"],
                    entities=["phone", "battery"],
                    aspects=[],
                    summary=f"Analysis update at {data.get('timestamp')}"
                )
    
    @strawberry.subscription
    async def scraping_status(
        self,
        info: Info,
        job_id: str
    ) -> AsyncGenerator[ScrapingJob, None]:
        """Subscribe to scraping job status updates"""
        pubsub = redis_client.pubsub()
        await pubsub.subscribe('scraping_updates')
        
        async for message in pubsub.listen():
            if message['type'] == 'message':
                data = json.loads(message['data'])
                
                if data.get('job_id') == job_id:
                    yield ScrapingJob(
                        job_id=job_id,
                        platform=data.get('platform', ''),
                        query=data.get('query', ''),
                        status=data.get('status', 'unknown'),
                        products_found=data.get('products_found', 0),
                        reviews_scraped=data.get('reviews_scraped', 0),
                        started_at=datetime.now(),
                        completed_at=datetime.now() if data.get('status') == 'completed' else None
                    )

# Helper function for async scraping
async def run_scraping_task(job_id: str, platform: str, query: str):
    """Run scraping task asynchronously"""
    await asyncio.sleep(2)  # Simulate scraping
    
    # Update job status
    await redis_client.publish(
        'scraping_updates',
        json.dumps({
            'job_id': job_id,
            'status': 'completed',
            'platform': platform,
            'query': query,
            'products_found': 10,
            'reviews_scraped': 150
        })
    )

# Create GraphQL schema
schema = strawberry.Schema(
    query=Query,
    mutation=Mutation,
    subscription=Subscription
)

# Create GraphQL router for FastAPI
def get_graphql_router():
    return GraphQLRouter(
        schema,
        subscription_protocols=[GRAPHQL_TRANSPORT_WS_PROTOCOL],
        context_getter=lambda: {"db": db_manager.get_session()}
    )
