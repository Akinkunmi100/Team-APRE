"""
Database Models for PostgreSQL
Using SQLAlchemy ORM
"""

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean, 
    DateTime, Text, JSON, ForeignKey, Table, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.sql import func
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

Base = declarative_base()

# Association tables for many-to-many relationships
product_aspect_association = Table(
    'product_aspect_association',
    Base.metadata,
    Column('product_id', Integer, ForeignKey('products.id')),
    Column('aspect_id', Integer, ForeignKey('aspects.id'))
)


class Product(Base):
    """Product model"""
    __tablename__ = 'products'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(500), nullable=False)
    brand = Column(String(200))
    model = Column(String(200))
    category = Column(String(100), default='phone')
    current_price = Column(Float)
    original_price = Column(Float)
    currency = Column(String(10), default='USD')
    overall_rating = Column(Float)
    total_reviews = Column(Integer, default=0)
    specifications = Column(JSON)
    image_url = Column(String(500))
    product_url = Column(String(500))
    source_platform = Column(String(50))  # Jumia, Temu, GSMArena, etc.
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    last_scraped = Column(DateTime)
    
    # Relationships
    reviews = relationship("Review", back_populates="product", cascade="all, delete-orphan")
    analyses = relationship("Analysis", back_populates="product", cascade="all, delete-orphan")
    aspects = relationship("Aspect", secondary=product_aspect_association, back_populates="products")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_product_name', 'name'),
        Index('idx_product_brand', 'brand'),
        Index('idx_product_rating', 'overall_rating'),
    )


class Review(Base):
    """Review model"""
    __tablename__ = 'reviews'
    
    id = Column(Integer, primary_key=True)
    product_id = Column(Integer, ForeignKey('products.id'), nullable=False)
    reviewer_name = Column(String(200))
    rating = Column(Float)
    title = Column(String(500))
    text = Column(Text, nullable=False)
    verified_purchase = Column(Boolean, default=False)
    helpful_count = Column(Integer, default=0)
    has_images = Column(Boolean, default=False)
    image_count = Column(Integer, default=0)
    
    # Sentiment analysis results
    sentiment = Column(String(20))  # positive, negative, neutral
    sentiment_confidence = Column(Float)
    emotion = Column(String(50))
    credibility_score = Column(Float)
    is_spam = Column(Boolean, default=False)
    spam_score = Column(Float)
    fake_probability = Column(Float)
    
    # Metadata
    source_platform = Column(String(50))
    review_date = Column(DateTime)
    scraped_at = Column(DateTime, default=func.now())
    processed_at = Column(DateTime)
    
    # Relationships
    product = relationship("Product", back_populates="reviews")
    aspect_sentiments = relationship("AspectSentiment", back_populates="review", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_review_product', 'product_id'),
        Index('idx_review_sentiment', 'sentiment'),
        Index('idx_review_date', 'review_date'),
        Index('idx_review_credibility', 'credibility_score'),
    )


class Aspect(Base):
    """Product aspect model"""
    __tablename__ = 'aspects'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    category = Column(String(100))  # technical, aesthetic, functional, etc.
    description = Column(Text)
    
    # Relationships
    products = relationship("Product", secondary=product_aspect_association, back_populates="aspects")
    aspect_sentiments = relationship("AspectSentiment", back_populates="aspect")


class AspectSentiment(Base):
    """Aspect-based sentiment for reviews"""
    __tablename__ = 'aspect_sentiments'
    
    id = Column(Integer, primary_key=True)
    review_id = Column(Integer, ForeignKey('reviews.id'), nullable=False)
    aspect_id = Column(Integer, ForeignKey('aspects.id'), nullable=False)
    sentiment = Column(String(20))
    confidence = Column(Float)
    context = Column(Text)
    
    # Relationships
    review = relationship("Review", back_populates="aspect_sentiments")
    aspect = relationship("Aspect", back_populates="aspect_sentiments")
    
    # Indexes
    __table_args__ = (
        Index('idx_aspect_sentiment_review', 'review_id'),
        Index('idx_aspect_sentiment_aspect', 'aspect_id'),
    )


class Analysis(Base):
    """Analysis results model"""
    __tablename__ = 'analyses'
    
    id = Column(Integer, primary_key=True)
    product_id = Column(Integer, ForeignKey('products.id'), nullable=False)
    analysis_type = Column(String(50))  # 'sentiment', 'fake_detection', 'summary', etc.
    
    # Results
    results = Column(JSON)
    summary = Column(Text)
    key_insights = Column(JSON)
    
    # Metrics
    positive_percentage = Column(Float)
    negative_percentage = Column(Float)
    neutral_percentage = Column(Float)
    average_rating = Column(Float)
    
    # AI model info
    model_version = Column(String(20))
    model_confidence = Column(Float)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    product = relationship("Product", back_populates="analyses")


class ScrapingJob(Base):
    """Scraping job tracking"""
    __tablename__ = 'scraping_jobs'
    
    id = Column(Integer, primary_key=True)
    job_id = Column(String(100), unique=True)
    platform = Column(String(50))
    query = Column(String(500))
    status = Column(String(20))  # pending, running, completed, failed
    
    # Statistics
    products_found = Column(Integer, default=0)
    reviews_scraped = Column(Integer, default=0)
    errors = Column(JSON)
    
    # Timestamps
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    created_at = Column(DateTime, default=func.now())


class User(Base):
    """User model for authentication"""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(100), unique=True, nullable=False)
    email = Column(String(200), unique=True, nullable=False)
    password_hash = Column(String(200), nullable=False)
    api_key = Column(String(100), unique=True)
    
    # User preferences
    preferences = Column(JSON)
    
    # Rate limiting
    api_calls_today = Column(Integer, default=0)
    last_api_call = Column(DateTime)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    last_login = Column(DateTime)
    
    # Relationships
    queries = relationship("UserQuery", back_populates="user")
    activities = relationship("UserActivity", back_populates="user")


class UserActivity(Base):
    """User activity tracking"""
    __tablename__ = 'user_activities'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    activity_type = Column(String(50))  # 'view', 'search', 'review', 'compare', etc.
    activity_data = Column(JSON)
    
    # What the user interacted with
    product_id = Column(Integer, ForeignKey('products.id'), nullable=True)
    review_id = Column(Integer, ForeignKey('reviews.id'), nullable=True)
    
    # Activity metadata
    session_id = Column(String(100))
    ip_address = Column(String(45))
    user_agent = Column(String(500))
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="activities")
    product = relationship("Product")
    review = relationship("Review")


class UserQuery(Base):
    """User query history"""
    __tablename__ = 'user_queries'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    query_type = Column(String(50))  # 'search', 'analysis', 'comparison'
    query_text = Column(Text)
    parameters = Column(JSON)
    results = Column(JSON)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="queries")


class ModelPerformance(Base):
    """Track AI model performance metrics"""
    __tablename__ = 'model_performance'
    
    id = Column(Integer, primary_key=True)
    model_name = Column(String(100))
    model_version = Column(String(20))
    
    # Metrics
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    
    # Test details
    test_size = Column(Integer)
    test_date = Column(DateTime, default=func.now())
    
    # Additional metrics
    confusion_matrix = Column(JSON)
    classification_report = Column(JSON)
    feature_importance = Column(JSON)


# Database connection and session management
class DatabaseManager:
    """Database connection manager"""
    
    def __init__(self):
        self.engine = None
        self.Session = None
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database connection"""
        # Get database URL from environment or use SQLite for local development
        database_url = os.getenv('DATABASE_URL', None)
        
        if database_url is None or 'postgresql' in database_url:
            # Use SQLite for local development if PostgreSQL is not available
            db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'review_engine.db')
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            database_url = f'sqlite:///{db_path}'
            print(f"Using SQLite database at: {db_path}")
            
            # Create engine with SQLite-specific settings
            self.engine = create_engine(
                database_url,
                connect_args={'check_same_thread': False},  # For SQLite
                echo=False  # Set to True for SQL query logging
            )
        else:
            # Use PostgreSQL with connection pooling
            self.engine = create_engine(
                database_url,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                echo=False
            )
        
        # Create session factory
        self.Session = sessionmaker(bind=self.engine)
        
        # Create tables if they don't exist
        Base.metadata.create_all(self.engine)
    
    def get_session(self):
        """Get a new database session"""
        return self.Session()
    
    def close_session(self, session):
        """Close a database session"""
        session.close()
    
    def create_tables(self):
        """Create all tables"""
        Base.metadata.create_all(self.engine)
    
    def drop_tables(self):
        """Drop all tables (use with caution!)"""
        Base.metadata.drop_all(self.engine)
    
    def get_or_create(self, session, model, defaults=None, **kwargs):
        """Get or create a database record"""
        instance = session.query(model).filter_by(**kwargs).first()
        if instance:
            return instance, False
        else:
            params = dict((k, v) for k, v in kwargs.items())
            if defaults:
                params.update(defaults)
            instance = model(**params)
            session.add(instance)
            session.commit()
            return instance, True


# Initialize database manager
db_manager = DatabaseManager()
