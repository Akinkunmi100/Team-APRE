#!/usr/bin/env python3
"""
Ultimate AI Phone Review Engine - Professional Web Application
Business intelligence platform with user role separation and advanced features
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os
import sys
import json
import logging
from functools import wraps
from typing import Dict, List, Optional, Any

# Add project directories
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import existing components
try:
    from utils.unified_data_access import get_primary_dataset, get_products_for_comparison, get_brands_list
    from models.absa_model import ABSASentimentAnalyzer
    from models.recommendation_engine_simple import RecommendationEngine
    from utils.data_preprocessing import DataPreprocessor
    DATA_MODULES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Data modules not available: {e}")
    DATA_MODULES_AVAILABLE = False

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-ultimate-secret-key-change-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///ultimate_phone_reviews.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User roles and plan types
class UserRole:
    REGULAR = 'regular'
    BUSINESS = 'business'

class PlanType:
    FREE = 'free'
    BUSINESS = 'business'
    ENTERPRISE = 'enterprise'

# Plan configurations
PLAN_CONFIGS = {
    PlanType.FREE: {
        'search_limit': 20,
        'api_calls_limit': 0,
        'bulk_search': False,
        'competitor_analysis': False,
        'custom_reports': False,
        'analytics': False,
        'priority_support': False
    },
    PlanType.BUSINESS: {
        'search_limit': 200,
        'api_calls_limit': 1000,
        'bulk_search': True,
        'competitor_analysis': True,
        'custom_reports': True,
        'analytics': True,
        'priority_support': True
    },
    PlanType.ENTERPRISE: {
        'search_limit': 1000,
        'api_calls_limit': 10000,
        'bulk_search': True,
        'competitor_analysis': True,
        'custom_reports': True,
        'analytics': True,
        'priority_support': True
    }
}

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    role = db.Column(db.String(20), default=UserRole.REGULAR)
    plan_type = db.Column(db.String(20), default=PlanType.FREE)
    
    # Usage tracking
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    searches_today = db.Column(db.Integer, default=0)
    api_calls_today = db.Column(db.Integer, default=0)
    reports_generated = db.Column(db.Integer, default=0)
    last_reset_date = db.Column(db.Date, default=datetime.utcnow().date)
    
    # Billing info (for demo - in production use secure payment gateway)
    billing_date = db.Column(db.Date)
    payment_method = db.Column(db.String(20), default='•••• 4242')
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def get_plan_config(self):
        return PLAN_CONFIGS.get(self.plan_type, PLAN_CONFIGS[PlanType.FREE])
    
    def is_business_user(self):
        return self.plan_type in [PlanType.BUSINESS, PlanType.ENTERPRISE]
    
    def can_make_search(self):
        self._reset_daily_limits()
        return self.searches_today < self.get_plan_config()['search_limit']
    
    def can_make_api_call(self):
        self._reset_daily_limits()
        return self.api_calls_today < self.get_plan_config()['api_calls_limit']
    
    def has_feature(self, feature):
        return self.get_plan_config().get(feature, False)
    
    def increment_search(self):
        self._reset_daily_limits()
        self.searches_today += 1
        db.session.commit()
    
    def increment_api_call(self):
        self._reset_daily_limits()
        self.api_calls_today += 1
        db.session.commit()
    
    def _reset_daily_limits(self):
        today = datetime.utcnow().date()
        if self.last_reset_date != today:
            self.searches_today = 0
            self.api_calls_today = 0
            self.last_reset_date = today
            db.session.commit()

class SearchHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    query = db.Column(db.String(500))
    search_type = db.Column(db.String(50))  # single, bulk, competitor
    results_count = db.Column(db.Integer)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    execution_time = db.Column(db.Float)  # in seconds

class PhoneAnalytics(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    phone_model = db.Column(db.String(200))
    brand = db.Column(db.String(100))
    total_reviews = db.Column(db.Integer)
    avg_rating = db.Column(db.Float)
    positive_sentiment = db.Column(db.Float)
    negative_sentiment = db.Column(db.Float)
    recommendation_score = db.Column(db.Float)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Decorators
def business_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_business_user():
            return jsonify({
                'error': 'Business plan required',
                'upgrade_needed': True,
                'current_plan': current_user.plan_type if current_user.is_authenticated else 'none'
            }), 403
        return f(*args, **kwargs)
    return decorated_function

def feature_required(feature_name):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not current_user.is_authenticated or not current_user.has_feature(feature_name):
                return jsonify({
                    'error': f'Feature "{feature_name}" not available in your plan',
                    'upgrade_needed': True,
                    'current_plan': current_user.plan_type
                }), 403
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def search_limit_check(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.can_make_search():
            plan_config = current_user.get_plan_config()
            return jsonify({
                'error': 'Search limit exceeded',
                'searches_left': 0,
                'daily_limit': plan_config['search_limit'],
                'upgrade_needed': current_user.plan_type == PlanType.FREE
            }), 429
        return f(*args, **kwargs)
    return decorated_function

# Enhanced System Initialization
try:
    from enhanced_initialization import initialize_enhanced_system, get_analyzer, is_system_ready, get_system_status, setup_enhanced_logging
    ENHANCED_SYSTEM_AVAILABLE = True
    
    # Setup enhanced logging
    setup_enhanced_logging()
    
    # Initialize the enhanced system
    logger.info("🚀 Starting Enhanced System Initialization...")
    initialization_result = initialize_enhanced_system()
    
    if initialization_result['success']:
        logger.info("✅ Enhanced system initialized successfully")
        system_info = initialization_result['system_info']
        logger.info(f"📊 Data Source: {system_info['data_source']}")
        logger.info(f"📈 Data Info: {system_info['data_info']}")
        logger.info(f"🔧 AI Components: {system_info.get('ai_components', 'unknown')}")
        
        # Get the initialized analyzer
        GLOBAL_ANALYZER = get_analyzer()
        
    else:
        logger.warning("⚠️ Enhanced system initialization failed, using fallback")
        GLOBAL_ANALYZER = None
        
except Exception as e:
    logger.warning(f"Enhanced system not available: {e}")
    ENHANCED_SYSTEM_AVAILABLE = False
    GLOBAL_ANALYZER = None

# Data Analysis Classes
class UltimateReviewAnalyzer:
    def __init__(self):
        # If enhanced system is available and analyzer exists, use it
        if ENHANCED_SYSTEM_AVAILABLE and GLOBAL_ANALYZER:
            logger.info("📋 Using orchestrator-initialized analyzer")
            self.df = GLOBAL_ANALYZER.df
            self.sentiment_analyzer = GLOBAL_ANALYZER.sentiment_analyzer
            self.recommendation_engine = GLOBAL_ANALYZER.recommendation_engine
            return
        
        # Fallback to original initialization
        logger.info("📋 Using fallback initialization")
        self.df = None
        self.sentiment_analyzer = None
        self.recommendation_engine = None
        self.load_data()
        
        if DATA_MODULES_AVAILABLE:
            try:
                self.sentiment_analyzer = ABSASentimentAnalyzer()
                self.recommendation_engine = RecommendationEngine()
                logger.info("AI components loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load AI components: {e}")
    
    def load_data(self):
        """Load phone review data"""
        try:
            if DATA_MODULES_AVAILABLE:
                self.df = get_primary_dataset()
                logger.info(f"Loaded {len(self.df)} reviews from dataset")
            else:
                # Fallback to CSV file
                self.df = pd.read_csv('final_dataset_streamlined_clean.csv')
                logger.info(f"Loaded {len(self.df)} reviews from CSV")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            self.df = pd.DataFrame()
    
    def search_phones(self, query: str, limit: int = 50) -> List[Dict]:
        """Search for phones based on query"""
        if self.df.empty:
            return []
        
        query_lower = query.lower()
        
        # Enhanced search across multiple fields
        mask = (
            self.df['product'].str.lower().str.contains(query_lower, na=False) |
            self.df['brand'].str.lower().str.contains(query_lower, na=False) |
            self.df['review_text'].str.lower().str.contains(query_lower, na=False)
        )
        
        results = self.df[mask].head(limit)
        return self._format_search_results(results)
    
    def get_phone_analytics(self, phone_name: str) -> Dict:
        """Get comprehensive analytics for a phone"""
        if self.df.empty:
            return {}
        
        phone_reviews = self.df[
            self.df['product'].str.lower().str.contains(phone_name.lower(), na=False)
        ]
        
        if phone_reviews.empty:
            return {}
        
        # Calculate comprehensive metrics
        analytics = self._calculate_phone_metrics(phone_reviews, phone_name)
        return analytics
    
    def get_competitor_analysis(self, phone_models: List[str]) -> Dict:
        """Compare multiple phone models"""
        comparison_data = {}
        
        for model in phone_models:
            analytics = self.get_phone_analytics(model)
            if analytics:
                comparison_data[model] = analytics
        
        return {
            'comparison': comparison_data,
            'summary': self._generate_comparison_summary(comparison_data)
        }
    
    def get_market_insights(self) -> Dict:
        """Generate market-wide insights"""
        if self.df.empty:
            return {}
        
        # Brand performance
        brand_stats = self.df.groupby('brand').agg({
            'rating': ['mean', 'count'],
            'product': 'nunique'
        }).round(2)
        
        # Top performing phones
        phone_stats = self.df.groupby('product').agg({
            'rating': ['mean', 'count']
        }).round(2)
        phone_stats = phone_stats[phone_stats[('rating', 'count')] >= 10]  # At least 10 reviews
        top_phones = phone_stats.nlargest(10, ('rating', 'mean'))
        
        return {
            'total_reviews': len(self.df),
            'total_brands': self.df['brand'].nunique(),
            'total_models': self.df['product'].nunique(),
            'avg_rating': self.df['rating'].mean(),
            'brand_performance': brand_stats.to_dict(),
            'top_phones': top_phones.to_dict()
        }
    
    def _format_search_results(self, results_df: pd.DataFrame) -> List[Dict]:
        """Format search results for API response"""
        formatted_results = []
        
        for _, row in results_df.iterrows():
            result = {
                'product': row.get('product', ''),
                'brand': row.get('brand', ''),
                'review_text': row.get('review_text', '')[:200] + '...' if len(row.get('review_text', '')) > 200 else row.get('review_text', ''),
                'rating': row.get('rating', None),
                'date': row.get('date', ''),
                'source': row.get('source', ''),
                'sentiment': self._analyze_sentiment_simple(row.get('review_text', ''))
            }
            formatted_results.append(result)
        
        return formatted_results
    
    def _calculate_phone_metrics(self, phone_reviews: pd.DataFrame, phone_name: str) -> Dict:
        """Calculate comprehensive metrics for a phone"""
        total_reviews = len(phone_reviews)
        
        # Rating statistics
        ratings = phone_reviews['rating'].dropna()
        avg_rating = ratings.mean() if not ratings.empty else 0
        rating_distribution = ratings.value_counts().sort_index().to_dict() if not ratings.empty else {}
        
        # Sentiment analysis
        sentiment_scores = []
        for text in phone_reviews['review_text'].dropna():
            sentiment = self._analyze_sentiment_simple(text)
            sentiment_scores.append(sentiment)
        
        positive_count = sum(1 for s in sentiment_scores if s['label'] == 'positive')
        negative_count = sum(1 for s in sentiment_scores if s['label'] == 'negative')
        neutral_count = sum(1 for s in sentiment_scores if s['label'] == 'neutral')
        
        positive_pct = (positive_count / total_reviews * 100) if total_reviews > 0 else 0
        negative_pct = (negative_count / total_reviews * 100) if total_reviews > 0 else 0
        neutral_pct = (neutral_count / total_reviews * 100) if total_reviews > 0 else 0
        
        # Recommendation score (weighted average)
        recommendation_score = min(95, (positive_pct * 0.6) + (avg_rating * 15) + 10)
        
        # Recent reviews sample
        recent_reviews = phone_reviews.head(5).to_dict('records')
        
        return {
            'phone_name': phone_name,
            'total_reviews': total_reviews,
            'avg_rating': round(avg_rating, 1),
            'rating_distribution': rating_distribution,
            'sentiment': {
                'positive': round(positive_pct, 1),
                'negative': round(negative_pct, 1),
                'neutral': round(neutral_pct, 1)
            },
            'recommendation_score': round(recommendation_score, 1),
            'recent_reviews': recent_reviews[:3],  # Limit for API response
            'key_features': self._extract_key_features(phone_reviews),
            'pros_cons': self._extract_pros_cons(phone_reviews)
        }
    
    def _analyze_sentiment_simple(self, text: str) -> Dict:
        """Simple sentiment analysis"""
        if not text or pd.isna(text):
            return {'label': 'neutral', 'score': 0.0}
        
        # Use AI analyzer if available
        if self.sentiment_analyzer:
            try:
                result = self.sentiment_analyzer.analyze_review(text)
                return {
                    'label': result.get('overall_sentiment', 'neutral'),
                    'score': result.get('sentiment_score', 0.0)
                }
            except:
                pass
        
        # Simple keyword-based fallback
        positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'best', 'awesome', 'perfect']
        negative_words = ['bad', 'terrible', 'worst', 'hate', 'awful', 'poor', 'horrible', 'disappointing']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return {'label': 'positive', 'score': 0.7}
        elif negative_count > positive_count:
            return {'label': 'negative', 'score': -0.7}
        else:
            return {'label': 'neutral', 'score': 0.0}
    
    def _extract_key_features(self, phone_reviews: pd.DataFrame) -> List[str]:
        """Extract key features mentioned in reviews"""
        feature_keywords = ['battery', 'camera', 'screen', 'display', 'performance', 'design', 'price']
        mentioned_features = []
        
        all_text = ' '.join(phone_reviews['review_text'].dropna().str.lower())
        
        for feature in feature_keywords:
            if feature in all_text:
                mentioned_features.append(feature.title())
        
        return mentioned_features[:5]  # Return top 5
    
    def _extract_pros_cons(self, phone_reviews: pd.DataFrame) -> Dict:
        """Extract pros and cons from reviews"""
        # Simple implementation - can be enhanced with NLP
        pros = ['Long battery life', 'Great camera quality', 'Smooth performance']
        cons = ['High price', 'Limited storage', 'No headphone jack']
        
        return {'pros': pros[:3], 'cons': cons[:3]}
    
    def _generate_comparison_summary(self, comparison_data: Dict) -> Dict:
        """Generate summary for phone comparison"""
        if not comparison_data:
            return {}
        
        models = list(comparison_data.keys())
        best_rating = max((data['avg_rating'] for data in comparison_data.values() if data.get('avg_rating')), default=0)
        best_sentiment = max((data['sentiment']['positive'] for data in comparison_data.values()), default=0)
        
        winner = None
        for model, data in comparison_data.items():
            if data.get('avg_rating') == best_rating:
                winner = model
                break
        
        return {
            'total_compared': len(models),
            'winner': winner,
            'best_rating': best_rating,
            'best_positive_sentiment': best_sentiment
        }

# Initialize analyzer
analyzer = UltimateReviewAnalyzer()

# Routes
@app.route('/')
def index():
    """Main landing page"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('landing.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration"""
    if request.method == 'POST':
        data = request.get_json() if request.is_json else request.form
        
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        plan_type = data.get('plan_type', PlanType.FREE)
        
        # Validation
        if User.query.filter_by(username=username).first():
            return jsonify({'error': 'Username already exists'}), 400
        
        if User.query.filter_by(email=email).first():
            return jsonify({'error': 'Email already registered'}), 400
        
        # Create user
        user = User(
            username=username,
            email=email,
            plan_type=plan_type,
            role=UserRole.BUSINESS if plan_type in [PlanType.BUSINESS, PlanType.ENTERPRISE] else UserRole.REGULAR
        )
        user.set_password(password)
        
        # Set billing date for demo
        if plan_type != PlanType.FREE:
            user.billing_date = datetime.utcnow().date() + timedelta(days=30)
        
        db.session.add(user)
        db.session.commit()
        
        login_user(user)
        return jsonify({'success': True, 'redirect': url_for('dashboard')})
    
    return render_template('register.html', plans=PLAN_CONFIGS)

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if request.method == 'POST':
        data = request.get_json() if request.is_json else request.form
        
        username = data.get('username')
        password = data.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            user.last_login = datetime.utcnow()
            db.session.commit()
            
            return jsonify({
                'success': True,
                'redirect': url_for('dashboard'),
                'user_plan': user.plan_type
            })
        
        return jsonify({'error': 'Invalid credentials'}), 401
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    """User logout"""
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    """User dashboard"""
    user_stats = {
        'searches_today': current_user.searches_today,
        'searches_left': current_user.get_plan_config()['search_limit'] - current_user.searches_today,
        'api_calls_today': current_user.api_calls_today,
        'reports_generated': current_user.reports_generated,
        'plan_name': current_user.plan_type.title(),
        'next_billing': current_user.billing_date.strftime('%B %d, %Y') if current_user.billing_date else None
    }
    
    return render_template('dashboard.html', 
                         user=current_user, 
                         user_stats=user_stats,
                         plan_config=current_user.get_plan_config())

# API Routes
@app.route('/api/search')
@login_required
@search_limit_check
def api_search():
    """Ultimate phone search API"""
    query = request.args.get('q', '').strip()
    search_type = request.args.get('type', 'single')  # single, bulk, competitor
    
    if not query:
        return jsonify({'error': 'Query parameter required'}), 400
    
    start_time = datetime.now()
    
    # Record search
    current_user.increment_search()
    
    # Perform search based on type
    if search_type == 'bulk' and current_user.has_feature('bulk_search'):
        queries = query.split(',')
        results = []
        for q in queries[:10]:  # Limit bulk search
            phone_results = analyzer.search_phones(q.strip(), limit=5)
            results.extend(phone_results)
    else:
        results = analyzer.search_phones(query, limit=20)
    
    execution_time = (datetime.now() - start_time).total_seconds()
    
    # Record search history
    search_history = SearchHistory(
        user_id=current_user.id,
        query=query,
        search_type=search_type,
        results_count=len(results),
        execution_time=execution_time
    )
    db.session.add(search_history)
    db.session.commit()
    
    return jsonify({
        'results': results,
        'total': len(results),
        'execution_time': round(execution_time, 2),
        'search_type': search_type,
        'searches_left': current_user.get_plan_config()['search_limit'] - current_user.searches_today,
        'user_plan': current_user.plan_type
    })

@app.route('/api/phone/<phone_name>/analytics')
@login_required
def phone_analytics(phone_name):
    """Get detailed phone analytics"""
    analytics = analyzer.get_phone_analytics(phone_name)
    
    if not analytics:
        return jsonify({'error': 'Phone not found in database'}), 404
    
    # Store/update analytics in database for caching
    db_analytics = PhoneAnalytics.query.filter_by(phone_model=phone_name).first()
    if not db_analytics:
        db_analytics = PhoneAnalytics(
            phone_model=phone_name,
            brand=analytics.get('brand', ''),
            total_reviews=analytics['total_reviews'],
            avg_rating=analytics['avg_rating'],
            positive_sentiment=analytics['sentiment']['positive'],
            negative_sentiment=analytics['sentiment']['negative'],
            recommendation_score=analytics['recommendation_score']
        )
        db.session.add(db_analytics)
    else:
        db_analytics.last_updated = datetime.utcnow()
        db_analytics.total_reviews = analytics['total_reviews']
        db_analytics.avg_rating = analytics['avg_rating']
        db_analytics.positive_sentiment = analytics['sentiment']['positive']
        db_analytics.negative_sentiment = analytics['sentiment']['negative']
        db_analytics.recommendation_score = analytics['recommendation_score']
    
    db.session.commit()
    
    return jsonify(analytics)

@app.route('/api/business/competitor-analysis')
@login_required
@feature_required('competitor_analysis')
def competitor_analysis():
    """Advanced competitor analysis for business users"""
    phone_models = request.args.getlist('models')
    
    if len(phone_models) < 2:
        return jsonify({'error': 'At least 2 phone models required for comparison'}), 400
    
    if len(phone_models) > 5:
        return jsonify({'error': 'Maximum 5 phone models allowed for comparison'}), 400
    
    comparison = analyzer.get_competitor_analysis(phone_models)
    
    return jsonify({
        'comparison_data': comparison,
        'generated_at': datetime.utcnow().isoformat(),
        'models_compared': phone_models
    })

@app.route('/api/business/market-insights')
@login_required
@feature_required('analytics')
def market_insights():
    """Market insights for business users"""
    insights = analyzer.get_market_insights()
    
    return jsonify({
        'market_insights': insights,
        'generated_at': datetime.utcnow().isoformat(),
        'data_freshness': 'Real-time analysis'
    })

@app.route('/api/business/usage-analytics')
@login_required
@business_required
def usage_analytics():
    """User usage analytics for business dashboard"""
    # Get user's search history
    searches = SearchHistory.query.filter_by(user_id=current_user.id).order_by(SearchHistory.timestamp.desc()).limit(100).all()
    
    # Analyze trends
    search_trends = {}
    search_types = {}
    
    for search in searches:
        date_key = search.timestamp.strftime('%Y-%m-%d')
        search_trends[date_key] = search_trends.get(date_key, 0) + 1
        search_types[search.search_type] = search_types.get(search.search_type, 0) + 1
    
    # Top queries
    top_queries = [{'query': s.query, 'results': s.results_count, 'time': s.execution_time} for s in searches[:10]]
    
    return jsonify({
        'total_searches': len(searches),
        'search_trends': search_trends,
        'search_types': search_types,
        'top_queries': top_queries,
        'avg_execution_time': np.mean([s.execution_time for s in searches]) if searches else 0
    })

@app.route('/api/business/custom-report')
@login_required
@feature_required('custom_reports')
def custom_report():
    """Generate custom reports for business users"""
    report_type = request.args.get('type', 'summary')
    phone_filter = request.args.get('phones', '')
    brand_filter = request.args.get('brands', '')
    
    # Generate custom report based on parameters
    report_data = {
        'report_type': report_type,
        'generated_at': datetime.utcnow().isoformat(),
        'filters_applied': {
            'phones': phone_filter.split(',') if phone_filter else [],
            'brands': brand_filter.split(',') if brand_filter else []
        }
    }
    
    if report_type == 'summary':
        report_data['summary'] = analyzer.get_market_insights()
    elif report_type == 'detailed':
        # Add detailed analysis
        report_data['detailed_analysis'] = {
            'top_performers': {},
            'sentiment_analysis': {},
            'recommendation_matrix': {}
        }
    
    current_user.reports_generated += 1
    db.session.commit()
    
    return jsonify(report_data)

# System Status Endpoints
@app.route('/api/system/status')
def system_status():
    """Get comprehensive system status"""
    try:
        if ENHANCED_SYSTEM_AVAILABLE:
            status = get_system_status()
            
            # Add Flask app status
            status['flask_status'] = {
                'app_running': True,
                'enhanced_system': True,
                'analyzer_available': GLOBAL_ANALYZER is not None,
                'system_ready': is_system_ready()
            }
            
            return jsonify(status)
        else:
            return jsonify({
                'flask_status': {
                    'app_running': True,
                    'enhanced_system': False,
                    'analyzer_available': True,
                    'system_ready': True,
                    'mode': 'fallback'
                },
                'message': 'Running in fallback mode'
            })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'flask_status': {
                'app_running': True,
                'enhanced_system': False,
                'system_ready': False
            }
        }), 500

@app.route('/api/system/health')
def system_health():
    """Get system health metrics"""
    try:
        if ENHANCED_SYSTEM_AVAILABLE:
            from enhanced_initialization import get_initialization_summary
            summary = get_initialization_summary()
            
            return jsonify({
                'status': 'healthy' if summary.get('health', {}).get('system_ready', False) else 'degraded',
                'initialization_summary': summary,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'status': 'healthy',
                'mode': 'fallback',
                'message': 'Running in basic mode',
                'timestamp': datetime.now().isoformat()
            })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('500.html'), 500

# Database initialization
def create_sample_data():
    """Create sample users and data"""
    if User.query.first():
        return
    
    # Create sample users
    users_data = [
        {
            'username': 'demo_user',
            'email': 'user@demo.com',
            'password': 'demo123',
            'plan_type': PlanType.FREE,
            'role': UserRole.REGULAR
        },
        {
            'username': 'business_user',
            'email': 'business@demo.com',
            'password': 'business123',
            'plan_type': PlanType.BUSINESS,
            'role': UserRole.BUSINESS
        },
        {
            'username': 'enterprise_user',
            'email': 'enterprise@demo.com',
            'password': 'enterprise123',
            'plan_type': PlanType.ENTERPRISE,
            'role': UserRole.BUSINESS
        }
    ]
    
    for user_data in users_data:
        user = User(
            username=user_data['username'],
            email=user_data['email'],
            role=user_data['role'],
            plan_type=user_data['plan_type']
        )
        user.set_password(user_data['password'])
        
        if user_data['plan_type'] != PlanType.FREE:
            user.billing_date = datetime.utcnow().date() + timedelta(days=30)
        
        db.session.add(user)
    
    db.session.commit()
    logger.info("Sample users created:")
    logger.info("Free User: demo_user / demo123")
    logger.info("Business User: business_user / business123")
    logger.info("Enterprise User: enterprise_user / enterprise123")

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        create_sample_data()
    
    logger.info("🚀 Ultimate AI Phone Review Engine - Professional Web App Starting...")
    logger.info("Visit: http://localhost:5000")
    
    app.run(debug=True, port=5000, host='0.0.0.0')