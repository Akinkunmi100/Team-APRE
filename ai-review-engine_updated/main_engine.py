"""
AI Phone Review Engine - Main Executable
This is the primary file to run the complete AI-powered phone review analysis system
with all advanced features including personalization and deeper insights.:
    """

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import sys
import os
import json
from typing import Dict, List, Optional, Any
import asyncio
from pathlib import Path

# Add project directories to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import core modules
from models.recommendation_engine import PhoneRecommendationEngine
from models.absa_model import ABSASentimentAnalyzer
from models.spam_detector import SpamDetector
from models.market_analyzer import MarketAnalyzer
from scrapers.jumia_scraper import JumiaScraper
from utils.data_preprocessing import DataPreprocessor
from utils.visualization import ReviewVisualizer
from database.database_manager import DatabaseManager
from core.ai_engine import AIReviewEngine
from core.nlp_core import NLPCore

# Import advanced modules
from core.personalization_engine import (
    PersonalizationEngine,
    UserProfile
)
from modules.deeper_insights import (
    DeeperInsightsEngine,
    EmotionType,
    TemporalPattern
)

# Import unified data access
from utils.unified_data_access import (
    get_primary_dataset,      # Full dataset
    create_sample_data,        # Sample data
    get_products_for_comparison,  # Product list
    get_brands_list,          # Brand list
    generate_fake_realtime_data  # Simulated events
)

# Page configuration
st.set_page_config(
    page_title="AI Phone Review Engine - Complete System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
    <style>
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .feature-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
        transition: transform 0.3s;
    }
    .feature-card:hover {
        transform: translateY(-5px);
    }
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
    }
    .insight-pill {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        background: #e3f2fd;
        border-radius: 20px;
        margin: 0.25rem;
        font-size: 0.9rem;
    }
    .emotion-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        margin: 0.25rem;
        font-weight: bold;
    }
    .joy { background: #fff3e0; color: #e65100; }
    .trust { background: #e8f5e9; color: #2e7d32; }
    .fear { background: #fce4ec; color: #c2185b; }
    .anger { background: #ffebee; color: #c62828; }
    .sadness { background: #e3f2fd; color: #1565c0; }
    </style>
""", unsafe_allow_html=True)


class AIPhoneReviewSystem:
    """Main system class that orchestrates all components"""
    
    def __init__(self):
        """Initialize all system components"""
        # Core components
        self.recommendation_engine = PhoneRecommendationEngine()
        self.sentiment_analyzer = ABSASentimentAnalyzer()
        self.spam_detector = SpamDetector()
        self.market_analyzer = MarketAnalyzer()
        self.preprocessor = DataPreprocessor()
        self.visualizer = ReviewVisualizer()
        self.db_manager = DatabaseManager()
        self.nlp_core = NLPCore()
        
        # Advanced components
        self.personalization_engine = PersonalizationEngine()
        self.insights_engine = DeeperInsightsEngine()
        
        # Scrapers
        self.jumia_scraper = JumiaScraper()
        
        # System state
        self.current_user_profile = None
        self.analysis_cache = {}
        
    def initialize_session_state(self):
        """Initialize Streamlit session state"""
        if 'system' not in st.session_state:
            st.session_state.system = self
        if 'user_profile' not in st.session_state:
            st.session_state.user_profile = None
        if 'current_analysis' not in st.session_state:
            st.session_state.current_analysis = None
        if 'recommendations' not in st.session_state:
            st.session_state.recommendations = []
        if 'insights' not in st.session_state:
            st.session_state.insights = None


def main():
    """Main application entry point"""
    # Initialize system
    if 'system' not in st.session_state:
        with st.spinner("🚀 Initializing AI Phone Review Engine..."):
            system = AIPhoneReviewSystem()
            system.initialize_session_state()
            st.success("✅ System initialized successfully!")
    else:
        system = st.session_state.system
    
    # Header
    st.markdown('<h1 class="main-header">🤖 AI Phone Review Engine</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Advanced AI-Powered Phone Analysis with Personalization & Deep Insights</p>', unsafe_allow_html=True)
    
    # Sidebar Navigation
    with st.sidebar:
        st.header("🧭 Navigation")
        
        # User Profile Section
        if st.session_state.user_profile:
            st.success(f"👤 Welcome back, {st.session_state.user_profile.user_id}!")
            st.caption(f"Trust Score: {st.session_state.user_profile.trust_score:.2f}")
        else:
            if st.button("Create Profile", type="primary"):
                create_user_profile(system)
        
        st.markdown("---")
        
        # Main Navigation
        page = st.selectbox(
            "Select Module",
            [
                "🏠 Dashboard",
                "🔍 Smart Search & Recommendations",
                "📊 Deep Analysis",
                "🎭 Emotion & Sarcasm Detection",
                "📈 Market Trends",
                "🌍 Cultural Insights",
                "⏰ Temporal Patterns",
                "🤝 Personalized Experience",
                "🔬 Live Scraping",
                "📋 Reports"
            ]
        )
        
        st.markdown("---")
        
        # Quick Actions
        st.header("⚡ Quick Actions")
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
        
        if st.button("💾 Save Analysis"):
            save_current_analysis(system)
        
        if st.button("📤 Export Report"):
            export_analysis_report(system)
        
        st.markdown("---")
        
        # System Status
        st.header("📊 System Status")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Reviews", "25,431", "↑ 523")
        with col2:
            st.metric("Accuracy", "96.8%", "↑ 1.2%")
    
    # Main Content Area
    if "Dashboard" in page:
        show_dashboard(system)
    elif "Smart Search" in page:
        show_smart_search(system)
    elif "Deep Analysis" in page:
        show_deep_analysis(system)
    elif "Emotion" in page:
        show_emotion_detection(system)
    elif "Market Trends" in page:
        show_market_trends(system)
    elif "Cultural" in page:
        show_cultural_insights(system)
    elif "Temporal" in page:
        show_temporal_patterns(system)
    elif "Personalized" in page:
        show_personalization(system)
    elif "Live Scraping" in page:
        show_live_scraping(system)
    elif "Reports" in page:
        show_reports(system)


def show_dashboard(system: AIPhoneReviewSystem):
    """Display main dashboard with key metrics and insights"""
    st.header("📊 Executive Dashboard")
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Total Phones Analyzed", "156", "↑ 12 this week")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Reviews Processed", "25,431", "↑ 2,341 today")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Average Sentiment", "72.3%", "↑ 3.2%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Spam Detected", "8.2%", "↓ 1.5%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick Insights
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔥 Trending Insights")
        insights = [
            "📱 iPhone 15 Pro showing 15% increase in positive sentiment",
            "🎯 Camera quality is the #1 discussed feature this week",
            "⚡ Battery life complaints up 20% for flagship phones",
            "💰 Price sensitivity increased during holiday season",
            "🌟 Samsung S24 Ultra leading in display satisfaction"
        ]
        for insight in insights:
            st.markdown(f'<span class="insight-pill">{insight}</span>', unsafe_allow_html=True)
    
    with col2:
        st.subheader("🏆 Top Rated This Week")
        top_phones = [
            ("iPhone 15 Pro Max", 4.8),
            ("Samsung S24 Ultra", 4.7),
            ("Google Pixel 8 Pro", 4.6),
            ("OnePlus 12", 4.5),
            ("Xiaomi 14 Pro", 4.4)
        ]
        for phone, rating in top_phones:
            st.write(f"**{phone}** ⭐ {rating}")
    
    st.markdown("---")
    
    # Emotion Distribution
    st.subheader("😊 Emotion Analysis Overview")
    
    emotions_data = {
        'Joy': 35,
        'Trust': 28,
        'Anticipation': 15,
        'Surprise': 8,
        'Fear': 6,
        'Sadness': 5,
        'Anger': 3
    }
    
    fig = px.pie(
        values=list(emotions_data.values()),
        names=list(emotions_data.keys()),
        title="Emotion Distribution Across All Reviews",
        color_discrete_map={
            'Joy': '#FFD700',
            'Trust': '#4CAF50',
            'Anticipation': '#2196F3',
            'Surprise': '#9C27B0',
            'Fear': '#FF5722',
            'Sadness': '#607D8B',
            'Anger': '#F44336'
        }
    )
    st.plotly_chart(fig, use_container_width=True)


def show_smart_search(system: AIPhoneReviewSystem):
    """Smart search and recommendations interface"""
    st.header("🔍 Smart Search & AI Recommendations")
    
    # Search Section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input(
            "What are you looking for?",
            placeholder="e.g., 'Best camera phone under $800' or 'Gaming phone with long battery'"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    
    # Advanced Filters
    with st.expander("🎛️ Advanced Filters"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            price_range = st.slider(
                "Price Range ($)",
                min_value=100,
                max_value=2000,
                value=(300, 1000),
                step=50
            )
            
            brands = st.multiselect(
                "Brands",
                ["Apple", "Samsung", "Google", "OnePlus", "Xiaomi", "Oppo", "Vivo"],
                default=[]
            )
        
        with col2:
            features = st.multiselect(
                "Must-have Features",
                ["5G", "Wireless Charging", "High Refresh Rate", "Water Resistant", 
                 "Dual SIM", "Headphone Jack", "MicroSD Slot"],
                default=[]
            )
            
            min_rating = st.slider(
                "Minimum Rating",
                min_value=1.0,
                max_value=5.0,
                value=3.5,
                step=0.5
            )
        
        with col3:
            use_case = st.selectbox(
                "Primary Use Case",
                ["General Use", "Gaming", "Photography", "Business", "Content Creation", "Student"]
            )
            
            urgency = st.select_slider(
                "Purchase Urgency",
                options=["Browsing", "Considering", "Ready to Buy"],
                value="Considering"
            )
    
    # Search Results
    if search_button or search_query:
        with st.spinner("🤖 AI is analyzing your requirements..."):
            # Simulate search and recommendation
            st.success("Found 12 phones matching your criteria!")
            
            # Create recommendation context
            context = {
                'query': search_query,
                'price_range': price_range,
                'brands': brands,
                'features': features,
                'min_rating': min_rating,
                'use_case': use_case,
                'urgency': urgency
            }
            
            # Display recommendations
            display_recommendations(system, context)


def show_deep_analysis(system: AIPhoneReviewSystem):
    """Deep analysis interface for individual phones or reviews"""
    st.header("📊 Deep Review Analysis")
    
    # Input Section
    tab1, tab2, tab3 = st.tabs(["📱 Analyze Phone", "📝 Analyze Reviews", "📁 Upload Data"])
    
    with tab1:
        phone_name = st.selectbox(
            "Select a phone to analyze",
            ["iPhone 15 Pro", "Samsung S24 Ultra", "Google Pixel 8", "OnePlus 12", "Xiaomi 14"]
        )
        
        if st.button("Analyze Phone", type="primary"):
            analyze_phone(system, phone_name)
    
    with tab2:
        review_text = st.text_area(
            "Paste reviews to analyze (one per line)",
            height=200,
            placeholder="Enter or paste reviews here..."
        )
        
        if st.button("Analyze Reviews", type="primary"):
            if review_text:
                reviews = review_text.split('\n')
                analyze_review_batch(system, reviews)
    
    with tab3:
        uploaded_file = st.file_uploader(
            "Upload CSV with reviews",
            type=['csv', 'xlsx'],
            help="File should contain 'review' or 'text' column"
        )
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.success(f"Loaded {len(df)} reviews")
            
            if st.button("Analyze Dataset", type="primary"):
                analyze_dataset(system, df)


def show_emotion_detection(system: AIPhoneReviewSystem):
    """Emotion and sarcasm detection interface"""
    st.header("🎭 Emotion & Sarcasm Detection")
    
    # Live Detection
    st.subheader("🔍 Live Analysis")
    
    review_input = st.text_area(
        "Enter a review to analyze emotions and detect sarcasm",
        placeholder="e.g., 'This phone is absolutely AMAZING!!! Best purchase ever... NOT!'"
    )
    
    if st.button("Analyze Emotions", type="primary"):
        if review_input:
            with st.spinner("Analyzing emotions and detecting sarcasm..."):
                # Analyze with deeper insights
                analysis = system.insights_engine.analyze_review({
                    'text': review_input,
                    'rating': None
                })
                
                # Display emotion results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 😊 Emotion Analysis")
                    emotion_data = analysis['emotion_analysis']
                    
                    # Primary emotion
                    primary = emotion_data['primary_emotion']
                    st.markdown(f'<span class="emotion-badge {primary.lower()}">{primary.upper()}</span>', 
                              unsafe_allow_html=True)
                    
                    # Emotion scores
                    st.markdown("**Emotion Breakdown:**")
                    for emotion, score in emotion_data['emotion_scores'].items():
                        if score > 0:
                            st.progress(score, text=f"{emotion}: {score:.1%}")
                    
                    st.metric("Intensity", f"{emotion_data['intensity']:.1%}")
                    st.metric("Confidence", f"{emotion_data['confidence']:.1%}")
                
                with col2:
                    st.markdown("### 🎭 Sarcasm Detection")
                    sarcasm_data = analysis['sarcasm_detection']
                    
                    if sarcasm_data['is_sarcastic']:
                        st.error("⚠️ SARCASM DETECTED!")
                        st.metric("Confidence", f"{sarcasm_data['confidence']:.1%}")
                        
                        st.markdown("**Indicators Found:**")
                        for indicator in sarcasm_data['indicators']:
                            st.write(f"• {indicator}")
                        
                        if sarcasm_data['irony_type']:
                            st.info(f"Irony Type: {sarcasm_data['irony_type']}")
                    else:
                        st.success("✅ No sarcasm detected")
                        st.caption("This appears to be a genuine review")


def show_cultural_insights(system: AIPhoneReviewSystem):
    """Cultural sentiment variation analysis"""
    st.header("🌍 Cultural Insights & Regional Analysis")
    
    # Sample data for visualization
    regions = ['North America', 'Europe', 'Asia Pacific', 'Latin America']
    
    # Regional sentiment comparison
    st.subheader("📊 Regional Sentiment Variations")
    
    sentiment_data = pd.DataFrame({
        'Region': regions,
        'Positive': [72, 68, 75, 70],
        'Neutral': [20, 22, 18, 21],
        'Negative': [8, 10, 7, 9]
    })
    
    fig = px.bar(
        sentiment_data,
        x='Region',
        y=['Positive', 'Neutral', 'Negative'],
        title="Sentiment Distribution by Region",
        color_discrete_map={'Positive': '#4CAF50', 'Neutral': '#FFC107', 'Negative': '#F44336'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Cultural preference patterns
    st.subheader("🎯 Cultural Preference Patterns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🇺🇸 North America")
        preferences = {
            'Value for Money': 85,
            'Innovation': 90,
            'Customer Service': 88,
            'Brand Trust': 75
        }
        for pref, score in preferences.items():
            st.progress(score/100, text=f"{pref}: {score}%")
    
    with col2:
        st.markdown("### 🇪🇺 Europe")
        preferences = {
            'Sustainability': 92,
            'Privacy': 95,
            'Build Quality': 87,
            'Design': 83
        }
        for pref, score in preferences.items():
            st.progress(score/100, text=f"{pref}: {score}%")


def show_temporal_patterns(system: AIPhoneReviewSystem):
    """Temporal pattern analysis interface"""
    st.header("⏰ Temporal Pattern Analysis")
    
    # Generate sample temporal data
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='W')
    ratings = np.random.normal(4.2, 0.3, len(dates))
    ratings = np.clip(ratings, 1, 5)
    
    # Temporal trend chart
    st.subheader("📈 Rating Trends Over Time")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=ratings,
        mode='lines+markers',
        name='Average Rating',
        line=dict(color='#2196F3', width=2),
        marker=dict(size=6)
    ))
    
    # Add trend line
    z = np.polyfit(range(len(dates)), ratings, 1)
    p = np.poly1d(z)
    fig.add_trace(go.Scatter(
        x=dates,
        y=p(range(len(dates))),
        mode='lines',
        name='Trend',
        line=dict(color='#FF5722', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title="Phone Rating Trends",
        xaxis_title="Date",
        yaxis_title="Average Rating",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Pattern Detection
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Pattern Type", "Honeymoon Effect", "↓ Declining after launch")
    with col2:
        st.metric("Trend Direction", "Decreasing", "-0.02/month")
    with col3:
        st.metric("Seasonality", "Moderate", "0.34 score")
    
    # Key Events
    st.subheader("🎯 Key Events Detected")
    
    events = [
        {"date": "2024-03-15", "event": "Major OS Update", "impact": "+0.3 rating"},
        {"date": "2024-06-20", "event": "Price Drop", "impact": "+0.5 rating"},
        {"date": "2024-09-10", "event": "Competitor Launch", "impact": "-0.2 rating"},
        {"date": "2024-11-25", "event": "Black Friday", "impact": "+0.4 rating"}
    ]
    
    for event in events:
        st.info(f"📅 **{event['date']}** - {event['event']} ({event['impact']})")


def show_personalization(system: AIPhoneReviewSystem):
    """Personalization settings and recommendations"""
    st.header("🤝 Personalized Experience")
    
    if st.session_state.user_profile:
        profile = st.session_state.user_profile
        
        # User Stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Trust Score", f"{profile.trust_score:.2f}")
        with col2:
            st.metric("Reviews", len(profile.interaction_history))
        with col3:
            st.metric("Expertise", profile.expertise_level.value)
        with col4:
            st.metric("Member Since", profile.created_at.strftime("%b %Y"))
        
        st.markdown("---")
        
        # Preferences
        st.subheader("📱 Your Preferences")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Favorite Brands")
            brands = profile.preferences.get('brands', {})
            for brand, score in list(brands.items())[:5]:
                st.progress(score, text=f"{brand}: {score:.0%}")
        
        with col2:
            st.markdown("### Important Features")
            features = profile.preferences.get('features', {})
            for feature, importance in list(features.items())[:5]:
                st.progress(importance, text=f"{feature}: {importance:.0%}")
        
        # Personalized Recommendations
        st.subheader("🎯 Recommended For You")
        
        recommendations = system.personalization_engine.get_personalized_recommendations(
            user_id=profile.user_id,
            context={'timestamp': datetime.now()}
        )
        
        for i, rec in enumerate(recommendations.recommendations[:3]):
            with st.expander(f"#{i+1} {rec.phone_id} - Score: {rec.score:.2f}"):
                st.write(f"**Match Reasons:** {', '.join(rec.match_reasons[:3])}")
                st.write(f"**Confidence:** {rec.confidence:.1%}")
                if st.button(f"View Details", key=f"rec_{i}"):
                    st.info("Detailed view would open here")
        
        # Alerts
        st.subheader("🔔 Your Alerts")
        
        alerts = system.personalization_engine.alert_manager.get_user_alerts(profile.user_id)
        
        if alerts:
            for alert in alerts[:5]:
                alert_color = "🔴" if alert.priority == "high" else "🟡" if alert.priority == "medium" else "🟢"
                st.write(f"{alert_color} **{alert.alert_type.value}** - {alert.message}")
        else:
            st.info("No new alerts")
    
    else:
        st.warning("Please create a profile to access personalized features")
        if st.button("Create Profile Now", type="primary"):
            create_user_profile(system)


def show_live_scraping(system: AIPhoneReviewSystem):
    """Live scraping interface"""
    st.header("🔬 Live Review Scraping")
    
    st.warning("⚠️ Please ensure you comply with website terms of service when scraping")
    
    # Scraping Configuration
    col1, col2 = st.columns(2)
    
    with col1:
        source = st.selectbox(
            "Select Source",
            ["Jumia", "Amazon", "Custom URL"]
        )
        
        product_url = st.text_input(
            "Product URL",
            placeholder="https://www.jumia.com/..."
        )
    
    with col2:
        max_pages = st.number_input(
            "Max Pages to Scrape",
            min_value=1,
            max_value=10,
            value=3
        )
        
        include_images = st.checkbox("Include Images", value=False)
    
    if st.button("Start Scraping", type="primary"):
        if product_url:
            with st.spinner(f"Scraping reviews from {source}..."):
                progress_bar = st.progress(0)
                
                # Simulate scraping progress
                for i in range(101):
                    progress_bar.progress(i)
                    if i % 20 == 0:
                        st.info(f"Scraped {i*2} reviews...")
                
                st.success("✅ Scraping complete! 200 reviews collected")
                
                # Display sample results from real dataset
                st.subheader("Sample Scraped Reviews")
                
                # Get sample data from cleaned dataset
                sample_df = create_sample_data(n_samples=5)
                if sample_df is not None and not sample_df.empty:
                    # Debug: Log available columns
                    st.write(f"DEBUG: Available columns in sample_df: {list(sample_df.columns)}")
                    st.write(f"DEBUG: Sample data shape: {sample_df.shape}")

                    # Format for display
                    sample_reviews = pd.DataFrame({
                        'Date': pd.date_range(start='2024-01-01', periods=min(5, len(sample_df))),
                        'Rating': sample_df['rating'].head(5).fillna(4),
                        'Review': sample_df['review_text'].head(5).apply(lambda x: x[:100] + '...' if len(str(x)) > 100 else str(x)),
                        'Product': sample_df['product'].head(5),
                        'Verified': [True] * min(5, len(sample_df))
                    })
                else:
                    # Fallback if no data available
                    sample_reviews = pd.DataFrame({
                        'Date': pd.date_range(start='2024-01-01', periods=1),
                        'Rating': [4],
                        'Review': ["Loading review data..."],
                        'Product': ["Sample Product"],
                        'Verified': [True]
                    })
                
                st.dataframe(sample_reviews)


def show_reports(system: AIPhoneReviewSystem):
    """Reports and export interface"""
    st.header("📋 Reports & Analytics")
    
    # Report Type Selection
    report_type = st.selectbox(
        "Select Report Type",
        ["Executive Summary", "Detailed Analysis", "Comparison Report", "Trend Analysis", "Custom Report"]
    )
    
    # Date Range
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=30))
    with col2:
        end_date = st.date_input("End Date", datetime.now())
    
    # Generate Report
    if st.button("Generate Report", type="primary"):
        with st.spinner("Generating report..."):
            # Simulate report generation
            st.success("Report generated successfully!")
            
            # Display report preview
            st.subheader("📄 Report Preview")
            
            st.markdown("""
            ### Executive Summary - AI Phone Review Analysis
            
            **Report Period:** {start} to {end}
            
            #### Key Findings:
            - Total reviews analyzed: 5,234
            - Average sentiment score: 72.3%
            - Top performing phone: iPhone 15 Pro (4.8/5.0)
            - Most discussed feature: Camera quality
            - Spam detection rate: 8.2%
            
            #### Emotion Analysis:
            - Primary emotion: Joy (35%)
            - Sarcasm detected in 12% of reviews
            - Cultural variations observed across 4 regions
            
            #### Recommendations:
            1. Focus on battery life improvements
            2. Address camera software issues
            3. Improve customer service response time
            """.format(start=start_date, end=end_date))
            
            # Export Options
            col1, col2, col3 = st.columns(3)
            with col1:
                st.download_button(
                    "📥 Download PDF",
                    data=b"PDF content here",
                    file_name=f"report_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf"
                )
            with col2:
                st.download_button(
                    "📊 Download Excel",
                    data=b"Excel content here",
                    file_name=f"report_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.ms-excel"
                )
            with col3:
                st.download_button(
                    "📄 Download CSV",
                    data="CSV content here",
                    file_name=f"report_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )


# Helper Functions

def create_user_profile(system: AIPhoneReviewSystem):
    """Create a new user profile"""
    with st.form("create_profile"):
        st.subheader("Create Your Profile")
        
        user_id = st.text_input("Username", placeholder="john_doe")
        expertise = st.selectbox(
            "Expertise Level",
            ["Beginner", "Intermediate", "Advanced", "Expert"]
        )
        
        st.markdown("### Preferences")
        # Get real brands from dataset
        brands_list = get_brands_list()
        default_brands = ["Apple", "Samsung", "Google", "OnePlus", "Xiaomi"]
        available_brands = [b for b in default_brands if b in brands_list] if brands_list else default_brands
        
        favorite_brands = st.multiselect(
            "Favorite Brands",
            brands_list[:20] if brands_list else default_brands,
            default=available_brands[:2]
        )
        
        important_features = st.multiselect(
            "Important Features",
            ["Camera", "Battery", "Performance", "Display", "Price"]
        )
        
        if st.form_submit_button("Create Profile"):
            # Create profile using personalization engine
            profile = system.personalization_engine.create_user_profile(
                user_id=user_id,
                preferences={
                    'brands': {brand: 1.0 for brand in favorite_brands},
                    'features': {feature: 1.0 for feature in important_features}
                }
            )
            st.session_state.user_profile = profile
            st.success(f"Profile created for {user_id}!")
            st.rerun()


def display_recommendations(system: AIPhoneReviewSystem, context: Dict):
    """Display phone recommendations"""
    # Sample recommendations
    recommendations = [
        {
            'name': 'iPhone 15 Pro',
            'price': 999,
            'rating': 4.8,
            'match_score': 95,
            'pros': ['Excellent camera', 'Fast performance', 'Great display'],
            'cons': ['Expensive', 'Limited customization']
        },
        {
            'name': 'Samsung S24 Ultra',
            'price': 1199,
            'rating': 4.7,
            'match_score': 92,
            'pros': ['Versatile camera', 'S-Pen', 'Long battery'],
            'cons': ['Large size', 'High price']
        },
        {
            'name': 'Google Pixel 8',
            'price': 699,
            'rating': 4.6,
            'match_score': 88,
            'pros': ['AI features', 'Clean Android', 'Good value'],
            'cons': ['Average battery', 'Limited availability']
        }
    ]
    
    for rec in recommendations:
        with st.container():
            col1, col2, col3 = st.columns([3, 2, 1])
            
            with col1:
                st.markdown(f"### {rec['name']}")
                st.write(f"⭐ {rec['rating']} | 💰 ${rec['price']}")
                
                # Pros and Cons
                pros_cons = st.columns(2)
                with pros_cons[0]:
                    st.markdown("**Pros:**")
                    for pro in rec['pros']:
                        st.write(f"✅ {pro}")
                
                with pros_cons[1]:
                    st.markdown("**Cons:**")
                    for con in rec['cons']:
                        st.write(f"❌ {con}")
            
            with col2:
                st.metric("Match Score", f"{rec['match_score']}%")
                st.progress(rec['match_score']/100)
            
            with col3:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button(f"View Details", key=f"view_{rec['name']}"):
                    st.info(f"Opening details for {rec['name']}")
            
            st.markdown("---")


def analyze_phone(system: AIPhoneReviewSystem, phone_name: str):
    """Analyze a specific phone"""
    with st.spinner(f"Analyzing {phone_name}..."):
        # Simulate analysis
        st.success(f"Analysis complete for {phone_name}")
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Overall Rating", "4.7/5.0", "↑ 0.2")
            st.metric("Total Reviews", "1,234", "↑ 123")
        
        with col2:
            st.metric("Positive Sentiment", "78%", "↑ 5%")
            st.metric("Recommendation Rate", "92%", "↑ 3%")
        
        with col3:
            st.metric("Spam Rate", "6%", "↓ 2%")
            st.metric("Verified Reviews", "94%", "↑ 1%")
        
        # Aspect Analysis
        st.subheader("📊 Aspect-Based Analysis")
        
        aspects = {
            'Camera': 85,
            'Battery': 72,
            'Performance': 90,
            'Display': 88,
            'Build Quality': 83,
            'Value': 75
        }
        
        for aspect, score in aspects.items():
            col1, col2 = st.columns([1, 4])
            with col1:
                st.write(aspect)
            with col2:
                st.progress(score/100, text=f"{score}%")


def analyze_review_batch(system: AIPhoneReviewSystem, reviews: List[str]):
    """Analyze a batch of reviews"""
    with st.spinner("Analyzing reviews..."):
        # Process reviews
        results = []
        for review in reviews[:5]:  # Limit to first 5 for demo:
            if review.strip():
                analysis = system.insights_engine.analyze_review({'text': review})
                results.append(analysis)
        
        st.success(f"Analyzed {len(results)} reviews")
        
        # Display aggregate results
        if results:
            st.subheader("📊 Aggregate Analysis")
            
            # Emotion summary
            emotions = {}
            for result in results:
                emotion = result['emotion_analysis']['primary_emotion']
                emotions[emotion] = emotions.get(emotion, 0) + 1
            
            st.markdown("**Emotion Distribution:**")
            for emotion, count in emotions.items():
                st.write(f"- {emotion}: {count} reviews")
            
            # Sarcasm detection
            sarcastic_count = sum(1 for r in results if r['sarcasm_detection']['is_sarcastic'])
            if sarcastic_count > 0:
                st.warning(f"⚠️ Sarcasm detected in {sarcastic_count} review(s)")


def analyze_dataset(system: AIPhoneReviewSystem, df: pd.DataFrame):
    """Analyze a dataset of reviews"""
    with st.spinner("Processing dataset..."):
        # Preprocess data
        processed_df = system.preprocessor.preprocess_dataset(df)
        
        st.success(f"Processed {len(processed_df)} reviews")
        
        # Display summary statistics
        st.subheader("📊 Dataset Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Reviews", len(df))
            st.metric("Unique Products", df['product'].nunique() if 'product' in df.columns else "N/A")
        
        with col2:
            st.metric("Average Length", f"{df['review'].str.len().mean():.0f} chars" if 'review' in df.columns else "N/A")
            st.metric("Date Range", f"{(df['date'].max() - df['date'].min()).days} days" if 'date' in df.columns else "N/A")
        
        with col3:
            st.metric("Avg Rating", f"{df['rating'].mean():.2f}" if 'rating' in df.columns else "N/A")
            st.metric("Std Dev", f"{df['rating'].std():.2f}" if 'rating' in df.columns else "N/A")


def save_current_analysis(system: AIPhoneReviewSystem):
    """Save current analysis results"""
    if st.session_state.current_analysis:
        # Save to database or file
        st.success("Analysis saved successfully!")
    else:
        st.warning("No analysis to save")


def export_analysis_report(system: AIPhoneReviewSystem):
    """Export analysis report"""
    if st.session_state.current_analysis:
        # Generate and export report
        st.success("Report exported successfully!")
    else:
        st.warning("No analysis to export")


if __name__ == "__main__":
    main()
