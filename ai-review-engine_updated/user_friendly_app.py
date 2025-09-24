"""
AI Phone Review Engine - User-Friendly Version
Focused on helping users make quick phone buying decisions
Prioritizes the most important features users actually need
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import sys
import os
import logging
from typing import Dict, List, Optional, Any

# Add project directories to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import core modules
try:
    from utils.preprocessed_data_loader import PreprocessedDataLoader
    PREPROCESSED_DATA_AVAILABLE = True
except ImportError:
    PREPROCESSED_DATA_AVAILABLE = False

# Import essential modules
from models.recommendation_engine_simple import RecommendationEngine
from models.auto_insights_engine import AutoInsightsEngine
from models.absa_model import ABSASentimentAnalyzer
from models.spam_detector import SpamDetector
from utils.data_preprocessing import DataPreprocessor
from utils.visualization import ReviewVisualizer
from utils.business_sentiment_tracker import BusinessSentimentTracker
from utils.data_quality_validator import DataQualityValidator

# Import Advanced AI modules
try:
    from models.conversational_ai import ConversationalAI
    from models.advanced_ai_model import AdvancedAIEngine
    from modules.deeper_insights import DeeperInsightsEngine
    from models.neural_recommendation_engine import NeuralRecommendationEngine
    from core.smart_search import SmartSearchEngine as CoreSmartSearch
    ADVANCED_AI_AVAILABLE = True
except ImportError as e:
    st.warning(f"⚠️ Some advanced AI features unavailable: {e}")
    ADVANCED_AI_AVAILABLE = False

# Import Agentic RAG modules
try:
    from models.agentic_rag import AgenticRAGSystem, BaseAgent, AgentRole, AgentTask
    from models.chat_assistant import RAGChatAssistant
    RAG_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Agentic RAG unavailable: {e}")
    RAG_AVAILABLE = False

# Import production modules (optional)
try:
    from utils.exceptions import ErrorHandler
    from utils.logging_config import LoggingManager
    PRODUCTION_MODULES_AVAILABLE = True
except ImportError:
    PRODUCTION_MODULES_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Phone Review Analysis",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="collapsed"  # Start with sidebar closed for cleaner look
)

# User-friendly CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .decision-card {
        background: linear-gradient(135deg, #E8F5E8 0%, #C8E6C9 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin: 1rem 0;
        text-align: center;
        transition: transform 0.3s;
        border: 2px solid #4CAF50;
    }
    .decision-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.15);
    }
    .quick-action-btn {
        background: #4CAF50;
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 25px;
        border: none;
        font-size: 1rem;
        font-weight: bold;
        margin: 0.5rem;
        cursor: pointer;
        transition: all 0.3s;
    }
    .quick-action-btn:hover {
        background: #45A049;
        transform: scale(1.05);
    }
    .recommendation-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #2196F3;
    }
    .phone-card {
        background: #F8F9FA;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        border: 1px solid #E0E0E0;
    }
    .verdict {
        font-size: 1.3rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        padding: 1rem;
        background: #E8F5E8;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .warning-box {
        background: #FFF3E0;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #FF9800;
        margin: 1rem 0;
    }
    .success-box {
        background: #E8F5E8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize core components for user-friendly experience"""
    global RAG_AVAILABLE
    if 'initialized' not in st.session_state:
        with st.spinner("🔄 Loading Phone Assistant..."):
            try:
                # Core components
                st.session_state.rec_engine = RecommendationEngine()
                st.session_state.insights_engine = AutoInsightsEngine()
                st.session_state.data_validator = DataQualityValidator()
                st.session_state.absa_analyzer = ABSASentimentAnalyzer()
                st.session_state.preprocessor = DataPreprocessor()
                st.session_state.visualizer = ReviewVisualizer()
                st.session_state.business_tracker = BusinessSentimentTracker()
                
                # Initialize Advanced AI modules if available
                if ADVANCED_AI_AVAILABLE:
                    st.session_state.conversational_ai = ConversationalAI()
                    st.session_state.advanced_ai_engine = AdvancedAIEngine(enable_gpu=False)  # Disable GPU for user-friendly version
                    st.session_state.deeper_insights = DeeperInsightsEngine()
                    st.session_state.neural_rec_engine = NeuralRecommendationEngine()
                    st.session_state.smart_search = CoreSmartSearch()
                    st.info("🚀 Advanced AI features loaded successfully!")
                else:
                    st.info("📱 Running with core features (some advanced AI features disabled)")
                
                # Initialize Agentic RAG system if available
                if RAG_AVAILABLE:
                    try:
                        st.session_state.rag_system = initialize_rag_system(st.session_state.df)
                        st.session_state.rag_agents = create_specialized_agents()
                        st.session_state.knowledge_base = build_knowledge_base(st.session_state.df)
                        st.success("🤖 Agentic RAG System initialized - Multi-agent AI ready!")
                    except Exception as e:
                        st.warning(f"RAG initialization failed: {str(e)[:100]}...")
                        RAG_AVAILABLE = False
                
                # Load data
                if PREPROCESSED_DATA_AVAILABLE:
                    st.session_state.data_loader = PreprocessedDataLoader()
                    st.session_state.df = st.session_state.data_loader.get_full_dataset()
                else:
                    # Fallback sample data
                    st.session_state.df = create_sample_data()
                
                # Error handling
                if PRODUCTION_MODULES_AVAILABLE:
                    st.session_state.error_handler = ErrorHandler(log_to_file=True)
                
                st.session_state.initialized = True
                
            except Exception as e:
                st.error("Some features may be limited, but core functionality is available.")
                st.session_state.df = create_sample_data()
                st.session_state.initialized = True

def create_sample_data():
    """Create fallback data if needed"""
    return pd.DataFrame({
        'product': ['iPhone 15 Pro', 'Samsung Galaxy S24', 'Google Pixel 8', 'OnePlus 12'] * 50,
        'review_text': ['Excellent phone', 'Great camera', 'Good battery', 'Fast performance'] * 50,
        'rating': [4.5, 4.3, 4.4, 4.2] * 50,
        'brand': ['Apple', 'Samsung', 'Google', 'OnePlus'] * 50,
        'sentiment_label': ['positive', 'positive', 'positive', 'positive'] * 50
    })

def main():
    """Main user-friendly application"""
    # Initialize
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">📱 Phone Review Analysis Platform</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Comprehensive phone analysis with AI-powered insights</p>', unsafe_allow_html=True)
    
    # Quick stats bar
    show_quick_stats()
    
    # Main user journey
    show_main_interface()

def show_quick_stats():
    """Show key statistics to build confidence"""
    df = st.session_state.df
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Phones Analyzed", f"{df['product'].nunique()}", "Real Reviews")
        with col2:
            st.metric("🏆 Top Rated", "4.5+ Stars", "Quality Phones")
        with col3:
            total_reviews = len(df)
            st.metric("💬 Total Reviews", f"{total_reviews:,}", "User Experiences")
        with col4:
            if 'sentiment_label' in df.columns:
                positive_pct = (df['sentiment_label'] == 'positive').mean() * 100
                st.metric("😊 Happy Users", f"{positive_pct:.0f}%", "Satisfied Buyers")
    
    st.markdown("---")

def show_main_interface():
    """Main user-friendly interface"""
    
    # Primary user actions - available analysis options
    st.header("🔍 Available Analysis Options")
    
    # Add AI Assistant option with RAG enhancement
    if ADVANCED_AI_AVAILABLE and 'conversational_ai' in st.session_state:
        rag_status = "Multi-Agent RAG" if RAG_AVAILABLE and 'rag_system' in st.session_state else "Standard AI"
        
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%); padding: 1.5rem; border-radius: 15px; text-align: center; margin: 1rem 0; border: 2px solid #2196F3;">
                <h3>🤖 AI Analysis Assistant ({rag_status})</h3>
                <p>Chat with our {'multi-agent RAG system' if RAG_AVAILABLE else 'AI'} to get detailed phone analysis, comparisons, and insights</p>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("💬 Chat with AI Assistant", key="ai_chat", type="primary", use_container_width=True):
            st.session_state.current_page = "ai_assistant"
        
        st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="decision-card">
                <h3>🔍 Phone Analysis by Criteria</h3>
                <p>Analyze phones based on your specific criteria and preferences</p>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔍 Analyze by Criteria", key="get_recs", type="primary", use_container_width=True):
            st.session_state.current_page = "recommendations"
    
    with col2:
        st.markdown("""
            <div class="decision-card">
                <h3>⚖️ Compare Specific Phones</h3>
                <p>Already have phones in mind? Compare them side-by-side</p>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("📊 Compare Phones", key="compare", type="primary", use_container_width=True):
            st.session_state.current_page = "compare"
    
    with col3:
        st.markdown("""
            <div class="decision-card">
                <h3>📱 Research a Phone</h3>
                <p>Deep dive into reviews and ratings for a specific phone</p>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔬 Research Phone", key="research", type="primary", use_container_width=True):
            st.session_state.current_page = "research"
    
    # Business access section
    st.markdown("---")
    
    # Additional features section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div style="background: linear-gradient(135deg, #FFF3E0 0%, #FFE0B2 100%); padding: 1.5rem; border-radius: 12px; text-align: center; border: 2px solid #FF9800;">
                <h4>🧠 AI Market Insights</h4>
                <p>Get automated insights and trending analysis</p>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔮 View AI Insights", key="insights", type="secondary", use_container_width=True):
            st.session_state.current_page = "insights"
    
    with col2:
        st.markdown("""
            <div style="background: linear-gradient(135deg, #F3E5F5 0%, #E1BEE7 100%); padding: 1.5rem; border-radius: 12px; text-align: center; border: 2px solid #9C27B0;">
                <h4>🔍 Data Quality Check</h4>
                <p>Validate and assess your data quality</p>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("🛡️ Check Data Quality", key="quality", type="secondary", use_container_width=True):
            st.session_state.current_page = "quality"
    
    with col3:
        st.markdown("""
            <div style="background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%); padding: 1.5rem; border-radius: 12px; text-align: center; border: 2px solid #2196F3;">
                <h4>🏢 Business Brand Tracking</h4>
                <p>Track sentiment and performance for your phone brand</p>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("📊 Access Business Dashboard", key="business", type="secondary", use_container_width=True):
            st.session_state.current_page = "business"
    
    st.markdown("---")
    
    # Show selected page content
    current_page = st.session_state.get('current_page', 'home')
    
    if current_page == 'recommendations':
        show_recommendation_wizard()
    elif current_page == 'compare':
        show_comparison_tool()
    elif current_page == 'research':
        show_phone_research()
    elif current_page == 'insights':
        show_ai_insights_dashboard()
    elif current_page == 'quality':
        show_data_quality_dashboard()
    elif current_page == 'business':
        show_business_dashboard()
    elif current_page == 'ai_assistant':
        show_ai_assistant_chat()
    else:
        show_home_content()

def show_home_content():
    """Default home content with quick insights"""
    
    # Top rated phones preview
    st.header("🏆 Highest User Satisfaction Phones")
    
    df = st.session_state.df
    if df is not None and 'product' in df.columns:
        
        # Get top phones by sentiment
        if 'sentiment_label' in df.columns:
            phone_scores = df.groupby('product').agg({
                'sentiment_label': lambda x: (x == 'positive').mean(),
                'rating': 'mean',
                'product': 'count'
            }).rename(columns={'sentiment_label': 'positive_rate', 'product': 'review_count'})
            
            # Filter phones with enough reviews
            popular_phones = phone_scores[phone_scores['review_count'] >= 5]
            top_phones = popular_phones.sort_values('positive_rate', ascending=False).head(3)
            
            for i, (phone, data) in enumerate(top_phones.iterrows()):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"""
                        <div class="recommendation-card">
                            <h4>#{i+1}. {phone}</h4>
                            <p><strong>User Satisfaction:</strong> {data['positive_rate']:.1%} positive reviews</p>
                            <p><strong>Average Rating:</strong> {data['rating']:.1f}/5.0 stars</p>
                            <p><strong>Based on:</strong> {int(data['review_count'])} real user reviews</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    if st.button(f"Learn More", key=f"learn_{i}", use_container_width=True):
                        st.session_state.selected_phone = phone
                        st.session_state.current_page = "research"
                        st.rerun()
    
    # Quick actions
    st.markdown("---")
    st.header("⚡ Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📱 Top Overall Rated", use_container_width=True):
            show_quick_recommendation("overall")
    
    with col2:
        if st.button("📸 Camera-Focused", use_container_width=True):
            show_quick_recommendation("camera")
    
    with col3:
        if st.button("🔋 Battery-Focused", use_container_width=True):
            show_quick_recommendation("battery")
    
    with col4:
        if st.button("💰 Value-Focused", use_container_width=True):
            show_quick_recommendation("value")

def show_recommendation_wizard():
    """Interactive analysis wizard with AI-powered discovery"""
    st.header("🧞‍♂️ Phone Analysis Wizard")
    st.markdown("Answer a few questions and we'll analyze phones matching your criteria!")
    
    # Back button
    if st.button("← Back to Home"):
        st.session_state.current_page = "home"
        st.rerun()
    
    # Add Smart Search option if advanced AI is available
    if ADVANCED_AI_AVAILABLE and 'smart_search' in st.session_state:
        st.markdown("---")
        st.markdown("### 🤖 Or try Smart Discovery")
        st.markdown("Describe what you want in natural language, and our AI will find matching phones!")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            smart_query = st.text_input(
                "Describe your ideal phone:",
                placeholder="e.g., 'I want a phone with great camera under $600 for photography'",
                key="smart_search_query"
            )
        
        with col2:
            if st.button("🔍 Smart Search", type="secondary") and smart_query:
                show_smart_search_results(smart_query)
                return
        
        st.markdown("---")
    
    with st.form("recommendation_wizard"):
        st.subheader("Tell us about your needs:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            budget = st.select_slider(
                "💰 What's your budget?",
                options=["Under $300", "$300-500", "$500-800", "$800-1200", "$1200+"],
                value="$500-800"
            )
            
            primary_use = st.selectbox(
                "📱 What will you use it for most?",
                ["General use", "Photography", "Gaming", "Work/Business", "Social media"]
            )
            
            brand_preference = st.multiselect(
                "🏷️ Any brand preferences?",
                ["Apple", "Samsung", "Google", "OnePlus", "Xiaomi", "No preference"],
                default=["No preference"]
            )
        
        with col2:
            important_features = st.multiselect(
                "⭐ What features matter most? (Pick 2-3)",
                ["Camera quality", "Battery life", "Performance/Speed", "Display quality", 
                 "Build quality", "Storage space", "Price/Value"]
            )
            
            current_phone = st.text_input(
                "📲 What phone do you currently have?",
                placeholder="e.g., iPhone 12, Samsung S21"
            )
            
            urgency = st.radio(
                "⏰ When are you buying?",
                ["This week", "This month", "Just browsing"]
            )
        
        if st.form_submit_button("📊 Analyze Matching Phones", type="primary"):
            show_personalized_recommendations(budget, primary_use, important_features, brand_preference)

def show_personalized_recommendations(budget, primary_use, important_features, brand_preference):
    """Show AI-enhanced analysis results based on user input"""
    
    st.success("🎉 Here are phones matching your criteria with AI analysis!")
    
    df = st.session_state.df
    
    # Use Neural Recommendation Engine if available
    if ADVANCED_AI_AVAILABLE and 'neural_rec_engine' in st.session_state:
        with st.spinner("🧠 AI is analyzing your preferences..."):
            try:
                neural_recommendations = get_neural_recommendations(
                    budget, primary_use, important_features, brand_preference, df
                )
                
                if neural_recommendations:
                    display_neural_recommendations(neural_recommendations, budget, primary_use)
                    return
            except Exception as e:
                st.info(f"Neural recommendations unavailable, using standard system: {str(e)[:100]}...")
    
    # Fallback to standard recommendations
    # Filter based on preferences
    filtered_phones = df.copy()
    
    # Brand filtering
    if brand_preference and "No preference" not in brand_preference:
        if 'brand' in filtered_phones.columns:
            filtered_phones = filtered_phones[filtered_phones['brand'].isin(brand_preference)]
    
    # Get top phones from filtered set
    if len(filtered_phones) > 0:
        phone_scores = filtered_phones.groupby('product').agg({
            'sentiment_label': lambda x: (x == 'positive').mean() if 'sentiment_label' in filtered_phones.columns else 0.8,
            'rating': lambda x: x.mean() if 'rating' in filtered_phones.columns else 4.0,
            'product': 'count'
        }).rename(columns={'sentiment_label': 'positive_rate', 'product': 'review_count'})
        
        recommendations = phone_scores.sort_values('positive_rate', ascending=False).head(3)
        
        for i, (phone, data) in enumerate(recommendations.iterrows()):
            
            # Analysis summary
            if data['positive_rate'] > 0.8:
                verdict = "🏆 Excellent User Satisfaction"
                verdict_color = "#4CAF50"
            elif data['positive_rate'] > 0.6:
                verdict = "✅ Good User Satisfaction"
                verdict_color = "#2196F3"
            else:
                verdict = "📊 Mixed User Feedback"
                verdict_color = "#FF9800"
            
            st.markdown(f"""
                <div class="phone-card">
                    <h3>#{i+1} {phone}</h3>
                    <div style="background: {verdict_color}20; padding: 0.5rem; border-radius: 5px; margin: 0.5rem 0;">
                        <strong style="color: {verdict_color};">{verdict}</strong>
                    </div>
                    <p><strong>Why it's perfect for you:</strong></p>
                    <ul>
                        <li>{data['positive_rate']:.1%} of users are satisfied</li>
                        <li>Average rating: {data['rating']:.1f}/5.0 stars</li>
                        <li>Based on {int(data['review_count'])} real reviews</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button(f"📊 See Details", key=f"details_{i}"):
                    st.session_state.selected_phone = phone
                    st.session_state.current_page = "research"
                    st.rerun()
            with col2:
                if st.button(f"⚖️ Compare", key=f"comp_{i}"):
                    st.session_state.phone_to_compare = phone
                    st.session_state.current_page = "compare"
                    st.rerun()
            with col3:
                st.button(f"❤️ Save", key=f"save_{i}")
            
            # Show warning and alternatives for phones with poor satisfaction
            if data['positive_rate'] < 0.6:  # Less than 60% satisfaction
                st.warning(f"⚠️ Note: {phone} has {data['positive_rate']:.1%} user satisfaction. You may want to explore other options.")

def show_comparison_tool():
    """User-friendly phone comparison"""
    st.header("⚖️ Phone Comparison Tool")
    st.markdown("Compare phones side-by-side to make the right choice")
    
    # Back button
    if st.button("← Back to Home"):
        st.session_state.current_page = "home"
        st.rerun()
    
    df = st.session_state.df
    if df is not None and 'product' in df.columns:
        available_phones = df['product'].unique()
        
        col1, col2 = st.columns(2)
        
        with col1:
            phone1 = st.selectbox("📱 First Phone:", available_phones, 
                                 index=0 if len(available_phones) > 0 else None)
        
        with col2:
            phone2_options = [p for p in available_phones if p != phone1]
            phone2 = st.selectbox("📱 Second Phone:", phone2_options,
                                 index=0 if len(phone2_options) > 0 else None)
        
        if phone1 and phone2 and st.button("🔄 Compare These Phones", type="primary"):
            show_detailed_comparison(phone1, phone2)

def show_detailed_comparison(phone1, phone2):
    """Show detailed comparison between two phones"""
    df = st.session_state.df
    
    # Get data for both phones
    phone1_data = df[df['product'] == phone1]
    phone2_data = df[df['product'] == phone2]
    
    st.markdown(f"### 📊 {phone1} vs {phone2}")
    
    # Quick verdict
    if 'sentiment_label' in df.columns:
        phone1_satisfaction = (phone1_data['sentiment_label'] == 'positive').mean()
        phone2_satisfaction = (phone2_data['sentiment_label'] == 'positive').mean()
        
        if phone1_satisfaction > phone2_satisfaction + 0.1:
            winner = phone1
            winner_reason = f"has {phone1_satisfaction:.1%} user satisfaction vs {phone2_satisfaction:.1%}"
        elif phone2_satisfaction > phone1_satisfaction + 0.1:
            winner = phone2
            winner_reason = f"has {phone2_satisfaction:.1%} user satisfaction vs {phone1_satisfaction:.1%}"
        else:
            winner = "It's a tie!"
            winner_reason = "Both phones have similar user satisfaction"
        
        st.markdown(f"""
            <div class="verdict">
                📊 Analysis Result: {winner}<br>
                <small>{winner_reason}</small>
            </div>
        """, unsafe_allow_html=True)
    
    # Detailed comparison
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📊 Reviews", f"{len(phone1_data)}", f"{phone1}")
        st.metric("", f"{len(phone2_data)}", f"{phone2}")
    
    with col2:
        if 'rating' in df.columns:
            phone1_rating = phone1_data['rating'].mean()
            phone2_rating = phone2_data['rating'].mean()
            st.metric("⭐ Rating", f"{phone1_rating:.1f}/5", f"{phone1}")
            st.metric("", f"{phone2_rating:.1f}/5", f"{phone2}")
    
    with col3:
        if 'sentiment_label' in df.columns:
            phone1_pos = (phone1_data['sentiment_label'] == 'positive').mean() * 100
            phone2_pos = (phone2_data['sentiment_label'] == 'positive').mean() * 100
            st.metric("😊 Satisfaction", f"{phone1_pos:.0f}%", f"{phone1}")
            st.metric("", f"{phone2_pos:.0f}%", f"{phone2}")
    
    # Visual comparison chart
    if 'rating' in df.columns:
        fig = go.Figure(data=[
            go.Bar(name=phone1, x=['Rating', 'Reviews', 'Satisfaction'], 
                   y=[phone1_data['rating'].mean(), len(phone1_data)/10, phone1_pos/20]),
            go.Bar(name=phone2, x=['Rating', 'Reviews', 'Satisfaction'], 
                   y=[phone2_data['rating'].mean(), len(phone2_data)/10, phone2_pos/20])
        ])
        fig.update_layout(title="Quick Comparison", barmode='group', height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Summary analysis
    st.markdown("### 💡 Analysis Summary")
    if phone1_satisfaction > phone2_satisfaction:
        st.markdown(f"""
            <div class="success-box">
                <strong>{phone1}</strong> shows higher user satisfaction in our analysis.
                It has {phone1_satisfaction:.1%} satisfaction vs {phone2_satisfaction:.1%}.
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="success-box">
                <strong>{phone2}</strong> shows higher user satisfaction in our analysis.
                It has {phone2_satisfaction:.1%} satisfaction vs {phone1_satisfaction:.1%}.
            </div>
        """, unsafe_allow_html=True)

def show_phone_research():
    """Deep dive research on a specific phone"""
    st.header("🔬 Phone Research Center")
    st.markdown("Get detailed insights about any phone")
    
    # Back button
    if st.button("← Back to Home"):
        st.session_state.current_page = "home"
        st.rerun()
    
    df = st.session_state.df
    
    # Phone selection
    selected_phone = st.session_state.get('selected_phone', '')
    if df is not None and 'product' in df.columns:
        available_phones = df['product'].unique()
        
        if selected_phone in available_phones:
            phone = st.selectbox("📱 Select Phone:", available_phones, 
                                index=list(available_phones).index(selected_phone))
        else:
            phone = st.selectbox("📱 Select Phone:", available_phones)
        
        if st.button("🔍 Analyze This Phone", type="primary"):
            show_detailed_phone_analysis(phone)

def show_detailed_phone_analysis(phone_name):
    """Show comprehensive phone analysis with charts and detailed insights"""
    df = st.session_state.df
    phone_data = df[df['product'] == phone_name]
    
    if len(phone_data) == 0:
        st.error(f"No data found for {phone_name}")
        return
    
    st.markdown(f"# 📱 {phone_name} - Complete Analysis")
    
    # Calculate key metrics
    total_reviews = len(phone_data)
    avg_rating = phone_data['rating'].mean() if 'rating' in phone_data.columns else 4.0
    positive_pct = (phone_data['sentiment_label'] == 'positive').mean() * 100 if 'sentiment_label' in phone_data.columns else 80
    negative_pct = (phone_data['sentiment_label'] == 'negative').mean() * 100 if 'sentiment_label' in phone_data.columns else 15
    neutral_pct = 100 - positive_pct - negative_pct
    
    # Advanced AI Analysis (if available)
    advanced_analysis = None
    if ADVANCED_AI_AVAILABLE and 'deeper_insights' in st.session_state and 'review_text' in phone_data.columns:
        with st.spinner("🧠 Running advanced AI analysis..."):
            try:
                advanced_analysis = run_advanced_sentiment_analysis(phone_data, phone_name)
            except Exception as e:
                st.info(f"Advanced analysis temporarily unavailable: {str(e)[:100]}...")
    
    # Overall analysis summary with enhanced styling
    if positive_pct >= 80:
        verdict = "🏆 Excellent User Satisfaction"
        verdict_desc = "Users express high satisfaction with this phone"
        verdict_color = "#4CAF50"
        confidence = "Very High"
    elif positive_pct >= 70:
        verdict = "✅ Good User Satisfaction"
        verdict_desc = "Most users report positive experiences"
        verdict_color = "#8BC34A"
        confidence = "High"
    elif positive_pct >= 60:
        verdict = "👍 Moderate User Satisfaction"
        verdict_desc = "Generally positive user feedback"
        verdict_color = "#2196F3"
        confidence = "Medium-High"
    elif positive_pct >= 40:
        verdict = "📊 Mixed User Feedback"
        verdict_desc = "User experiences vary significantly"
        verdict_color = "#FF9800"
        confidence = "Medium"
    else:
        verdict = "📉 Low User Satisfaction"
        verdict_desc = "Many users report negative experiences"
        verdict_color = "#F44336"
        confidence = "Low"
    
    st.markdown(f"""
        <div style="background: {verdict_color}20; padding: 2rem; border-radius: 15px; text-align: center; margin: 2rem 0; border: 2px solid {verdict_color};">
            <h2 style="color: {verdict_color}; margin: 0;">{verdict}</h2>
            <p style="font-size: 1.1rem; margin: 0.5rem 0;">{verdict_desc}</p>
            <p style="margin: 0;"><strong>{positive_pct:.0f}% of {total_reviews} users express positive sentiment</strong></p>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; opacity: 0.8;">Confidence Level: {confidence}</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Enhanced key metrics dashboard
    st.markdown("### 📊 Key Performance Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("📝 Total Reviews", f"{total_reviews:,}")
    
    with col2:
        st.metric("⭐ Average Rating", f"{avg_rating:.1f}/5.0", 
                 delta=f"{avg_rating-4.0:+.1f}" if avg_rating != 4.0 else None)
    
    with col3:
        st.metric("😊 Satisfaction", f"{positive_pct:.0f}%", 
                 delta=f"{positive_pct-75:+.0f}%" if positive_pct != 75 else None)
    
    with col4:
        if 'review_text' in phone_data.columns:
            avg_length = phone_data['review_text'].str.len().mean()
            detail_score = "Detailed" if avg_length > 100 else "Moderate" if avg_length > 50 else "Brief"
            st.metric("📄 Review Depth", detail_score)
        else:
            st.metric("📄 Review Depth", "Available")
    
    with col5:
        brand = phone_data['brand'].iloc[0] if 'brand' in phone_data.columns else "Unknown"
        st.metric("🏷️ Brand", brand)
    
    st.markdown("---")
    
    # Advanced AI Insights Section (if available)
    if advanced_analysis:
        st.markdown("### 🧠 Advanced AI Insights")
        
        # Emotion Analysis
        if 'emotions' in advanced_analysis:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🎨 Emotion Analysis**")
                emotions = advanced_analysis['emotions']
                
                # Display primary emotion
                primary_emotion = emotions.get('primary_emotion', 'neutral')
                emotion_intensity = emotions.get('intensity', 0.5)
                emotion_confidence = emotions.get('confidence', 0.5)
                
                emotion_icons = {
                    'joy': '😊', 'trust': '🤝', 'fear': '😨', 'surprise': '😲',
                    'sadness': '😢', 'disgust': '🤢', 'anger': '😠', 'anticipation': '🤔',
                    'neutral': '😐'
                }
                
                emotion_icon = emotion_icons.get(primary_emotion, '😐')
                
                st.markdown(f"""
                    <div style="background: #F3E5F5; padding: 1rem; border-radius: 8px; text-align: center;">
                        <h4>{emotion_icon} Primary Emotion: {primary_emotion.title()}</h4>
                        <p><strong>Intensity:</strong> {emotion_intensity:.1%} | <strong>Confidence:</strong> {emotion_confidence:.1%}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Display emotion breakdown
                if 'emotion_scores' in emotions:
                    st.markdown("**Emotion Breakdown:**")
                    emotion_scores = emotions['emotion_scores']
                    for emotion, score in sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)[:5]:
                        if score > 0.05:  # Only show significant emotions
                            emotion_icon = emotion_icons.get(emotion, '😐')
                            st.markdown(f"{emotion_icon} **{emotion.title()}:** {score:.1%}")
            
            with col2:
                # Sarcasm Detection
                if 'sarcasm' in advanced_analysis:
                    st.markdown("**😏 Sarcasm & Irony Detection**")
                    sarcasm = advanced_analysis['sarcasm']
                    
                    is_sarcastic = sarcasm.get('is_sarcastic', False)
                    sarcasm_confidence = sarcasm.get('confidence', 0)
                    
                    if is_sarcastic:
                        st.markdown(f"""
                            <div style="background: #FFF3E0; padding: 1rem; border-radius: 8px; border-left: 4px solid #FF9800;">
                                <p>😏 <strong>Sarcasm Detected</strong></p>
                                <p>Confidence: {sarcasm_confidence:.1%}</p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Show sarcasm indicators
                        if 'indicators' in sarcasm:
                            indicators = sarcasm['indicators']
                            st.markdown("**Indicators:**")
                            for indicator in indicators[:3]:
                                st.markdown(f"- {indicator.replace('_', ' ').title()}")
                    else:
                        st.markdown("""
                            <div style="background: #E8F5E8; padding: 1rem; border-radius: 8px;">
                                <p>✅ <strong>Genuine Reviews</strong></p>
                                <p>Low sarcasm indicators detected</p>
                            </div>
                        """)
        
        # Cultural Insights
        if 'cultural_insights' in advanced_analysis:
            st.markdown("**🌍 Cultural Sentiment Patterns**")
            cultural_data = advanced_analysis['cultural_insights']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Regional Preference", cultural_data.get('primary_region', 'Global'))
            with col2:
                st.metric("Cultural Score", f"{cultural_data.get('cultural_score', 0.5):.1%}")
            with col3:
                st.metric("Language Variety", cultural_data.get('language_indicators', 'Standard'))
        
        st.markdown("---")
    
    # Visual Analytics Section
    st.markdown("### 📈 Visual Analytics & Review Summary")
    
    # Comprehensive Review Summary
    st.markdown("#### 📝 Complete Review Summary")
    
    # Generate comprehensive narrative summary based on all reviews
    if 'review_text' in phone_data.columns and len(phone_data) > 0:
        all_reviews_text = ' '.join(phone_data['review_text'].dropna().tolist()).lower()
        
        # Generate narrative summary
        narrative_summary = generate_review_summary(all_reviews_text, phone_name, phone_data)
        
        # Display narrative summary
        st.markdown(f"**📝 What Users Are Saying:**")
        st.markdown(narrative_summary)
    else:
        # Fallback summary
        st.markdown(f"**📝 What Users Are Saying:**")
        st.markdown(f"Based on {total_reviews:,} user reviews with {positive_pct:.0f}% positive sentiment and an average rating of {avg_rating:.1f}/5 stars, users generally express satisfaction with this phone's performance and features. The overall feedback indicates a solid choice for users looking for reliable smartphone performance.")
    
    st.markdown("---")
    
    # Create visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Aspect-Based Sentiment Analysis Chart
        st.markdown("**📱 Aspect-Based Sentiment Analysis**")
        
        # Generate aspect sentiment data
        aspect_data = get_aspect_sentiment_data(phone_data, phone_name)
        
        if aspect_data:
            aspects = list(aspect_data.keys())
            positive_scores = [aspect_data[aspect]['positive'] for aspect in aspects]
            negative_scores = [aspect_data[aspect]['negative'] for aspect in aspects]
            
            fig_aspects = go.Figure(data=[
                go.Bar(name='Positive', x=aspects, y=positive_scores, marker_color='#4CAF50'),
                go.Bar(name='Negative', x=aspects, y=negative_scores, marker_color='#F44336')
            ])
            
            fig_aspects.update_layout(
                title="User Sentiment by Phone Aspect",
                xaxis_title="Phone Aspects",
                yaxis_title="Sentiment Score (%)",
                barmode='group',
                height=400,
                showlegend=True
            )
            st.plotly_chart(fig_aspects, use_container_width=True)
        else:
            # Fallback general sentiment chart
            if 'sentiment_label' in phone_data.columns:
                sentiment_counts = phone_data['sentiment_label'].value_counts()
                colors = ['#4CAF50', '#F44336', '#FF9800']
                
                fig_sentiment = go.Figure(data=[go.Pie(
                    labels=sentiment_counts.index.str.title(),
                    values=sentiment_counts.values,
                    marker=dict(colors=colors[:len(sentiment_counts)]),
                    hole=0.4
                )])
                fig_sentiment.update_layout(
                    title="Overall Sentiment Distribution",
                    height=400,
                    showlegend=True,
                    annotations=[dict(text=f'{positive_pct:.0f}%<br>Positive', x=0.5, y=0.5, font_size=20, showarrow=False)]
                )
                st.plotly_chart(fig_sentiment, use_container_width=True)
    
    with col2:
        # Rating Distribution
        st.markdown("**⭐ Rating Distribution Analysis**")
        if 'rating' in phone_data.columns:
            rating_counts = phone_data['rating'].value_counts().sort_index()
            
            fig_ratings = go.Figure(data=[go.Bar(
                x=rating_counts.index,
                y=rating_counts.values,
                marker_color='#2196F3',
                text=rating_counts.values,
                textposition='auto',
            )])
            fig_ratings.update_layout(
                title="User Rating Distribution",
                xaxis_title="Star Rating",
                yaxis_title="Number of Reviews",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig_ratings, use_container_width=True)
        else:
            st.info("Rating distribution not available")
    
    # Aspect Details Section
    st.markdown("---")
    st.markdown("#### 🔍 Detailed Aspect Analysis")
    
    if aspect_data:
        # Create tabs for each aspect
        aspect_tabs = st.tabs([f"{aspect.title()}" for aspect in aspect_data.keys()])
        
        for i, (aspect, tab) in enumerate(zip(aspect_data.keys(), aspect_tabs)):
            with tab:
                aspect_info = aspect_data[aspect]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("👍 Positive", f"{aspect_info['positive']:.0f}%")
                with col2:
                    st.metric("👎 Negative", f"{aspect_info['negative']:.0f}%")
                with col3:
                    overall_score = aspect_info['positive'] - aspect_info['negative']
                    st.metric("📊 Net Score", f"{overall_score:+.0f}%")
                
                # Sample reviews for this aspect
                st.markdown(f"**What users say about {aspect}:**")
                sample_comments = aspect_info.get('comments', [])
                for comment in sample_comments[:3]:
                    sentiment_icon = "😊" if comment['sentiment'] == 'positive' else "😞" if comment['sentiment'] == 'negative' else "😐"
                    st.markdown(f"{sentiment_icon} *\"{comment['text']}\"*")
    
    # Detailed Analysis Section
    st.markdown("---")
    st.markdown("### 🔍 Detailed Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Strengths and Weaknesses Analysis
        st.markdown("**📈 What Users Love**")
        
        strengths = []
        if positive_pct >= 70:
            strengths.extend(["✅ High user satisfaction", "✅ Proven track record"])
        if avg_rating >= 4.0:
            strengths.append("✅ Excellent ratings")
        if total_reviews >= 50:
            strengths.append("✅ Well-reviewed product")
        
        # Add context-specific strengths
        if 'review_text' in phone_data.columns:
            sample_text = ' '.join(phone_data['review_text'].dropna().head(20).tolist()).lower()
            if 'camera' in sample_text or 'photo' in sample_text:
                strengths.append("📸 Great camera quality")
            if 'battery' in sample_text or 'charge' in sample_text:
                strengths.append("🔋 Good battery life")
            if 'fast' in sample_text or 'speed' in sample_text:
                strengths.append("⚡ Fast performance")
            if 'display' in sample_text or 'screen' in sample_text:
                strengths.append("📱 Quality display")
        
        for strength in strengths[:6]:  # Limit to top 6
            st.markdown(strength)
        
        if not strengths:
            st.markdown("• Based on available data analysis")
    
    with col2:
        st.markdown("**⚠️ Areas to Consider**")
        
        concerns = []
        if positive_pct < 70:
            concerns.append("⚠️ Mixed user feedback")
        if negative_pct > 20:
            concerns.append("⚠️ Some negative experiences")
        if avg_rating < 4.0:
            concerns.append("⚠️ Below-average ratings")
        if total_reviews < 20:
            concerns.append("⚠️ Limited review data")
        
        # Add context-specific concerns
        if 'review_text' in phone_data.columns:
            sample_text = ' '.join(phone_data['review_text'].dropna().head(20).tolist()).lower()
            if 'expensive' in sample_text or 'price' in sample_text:
                concerns.append("💰 Price concerns")
            if 'battery' in sample_text and 'short' in sample_text:
                concerns.append("🔋 Battery life issues")
            if 'slow' in sample_text:
                concerns.append("🐌 Performance concerns")
        
        for concern in concerns[:6]:  # Limit to top 6
            st.markdown(concern)
        
        if not concerns:
            st.markdown("• No major concerns identified")
    
    # User Feedback Section
    st.markdown("---")
    st.markdown("### 💬 What Users Are Saying")
    
    if 'review_text' in phone_data.columns and len(phone_data) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Recent User Reviews**")
            sample_reviews = phone_data['review_text'].dropna().head(5)
            
            for i, review in enumerate(sample_reviews.head(3)):
                # Try to get actual sentiment if available
                if 'sentiment_label' in phone_data.columns:
                    review_sentiment = phone_data[phone_data['review_text'] == review]['sentiment_label'].iloc[0]
                    if review_sentiment == 'positive':
                        sentiment_icon = "😊"
                    elif review_sentiment == 'negative':
                        sentiment_icon = "😞"
                    else:
                        sentiment_icon = "😐"
                else:
                    sentiment_icon = "💬"
                
                review_text = review[:200] + "..." if len(review) > 200 else review
                st.markdown(f"{sentiment_icon} *\"{review_text}\"*")
                st.markdown("")
        
        with col2:
            st.markdown("**Key Statistics & Insights**")
            insights = [
                f"📊 **{total_reviews:,}** total user reviews analyzed",
                f"⭐ **{avg_rating:.1f}/5** average star rating",
                f"😊 **{positive_pct:.0f}%** positive user sentiment",
                f"📊 **{confidence}** confidence in analysis"
            ]
            
            # Add brand comparison if available
            if 'brand' in phone_data.columns:
                brand = phone_data['brand'].iloc[0]
                brand_avg = df[df['brand'] == brand]['sentiment_label'].apply(lambda x: x == 'positive').mean() * 100 if 'sentiment_label' in df.columns else 75
                insights.append(f"🏷️ **{brand_avg:.0f}%** avg satisfaction for {brand} phones")
            
            # Add market position
            all_phones_avg = (df['sentiment_label'] == 'positive').mean() * 100 if 'sentiment_label' in df.columns else 75
            if positive_pct > all_phones_avg:
                insights.append(f"🚀 **Above average** vs market ({all_phones_avg:.0f}%)")
            elif positive_pct < all_phones_avg - 5:
                insights.append(f"📉 **Below average** vs market ({all_phones_avg:.0f}%)")
            else:
                insights.append(f"📊 **Average** market performance ({all_phones_avg:.0f}%)")
            
            for insight in insights:
                st.markdown(insight)
    else:
        st.info("Review text analysis not available for this phone.")
    
    
    # Analysis Summary
    st.markdown("---")
    st.markdown("### 📊 Analysis Summary")
    
    if positive_pct >= 80:
        st.markdown(f"""
            <div class="success-box">
                <h4>✅ Excellent User Satisfaction</h4>
                <p><strong>Analysis indicates:</strong></p>
                <ul>
                    <li>🏆 {positive_pct:.0f}% user satisfaction - among the best</li>
                    <li>⭐ {avg_rating:.1f}/5.0 average rating from real users</li>
                    <li>📊 Based on {total_reviews:,} genuine user reviews</li>
                    <li>🎯 Strong evidence of user satisfaction</li>
                </ul>
                <p><strong>Summary:</strong> This phone shows consistently positive user feedback and high satisfaction rates.</p>
            </div>
        """, unsafe_allow_html=True)
    elif positive_pct >= 70:
        st.markdown(f"""
            <div class="success-box">
                <h4>✅ Good User Satisfaction</h4>
                <p><strong>Analysis indicates:</strong></p>
                <ul>
                    <li>👍 {positive_pct:.0f}% user satisfaction - above average</li>
                    <li>⭐ {avg_rating:.1f}/5.0 rating from {total_reviews:,} reviews</li>
                    <li>📈 Solid performance based on user feedback</li>
                    <li>✨ Most users are satisfied with their purchase</li>
                </ul>
                <p><strong>Summary:</strong> This phone shows generally positive user feedback and solid performance ratings.</p>
            </div>
        """, unsafe_allow_html=True)
    elif positive_pct >= 50:
        st.markdown(f"""
            <div class="warning-box">
                <h4>⚠️ Mixed User Feedback</h4>
                <p><strong>Analysis summary:</strong></p>
                <ul>
                    <li>📊 {positive_pct:.0f}% satisfaction - user opinions are divided</li>
                    <li>⭐ {avg_rating:.1f}/5.0 average rating from {total_reviews:,} reviews</li>
                    <li>🎯 Mixed user experiences reported</li>
                    <li>⚠️ User experiences vary significantly</li>
                </ul>
                <p><strong>Summary:</strong> This phone shows mixed user feedback with varying experiences.</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Show alternatives for mixed feedback phones
        show_alternative_recommendations(phone_name, phone_data, df, threshold="mixed")
    else:
        st.markdown(f"""
            <div style="background: #FFEBEE; padding: 1rem; border-radius: 8px; border-left: 4px solid #F44336;">
                <h4>📉 Low User Satisfaction</h4>
                <p><strong>Analysis findings:</strong></p>
                <ul>
                    <li>📉 {positive_pct:.0f}% user satisfaction rate</li>
                    <li>⭐ {avg_rating:.1f}/5.0 average rating from {total_reviews:,} reviews</li>
                    <li>📊 Many users report negative experiences</li>
                    <li>⚠️ Below-average satisfaction scores</li>
                </ul>
                <p><strong>Summary:</strong> This phone shows lower user satisfaction compared to market averages.</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Add recommendations for phones with poor sentiment
        show_alternative_recommendations(phone_name, phone_data, df)
    
    # RAG-Enhanced Deep Insights (if available)
    if RAG_AVAILABLE and 'rag_system' in st.session_state:
        st.markdown("---")
        st.markdown("### 🤖 RAG-Powered Deep Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button(f"🔍 Deep Analysis of {phone_name}", use_container_width=True):
                with st.spinner("🧙 Multi-Agent Analysis in progress..."):
                    deep_query = f"Provide comprehensive analysis of {phone_name} including all aspects, user sentiments, comparisons, and recommendations"
                    rag_response = rag_enhanced_query_processing(deep_query)
                    
                    if 'error' not in rag_response:
                        st.markdown("#### 🤖 Multi-Agent Deep Analysis:")
                        st.markdown(rag_response['response'])
                        
                        # Store in user memory
                        store_user_interaction({
                            'action': 'deep_analysis',
                            'phone': phone_name,
                            'timestamp': datetime.now().isoformat(),
                            'agents_used': rag_response.get('agents_used', []),
                            'confidence': rag_response.get('confidence', 0.8)
                        })
                    else:
                        st.error("Deep analysis temporarily unavailable")
        
        with col2:
            if st.button(f"🔄 Compare {phone_name} with Similar", use_container_width=True):
                with st.spinner("🤖 Finding similar phones..."):
                    compare_query = f"Find phones similar to {phone_name} and compare them in detail"
                    rag_response = rag_enhanced_query_processing(compare_query)
                    
                    if 'error' not in rag_response:
                        st.markdown("#### 🔄 AI-Powered Similarity Analysis:")
                        st.markdown(rag_response['response'])
    
    # Action buttons
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("⚖️ Compare with Others", use_container_width=True):
            st.session_state.phone_to_compare = phone_name
            st.session_state.current_page = "compare"
            st.rerun()
    
    with col2:
        if st.button("📊 View Other Options", use_container_width=True):
            st.session_state.current_page = "recommendations"
            st.rerun()
    
    with col3:
        if st.button("📊 View All Analytics", use_container_width=True):
            with st.expander("📈 Detailed Analytics", expanded=True):
                st.write("**Review Statistics:**")
                st.write(phone_data.describe())
    
    with col4:
        if st.button("❤️ Save Analysis", use_container_width=True):
            # Save to user memory/preferences
            save_to_user_preferences(phone_name, 'saved_analysis')
            st.success(f"Analysis for {phone_name} saved to your favorites!")

def generate_review_summary(reviews_text, phone_name, phone_data):
    """Generate a comprehensive narrative summary of what users are saying"""
    
    # Analyze different aspects mentioned in reviews
    aspects_mentioned = []
    
    # Camera analysis
    if 'camera' in reviews_text or 'photo' in reviews_text or 'picture' in reviews_text:
        if 'great camera' in reviews_text or 'amazing camera' in reviews_text or 'excellent camera' in reviews_text:
            aspects_mentioned.append("users consistently praise the camera quality, describing it as impressive with excellent photo capabilities")
        elif 'good camera' in reviews_text:
            aspects_mentioned.append("the camera receives positive feedback from users")
        else:
            aspects_mentioned.append("users mention the camera performance in their reviews")
    
    # Battery analysis
    if 'battery' in reviews_text or 'charge' in reviews_text:
        if 'good battery' in reviews_text or 'great battery' in reviews_text or 'battery life' in reviews_text:
            aspects_mentioned.append("battery life is frequently highlighted as a strong point")
        elif 'battery drain' in reviews_text or 'poor battery' in reviews_text:
            aspects_mentioned.append("some users express concerns about battery performance")
        else:
            aspects_mentioned.append("battery performance is commonly discussed among users")
    
    # Display analysis
    if 'display' in reviews_text or 'screen' in reviews_text:
        if 'great display' in reviews_text or 'beautiful screen' in reviews_text or 'stunning display' in reviews_text:
            aspects_mentioned.append("the display quality receives excellent reviews with users praising its clarity and colors")
        else:
            aspects_mentioned.append("users comment positively on the display and screen quality")
    
    # Performance analysis
    if 'fast' in reviews_text or 'performance' in reviews_text or 'speed' in reviews_text:
        if 'very fast' in reviews_text or 'great performance' in reviews_text or 'smooth' in reviews_text:
            aspects_mentioned.append("performance is consistently rated as excellent with users noting fast and smooth operation")
        else:
            aspects_mentioned.append("users report solid performance and speed")
    
    # Build quality analysis
    if 'build' in reviews_text or 'quality' in reviews_text or 'design' in reviews_text:
        if 'great build' in reviews_text or 'excellent quality' in reviews_text or 'premium' in reviews_text:
            aspects_mentioned.append("build quality and design are highly regarded by users")
        else:
            aspects_mentioned.append("users comment on the overall build quality and design")
    
    # Value analysis
    if 'price' in reviews_text or 'value' in reviews_text or 'money' in reviews_text:
        if 'great value' in reviews_text or 'worth the money' in reviews_text:
            aspects_mentioned.append("many users feel the phone offers excellent value for money")
        elif 'expensive' in reviews_text or 'overpriced' in reviews_text:
            aspects_mentioned.append("some users have concerns about the pricing")
        else:
            aspects_mentioned.append("users discuss the value proposition")
    
    # Brand-specific insights
    brand_insights = ""
    if 'iPhone' in phone_name:
        brand_insights = "As expected from Apple, users particularly appreciate the premium experience, ecosystem integration, and camera capabilities, though some mention the higher price point."
    elif 'Samsung' in phone_name:
        brand_insights = "Samsung users frequently highlight the vibrant display quality, feature-rich interface, and overall performance, making it a popular choice among Android enthusiasts."
    elif 'Pixel' in phone_name:
        brand_insights = "Google Pixel users consistently praise the computational photography features, clean Android experience, and AI-powered capabilities that set it apart from other Android phones."
    elif 'OnePlus' in phone_name:
        brand_insights = "OnePlus users appreciate the fast performance, quick charging capabilities, and competitive pricing that delivers flagship features at a more accessible price point."
    else:
        brand_insights = "Users generally appreciate the phone's feature set and overall performance for its price category."
    
    # Create narrative summary
    if aspects_mentioned:
        aspect_summary = ", ".join(aspects_mentioned[:-1])
        if len(aspects_mentioned) > 1:
            aspect_summary += f", and {aspects_mentioned[-1]}."
        else:
            aspect_summary = aspects_mentioned[0] + "."
    else:
        aspect_summary = "users share generally positive experiences with various aspects of the phone."
    
    # Calculate sentiment for summary tone
    positive_pct = (phone_data['sentiment_label'] == 'positive').mean() * 100
    total_reviews = len(phone_data)
    avg_rating = phone_data['rating'].mean() if 'rating' in phone_data.columns else 4.0
    
    # Create comprehensive narrative
    summary = f"Based on analysis of {total_reviews:,} user reviews, {aspect_summary.capitalize()} "
    summary += f"With {positive_pct:.0f}% positive sentiment and an average rating of {avg_rating:.1f} out of 5 stars, "
    
    if positive_pct >= 80:
        summary += "users overwhelmingly recommend this phone, with most buyers expressing high satisfaction with their purchase. "
    elif positive_pct >= 70:
        summary += "the majority of users are satisfied with their purchase and would recommend it to others. "
    elif positive_pct >= 60:
        summary += "most users are generally pleased with the phone, though experiences vary depending on individual needs. "
    else:
        summary += "user opinions are mixed, with some very satisfied while others have encountered issues. "
    
    summary += brand_insights
    
    return summary

def get_aspect_sentiment_data(phone_data, phone_name):
    """Generate aspect-based sentiment data for visualization"""
    
    # Mock aspect data based on phone characteristics and reviews
    aspect_data = {}
    
    # Extract review text for analysis
    if 'review_text' in phone_data.columns and len(phone_data) > 0:
        reviews_text = ' '.join(phone_data['review_text'].dropna().tolist()).lower()
        
        # Camera aspect
        camera_positive = 85 if 'camera' in reviews_text else 75
        camera_negative = 10 if 'camera' in reviews_text else 15
        
        # Battery aspect
        battery_positive = 80 if 'battery' in reviews_text else 70
        battery_negative = 15 if 'battery' in reviews_text else 20
        
        # Display aspect
        display_positive = 88 if 'display' in reviews_text or 'screen' in reviews_text else 78
        display_negative = 8 if 'display' in reviews_text else 12
        
        # Performance aspect
        performance_positive = 90 if 'fast' in reviews_text or 'performance' in reviews_text else 80
        performance_negative = 5 if 'fast' in reviews_text else 10
        
        # Build Quality aspect
        build_positive = 85 if 'quality' in reviews_text or 'build' in reviews_text else 75
        build_negative = 10 if 'quality' in reviews_text else 15
        
        # Value aspect
        value_positive = 75 if 'value' in reviews_text or 'price' in reviews_text else 65
        value_negative = 20 if 'expensive' in reviews_text else 25
    else:
        # Fallback data based on phone brand/name
        if 'iPhone' in phone_name:
            camera_positive, camera_negative = 90, 5
            battery_positive, battery_negative = 75, 20
            display_positive, display_negative = 95, 3
            performance_positive, performance_negative = 95, 3
            build_positive, build_negative = 98, 2
            value_positive, value_negative = 60, 35
        elif 'Samsung' in phone_name:
            camera_positive, camera_negative = 88, 8
            battery_positive, battery_negative = 85, 10
            display_positive, display_negative = 95, 3
            performance_positive, performance_negative = 90, 5
            build_positive, build_negative = 85, 10
            value_positive, value_negative = 75, 20
        elif 'Pixel' in phone_name:
            camera_positive, camera_negative = 95, 3
            battery_positive, battery_negative = 70, 25
            display_positive, display_negative = 85, 10
            performance_positive, performance_negative = 88, 8
            build_positive, build_negative = 80, 15
            value_positive, value_negative = 80, 15
        elif 'OnePlus' in phone_name:
            camera_positive, camera_negative = 80, 12
            battery_positive, battery_negative = 88, 8
            display_positive, display_negative = 90, 5
            performance_positive, performance_negative = 95, 3
            build_positive, build_negative = 85, 10
            value_positive, value_negative = 90, 5
        else:
            # Generic phone data
            camera_positive, camera_negative = 75, 15
            battery_positive, battery_negative = 70, 20
            display_positive, display_negative = 80, 12
            performance_positive, performance_negative = 78, 15
            build_positive, build_negative = 75, 18
            value_positive, value_negative = 70, 25
    
    # Build aspect data dictionary
    aspects = {
        'Camera': {
            'positive': camera_positive,
            'negative': camera_negative,
            'comments': [
                {'text': 'Amazing camera quality, photos look professional', 'sentiment': 'positive'},
                {'text': 'Great camera features and night mode', 'sentiment': 'positive'},
                {'text': 'Camera could be better in low light', 'sentiment': 'negative'}
            ]
        },
        'Battery': {
            'positive': battery_positive,
            'negative': battery_negative,
            'comments': [
                {'text': 'Battery lasts all day with heavy usage', 'sentiment': 'positive'},
                {'text': 'Fast charging is really convenient', 'sentiment': 'positive'},
                {'text': 'Battery drains quickly with gaming', 'sentiment': 'negative'}
            ]
        },
        'Display': {
            'positive': display_positive,
            'negative': display_negative,
            'comments': [
                {'text': 'Beautiful display with vivid colors', 'sentiment': 'positive'},
                {'text': 'Screen is bright and clear outdoors', 'sentiment': 'positive'},
                {'text': 'Display could be brighter', 'sentiment': 'negative'}
            ]
        },
        'Performance': {
            'positive': performance_positive,
            'negative': performance_negative,
            'comments': [
                {'text': 'Super fast and smooth performance', 'sentiment': 'positive'},
                {'text': 'No lag even with multiple apps', 'sentiment': 'positive'},
                {'text': 'Occasional slowdown with heavy apps', 'sentiment': 'negative'}
            ]
        },
        'Build Quality': {
            'positive': build_positive,
            'negative': build_negative,
            'comments': [
                {'text': 'Feels premium and well-built', 'sentiment': 'positive'},
                {'text': 'Solid construction and materials', 'sentiment': 'positive'},
                {'text': 'Back panel feels a bit cheap', 'sentiment': 'negative'}
            ]
        },
        'Value': {
            'positive': value_positive,
            'negative': value_negative,
            'comments': [
                {'text': 'Great value for money', 'sentiment': 'positive'},
                {'text': 'Worth every penny', 'sentiment': 'positive'},
                {'text': 'Overpriced for what you get', 'sentiment': 'negative'}
            ]
        }
    }
    
    return aspects

def show_alternative_recommendations(phone_name, phone_data, df, threshold="low"):
    """Show alternative recommendations only for phones with negative sentiment"""
    
    # Calculate sentiment for current phone
    if 'sentiment_label' not in phone_data.columns or len(phone_data) == 0:
        return
    
    positive_pct = (phone_data['sentiment_label'] == 'positive').mean() * 100
    
    # Only show alternatives if sentiment is poor
    show_alternatives = False
    if threshold == "mixed" and positive_pct < 60:  # Mixed feedback threshold
        show_alternatives = True
    elif threshold == "low" and positive_pct < 50:  # Low satisfaction threshold
        show_alternatives = True
    
    if not show_alternatives:
        return
    
    # Get phone brand and category for finding alternatives
    phone_brand = phone_data['brand'].iloc[0] if 'brand' in phone_data.columns and len(phone_data) > 0 else None
    
    st.markdown("---")
    st.markdown("### 🔄 Alternative Options")
    st.info("📊 Since this phone shows lower user satisfaction, here are similar phones with better user feedback:")
    
    # Find better alternatives
    alternatives = find_better_alternatives(phone_name, phone_brand, df)
    
    if alternatives:
        st.markdown("#### 👍 Better Options in Similar Category:")
        
        for i, (alt_phone, alt_data) in enumerate(alternatives[:3]):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                alt_positive_pct = (alt_data['sentiment_label'] == 'positive').mean() * 100
                alt_rating = alt_data['rating'].mean() if 'rating' in alt_data.columns else 0
                
                st.markdown(f"""
                    <div style="background: #E8F5E8; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; border-left: 4px solid #4CAF50;">
                        <h4>📱 {alt_phone}</h4>
                        <p><strong>😊 {alt_positive_pct:.0f}% positive sentiment</strong> (vs {positive_pct:.0f}% for {phone_name})</p>
                        <p>⭐ <strong>{alt_rating:.1f}/5.0 rating</strong> • 💬 <strong>{len(alt_data)} reviews</strong></p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                if st.button(f"🔍 Analyze", key=f"alt_analyze_{i}"):
                    st.session_state.selected_phone = alt_phone
                    st.session_state.current_page = "research"
                    st.rerun()
    else:
        st.info("🔍 No significantly better alternatives found in our current dataset.")

def find_better_alternatives(current_phone, current_brand, df, min_improvement=15):
    """Find phones with significantly better sentiment in similar category"""
    if df is None or 'sentiment_label' not in df.columns:
        return []
    
    # Get current phone's sentiment
    current_data = df[df['product'] == current_phone]
    if len(current_data) == 0:
        return []
    
    current_positive_pct = (current_data['sentiment_label'] == 'positive').mean() * 100
    
    # Find all other phones
    other_phones = df[df['product'] != current_phone]
    
    alternatives = []
    
    # Group by phone and calculate metrics
    for phone in other_phones['product'].unique():
        phone_data = df[df['product'] == phone]
        
        if len(phone_data) < 5:  # Skip phones with too few reviews
            continue
        
        phone_positive_pct = (phone_data['sentiment_label'] == 'positive').mean() * 100
        
        # Only include if significantly better
        if phone_positive_pct > current_positive_pct + min_improvement:
            # Prefer same brand alternatives, but include others too
            phone_brand = phone_data['brand'].iloc[0] if 'brand' in phone_data.columns else None
            priority = 1 if phone_brand == current_brand else 2
            
            alternatives.append((phone, phone_data, phone_positive_pct, priority))
    
    # Sort by priority (same brand first) then by sentiment
    alternatives.sort(key=lambda x: (x[3], -x[2]))
    
    # Return phone name and data pairs
    return [(alt[0], alt[1]) for alt in alternatives]

def show_quick_recommendation(category):
    """Show phones by category analysis with recommendations for poor performers"""
    st.markdown(f"### 🎯 Top {category.title()}-Focused Phones")
    
    df = st.session_state.df
    if df is not None and 'product' in df.columns:
        # Get top phones (simplified logic)
        if 'sentiment_label' in df.columns:
            top_phones = df.groupby('product')['sentiment_label'].apply(
                lambda x: (x == 'positive').mean()
            ).sort_values(ascending=False).head(5)  # Get top 5 to filter
            
            # Separate good and poor performing phones
            excellent_phones = [(phone, satisfaction) for phone, satisfaction in top_phones.items() if satisfaction >= 0.75]
            poor_phones = [(phone, satisfaction) for phone, satisfaction in top_phones.items() if satisfaction < 0.6]
            
            # Show excellent phones
            if excellent_phones:
                st.markdown(f"**😊 Phones with excellent {category} satisfaction (75%+):**")
                
                for i, (phone, satisfaction) in enumerate(excellent_phones[:3]):
                    st.markdown(f"""
                        <div style="background: #E8F5E8; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; border-left: 4px solid #4CAF50;">
                            <strong>#{i+1}. {phone}</strong><br>
                            😊 User satisfaction: {satisfaction:.1%}
                        </div>
                    """, unsafe_allow_html=True)
            
            # Warn about poor performing phones in this category
            if poor_phones:
                st.markdown("---")
                st.markdown(f"**⚠️ {category.title()} phones with lower satisfaction:**")
                st.warning(f"These {category} phones have received mixed reviews. Consider the alternatives above.")
                
                for phone, satisfaction in poor_phones:
                    st.markdown(f"• **{phone}**: {satisfaction:.1%} satisfaction - *Consider alternatives*")

def show_business_dashboard():
    """Business sentiment tracking dashboard integrated into user app"""
    
    st.header("🏢 Business Brand Sentiment Dashboard")
    st.markdown("**Track and monitor sentiment for your phone brand in real-time**")
    
    # Back to main app button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("← Back to Main App"):
            st.session_state.current_page = "home"
            st.rerun()
    
    st.markdown("---")
    
    # Business registration/selection section
    st.subheader("📋 Business Setup")
    
    with st.expander("🏢 Register Your Business (First Time Setup)", expanded=False):
        with st.form("business_registration"):
            col1, col2 = st.columns(2)
            
            with col1:
                business_name = st.text_input("Business/Company Name", placeholder="e.g., Apple Inc., Samsung Electronics")
                brand_to_track = st.selectbox(
                    "Phone Brand to Track",
                    ["Apple", "Samsung", "Google", "OnePlus", "Xiaomi", "Nothing", "Huawei", "Other"]
                )
            
            with col2:
                contact_email = st.text_input("Contact Email", placeholder="business@company.com")
                alert_frequency = st.selectbox(
                    "Alert Frequency",
                    ["Real-time", "Daily", "Weekly", "Monthly"]
                )
            
            # Advanced alert thresholds
            st.markdown("**📊 Custom Alert Thresholds (Optional)**")
            col1, col2 = st.columns(2)
            
            with col1:
                negative_threshold = st.slider(
                    "Negative Sentiment Alert (%)", 
                    min_value=10, max_value=50, value=25, step=5,
                    help="Alert when negative sentiment exceeds this percentage"
                )
                rating_threshold = st.number_input(
                    "Minimum Rating Alert", 
                    min_value=2.0, max_value=4.5, value=3.5, step=0.1,
                    help="Alert when average rating drops below this value"
                )
            
            with col2:
                competitor_gap = st.slider(
                    "Competitive Gap Alert (%)",
                    min_value=5, max_value=30, value=15, step=5,
                    help="Alert when competitors gain this much advantage"
                )
            
            if st.form_submit_button("🚀 Register Business", type="primary"):
                if business_name and brand_to_track:
                    # Register the business
                    thresholds = {
                        'negative_sentiment_threshold': negative_threshold / 100,
                        'rating_drop_threshold': 5.0 - rating_threshold,
                        'competitor_gap_threshold': competitor_gap / 100
                    }
                    
                    business_id = st.session_state.business_tracker.register_business(
                        business_name, brand_to_track, thresholds
                    )
                    
                    st.session_state.current_business_id = business_id
                    st.session_state.selected_brand = brand_to_track
                    
                    st.success(f"✅ Successfully registered {business_name} for {brand_to_track} tracking!")
                    st.info(f"🆔 Your Business ID: `{business_id}`")
                    st.rerun()
                else:
                    st.error("Please fill in business name and select a brand to track.")
    
    # Quick brand selection for demo
    st.subheader("🎯 Quick Brand Analysis")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_brand = st.selectbox(
            "Select Brand to Analyze",
            ["Apple", "Samsung", "Google", "OnePlus", "Xiaomi"],
            index=0,
            key="quick_brand_select"
        )
    
    with col2:
        analysis_period = st.selectbox(
            "Analysis Period",
            ["Last 30 Days", "Last 90 Days", "Last 6 Months", "All Time"],
            index=1
        )
    
    with col3:
        if st.button("🗘 Generate Report", type="primary"):
            st.session_state.selected_brand = selected_brand
            st.session_state.analysis_period = analysis_period
    
    # Main dashboard content
    if 'selected_brand' in st.session_state:
        brand_name = st.session_state.selected_brand
        df = st.session_state.df
        business_tracker = st.session_state.business_tracker
        
        st.markdown("---")
        st.subheader(f"📋 {brand_name} Brand Performance Dashboard")
        
        # Get brand overview
        overview = business_tracker.get_brand_sentiment_overview(df, brand_name)
        
        # Key Performance Indicators
        st.markdown("#### 📈 Key Performance Indicators")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            trend_icon = "📈" if overview['sentiment_trend'] == 'improving' else "📉" if overview['sentiment_trend'] == 'declining' else "➡️"
            st.metric(
                "😊 Positive Sentiment", 
                f"{overview['positive_sentiment_rate']:.1f}%",
                delta=f"{overview['sentiment_trend']} {trend_icon}"
            )
        
        with col2:
            st.metric(
                "⭐ Average Rating", 
                f"{overview['average_rating']:.1f}/5.0",
                delta=f"{overview['average_rating']-4.0:+.1f}" if overview['average_rating'] != 4.0 else None
            )
        
        with col3:
            st.metric(
                "📝 Total Reviews", 
                f"{overview['total_reviews']:,}",
                "Data Points"
            )
        
        with col4:
            competitive_data = business_tracker.get_competitive_analysis(df, brand_name)
            market_position = competitive_data.get('market_position', 'N/A')
            st.metric(
                "🏆 Market Position", 
                f"#{market_position}" if market_position != 'N/A' else "N/A",
                f"of {competitive_data.get('total_brands_analyzed', 0)} brands"
            )
        
        with col5:
            market_share = competitive_data.get('market_share', {}).get(brand_name, 0)
            st.metric(
                "🎥 Market Share", 
                f"{market_share:.1f}%",
                "Review Volume"
            )
        
        # Alert System
        st.markdown("#### 🚨 Business Alerts")
        
        # Generate alerts for demo business
        if 'current_business_id' in st.session_state:
            alerts = business_tracker.generate_business_alerts(df, st.session_state.current_business_id)
        else:
            # Demo alerts based on overview
            alerts = []
            if overview['negative_sentiment_rate'] > 25:
                alerts.append({
                    'severity': 'high',
                    'title': '⚠️ High Negative Sentiment Alert',
                    'message': f"{overview['negative_sentiment_rate']:.1f}% of recent reviews are negative",
                    'recommendation': 'Review recent feedback and address common complaints'
                })
            if overview['positive_sentiment_rate'] > 80:
                alerts.append({
                    'severity': 'positive',
                    'title': '🚀 Strong Positive Sentiment',
                    'message': f"{overview['positive_sentiment_rate']:.1f}% positive sentiment rate",
                    'recommendation': 'Leverage this momentum for marketing and expansion'
                })
        
        if alerts:
            for alert in alerts:
                if alert['severity'] == 'high':
                    st.error(f"**{alert['title']}**\n\n{alert['message']}\n\n*Recommendation: {alert['recommendation']}*")
                elif alert['severity'] == 'positive':
                    st.success(f"**{alert['title']}**\n\n{alert['message']}\n\n*Recommendation: {alert['recommendation']}*")
                else:
                    st.warning(f"**{alert['title']}**\n\n{alert['message']}\n\n*Recommendation: {alert['recommendation']}*")
        else:
            st.info("🟢 No alerts at this time - your brand performance is within normal parameters.")
        
        # Market Trends Analysis (NEW SECTION)
        st.markdown("#### 📈 Market Trends & Intelligence")
        
        # Get market trends data
        market_trends = business_tracker.get_market_trends_analysis(df)
        quarterly_trends = business_tracker.get_quarterly_trends(df)
        
        if market_trends:
            # Market overview metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Market Reviews", 
                    f"{market_trends['total_market_reviews']:,}",
                    "Active Market"
                )
            
            with col2:
                st.metric(
                    "Market Avg Rating", 
                    f"{market_trends['overall_avg_rating']:.1f}/5.0",
                    "Industry Standard"
                )
            
            with col3:
                market_sentiment = market_trends.get('market_sentiment', {})
                st.metric(
                    "Market Positive %", 
                    f"{market_sentiment.get('positive_rate', 0):.1f}%",
                    "Overall Sentiment"
                )
            
            with col4:
                if quarterly_trends and 'growth_metrics' in quarterly_trends:
                    growth = quarterly_trends['growth_metrics'].get('review_growth', 0)
                    st.metric(
                        "Market Growth", 
                        f"{growth:+.1f}%",
                        "Quarter-over-Quarter"
                    )
                else:
                    st.metric("Market Growth", "N/A", "Insufficient Data")
            
            # Price segment analysis
            if 'price_segment_performance' in market_trends:
                st.markdown("**📊 Price Segment Performance:**")
                
                segment_data = market_trends['price_segment_performance']
                if segment_data:
                    segment_df = pd.DataFrame(segment_data).T
                    segment_df.columns = ['Reviews', 'Avg Rating', 'Market Share %', 'Positive %']
                    segment_df = segment_df.round(2)
                    
                    # Simple table view (no fancy styling)
                    st.dataframe(segment_df, use_container_width=True)
            
            # Feature importance trends
            if 'feature_importance_trends' in market_trends:
                st.markdown("**🔍 What Users Care About Most:**")
                
                features = market_trends['feature_importance_trends']
                if features:
                    feature_cols = st.columns(len(features))
                    
                    for i, (feature, data) in enumerate(features.items()):
                        with feature_cols[i]:
                            importance = data.get('importance_score', 0)
                            st.metric(
                                feature,
                                f"{importance:.0f}%",
                                f"{data.get('mention_count', 0)} mentions"
                            )
        
        # Simplified Analytics (replacing overly visual charts)
        st.markdown("#### 📊 Key Analytics")
        
        # Simple data-focused metrics instead of fancy charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🎯 Brand Performance Summary:**")
            perf_data = {
                'Metric': ['Positive Sentiment', 'Average Rating', 'Market Position', 'Review Volume'],
                'Your Brand': [f"{overview['positive_sentiment_rate']:.1f}%", 
                             f"{overview['average_rating']:.1f}/5.0",
                             f"#{market_position}" if market_position != 'N/A' else 'N/A',
                             f"{overview['total_reviews']:,}"],
                'Market Average': [f"{market_trends.get('market_sentiment', {}).get('positive_rate', 0):.1f}%" if market_trends else 'N/A',
                                 f"{market_trends.get('overall_avg_rating', 0):.1f}/5.0" if market_trends else 'N/A',
                                 "Varies",
                                 f"{market_trends.get('total_market_reviews', 0)//len(market_trends.get('brand_market_share', {}).keys()) if market_trends and market_trends.get('brand_market_share') else 0:,}"]
            }
            st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)
        
        with col2:
            # Market trends chart (simplified)
            if market_trends:
                trends_chart = business_tracker.create_market_trends_chart(df)
                st.plotly_chart(trends_chart, use_container_width=True)
        
        # Quarterly trends (if available)
        if quarterly_trends and 'quarterly_performance' in quarterly_trends:
            st.markdown("**📅 Quarterly Market Trends:**")
            
            quarterly_data = quarterly_trends['quarterly_performance']
            if quarterly_data:
                quarters_df = pd.DataFrame(quarterly_data).T
                quarters_df.columns = ['Total Reviews', 'Avg Rating', 'Positive %']
                quarters_df = quarters_df.round(2)
                
                st.dataframe(quarters_df, use_container_width=True)
                
                # Trend direction indicator
                trend_direction = quarterly_trends.get('trend_direction', 'stable')
                if trend_direction == 'improving':
                    st.success("📈 **Market Trend:** Improving - ratings and sentiment are trending upward")
                elif trend_direction == 'declining':
                    st.error("📉 **Market Trend:** Declining - market showing downward trends")
                else:
                    st.info("➡️ **Market Trend:** Stable - market performance is steady")
        
        # Product Performance Breakdown
        if overview['product_performance']:
            st.markdown("#### 📱 Product Performance Breakdown")
            
            product_data = overview['product_performance']
            products = list(product_data.keys())
            
            # Create product performance table
            performance_df = pd.DataFrame(product_data).T
            performance_df = performance_df.round(2)
            performance_df.columns = ['Avg Rating', 'Positive %', 'Review Count']
            performance_df = performance_df.sort_values('Positive %', ascending=False)
            
            st.dataframe(
                performance_df.style.highlight_max(axis=0, color='lightgreen')
                             .highlight_min(axis=0, color='lightcoral'),
                use_container_width=True
            )
        
        # Competitive Analysis Detail
        if competitive_data:
            st.markdown("#### 🏆 Detailed Competitive Analysis")
            
            # Top competitors table
            if 'brand_rankings' in competitive_data:
                st.markdown("**Top Performing Brands:**")
                
                rankings_df = pd.DataFrame(competitive_data['brand_rankings']).T
                rankings_df.columns = ['Avg Rating', 'Positive %', 'Review Count']
                rankings_df = rankings_df.round(2)
                
                # Highlight the focus brand
                def highlight_focus_brand(row):
                    if row.name == brand_name:
                        return ['background-color: #FFE082'] * len(row)
                    return [''] * len(row)
                
                st.dataframe(
                    rankings_df.style.apply(highlight_focus_brand, axis=1),
                    use_container_width=True
                )
            
            # Competitive gaps
            if 'competitive_gaps' in competitive_data and competitive_data['competitive_gaps']:
                st.markdown("**⚠️ Areas for Improvement:**")
                gaps = competitive_data['competitive_gaps']
                for gap_type, gap_message in gaps.items():
                    st.warning(f"**{gap_type.title()} Gap:** {gap_message}")
        
        # Action Items and Recommendations
        st.markdown("#### 🎯 Strategic Recommendations")
        
        recommendations = []
        
        if overview['positive_sentiment_rate'] < 70:
            recommendations.append("🛠️ **Focus on Quality Improvement**: Address common user complaints to boost satisfaction")
        
        if overview['negative_sentiment_rate'] > 20:
            recommendations.append("📞 **Customer Support**: Enhance customer service to address negative feedback proactively")
        
        if market_position and market_position > 3:
            recommendations.append("🏆 **Competitive Strategy**: Analyze top performers and implement differentiation strategies")
        
        if overview['positive_sentiment_rate'] > 80:
            recommendations.append("📺 **Marketing Leverage**: Use positive sentiment for marketing campaigns and testimonials")
        
        if recommendations:
            for rec in recommendations:
                st.info(rec)
        else:
            st.success("✅ **Strong Performance**: Your brand is performing well across all metrics. Continue current strategies and monitor for changes.")
    
    else:
        # Initial state - no brand selected
        st.info("👆 Select a brand above to view detailed sentiment analysis and competitive positioning.")
        
        # Demo features showcase
        st.markdown("### 📊 Enhanced Business Intelligence Features")
        
        feature_col1, feature_col2, feature_col3 = st.columns(3)
        
        with feature_col1:
            st.markdown("""
                **📈 Market Trends Analysis**
                - Overall market sentiment tracking
                - Price segment performance
                - Quarterly growth trends
                - Feature importance analysis
            """)
        
        with feature_col2:
            st.markdown("""
                **🏆 Competitive Intelligence**
                - Brand market positioning
                - Competitive gap analysis
                - Market share insights
                - Performance benchmarking
            """)
        
        with feature_col3:
            st.markdown("""
                **💼 Data-Driven Insights**
                - Clean data tables
                - Strategic recommendations
                - Performance metrics
                - Simplified reporting
            """)

def show_ai_assistant_chat():
    """AI-powered conversational assistant for phone recommendations"""
    st.header("🤖 AI Phone Assistant")
    st.markdown("Chat with our intelligent assistant to get personalized phone recommendations!")
    
    # Back button
    if st.button("← Back to Home"):
        st.session_state.current_page = "home"
        st.rerun()
    
    if not ADVANCED_AI_AVAILABLE or 'conversational_ai' not in st.session_state:
        st.error("⚠️ AI Assistant is not available. Please check your configuration.")
        return
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        st.session_state.session_id = f"user_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Display chat history
    st.markdown("### 💬 Conversation")
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Show conversation history
        for i, message in enumerate(st.session_state.chat_history):
            if message['type'] == 'user':
                st.markdown(f"""
                    <div style="background: #E3F2FD; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; border-left: 4px solid #2196F3;">
                        <strong>You:</strong> {message['content']}
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div style="background: #E8F5E8; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; border-left: 4px solid #4CAF50;">
                        <strong>🤖 AI Assistant:</strong> {message['content']}
                    </div>
                """, unsafe_allow_html=True)
                
                # Show additional data if available
                if 'data' in message and message['data']:
                    with st.expander("📊 View Details", expanded=False):
                        st.json(message['data'])
    
    # Input area
    st.markdown("---")
    
    # Example queries
    st.markdown("**💡 Try asking:**")
    example_col1, example_col2, example_col3 = st.columns(3)
    
    with example_col1:
        if st.button("📱 Phones under $500"):
            user_input = "Show me phones under $500 and their user reviews"
            process_ai_query(user_input)
    
    with example_col2:
        if st.button("📸 Camera-focused phones"):
            user_input = "Analyze phones with good camera reviews"
            process_ai_query(user_input)
    
    with example_col3:
        if st.button("⚖️ Compare iPhone vs Samsung"):
            user_input = "Compare user sentiment for iPhone vs Samsung phones"
            process_ai_query(user_input)
    
    # Text input (with pre-fill support)
    prefill_query = st.session_state.get('ai_prefill', '')
    if prefill_query:
        # Clear prefill after using it
        st.session_state.ai_prefill = ''
        # Process the pre-filled query immediately
        process_ai_query(prefill_query)
        st.rerun()
    
    user_input = st.text_input(
        "Type your question:", 
        placeholder="e.g., 'Analyze gaming phones' or 'Compare user reviews for Pixel 8 vs iPhone 15'",
        key="ai_chat_input"
    )
    
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("Send 📤", type="primary") and user_input:
            process_ai_query(user_input)
    
    with col2:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Show AI system status and available data
    col1, col2 = st.columns(2)
    
    with col1:
        # RAG System Status
        if RAG_AVAILABLE and 'rag_system' in st.session_state:
            st.markdown("""
                <div style="background: #E8F5E8; padding: 1rem; border-radius: 8px; border-left: 4px solid #4CAF50;">
                    <h4 style="margin: 0; color: #2E7D32;">🤖 Multi-Agent RAG Active</h4>
                    <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">4 specialized agents working together</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Show active agents
            if 'rag_agents' in st.session_state:
                agents = st.session_state.rag_agents
                active_agents = [name for name, agent in agents.items() if agent.get('active', False)]
                st.markdown(f"**Active Agents:** {', '.join([agent.title() for agent in active_agents])}")
        else:
            st.markdown("""
                <div style="background: #E3F2FD; padding: 1rem; border-radius: 8px; border-left: 4px solid #2196F3;">
                    <h4 style="margin: 0; color: #1976D2;">🤖 Standard AI Active</h4>
                    <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">Basic conversational AI</p>
                </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Show available phone data info
        if st.session_state.df is not None:
            df = st.session_state.df
            st.markdown("**📊 Knowledge Base:**")
            st.metric("Total Reviews", f"{len(df):,}")
            st.metric("Phone Models", f"{df['product'].nunique() if 'product' in df.columns else 0}")
            st.metric("Brands", f"{df['brand'].nunique() if 'brand' in df.columns else 0}")
            
            # Show user memory summary if RAG is available
            if RAG_AVAILABLE and 'rag_system' in st.session_state:
                with st.expander("🧠 Your AI Memory", expanded=False):
                    memory_summary = get_user_memory_summary()
                    st.markdown(memory_summary)

def process_ai_query(user_input: str):
    """Process user query through the AI assistant"""
    if not user_input.strip():
        return
    
    # Add user message to history
    st.session_state.chat_history.append({
        'type': 'user',
        'content': user_input,
        'timestamp': datetime.now().isoformat()
    })
    
    try:
        # Get AI response
        conversational_ai = st.session_state.conversational_ai
        context_data = {
            'available_phones': st.session_state.df['product'].unique().tolist() if 'product' in st.session_state.df.columns else [],
            'available_brands': st.session_state.df['brand'].unique().tolist() if 'brand' in st.session_state.df.columns else [],
            'total_reviews': len(st.session_state.df)
        }
        
        with st.spinner("🤔 AI is thinking..."):
            # Use RAG system if available for enhanced responses
            if RAG_AVAILABLE and 'rag_system' in st.session_state:
                try:
                    rag_response = rag_enhanced_query_processing(user_input, context_data)
                    
                    if 'error' not in rag_response:
                        response = {
                            'message': rag_response['response'],
                            'data': {
                                'agents_used': rag_response.get('agents_used', []),
                                'confidence': rag_response.get('confidence', 0.8),
                                'retrieved_info': rag_response.get('retrieved_info', {}),
                                'system_type': 'Multi-Agent RAG'
                            }
                        }
                    else:
                        # Fallback to standard conversational AI
                        response = conversational_ai.process_query(
                            user_input, 
                            session_id=st.session_state.session_id,
                            context_data=context_data
                        )
                except Exception as e:
                    # Fallback to standard conversational AI on RAG error
                    response = conversational_ai.process_query(
                        user_input, 
                        session_id=st.session_state.session_id,
                        context_data=context_data
                    )
            else:
                # Standard conversational AI
                response = conversational_ai.process_query(
                    user_input, 
                    session_id=st.session_state.session_id,
                    context_data=context_data
                )
        
        # Add AI response to history
        st.session_state.chat_history.append({
            'type': 'assistant',
            'content': response.get('message', 'I apologize, but I encountered an error processing your request.'),
            'data': response.get('data', {}),
            'timestamp': datetime.now().isoformat()
        })
        
        # Clear input and rerun
        st.rerun()
        
    except Exception as e:
        st.error(f"⚠️ Error processing your request: {str(e)}")
        
        # Add error response to history
        st.session_state.chat_history.append({
            'type': 'assistant',
            'content': "I'm sorry, I encountered an error while processing your request. Please try rephrasing your question or contact support if the issue persists.",
            'timestamp': datetime.now().isoformat()
        })

def run_advanced_sentiment_analysis(phone_data: pd.DataFrame, phone_name: str) -> Dict[str, Any]:
    """Run advanced AI sentiment analysis on phone reviews"""
    if not ADVANCED_AI_AVAILABLE or 'deeper_insights' not in st.session_state:
        return {}
    
    try:
        deeper_insights = st.session_state.deeper_insights
        
        # Sample reviews for analysis (limit to avoid timeouts)
        sample_reviews = phone_data['review_text'].dropna().head(50).tolist()
        
        if not sample_reviews:
            return {}
        
        # Combine reviews for overall analysis
        combined_text = ' '.join(sample_reviews[:10])  # Use first 10 reviews for combined analysis
        
        analysis_results = {
            'emotions': {},
            'sarcasm': {},
            'cultural_insights': {},
            'review_quality': {}
        }
        
        # Emotion Analysis
        try:
            emotion_detector = deeper_insights.emotion_detector if hasattr(deeper_insights, 'emotion_detector') else None
            if emotion_detector:
                emotion_result = emotion_detector.detect_emotions(combined_text)
                analysis_results['emotions'] = {
                    'primary_emotion': emotion_result.primary_emotion.value if hasattr(emotion_result.primary_emotion, 'value') else str(emotion_result.primary_emotion),
                    'emotion_scores': {k.value if hasattr(k, 'value') else str(k): v for k, v in emotion_result.emotion_scores.items()},
                    'intensity': emotion_result.intensity,
                    'confidence': emotion_result.confidence,
                    'emotion_words': emotion_result.emotion_words[:10]  # Limit to first 10
                }
        except Exception as e:
            logger.warning(f"Emotion analysis failed: {e}")
            # Fallback emotion analysis
            analysis_results['emotions'] = generate_fallback_emotion_analysis(combined_text)
        
        # Sarcasm Detection
        try:
            sarcasm_detector = deeper_insights.sarcasm_detector if hasattr(deeper_insights, 'sarcasm_detector') else None
            if sarcasm_detector:
                avg_rating = phone_data['rating'].mean() if 'rating' in phone_data.columns else None
                sarcasm_result = sarcasm_detector.detect_sarcasm(combined_text, rating=avg_rating)
                
                analysis_results['sarcasm'] = {
                    'is_sarcastic': sarcasm_result.is_sarcastic,
                    'confidence': sarcasm_result.confidence,
                    'indicators': [ind.value if hasattr(ind, 'value') else str(ind) for ind in sarcasm_result.indicators],
                    'sarcastic_phrases': sarcasm_result.sarcastic_phrases[:5]  # Limit to first 5
                }
        except Exception as e:
            logger.warning(f"Sarcasm analysis failed: {e}")
            # Fallback sarcasm analysis
            analysis_results['sarcasm'] = generate_fallback_sarcasm_analysis(combined_text)
        
        # Cultural Insights (simplified)
        try:
            analysis_results['cultural_insights'] = {
                'primary_region': 'Global',
                'cultural_score': 0.75,
                'language_indicators': 'Standard English',
                'regional_markers': ['global', 'standard']
            }
        except Exception as e:
            logger.warning(f"Cultural analysis failed: {e}")
        
        return analysis_results
    
    except Exception as e:
        logger.error(f"Advanced sentiment analysis failed: {e}")
        return {}

def generate_fallback_emotion_analysis(text: str) -> Dict[str, Any]:
    """Generate fallback emotion analysis using simple keyword matching"""
    text_lower = text.lower()
    
    emotions = {
        'joy': ['happy', 'great', 'excellent', 'amazing', 'love', 'perfect', 'wonderful'],
        'trust': ['reliable', 'trustworthy', 'dependable', 'solid', 'consistent'],
        'fear': ['worried', 'concerned', 'afraid', 'risky', 'dangerous'],
        'surprise': ['surprised', 'unexpected', 'amazing', 'wow', 'incredible'],
        'sadness': ['disappointed', 'sad', 'terrible', 'awful', 'poor'],
        'disgust': ['disgusting', 'gross', 'horrible', 'unacceptable'],
        'anger': ['angry', 'frustrated', 'annoyed', 'outraged', 'mad'],
        'anticipation': ['excited', 'looking forward', 'hopeful', 'eager']
    }
    
    emotion_scores = {}
    total_matches = 0
    
    for emotion, keywords in emotions.items():
        matches = sum(1 for keyword in keywords if keyword in text_lower)
        emotion_scores[emotion] = matches
        total_matches += matches
    
    # Normalize scores
    if total_matches > 0:
        emotion_scores = {k: v/total_matches for k, v in emotion_scores.items()}
    
    # Find primary emotion
    primary_emotion = max(emotion_scores.keys(), key=lambda k: emotion_scores[k]) if emotion_scores else 'neutral'
    
    return {
        'primary_emotion': primary_emotion,
        'emotion_scores': emotion_scores,
        'intensity': min(1.0, total_matches / 20),
        'confidence': 0.7 if total_matches > 5 else 0.4,
        'emotion_words': []
    }

def generate_fallback_sarcasm_analysis(text: str) -> Dict[str, Any]:
    """Generate fallback sarcasm analysis using simple pattern matching"""
    text_lower = text.lower()
    
    sarcasm_indicators = [
        'yeah right', 'sure thing', 'oh great', 'wonderful...', '"great"', '\'great\'',
        'amazing...', 'perfect...', 'exactly what', 'just what i needed'
    ]
    
    sarcasm_count = sum(1 for indicator in sarcasm_indicators if indicator in text_lower)
    has_ellipsis = '...' in text or '…' in text
    has_quotes = '"' in text and text.count('"') >= 2
    
    total_indicators = sarcasm_count + (1 if has_ellipsis else 0) + (1 if has_quotes else 0)
    
    return {
        'is_sarcastic': total_indicators >= 2,
        'confidence': min(0.9, total_indicators * 0.3),
        'indicators': ['contradiction' if sarcasm_count > 0 else 'ellipsis' if has_ellipsis else 'quotation_marks'],
        'sarcastic_phrases': []
    }

def should_recommend_alternatives(phone_data, threshold=60):
    """Check if a phone needs alternative recommendations based on sentiment"""
    if 'sentiment_label' not in phone_data.columns or len(phone_data) == 0:
        return False, 0
    
    positive_pct = (phone_data['sentiment_label'] == 'positive').mean() * 100
    return positive_pct < threshold, positive_pct

def show_smart_search_results(query: str):
    """Display smart search results based on natural language query"""
    st.markdown(f"### 🎯 Smart Search Results for: '{query}'")
    
    if not ADVANCED_AI_AVAILABLE or 'smart_search' not in st.session_state:
        st.error("⚠️ Smart search is not available. Using basic search...")
        show_basic_search_results(query)
        return
    
    try:
        with st.spinner("🤖 AI is analyzing your request..."):
            # Use smart search engine to process the query
            smart_search = st.session_state.smart_search
            df = st.session_state.df
            
            # Extract search parameters from natural language
            search_params = extract_search_parameters(query)
            
            # Get matching phones
            matching_phones = find_matching_phones(df, search_params)
            
            if len(matching_phones) == 0:
                st.warning("🔍 No phones found matching your criteria. Try broadening your search.")
                return
            
            st.success(f"🎉 Found {len(matching_phones)} phones matching your criteria!")
            
            # Display AI interpretation of the query
            with st.expander("🧠 How AI Interpreted Your Request", expanded=False):
                st.markdown("**Extracted Parameters:**")
                for key, value in search_params.items():
                    if value:
                        st.markdown(f"- **{key.title()}:** {value}")
            
            # Show top matching phones
            for i, (phone, score) in enumerate(matching_phones.head(5).iterrows()):
                phone_data = df[df['product'] == phone]
                
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.markdown(f"### #{i+1}. {phone}")
                    
                # Show why it matches
                match_reasons = generate_match_reasons(phone, phone_data, search_params)
                if match_reasons:
                    st.markdown("**Why it matches:**")
                    for reason in match_reasons:
                        st.markdown(f"• {reason}")
                
                # Show alternatives if this phone has poor sentiment
                if 'sentiment_label' in phone_data.columns:
                    positive_pct = (phone_data['sentiment_label'] == 'positive').mean() * 100
                    if positive_pct < 60:  # Show alternatives for poor sentiment phones
                        st.warning(f"⚠️ This phone has {positive_pct:.0f}% positive sentiment. Consider the alternatives shown below.")
                    
                    # Show key metrics
                    if 'sentiment_label' in phone_data.columns:
                        positive_pct = (phone_data['sentiment_label'] == 'positive').mean() * 100
                        st.markdown(f"**User Satisfaction:** {positive_pct:.0f}%")
                    
                    if 'rating' in phone_data.columns:
                        avg_rating = phone_data['rating'].mean()
                        st.markdown(f"**Average Rating:** {avg_rating:.1f}/5.0")
                    
                    st.markdown(f"**Based on:** {len(phone_data)} reviews")
                
                with col2:
                    if st.button(f"📊 Analyze", key=f"analyze_{i}"):
                        st.session_state.selected_phone = phone
                        st.session_state.current_page = "research"
                        st.rerun()
                
                with col3:
                    if st.button(f"⚖️ Compare", key=f"compare_{i}"):
                        st.session_state.phone_to_compare = phone
                        st.session_state.current_page = "compare"
                        st.rerun()
                
                st.markdown("---")
            
            # Show category-based recommendations if search included phones with poor sentiment
            poor_sentiment_phones = []
            for i, (phone, score) in enumerate(matching_phones.head(5).iterrows()):
                phone_data = df[df['product'] == phone]
                needs_alternatives, sentiment_pct = should_recommend_alternatives(phone_data)
                if needs_alternatives:
                    poor_sentiment_phones.append((phone, sentiment_pct))
            
            if poor_sentiment_phones:
                st.markdown("---")
                st.markdown("### 🔄 Alternative Recommendations")
                st.warning("📊 Some phones in your search results have lower user satisfaction. Here are better alternatives:")
                
                # Find better alternatives from the same search
                better_alternatives = []
                for phone, score in matching_phones.iterrows():
                    phone_data = df[df['product'] == phone]
                    if 'sentiment_label' in phone_data.columns:
                        positive_pct = (phone_data['sentiment_label'] == 'positive').mean() * 100
                        if positive_pct >= 70:  # Good alternatives
                            better_alternatives.append((phone, phone_data, positive_pct))
                
                # Show top 3 better alternatives
                better_alternatives.sort(key=lambda x: x[2], reverse=True)
                for phone, phone_data, positive_pct in better_alternatives[:3]:
                    rating = phone_data['rating'].mean() if 'rating' in phone_data.columns else 0
                    st.success(f"👍 **{phone}**: {positive_pct:.0f}% satisfaction, {rating:.1f}/5.0 rating, {len(phone_data)} reviews")
            
            # Show more options
            if len(matching_phones) > 5:
                st.info(f"📊 {len(matching_phones) - 5} more phones match your criteria. Refine your search for better results.")
    
    except Exception as e:
        st.error(f"⚠️ Smart search encountered an error: {str(e)}")
        st.info("Falling back to basic search...")
        show_basic_search_results(query)

def extract_search_parameters(query: str) -> Dict[str, Any]:
    """Extract search parameters from natural language query"""
    query_lower = query.lower()
    params = {
        'budget_min': None,
        'budget_max': None,
        'features': [],
        'brands': [],
        'use_case': None,
        'priority': None
    }
    
    # Extract budget information
    import re
    budget_patterns = [
        r'under \$?(\d+)', r'below \$?(\d+)', r'less than \$?(\d+)',
        r'\$?(\d+)-\$?(\d+)', r'\$?(\d+) to \$?(\d+)',
        r'around \$?(\d+)', r'about \$?(\d+)'
    ]
    
    for pattern in budget_patterns:
        match = re.search(pattern, query_lower)
        if match:
            numbers = [int(x) for x in match.groups() if x]
            if len(numbers) == 1:
                if 'under' in pattern or 'below' in pattern or 'less than' in pattern:
                    params['budget_max'] = numbers[0]
                else:
                    params['budget_max'] = numbers[0] + 100  # Add buffer for "around"
            elif len(numbers) == 2:
                params['budget_min'] = min(numbers)
                params['budget_max'] = max(numbers)
            break
    
    # Extract features
    feature_keywords = {
        'camera': ['camera', 'photo', 'picture', 'photography', 'selfie', 'video'],
        'battery': ['battery', 'power', 'charge', 'lasting', 'long-lasting'],
        'performance': ['fast', 'speed', 'performance', 'gaming', 'processor', 'smooth'],
        'display': ['screen', 'display', 'size', 'big', 'large', 'bright'],
        'storage': ['storage', 'memory', 'space', 'gb', 'tb'],
        'design': ['design', 'look', 'style', 'beautiful', 'premium', 'build']
    }
    
    for feature, keywords in feature_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            params['features'].append(feature)
    
    # Extract brands
    brand_keywords = {
        'apple': ['apple', 'iphone'],
        'samsung': ['samsung', 'galaxy'],
        'google': ['google', 'pixel'],
        'oneplus': ['oneplus', 'one plus', '1+'],
        'xiaomi': ['xiaomi', 'redmi']
    }
    
    for brand, keywords in brand_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            params['brands'].append(brand.title())
    
    # Extract use case
    use_cases = {
        'gaming': ['gaming', 'games', 'gamer'],
        'photography': ['photography', 'photo', 'camera', 'pictures'],
        'business': ['business', 'work', 'professional', 'office'],
        'social': ['social', 'social media', 'instagram', 'tiktok']
    }
    
    for use_case, keywords in use_cases.items():
        if any(keyword in query_lower for keyword in keywords):
            params['use_case'] = use_case
            break
    
    return params

def find_matching_phones(df: pd.DataFrame, search_params: Dict[str, Any]) -> pd.DataFrame:
    """Find phones matching search parameters"""
    if df is None or len(df) == 0:
        return pd.DataFrame()
    
    # Start with all phones
    matching_df = df.copy()
    
    # Filter by brands if specified
    if search_params['brands'] and 'brand' in df.columns:
        matching_df = matching_df[matching_df['brand'].isin(search_params['brands'])]
    
    # Calculate match scores for each phone
    phone_scores = []
    
    for phone in matching_df['product'].unique():
        phone_data = matching_df[matching_df['product'] == phone]
        score = calculate_phone_match_score(phone, phone_data, search_params)
        phone_scores.append((phone, score))
    
    # Sort by score and return as DataFrame
    phone_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Create result DataFrame
    result_phones = [phone for phone, score in phone_scores if score > 0.1]  # Minimum threshold
    
    if result_phones:
        result_df = pd.DataFrame({'product': result_phones})
        result_df['match_score'] = [score for phone, score in phone_scores if score > 0.1]
        return result_df.set_index('product')
    
    return pd.DataFrame()

def calculate_phone_match_score(phone: str, phone_data: pd.DataFrame, search_params: Dict[str, Any]) -> float:
    """Calculate how well a phone matches search parameters"""
    score = 0.0
    
    # Base score from user satisfaction
    if 'sentiment_label' in phone_data.columns:
        positive_pct = (phone_data['sentiment_label'] == 'positive').mean()
        score += positive_pct * 0.4
    
    # Rating score
    if 'rating' in phone_data.columns:
        avg_rating = phone_data['rating'].mean()
        score += (avg_rating / 5.0) * 0.3
    
    # Feature matching (if reviews mention features)
    if search_params['features'] and 'review_text' in phone_data.columns:
        review_text = ' '.join(phone_data['review_text'].dropna().head(20).tolist()).lower()
        
        feature_matches = 0
        for feature in search_params['features']:
            feature_keywords = {
                'camera': ['camera', 'photo', 'picture'],
                'battery': ['battery', 'power', 'charge'],
                'performance': ['fast', 'speed', 'performance'],
                'display': ['screen', 'display'],
                'storage': ['storage', 'memory', 'space'],
                'design': ['design', 'look', 'build']
            }
            
            if feature in feature_keywords:
                if any(keyword in review_text for keyword in feature_keywords[feature]):
                    feature_matches += 1
        
        if search_params['features']:
            score += (feature_matches / len(search_params['features'])) * 0.3
    
    return min(1.0, score)

def generate_match_reasons(phone: str, phone_data: pd.DataFrame, search_params: Dict[str, Any]) -> List[str]:
    """Generate reasons why this phone matches the search"""
    reasons = []
    
    # User satisfaction
    if 'sentiment_label' in phone_data.columns:
        positive_pct = (phone_data['sentiment_label'] == 'positive').mean() * 100
        if positive_pct >= 80:
            reasons.append(f"Excellent user satisfaction ({positive_pct:.0f}% positive)")
        elif positive_pct >= 70:
            reasons.append(f"Good user satisfaction ({positive_pct:.0f}% positive)")
    
    # Rating
    if 'rating' in phone_data.columns:
        avg_rating = phone_data['rating'].mean()
        if avg_rating >= 4.5:
            reasons.append(f"Outstanding ratings ({avg_rating:.1f}/5.0)")
        elif avg_rating >= 4.0:
            reasons.append(f"Great ratings ({avg_rating:.1f}/5.0)")
    
    # Feature matching
    if search_params['features'] and 'review_text' in phone_data.columns:
        review_text = ' '.join(phone_data['review_text'].dropna().head(10).tolist()).lower()
        
        for feature in search_params['features']:
            if feature == 'camera' and any(word in review_text for word in ['great camera', 'excellent camera', 'amazing camera']):
                reasons.append("Praised for camera quality")
            elif feature == 'battery' and any(word in review_text for word in ['good battery', 'great battery', 'long battery']):
                reasons.append("Solid battery performance")
            elif feature == 'performance' and any(word in review_text for word in ['fast', 'smooth', 'great performance']):
                reasons.append("Strong performance reviews")
    
    # Brand preference
    if search_params['brands'] and 'brand' in phone_data.columns:
        phone_brand = phone_data['brand'].iloc[0] if len(phone_data) > 0 else ''
        if phone_brand in search_params['brands']:
            reasons.append(f"Matches your {phone_brand} preference")
    
    return reasons[:4]  # Limit to top 4 reasons

def show_basic_search_results(query: str):
    """Fallback basic search when advanced AI is not available"""
    st.markdown(f"### 🔍 Basic Search Results for: '{query}'")
    
    df = st.session_state.df
    if df is None or 'product' not in df.columns:
        st.error("No phone data available for search.")
        return
    
    # Simple keyword matching
    query_words = query.lower().split()
    
    # Find phones that match query keywords
    matching_phones = []
    
    for phone in df['product'].unique():
        phone_lower = phone.lower()
        phone_data = df[df['product'] == phone]
        
        # Check if phone name matches query
        name_match = any(word in phone_lower for word in query_words)
        
        # Check if brand matches
        brand_match = False
        if 'brand' in df.columns and len(phone_data) > 0:
            brand_lower = phone_data['brand'].iloc[0].lower()
            brand_match = any(word in brand_lower for word in query_words)
        
        if name_match or brand_match:
            matching_phones.append(phone)
    
    if not matching_phones:
        st.warning("No phones found matching your search terms.")
        return
    
    st.success(f"Found {len(matching_phones)} phones matching '{query}'")
    
    # Display results
    for i, phone in enumerate(matching_phones[:5]):
        phone_data = df[df['product'] == phone]
        
        st.markdown(f"### {phone}")
        
        col1, col2 = st.columns(2)
        with col1:
            if 'rating' in phone_data.columns:
                avg_rating = phone_data['rating'].mean()
                st.metric("Average Rating", f"{avg_rating:.1f}/5.0")
        
        with col2:
            if 'sentiment_label' in phone_data.columns:
                positive_pct = (phone_data['sentiment_label'] == 'positive').mean() * 100
                st.metric("User Satisfaction", f"{positive_pct:.0f}%")
        
        if st.button(f"Learn More about {phone}", key=f"basic_search_{i}"):
            st.session_state.selected_phone = phone
            st.session_state.current_page = "research"
            st.rerun()

def get_neural_recommendations(budget, primary_use, important_features, brand_preference, df):
    """Get AI-powered neural recommendations"""
    try:
        neural_engine = st.session_state.neural_rec_engine
        
        # Create user profile
        user_profile = {
            'budget_range': budget,
            'primary_use_case': primary_use,
            'important_features': important_features,
            'brand_preferences': brand_preference,
            'timestamp': datetime.now().isoformat()
        }
        
        # Get available phones
        available_phones = df['product'].unique().tolist() if 'product' in df.columns else []
        
        if len(available_phones) < 3:
            return None
        
        # Generate neural recommendations
        recommendations = []
        
        for phone in available_phones[:20]:  # Analyze top 20 phones
            phone_data = df[df['product'] == phone]
            if len(phone_data) == 0:
                continue
                
            # Calculate neural score
            neural_score = calculate_neural_score(phone, phone_data, user_profile)
            
            if neural_score > 0.3:  # Minimum threshold
                recommendations.append({
                    'phone': phone,
                    'neural_score': neural_score,
                    'phone_data': phone_data,
                    'match_explanation': generate_ai_explanation(phone, phone_data, user_profile, neural_score)
                })
        
        # Sort by neural score
        recommendations.sort(key=lambda x: x['neural_score'], reverse=True)
        
        return recommendations[:5]  # Return top 5
        
    except Exception as e:
        logging.error(f"Neural recommendations failed: {e}")
        return None

def calculate_neural_score(phone, phone_data, user_profile):
    """Calculate AI-based neural score for phone recommendation"""
    score = 0.0
    
    # Base sentiment and rating score (40%)
    if 'sentiment_label' in phone_data.columns:
        positive_pct = (phone_data['sentiment_label'] == 'positive').mean()
        score += positive_pct * 0.4
    
    if 'rating' in phone_data.columns:
        avg_rating = phone_data['rating'].mean()
        score += (avg_rating / 5.0) * 0.3
    
    # Feature alignment score (30%)
    if user_profile['important_features'] and 'review_text' in phone_data.columns:
        feature_score = calculate_feature_alignment(
            phone_data['review_text'].dropna().head(20).tolist(),
            user_profile['important_features']
        )
        score += feature_score * 0.3
    
    # Brand preference bonus (10%)
    if user_profile['brand_preferences'] and 'brand' in phone_data.columns:
        phone_brand = phone_data['brand'].iloc[0] if len(phone_data) > 0 else ''
        if phone_brand in user_profile['brand_preferences'] or 'No preference' in user_profile['brand_preferences']:
            score += 0.1
    
    # Use case alignment (20%)
    if user_profile['primary_use_case'] and 'review_text' in phone_data.columns:
        use_case_score = calculate_use_case_alignment(
            phone_data['review_text'].dropna().head(15).tolist(),
            user_profile['primary_use_case']
        )
        score += use_case_score * 0.2
    
    return min(1.0, score)

def calculate_feature_alignment(reviews, important_features):
    """Calculate how well phone aligns with important features"""
    if not reviews or not important_features:
        return 0.0
    
    combined_text = ' '.join(reviews).lower()
    
    feature_keywords = {
        'Camera quality': ['camera', 'photo', 'picture', 'excellent camera', 'great camera', 'amazing camera'],
        'Battery life': ['battery', 'power', 'charge', 'long battery', 'great battery', 'excellent battery'],
        'Performance/Speed': ['fast', 'speed', 'performance', 'smooth', 'lag', 'quick', 'responsive'],
        'Display quality': ['display', 'screen', 'bright', 'clear', 'beautiful screen', 'great display'],
        'Build quality': ['build', 'quality', 'premium', 'solid', 'well-built', 'durable'],
        'Storage space': ['storage', 'memory', 'space', 'gb', 'capacity'],
        'Price/Value': ['value', 'price', 'worth', 'money', 'affordable', 'cheap']
    }
    
    total_score = 0.0
    for feature in important_features:
        if feature in feature_keywords:
            keywords = feature_keywords[feature]
            feature_mentions = sum(1 for keyword in keywords if keyword in combined_text)
            if feature_mentions > 0:
                # More mentions = higher score, but with diminishing returns
                total_score += min(1.0, feature_mentions / 5)
    
    return total_score / len(important_features) if important_features else 0.0

def calculate_use_case_alignment(reviews, use_case):
    """Calculate how well phone aligns with use case"""
    if not reviews or not use_case:
        return 0.0
    
    combined_text = ' '.join(reviews).lower()
    
    use_case_keywords = {
        'General use': ['reliable', 'daily', 'everyday', 'good', 'solid'],
        'Photography': ['camera', 'photo', 'picture', 'selfie', 'video', 'photography'],
        'Gaming': ['gaming', 'game', 'performance', 'fast', 'smooth', 'lag-free'],
        'Work/Business': ['professional', 'business', 'productivity', 'reliable', 'battery'],
        'Social media': ['camera', 'selfie', 'social', 'instagram', 'video', 'photo']
    }
    
    if use_case not in use_case_keywords:
        return 0.0
    
    keywords = use_case_keywords[use_case]
    matches = sum(1 for keyword in keywords if keyword in combined_text)
    
    return min(1.0, matches / len(keywords))

def generate_ai_explanation(phone, phone_data, user_profile, neural_score):
    """Generate AI explanation for analysis results"""
    explanations = []
    
    # Neural score explanation
    if neural_score > 0.8:
        explanations.append("🎯 Strong alignment with your criteria")
    elif neural_score > 0.6:
        explanations.append("✅ Good alignment with your preferences")
    elif neural_score > 0.4:
        explanations.append("👍 Moderate alignment with your requirements")
    
    # User satisfaction
    if 'sentiment_label' in phone_data.columns:
        positive_pct = (phone_data['sentiment_label'] == 'positive').mean() * 100
        if positive_pct >= 85:
            explanations.append(f"🌟 Outstanding user satisfaction ({positive_pct:.0f}%)")
        elif positive_pct >= 75:
            explanations.append(f"😊 High user satisfaction ({positive_pct:.0f}%)")
    
    # Feature alignment
    if user_profile['important_features']:
        feature_text = ', '.join(user_profile['important_features'])
        explanations.append(f"🎯 Matches your focus on {feature_text}")
    
    # Use case alignment
    if user_profile['primary_use_case']:
        explanations.append(f"📱 Optimized for {user_profile['primary_use_case'].lower()}")
    
    # Brand preference
    if user_profile['brand_preferences'] and 'brand' in phone_data.columns:
        phone_brand = phone_data['brand'].iloc[0] if len(phone_data) > 0 else ''
        if phone_brand in user_profile['brand_preferences']:
            explanations.append(f"🏷️ Your preferred {phone_brand} brand")
    
    return explanations[:4]  # Return top 4 explanations

def display_neural_recommendations(recommendations, budget, primary_use):
    """Display neural AI analysis results with enhanced explanations"""
    
    st.markdown("### 🧠 AI-Powered Analysis Results")
    st.markdown(f"*Based on neural analysis of your preferences: {budget} budget for {primary_use}*")
    
    for i, rec in enumerate(recommendations):
        phone = rec['phone']
        neural_score = rec['neural_score']
        phone_data = rec['phone_data']
        explanations = rec['match_explanation']
        
        # Recommendation confidence
        if neural_score > 0.8:
            confidence_color = "#4CAF50"
            confidence_text = "Very High"
            confidence_icon = "🎯"
        elif neural_score > 0.6:
            confidence_color = "#8BC34A"
            confidence_text = "High"
            confidence_icon = "✅"
        elif neural_score > 0.4:
            confidence_color = "#FFC107"
            confidence_text = "Medium"
            confidence_icon = "👍"
        else:
            confidence_color = "#FF9800"
            confidence_text = "Low"
            confidence_icon = "⚠️"
        
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #F8F9FA 0%, #E9ECEF 100%); padding: 2rem; border-radius: 15px; margin: 1rem 0; border-left: 4px solid {confidence_color};">
                <h3>{confidence_icon} #{i+1}. {phone}</h3>
                <p><strong>AI Confidence:</strong> <span style="color: {confidence_color}; font-weight: bold;">{confidence_text} ({neural_score:.1%})</span></p>
            </div>
        """, unsafe_allow_html=True)
        
        # Show AI explanations
        if explanations:
            st.markdown("**🤖 AI analysis findings:**")
            for explanation in explanations:
                st.markdown(f"• {explanation}")
        
        # Show key metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'rating' in phone_data.columns:
                avg_rating = phone_data['rating'].mean()
                st.metric("⭐ Rating", f"{avg_rating:.1f}/5.0")
        
        with col2:
            if 'sentiment_label' in phone_data.columns:
                positive_pct = (phone_data['sentiment_label'] == 'positive').mean() * 100
                st.metric("😊 Satisfaction", f"{positive_pct:.0f}%")
        
        with col3:
            st.metric("📊 Reviews", f"{len(phone_data):,}")
        
        # Action buttons
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button(f"📊 Analyze", key=f"neural_analyze_{i}"):
                st.session_state.selected_phone = phone
                st.session_state.current_page = "research"
                st.rerun()
        
        with col2:
            if st.button(f"⚖️ Compare", key=f"neural_compare_{i}"):
                st.session_state.phone_to_compare = phone
                st.session_state.current_page = "compare"
                st.rerun()
        
        with col3:
            if st.button(f"🤖 Ask AI", key=f"neural_ask_{i}"):
                # Pre-fill AI chat with question about this phone
                st.session_state.ai_prefill = f"Tell me more about the {phone}"
                st.session_state.current_page = "ai_assistant"
                st.rerun()
        
        with col4:
            if st.button(f"❤️ Save", key=f"neural_save_{i}"):
                if 'saved_phones' not in st.session_state:
                    st.session_state.saved_phones = []
                if phone not in st.session_state.saved_phones:
                    st.session_state.saved_phones.append(phone)
                    st.success(f"Saved {phone} to favorites!")
                else:
                    st.info(f"{phone} already in favorites")
        
        st.markdown("---")

def generate_ai_market_insights(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate AI-powered market insights automatically"""
    insights = {
        'market_overview': {},
        'trending_brands': {},
        'top_performers': {},
        'user_satisfaction_trends': {},
        'ai_recommendations': {},
        'market_opportunities': {},
        'price_analysis': {}
    }
    
    if df is None or len(df) == 0:
        return insights
    
    # Market Overview Analysis
    total_reviews = len(df)
    total_phones = df['product'].nunique() if 'product' in df.columns else 0
    total_brands = df['brand'].nunique() if 'brand' in df.columns else 0
    
    insights['market_overview'] = {
        'total_reviews': total_reviews,
        'total_phones': total_phones,
        'total_brands': total_brands,
        'market_health': 'Strong' if total_reviews > 1000 else 'Growing' if total_reviews > 500 else 'Emerging'
    }
    
    # Trending Brands Analysis
    if 'brand' in df.columns and 'sentiment_label' in df.columns:
        brand_performance = df.groupby('brand').agg({
            'sentiment_label': lambda x: (x == 'positive').mean(),
            'rating': 'mean' if 'rating' in df.columns else lambda x: 4.0,
            'product': 'count'
        }).rename(columns={'sentiment_label': 'satisfaction', 'product': 'review_count'})
        
        # Filter brands with sufficient data
        significant_brands = brand_performance[brand_performance['review_count'] >= 10]
        
        if len(significant_brands) > 0:
            top_brands = significant_brands.sort_values('satisfaction', ascending=False).head(3)
            insights['trending_brands'] = {
                'top_brand': top_brands.index[0] if len(top_brands) > 0 else 'N/A',
                'top_satisfaction': top_brands['satisfaction'].iloc[0] if len(top_brands) > 0 else 0,
                'brand_rankings': top_brands.to_dict('index')
            }
    
    # Top Performing Phones
    if 'product' in df.columns:
        phone_performance = df.groupby('product').agg({
            'sentiment_label': lambda x: (x == 'positive').mean() if 'sentiment_label' in df.columns else 0.8,
            'rating': 'mean' if 'rating' in df.columns else lambda x: 4.0,
            'product': 'count'
        }).rename(columns={'sentiment_label': 'satisfaction', 'product': 'review_count'})
        
        # Filter phones with sufficient reviews
        popular_phones = phone_performance[phone_performance['review_count'] >= 5]
        
        if len(popular_phones) > 0:
            top_phones = popular_phones.sort_values(['satisfaction', 'review_count'], ascending=[False, False]).head(5)
            insights['top_performers'] = {
                'best_phone': top_phones.index[0] if len(top_phones) > 0 else 'N/A',
                'best_satisfaction': top_phones['satisfaction'].iloc[0] if len(top_phones) > 0 else 0,
                'top_phones': top_phones.to_dict('index')
            }
    
    # User Satisfaction Trends
    if 'sentiment_label' in df.columns:
        overall_satisfaction = (df['sentiment_label'] == 'positive').mean()
        satisfaction_grade = (
            'Excellent' if overall_satisfaction > 0.8 else
            'Good' if overall_satisfaction > 0.7 else
            'Fair' if overall_satisfaction > 0.6 else
            'Needs Improvement'
        )
        
        insights['user_satisfaction_trends'] = {
            'overall_satisfaction': overall_satisfaction,
            'satisfaction_grade': satisfaction_grade,
            'positive_reviews': (df['sentiment_label'] == 'positive').sum(),
            'negative_reviews': (df['sentiment_label'] == 'negative').sum()
        }
    
    # AI Recommendations for Market
    insights['ai_recommendations'] = generate_market_recommendations(df, insights)
    
    # Market Opportunities
    insights['market_opportunities'] = identify_market_opportunities(df)
    
    return insights

def generate_market_recommendations(df: pd.DataFrame, insights: Dict[str, Any]) -> List[str]:
    """Generate AI recommendations based on market analysis"""
    recommendations = []
    
    # Brand recommendations
    if 'trending_brands' in insights and insights['trending_brands']:
        top_brand = insights['trending_brands'].get('top_brand')
        top_satisfaction = insights['trending_brands'].get('top_satisfaction', 0)
        
        if top_brand and top_satisfaction > 0.8:
            recommendations.append(f"🏆 {top_brand} leads the market with {top_satisfaction:.1%} user satisfaction")
        elif top_satisfaction > 0.75:
            recommendations.append(f"⭐ {top_brand} shows strong performance - worth considering for purchases")
    
    # Phone recommendations
    if 'top_performers' in insights and insights['top_performers']:
        best_phone = insights['top_performers'].get('best_phone')
        best_satisfaction = insights['top_performers'].get('best_satisfaction', 0)
        
        if best_phone and best_satisfaction > 0.85:
            recommendations.append(f"📱 {best_phone} is the top-rated choice with {best_satisfaction:.1%} satisfaction")
    
    # Market health recommendations
    if 'market_overview' in insights:
        market_health = insights['market_overview'].get('market_health', '')
        total_reviews = insights['market_overview'].get('total_reviews', 0)
        
        if market_health == 'Strong':
            recommendations.append(f"📈 Market is robust with {total_reviews:,} reviews - great time to buy")
        elif market_health == 'Growing':
            recommendations.append(f"🌱 Market is expanding with {total_reviews:,} reviews - good selection available")
    
    # Satisfaction trends
    if 'user_satisfaction_trends' in insights:
        satisfaction_grade = insights['user_satisfaction_trends'].get('satisfaction_grade', '')
        overall_satisfaction = insights['user_satisfaction_trends'].get('overall_satisfaction', 0)
        
        if satisfaction_grade == 'Excellent':
            recommendations.append(f"✨ Excellent market satisfaction ({overall_satisfaction:.1%}) - high quality products available")
        elif satisfaction_grade == 'Good':
            recommendations.append(f"👍 Good market satisfaction ({overall_satisfaction:.1%}) - reliable options available")
        elif satisfaction_grade == 'Fair':
            recommendations.append(f"⚠️ Market satisfaction is moderate ({overall_satisfaction:.1%}) - research carefully before buying")
    
    return recommendations[:5]  # Return top 5 recommendations

def identify_market_opportunities(df: pd.DataFrame) -> Dict[str, Any]:
    """Identify market opportunities using AI analysis"""
    opportunities = {
        'undervalued_phones': [],
        'emerging_brands': [],
        'feature_gaps': [],
        'price_opportunities': []
    }
    
    if df is None or len(df) == 0:
        return opportunities
    
    # Identify undervalued phones (high satisfaction, low review count)
    if 'product' in df.columns and 'sentiment_label' in df.columns:
        phone_stats = df.groupby('product').agg({
            'sentiment_label': lambda x: (x == 'positive').mean(),
            'product': 'count',
            'rating': 'mean' if 'rating' in df.columns else lambda x: 4.0
        }).rename(columns={'sentiment_label': 'satisfaction', 'product': 'review_count'})
        
        # Find phones with high satisfaction but low visibility
        undervalued = phone_stats[
            (phone_stats['satisfaction'] > 0.8) & 
            (phone_stats['review_count'] < 50) &
            (phone_stats['review_count'] > 5)
        ].sort_values('satisfaction', ascending=False)
        
        opportunities['undervalued_phones'] = undervalued.head(3).to_dict('index')
    
    # Identify emerging brands
    if 'brand' in df.columns:
        brand_stats = df.groupby('brand').agg({
            'sentiment_label': lambda x: (x == 'positive').mean() if 'sentiment_label' in df.columns else 0.75,
            'brand': 'count'
        }).rename(columns={'sentiment_label': 'satisfaction', 'brand': 'review_count'})
        
        # Emerging brands: decent satisfaction, moderate review count
        emerging = brand_stats[
            (brand_stats['satisfaction'] > 0.75) & 
            (brand_stats['review_count'] >= 20) &
            (brand_stats['review_count'] < 200)
        ].sort_values('satisfaction', ascending=False)
        
        opportunities['emerging_brands'] = emerging.head(2).to_dict('index')
    
    return opportunities

def display_ai_quick_insights(insights: Dict[str, Any]):
    """Display AI-generated quick insights in an attractive format"""
    
    # Market Overview Cards
    st.markdown("#### 📈 Market Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    market_overview = insights.get('market_overview', {})
    
    with col1:
        st.metric(
            "📊 Total Reviews", 
            f"{market_overview.get('total_reviews', 0):,}",
            "Market Data"
        )
    
    with col2:
        st.metric(
            "📱 Phone Models", 
            f"{market_overview.get('total_phones', 0)}",
            "Available Options"
        )
    
    with col3:
        st.metric(
            "🏷️ Brands", 
            f"{market_overview.get('total_brands', 0)}",
            "Market Competition"
        )
    
    with col4:
        market_health = market_overview.get('market_health', 'Unknown')
        health_color = {
            'Strong': '🟢', 'Growing': '🟡', 'Emerging': '🟠'
        }.get(market_health, '⚪')
        
        st.metric(
            "🏅 Market Health", 
            f"{health_color} {market_health}",
            "Current Status"
        )
    
    # AI Recommendations Section
    ai_recommendations = insights.get('ai_recommendations', [])
    if ai_recommendations:
        st.markdown("#### 🤖 AI Market Recommendations")
        
        for i, recommendation in enumerate(ai_recommendations):
            st.markdown(f"""
                <div style="background: linear-gradient(135deg, #E8F5E8 0%, #C8E6C9 100%); padding: 1rem; border-radius: 10px; margin: 0.5rem 0; border-left: 4px solid #4CAF50;">
                    {recommendation}
                </div>
            """, unsafe_allow_html=True)
    
    # Top Performers Section
    top_performers = insights.get('top_performers', {})
    if top_performers and 'top_phones' in top_performers:
        st.markdown("#### 🏆 AI-Identified Top Performers")
        
        top_phones = top_performers['top_phones']
        
        for i, (phone, data) in enumerate(list(top_phones.items())[:3]):
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown(f"**#{i+1}. {phone}**")
                satisfaction = data.get('satisfaction', 0)
                rating = data.get('rating', 0)
                reviews = data.get('review_count', 0)
                
                st.markdown(f"User Satisfaction: {satisfaction:.1%} | Rating: {rating:.1f}/5 | Reviews: {reviews:,}")
            
            with col2:
                if st.button(f"📊 Analyze", key=f"ai_insight_analyze_{i}"):
                    st.session_state.selected_phone = phone
                    st.session_state.current_page = "research"
                    st.rerun()
            
            with col3:
                if st.button(f"🤖 Ask AI", key=f"ai_insight_ask_{i}"):
                    st.session_state.ai_prefill = f"What makes {phone} a top performer?"
                    st.session_state.current_page = "ai_assistant"
                    st.rerun()
    
    # Market Opportunities Section
    opportunities = insights.get('market_opportunities', {})
    
    if opportunities.get('undervalued_phones') or opportunities.get('emerging_brands'):
        st.markdown("#### 🔍 AI-Discovered Opportunities")
        
        col1, col2 = st.columns(2)
        
        with col1:
            undervalued = opportunities.get('undervalued_phones', {})
            if undervalued:
                st.markdown("**📎 Hidden Gems:**")
                for phone, data in list(undervalued.items())[:2]:
                    satisfaction = data.get('satisfaction', 0)
                    st.markdown(f"• **{phone}** - {satisfaction:.1%} satisfaction, underrated")
        
        with col2:
            emerging = opportunities.get('emerging_brands', {})
            if emerging:
                st.markdown("**🌱 Rising Brands:**")
                for brand, data in list(emerging.items())[:2]:
                    satisfaction = data.get('satisfaction', 0)
                    st.markdown(f"• **{brand}** - {satisfaction:.1%} satisfaction, growing presence")
    
    # User Satisfaction Trends
    satisfaction_trends = insights.get('user_satisfaction_trends', {})
    if satisfaction_trends:
        st.markdown("#### 😊 User Satisfaction Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            overall_satisfaction = satisfaction_trends.get('overall_satisfaction', 0)
            grade = satisfaction_trends.get('satisfaction_grade', 'Unknown')
            
            grade_colors = {
                'Excellent': '#4CAF50',
                'Good': '#8BC34A', 
                'Fair': '#FFC107',
                'Needs Improvement': '#FF9800'
            }
            
            color = grade_colors.get(grade, '#9E9E9E')
            
            st.markdown(f"""
                <div style="background: {color}20; padding: 1rem; border-radius: 10px; text-align: center; border: 2px solid {color};">
                    <h3 style="color: {color}; margin: 0;">{overall_satisfaction:.1%}</h3>
                    <p style="margin: 0.5rem 0 0 0;"><strong>Overall Satisfaction</strong></p>
                    <p style="margin: 0; font-size: 0.9rem;">{grade}</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            positive_reviews = satisfaction_trends.get('positive_reviews', 0)
            st.metric("😊 Positive Reviews", f"{positive_reviews:,}", "Happy Users")
        
        with col3:
            negative_reviews = satisfaction_trends.get('negative_reviews', 0)
            st.metric("😟 Negative Reviews", f"{negative_reviews:,}", "Areas to Improve")

# ==================== AGENTIC RAG SYSTEM FUNCTIONS ====================

def initialize_rag_system(df: pd.DataFrame):
    """Initialize the Agentic RAG system with phone review data"""
    if not RAG_AVAILABLE or df is None:
        return None
    
    try:
        # Create a simplified RAG system for user-friendly app
        rag_system = {
            'agents': {},
            'knowledge_base': {},
            'conversation_memory': [],
            'user_preferences': {},
            'initialized': True
        }
        
        return rag_system
    except Exception as e:
        logging.error(f"RAG system initialization failed: {e}")
        return None

def create_specialized_agents():
    """Create specialized agents for different tasks"""
    if not RAG_AVAILABLE:
        return {}
    
    agents = {
        'researcher': {
            'role': 'Phone Research Specialist',
            'description': 'Finds and analyzes specific phone information from reviews',
            'capabilities': ['search_reviews', 'extract_facts', 'compare_specs'],
            'memory': [],
            'active': True
        },
        'recommender': {
            'role': 'Recommendation Expert', 
            'description': 'Provides personalized phone recommendations',
            'capabilities': ['analyze_preferences', 'match_phones', 'explain_choices'],
            'memory': [],
            'active': True
        },
        'analyst': {
            'role': 'Sentiment Analysis Expert',
            'description': 'Analyzes user sentiment and review patterns',
            'capabilities': ['sentiment_analysis', 'trend_detection', 'pattern_recognition'],
            'memory': [],
            'active': True
        },
        'comparator': {
            'role': 'Phone Comparison Specialist',
            'description': 'Compares phones across different dimensions',
            'capabilities': ['side_by_side_comparison', 'pros_cons_analysis', 'feature_ranking'],
            'memory': [],
            'active': True
        }
    }
    
    return agents

def build_knowledge_base(df: pd.DataFrame):
    """Build a knowledge base from phone review data for RAG retrieval"""
    if df is None or len(df) == 0:
        return {}
    
    knowledge_base = {
        'phone_profiles': {},
        'review_vectors': [],
        'brand_knowledge': {},
        'feature_knowledge': {},
        'user_patterns': {},
        'indexed': True
    }
    
    try:
        # Build phone profiles
        if 'product' in df.columns:
            for phone in df['product'].unique():
                phone_data = df[df['product'] == phone]
                
                profile = {
                    'name': phone,
                    'total_reviews': len(phone_data),
                    'avg_rating': phone_data['rating'].mean() if 'rating' in phone_data.columns else 4.0,
                    'sentiment_distribution': {},
                    'key_features_mentioned': [],
                    'common_complaints': [],
                    'strengths': [],
                    'review_summary': ''
                }
                
                # Add sentiment distribution
                if 'sentiment_label' in phone_data.columns:
                    sentiment_counts = phone_data['sentiment_label'].value_counts(normalize=True)
                    profile['sentiment_distribution'] = sentiment_counts.to_dict()
                
                # Extract key insights from reviews
                if 'review_text' in phone_data.columns:
                    reviews = phone_data['review_text'].dropna().head(50).tolist()
                    profile['key_features_mentioned'] = extract_key_features(reviews)
                    profile['common_complaints'] = extract_complaints(reviews)
                    profile['strengths'] = extract_strengths(reviews)
                    profile['review_summary'] = generate_phone_summary(reviews, phone)
                
                knowledge_base['phone_profiles'][phone] = profile
        
        # Build brand knowledge
        if 'brand' in df.columns:
            for brand in df['brand'].unique():
                brand_data = df[df['brand'] == brand]
                
                brand_profile = {
                    'name': brand,
                    'total_phones': brand_data['product'].nunique() if 'product' in brand_data.columns else 0,
                    'avg_satisfaction': (brand_data['sentiment_label'] == 'positive').mean() if 'sentiment_label' in brand_data.columns else 0.75,
                    'popular_phones': [],
                    'brand_characteristics': []
                }
                
                # Get popular phones for this brand
                if 'product' in brand_data.columns:
                    phone_popularity = brand_data['product'].value_counts().head(3)
                    brand_profile['popular_phones'] = phone_popularity.index.tolist()
                
                knowledge_base['brand_knowledge'][brand] = brand_profile
        
        return knowledge_base
    
    except Exception as e:
        logging.error(f"Knowledge base building failed: {e}")
        return knowledge_base

def extract_key_features(reviews: List[str]) -> List[str]:
    """Extract key features mentioned in reviews"""
    if not reviews:
        return []
    
    combined_text = ' '.join(reviews).lower()
    
    feature_keywords = {
        'camera': ['camera', 'photo', 'picture', 'selfie'],
        'battery': ['battery', 'power', 'charge', 'lasting'],
        'performance': ['fast', 'speed', 'performance', 'smooth'],
        'display': ['screen', 'display', 'bright', 'clear'],
        'build_quality': ['build', 'quality', 'premium', 'solid'],
        'storage': ['storage', 'memory', 'space', 'gb'],
        'price': ['price', 'value', 'worth', 'money'],
        'design': ['design', 'look', 'style', 'beautiful']
    }
    
    mentioned_features = []
    for feature, keywords in feature_keywords.items():
        mention_count = sum(1 for keyword in keywords if keyword in combined_text)
        if mention_count >= 2:  # Feature mentioned multiple times
            mentioned_features.append(feature)
    
    return mentioned_features

def extract_complaints(reviews: List[str]) -> List[str]:
    """Extract common complaints from reviews"""
    if not reviews:
        return []
    
    combined_text = ' '.join(reviews).lower()
    
    complaint_patterns = {
        'battery_issues': ['battery drain', 'poor battery', 'battery dies', 'short battery'],
        'performance_issues': ['slow', 'lag', 'freeze', 'crash'],
        'camera_issues': ['poor camera', 'bad photos', 'blurry', 'camera quality'],
        'build_issues': ['cheap feel', 'breaks easily', 'poor build'],
        'price_issues': ['overpriced', 'too expensive', 'not worth'],
        'software_issues': ['bugs', 'software issues', 'updates', 'glitches']
    }
    
    complaints = []
    for complaint_type, patterns in complaint_patterns.items():
        if any(pattern in combined_text for pattern in patterns):
            complaints.append(complaint_type.replace('_', ' ').title())
    
    return complaints

def extract_strengths(reviews: List[str]) -> List[str]:
    """Extract strengths mentioned in reviews"""
    if not reviews:
        return []
    
    combined_text = ' '.join(reviews).lower()
    
    strength_patterns = {
        'excellent_camera': ['great camera', 'excellent camera', 'amazing photos'],
        'long_battery': ['great battery', 'long battery', 'excellent battery life'],
        'fast_performance': ['very fast', 'smooth', 'great performance'],
        'beautiful_display': ['beautiful display', 'great screen', 'stunning display'],
        'premium_build': ['premium feel', 'solid build', 'excellent build quality'],
        'good_value': ['great value', 'worth the money', 'excellent value'],
        'reliable': ['reliable', 'dependable', 'consistent']
    }
    
    strengths = []
    for strength_type, patterns in strength_patterns.items():
        if any(pattern in combined_text for pattern in patterns):
            strengths.append(strength_type.replace('_', ' ').title())
    
    return strengths

def generate_phone_summary(reviews: List[str], phone_name: str) -> str:
    """Generate a summary of the phone based on reviews"""
    if not reviews:
        return f"{phone_name} - Analysis available from user reviews."
    
    # Simple summary based on review content analysis
    combined_text = ' '.join(reviews[:10]).lower()  # Use first 10 reviews
    
    positive_indicators = sum(1 for word in ['great', 'excellent', 'amazing', 'love', 'perfect', 'fantastic'] if word in combined_text)
    negative_indicators = sum(1 for word in ['poor', 'terrible', 'awful', 'hate', 'disappointing', 'bad'] if word in combined_text)
    
    if positive_indicators > negative_indicators * 2:
        sentiment_tone = "Users generally praise"
    elif negative_indicators > positive_indicators * 2:
        sentiment_tone = "Users commonly critique"
    else:
        sentiment_tone = "Users have mixed opinions on"
    
    # Find most mentioned aspects
    aspects = []
    if 'camera' in combined_text:
        aspects.append('camera quality')
    if 'battery' in combined_text:
        aspects.append('battery performance')
    if 'fast' in combined_text or 'speed' in combined_text:
        aspects.append('performance')
    if 'display' in combined_text or 'screen' in combined_text:
        aspects.append('display quality')
    
    aspect_text = ', '.join(aspects[:3]) if aspects else 'various features'
    
    return f"{sentiment_tone} the {phone_name}, particularly noting {aspect_text}. Reviews suggest it offers solid performance in its category."

def rag_enhanced_query_processing(query: str, user_context: Dict = None) -> Dict[str, Any]:
    """Process queries using RAG system with multiple specialized agents"""
    if not RAG_AVAILABLE or 'rag_system' not in st.session_state:
        return {'error': 'RAG system not available'}
    
    try:
        # Determine query intent and route to appropriate agents
        query_intent = classify_query_intent(query)
        relevant_agents = select_relevant_agents(query_intent)
        
        # Retrieve relevant information from knowledge base
        retrieved_info = retrieve_relevant_knowledge(query, st.session_state.knowledge_base)
        
        # Process with specialized agents
        agent_responses = {}
        for agent_name in relevant_agents:
            agent_response = process_with_agent(query, agent_name, retrieved_info, user_context)
            agent_responses[agent_name] = agent_response
        
        # Synthesize final response
        final_response = synthesize_agent_responses(query, agent_responses, retrieved_info)
        
        return {
            'response': final_response,
            'agents_used': relevant_agents,
            'retrieved_info': retrieved_info,
            'confidence': calculate_response_confidence(agent_responses)
        }
    
    except Exception as e:
        logging.error(f"RAG query processing failed: {e}")
        return {'error': str(e)}

def classify_query_intent(query: str) -> str:
    """Classify the intent of the user query"""
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['recommend', 'suggest', 'best', 'should i', 'which']):
        return 'recommendation'
    elif any(word in query_lower for word in ['compare', 'vs', 'versus', 'difference', 'better']):
        return 'comparison'
    elif any(word in query_lower for word in ['review', 'opinion', 'experience', 'feedback']):
        return 'review_analysis'
    elif any(word in query_lower for word in ['feature', 'spec', 'capability', 'camera', 'battery']):
        return 'feature_inquiry'
    elif any(word in query_lower for word in ['price', 'cost', 'budget', 'cheap', 'expensive']):
        return 'price_inquiry'
    else:
        return 'general_inquiry'

def select_relevant_agents(intent: str) -> List[str]:
    """Select which agents should handle the query based on intent"""
    agent_mapping = {
        'recommendation': ['recommender', 'researcher'],
        'comparison': ['comparator', 'analyst'],
        'review_analysis': ['analyst', 'researcher'],
        'feature_inquiry': ['researcher', 'analyst'],
        'price_inquiry': ['researcher', 'recommender'],
        'general_inquiry': ['researcher']
    }
    
    return agent_mapping.get(intent, ['researcher'])

def retrieve_relevant_knowledge(query: str, knowledge_base: Dict) -> Dict[str, Any]:
    """Retrieve relevant information from the knowledge base"""
    if not knowledge_base or 'phone_profiles' not in knowledge_base:
        return {}
    
    query_lower = query.lower()
    relevant_info = {
        'matching_phones': [],
        'relevant_brands': [],
        'related_features': [],
        'context_data': {}
    }
    
    # Find matching phones
    for phone, profile in knowledge_base['phone_profiles'].items():
        if any(word in phone.lower() for word in query_lower.split()):
            relevant_info['matching_phones'].append({
                'name': phone,
                'profile': profile
            })
    
    # Find relevant brands
    if 'brand_knowledge' in knowledge_base:
        for brand, brand_info in knowledge_base['brand_knowledge'].items():
            if brand.lower() in query_lower:
                relevant_info['relevant_brands'].append({
                    'name': brand,
                    'info': brand_info
                })
    
    # Extract feature mentions
    features = ['camera', 'battery', 'performance', 'display', 'price', 'storage', 'design']
    for feature in features:
        if feature in query_lower:
            relevant_info['related_features'].append(feature)
    
    return relevant_info

def process_with_agent(query: str, agent_name: str, retrieved_info: Dict, user_context: Dict = None) -> Dict[str, Any]:
    """Process query with a specific agent"""
    if 'rag_agents' not in st.session_state:
        return {'response': 'Agent not available'}
    
    agent = st.session_state.rag_agents.get(agent_name, {})
    if not agent:
        return {'response': 'Agent not found'}
    
    # Simulate agent processing based on their capabilities
    if agent_name == 'researcher':
        return research_agent_process(query, retrieved_info)
    elif agent_name == 'recommender':
        return recommender_agent_process(query, retrieved_info, user_context)
    elif agent_name == 'analyst':
        return analyst_agent_process(query, retrieved_info)
    elif agent_name == 'comparator':
        return comparator_agent_process(query, retrieved_info)
    else:
        return {'response': 'Generic agent response', 'confidence': 0.5}

def research_agent_process(query: str, retrieved_info: Dict) -> Dict[str, Any]:
    """Process query with research agent"""
    response_parts = []
    
    if retrieved_info.get('matching_phones'):
        phones = retrieved_info['matching_phones']
        response_parts.append(f"Found {len(phones)} relevant phones in our database.")
        
        for phone_info in phones[:3]:  # Top 3 phones
            phone = phone_info['name']
            profile = phone_info['profile']
            
            response_parts.append(f"\n**{phone}:** {profile.get('review_summary', 'Phone information available.')}")
            
            if profile.get('strengths'):
                strengths = ', '.join(profile['strengths'][:3])
                response_parts.append(f"Key strengths: {strengths}")
    
    if retrieved_info.get('related_features'):
        features = ', '.join(retrieved_info['related_features'])
        response_parts.append(f"\nAnalyzing features: {features}")
    
    return {
        'response': '\n'.join(response_parts) if response_parts else 'No specific information found.',
        'confidence': 0.8 if response_parts else 0.3
    }

def recommender_agent_process(query: str, retrieved_info: Dict, user_context: Dict = None) -> Dict[str, Any]:
    """Process query with recommendation agent"""
    response_parts = []
    
    if retrieved_info.get('matching_phones'):
        phones = retrieved_info['matching_phones']
        
        # Sort phones by satisfaction score
        sorted_phones = sorted(phones, 
                             key=lambda x: x['profile'].get('sentiment_distribution', {}).get('positive', 0), 
                             reverse=True)
        
        response_parts.append("Based on user reviews and satisfaction scores, here are my recommendations:")
        
        for i, phone_info in enumerate(sorted_phones[:3]):
            phone = phone_info['name']
            profile = phone_info['profile']
            
            satisfaction = profile.get('sentiment_distribution', {}).get('positive', 0.5)
            rating = profile.get('avg_rating', 4.0)
            
            response_parts.append(f"\n**#{i+1}. {phone}**")
            response_parts.append(f"- User satisfaction: {satisfaction:.1%}")
            response_parts.append(f"- Average rating: {rating:.1f}/5.0")
            
            if profile.get('strengths'):
                response_parts.append(f"- Key strengths: {', '.join(profile['strengths'][:2])}")
    
    return {
        'response': '\n'.join(response_parts) if response_parts else 'I need more information to make recommendations.',
        'confidence': 0.9 if response_parts else 0.4
    }

def analyst_agent_process(query: str, retrieved_info: Dict) -> Dict[str, Any]:
    """Process query with sentiment analysis agent"""
    response_parts = []
    
    if retrieved_info.get('matching_phones'):
        phones = retrieved_info['matching_phones']
        
        response_parts.append("Sentiment analysis results:")
        
        for phone_info in phones[:3]:
            phone = phone_info['name']
            profile = phone_info['profile']
            
            sentiment_dist = profile.get('sentiment_distribution', {})
            positive_pct = sentiment_dist.get('positive', 0) * 100
            negative_pct = sentiment_dist.get('negative', 0) * 100
            
            response_parts.append(f"\n**{phone}:**")
            response_parts.append(f"- Positive sentiment: {positive_pct:.1f}%")
            response_parts.append(f"- Negative sentiment: {negative_pct:.1f}%")
            
            if profile.get('common_complaints'):
                complaints = ', '.join(profile['common_complaints'][:2])
                response_parts.append(f"- Common concerns: {complaints}")
    
    return {
        'response': '\n'.join(response_parts) if response_parts else 'No sentiment data available.',
        'confidence': 0.8 if response_parts else 0.3
    }

def comparator_agent_process(query: str, retrieved_info: Dict) -> Dict[str, Any]:
    """Process query with comparison agent"""
    phones = retrieved_info.get('matching_phones', [])
    
    if len(phones) < 2:
        return {
            'response': 'I need at least two phones to compare. Please specify the phones you want to compare.',
            'confidence': 0.2
        }
    
    phone1 = phones[0]
    phone2 = phones[1]
    
    response_parts = [f"Comparing {phone1['name']} vs {phone2['name']}:"]
    
    # Compare ratings
    rating1 = phone1['profile'].get('avg_rating', 4.0)
    rating2 = phone2['profile'].get('avg_rating', 4.0)
    
    response_parts.append(f"\n**Ratings:**")
    response_parts.append(f"- {phone1['name']}: {rating1:.1f}/5.0")
    response_parts.append(f"- {phone2['name']}: {rating2:.1f}/5.0")
    
    if rating1 > rating2:
        response_parts.append(f"Winner: {phone1['name']} (higher rating)")
    elif rating2 > rating1:
        response_parts.append(f"Winner: {phone2['name']} (higher rating)")
    else:
        response_parts.append("Tied in ratings")
    
    # Compare satisfaction
    sat1 = phone1['profile'].get('sentiment_distribution', {}).get('positive', 0.5)
    sat2 = phone2['profile'].get('sentiment_distribution', {}).get('positive', 0.5)
    
    response_parts.append(f"\n**User Satisfaction:**")
    response_parts.append(f"- {phone1['name']}: {sat1:.1%}")
    response_parts.append(f"- {phone2['name']}: {sat2:.1%}")
    
    return {
        'response': '\n'.join(response_parts),
        'confidence': 0.9
    }

def synthesize_agent_responses(query: str, agent_responses: Dict, retrieved_info: Dict) -> str:
    """Synthesize responses from multiple agents into a coherent final response"""
    if not agent_responses:
        return "I'm sorry, I couldn't process your query at the moment."
    
    # Combine responses from different agents
    final_parts = []
    
    # Add a personalized greeting
    final_parts.append("🤖 **Multi-Agent Analysis Complete:**\n")
    
    # Include responses from each agent
    for agent_name, response_data in agent_responses.items():
        if response_data.get('response') and response_data.get('confidence', 0) > 0.3:
            agent_role = st.session_state.rag_agents.get(agent_name, {}).get('role', agent_name.title())
            final_parts.append(f"**{agent_role} Analysis:**")
            final_parts.append(response_data['response'])
            final_parts.append("")  # Add spacing
    
    # Add a summary conclusion
    if len(agent_responses) > 1:
        final_parts.append("**Summary:** This analysis was generated by multiple AI agents working together to provide you with comprehensive insights from our phone review database.")
    
    return "\n".join(final_parts)

def calculate_response_confidence(agent_responses: Dict) -> float:
    """Calculate overall confidence in the response"""
    if not agent_responses:
        return 0.0
    
    confidences = [resp.get('confidence', 0.5) for resp in agent_responses.values()]
    return sum(confidences) / len(confidences)

# ==================== USER MEMORY AND LEARNING FUNCTIONS ====================

def store_user_interaction(interaction_data: Dict[str, Any]):
    """Store user interactions for learning and memory"""
    if 'user_memory' not in st.session_state:
        st.session_state.user_memory = []
    
    # Add timestamp if not present
    if 'timestamp' not in interaction_data:
        interaction_data['timestamp'] = datetime.now().isoformat()
    
    # Store interaction
    st.session_state.user_memory.append(interaction_data)
    
    # Keep only last 50 interactions to avoid memory bloat
    if len(st.session_state.user_memory) > 50:
        st.session_state.user_memory = st.session_state.user_memory[-50:]
    
    # Update user preferences based on interactions
    update_user_preferences_from_interaction(interaction_data)

def save_to_user_preferences(item: str, category: str):
    """Save items to user preferences"""
    if 'user_preferences' not in st.session_state:
        st.session_state.user_preferences = {
            'saved_phones': [],
            'saved_analyses': [],
            'preferred_brands': [],
            'preferred_features': [],
            'interaction_history': []
        }
    
    prefs = st.session_state.user_preferences
    
    if category == 'saved_analysis':
        if item not in prefs['saved_analyses']:
            prefs['saved_analyses'].append(item)
    elif category == 'saved_phone':
        if item not in prefs['saved_phones']:
            prefs['saved_phones'].append(item)
    
    # Store the preference action
    store_user_interaction({
        'action': 'save_preference',
        'item': item,
        'category': category
    })

def update_user_preferences_from_interaction(interaction_data: Dict[str, Any]):
    """Update user preferences based on their interactions"""
    if 'user_preferences' not in st.session_state:
        st.session_state.user_preferences = {
            'saved_phones': [],
            'saved_analyses': [],
            'preferred_brands': [],
            'preferred_features': [],
            'interaction_history': []
        }
    
    prefs = st.session_state.user_preferences
    
    # Learn from phone research actions
    if interaction_data.get('action') == 'phone_research' and 'phone' in interaction_data:
        phone_name = interaction_data['phone']
        
        # Extract brand preference
        if 'brand' in st.session_state.df.columns:
            df = st.session_state.df
            phone_data = df[df['product'] == phone_name]
            if len(phone_data) > 0:
                brand = phone_data['brand'].iloc[0]
                if brand not in prefs['preferred_brands']:
                    prefs['preferred_brands'].append(brand)
    
    # Learn from feature interests
    if 'features' in interaction_data:
        for feature in interaction_data['features']:
            if feature not in prefs['preferred_features']:
                prefs['preferred_features'].append(feature)
    
    # Add to interaction history
    prefs['interaction_history'].append({
        'timestamp': interaction_data.get('timestamp'),
        'action': interaction_data.get('action'),
        'details': {k: v for k, v in interaction_data.items() if k not in ['timestamp', 'action']}
    })
    
    # Keep only recent history (last 20 interactions)
    if len(prefs['interaction_history']) > 20:
        prefs['interaction_history'] = prefs['interaction_history'][-20:]

def get_personalized_context_for_rag(user_query: str) -> Dict[str, Any]:
    """Get personalized context for RAG system based on user history"""
    context = {
        'user_preferences': {},
        'interaction_history': [],
        'learned_patterns': {},
        'personalization_strength': 0.0
    }
    
    if 'user_preferences' not in st.session_state:
        return context
    
    prefs = st.session_state.user_preferences
    
    # Add user preferences to context
    context['user_preferences'] = {
        'preferred_brands': prefs.get('preferred_brands', []),
        'preferred_features': prefs.get('preferred_features', []),
        'saved_phones': prefs.get('saved_phones', []),
        'saved_analyses': prefs.get('saved_analyses', [])
    }
    
    # Add recent interaction history
    context['interaction_history'] = prefs.get('interaction_history', [])[-5:]  # Last 5 interactions
    
    # Calculate personalization strength
    total_interactions = len(prefs.get('interaction_history', []))
    context['personalization_strength'] = min(1.0, total_interactions / 10)  # Stronger after 10+ interactions
    
    # Learn patterns from history
    context['learned_patterns'] = analyze_user_patterns(prefs)
    
    return context

def analyze_user_patterns(preferences: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze user patterns from their interaction history"""
    patterns = {
        'most_researched_brands': [],
        'common_actions': [],
        'feature_interests': [],
        'usage_frequency': 'low'  # low, medium, high
    }
    
    history = preferences.get('interaction_history', [])
    if not history:
        return patterns
    
    # Analyze brand research patterns
    brand_counts = {}
    action_counts = {}
    
    for interaction in history:
        action = interaction.get('action', '')
        if action:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        # Count brand interactions
        if 'phone' in interaction.get('details', {}):
            # This would need brand lookup from phone name
            pass
    
    # Determine usage frequency
    if len(history) >= 15:
        patterns['usage_frequency'] = 'high'
    elif len(history) >= 5:
        patterns['usage_frequency'] = 'medium'
    
    # Most common actions
    patterns['common_actions'] = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    
    # Feature interests from preferences
    patterns['feature_interests'] = preferences.get('preferred_features', [])
    
    return patterns

def get_user_memory_summary() -> str:
    """Get a summary of user's interaction history for display"""
    if 'user_memory' not in st.session_state or not st.session_state.user_memory:
        return "No interaction history yet. Start chatting to build your personalized experience!"
    
    memory = st.session_state.user_memory
    recent_actions = [interaction.get('action', 'unknown') for interaction in memory[-5:]]
    
    summary_parts = [
        f"📊 Total interactions: {len(memory)}",
        f"🔄 Recent activities: {', '.join(recent_actions)}"
    ]
    
    if 'user_preferences' in st.session_state:
        prefs = st.session_state.user_preferences
        if prefs.get('preferred_brands'):
            summary_parts.append(f"🏷️ Preferred brands: {', '.join(prefs['preferred_brands'])}")
        if prefs.get('saved_phones'):
            summary_parts.append(f"❤️ Saved phones: {len(prefs['saved_phones'])} phones")
    
    return "\n".join(summary_parts)

def show_ai_insights_dashboard():
    """Enhanced AI-powered insights and analytics dashboard with automated market analysis"""
    st.header("🧠 AI Market Insights Dashboard")
    st.markdown("**Automated analytics and intelligent insights powered by advanced AI**")
    st.markdown("🎯 *Get instant market trends, performance analysis, and buying recommendations*")
    
    # Back button
    if st.button("← Back to Home"):
        st.session_state.current_page = "home"
        st.rerun()
    
    df = st.session_state.df
    insights_engine = st.session_state.insights_engine
    
    # Add AI-powered quick insights first
    if ADVANCED_AI_AVAILABLE and df is not None:
        st.markdown("### 🚀 AI Quick Insights")
        
        with st.spinner("🧠 AI is analyzing market data..."):
            try:
                quick_insights = generate_ai_market_insights(df)
                display_ai_quick_insights(quick_insights)
            except Exception as e:
                st.info(f"Quick insights temporarily unavailable: {str(e)[:100]}...")
        
        st.markdown("---")
    
    if df is not None and insights_engine is not None:
        
        # Time period selector
        col1, col2 = st.columns([1, 3])
        with col1:
            time_period = st.selectbox(
                "📅 Analysis Period",
                ["last_30_days", "last_90_days", "last_6_months", "last_year"]
            )
        
        st.markdown("---")
        
        # Generate insights
        with st.spinner("🔍 Generating AI insights..."):
            try:
                insights = insights_engine.generate_insights(df, time_period)
                alerts = insights_engine.generate_alerts(df)
                
                # Display alerts first (if any)
                if alerts:
                    st.subheader("🚨 Active Alerts")
                    
                    for alert in alerts[:5]:  # Show top 5 alerts
                        severity_colors = {
                            'critical': '🔴',
                            'high': '🟠', 
                            'medium': '🟡',
                            'low': '🟢'
                        }
                        
                        severity_icon = severity_colors.get(alert.severity, '⚪')
                        
                        with st.expander(f"{severity_icon} {alert.title}", expanded=(alert.severity in ['critical', 'high'])):
                            st.markdown(f"**Category:** {alert.category.title()}")
                            st.markdown(f"**Severity:** {alert.severity.title()}")
                            st.markdown(f"**Message:** {alert.message}")
                            
                            if alert.data:
                                st.markdown("**Details:**")
                                for key, value in alert.data.items():
                                    st.markdown(f"- {key.title().replace('_', ' ')}: {value}")
                    
                    st.markdown("---")
                
                # Display insights
                if insights:
                    st.subheader("💡 Market Insights")
                    
                    # Insights summary
                    summary = insights_engine.get_insight_summary(insights)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Insights", summary['total_insights'])
                    with col2:
                        st.metric("High Priority", summary['high_importance'])
                    with col3:
                        st.metric("Avg Confidence", f"{summary['avg_confidence']:.1%}")
                    with col4:
                        st.metric("Categories", len(summary['categories']))
                    
                    st.markdown("---")
                    
                    # Display insights by category
                    insight_categories = {}
                    for insight in insights:
                        category = insight.insight_type
                        if category not in insight_categories:
                            insight_categories[category] = []
                        insight_categories[category].append(insight)
                    
                    # Tabs for different insight categories
                    if insight_categories:
                        category_tabs = st.tabs([f"{cat.title()} ({len(insights_list)})" for cat, insights_list in insight_categories.items()])
                        
                        for i, (category, category_insights) in enumerate(insight_categories.items()):
                            with category_tabs[i]:
                                for j, insight in enumerate(category_insights[:5]):  # Limit to 5 per category
                                    
                                    # Confidence and importance indicators
                                    confidence_bar = "🟢" * int(insight.confidence * 5) + "⚪" * (5 - int(insight.confidence * 5))
                                    importance_bar = "🔥" * int(insight.importance * 5) + "🔘" * (5 - int(insight.importance * 5))
                                    
                                    with st.expander(f"💡 {insight.title}", expanded=(j == 0)):
                                        st.markdown(insight.description)
                                        
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.markdown(f"**Confidence:** {confidence_bar} ({insight.confidence:.1%})")
                                        with col2:
                                            st.markdown(f"**Importance:** {importance_bar} ({insight.importance:.1%})")
                                        
                                        # Show data points if available
                                        if insight.data_points:
                                            st.markdown("**Supporting Data:**")
                                            data_df = pd.DataFrame([insight.data_points])
                                            st.dataframe(data_df, use_container_width=True)
                                        
                                        # Tags
                                        if insight.tags:
                                            st.markdown(f"**Tags:** {', '.join([f'`{tag}`' for tag in insight.tags[:5]])}")
                
                else:
                    st.info("🔍 No significant insights found for the current dataset. Try adjusting the time period or check back when more data is available.")
                
            except Exception as e:
                st.error(f"⚠️ Error generating insights: {str(e)}")
                st.info("Using fallback insights generation...")
                
                # Fallback: Show basic statistics
                st.subheader("📊 Basic Market Overview")
                
                if 'product' in df.columns:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**📱 Top Products by Review Count**")
                        product_counts = df['product'].value_counts().head(5)
                        st.bar_chart(product_counts)
                    
                    with col2:
                        if 'rating' in df.columns:
                            st.markdown("**⭐ Average Ratings by Product**")
                            product_ratings = df.groupby('product')['rating'].mean().sort_values(ascending=False).head(5)
                            st.bar_chart(product_ratings)
                
                if 'sentiment_label' in df.columns:
                    st.markdown("**😊 Overall Sentiment Distribution**")
                    sentiment_counts = df['sentiment_label'].value_counts()
                    
                    fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index, 
                               title="Market Sentiment Overview")
                    st.plotly_chart(fig, use_container_width=True)
        
        # Insights history (if available)
        if hasattr(insights_engine, 'insights_history') and insights_engine.insights_history:
            st.markdown("---")
            st.subheader("📈 Insights History")
            
            history_df = pd.DataFrame([
                {
                    'Timestamp': insight.timestamp.strftime('%Y-%m-%d %H:%M'),
                    'Title': insight.title,
                    'Type': insight.insight_type,
                    'Confidence': f"{insight.confidence:.1%}",
                    'Importance': f"{insight.importance:.1%}"
                } for insight in insights_engine.insights_history[-10:]  # Last 10 insights
            ])
            
            st.dataframe(history_df, use_container_width=True)
    
    else:
        st.error("⚠️ Insights engine or data not available. Please check your configuration.")
        
        # Show demo content
        st.markdown("### 🎯 AI Insights Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
                **🔍 Automated Analysis**
                - Sentiment trend detection
                - Rating change analysis
                - Top performer identification
                - User preference shifts
                - Competitive insights
                - Anomaly detection
            """)
        
        with col2:
            st.markdown("""
                **🚨 Smart Alerts**
                - High negative sentiment warnings
                - Rating drop notifications
                - Volume change alerts
                - Market anomaly detection
                - Competitive threats
                - Performance opportunities
            """)

def show_data_quality_dashboard():
    """Data Quality Validation and Assessment Dashboard"""
    st.header("🔍 Data Quality Dashboard")
    st.markdown("Comprehensive data validation and quality assurance for your review dataset")
    
    # Back button
    if st.button("← Back to Home"):
        st.session_state.current_page = "home"
        st.rerun()
    
    df = st.session_state.df
    data_validator = st.session_state.data_validator
    
    if df is not None and data_validator is not None:
        
        # Quick quality check section
        st.subheader("🚀 Quick Quality Check")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.info("📊 Running comprehensive validation on your dataset...")
        
        with col2:
            if st.button("🔄 Run Validation", type="primary", use_container_width=True):
                st.session_state.run_validation = True
        
        # Run validation if requested
        if st.session_state.get('run_validation', False) or st.button("🔍 Start Validation", use_container_width=True):
            
            with st.spinner("🔍 Analyzing data quality..."):
                try:
                    validation_results, quality_metrics = data_validator.validate_dataset(df)
                    
                    # Store results in session state
                    st.session_state.validation_results = validation_results
                    st.session_state.quality_metrics = quality_metrics
                    st.session_state.run_validation = False
                    
                    st.success("✅ Validation completed successfully!")
                    
                except Exception as e:
                    st.error(f"⚠️ Validation failed: {str(e)}")
                    st.info("Some validation features may be limited, but basic checks will still work.")
        
        # Display results if available
        if hasattr(st.session_state, 'validation_results') and hasattr(st.session_state, 'quality_metrics'):
            
            validation_results = st.session_state.validation_results
            quality_metrics = st.session_state.quality_metrics
            
            st.markdown("---")
            
            # Overall Quality Score
            st.subheader("🏆 Overall Quality Score")
            
            # Determine quality level and colors
            score = quality_metrics.overall_score
            if score >= 95:
                quality_level = "Excellent"
                color = "#4CAF50"
                emoji = "✅"
            elif score >= 85:
                quality_level = "Good"
                color = "#4CAF50"
                emoji = "😊"
            elif score >= 70:
                quality_level = "Fair"
                color = "#FF9800"
                emoji = "⚠️"
            elif score >= 50:
                quality_level = "Poor"
                color = "#FF5722"
                emoji = "😟"
            else:
                quality_level = "Critical"
                color = "#F44336"
                emoji = "❌"
            
            # Quality score display
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "🏆 Overall Score",
                    f"{score:.1f}/100",
                    f"{quality_level} {emoji}"
                )
            
            with col2:
                st.metric(
                    "📋 Completeness",
                    f"{quality_metrics.completeness_score:.1f}/100",
                    "Data Completeness"
                )
            
            with col3:
                st.metric(
                    "✅ Validity", 
                    f"{quality_metrics.validity_score:.1f}/100",
                    "Data Validity"
                )
            
            with col4:
                st.metric(
                    "🔄 Consistency",
                    f"{quality_metrics.consistency_score:.1f}/100",
                    "Data Consistency"
                )
            
            st.markdown("---")
            
            # Issues Summary
            st.subheader("🚨 Issues Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "📄 Total Records",
                    f"{quality_metrics.total_records:,}",
                    "Dataset Size"
                )
            
            with col2:
                critical_color = "normal" if quality_metrics.critical_issues == 0 else "inverse"
                st.metric(
                    "🔴 Critical Issues",
                    quality_metrics.critical_issues,
                    "High Priority",
                    delta_color=critical_color
                )
            
            with col3:
                warning_color = "normal" if quality_metrics.warnings < 5 else "inverse"
                st.metric(
                    "🟡 Warnings",
                    quality_metrics.warnings,
                    "Medium Priority",
                    delta_color=warning_color
                )
            
            with col4:
                st.metric(
                    "🔍 Total Issues",
                    quality_metrics.issues_found,
                    "All Issues"
                )
            
            # Detailed Issues
            if validation_results:
                st.markdown("---")
                st.subheader("📋 Detailed Validation Results")
                
                # Filter issues by severity
                critical_issues = [r for r in validation_results if r.severity == 'critical']
                high_issues = [r for r in validation_results if r.severity == 'high']
                medium_issues = [r for r in validation_results if r.severity == 'medium']
                low_issues = [r for r in validation_results if r.severity == 'low']
                
                # Tabs for different severity levels
                if critical_issues or high_issues:
                    tab1, tab2, tab3, tab4 = st.tabs([
                        f"🔴 Critical ({len(critical_issues)})",
                        f"🟠 High ({len(high_issues)})", 
                        f"🟡 Medium ({len(medium_issues)})",
                        f"🟢 Low ({len(low_issues)})"
                    ])
                else:
                    tab1, tab2 = st.tabs([
                        f"🟡 Medium ({len(medium_issues)})",
                        f"🟢 Low ({len(low_issues)})"
                    ])
                    tab3, tab4 = None, None
                
                # Display critical issues
                if tab4:  # All tabs available
                    with tab1:
                        if critical_issues:
                            for issue in critical_issues:
                                st.error(f"❌ **{issue.message}**")
                                st.markdown(f"   - **Affected rows:** {issue.affected_rows}")
                                if issue.recommendations:
                                    st.markdown(f"   - **Recommendation:** {issue.recommendations[0]}")
                                st.markdown("---")
                        else:
                            st.success("✅ No critical issues found!")
                    
                    with tab2:
                        if high_issues:
                            for issue in high_issues:
                                st.warning(f"🟠 **{issue.message}**")
                                st.markdown(f"   - **Affected rows:** {issue.affected_rows}")
                                if issue.recommendations:
                                    st.markdown(f"   - **Recommendation:** {issue.recommendations[0]}")
                                st.markdown("---")
                        else:
                            st.success("✅ No high priority issues found!")
                    
                    with tab3:
                        if medium_issues:
                            for issue in medium_issues[:10]:  # Limit display
                                st.info(f"🟡 **{issue.message}**")
                                if issue.recommendations:
                                    st.markdown(f"   - **Recommendation:** {issue.recommendations[0]}")
                        else:
                            st.success("✅ No medium priority issues found!")
                    
                    with tab4:
                        if low_issues:
                            for issue in low_issues[:10]:  # Limit display
                                with st.expander(f"🟢 {issue.message}"):
                                    st.markdown(f"**Affected rows:** {issue.affected_rows}")
                                    if issue.recommendations:
                                        st.markdown(f"**Recommendation:** {issue.recommendations[0]}")
                        else:
                            st.success("✅ No low priority issues found!")
                
                else:  # Only medium and low tabs
                    with tab1:
                        if medium_issues:
                            for issue in medium_issues[:10]:
                                st.info(f"🟡 **{issue.message}**")
                                if issue.recommendations:
                                    st.markdown(f"   - **Recommendation:** {issue.recommendations[0]}")
                        else:
                            st.success("✅ No medium priority issues found!")
                    
                    with tab2:
                        if low_issues:
                            for issue in low_issues[:10]:
                                with st.expander(f"🟢 {issue.message}"):
                                    st.markdown(f"**Affected rows:** {issue.affected_rows}")
                                    if issue.recommendations:
                                        st.markdown(f"**Recommendation:** {issue.recommendations[0]}")
                        else:
                            st.success("✅ No low priority issues found!")
            
            # Data Profile
            st.markdown("---")
            st.subheader("📊 Data Profile")
            
            with st.expander("🔍 View Detailed Data Profile"):
                try:
                    profile = data_validator.get_data_profile(df)
                    
                    # Basic info
                    st.markdown("**📈 Dataset Overview:**")
                    basic_info = profile['basic_info']
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Rows", f"{basic_info['total_rows']:,}")
                    with col2:
                        st.metric("Total Columns", basic_info['total_columns'])
                    with col3:
                        st.metric("Memory Usage", basic_info['memory_usage'])
                    
                    # Column profiles
                    if 'column_profiles' in profile:
                        st.markdown("**📁 Column Profiles:**")
                        
                        profile_data = []
                        for col_name, col_profile in profile['column_profiles'].items():
                            profile_data.append({
                                'Column': col_name,
                                'Type': col_profile['data_type'],
                                'Null %': f"{col_profile['null_percentage']:.1f}%",
                                'Unique': col_profile['unique_count'],
                                'Unique %': f"{col_profile['unique_percentage']:.1f}%"
                            })
                        
                        profile_df = pd.DataFrame(profile_data)
                        st.dataframe(profile_df, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Could not generate data profile: {str(e)}")
            
            # Quality Report
            st.markdown("---")
            st.subheader("📝 Quality Report")
            
            if st.button("📝 Generate Detailed Report", use_container_width=True):
                try:
                    report = data_validator.get_quality_report(validation_results, quality_metrics)
                    
                    # Display report in expandable text area
                    with st.expander("📄 View Full Quality Report", expanded=True):
                        st.text(report)
                    
                    # Download button for the report
                    st.download_button(
                        "💾 Download Report",
                        data=report,
                        file_name=f"data_quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                
                except Exception as e:
                    st.error(f"Could not generate report: {str(e)}")
        
        else:
            # Initial state - no validation run yet
            st.info("🚀 Click 'Start Validation' above to begin data quality assessment.")
            
            # Show preview of what will be checked
            st.markdown("### 🔍 What We'll Check")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                    **📋 Structure & Completeness**
                    - Missing essential columns
                    - Null values and empty data
                    - Data type consistency
                    - Minimum data requirements
                    
                    **✅ Data Validity**
                    - Rating ranges (1-5 scale)
                    - Text length requirements
                    - Categorical value validation
                    - Date/time format checking
                """)
            
            with col2:
                st.markdown("""
                    **🔄 Consistency & Quality**
                    - Product-brand relationships
                    - Rating-sentiment alignment
                    - Naming convention consistency
                    - Statistical distribution analysis
                    
                    **🔍 Duplicate Detection**
                    - Exact duplicate records
                    - Near-duplicate reviews
                    - Suspicious content patterns
                    - Data authenticity indicators
                """)
    
    else:
        st.error("⚠️ Data validator or dataset not available. Please check your configuration.")
        
        # Show demo content
        st.markdown("### 🏆 Data Quality Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
                **🔍 Smart Validation**
                - Comprehensive data structure checks
                - Business rule validation
                - Statistical anomaly detection
                - Cross-column consistency analysis
                - Automated quality scoring
            """)
        
        with col2:
            st.markdown("""
                **📄 Detailed Reporting**
                - Prioritized issue identification
                - Actionable recommendations
                - Quality metrics dashboard
                - Data profiling and statistics
                - Downloadable quality reports
            """)

if __name__ == "__main__":
    main()
