"""
AI Phone Review Engine - Enhanced User-Friendly Version
Now with integrated Web Search Agent and Orchestrator
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import sys
import os
from typing import Dict, List, Optional, Any

# Add project directories to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import enhanced components
from core.api_search_orchestrator import APISearchOrchestrator, create_api_search_orchestrator, APISearchResult as SearchResult
from utils.enhanced_ui_components import (
    display_complete_search_result, enhanced_search_interface,
    inject_enhanced_ui_css, create_search_statistics_chart
)

# Import existing components (with fallback)
try:
    from utils.preprocessed_data_loader import PreprocessedDataLoader
    PREPROCESSED_DATA_AVAILABLE = True
except ImportError:
    PREPROCESSED_DATA_AVAILABLE = False

try:
    from models.recommendation_engine_simple import RecommendationEngine
    from models.auto_insights_engine import AutoInsightsEngine
    from models.absa_model import ABSASentimentAnalyzer
    from utils.data_preprocessing import DataPreprocessor
    from utils.visualization import ReviewVisualizer
    CORE_MODULES_AVAILABLE = True
except ImportError:
    CORE_MODULES_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="AI Phone Review Engine - Enhanced",
    page_icon="🤖📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject enhanced CSS
inject_enhanced_ui_css()

def initialize_enhanced_system():
    """Initialize the enhanced system with orchestrator"""
    if 'orchestrator' not in st.session_state:
        with st.spinner("🚀 Initializing Enhanced AI System..."):
            try:
                # Load data
                df = None
                if PREPROCESSED_DATA_AVAILABLE:
                    try:
                        data_loader = PreprocessedDataLoader()
                        df = data_loader.get_full_dataset()
                        st.session_state.data_source = "preprocessed_data"
                        st.session_state.data_info = f"{len(df)} preprocessed reviews loaded"
                    except Exception as e:
                        st.warning(f"Could not load preprocessed data: {str(e)}")
                        df = create_fallback_data()
                        st.session_state.data_source = "fallback_data"
                        st.session_state.data_info = "Using fallback sample data"
                else:
                    df = create_fallback_data()
                    st.session_state.data_source = "sample_data"
                    st.session_state.data_info = "Using sample data (preprocessed data not available)"
                
                # Configuration for orchestrator
                orchestrator_config = {
                    'local_confidence_threshold': 0.7,  # Lower threshold for better web integration
                    'enable_web_fallback': True,
                    'enable_hybrid_search': True,
                    'max_search_timeout': 30,
                    'cache_results': True,
                    'log_searches': True
                }
                
                # Create API-based orchestrator
                st.session_state.orchestrator = create_api_search_orchestrator(
                    local_data=df,
                    config=orchestrator_config
                )
                
                # Initialize other components if available
                if CORE_MODULES_AVAILABLE:
                    st.session_state.rec_engine = RecommendationEngine()
                    st.session_state.insights_engine = AutoInsightsEngine()
                    st.session_state.absa_analyzer = ABSASentimentAnalyzer()
                
                st.session_state.initialized = True
                st.session_state.df = df
                
            except Exception as e:
                st.error(f"⚠️ Initialization error: {str(e)}")
                st.session_state.orchestrator = None

def create_fallback_data():
    """Create fallback data if real data not available"""
    data = {
        'product': [
            'iPhone 15 Pro', 'iPhone 15 Pro Max', 'iPhone 14 Pro', 'iPhone 13 Pro',
            'Samsung Galaxy S24 Ultra', 'Samsung Galaxy S24', 'Samsung Galaxy S23 Ultra',
            'Google Pixel 8 Pro', 'Google Pixel 8', 'Google Pixel 7a',
            'OnePlus 12', 'OnePlus 11', 'Nothing Phone 2', 'Xiaomi 14 Pro'
        ] * 20,
        'brand': [
            'Apple', 'Apple', 'Apple', 'Apple',
            'Samsung', 'Samsung', 'Samsung',
            'Google', 'Google', 'Google',
            'OnePlus', 'OnePlus', 'Nothing', 'Xiaomi'
        ] * 20,
        'rating': np.random.normal(4.2, 0.8, 280),
        'review_text': ['Great phone with excellent features'] * 280,
        'sentiment_label': np.random.choice(['positive', 'neutral', 'negative'], 280, p=[0.6, 0.3, 0.1])
    }
    
    df = pd.DataFrame(data)
    df['rating'] = df['rating'].clip(1, 5)
    return df

def main():
    """Main enhanced application"""
    
    # Initialize system
    initialize_enhanced_system()
    
    if not st.session_state.get('orchestrator'):
        st.error("❌ System initialization failed. Please refresh the page.")
        return
    
    # Header
    st.markdown("""
        <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 2rem;">
            <h1 style="color: white; margin: 0;">🤖📱 AI Phone Review Engine</h1>
            <h3 style="color: #E0E0E0; margin: 0.5rem 0;">Enhanced with API Search Intelligence</h3>
            <p style="color: #E0E0E0; margin: 0;">Local Database + API Search + Fallback System + AI Analysis</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Show data source info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"📊 Data: {st.session_state.data_info}")
    with col2:
        st.success("🌐 API Search: Enabled")
    with col3:
        st.success("🤖 AI Analysis: Active")
    
    # Sidebar navigation
    with st.sidebar:
        st.header("🧭 Navigation")
        
        page = st.selectbox(
            "Choose Feature",
            [
                "🔍 Enhanced Search",
                "📊 Search Statistics", 
                "⚙️ System Settings",
                "📱 Quick Analysis",
                "🏆 Top Phones",
                "❓ Help & Info"
            ]
        )
        
        st.markdown("---")
        st.markdown("### 📈 Quick Stats")
        
        if hasattr(st.session_state, 'orchestrator') and st.session_state.orchestrator:
            stats = st.session_state.orchestrator.get_search_statistics()
            if 'total_searches' in stats:
                st.metric("Total Searches", stats['total_searches'])
                st.metric("Success Rate", f"{stats.get('success_rate', 0):.1f}%")
        
        st.markdown("---")
        st.markdown("### 🔧️ System Status")
        st.write("✅ API Search Orchestrator")
        st.write("✅ API Web Search Agent")
        st.write("✅ Fallback Search System")
        st.write("✅ Enhanced UI")
        if CORE_MODULES_AVAILABLE:
            st.write("✅ Core ML Modules")
        if PREPROCESSED_DATA_AVAILABLE:
            st.write("✅ Preprocessed Data")
    
    # Main content based on selected page
    if "Enhanced Search" in page:
        show_enhanced_search_page()
    elif "Search Statistics" in page:
        show_statistics_page()
    elif "System Settings" in page:
        show_settings_page()
    elif "Quick Analysis" in page:
        show_quick_analysis_page()
    elif "Top Phones" in page:
        show_top_phones_page()
    elif "Help" in page:
        show_help_page()

def show_enhanced_search_page():
    """Enhanced search page with orchestrator integration"""
    
    st.header("🔍 Enhanced Phone Search")
    st.markdown("Search across local database and web sources for comprehensive phone analysis.")
    
    # Enhanced search interface
    search_query, search_button, search_options = enhanced_search_interface()
    
    # Perform search
    if search_button and search_query:
        with st.spinner("🔍 Searching across all sources..."):
            try:
                # Use orchestrator for search
                orchestrator = st.session_state.orchestrator
                search_result = orchestrator.search_phone(search_query, search_options)
                
                # Store result for comparison
                if 'search_history' not in st.session_state:
                    st.session_state.search_history = []
                st.session_state.search_history.append(search_result)
                
                # Display complete result
                st.markdown("---")
                display_complete_search_result(search_result)
                
                # Search history
                if len(st.session_state.search_history) > 1:
                    st.markdown("---")
                    st.subheader("📋 Recent Searches")
                    
                    recent_searches = st.session_state.search_history[-5:]  # Last 5 searches
                    for i, result in enumerate(reversed(recent_searches[:-1]), 1):
                        if result.phone_found:
                            phone_name = result.phone_data.get('product_name', 'Unknown')
                            source_icon = {"local": "💾", "web": "🌐", "hybrid": "📊"}.get(result.source, "❓")
                            st.write(f"{i}. {source_icon} {phone_name} (Confidence: {result.confidence:.1%})")
                
            except Exception as e:
                st.error(f"Search error: {str(e)}")
    
    elif search_query and not search_button:
        st.info("👆 Click Search to analyze this phone")

def show_statistics_page():
    """Show search statistics and analytics"""
    
    st.header("📊 Search Statistics & Analytics")
    
    orchestrator = st.session_state.orchestrator
    if not orchestrator:
        st.error("Orchestrator not available")
        return
    
    # Get statistics
    stats = orchestrator.get_search_statistics()
    
    if 'total_searches' not in stats:
        st.info("📊 No searches performed yet. Try the Enhanced Search feature!")
        return
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Searches", stats['total_searches'])
    with col2:
        st.metric("Success Rate", f"{stats.get('success_rate', 0):.1f}%")
    with col3:
        st.metric("Local Hit Rate", f"{stats.get('local_hit_rate', 0):.1f}%")
    with col4:
        st.metric("Web Search Rate", f"{stats.get('web_search_rate', 0):.1f}%")
    
    # Statistics chart
    st.subheader("📈 Search Source Distribution")
    
    chart = create_search_statistics_chart(orchestrator)
    if chart:
        st.plotly_chart(chart, use_container_width=True)
    
    # Detailed statistics
    with st.expander("📋 Detailed Statistics"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Search Patterns")
            st.write(f"• Hybrid Results: {stats.get('hybrid_result_rate', 0):.1f}%")
            st.write(f"• Failed Searches: {stats.get('failure_rate', 0):.1f}%")
            
        with col2:
            st.subheader("Performance Insights")
            if stats.get('web_search_rate', 0) > 50:
                st.warning("High web search usage - consider expanding local database")
            elif stats.get('local_hit_rate', 0) > 80:
                st.success("Excellent local database coverage")
            else:
                st.info("Balanced local and web search usage")

def show_settings_page():
    """System settings and configuration"""
    
    st.header("⚙️ System Settings")
    
    orchestrator = st.session_state.orchestrator
    if not orchestrator:
        st.error("Orchestrator not available")
        return
    
    # Current configuration
    current_config = orchestrator.config
    
    st.subheader("🔧 Search Configuration")
    
    with st.form("settings_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            confidence_threshold = st.slider(
                "Local Confidence Threshold",
                min_value=0.5,
                max_value=1.0,
                value=current_config.get('local_confidence_threshold', 0.8),
                step=0.05,
                help="Minimum confidence to use local results without web search"
            )
            
            enable_web_fallback = st.checkbox(
                "Enable Web Search Fallback",
                value=current_config.get('enable_web_fallback', True),
                help="Search web sources when local data is insufficient"
            )
            
            enable_hybrid = st.checkbox(
                "Enable Hybrid Search",
                value=current_config.get('enable_hybrid_search', True),
                help="Combine local and web results when beneficial"
            )
        
        with col2:
            search_timeout = st.number_input(
                "Search Timeout (seconds)",
                min_value=10,
                max_value=60,
                value=current_config.get('max_search_timeout', 30),
                step=5,
                help="Maximum time to wait for web searches"
            )
            
            cache_results = st.checkbox(
                "Cache Search Results",
                value=current_config.get('cache_results', True),
                help="Cache web search results to improve performance"
            )
            
            log_searches = st.checkbox(
                "Log Search Activity",
                value=current_config.get('log_searches', True),
                help="Keep logs of search activity for analytics"
            )
        
        if st.form_submit_button("💾 Save Settings", type="primary"):
            new_config = {
                'local_confidence_threshold': confidence_threshold,
                'enable_web_fallback': enable_web_fallback,
                'enable_hybrid_search': enable_hybrid,
                'max_search_timeout': search_timeout,
                'cache_results': cache_results,
                'log_searches': log_searches
            }
            
            orchestrator.configure(new_config)
            st.success("✅ Settings saved successfully!")
            st.rerun()
    
    # System information
    st.markdown("---")
    st.subheader("ℹ️ System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Components Status:**")
        st.write(f"✅ Search Orchestrator: Active")
        st.write(f"{'✅' if orchestrator.web_agent else '❌'} Web Search Agent: {'Active' if orchestrator.web_agent else 'Disabled'}")
        st.write(f"{'✅' if CORE_MODULES_AVAILABLE else '⚠️'} Core ML Modules: {'Available' if CORE_MODULES_AVAILABLE else 'Limited'}")
        st.write(f"{'✅' if PREPROCESSED_DATA_AVAILABLE else '⚠️'} Preprocessed Data: {'Available' if PREPROCESSED_DATA_AVAILABLE else 'Sample Data'}")
    
    with col2:
        st.write("**Data Sources:**")
        st.write(f"📊 Local Database: {len(st.session_state.df) if st.session_state.df is not None else 0} records")
        st.write(f"🌐 Web Sources: {'5 sources available' if orchestrator.web_agent else 'Disabled'}")
        st.write(f"💾 Cache Status: {'Enabled' if current_config.get('cache_results') else 'Disabled'}")

def show_quick_analysis_page():
    """Quick analysis for popular phones"""
    
    st.header("📱 Quick Phone Analysis")
    st.markdown("Get instant analysis for popular phone models")
    
    # Popular phones
    popular_phones = [
        "iPhone 15 Pro", "iPhone 15 Pro Max", "Samsung Galaxy S24 Ultra",
        "Google Pixel 8 Pro", "OnePlus 12", "Nothing Phone 2",
        "Xiaomi 14 Pro", "Samsung Galaxy S24"
    ]
    
    st.subheader("🏆 Popular Phones")
    
    cols = st.columns(4)
    for i, phone in enumerate(popular_phones[:8]):
        with cols[i % 4]:
            if st.button(f"📱 {phone}", key=f"quick_{i}", use_container_width=True):
                with st.spinner(f"Analyzing {phone}..."):
                    orchestrator = st.session_state.orchestrator
                    result = orchestrator.search_phone(phone)
                    
                    st.markdown("---")
                    st.subheader(f"Analysis: {phone}")
                    display_complete_search_result(result)

def show_top_phones_page():
    """Show top-rated phones from local database"""
    
    st.header("🏆 Top-Rated Phones")
    
    if st.session_state.df is None:
        st.error("No data available")
        return
    
    df = st.session_state.df
    
    # Calculate phone rankings
    if 'product' in df.columns and 'rating' in df.columns:
        phone_stats = df.groupby('product').agg({
            'rating': ['mean', 'count'],
            'sentiment_label': lambda x: (x == 'positive').mean() if 'sentiment_label' in df.columns else 0.8
        }).round(2)
        
        phone_stats.columns = ['avg_rating', 'review_count', 'positive_rate']
        phone_stats = phone_stats[phone_stats['review_count'] >= 5]  # At least 5 reviews
        phone_stats = phone_stats.sort_values('avg_rating', ascending=False)
        
        st.subheader("🥇 Top 10 by Average Rating")
        
        top_phones = phone_stats.head(10)
        
        for i, (phone, stats) in enumerate(top_phones.iterrows(), 1):
            col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])
            
            with col1:
                st.write(f"**{i}. {phone}**")
            with col2:
                st.write(f"⭐ {stats['avg_rating']:.1f}")
            with col3:
                st.write(f"💬 {int(stats['review_count'])}")
            with col4:
                st.write(f"😊 {stats['positive_rate']:.1%}")
            with col5:
                if st.button("🔍", key=f"top_{i}", help=f"Analyze {phone}"):
                    orchestrator = st.session_state.orchestrator
                    result = orchestrator.search_phone(phone)
                    
                    st.markdown("---")
                    display_complete_search_result(result)
        
        # Visual chart
        st.subheader("📊 Rating Distribution")
        
        fig = px.bar(
            x=top_phones.index[:8],
            y=top_phones['avg_rating'][:8],
            title="Top Phones by Average Rating",
            labels={'x': 'Phone Model', 'y': 'Average Rating'},
            color=top_phones['avg_rating'][:8],
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

def show_help_page():
    """Help and information page"""
    
    st.header("❓ Help & Information")
    
    st.subheader("🚀 Getting Started")
    st.markdown("""
    1. **🔍 Enhanced Search**: Use natural language queries like "iPhone 15 Pro reviews" or "best camera phone under $800"
    2. **📊 View Statistics**: Check how the system is performing and which sources are being used
    3. **⚙️ Adjust Settings**: Configure confidence thresholds and search behavior
    4. **📱 Quick Analysis**: Get instant analysis for popular phone models
    """)
    
    st.subheader("🌐 How Web Search Works")
    st.markdown("""
    - **Intelligent Fallback**: Web search activates when local database doesn't have sufficient information
    - **Multi-Source**: Searches GSMArena, PhoneArena, CNET, TechCrunch, and Google Shopping
    - **Quality Assessment**: Each result includes confidence scores and data quality indicators
    - **Hybrid Results**: Combines local database and web data for comprehensive analysis
    """)
    
    st.subheader("📊 Understanding Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Source Indicators:**")
        st.markdown("- 💾 **Local Database**: High confidence, curated data")
        st.markdown("- 🌐 **Web Search**: Real-time data from multiple sources") 
        st.markdown("- 📊 **Combined**: Best of both local and web data")
        st.markdown("- ⚠️ **Low Confidence**: Limited data available")
    
    with col2:
        st.markdown("**Confidence Levels:**")
        st.markdown("- **High (80%+)**: Very reliable results")
        st.markdown("- **Medium (60-80%)**: Good quality data")
        st.markdown("- **Low (<60%)**: Limited information available")
        st.markdown("- **Hybrid**: Combined confidence from multiple sources")
    
    st.subheader("🔧 Troubleshooting")
    
    with st.expander("Common Issues"):
        st.markdown("""
        **Phone not found?**
        - Check spelling of phone model
        - Try searching for just the brand name
        - Use more general terms (e.g., "iPhone 15" instead of "iPhone 15 Pro Max 256GB")
        
        **Slow search results?**
        - Web searches can take 10-30 seconds depending on your internet connection
        - Adjust search timeout in Settings if needed
        - Enable result caching to speed up repeated searches
        
        **Low confidence results?**
        - This means limited data is available for that phone
        - Try enabling web search fallback in Settings
        - Consider that very new or regional phones may have limited data
        """)
    
    st.subheader("💡 Tips for Best Results")
    st.markdown("""
    - Use specific phone model names for better accuracy
    - Try different phrasings if you don't get results initially
    - Enable hybrid search for the most comprehensive analysis
    - Check the search statistics to optimize your settings
    """)

if __name__ == "__main__":
    main()