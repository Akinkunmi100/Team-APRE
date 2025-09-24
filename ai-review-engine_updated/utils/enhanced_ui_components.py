"""
Enhanced UI Components for Web Search Integration
Provides UI components and helpers for displaying search results from multiple sources
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd

# Import search orchestrator
from core.search_orchestrator import SearchResult

def display_search_source_indicator(search_result: SearchResult):
    """Display indicator showing the source of search results"""
    
    source = search_result.source
    confidence = search_result.confidence
    
    if source == 'local':
        st.success(f"💾 **Local Database** (Confidence: {confidence:.1%})")
        st.caption("✅ High confidence results from curated database")
        
    elif source == 'web':
        st.warning(f"🌐 **Web Search** (Confidence: {confidence:.1%})")
        st.caption("⚡ Real-time data from multiple web sources")
        
    elif source == 'hybrid':
        st.info(f"📊 **Combined Sources** (Confidence: {confidence:.1%})")
        st.caption("🔄 Local database + web search for comprehensive analysis")
        
    elif source == 'local_low_confidence':
        st.warning(f"💾 **Local Database - Low Confidence** (Confidence: {confidence:.1%})")
        st.caption("⚠️ Limited local data available")
        
    elif source == 'none':
        st.error("❌ **No Results Found**")
        st.caption("🔍 Phone not found in database or web sources")
        
    elif source == 'error':
        st.error("⚠️ **Search Error**")
        st.caption("❌ Technical issue occurred during search")

def display_search_metadata(search_result: SearchResult):
    """Display search metadata and performance info"""
    
    metadata = search_result.search_metadata
    
    with st.expander("🔍 Search Details"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Search Time", f"{metadata.get('search_time', 0):.2f}s")
            if 'local_search_used' in metadata:
                st.write(f"📊 Local Search: {'✅' if metadata['local_search_used'] else '❌'}")
            if 'web_search_used' in metadata:
                st.write(f"🌐 Web Search: {'✅' if metadata['web_search_used'] else '❌'}")
        
        with col2:
            if 'sources_combined' in metadata:
                st.metric("Sources Combined", metadata['sources_combined'])
            if 'web_sources' in metadata:
                st.write("**Web Sources:**")
                for source in metadata['web_sources']:
                    st.write(f"• {source.title()}")

def display_phone_overview_card(search_result: SearchResult):
    """Display main phone overview card"""
    
    phone_data = search_result.phone_data
    
    # Create overview card
    st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 2rem; border-radius: 15px; color: white; margin: 1rem 0;">
            <h2 style="margin: 0; color: white;">📱 {phone_data.get('product_name', 'Unknown Phone')}</h2>
            <h4 style="margin: 0.5rem 0; color: #E0E0E0;">🏷️ {phone_data.get('brand', 'Unknown Brand')}</h4>
        </div>
    """, unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        rating = phone_data.get('overall_rating') or phone_data.get('combined_rating')
        if rating:
            st.metric("⭐ Overall Rating", f"{rating:.1f}/5.0")
        else:
            st.metric("⭐ Overall Rating", "N/A")
    
    with col2:
        local_count = phone_data.get('review_count', 0)
        web_count = phone_data.get('web_review_count', 0)
        total_reviews = local_count + web_count
        st.metric("💬 Reviews", f"{total_reviews:,}")
    
    with col3:
        if search_result.source == 'hybrid':
            sources_count = len(phone_data.get('total_sources', []))
            st.metric("📊 Sources", sources_count)
        else:
            confidence = search_result.confidence
            st.metric("🎯 Confidence", f"{confidence:.1%}")
    
    with col4:
        if 'price_info' in phone_data and phone_data['price_info'].get('status') == 'found':
            avg_price = phone_data['price_info'].get('avg_price', 'N/A')
            st.metric("💰 Avg Price", avg_price)
        else:
            st.metric("💰 Price", "N/A")

def display_recommendations_card(search_result: SearchResult):
    """Display AI recommendations"""
    
    recommendations = search_result.recommendations
    if not recommendations:
        return
    
    # Recommendation card
    verdict = recommendations.get('verdict', 'Analysis Complete')
    reason = recommendations.get('reason', '')
    
    if 'Highly Recommended' in verdict:
        color = "#4CAF50"
        icon = "🏆"
    elif 'Recommended' in verdict:
        color = "#2196F3"
        icon = "✅"
    elif 'Consider' in verdict:
        color = "#FF9800"
        icon = "⚠️"
    else:
        color = "#F44336"
        icon = "❌"
    
    st.markdown(f"""
        <div style="background: {color}20; padding: 1.5rem; border-radius: 12px; 
                    border-left: 4px solid {color}; margin: 1rem 0;">
            <h3 style="margin: 0; color: {color};">{icon} {verdict}</h3>
            <p style="margin: 0.5rem 0; color: #333;">{reason}</p>
            <small style="color: #666;">
                {recommendations.get('note', '')} | 
                Reliability: {recommendations.get('reliability', 'medium').title()}
            </small>
        </div>
    """, unsafe_allow_html=True)

def display_hybrid_analysis(search_result: SearchResult):
    """Display analysis for hybrid results (local + web)"""
    
    if search_result.source != 'hybrid':
        return
    
    phone_data = search_result.phone_data
    
    st.subheader("📊 Combined Analysis")
    
    # Rating comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Local Database Results**")
        local_rating = phone_data.get('local_rating')
        local_count = phone_data.get('local_review_count', 0)
        
        if local_rating:
            st.metric("⭐ Local Rating", f"{local_rating:.1f}/5.0")
        st.metric("💬 Local Reviews", f"{local_count:,}")
        
        # Local sentiment
        if 'local_sentiment' in phone_data and phone_data['local_sentiment']:
            local_sent = phone_data['local_sentiment']
            st.write("**Sentiment Distribution:**")
            st.write(f"😊 Positive: {local_sent.get('positive', 0):.1f}%")
            st.write(f"😐 Neutral: {local_sent.get('neutral', 0):.1f}%")
            st.write(f"😞 Negative: {local_sent.get('negative', 0):.1f}%")
    
    with col2:
        st.markdown("**Web Sources Results**")
        web_rating = phone_data.get('web_rating')
        web_count = phone_data.get('web_review_count', 0)
        
        if web_rating:
            st.metric("⭐ Web Rating", f"{web_rating:.1f}/5.0")
        st.metric("💬 Web Reviews", f"{web_count:,}")
        
        # Web sources
        if 'total_sources' in phone_data:
            st.write("**Sources:**")
            for source in phone_data['total_sources']:
                if source != 'local_database':
                    st.write(f"• {source.title()}")
    
    # Combined rating
    combined_rating = phone_data.get('combined_rating')
    if combined_rating:
        st.markdown("---")
        st.metric("🎯 **Combined Rating**", f"{combined_rating:.1f}/5.0", 
                 help="Weighted average based on review counts")

def display_web_features_analysis(search_result: SearchResult):
    """Display features analysis from web sources"""
    
    phone_data = search_result.phone_data
    
    if search_result.source not in ['web', 'hybrid']:
        return
    
    # Features section
    features = phone_data.get('web_features', []) or phone_data.get('key_features', [])
    pros = phone_data.get('web_pros', []) or phone_data.get('pros', [])
    cons = phone_data.get('web_cons', []) or phone_data.get('cons', [])
    
    if features or pros or cons:
        st.subheader("🔍 Key Features & Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if pros:
                st.markdown("**✅ Strengths**")
                for pro in pros[:5]:
                    st.markdown(f"• {pro}")
        
        with col2:
            if cons:
                st.markdown("**⚠️ Areas for Improvement**")
                for con in cons[:5]:
                    st.markdown(f"• {con}")
        
        if features:
            st.markdown("**📱 Mentioned Features**")
            feature_cols = st.columns(min(len(features), 4))
            for i, feature in enumerate(features[:8]):
                with feature_cols[i % 4]:
                    st.markdown(f"🔧 {feature.title()}")

def display_price_analysis(search_result: SearchResult):
    """Display price information if available"""
    
    phone_data = search_result.phone_data
    price_info = phone_data.get('price_info', {})
    
    if price_info.get('status') == 'found':
        st.subheader("💰 Pricing Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_price = price_info.get('min_price', 'N/A')
            st.metric("💵 Min Price", min_price)
        
        with col2:
            avg_price = price_info.get('avg_price', 'N/A')
            st.metric("💸 Avg Price", avg_price)
        
        with col3:
            max_price = price_info.get('max_price', 'N/A')
            st.metric("💎 Max Price", max_price)
        
        sources_count = price_info.get('price_sources', 0)
        st.caption(f"📊 Based on {sources_count} price sources")

def display_no_results_help(search_result: SearchResult):
    """Display helpful suggestions when no results found"""
    
    if search_result.phone_found:
        return
    
    phone_data = search_result.phone_data
    suggestions = phone_data.get('suggestions', [])
    
    st.error("🔍 Phone Not Found")
    
    st.markdown("### 💡 Try these suggestions:")
    
    for i, suggestion in enumerate(suggestions, 1):
        st.markdown(f"{i}. {suggestion}")
    
    st.markdown("---")
    st.info("🌐 **Web Search Attempted**: We searched multiple online sources but couldn't find this phone model.")

def create_search_statistics_chart(orchestrator):
    """Create search statistics visualization"""
    
    stats = orchestrator.get_search_statistics()
    
    if 'message' in stats:
        st.info(stats['message'])
        return
    
    # Create pie chart of search sources
    labels = ['Local Hits', 'Web Searches', 'Hybrid Results', 'Failed']
    values = [
        stats.get('local_hit_rate', 0),
        stats.get('web_search_rate', 0),
        stats.get('hybrid_result_rate', 0),
        stats.get('failure_rate', 0)
    ]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels, 
        values=values,
        hole=0.4,
        marker_colors=['#4CAF50', '#2196F3', '#FF9800', '#F44336']
    )])
    
    fig.update_layout(
        title="Search Source Distribution",
        height=400,
        showlegend=True
    )
    
    return fig

def enhanced_search_interface():
    """Enhanced search interface with web fallback options"""
    
    st.subheader("🔍 Enhanced Phone Search")
    
    # Search input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input(
            "Search for a phone:",
            placeholder="e.g., 'iPhone 15 Pro', 'Samsung Galaxy S24 reviews', 'Nothing Phone 2'"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    
    # Search options
    with st.expander("⚙️ Search Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            force_web = st.checkbox("🌐 Force web search", 
                                   help="Search web sources even if local data is available")
            skip_local = st.checkbox("📊 Skip local database", 
                                   help="Only search web sources")
        
        with col2:
            max_sources = st.slider("Web sources to check", 1, 5, 3,
                                   help="Maximum number of web sources to search")
    
    return search_query, search_button, {
        'force_web_search': force_web,
        'skip_local_search': skip_local,
        'max_sources': max_sources
    }

def display_complete_search_result(search_result: SearchResult):
    """Display complete search result with all components"""
    
    # Source indicator
    display_search_source_indicator(search_result)
    
    if not search_result.phone_found:
        # No results - show help
        display_no_results_help(search_result)
        display_search_metadata(search_result)
        return
    
    # Main phone overview
    display_phone_overview_card(search_result)
    
    # Recommendations
    display_recommendations_card(search_result)
    
    # Source-specific analysis
    if search_result.source == 'hybrid':
        display_hybrid_analysis(search_result)
    
    # Features and analysis
    display_web_features_analysis(search_result)
    
    # Price information
    display_price_analysis(search_result)
    
    # Search metadata
    display_search_metadata(search_result)

def create_search_comparison_chart(results: List[SearchResult]):
    """Create comparison chart for multiple search results"""
    
    if len(results) < 2:
        return None
    
    phone_names = []
    ratings = []
    confidences = []
    sources = []
    
    for result in results:
        if result.phone_found:
            phone_data = result.phone_data
            phone_names.append(phone_data.get('product_name', 'Unknown'))
            rating = phone_data.get('overall_rating') or phone_data.get('combined_rating') or 0
            ratings.append(rating)
            confidences.append(result.confidence)
            sources.append(result.source)
    
    if not phone_names:
        return None
    
    # Create comparison chart
    fig = go.Figure()
    
    # Add rating bars
    fig.add_trace(go.Bar(
        name='Rating',
        x=phone_names,
        y=ratings,
        yaxis='y',
        marker_color='#4CAF50',
        text=[f"{r:.1f}" for r in ratings],
        textposition='auto'
    ))
    
    # Add confidence line
    fig.add_trace(go.Scatter(
        name='Confidence',
        x=phone_names,
        y=[c * 5 for c in confidences],  # Scale to 0-5 range
        yaxis='y2',
        mode='lines+markers',
        line=dict(color='#FF9800', width=3),
        marker=dict(size=8)
    ))
    
    # Update layout
    fig.update_layout(
        title='Phone Comparison: Rating vs Search Confidence',
        xaxis_title='Phones',
        yaxis_title='Rating (0-5)',
        yaxis2=dict(
            title='Confidence (%)',
            overlaying='y',
            side='right',
            range=[0, 5]
        ),
        height=500,
        showlegend=True
    )
    
    return fig

# CSS for enhanced styling
ENHANCED_UI_CSS = """
<style>
.search-result-card {
    background: white;
    padding: 1.5rem;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    margin: 1rem 0;
    border-left: 4px solid #2196F3;
}

.source-indicator {
    display: inline-block;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-weight: bold;
    margin: 0.5rem 0;
}

.source-local { background: #E8F5E8; color: #2E7D32; }
.source-web { background: #FFF3E0; color: #E65100; }
.source-hybrid { background: #E3F2FD; color: #1565C0; }
.source-error { background: #FFEBEE; color: #C62828; }

.metric-card {
    background: #F8F9FA;
    padding: 1rem;
    border-radius: 8px;
    text-align: center;
    margin: 0.5rem 0;
}

.recommendation-high { border-left-color: #4CAF50; }
.recommendation-medium { border-left-color: #FF9800; }
.recommendation-low { border-left-color: #F44336; }
</style>
"""

def inject_enhanced_ui_css():
    """Inject enhanced UI CSS"""
    st.markdown(ENHANCED_UI_CSS, unsafe_allow_html=True)