"""
Business User Interface Components
Advanced features specifically for business users
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Optional

def display_business_dashboard(role_manager, user_memory=None):
    """Display business-specific dashboard"""
    st.markdown("### 📊 Business Analytics Dashboard")
    
    # Usage summary
    usage = role_manager.get_usage_summary()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Searches Today", usage['searches_today'], 
                 delta=f"{usage['daily_limit'] - usage['searches_today']} remaining")
    with col2:
        st.metric("This Month", usage['searches_this_month'])
    with col3:
        st.metric("API Calls", usage['api_calls'])
    with col4:
        st.metric("Reports Generated", usage['reports_generated'])

def display_bulk_search_interface():
    """Bulk search interface for business users"""
    st.markdown("### 🔍 Bulk Phone Search")
    st.markdown("*Search multiple phones at once for competitive analysis*")
    
    # Input methods
    input_method = st.radio("Choose input method:", 
                           ["Manual Entry", "Upload CSV", "Paste List"])
    
    phone_list = []
    
    if input_method == "Manual Entry":
        st.markdown("**Enter phone models (one per line):**")
        phones_text = st.text_area("Phone Models", 
                                  placeholder="iPhone 15 Pro\nSamsung Galaxy S24\nGoogle Pixel 8\nOnePlus 12", 
                                  height=150)
        if phones_text:
            phone_list = [phone.strip() for phone in phones_text.split('\n') if phone.strip()]
            
    elif input_method == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV with phone models", type=['csv'])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            if 'phone_model' in df.columns:
                phone_list = df['phone_model'].tolist()
                st.success(f"Loaded {len(phone_list)} phone models")
            else:
                st.error("CSV must contain a 'phone_model' column")
                
    elif input_method == "Paste List":
        pasted_text = st.text_area("Paste phone list", placeholder="Paste comma or line separated phone models")
        if pasted_text:
            # Handle both comma and line separated
            if ',' in pasted_text:
                phone_list = [phone.strip() for phone in pasted_text.split(',') if phone.strip()]
            else:
                phone_list = [phone.strip() for phone in pasted_text.split('\n') if phone.strip()]
    
    if phone_list:
        st.info(f"Ready to search {len(phone_list)} phones: {', '.join(phone_list[:3])}{'...' if len(phone_list) > 3 else ''}")
        
        if st.button("🚀 Start Bulk Search", type="primary"):
            return phone_list
    
    return None

def display_competitor_analysis(search_results: List[Dict]):
    """Display competitor analysis for business users"""
    st.markdown("### 🏆 Competitive Analysis")
    
    if not search_results:
        st.info("No search results available for analysis")
        return
    
    # Create comparison DataFrame
    comparison_data = []
    for result in search_results:
        if result.get('success', False):
            comparison_data.append({
                'Phone': result.get('phone_model', 'Unknown'),
                'Rating': result.get('overall_rating', 0),
                'Price Range': result.get('price_range', 'Unknown'),
                'Key Strengths': ', '.join(result.get('strengths', [])[:2]),
                'Key Weaknesses': ', '.join(result.get('weaknesses', [])[:2]),
                'Market Position': result.get('market_position', 'Unknown')
            })
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        
        # Display comparison table
        st.dataframe(df, use_container_width=True)
        
        # Rating comparison chart
        if len(df) > 1:
            fig = px.bar(df, x='Phone', y='Rating', 
                        title="Overall Rating Comparison",
                        color='Rating', color_continuous_scale='viridis')
            st.plotly_chart(fig, use_container_width=True)
            
        # Market positioning insights
        st.markdown("#### 💡 Market Insights")
        
        # Best performers
        best_phone = df.loc[df['Rating'].idxmax()]
        st.success(f"**Top Performer**: {best_phone['Phone']} (Rating: {best_phone['Rating']})")
        
        # Price analysis
        price_ranges = df['Price Range'].value_counts()
        most_common_price = price_ranges.index[0] if len(price_ranges) > 0 else "Unknown"
        st.info(f"**Most Common Price Range**: {most_common_price}")

def display_api_access_panel():
    """API access panel for business users"""
    st.markdown("### 🔌 API Access")
    st.markdown("*Integrate phone review data into your applications*")
    
    # API key management
    if 'api_key' not in st.session_state:
        if st.button("Generate API Key"):
            import secrets
            api_key = f"pr_{''.join(secrets.choice('abcdefghijklmnopqrstuvwxyz0123456789') for _ in range(32))}"
            st.session_state.api_key = api_key
            st.success("API Key Generated!")
    
    if 'api_key' in st.session_state:
        st.code(f"API Key: {st.session_state.api_key}")
        
        # API documentation
        st.markdown("#### 📚 Quick Start")
        
        st.code('''
# Python Example
import requests

headers = {
    "Authorization": f"Bearer {your_api_key}",
    "Content-Type": "application/json"
}

# Search phones
response = requests.post(
    "https://your-api.com/v1/search",
    headers=headers,
    json={"query": "iPhone 15 Pro"}
)

data = response.json()
''', language='python')
        
        # Usage limits
        st.markdown("#### 📊 API Usage")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("API Calls Today", "47", "953 remaining")
        with col2:
            st.metric("Monthly Limit", "1000", "Premium: 5000")

def display_export_options(search_results: List[Dict], user_role):
    """Export options for business users"""
    if not search_results:
        return
    
    st.markdown("### 📤 Export Data")
    
    export_format = st.selectbox("Export Format", 
                                ["CSV", "JSON", "Excel", "PDF Report"])
    
    include_options = st.multiselect("Include Data:", 
                                   ["Basic Info", "Detailed Reviews", "Ratings", 
                                    "Price History", "Specifications", "Market Analysis"],
                                   default=["Basic Info", "Ratings"])
    
    if st.button("Generate Export"):
        # Simulate export generation
        with st.spinner("Generating export..."):
            import time
            time.sleep(2)
        
        # Create downloadable content
        if export_format == "CSV":
            df = pd.DataFrame(search_results)
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"phone_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        elif export_format == "JSON":
            import json
            json_data = json.dumps(search_results, indent=2)
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name=f"phone_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

def display_custom_reports_builder():
    """Custom reports builder for business users"""
    st.markdown("### 📋 Custom Reports Builder")
    
    report_type = st.selectbox("Report Type", [
        "Market Overview", "Competitive Analysis", "Price Trends", 
        "Feature Comparison", "Consumer Sentiment Analysis"
    ])
    
    # Report parameters
    st.markdown("#### Report Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        time_range = st.selectbox("Time Range", 
                                 ["Last 7 days", "Last 30 days", "Last 3 months", "Last year"])
        brands = st.multiselect("Focus Brands", 
                               ["Apple", "Samsung", "Google", "OnePlus", "Xiaomi"],
                               default=["Apple", "Samsung"])
    
    with col2:
        price_range = st.selectbox("Price Range", 
                                  ["All", "Budget (<$300)", "Mid-range ($300-$700)", "Premium (>$700)"])
        include_charts = st.checkbox("Include Charts", value=True)
    
    if st.button("Generate Custom Report"):
        with st.spinner("Generating your custom report..."):
            import time
            time.sleep(3)
        
        st.success("✅ Report generated successfully!")
        
        # Mock report preview
        st.markdown("#### 📊 Report Preview")
        
        # Create sample chart
        sample_data = pd.DataFrame({
            'Brand': ['Apple', 'Samsung', 'Google', 'OnePlus'],
            'Market Share': [35, 28, 12, 8],
            'Avg Rating': [4.2, 4.0, 4.3, 4.1]
        })
        
        fig = px.bar(sample_data, x='Brand', y='Market Share', 
                    title=f"{report_type} - Market Share Analysis")
        st.plotly_chart(fig, use_container_width=True)
        
        st.download_button(
            label="📥 Download Full Report (PDF)",
            data="Sample report content...",
            file_name=f"{report_type.lower().replace(' ', '_')}_report.pdf",
            mime="application/pdf"
        )

def display_usage_dashboard(role_manager):
    """Usage tracking dashboard for business users"""
    st.markdown("### 📈 Usage Analytics")
    
    # Create sample usage data
    dates = pd.date_range(start=datetime.now() - timedelta(days=30), 
                         end=datetime.now(), freq='D')
    usage_data = pd.DataFrame({
        'Date': dates,
        'Searches': [max(0, int(50 + 30 * np.sin(i/5) + np.random.normal(0, 10))) 
                    for i in range(len(dates))],
        'API Calls': [max(0, int(20 + 15 * np.sin(i/3) + np.random.normal(0, 5))) 
                     for i in range(len(dates))]
    })
    
    # Usage trend chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=usage_data['Date'], y=usage_data['Searches'],
                            mode='lines+markers', name='Searches',
                            line=dict(color='blue', width=3)))
    fig.add_trace(go.Scatter(x=usage_data['Date'], y=usage_data['API Calls'],
                            mode='lines+markers', name='API Calls',
                            line=dict(color='green', width=3)))
    
    fig.update_layout(title="30-Day Usage Trends",
                     xaxis_title="Date", yaxis_title="Count",
                     hovermode='x unified')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Usage summary
    col1, col2, col3 = st.columns(3)
    with col1:
        total_searches = usage_data['Searches'].sum()
        st.metric("Total Searches (30d)", total_searches)
    with col2:
        avg_daily = usage_data['Searches'].mean()
        st.metric("Daily Average", f"{avg_daily:.1f}")
    with col3:
        peak_day = usage_data.loc[usage_data['Searches'].idxmax(), 'Searches']
        st.metric("Peak Day", peak_day)

# Import numpy for sample data generation
try:
    import numpy as np
except ImportError:
    # Fallback if numpy not available
    import random
    class np:
        @staticmethod
        def sin(x):
            return random.uniform(-1, 1)
        @staticmethod  
        def random():
            class RandomClass:
                @staticmethod
                def normal(mean, std):
                    return random.gauss(mean, std)
            return RandomClass()