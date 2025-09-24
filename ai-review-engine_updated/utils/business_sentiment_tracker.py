"""
Business Sentiment Tracker
Tracks sentiment and performance metrics for specific phone brands
Designed for integration into the user-friendly app
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict


class BusinessSentimentTracker:
    """
    Tracks sentiment and key metrics for businesses to monitor their phone brand performance
    Integrates seamlessly with the user-friendly app interface
    """
    
    def __init__(self):
        """Initialize the business sentiment tracker"""
        self.tracked_brands = {}  # Store business tracking configurations
        self.alerts = []  # Store active alerts for businesses
        
    def register_business(self, business_name: str, brand_name: str, 
                         alert_thresholds: Optional[Dict] = None) -> str:
        """
        Register a business to track their brand sentiment
        
        Args:
            business_name: Name of the business
            brand_name: Phone brand to track (e.g., "Apple", "Samsung")
            alert_thresholds: Custom alert thresholds for the business
            
        Returns:
            Unique business tracking ID
        """
        business_id = f"{business_name.lower().replace(' ', '_')}_{brand_name.lower()}"
        
        # Default alert thresholds
        default_thresholds = {
            'negative_sentiment_threshold': 0.25,  # Alert if >25% negative
            'rating_drop_threshold': 0.3,          # Alert if rating drops by 0.3
            'review_volume_drop': 0.50,            # Alert if reviews drop by 50%
            'competitor_gap_threshold': 0.15       # Alert if competitor gains 15% advantage
        }
        
        if alert_thresholds:
            default_thresholds.update(alert_thresholds)
        
        self.tracked_brands[business_id] = {
            'business_name': business_name,
            'brand_name': brand_name,
            'registered_date': datetime.now(),
            'alert_thresholds': default_thresholds,
            'last_analysis': None
        }
        
        return business_id
    
    def get_brand_sentiment_overview(self, df: pd.DataFrame, brand_name: str) -> Dict[str, Any]:
        """
        Get comprehensive sentiment overview for a specific brand
        
        Args:
            df: Review data DataFrame
            brand_name: Brand to analyze
            
        Returns:
            Dictionary with brand sentiment metrics
        """
        if df is None or df.empty:
            return self._get_empty_overview()
        
        # Filter data for the specific brand
        brand_data = df[df['brand'].str.lower() == brand_name.lower()] if 'brand' in df.columns else pd.DataFrame()
        
        if brand_data.empty:
            return self._get_empty_overview(brand_name)
        
        # Calculate key metrics
        total_reviews = len(brand_data)
        avg_rating = brand_data['rating'].mean() if 'rating' in brand_data.columns else 0
        
        # Sentiment analysis
        sentiment_counts = brand_data['sentiment_label'].value_counts() if 'sentiment_label' in brand_data.columns else pd.Series()
        positive_rate = (sentiment_counts.get('positive', 0) / total_reviews) * 100 if total_reviews > 0 else 0
        negative_rate = (sentiment_counts.get('negative', 0) / total_reviews) * 100 if total_reviews > 0 else 0
        neutral_rate = (sentiment_counts.get('neutral', 0) / total_reviews) * 100 if total_reviews > 0 else 0
        
        # Recent trends (last 30 days simulation)
        recent_data = brand_data.tail(min(50, len(brand_data)))  # Simulate recent data
        recent_positive_rate = (recent_data['sentiment_label'].eq('positive').mean() * 100) if not recent_data.empty else 0
        
        # Product performance breakdown
        product_performance = {}
        if 'product' in brand_data.columns:
            product_stats = brand_data.groupby('product').agg({
                'rating': 'mean',
                'sentiment_label': lambda x: (x == 'positive').mean() * 100,
                'product': 'count'
            }).rename(columns={'sentiment_label': 'positive_rate', 'product': 'review_count'})
            
            product_performance = product_stats.to_dict('index')
        
        return {
            'brand_name': brand_name,
            'total_reviews': total_reviews,
            'average_rating': avg_rating,
            'positive_sentiment_rate': positive_rate,
            'negative_sentiment_rate': negative_rate,
            'neutral_sentiment_rate': neutral_rate,
            'recent_positive_trend': recent_positive_rate,
            'sentiment_trend': self._calculate_trend(positive_rate, recent_positive_rate),
            'product_performance': product_performance,
            'competitor_comparison': self._get_competitor_comparison(df, brand_name),
            'alerts': self._check_brand_alerts(brand_name, positive_rate, negative_rate, avg_rating),
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def get_competitive_analysis(self, df: pd.DataFrame, focus_brand: str) -> Dict[str, Any]:
        """
        Get competitive analysis comparing the focus brand against competitors
        
        Args:
            df: Review data DataFrame  
            focus_brand: Brand to focus analysis on
            
        Returns:
            Competitive analysis data
        """
        if df is None or df.empty or 'brand' not in df.columns:
            return {}
        
        # Get all brand performance
        brand_stats = df.groupby('brand').agg({
            'rating': 'mean',
            'sentiment_label': lambda x: (x == 'positive').mean() * 100 if len(x) > 0 else 0,
            'brand': 'count'
        }).rename(columns={'sentiment_label': 'positive_rate', 'brand': 'review_count'})
        
        # Sort by positive sentiment rate
        brand_stats = brand_stats.sort_values('positive_rate', ascending=False)
        
        # Find focus brand position
        focus_position = None
        if focus_brand in brand_stats.index:
            brands_list = brand_stats.index.tolist()
            focus_position = brands_list.index(focus_brand) + 1
        
        # Get top competitors
        top_competitors = brand_stats.head(5).to_dict('index')
        
        # Calculate market share approximation
        total_reviews = brand_stats['review_count'].sum()
        market_share = (brand_stats['review_count'] / total_reviews * 100).round(1)
        
        return {
            'focus_brand': focus_brand,
            'market_position': focus_position,
            'total_brands_analyzed': len(brand_stats),
            'brand_rankings': top_competitors,
            'market_share': market_share.to_dict(),
            'focus_brand_metrics': brand_stats.loc[focus_brand].to_dict() if focus_brand in brand_stats.index else {},
            'competitive_gaps': self._identify_competitive_gaps(brand_stats, focus_brand)
        }
    
    def generate_business_alerts(self, df: pd.DataFrame, business_id: str) -> List[Dict]:
        """
        Generate alerts for a registered business
        
        Args:
            df: Review data DataFrame
            business_id: Business tracking ID
            
        Returns:
            List of alert dictionaries
        """
        if business_id not in self.tracked_brands:
            return []
        
        brand_info = self.tracked_brands[business_id]
        brand_name = brand_info['brand_name']
        thresholds = brand_info['alert_thresholds']
        
        overview = self.get_brand_sentiment_overview(df, brand_name)
        alerts = []
        
        # High negative sentiment alert
        if overview['negative_sentiment_rate'] > thresholds['negative_sentiment_threshold'] * 100:
            alerts.append({
                'type': 'high_negative_sentiment',
                'severity': 'high',
                'title': '⚠️ High Negative Sentiment Alert',
                'message': f"{overview['negative_sentiment_rate']:.1f}% of recent reviews are negative",
                'recommendation': 'Review recent feedback and address common complaints',
                'timestamp': datetime.now()
            })
        
        # Low rating alert  
        if overview['average_rating'] < 4.0:
            alerts.append({
                'type': 'low_rating',
                'severity': 'medium',
                'title': '📉 Rating Below Average',
                'message': f"Average rating is {overview['average_rating']:.1f}/5.0",
                'recommendation': 'Focus on improving product quality and user experience',
                'timestamp': datetime.now()
            })
        
        # Competitive disadvantage alert
        competitive_analysis = self.get_competitive_analysis(df, brand_name)
        if competitive_analysis and competitive_analysis.get('market_position', 1) > 3:
            alerts.append({
                'type': 'competitive_position',
                'severity': 'medium',
                'title': '📊 Competitive Position Alert',
                'message': f"Currently ranked #{competitive_analysis['market_position']} in market",
                'recommendation': 'Analyze top competitors and improve key differentiators',
                'timestamp': datetime.now()
            })
        
        # Positive alerts (opportunities)
        if overview['positive_sentiment_rate'] > 80:
            alerts.append({
                'type': 'positive_momentum',
                'severity': 'positive',
                'title': '🚀 Strong Positive Sentiment',
                'message': f"{overview['positive_sentiment_rate']:.1f}% positive sentiment rate",
                'recommendation': 'Leverage this momentum for marketing and expansion',
                'timestamp': datetime.now()
            })
        
        return alerts
    
    def create_sentiment_dashboard_chart(self, df: pd.DataFrame, brand_name: str) -> go.Figure:
        """
        Create a comprehensive sentiment dashboard chart for a brand
        
        Args:
            df: Review data DataFrame
            brand_name: Brand to visualize
            
        Returns:
            Plotly figure for dashboard
        """
        overview = self.get_brand_sentiment_overview(df, brand_name)
        
        # Create subplot figure
        fig = go.Figure()
        
        # Sentiment distribution pie chart
        sentiments = ['Positive', 'Neutral', 'Negative']
        values = [
            overview['positive_sentiment_rate'],
            overview['neutral_sentiment_rate'], 
            overview['negative_sentiment_rate']
        ]
        colors = ['#4CAF50', '#FF9800', '#F44336']
        
        fig.add_trace(go.Pie(
            labels=sentiments,
            values=values,
            hole=0.4,
            marker_colors=colors,
            textinfo='label+percent',
            textfont_size=12,
            name="Sentiment Distribution"
        ))
        
        fig.update_layout(
            title=f"📊 {brand_name} Sentiment Overview",
            title_font_size=20,
            title_x=0.5,
            showlegend=True,
            height=400,
            annotations=[dict(text=f'{overview["average_rating"]:.1f}⭐<br>Rating', x=0.5, y=0.5, font_size=16, showarrow=False)]
        )
        
        return fig
    
    def create_competitive_comparison_chart(self, df: pd.DataFrame, focus_brand: str) -> go.Figure:
        """
        Create competitive comparison chart
        
        Args:
            df: Review data DataFrame
            focus_brand: Brand to focus on
            
        Returns:
            Plotly figure showing competitive positioning
        """
        competitive_data = self.get_competitive_analysis(df, focus_brand)
        
        if not competitive_data or not competitive_data.get('brand_rankings'):
            return go.Figure().add_annotation(text="No competitive data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        
        brands = list(competitive_data['brand_rankings'].keys())
        positive_rates = [competitive_data['brand_rankings'][brand]['positive_rate'] for brand in brands]
        ratings = [competitive_data['brand_rankings'][brand]['rating'] for brand in brands]
        
        # Highlight focus brand
        colors = ['#1f77b4' if brand != focus_brand else '#ff7f0e' for brand in brands]
        
        fig = go.Figure()
        
        # Add scatter plot
        fig.add_trace(go.Scatter(
            x=positive_rates,
            y=ratings,
            mode='markers+text',
            text=brands,
            textposition='top center',
            marker=dict(size=12, color=colors, line=dict(width=2, color='white')),
            name='Brands'
        ))
        
        fig.update_layout(
            title=f"🏆 Competitive Positioning: {focus_brand} vs Competitors",
            xaxis_title="Positive Sentiment Rate (%)",
            yaxis_title="Average Rating",
            height=500,
            showlegend=False
        )
        
        return fig
    
    def _get_empty_overview(self, brand_name: str = "Unknown") -> Dict[str, Any]:
        """Return empty overview when no data is available"""
        return {
            'brand_name': brand_name,
            'total_reviews': 0,
            'average_rating': 0,
            'positive_sentiment_rate': 0,
            'negative_sentiment_rate': 0,
            'neutral_sentiment_rate': 0,
            'recent_positive_trend': 0,
            'sentiment_trend': 'stable',
            'product_performance': {},
            'competitor_comparison': {},
            'alerts': [],
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def _calculate_trend(self, current_rate: float, recent_rate: float) -> str:
        """Calculate sentiment trend direction"""
        diff = recent_rate - current_rate
        if diff > 2:
            return 'improving'
        elif diff < -2:
            return 'declining'
        else:
            return 'stable'
    
    def _get_competitor_comparison(self, df: pd.DataFrame, brand_name: str) -> Dict:
        """Get basic competitor comparison"""
        if df is None or 'brand' not in df.columns:
            return {}
            
        brand_stats = df.groupby('brand').agg({
            'sentiment_label': lambda x: (x == 'positive').mean() * 100,
            'rating': 'mean'
        }).round(2)
        
        if brand_name in brand_stats.index:
            brand_rank = brand_stats['sentiment_label'].rank(ascending=False)[brand_name]
            return {
                'market_position': int(brand_rank),
                'total_competitors': len(brand_stats)
            }
        
        return {}
    
    def _check_brand_alerts(self, brand_name: str, positive_rate: float, 
                          negative_rate: float, avg_rating: float) -> List[str]:
        """Check for basic alerts"""
        alerts = []
        
        if negative_rate > 30:
            alerts.append("High negative sentiment detected")
        if avg_rating < 3.5:
            alerts.append("Below average rating")
        if positive_rate > 85:
            alerts.append("Excellent sentiment - leverage for marketing")
            
        return alerts
    
    def _identify_competitive_gaps(self, brand_stats: pd.DataFrame, focus_brand: str) -> Dict:
        """Identify competitive gaps and opportunities"""
        if focus_brand not in brand_stats.index:
            return {}
        
        focus_metrics = brand_stats.loc[focus_brand]
        top_competitor = brand_stats.iloc[0]  # Best performing brand
        
        gaps = {}
        
        # Sentiment gap
        sentiment_gap = top_competitor['positive_rate'] - focus_metrics['positive_rate']
        if sentiment_gap > 5:
            gaps['sentiment'] = f"Behind by {sentiment_gap:.1f}% in positive sentiment"
        
        # Rating gap  
        rating_gap = top_competitor['rating'] - focus_metrics['rating']
        if rating_gap > 0.2:
            gaps['rating'] = f"Behind by {rating_gap:.1f} stars in average rating"
        
        return gaps
    
    def get_market_trends_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze overall market trends across all phone brands
        
        Args:
            df: Review data DataFrame
            
        Returns:
            Dictionary with market trends data
        """
        if df is None or df.empty:
            return {}
        
        # Overall market metrics
        total_reviews = len(df)
        overall_avg_rating = df['rating'].mean() if 'rating' in df.columns else 0
        
        # Brand market share by review volume
        if 'brand' in df.columns:
            brand_volumes = df['brand'].value_counts()
            market_share = (brand_volumes / total_reviews * 100).round(1)
        else:
            brand_volumes = pd.Series()
            market_share = pd.Series()
        
        # Sentiment trends across market
        if 'sentiment_label' in df.columns:
            overall_sentiment = df['sentiment_label'].value_counts(normalize=True) * 100
            positive_trend = overall_sentiment.get('positive', 0)
            negative_trend = overall_sentiment.get('negative', 0)
        else:
            positive_trend = 0
            negative_trend = 0
        
        # Price segment analysis (simulated based on brand positioning)
        price_segments = {
            'Premium': ['Apple', 'Samsung'],
            'Mid-Range': ['Google', 'OnePlus'], 
            'Budget': ['Xiaomi', 'Nothing', 'Huawei']
        }
        
        segment_performance = {}
        for segment, brands in price_segments.items():
            segment_data = df[df['brand'].isin(brands)] if 'brand' in df.columns else pd.DataFrame()
            if not segment_data.empty:
                segment_performance[segment] = {
                    'review_volume': len(segment_data),
                    'avg_rating': segment_data['rating'].mean() if 'rating' in segment_data.columns else 0,
                    'market_share': len(segment_data) / total_reviews * 100 if total_reviews > 0 else 0,
                    'sentiment_positive': (segment_data['sentiment_label'] == 'positive').mean() * 100 if 'sentiment_label' in segment_data.columns else 0
                }
        
        # Feature importance trends (simulated based on review content)
        feature_trends = {}
        if 'review_text' in df.columns:
            all_reviews = ' '.join(df['review_text'].dropna().astype(str)).lower()
            
            feature_keywords = {
                'Camera': ['camera', 'photo', 'picture', 'lens', 'zoom'],
                'Battery': ['battery', 'charge', 'power', 'lasting'],
                'Performance': ['fast', 'speed', 'lag', 'smooth', 'performance'],
                'Display': ['screen', 'display', 'bright', 'resolution'],
                'Design': ['design', 'look', 'beautiful', 'premium', 'build']
            }
            
            for feature, keywords in feature_keywords.items():
                mentions = sum(all_reviews.count(keyword) for keyword in keywords)
                feature_trends[feature] = {
                    'mention_count': mentions,
                    'importance_score': min(mentions / 10, 100)  # Normalize to 0-100
                }
        
        return {
            'total_market_reviews': total_reviews,
            'overall_avg_rating': overall_avg_rating,
            'market_sentiment': {
                'positive_rate': positive_trend,
                'negative_rate': negative_trend
            },
            'brand_market_share': market_share.to_dict() if not market_share.empty else {},
            'price_segment_performance': segment_performance,
            'feature_importance_trends': feature_trends,
            'market_leaders': market_share.head(3).to_dict() if not market_share.empty else {},
            'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def get_quarterly_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Simulate quarterly trends for market analysis
        
        Args:
            df: Review data DataFrame
            
        Returns:
            Dictionary with quarterly trend data
        """
        if df is None or df.empty:
            return {}
        
        # Simulate quarterly data by segmenting the dataset
        total_reviews = len(df)
        quarter_size = total_reviews // 4
        
        quarterly_data = {}
        quarters = ['Q1 2024', 'Q2 2024', 'Q3 2024', 'Q4 2024']
        
        for i, quarter in enumerate(quarters):
            start_idx = i * quarter_size
            end_idx = (i + 1) * quarter_size if i < 3 else total_reviews
            quarter_df = df.iloc[start_idx:end_idx]
            
            if not quarter_df.empty:
                quarterly_data[quarter] = {
                    'total_reviews': len(quarter_df),
                    'avg_rating': quarter_df['rating'].mean() if 'rating' in quarter_df.columns else 0,
                    'positive_sentiment': (quarter_df['sentiment_label'] == 'positive').mean() * 100 if 'sentiment_label' in quarter_df.columns else 0
                }
        
        # Calculate growth rates
        growth_metrics = {}
        if len(quarterly_data) >= 2:
            quarters_list = list(quarterly_data.keys())
            latest_quarter = quarterly_data[quarters_list[-1]]
            previous_quarter = quarterly_data[quarters_list[-2]]
            
            growth_metrics = {
                'review_growth': ((latest_quarter['total_reviews'] - previous_quarter['total_reviews']) / previous_quarter['total_reviews'] * 100) if previous_quarter['total_reviews'] > 0 else 0,
                'rating_change': latest_quarter['avg_rating'] - previous_quarter['avg_rating'],
                'sentiment_change': latest_quarter['positive_sentiment'] - previous_quarter['positive_sentiment']
            }
        
        return {
            'quarterly_performance': quarterly_data,
            'growth_metrics': growth_metrics,
            'trend_direction': self._analyze_trend_direction(quarterly_data)
        }
    
    def _analyze_trend_direction(self, quarterly_data: Dict) -> str:
        """Analyze overall trend direction from quarterly data"""
        if len(quarterly_data) < 2:
            return 'insufficient_data'
        
        quarters = list(quarterly_data.keys())
        ratings = [quarterly_data[q]['avg_rating'] for q in quarters]
        
        if len(ratings) >= 2:
            if ratings[-1] > ratings[-2]:
                return 'improving'
            elif ratings[-1] < ratings[-2]:
                return 'declining'
            else:
                return 'stable'
        
        return 'stable'
    
    def create_market_trends_chart(self, df: pd.DataFrame) -> go.Figure:
        """
        Create a simplified market trends chart focusing on data clarity
        
        Args:
            df: Review data DataFrame
            
        Returns:
            Plotly figure for market trends
        """
        trends = self.get_market_trends_analysis(df)
        
        if not trends or not trends.get('brand_market_share'):
            fig = go.Figure()
            fig.add_annotation(
                text="No market data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        # Simple bar chart showing market share
        brands = list(trends['brand_market_share'].keys())
        shares = list(trends['brand_market_share'].values())
        
        fig = go.Figure(data=[
            go.Bar(
                x=brands,
                y=shares,
                marker_color=['#2E7D32', '#1976D2', '#F57C00', '#7B1FA2', '#C62828'][:len(brands)],
                text=[f'{share:.1f}%' for share in shares],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title=dict(
                text="Market Share by Review Volume",
                x=0.5,
                font=dict(size=16)
            ),
            xaxis_title="Brand",
            yaxis_title="Market Share (%)",
            height=400,
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
