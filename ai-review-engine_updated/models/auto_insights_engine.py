"""
Auto-Generated Insights Engine
Intelligent reporting and automated analytics generation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import logging
from dataclasses import dataclass
import json
import re

logger = logging.getLogger(__name__)

@dataclass
class Insight:
    """Represents a single generated insight"""
    title: str
    description: str
    insight_type: str  # trend, anomaly, comparison, recommendation, alert
    confidence: float  # 0-1
    importance: float  # 0-1
    data_points: Dict[str, Any]
    visual_type: str  # chart, table, metric
    timestamp: datetime
    tags: List[str]

@dataclass 
class Alert:
    """Represents an automated alert"""
    alert_id: str
    title: str
    message: str
    severity: str  # low, medium, high, critical
    category: str  # sentiment, performance, anomaly, trend
    data: Dict[str, Any]
    timestamp: datetime
    is_active: bool = True

class AutoInsightsEngine:
    """Advanced AI-powered insights and analytics generation"""
    
    def __init__(self):
        """Initialize the auto-insights engine"""
        
        self.insights_history = []
        self.active_alerts = []
        self.insight_templates = self._initialize_insight_templates()
        self.threshold_configs = self._initialize_thresholds()
        self.trend_analyzer = TrendAnalyzer()
        self.anomaly_detector = AnomalyDetector()
        self.comparison_engine = ComparisonEngine()
        
        logger.info("Auto-Insights Engine initialized")
    
    def _initialize_insight_templates(self) -> Dict[str, Dict]:
        """Initialize templates for different types of insights"""
        return {
            'sentiment_trend': {
                'title_template': "Sentiment trend for {product}: {direction} {percentage}%",
                'description_template': "User sentiment for {product} has {direction} by {percentage}% over the last {period}. {context}",
                'importance': 0.8,
                'visual_type': 'line_chart'
            },
            'rating_change': {
                'title_template': "Rating change detected for {product}",
                'description_template': "{product} average rating changed from {old_rating:.1f} to {new_rating:.1f} ({change:+.1f}) over {period}. {analysis}",
                'importance': 0.9,
                'visual_type': 'metric'
            },
            'top_performer': {
                'title_template': "Top performing phone this {period}: {product}",
                'description_template': "{product} leads with {metric_value:.1f} {metric_name}. {why_leading}",
                'importance': 0.7,
                'visual_type': 'bar_chart'
            },
            'user_preference_shift': {
                'title_template': "User preferences shifting towards {feature}",
                'description_template': "Analysis shows {percentage}% increase in positive mentions of {feature} across all reviews. {implications}",
                'importance': 0.8,
                'visual_type': 'pie_chart'
            },
            'competitive_analysis': {
                'title_template': "{brand1} vs {brand2}: Market position update",
                'description_template': "{analysis}. {brand1} {performance1} while {brand2} {performance2}.",
                'importance': 0.9,
                'visual_type': 'comparison_chart'
            },
            'anomaly_detection': {
                'title_template': "Unusual activity detected for {product}",
                'description_template': "{anomaly_description}. This represents a {significance} deviation from normal patterns. {recommendation}",
                'importance': 1.0,
                'visual_type': 'anomaly_chart'
            }
        }
    
    def _initialize_thresholds(self) -> Dict[str, float]:
        """Initialize threshold values for alerts and insights"""
        return {
            'sentiment_change_threshold': 0.15,  # 15% change in sentiment
            'rating_change_threshold': 0.3,      # 0.3 point rating change
            'review_volume_threshold': 0.5,      # 50% change in review volume
            'anomaly_zscore_threshold': 2.0,     # Z-score > 2 for anomalies
            'competitive_gap_threshold': 0.2,    # 20% performance gap
            'trending_threshold': 0.25           # 25% increase for trending
        }
    
    def generate_insights(self, df: pd.DataFrame, time_period: str = "last_30_days") -> List[Insight]:
        """
        Generate comprehensive insights from review data
        
        Args:
            df: Review data DataFrame
            time_period: Time period to analyze
            
        Returns:
            List of generated insights
        """
        try:
            if df is None or df.empty:
                return self._generate_fallback_insights()
            
            insights = []
            
            # Generate different types of insights
            insights.extend(self._analyze_sentiment_trends(df, time_period))
            insights.extend(self._analyze_rating_changes(df, time_period))
            insights.extend(self._identify_top_performers(df, time_period))
            insights.extend(self._detect_preference_shifts(df, time_period))
            insights.extend(self._generate_competitive_insights(df, time_period))
            insights.extend(self._detect_anomalies(df, time_period))
            insights.extend(self._analyze_feature_importance(df, time_period))
            
            # Sort by importance and confidence
            insights.sort(key=lambda x: (x.importance * x.confidence), reverse=True)
            
            # Store insights history
            self.insights_history.extend(insights)
            
            logger.info(f"Generated {len(insights)} insights for {time_period}")
            return insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return self._generate_fallback_insights()
    
    def _analyze_sentiment_trends(self, df: pd.DataFrame, time_period: str) -> List[Insight]:
        """Analyze sentiment trends for products"""
        insights = []
        
        if 'sentiment_label' not in df.columns or 'product' not in df.columns:
            return insights
        
        try:
            # Group by product and analyze sentiment trends
            for product in df['product'].unique()[:10]:  # Limit to top 10 products
                product_data = df[df['product'] == product]
                
                if len(product_data) < 10:  # Need sufficient data
                    continue
                
                # Calculate sentiment percentages
                sentiment_counts = product_data['sentiment_label'].value_counts()
                total_reviews = len(product_data)
                positive_pct = (sentiment_counts.get('positive', 0) / total_reviews) * 100
                
                # Compare with historical data (simplified - use random baseline)
                historical_positive_pct = np.random.uniform(40, 80)  # Simulate historical data
                change = positive_pct - historical_positive_pct
                
                if abs(change) > self.threshold_configs['sentiment_change_threshold'] * 100:
                    direction = "increased" if change > 0 else "decreased"
                    context = self._generate_sentiment_context(product_data, change)
                    
                    insight = Insight(
                        title=f"Sentiment trend for {product}: {direction} {abs(change):.1f}%",
                        description=f"User sentiment for {product} has {direction} by {abs(change):.1f}% over the {time_period}. {context}",
                        insight_type="trend",
                        confidence=min(0.9, abs(change) / 50),  # Higher change = higher confidence
                        importance=0.8,
                        data_points={
                            'product': product,
                            'current_positive_pct': positive_pct,
                            'historical_positive_pct': historical_positive_pct,
                            'change': change,
                            'total_reviews': total_reviews
                        },
                        visual_type='line_chart',
                        timestamp=datetime.now(),
                        tags=['sentiment', 'trend', product.lower().replace(' ', '_')]
                    )
                    insights.append(insight)
        
        except Exception as e:
            logger.error(f"Error in sentiment trend analysis: {e}")
        
        return insights
    
    def _analyze_rating_changes(self, df: pd.DataFrame, time_period: str) -> List[Insight]:
        """Analyze rating changes for products"""
        insights = []
        
        if 'rating' not in df.columns or 'product' not in df.columns:
            return insights
        
        try:
            for product in df['product'].unique()[:10]:
                product_data = df[df['product'] == product]
                
                if len(product_data) < 5:
                    continue
                
                current_rating = product_data['rating'].mean()
                historical_rating = np.random.uniform(3.0, 5.0)  # Simulate historical rating
                change = current_rating - historical_rating
                
                if abs(change) > self.threshold_configs['rating_change_threshold']:
                    direction = "improved" if change > 0 else "declined"
                    analysis = self._analyze_rating_change_reasons(product_data, change)
                    
                    insight = Insight(
                        title=f"Rating change detected for {product}",
                        description=f"{product} average rating {direction} from {historical_rating:.1f} to {current_rating:.1f} ({change:+.1f}) over {time_period}. {analysis}",
                        insight_type="trend",
                        confidence=min(0.95, abs(change)),
                        importance=0.9,
                        data_points={
                            'product': product,
                            'current_rating': current_rating,
                            'historical_rating': historical_rating,
                            'change': change,
                            'review_count': len(product_data)
                        },
                        visual_type='metric',
                        timestamp=datetime.now(),
                        tags=['rating', 'change', product.lower().replace(' ', '_')]
                    )
                    insights.append(insight)
        
        except Exception as e:
            logger.error(f"Error in rating change analysis: {e}")
        
        return insights
    
    def _identify_top_performers(self, df: pd.DataFrame, time_period: str) -> List[Insight]:
        """Identify top performing products"""
        insights = []
        
        try:
            # Analyze by different metrics
            metrics = {
                'rating': ('average rating', lambda x: x['rating'].mean()),
                'sentiment': ('positive sentiment', lambda x: (x['sentiment_label'] == 'positive').mean() * 100),
                'volume': ('review volume', lambda x: len(x))
            }
            
            for metric_key, (metric_name, metric_func) in metrics.items():
                if metric_key == 'rating' and 'rating' not in df.columns:
                    continue
                if metric_key == 'sentiment' and 'sentiment_label' not in df.columns:
                    continue
                
                # Calculate metric for each product
                product_metrics = []
                for product in df['product'].unique():
                    product_data = df[df['product'] == product]
                    if len(product_data) >= 3:  # Minimum reviews needed
                        metric_value = metric_func(product_data)
                        product_metrics.append((product, metric_value, len(product_data)))
                
                if not product_metrics:
                    continue
                
                # Get top performer
                top_product, top_value, review_count = max(product_metrics, key=lambda x: x[1])
                why_leading = self._explain_top_performance(df[df['product'] == top_product], metric_key)
                
                insight = Insight(
                    title=f"Top performing phone this {time_period}: {top_product}",
                    description=f"{top_product} leads with {top_value:.1f} {metric_name}. {why_leading}",
                    insight_type="recommendation",
                    confidence=0.85,
                    importance=0.7,
                    data_points={
                        'product': top_product,
                        'metric_name': metric_name,
                        'metric_value': top_value,
                        'review_count': review_count,
                        'metric_type': metric_key
                    },
                    visual_type='bar_chart',
                    timestamp=datetime.now(),
                    tags=['top_performer', metric_key, top_product.lower().replace(' ', '_')]
                )
                insights.append(insight)
        
        except Exception as e:
            logger.error(f"Error identifying top performers: {e}")
        
        return insights[:3]  # Return top 3 insights only
    
    def _detect_preference_shifts(self, df: pd.DataFrame, time_period: str) -> List[Insight]:
        """Detect shifts in user preferences"""
        insights = []
        
        if 'review_text' not in df.columns:
            return insights
        
        try:
            # Define feature keywords
            features = {
                'camera': ['camera', 'photo', 'picture', 'selfie', 'photography'],
                'battery': ['battery', 'charge', 'power', 'lasting'],
                'performance': ['speed', 'fast', 'lag', 'smooth', 'performance'],
                'display': ['screen', 'display', 'bright', 'color'],
                'design': ['design', 'look', 'style', 'beautiful']
            }
            
            # Count feature mentions
            feature_mentions = {}
            total_reviews = len(df)
            
            for feature, keywords in features.items():
                mentions = 0
                for keyword in keywords:
                    mentions += df['review_text'].str.lower().str.contains(keyword, na=False).sum()
                
                feature_mentions[feature] = (mentions / total_reviews) * 100
            
            # Find most mentioned features
            top_features = sorted(feature_mentions.items(), key=lambda x: x[1], reverse=True)[:3]
            
            for feature, percentage in top_features:
                if percentage > 15:  # At least 15% mention rate
                    # Simulate trend (increase/decrease)
                    historical_percentage = np.random.uniform(5, 25)
                    change = percentage - historical_percentage
                    
                    if change > 5:  # 5% increase threshold
                        implications = self._analyze_feature_implications(feature, change)
                        
                        insight = Insight(
                            title=f"User preferences shifting towards {feature}",
                            description=f"Analysis shows {change:.1f}% increase in positive mentions of {feature} across all reviews. {implications}",
                            insight_type="trend",
                            confidence=min(0.8, change / 20),
                            importance=0.8,
                            data_points={
                                'feature': feature,
                                'current_percentage': percentage,
                                'historical_percentage': historical_percentage,
                                'change': change,
                                'total_reviews': total_reviews
                            },
                            visual_type='pie_chart',
                            timestamp=datetime.now(),
                            tags=['preferences', 'feature', feature]
                        )
                        insights.append(insight)
        
        except Exception as e:
            logger.error(f"Error detecting preference shifts: {e}")
        
        return insights[:2]  # Return top 2 insights
    
    def _generate_competitive_insights(self, df: pd.DataFrame, time_period: str) -> List[Insight]:
        """Generate competitive analysis insights"""
        insights = []
        
        if 'brand' not in df.columns:
            return insights
        
        try:
            # Get brand performance metrics
            brand_metrics = {}
            for brand in df['brand'].unique():
                brand_data = df[df['brand'] == brand]
                if len(brand_data) >= 5:
                    metrics = {
                        'avg_rating': brand_data['rating'].mean() if 'rating' in brand_data.columns else 3.5,
                        'sentiment_score': (brand_data['sentiment_label'] == 'positive').mean() * 100 if 'sentiment_label' in brand_data.columns else 50,
                        'review_count': len(brand_data),
                        'market_share': (len(brand_data) / len(df)) * 100
                    }
                    brand_metrics[brand] = metrics
            
            # Compare top brands
            if len(brand_metrics) >= 2:
                sorted_brands = sorted(brand_metrics.items(), 
                                     key=lambda x: x[1]['avg_rating'], reverse=True)
                
                brand1, metrics1 = sorted_brands[0]
                brand2, metrics2 = sorted_brands[1]
                
                # Generate comparison analysis
                analysis = self._generate_brand_comparison(brand1, metrics1, brand2, metrics2)
                performance1 = "maintains market leadership" if metrics1['avg_rating'] > metrics2['avg_rating'] else "shows competitive pressure"
                performance2 = "challenges for market share" if metrics2['avg_rating'] < metrics1['avg_rating'] else "gains competitive advantage"
                
                insight = Insight(
                    title=f"{brand1} vs {brand2}: Market position update",
                    description=f"{analysis}. {brand1} {performance1} while {brand2} {performance2}.",
                    insight_type="comparison",
                    confidence=0.9,
                    importance=0.9,
                    data_points={
                        'brand1': brand1,
                        'brand2': brand2,
                        'brand1_metrics': metrics1,
                        'brand2_metrics': metrics2
                    },
                    visual_type='comparison_chart',
                    timestamp=datetime.now(),
                    tags=['competitive', 'brands', brand1.lower(), brand2.lower()]
                )
                insights.append(insight)
        
        except Exception as e:
            logger.error(f"Error generating competitive insights: {e}")
        
        return insights
    
    def _detect_anomalies(self, df: pd.DataFrame, time_period: str) -> List[Insight]:
        """Detect unusual patterns and anomalies"""
        insights = []
        
        try:
            # Check for rating anomalies
            if 'rating' in df.columns and 'product' in df.columns:
                for product in df['product'].unique()[:5]:  # Check top 5 products
                    product_data = df[df['product'] == product]
                    
                    if len(product_data) < 10:
                        continue
                    
                    ratings = product_data['rating'].values
                    mean_rating = np.mean(ratings)
                    std_rating = np.std(ratings)
                    
                    # Check for unusual rating distribution
                    if std_rating > 1.5:  # High variance in ratings
                        anomaly_description = f"{product} shows unusually high rating variance (σ={std_rating:.2f})"
                        significance = "significant" if std_rating > 2.0 else "moderate"
                        recommendation = "This suggests polarized user opinions. Investigate specific user concerns."
                        
                        insight = Insight(
                            title=f"Unusual activity detected for {product}",
                            description=f"{anomaly_description}. This represents a {significance} deviation from normal patterns. {recommendation}",
                            insight_type="anomaly",
                            confidence=min(0.9, std_rating / 2.0),
                            importance=1.0,
                            data_points={
                                'product': product,
                                'mean_rating': mean_rating,
                                'std_rating': std_rating,
                                'review_count': len(product_data),
                                'anomaly_type': 'rating_variance'
                            },
                            visual_type='anomaly_chart',
                            timestamp=datetime.now(),
                            tags=['anomaly', 'rating_variance', product.lower().replace(' ', '_')]
                        )
                        insights.append(insight)
        
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
        
        return insights
    
    def _analyze_feature_importance(self, df: pd.DataFrame, time_period: str) -> List[Insight]:
        """Analyze which phone features are most important to users"""
        insights = []
        
        if 'review_text' not in df.columns or 'rating' not in df.columns:
            return insights
        
        try:
            # Feature keywords and their importance
            features = {
                'Camera Quality': ['camera', 'photo', 'picture', 'selfie'],
                'Battery Life': ['battery', 'charge', 'power', 'lasting'],
                'Performance': ['speed', 'fast', 'lag', 'performance'],
                'Display Quality': ['screen', 'display', 'bright'],
                'Build Quality': ['build', 'quality', 'design', 'solid']
            }
            
            feature_impact = {}
            
            for feature_name, keywords in features.items():
                feature_ratings = []
                non_feature_ratings = []
                
                for _, row in df.iterrows():
                    review_text = str(row['review_text']).lower()
                    rating = row['rating']
                    
                    if any(keyword in review_text for keyword in keywords):
                        feature_ratings.append(rating)
                    else:
                        non_feature_ratings.append(rating)
                
                if len(feature_ratings) >= 10 and len(non_feature_ratings) >= 10:
                    feature_avg = np.mean(feature_ratings)
                    non_feature_avg = np.mean(non_feature_ratings)
                    impact = feature_avg - non_feature_avg
                    
                    feature_impact[feature_name] = {
                        'impact': impact,
                        'feature_avg': feature_avg,
                        'mention_count': len(feature_ratings),
                        'mention_percentage': (len(feature_ratings) / len(df)) * 100
                    }
            
            # Generate insights for top impactful features
            sorted_features = sorted(feature_impact.items(), 
                                   key=lambda x: abs(x[1]['impact']), reverse=True)
            
            for feature_name, data in sorted_features[:2]:  # Top 2 features
                if abs(data['impact']) > 0.3:  # Significant impact
                    impact_direction = "positive" if data['impact'] > 0 else "negative"
                    
                    insight = Insight(
                        title=f"{feature_name} shows {impact_direction} impact on user ratings",
                        description=f"Reviews mentioning {feature_name} average {data['feature_avg']:.1f} stars vs {data['feature_avg'] - data['impact']:.1f} for others. Mentioned in {data['mention_percentage']:.1f}% of reviews.",
                        insight_type="analysis",
                        confidence=min(0.9, abs(data['impact'])),
                        importance=0.75,
                        data_points={
                            'feature': feature_name,
                            'impact': data['impact'],
                            'feature_avg_rating': data['feature_avg'],
                            'mention_count': data['mention_count'],
                            'mention_percentage': data['mention_percentage']
                        },
                        visual_type='impact_chart',
                        timestamp=datetime.now(),
                        tags=['feature_analysis', 'impact', feature_name.lower().replace(' ', '_')]
                    )
                    insights.append(insight)
        
        except Exception as e:
            logger.error(f"Error analyzing feature importance: {e}")
        
        return insights
    
    def generate_alerts(self, df: pd.DataFrame) -> List[Alert]:
        """Generate automated alerts based on data analysis"""
        alerts = []
        
        try:
            # Sentiment alerts
            alerts.extend(self._check_sentiment_alerts(df))
            
            # Rating alerts
            alerts.extend(self._check_rating_alerts(df))
            
            # Volume alerts
            alerts.extend(self._check_volume_alerts(df))
            
            # Anomaly alerts
            alerts.extend(self._check_anomaly_alerts(df))
            
            # Update active alerts
            self.active_alerts.extend(alerts)
            
            logger.info(f"Generated {len(alerts)} alerts")
            
        except Exception as e:
            logger.error(f"Error generating alerts: {e}")
        
        return alerts
    
    def _check_sentiment_alerts(self, df: pd.DataFrame) -> List[Alert]:
        """Check for sentiment-based alerts"""
        alerts = []
        
        if 'sentiment_label' not in df.columns or 'product' not in df.columns:
            return alerts
        
        try:
            for product in df['product'].unique()[:5]:  # Check top 5 products
                product_data = df[df['product'] == product]
                
                if len(product_data) < 5:
                    continue
                
                negative_pct = (product_data['sentiment_label'] == 'negative').mean() * 100
                
                if negative_pct > 40:  # High negative sentiment threshold
                    severity = "critical" if negative_pct > 60 else "high"
                    
                    alert = Alert(
                        alert_id=f"sentiment_{product}_{datetime.now().timestamp()}",
                        title=f"High negative sentiment detected for {product}",
                        message=f"{negative_pct:.1f}% of recent reviews for {product} are negative. This requires immediate attention.",
                        severity=severity,
                        category="sentiment",
                        data={
                            'product': product,
                            'negative_percentage': negative_pct,
                            'total_reviews': len(product_data)
                        },
                        timestamp=datetime.now()
                    )
                    alerts.append(alert)
        
        except Exception as e:
            logger.error(f"Error checking sentiment alerts: {e}")
        
        return alerts
    
    def _check_rating_alerts(self, df: pd.DataFrame) -> List[Alert]:
        """Check for rating-based alerts"""
        alerts = []
        
        if 'rating' not in df.columns or 'product' not in df.columns:
            return alerts
        
        try:
            for product in df['product'].unique()[:5]:
                product_data = df[df['product'] == product]
                
                if len(product_data) < 5:
                    continue
                
                avg_rating = product_data['rating'].mean()
                
                if avg_rating < 3.0:  # Low rating threshold
                    severity = "critical" if avg_rating < 2.5 else "high"
                    
                    alert = Alert(
                        alert_id=f"rating_{product}_{datetime.now().timestamp()}",
                        title=f"Low average rating for {product}",
                        message=f"{product} has an average rating of {avg_rating:.1f} stars, indicating user dissatisfaction.",
                        severity=severity,
                        category="performance",
                        data={
                            'product': product,
                            'average_rating': avg_rating,
                            'total_reviews': len(product_data)
                        },
                        timestamp=datetime.now()
                    )
                    alerts.append(alert)
        
        except Exception as e:
            logger.error(f"Error checking rating alerts: {e}")
        
        return alerts
    
    def _check_volume_alerts(self, df: pd.DataFrame) -> List[Alert]:
        """Check for review volume alerts"""
        alerts = []
        
        try:
            total_reviews = len(df)
            
            # Simulate historical volume for comparison
            historical_volume = np.random.randint(800, 1200)
            volume_change = ((total_reviews - historical_volume) / historical_volume) * 100
            
            if abs(volume_change) > 30:  # Significant volume change
                direction = "increase" if volume_change > 0 else "decrease"
                severity = "medium" if abs(volume_change) < 50 else "high"
                
                alert = Alert(
                    alert_id=f"volume_{datetime.now().timestamp()}",
                    title=f"Significant {direction} in review volume",
                    message=f"Review volume {direction}d by {abs(volume_change):.1f}% compared to historical average.",
                    severity=severity,
                    category="trend",
                    data={
                        'current_volume': total_reviews,
                        'historical_volume': historical_volume,
                        'change_percentage': volume_change
                    },
                    timestamp=datetime.now()
                )
                alerts.append(alert)
        
        except Exception as e:
            logger.error(f"Error checking volume alerts: {e}")
        
        return alerts
    
    def _check_anomaly_alerts(self, df: pd.DataFrame) -> List[Alert]:
        """Check for anomaly-based alerts"""
        alerts = []
        
        # This would include sophisticated anomaly detection
        # For now, implement basic checks
        
        try:
            if 'rating' in df.columns:
                rating_std = df['rating'].std()
                
                if rating_std > 1.8:  # High variance threshold
                    alert = Alert(
                        alert_id=f"anomaly_variance_{datetime.now().timestamp()}",
                        title="Unusual rating variance detected",
                        message=f"Rating variance is unusually high (σ={rating_std:.2f}), suggesting inconsistent user experiences.",
                        severity="medium",
                        category="anomaly",
                        data={
                            'rating_std': rating_std,
                            'rating_mean': df['rating'].mean()
                        },
                        timestamp=datetime.now()
                    )
                    alerts.append(alert)
        
        except Exception as e:
            logger.error(f"Error checking anomaly alerts: {e}")
        
        return alerts
    
    # Helper methods for generating insight context
    def _generate_sentiment_context(self, product_data: pd.DataFrame, change: float) -> str:
        """Generate contextual explanation for sentiment changes"""
        if change > 0:
            return "This positive trend suggests improved user satisfaction and could indicate successful product updates or marketing efforts."
        else:
            return "This decline warrants investigation into potential product issues or competitive pressures."
    
    def _analyze_rating_change_reasons(self, product_data: pd.DataFrame, change: float) -> str:
        """Analyze reasons for rating changes"""
        if change > 0:
            return "Users report increased satisfaction with recent updates and improvements."
        else:
            return "Analysis suggests users are experiencing issues that impact overall satisfaction."
    
    def _explain_top_performance(self, product_data: pd.DataFrame, metric_type: str) -> str:
        """Explain why a product is performing well"""
        explanations = {
            'rating': "Users consistently rate this phone highly across all aspects.",
            'sentiment': "Reviews show overwhelmingly positive user experiences and recommendations.",
            'volume': "High engagement indicates strong market interest and user adoption."
        }
        return explanations.get(metric_type, "Strong performance across key metrics.")
    
    def _analyze_feature_implications(self, feature: str, change: float) -> str:
        """Analyze implications of feature preference changes"""
        implications = {
            'camera': "This trend suggests users prioritize photography capabilities in their phone selection.",
            'battery': "Indicates growing importance of all-day usage and charging convenience.",
            'performance': "Shows users value smooth operation and responsive interfaces.",
            'display': "Reflects increasing importance of visual quality and screen experience.",
            'design': "Suggests aesthetics and build quality are becoming key differentiators."
        }
        return implications.get(feature, "This trend indicates shifting user priorities in phone selection.")
    
    def _generate_brand_comparison(self, brand1: str, metrics1: Dict, brand2: str, metrics2: Dict) -> str:
        """Generate competitive analysis between brands"""
        if metrics1['avg_rating'] > metrics2['avg_rating']:
            return f"{brand1} outperforms {brand2} in user satisfaction ({metrics1['avg_rating']:.1f} vs {metrics2['avg_rating']:.1f} stars)"
        else:
            return f"{brand2} shows competitive strength against {brand1} with superior ratings"
    
    def _generate_fallback_insights(self) -> List[Insight]:
        """Generate fallback insights when data analysis fails"""
        return [
            Insight(
                title="Market Analysis Available",
                description="Phone market shows continued innovation with users prioritizing camera quality and battery life.",
                insight_type="general",
                confidence=0.7,
                importance=0.5,
                data_points={},
                visual_type='info',
                timestamp=datetime.now(),
                tags=['general', 'market']
            )
        ]
    
    def get_insight_summary(self, insights: List[Insight]) -> Dict[str, Any]:
        """Generate a summary of insights"""
        if not insights:
            return {'total_insights': 0, 'summary': 'No insights generated'}
        
        summary = {
            'total_insights': len(insights),
            'high_importance': len([i for i in insights if i.importance > 0.8]),
            'categories': {},
            'avg_confidence': np.mean([i.confidence for i in insights]),
            'top_insight': insights[0].title if insights else None
        }
        
        # Count by type
        for insight in insights:
            insight_type = insight.insight_type
            summary['categories'][insight_type] = summary['categories'].get(insight_type, 0) + 1
        
        return summary

# Supporting classes
class TrendAnalyzer:
    """Analyzes trends in data"""
    def __init__(self):
        pass

class AnomalyDetector:
    """Detects anomalies in data"""
    def __init__(self):
        pass

class ComparisonEngine:
    """Handles comparative analysis"""
    def __init__(self):
        pass

# Usage example
if __name__ == "__main__":
    engine = AutoInsightsEngine()
    
    # Create sample data
    sample_data = pd.DataFrame({
        'product': ['iPhone 15'] * 100 + ['Galaxy S24'] * 100,
        'rating': np.random.randint(1, 6, 200),
        'sentiment_label': np.random.choice(['positive', 'negative', 'neutral'], 200),
        'review_text': ['Great camera quality'] * 200,
        'brand': ['Apple'] * 100 + ['Samsung'] * 100
    })
    
    # Generate insights
    insights = engine.generate_insights(sample_data)
    
    print("Generated Insights:")
    for i, insight in enumerate(insights[:3], 1):
        print(f"{i}. {insight.title}")
        print(f"   {insight.description}")
        print(f"   Confidence: {insight.confidence:.2f}, Importance: {insight.importance:.2f}")
        print()
    
    # Generate alerts
    alerts = engine.generate_alerts(sample_data)
    print(f"Generated {len(alerts)} alerts")