"""
Robust and Realistic Review Analysis Engine
Handles missing data, provides confidence levels, and uses fallback strategies
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import logging
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataQuality(Enum):
    """Data quality levels"""
    HIGH = "high"          # >1000 reviews, all aspects covered
    MEDIUM = "medium"      # 100-1000 reviews, most aspects
    LOW = "low"           # 10-100 reviews, some aspects
    INSUFFICIENT = "insufficient"  # <10 reviews
    NO_DATA = "no_data"   # No reviews available

class ConfidenceLevel(Enum):
    """Analysis confidence levels"""
    VERY_HIGH = 0.9    # Strong data, clear patterns
    HIGH = 0.75        # Good data, reliable patterns
    MODERATE = 0.6     # Adequate data, some uncertainty
    LOW = 0.4          # Limited data, high uncertainty
    VERY_LOW = 0.2     # Minimal data, mostly estimates

@dataclass
class AnalysisResult:
    """Structured analysis result with confidence metrics"""
    phone_model: str
    data_quality: DataQuality
    confidence_level: float
    total_reviews: int
    actual_reviews_analyzed: int
    sentiment: Dict[str, float]
    aspects: Dict[str, Any]
    warnings: List[str]
    recommendations: List[str]
    metadata: Dict[str, Any]

class RobustReviewAnalyzer:
    """
    Robust review analyzer that handles incomplete data gracefully
    """
    
    def __init__(self):
        # Default sentiment distributions by phone category
        self.category_baselines = {
            'flagship': {'positive': 0.70, 'neutral': 0.20, 'negative': 0.10},
            'mid-range': {'positive': 0.65, 'neutral': 0.25, 'negative': 0.10},
            'budget': {'positive': 0.60, 'neutral': 0.28, 'negative': 0.12},
            'unknown': {'positive': 0.65, 'neutral': 0.23, 'negative': 0.12}
        }
        
        # Aspect importance weights
        self.aspect_weights = {
            'camera': 0.25,
            'battery': 0.20,
            'performance': 0.20,
            'display': 0.15,
            'price': 0.10,
            'design': 0.05,
            'software': 0.05
        }
        
        # Minimum data thresholds
        self.min_reviews_for_analysis = 5
        self.min_reviews_for_aspect = 3
        self.min_confidence_threshold = 0.3
        
        # Load historical data patterns
        self.historical_patterns = self._load_historical_patterns()
    
    def analyze_phone(self,
        phone_model: str,
        reviews_df: Optional[pd.DataFrame] = None,
        requested_aspects: List[str] = None
    ) -> AnalysisResult:
        """
        Analyze phone reviews with robust handling of missing data
        
        Args:
            phone_model: Phone model name
            reviews_df: DataFrame with reviews (can be None or empty)
            requested_aspects: Specific aspects to analyze
        
        Returns:
            AnalysisResult with confidence metrics and warnings
        """
        
        logger.info(f"Starting robust analysis for {phone_model}")
        
        # Initialize result structure
        warnings = []
        recommendations = []
        
        # Determine phone category for baseline
        category = self._determine_phone_category(phone_model)
        
        # Check data availability
        if reviews_df is None or len(reviews_df) == 0:
            return self._handle_no_data(phone_model, category, requested_aspects)
        
        # Assess data quality
        data_quality = self._assess_data_quality(reviews_df)
        
        # Calculate confidence level
        confidence_level = self._calculate_confidence(len(reviews_df), data_quality)
        
        # Perform sentiment analysis with fallbacks
        sentiment_result = self._analyze_sentiment_robust(
            reviews_df, 
            category, 
            confidence_level
        )
        
        # Analyze aspects with missing data handling
        aspect_result = self._analyze_aspects_robust(
            reviews_df,
            requested_aspects or list(self.aspect_weights.keys()),
            confidence_level
        )
        
        # Generate warnings for missing or low-quality data
        warnings.extend(self._generate_warnings(
            data_quality,
            len(reviews_df),
            aspect_result['missing_aspects']
        ))
        
        # Generate recommendations based on analysis
        recommendations.extend(self._generate_recommendations(
            sentiment_result,
            aspect_result,
            data_quality
        ))
        
        # Compile final result
        return AnalysisResult(
            phone_model=phone_model,
            data_quality=data_quality,
            confidence_level=confidence_level,
            total_reviews=len(reviews_df),
            actual_reviews_analyzed=min(len(reviews_df), 1000),  # Cap at 1000 for performance
            sentiment=sentiment_result,
            aspects=aspect_result,
            warnings=warnings,
            recommendations=recommendations,
            metadata={
                'analysis_date': datetime.now().isoformat(),
                'category': category,
                'fallback_used': data_quality in [DataQuality.LOW, DataQuality.INSUFFICIENT],
                'estimated_metrics': aspect_result.get('estimated', [])
            }
        )
    
    def _handle_no_data(self,
        phone_model: str,
        category: str,
        requested_aspects: List[str]
    ) -> AnalysisResult:
        """Handle case when no review data is available"""
        
        logger.warning(f"No data available for {phone_model}, using estimates")
        
        # Use category baseline for sentiment
        baseline_sentiment = self.category_baselines[category]
        
        # Estimate aspects based on phone category and historical patterns
        estimated_aspects = self._estimate_aspects_from_category(category, requested_aspects)
        
        warnings = [
            "No review data available for this phone model",
            "All metrics are estimated based on similar phones",
            "Please check back later for actual review data"
        ]
        
        recommendations = [
            "Consider looking at reviews for similar models",
            "Check multiple sources for more comprehensive information",
            f"Based on category trends, this {category} phone likely performs well in typical use cases"
        ]
        
        return AnalysisResult(
            phone_model=phone_model,
            data_quality=DataQuality.NO_DATA,
            confidence_level=ConfidenceLevel.VERY_LOW.value,
            total_reviews=0,
            actual_reviews_analyzed=0,
            sentiment={
                'positive': baseline_sentiment['positive'] * 100,
                'neutral': baseline_sentiment['neutral'] * 100,
                'negative': baseline_sentiment['negative'] * 100,
                'confidence': ConfidenceLevel.VERY_LOW.value
            },
            aspects=estimated_aspects,
            warnings=warnings,
            recommendations=recommendations,
            metadata={
                'analysis_date': datetime.now().isoformat(),
                'category': category,
                'all_estimated': True
            }
        )
    
    def _assess_data_quality(self, reviews_df: pd.DataFrame) -> DataQuality:
        """Assess the quality of available review data"""
        
        review_count = len(reviews_df)
        
        if review_count >= 1000:
            return DataQuality.HIGH
        elif review_count >= 100:
            return DataQuality.MEDIUM
        elif review_count >= 10:
            return DataQuality.LOW
        elif review_count >= self.min_reviews_for_analysis:
            return DataQuality.INSUFFICIENT
        else:
            return DataQuality.NO_DATA
    
    def _calculate_confidence(self,
        review_count: int,
        data_quality: DataQuality
    ) -> float:
        """Calculate confidence level based on data availability"""
        
        # Base confidence from data quality
        quality_confidence = {
            DataQuality.HIGH: 0.9,
            DataQuality.MEDIUM: 0.75,
            DataQuality.LOW: 0.5,
            DataQuality.INSUFFICIENT: 0.3,
            DataQuality.NO_DATA: 0.1
        }
        
        base_confidence = quality_confidence[data_quality]
        
        # Adjust based on review count (logarithmic scale)
        if review_count > 0:
            count_factor = min(np.log10(review_count) / 4, 1.0)  # Max at 10,000 reviews
            adjusted_confidence = base_confidence * 0.7 + count_factor * 0.3
        else:
            adjusted_confidence = base_confidence * 0.5
        
        return min(max(adjusted_confidence, self.min_confidence_threshold), 0.95)
    
    def _analyze_sentiment_robust(self,
        reviews_df: pd.DataFrame,
        category: str,
        confidence: float
    ) -> Dict[str, float]:
        """Analyze sentiment with fallback strategies"""
        
        result = {}
        
        # Check if sentiment column exists
        if 'sentiment' in reviews_df.columns:
            # Calculate actual sentiment distribution
            sentiment_counts = reviews_df['sentiment'].value_counts(normalize=True)
            
            result['positive'] = sentiment_counts.get('positive', 0) * 100
            result['neutral'] = sentiment_counts.get('neutral', 0) * 100
            result['negative'] = sentiment_counts.get('negative', 0) * 100
            
        elif 'rating' in reviews_df.columns:
            # Estimate sentiment from ratings
            ratings = reviews_df['rating'].dropna()
            
            if len(ratings) > 0:
                result['positive'] = (ratings >= 4).sum() / len(ratings) * 100
                result['negative'] = (ratings <= 2).sum() / len(ratings) * 100
                result['neutral'] = 100 - result['positive'] - result['negative']
                confidence *= 0.8  # Reduce confidence for estimated sentiment
            else:
                # Use baseline
                baseline = self.category_baselines[category]
                result = {k: v * 100 for k, v in baseline.items()}
                confidence *= 0.5
        else:
            # Use category baseline
            baseline = self.category_baselines[category]
            result = {k: v * 100 for k, v in baseline.items()}
            confidence *= 0.4
        
        # Apply smoothing with baseline for small samples
        if len(reviews_df) < 30:
            baseline = self.category_baselines[category]
            weight = len(reviews_df) / 30  # Linear weight based on sample size
            
            for key in ['positive', 'neutral', 'negative']:
                result[key] = result.get(key, 0) * weight + baseline[key] * 100 * (1 - weight)
        
        result['confidence'] = confidence
        return result
    
    def _analyze_aspects_robust(self,
        reviews_df: pd.DataFrame,
        requested_aspects: List[str],
        confidence: float
    ) -> Dict[str, Any]:
        """Analyze specific aspects with missing data handling"""
        
        aspect_results = {}
        missing_aspects = []
        estimated_aspects = []
        
        for aspect in requested_aspects:
            aspect_data = self._extract_aspect_data(reviews_df, aspect)
            
            if aspect_data['count'] >= self.min_reviews_for_aspect:
                # Sufficient data for this aspect
                aspect_results[aspect] = {
                    'sentiment': aspect_data['sentiment'],
                    'count': aspect_data['count'],
                    'confidence': confidence * (aspect_data['count'] / len(reviews_df)),
                    'keywords': aspect_data['keywords'],
                    'is_estimated': False
                }
            
            elif aspect_data['count'] > 0:
                # Limited data - use with caution
                aspect_results[aspect] = {
                    'sentiment': aspect_data['sentiment'],
                    'count': aspect_data['count'],
                    'confidence': confidence * 0.5,
                    'warning': f"Only {aspect_data['count']} reviews mention {aspect}",
                    'is_estimated': False
                }
                missing_aspects.append(aspect)
            
            else:
                # No data - estimate from similar aspects or baseline
                estimated_value = self._estimate_aspect(aspect, reviews_df, aspect_results)
                aspect_results[aspect] = {
                    'sentiment': estimated_value,
                    'count': 0,
                    'confidence': confidence * 0.3,
                    'warning': f"No specific data for {aspect} - estimated from overall sentiment",
                    'is_estimated': True
                }
                missing_aspects.append(aspect)
                estimated_aspects.append(aspect)
        
        return {
            'details': aspect_results,
            'missing_aspects': missing_aspects,
            'estimated': estimated_aspects,
            'coverage': (len(requested_aspects) - len(missing_aspects)) / len(requested_aspects)
        }
    
    def _extract_aspect_data(self,
        reviews_df: pd.DataFrame,
        aspect: str
    ) -> Dict[str, Any]:
        """Extract data for a specific aspect from reviews"""
        
        aspect_keywords = {
            'camera': ['camera', 'photo', 'picture', 'lens', 'zoom', 'selfie', 'video'],
            'battery': ['battery', 'charging', 'charge', 'power', 'lasting', 'drain'],
            'performance': ['fast', 'slow', 'lag', 'smooth', 'speed', 'processor', 'ram'],
            'display': ['screen', 'display', 'bright', 'color', 'resolution', 'touch'],
            'price': ['price', 'expensive', 'cheap', 'value', 'worth', 'money', 'cost'],
            'design': ['design', 'build', 'quality', 'feel', 'premium', 'plastic', 'metal'],
            'software': ['software', 'ui', 'android', 'ios', 'update', 'bug', 'app']
        }
        
        keywords = aspect_keywords.get(aspect, [aspect])
        
        # Find reviews mentioning this aspect
        aspect_reviews = []
        
        if 'review_text' in reviews_df.columns:
            for _, row in reviews_df.iterrows():
                if pd.notna(row.get('review_text')):
                    text = str(row['review_text']).lower()
                    if any(keyword in text for keyword in keywords):
                        aspect_reviews.append(row)
        
        if not aspect_reviews:
            return {'count': 0, 'sentiment': None, 'keywords': []}
        
        # Analyze sentiment for aspect-specific reviews
        aspect_df = pd.DataFrame(aspect_reviews)
        
        sentiment = {}
        if 'sentiment' in aspect_df.columns:
            sentiment_dist = aspect_df['sentiment'].value_counts(normalize=True)
            sentiment = {
                'positive': sentiment_dist.get('positive', 0) * 100,
                'negative': sentiment_dist.get('negative', 0) * 100,
                'neutral': sentiment_dist.get('neutral', 0) * 100
            }
        elif 'rating' in aspect_df.columns:
            ratings = aspect_df['rating'].dropna()
            if len(ratings) > 0:
                sentiment = {
                    'positive': (ratings >= 4).sum() / len(ratings) * 100,
                    'negative': (ratings <= 2).sum() / len(ratings) * 100,
                    'neutral': 0  # Simplified
                }
                sentiment['neutral'] = 100 - sentiment['positive'] - sentiment['negative']
        
        return {
            'count': len(aspect_reviews),
            'sentiment': sentiment,
            'keywords': keywords[:3]  # Top keywords used
        }
    
    def _estimate_aspect(self,
        aspect: str,
        reviews_df: pd.DataFrame,
        existing_aspects: Dict
    ) -> Dict[str, float]:
        """Estimate aspect sentiment when no direct data available"""
        
        # Strategy 1: Use overall sentiment as baseline
        if 'sentiment' in reviews_df.columns:
            overall_sentiment = reviews_df['sentiment'].value_counts(normalize=True)
            base_estimate = {
                'positive': overall_sentiment.get('positive', 0) * 100,
                'negative': overall_sentiment.get('negative', 0) * 100,
                'neutral': overall_sentiment.get('neutral', 0) * 100
            }
        else:
            # Use category baseline
            base_estimate = {'positive': 65, 'negative': 15, 'neutral': 20}
        
        # Strategy 2: Adjust based on aspect type
        aspect_adjustments = {
            'camera': {'positive': +5},     # Cameras generally reviewed positively
            'battery': {'negative': +5},    # Battery often criticized
            'price': {'negative': +10},     # Price sensitivity
            'performance': {'positive': +3}, # Performance usually good in modern phones
        }
        
        if aspect in aspect_adjustments:
            for key, adjustment in aspect_adjustments[aspect].items():
                base_estimate[key] = min(100, max(0, base_estimate[key] + adjustment))
            
            # Normalize to 100%
            total = sum(base_estimate.values())
            if total > 0:
                base_estimate = {k: v / total * 100 for k, v in base_estimate.items()}
        
        return base_estimate
    
    def _estimate_aspects_from_category(self,
        category: str,
        requested_aspects: List[str]
    ) -> Dict[str, Any]:
        """Estimate aspects based on phone category"""
        
        # Category-specific aspect patterns
        category_patterns = {
            'flagship': {
                'camera': {'positive': 75, 'negative': 10, 'neutral': 15},
                'performance': {'positive': 80, 'negative': 5, 'neutral': 15},
                'display': {'positive': 85, 'negative': 5, 'neutral': 10},
                'battery': {'positive': 60, 'negative': 20, 'neutral': 20},
                'price': {'positive': 40, 'negative': 40, 'neutral': 20}
            },
            'mid-range': {
                'camera': {'positive': 65, 'negative': 15, 'neutral': 20},
                'performance': {'positive': 70, 'negative': 10, 'neutral': 20},
                'display': {'positive': 70, 'negative': 10, 'neutral': 20},
                'battery': {'positive': 65, 'negative': 15, 'neutral': 20},
                'price': {'positive': 70, 'negative': 10, 'neutral': 20}
            },
            'budget': {
                'camera': {'positive': 50, 'negative': 25, 'neutral': 25},
                'performance': {'positive': 55, 'negative': 20, 'neutral': 25},
                'display': {'positive': 60, 'negative': 15, 'neutral': 25},
                'battery': {'positive': 70, 'negative': 10, 'neutral': 20},
                'price': {'positive': 80, 'negative': 5, 'neutral': 15}
            }
        }
        
        patterns = category_patterns.get(category, category_patterns['mid-range'])
        
        aspects = {}
        for aspect in requested_aspects or self.aspect_weights.keys():
            if aspect in patterns:
                aspects[aspect] = {
                    'sentiment': patterns[aspect],
                    'confidence': ConfidenceLevel.VERY_LOW.value,
                    'is_estimated': True,
                    'note': f"Estimated based on {category} category trends"
                }
            else:
                # Generic estimate
                aspects[aspect] = {
                    'sentiment': {'positive': 60, 'negative': 20, 'neutral': 20},
                    'confidence': ConfidenceLevel.VERY_LOW.value,
                    'is_estimated': True,
                    'note': "Generic estimate - no specific data available"
                }
        
        return {
            'details': aspects,
            'all_estimated': True,
            'estimation_basis': f'{category} category patterns'
        }
    
    def _determine_phone_category(self, phone_model: str) -> str:
        """Determine phone category from model name"""
        
        model_lower = phone_model.lower()
        
        # Flagship indicators
        flagship_keywords = ['pro', 'max', 'ultra', 'plus', 'fold', 'flip']
        if any(keyword in model_lower for keyword in flagship_keywords):
            return 'flagship'
        
        # Budget indicators
        budget_keywords = ['lite', 'go', 'c', 'y', 'a0', 'a1', 'm']
        if any(keyword in model_lower.split() for keyword in budget_keywords):
            return 'budget'
        
        # Brand-specific patterns
        if 'iphone' in model_lower:
            if any(x in model_lower for x in ['se', 'mini']):
                return 'mid-range'
            return 'flagship'
        
        if 'galaxy' in model_lower:
            if 's' in model_lower or 'note' in model_lower:
                return 'flagship'
            elif 'a' in model_lower:
                return 'mid-range'
        
        if 'pixel' in model_lower:
            if 'a' in model_lower:
                return 'mid-range'
            return 'flagship'
        
        # Default to mid-range
        return 'mid-range'
    
    def _generate_warnings(self,
        data_quality: DataQuality,
        review_count: int,
        missing_aspects: List[str]
    ) -> List[str]:
        """Generate appropriate warnings based on data quality"""
        
        warnings = []
        
        if data_quality == DataQuality.NO_DATA:
            warnings.append("⚠️ No review data available - all metrics are estimates")
        elif data_quality == DataQuality.INSUFFICIENT:
            warnings.append(f"⚠️ Very limited data ({review_count} reviews) - results may not be representative")
        elif data_quality == DataQuality.LOW:
            warnings.append(f"⚠️ Limited data ({review_count} reviews) - consider checking multiple sources")
        
        if missing_aspects:
            if len(missing_aspects) <= 3:
                warnings.append(f"⚠️ Limited or no data for: {', '.join(missing_aspects)}")
            else:
                warnings.append(f"⚠️ Limited data for {len(missing_aspects)} aspects")
        
        if review_count > 0 and review_count < 100:
            warnings.append("💡 Results based on small sample size - trends may change with more reviews")
        
        return warnings
    
    def _generate_recommendations(self,
        sentiment: Dict,
        aspects: Dict,
        data_quality: DataQuality
    ) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # Sentiment-based recommendations
        if sentiment.get('positive', 0) > 75:
            recommendations.append("✅ Highly recommended by users - strong positive consensus")
        elif sentiment.get('positive', 0) > 60:
            recommendations.append("👍 Generally well-received - most users are satisfied")
        elif sentiment.get('negative', 0) > 30:
            recommendations.append("⚠️ Mixed reviews - consider specific needs before purchasing")
        
        # Aspect-based recommendations
        if 'details' in aspects:
            strong_aspects = []
            weak_aspects = []
            
            for aspect, data in aspects['details'].items():
                if not data.get('is_estimated', False):
                    aspect_sentiment = data.get('sentiment', {})
                    if aspect_sentiment.get('positive', 0) > 70:
                        strong_aspects.append(aspect)
                    elif aspect_sentiment.get('negative', 0) > 30:
                        weak_aspects.append(aspect)
            
            if strong_aspects:
                recommendations.append(f"💪 Strong points: {', '.join(strong_aspects)}")
            if weak_aspects:
                recommendations.append(f"📉 Consider alternatives if these are important: {', '.join(weak_aspects)}")
        
        # Data quality recommendations
        if data_quality in [DataQuality.NO_DATA, DataQuality.INSUFFICIENT]:
            recommendations.append("🔍 Check back later for more reviews or consult additional sources")
            recommendations.append("💬 Consider asking in forums or communities for user experiences")
        
        return recommendations
    
    def _load_historical_patterns(self) -> Dict:
        """Load historical review patterns for better predictions"""
        
        # In production, this would load from a database or file
        return {
            'average_sentiments': {
                'flagship': {'positive': 0.72, 'neutral': 0.18, 'negative': 0.10},
                'mid-range': {'positive': 0.65, 'neutral': 0.23, 'negative': 0.12},
                'budget': {'positive': 0.58, 'neutral': 0.28, 'negative': 0.14}
            },
            'aspect_correlations': {
                'camera': {'with_price': -0.3, 'with_overall': 0.7},
                'battery': {'with_performance': -0.2, 'with_overall': 0.5},
                'price': {'with_overall': -0.4, 'with_features': -0.3}
            }
        }
    
    def format_result_for_display(self, result: AnalysisResult) -> Dict[str, Any]:
        """Format analysis result for user-friendly display"""
        
        display_data = {
            'phone_model': result.phone_model,
            'summary': self._generate_summary(result),
            'confidence': {
                'level': result.confidence_level,
                'quality': result.data_quality.value,
                'label': self._get_confidence_label(result.confidence_level)
            },
            'metrics': {
                'total_reviews': result.total_reviews,
                'analyzed': result.actual_reviews_analyzed,
                'sentiment': result.sentiment,
                'aspects': result.aspects
            },
            'insights': {
                'warnings': result.warnings,
                'recommendations': result.recommendations
            },
            'metadata': result.metadata
        }
        
        return display_data
    
    def _generate_summary(self, result: AnalysisResult) -> str:
        """Generate human-readable summary"""
        
        if result.data_quality == DataQuality.NO_DATA:
            return f"No reviews available for {result.phone_model}. Showing estimated metrics based on category trends."
        
        sentiment = result.sentiment
        positive_pct = sentiment.get('positive', 0)
        
        if positive_pct > 75:
            tone = "overwhelmingly positive"
        elif positive_pct > 60:
            tone = "mostly positive"
        elif positive_pct > 40:
            tone = "mixed"
        else:
            tone = "concerning"
        
        confidence_text = self._get_confidence_label(result.confidence_level)
        
        summary = f"{result.phone_model} has {tone} reviews based on {result.total_reviews} user opinions. "
        summary += f"Analysis confidence: {confidence_text}."
        
        if result.warnings:
            summary += f" Note: {result.warnings[0]}"
        
        return summary
    
    def _get_confidence_label(self, confidence: float) -> str:
        """Get human-readable confidence label"""
        
        if confidence >= ConfidenceLevel.VERY_HIGH.value:
            return "Very High"
        elif confidence >= ConfidenceLevel.HIGH.value:
            return "High"
        elif confidence >= ConfidenceLevel.MODERATE.value:
            return "Moderate"
        elif confidence >= ConfidenceLevel.LOW.value:
            return "Low"
        else:
            return "Very Low"


# Example usage and testing
if __name__ == "__main__":
    analyzer = RobustReviewAnalyzer()
    
    # Test scenarios
    print("=" * 60)
    print("ROBUST ANALYZER TEST SCENARIOS")
    print("=" * 60)
    
    # Scenario 1: No data
    print("\n📱 Scenario 1: No data available")
    result1 = analyzer.analyze_phone("iPhone 16 Pro Max", None, ['camera', 'battery'])
    fromatted1 = analyzer.fromat_result_for_display(result1)
    print(f"Summary: {formatted1['summary']}")
    print(f"Confidence: {formatted1['confidence']['label']}")
    print(f"Warnings: {formatted1['insights']['warnings']}")
    
    # Scenario 2: Very limited data
    print("\n📱 Scenario 2: Limited data (5 reviews)")
    limited_df = pd.DataFrame({
        'review_text': ['Great phone!', 'Camera is amazing', 'Too expensive', 'Good', 'Nice'],
        'rating': [5, 5, 3, 4, 4],
        'sentiment': ['positive', 'positive', 'negative', 'positive', 'positive']
    })
    result2 = analyzer.analyze_phone("OnePlus 13", limited_df, ['camera', 'price'])
    fromatted2 = analyzer.fromat_result_for_display(result2)
    print(f"Summary: {formatted2['summary']}")
    print(f"Confidence: {formatted2['confidence']['label']}")
    print(f"Warnings: {formatted2['insights']['warnings']}")
    
    # Scenario 3: Missing aspects
    print("\n📱 Scenario 3: Missing aspect data")
    missing_df = pd.DataFrame({
        'review_text': ['Great display'] * 20 + ['Nice phone'] * 10,
        'rating': [5] * 20 + [4] * 10,
        'sentiment': ['positive'] * 25 + ['neutral'] * 5
    })
    result3 = analyzer.analyze_phone("Pixel 9", missing_df, ['camera', 'battery', 'gaming'])
    fromatted3 = analyzer.fromat_result_for_display(result3)
    print(f"Summary: {formatted3['summary']}")
    print(f"Aspects coverage: {result3.aspects['coverage']:.1%}")
    print(f"Estimated aspects: {result3.aspects.get('estimated', [])}")
    
    print("\n" + "=" * 60)
