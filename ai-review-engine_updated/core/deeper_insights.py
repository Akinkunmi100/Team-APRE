"""
Deeper Insights Module
Advanced sentiment analysis beyond basic positive/negative classification
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import numpy as np
from collections import defaultdict, Counter
import re
import statistics

# NLP imports
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import spacy
    from spacy import displacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

logger = logging.getLogger(__name__)


# ==================== EMOTION DETECTION ====================

class EmotionDetector:
    """Detect detailed emotions beyond basic sentiment"""
    
    def __init__(self):
        self.emotions = [
            'joy', 'sadness', 'anger', 'fear', 'surprise', 
            'disgust', 'trust', 'anticipation', 'love', 'optimism',
            'pessimism', 'frustration', 'excitement', 'disappointment'
        ]
        
        if TRANSFORMERS_AVAILABLE:
            try:
                self.emotion_pipeline = pipeline(
                    "text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    return_all_scores=True
                )
                logger.info("Emotion detection model loaded")
            except:
                self.emotion_pipeline = None
                logger.warning("Could not load emotion model")
        else:
            self.emotion_pipeline = None
    
    def detect_emotions(self, text: str) -> Dict[str, float]:
        """
        Detect emotions in text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of emotion scores
        """
        if not self.emotion_pipeline:
            return self._fallback_emotion_detection(text)
        
        try:
            # Get emotion predictions
            results = self.emotion_pipeline(text)[0]
            
            # Convert to dictionary
            emotions = {}
            for result in results:
                emotion = result['label'].lower()
                score = result['score']
                emotions[emotion] = score
            
            # Add derived emotions
            emotions.update(self._derive_complex_emotions(emotions))
            
            return emotions
            
        except Exception as e:
            logger.error(f"Error in emotion detection: {e}")
            return self._fallback_emotion_detection(text)
    
    def _derive_complex_emotions(self, base_emotions: Dict[str, float]) -> Dict[str, float]:
        """Derive complex emotions from base emotions"""
        derived = {}
        
        # Frustration = Anger + Sadness
        if 'anger' in base_emotions and 'sadness' in base_emotions:
            derived['frustration'] = (base_emotions['anger'] + base_emotions['sadness']) / 2
        
        # Disappointment = Sadness + Surprise (negative)
        if 'sadness' in base_emotions and 'surprise' in base_emotions:
            derived['disappointment'] = base_emotions['sadness'] * 0.7 + base_emotions['surprise'] * 0.3
        
        # Excitement = Joy + Surprise
        if 'joy' in base_emotions and 'surprise' in base_emotions:
            derived['excitement'] = (base_emotions['joy'] + base_emotions['surprise']) / 2
        
        # Optimism = Joy + Trust
        if 'joy' in base_emotions:
            derived['optimism'] = base_emotions['joy'] * 0.8
        
        # Pessimism = Sadness + Fear
        if 'sadness' in base_emotions or 'fear' in base_emotions:
            sadness = base_emotions.get('sadness', 0)
            fear = base_emotions.get('fear', 0)
            derived['pessimism'] = (sadness + fear) / 2
        
        return derived
    
    def _fallback_emotion_detection(self, text: str) -> Dict[str, float]:
        """Fallback emotion detection using keywords"""
        emotions = defaultdict(float)
        text_lower = text.lower()
        
        # Keyword-based detection
        emotion_keywords = {
            'joy': ['happy', 'joy', 'excited', 'love', 'great', 'amazing', 'wonderful'],
            'sadness': ['sad', 'disappointed', 'unhappy', 'terrible', 'awful'],
            'anger': ['angry', 'furious', 'mad', 'annoyed', 'frustrated'],
            'fear': ['afraid', 'scared', 'worried', 'concerned', 'anxious'],
            'surprise': ['surprised', 'shocked', 'unexpected', 'wow', 'amazing'],
            'disgust': ['disgusted', 'horrible', 'gross', 'terrible', 'awful'],
            'trust': ['trust', 'reliable', 'confident', 'secure', 'dependable'],
            'anticipation': ['excited', 'looking forward', 'can\'t wait', 'eager']
        }
        
        for emotion, keywords in emotion_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    emotions[emotion] += 0.3
        
        # Normalize scores
        total = sum(emotions.values())
        if total > 0:
            emotions = {k: v/total for k, v in emotions.items()}
        
        return dict(emotions)
    
    def get_dominant_emotion(self, emotions: Dict[str, float]) -> Tuple[str, float]:
        """Get the dominant emotion from emotion scores"""
        if not emotions:
            return 'neutral', 0.0
        
        dominant = max(emotions.items(), key=lambda x: x[1])
        return dominant


# ==================== SARCASM AND IRONY DETECTION ====================

class SarcasmDetector:
    """Detect sarcasm and irony in text"""
    
    def __init__(self):
        self.sarcasm_indicators = [
            r'\b(yeah right|sure|oh really|wow|great job|brilliant)\b',
            r'\.{3,}',  # Ellipsis
            r'\!{2,}',  # Multiple exclamation marks
            r'["\'].*["\']',  # Quoted text (often sarcastic)
            r'\b(totally|absolutely|definitely)\b.*\b(not|never)\b',
            r'\b(not|never)\b.*\b(at all)\b'
        ]
        
        if TRANSFORMERS_AVAILABLE:
            try:
                # Load sarcasm detection model if available
                self.sarcasm_model = pipeline(
                    "text-classification",
                    model="mrm8488/t5-base-finetuned-sarcasm-twitter"
                )
                logger.info("Sarcasm detection model loaded")
            except:
                self.sarcasm_model = None
        else:
            self.sarcasm_model = None
    
    def detect_sarcasm(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Detect sarcasm in text
        
        Args:
            text: Input text
            context: Additional context (rating vs text sentiment mismatch)
            
        Returns:
            Sarcasm detection results
        """
        result = {
            'is_sarcastic': False,
            'confidence': 0.0,
            'indicators': [],
            'type': None  # 'sarcasm', 'irony', 'hyperbole'
        }
        
        # Model-based detection
        if self.sarcasm_model:
            try:
                prediction = self.sarcasm_model(text)[0]
                if prediction['label'] == 'SARCASM':
                    result['is_sarcastic'] = True
                    result['confidence'] = prediction['score']
            except:
                pass
        
        # Pattern-based detection
        pattern_score = self._detect_sarcasm_patterns(text)
        
        # Context-based detection (e.g., positive text with low rating)
        context_score = 0.0
        if context:
            context_score = self._analyze_context_mismatch(text, context)
        
        # Combine scores
        combined_score = max(
            result['confidence'],
            pattern_score * 0.6 + context_score * 0.4
        )
        
        if combined_score > 0.5:
            result['is_sarcastic'] = True
            result['confidence'] = combined_score
            result['type'] = self._classify_sarcasm_type(text, context)
        
        return result
    
    def _detect_sarcasm_patterns(self, text: str) -> float:
        """Detect sarcasm using linguistic patterns"""
        score = 0.0
        indicators = []
        
        for pattern in self.sarcasm_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                score += 0.2
                indicators.append(pattern)
        
        # Check for contradiction patterns
        if self._has_contradiction(text):
            score += 0.3
            indicators.append('contradiction')
        
        # Check for exaggeration
        if self._has_exaggeration(text):
            score += 0.2
            indicators.append('exaggeration')
        
        return min(1.0, score)
    
    def _has_contradiction(self, text: str) -> bool:
        """Check if text contains contradictions"""
        contradiction_patterns = [
            (r'\bgreat\b', r'\bbut\b.*\bterrible\b'),
            (r'\blove\b', r'\bhate\b'),
            (r'\bamazing\b', r'\bawful\b'),
            (r'\bbest\b', r'\bworst\b')
        ]
        
        for pos_pattern, neg_pattern in contradiction_patterns:
            if re.search(pos_pattern, text, re.IGNORECASE) and \
                re.search(neg_pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def _has_exaggeration(self, text: str) -> bool:
        """Check for exaggerated language"""
        exaggeration_words = [
            'absolutely', 'totally', 'completely', 'utterly',
            'literally', 'obviously', 'clearly', 'definitely'
        ]
        
        count = sum(1 for word in exaggeration_words if word in text.lower())
        return count >= 2
    
    def _analyze_context_mismatch(self, text: str, context: Dict[str, Any]) -> float:
        """Analyze mismatch between text sentiment and context"""
        if 'rating' not in context:
            return 0.0
        
        rating = context['rating']
        
        # Simple sentiment check
        if TEXTBLOB_AVAILABLE:
            blob = TextBlob(text)
            text_sentiment = blob.sentiment.polarity
            
            # High rating with negative text or vice versa
            if rating >= 4 and text_sentiment < -0.3:
                return 0.8
            elif rating <= 2 and text_sentiment > 0.3:
                return 0.8
        
        return 0.0
    
    def _classify_sarcasm_type(self, text: str, context: Optional[Dict[str, Any]]) -> str:
        """Classify the type of sarcasm"""
        if self._has_exaggeration(text):
            return 'hyperbole'
        elif self._has_contradiction(text):
            return 'irony'
        else:
            return 'sarcasm'


# ==================== CULTURAL SENTIMENT VARIATIONS ====================

class CulturalSentimentAnalyzer:
    """Analyze sentiment variations across cultures"""
    
    def __init__(self):
        # Cultural sentiment modifiers
        self.cultural_patterns = {
            'american': {
                'direct': True,
                'emotion_expression': 'high',
                'positive_bias': 0.1,
                'superlatives': ['awesome', 'amazing', 'incredible']
            },
            'british': {
                'direct': False,
                'emotion_expression': 'moderate',
                'understatement': True,
                'phrases': ['quite good', 'not bad', 'rather nice']
            },
            'japanese': {
                'direct': False,
                'emotion_expression': 'low',
                'politeness': 'high',
                'indirect_criticism': True
            },
            'indian': {
                'direct': 'moderate',
                'emotion_expression': 'high',
                'detailed': True,
                'value_focus': True
            }
        }
        
        # Language indicators for culture detection
        self.cultural_indicators = {
            'american': ['awesome', 'cool', 'bucks', 'cell phone'],
            'british': ['brilliant', 'lovely', 'queue', 'mobile'],
            'indian': ['lakh', 'crore', 'value for money', 'kindly'],
            'australian': ['mate', 'heaps', 'arvo', 'ripper']
        }
    
    def analyze_cultural_sentiment(
        self, 
        text: str,
        detected_culture: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze sentiment with cultural context
        
        Args:
            text: Input text
            detected_culture: Pre-detected culture or None for auto-detection
            
        Returns:
            Cultural sentiment analysis
        """
        # Detect culture if not provided
        if not detected_culture:
            detected_culture = self._detect_culture(text)
        
        # Get base sentiment
        base_sentiment = self._get_base_sentiment(text)
        
        # Apply cultural adjustments
        adjusted_sentiment = self._apply_cultural_adjustments(
            base_sentiment,
            text,
            detected_culture
        )
        
        # Detect cultural-specific expressions
        cultural_expressions = self._detect_cultural_expressions(text, detected_culture)
        
        return {
            'detected_culture': detected_culture,
            'base_sentiment': base_sentiment,
            'adjusted_sentiment': adjusted_sentiment,
            'cultural_expressions': cultural_expressions,
            'confidence': self._calculate_confidence(text, detected_culture)
        }
    
    def _detect_culture(self, text: str) -> str:
        """Detect cultural context from text"""
        scores = defaultdict(float)
        text_lower = text.lower()
        
        for culture, indicators in self.cultural_indicators.items():
            for indicator in indicators:
                if indicator in text_lower:
                    scores[culture] += 1
        
        # Check for spelling variations
        if 'color' in text_lower:
            scores['american'] += 0.5
        elif 'colour' in text_lower:
            scores['british'] += 0.5
        
        if 'analyze' in text_lower:
            scores['american'] += 0.5
        elif 'analyse' in text_lower:
            scores['british'] += 0.5
        
        # Return culture with highest score or 'neutral'
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        return 'neutral'
    
    def _get_base_sentiment(self, text: str) -> float:
        """Get base sentiment score"""
        if TEXTBLOB_AVAILABLE:
            blob = TextBlob(text)
            return blob.sentiment.polarity
        return 0.0
    
    def _apply_cultural_adjustments(:
        self,
        base_sentiment: float,
        text: str,
        culture: str
    ) -> float:
        """Apply cultural adjustments to sentiment"""
        if culture not in self.cultural_patterns:
            return base_sentiment
        
        pattern = self.cultural_patterns[culture]
        adjusted = base_sentiment
        
        # Apply positive bias if exists
        if 'positive_bias' in pattern:
            adjusted += pattern['positive_bias']
        
        # Adjust for understatement
        if pattern.get('understatement'):
            # British understatement - "not bad" actually means good
            if 'not bad' in text.lower() or 'quite good' in text.lower():
                adjusted = max(adjusted, 0.6)
        
        # Adjust for indirect criticism
        if pattern.get('indirect_criticism'):
            # Polite criticism might seem less negative
            if adjusted < 0 and 'perhaps' in text.lower() or 'might' in text.lower():
                adjusted *= 0.7  # Reduce negativity
        
        return max(-1, min(1, adjusted))
    
    def _detect_cultural_expressions(self, text: str, culture: str) -> List[str]:
        """Detect culture-specific expressions"""
        expressions = []
        
        if culture == 'british':
            british_expressions = [
                'brilliant', 'lovely', 'quite good', 'rather',
                'indeed', 'perhaps', 'might be better'
            ]
            expressions = [expr for expr in british_expressions if expr in text.lower()]
        
        elif culture == 'american':
            american_expressions = [
                'awesome', 'amazing', 'totally', 'super',
                'cool', 'great job', 'way to go'
            ]
            expressions = [expr for expr in american_expressions if expr in text.lower()]
        
        return expressions
    
    def _calculate_confidence(self, text: str, culture: str) -> float:
        """Calculate confidence in cultural detection"""
        indicator_count = sum(
            1 for indicator in self.cultural_indicators.get(culture, [])
            if indicator in text.lower():
        )
        
        # More indicators = higher confidence
        confidence = min(1.0, indicator_count * 0.25)
        
        # Boost confidence for longer texts
        if len(text.split()) > 20:
            confidence = min(1.0, confidence * 1.2)
        
        return confidence


# ==================== TEMPORAL PATTERN ANALYSIS ====================

class TemporalPatternAnalyzer:
    """Analyze temporal patterns in sentiment and reviews"""
    
    def __init__(self):
        self.time_windows = {
            'hourly': timedelta(hours=1),
            'daily': timedelta(days=1),
            'weekly': timedelta(weeks=1),
            'monthly': timedelta(days=30)
        }
    
    def analyze_temporal_patterns(:
        self,
        reviews: List[Dict[str, Any]],
        granularity: str = 'daily'
    ) -> Dict[str, Any]:
        """
        Analyze temporal patterns in reviews
        
        Args:
            reviews: List of reviews with timestamps
            granularity: Time granularity for analysis
            
        Returns:
            Temporal pattern analysis
        """
        if not reviews:
            return {'status': 'no_data'}
        
        # Group reviews by time window
        time_groups = self._group_by_time(reviews, granularity)
        
        # Analyze sentiment trends
        sentiment_trends = self._analyze_sentiment_trends(time_groups)
        
        # Detect anomalies
        anomalies = self._detect_temporal_anomalies(time_groups)
        
        # Identify patterns
        patterns = self._identify_patterns(time_groups)
        
        # Predict future trends
        predictions = self._predict_trends(sentiment_trends)
        
        return {
            'granularity': granularity,
            'time_groups': len(time_groups),
            'sentiment_trends': sentiment_trends,
            'anomalies': anomalies,
            'patterns': patterns,
            'predictions': predictions,
            'summary': self._generate_temporal_summary(sentiment_trends, patterns)
        }
    
    def _group_by_time(:
        self,
        reviews: List[Dict[str, Any]],
        granularity: str
    ) -> Dict[datetime, List[Dict[str, Any]]]:
        """Group reviews by time window"""
        groups = defaultdict(list)
        window = self.time_windows.get(granularity, timedelta(days=1))
        
        for review in reviews:
            if 'timestamp' in review:
                timestamp = review['timestamp']
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp)
                
                # Round to nearest window
                if granularity == 'hourly':
                    key = timestamp.replace(minute=0, second=0, microsecond=0)
                elif granularity == 'daily':
                    key = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
                elif granularity == 'weekly':
                    # Round to Monday
                    days_since_monday = timestamp.weekday()
                    key = timestamp - timedelta(days=days_since_monday)
                    key = key.replace(hour=0, minute=0, second=0, microsecond=0)
                else:  # monthly
                    key = timestamp.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                
                groups[key].append(review)
        
        return dict(sorted(groups.items()))
    
    def _analyze_sentiment_trends(:
        self,
        time_groups: Dict[datetime, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Analyze sentiment trends over time"""
        trends = []
        
        for timestamp, reviews in time_groups.items():
            sentiments = [r.get('sentiment', 0) for r in reviews]
            ratings = [r.get('rating', 0) for r in reviews if 'rating' in r]
            
            if sentiments:
                trend_point = {
                    'timestamp': timestamp.isoformat(),
                    'avg_sentiment': statistics.mean(sentiments),
                    'sentiment_std': statistics.stdev(sentiments) if len(sentiments) > 1 else 0,
                    'review_count': len(reviews),
                    'avg_rating': statistics.mean(ratings) if ratings else None,
                    'positive_ratio': sum(1 for s in sentiments if s > 0) / len(sentiments),
                    'negative_ratio': sum(1 for s in sentiments if s < 0) / len(sentiments)
                }
                trends.append(trend_point)
        
        return trends
    
    def _detect_temporal_anomalies(:
        self,
        time_groups: Dict[datetime, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Detect anomalies in temporal patterns"""
        anomalies = []
        
        # Calculate baseline statistics
        all_counts = [len(reviews) for reviews in time_groups.values()]
        if len(all_counts) < 3:
            return anomalies
        
        mean_count = statistics.mean(all_counts)
        std_count = statistics.stdev(all_counts)
        
        for timestamp, reviews in time_groups.items():
            count = len(reviews)
            
            # Volume anomaly
            if abs(count - mean_count) > 2 * std_count:
                anomalies.append({
                    'timestamp': timestamp.isoformat(),
                    'type': 'volume_spike' if count > mean_count else 'volume_drop',
                    'severity': abs(count - mean_count) / std_count,
                    'review_count': count,
                    'expected_count': mean_count
                })
            
            # Sentiment anomaly
            sentiments = [r.get('sentiment', 0) for r in reviews]
            if sentiments:
                avg_sentiment = statistics.mean(sentiments)
                if abs(avg_sentiment) > 0.7:  # Strong sentiment:
                    anomalies.append({
                        'timestamp': timestamp.isoformat(),
                        'type': 'sentiment_extreme',
                        'sentiment': avg_sentiment,
                        'direction': 'positive' if avg_sentiment > 0 else 'negative'
                    })
        
        return anomalies
    
    def _identify_patterns(:
        self,
        time_groups: Dict[datetime, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Identify recurring patterns"""
        patterns = {
            'weekly_pattern': None,
            'daily_pattern': None,
            'trend_direction': None,
            'seasonality': None
        }
        
        # Analyze by day of week
        day_groups = defaultdict(list)
        for timestamp, reviews in time_groups.items():
            day_of_week = timestamp.strftime('%A')
            day_groups[day_of_week].extend(reviews)
        
        if day_groups:
            # Find busiest day
            busiest_day = max(day_groups.items(), key=lambda x: len(x[1]))
            patterns['weekly_pattern'] = {
                'busiest_day': busiest_day[0],
                'busiest_day_count': len(busiest_day[1])
            }
        
        # Analyze trend direction
        if len(time_groups) >= 3:
            timestamps = sorted(time_groups.keys())
            early_sentiment = statistics.mean([
                r.get('sentiment', 0) 
                for r in time_groups[timestamps[0]]:
            ])
            late_sentiment = statistics.mean([
                r.get('sentiment', 0) 
                for r in time_groups[timestamps[-1]]:
            ])
            
            if late_sentiment > early_sentiment + 0.1:
                patterns['trend_direction'] = 'improving'
            elif late_sentiment < early_sentiment - 0.1:
                patterns['trend_direction'] = 'declining'
            else:
                patterns['trend_direction'] = 'stable'
        
        return patterns
    
    def _predict_trends(:
        self,
        sentiment_trends: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Predict future trends based on historical data"""
        if len(sentiment_trends) < 3:
            return {'status': 'insufficient_data'}
        
        # Simple linear regression for trend prediction
        sentiments = [t['avg_sentiment'] for t in sentiment_trends]
        
        # Calculate trend
        n = len(sentiments)
        if n >= 2:
            # Simple slope calculation
            x_mean = n / 2
            y_mean = sum(sentiments) / n
            
            numerator = sum((i - x_mean) * (sentiments[i] - y_mean) for i in range(n))
            denominator = sum((i - x_mean) ** 2 for i in range(n))
            
            if denominator != 0:
                slope = numerator / denominator
                
                # Predict next period
                next_sentiment = sentiments[-1] + slope
                
                return {
                    'next_period_sentiment': max(-1, min(1, next_sentiment)),
                    'trend_slope': slope,
                    'trend_interpretation': self._interpret_slope(slope),
                    'confidence': min(0.9, n * 0.1)  # Higher confidence with more data
                }
        
        return {'status': 'cannot_predict'}
    
    def _interpret_slope(self, slope: float) -> str:
        """Interpret trend slope"""
        if slope > 0.05:
            return 'strongly_improving'
        elif slope > 0.01:
            return 'improving'
        elif slope < -0.05:
            return 'strongly_declining'
        elif slope < -0.01:
            return 'declining'
        else:
            return 'stable'
    
    def _generate_temporal_summary(:
        self,
        sentiment_trends: List[Dict[str, Any]],
        patterns: Dict[str, Any]
    ) -> str:
        """Generate human-readable temporal summary"""
        if not sentiment_trends:
            return "No temporal data available"
        
        summary_parts = []
        
        # Overall trend
        if patterns.get('trend_direction'):
            summary_parts.append(f"Sentiment is {patterns['trend_direction']}")
        
        # Volume information
        total_reviews = sum(t['review_count'] for t in sentiment_trends)
        avg_reviews = total_reviews / len(sentiment_trends)
        summary_parts.append(f"Average {avg_reviews:.0f} reviews per period")
        
        # Weekly pattern
        if patterns.get('weekly_pattern'):
            summary_parts.append(
                f"Most active on {patterns['weekly_pattern']['busiest_day']}"
            )
        
        return ". ".join(summary_parts)


# ==================== REVIEW HELPFULNESS PREDICTION ====================

class HelpfulnessPredictor:
    """Predict review helpfulness"""
    
    def __init__(self):
        self.feature_weights = {
            'length': 0.15,
            'detail_level': 0.20,
            'pros_cons': 0.15,
            'specificity': 0.20,
            'readability': 0.10,
            'emotion_balance': 0.10,
            'verified_purchase': 0.10
        }
    
    def predict_helpfulness(:
        self,
        review: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Predict how helpful a review will be
        
        Args:
            review: Review data
            
        Returns:
            Helpfulness prediction
        """
        features = self._extract_features(review)
        score = self._calculate_helpfulness_score(features)
        factors = self._identify_factors(features)
        
        return {
            'helpfulness_score': score,
            'helpfulness_category': self._categorize_score(score),
            'confidence': features.get('confidence', 0.7),
            'positive_factors': factors['positive'],
            'negative_factors': factors['negative'],
            'suggestions': self._generate_suggestions(features)
        }
    
    def _extract_features(self, review: Dict[str, Any]) -> Dict[str, float]:
        """Extract features for helpfulness prediction"""
        features = {}
        text = review.get('text', '')
        
        # Length feature
        word_count = len(text.split())
        if word_count < 20:
            features['length'] = 0.2
        elif word_count < 50:
            features['length'] = 0.5
        elif word_count < 200:
            features['length'] = 0.9
        else:
            features['length'] = 0.7  # Too long
        
        # Detail level
        detail_keywords = ['because', 'specifically', 'compared', 'example', 'instance']
        detail_count = sum(1 for kw in detail_keywords if kw in text.lower())
        features['detail_level'] = min(1.0, detail_count * 0.25)
        
        # Pros and cons
        has_pros = any(word in text.lower() for word in ['pros', 'advantages', 'good'])
        has_cons = any(word in text.lower() for word in ['cons', 'disadvantages', 'bad'])
        features['pros_cons'] = 1.0 if (has_pros and has_cons) else 0.5 if (has_pros or has_cons) else 0.0
        
        # Specificity
        specific_terms = ['model', 'version', 'gb', 'mp', 'mah', 'inch', 'hours', 'days']
        specificity_count = sum(1 for term in specific_terms if term in text.lower())
        features['specificity'] = min(1.0, specificity_count * 0.2)
        
        # Readability (simple approximation)
        avg_word_length = sum(len(word) for word in text.split()) / max(1, word_count)
        if 4 <= avg_word_length <= 6:
            features['readability'] = 0.9
        else:
            features['readability'] = 0.6
        
        # Emotion balance
        if 'sentiment' in review:
            sentiment = abs(review['sentiment'])
            if 0.3 <= sentiment <= 0.7:
                features['emotion_balance'] = 0.9  # Balanced
            else:
                features['emotion_balance'] = 0.5  # Too extreme
        else:
            features['emotion_balance'] = 0.5
        
        # Verified purchase
        features['verified_purchase'] = 1.0 if review.get('verified_purchase') else 0.5
        
        # Confidence based on available features
        features['confidence'] = sum(1 for v in features.values() if v > 0) / len(features)
        
        return features
    
    def _calculate_helpfulness_score(self, features: Dict[str, float]) -> float:
        """Calculate overall helpfulness score"""
        score = 0.0
        
        for feature, weight in self.feature_weights.items():
            if feature in features:
                score += features[feature] * weight
        
        return min(1.0, score)
    
    def _categorize_score(self, score: float) -> str:
        """Categorize helpfulness score"""
        if score >= 0.8:
            return 'very_helpful'
        elif score >= 0.6:
            return 'helpful'
        elif score >= 0.4:
            return 'somewhat_helpful'
        else:
            return 'not_helpful'
    
    def _identify_factors(self, features: Dict[str, float]) -> Dict[str, List[str]]:
        """Identify positive and negative factors"""
        factors = {'positive': [], 'negative': []}
        
        for feature, value in features.items():
            if feature == 'confidence':
                continue
                
            if value >= 0.7:
                factors['positive'].append(self._feature_to_description(feature, 'positive'))
            elif value <= 0.3:
                factors['negative'].append(self._feature_to_description(feature, 'negative'))
        
        return factors
    
    def _feature_to_description(self, feature: str, sentiment: str) -> str:
        """Convert feature to human-readable description"""
        descriptions = {
            'length': {
                'positive': 'Good review length',
                'negative': 'Review too short or too long'
            },
            'detail_level': {
                'positive': 'Provides specific details',
                'negative': 'Lacks specific details'
            },
            'pros_cons': {
                'positive': 'Discusses both pros and cons',
                'negative': 'One-sided review'
            },
            'specificity': {
                'positive': 'Mentions specific features',
                'negative': 'Too generic'
            },
            'readability': {
                'positive': 'Easy to read',
                'negative': 'Difficult to read'
            },
            'emotion_balance': {
                'positive': 'Balanced emotional tone',
                'negative': 'Too emotional or extreme'
            },
            'verified_purchase': {
                'positive': 'Verified purchase',
                'negative': 'Unverified review'
            }
        }
        
        return descriptions.get(feature, {}).get(sentiment, feature)
    
    def _generate_suggestions(self, features: Dict[str, float]) -> List[str]:
        """Generate suggestions to improve helpfulness"""
        suggestions = []
        
        if features.get('length', 0) < 0.5:
            suggestions.append("Add more details to your review (aim for 50-150 words)")
        
        if features.get('detail_level', 0) < 0.5:
            suggestions.append("Include specific examples and experiences")
        
        if features.get('pros_cons', 0) < 0.7:
            suggestions.append("Discuss both advantages and disadvantages")
        
        if features.get('specificity', 0) < 0.5:
            suggestions.append("Mention specific features, models, or measurements")
        
        if features.get('emotion_balance', 0) < 0.7:
            suggestions.append("Maintain a balanced tone - avoid being too extreme")
        
        return suggestions


# ==================== MAIN DEEPER INSIGHTS ENGINE ====================

class DeeperInsightsEngine:
    """
    Orchestrates all deeper insight components
    """
    
    def __init__(self, enabled: bool = True):
        """
        Initialize deeper insights engine
        
        Args:
            enabled: Whether engine is enabled
        """
        self.enabled = enabled
        
        if not self.enabled:
            logger.info("Deeper insights disabled")
            return
        
        # Initialize components
        self.emotion_detector = EmotionDetector()
        self.sarcasm_detector = SarcasmDetector()
        self.cultural_analyzer = CulturalSentimentAnalyzer()
        self.temporal_analyzer = TemporalPatternAnalyzer()
        self.helpfulness_predictor = HelpfulnessPredictor()
        
        logger.info("Deeper insights engine initialized")
    
    def analyze(:
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive deeper analysis
        
        Args:
            text: Input text
            context: Additional context
            
        Returns:
            Complete deeper insights
        """
        if not self.enabled:
            return {'status': 'disabled'}
        
        insights = {}
        
        # Emotion detection
        insights['emotions'] = self.emotion_detector.detect_emotions(text)
        dominant_emotion = self.emotion_detector.get_dominant_emotion(insights['emotions'])
        insights['dominant_emotion'] = {
            'emotion': dominant_emotion[0],
            'confidence': dominant_emotion[1]
        }
        
        # Sarcasm detection
        insights['sarcasm'] = self.sarcasm_detector.detect_sarcasm(text, context)
        
        # Cultural sentiment
        insights['cultural_sentiment'] = self.cultural_analyzer.analyze_cultural_sentiment(text)
        
        # Helpfulness prediction
        review_data = {'text': text}
        if context:
            review_data.update(context)
        insights['helpfulness'] = self.helpfulness_predictor.predict_helpfulness(review_data)
        
        # Generate summary
        insights['summary'] = self._generate_insights_summary(insights)
        
        return insights
    
    def analyze_temporal(:
        self,
        reviews: List[Dict[str, Any]],
        granularity: str = 'daily'
    ) -> Dict[str, Any]:
        """
        Analyze temporal patterns
        
        Args:
            reviews: List of reviews with timestamps
            granularity: Time granularity
            
        Returns:
            Temporal analysis
        """
        if not self.enabled:
            return {'status': 'disabled'}
        
        return self.temporal_analyzer.analyze_temporal_patterns(reviews, granularity)
    
    def _generate_insights_summary(self, insights: Dict[str, Any]) -> str:
        """Generate human-readable insights summary"""
        summary_parts = []
        
        # Emotion summary
        if insights.get('dominant_emotion'):
            emotion = insights['dominant_emotion']['emotion']
            confidence = insights['dominant_emotion']['confidence']
            if confidence > 0.5:
                summary_parts.append(f"Primarily expresses {emotion}")
        
        # Sarcasm warning
        if insights.get('sarcasm', {}).get('is_sarcastic'):
            summary_parts.append("Contains sarcasm/irony")
        
        # Cultural context
        culture = insights.get('cultural_sentiment', {}).get('detected_culture')
        if culture and culture != 'neutral':
            summary_parts.append(f"Shows {culture} cultural patterns")
        
        # Helpfulness
        helpfulness = insights.get('helpfulness', {}).get('helpfulness_category')
        if helpfulness:
            summary_parts.append(f"Review is {helpfulness.replace('_', ' ')}")
        
        return ". ".join(summary_parts) if summary_parts else "Standard review"
    
    def get_status(self) -> Dict[str, Any]:
        """Get engine status"""
        if not self.enabled:
            return {'status': 'disabled'}
        
        return {
            'status': 'enabled',
            'components': {
                'emotion_detection': 'active',
                'sarcasm_detection': 'active',
                'cultural_analysis': 'active',
                'temporal_analysis': 'active',
                'helpfulness_prediction': 'active'
            },
            'models_loaded': {
                'emotion_model': self.emotion_detector.emotion_pipeline is not None,
                'sarcasm_model': self.sarcasm_detector.sarcasm_model is not None
            }
        }


# Example usage
if __name__ == "__main__":
    # Initialize engine
    engine = DeeperInsightsEngine(enabled=True)
    
    # Analyze text
    text = "This phone is absolutely AMAZING!!! Best purchase ever... NOT!"
    context = {'rating': 2}
    
    insights = engine.analyze(text, context)
    print(json.dumps(insights, indent=2, default=str))
