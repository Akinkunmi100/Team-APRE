"""
Deeper Insights Module - Advanced Analytical Capabilities
Provides emotion detection, sarcasm/irony detection, cultural sentiment variation,
temporal pattern analysis, and review helpfulness prediction.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any
from datetime import datetime, timedelta
from enum import Enum
import re
import json
from collections import defaultdict, Counter
import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class EmotionType(Enum):
    """Emotion categories for detection"""
    JOY = "joy"
    TRUST = "trust"
    FEAR = "fear"
    SURPRISE = "surprise"
    SADNESS = "sadness"
    DISGUST = "disgust"
    ANGER = "anger"
    ANTICIPATION = "anticipation"
    NEUTRAL = "neutral"


class SarcasmIndicator(Enum):
    """Indicators of sarcasm or irony"""
    CONTRADICTION = "contradiction"  # Positive words with negative context
    EXAGGERATION = "exaggeration"  # Over-the-top statements
    RHETORICAL_QUESTION = "rhetorical_question"
    QUOTATION_MARKS = "quotation_marks"  # "Great" product
    ELLIPSIS = "ellipsis"  # Yeah, right...
    EMOJI_CONTRAST = "emoji_contrast"  # Negative text with positive emoji
    RATING_MISMATCH = "rating_mismatch"  # 5 stars with negative review


class CulturalRegion(Enum):
    """Cultural regions for sentiment variation analysis"""
    NORTH_AMERICA = "north_america"
    EUROPE = "europe"
    ASIA_PACIFIC = "asia_pacific"
    LATIN_AMERICA = "latin_america"
    MIDDLE_EAST = "middle_east"
    AFRICA = "africa"
    GLOBAL = "global"


class TemporalPattern(Enum):
    """Types of temporal patterns in reviews"""
    HONEYMOON = "honeymoon"  # High initial rating that decreases
    GROWING_SATISFACTION = "growing_satisfaction"  # Increasing satisfaction
    CONSISTENT = "consistent"  # Stable ratings over time
    VOLATILE = "volatile"  # Highly variable ratings
    DECLINING = "declining"  # Steadily decreasing satisfaction
    SEASONAL = "seasonal"  # Pattern varies by season/time


@dataclass
class EmotionScore:
    """Emotion detection results"""
    primary_emotion: EmotionType
    emotion_scores: Dict[EmotionType, float]
    intensity: float  # 0-1 scale
    confidence: float
    emotion_words: List[str]
    emotion_phrases: List[str]


@dataclass
class SarcasmDetection:
    """Sarcasm and irony detection results"""
    is_sarcastic: bool
    confidence: float
    indicators: List[SarcasmIndicator]
    sarcastic_phrases: List[str]
    context_clues: Dict[str, Any]
    irony_type: Optional[str]  # verbal, situational, dramatic


@dataclass
class CulturalSentiment:
    """Cultural sentiment variation analysis"""
    region: CulturalRegion
    sentiment_score: float
    cultural_markers: List[str]
    language_indicators: List[str]
    regional_preferences: Dict[str, float]
    cultural_context: Dict[str, Any]


@dataclass
class TemporalAnalysis:
    """Temporal pattern analysis results"""
    pattern_type: TemporalPattern
    trend_direction: str  # increasing, decreasing, stable
    change_rate: float  # Rate of change over time
    key_events: List[Dict[str, Any]]  # Significant temporal events
    seasonality_score: float
    prediction_window: Dict[str, float]  # Future trend predictions


@dataclass
class ReviewHelpfulness:
    """Review helpfulness prediction"""
    helpfulness_score: float  # 0-1 scale
    predicted_helpful_votes: int
    quality_indicators: Dict[str, float]
    missing_elements: List[str]
    improvement_suggestions: List[str]
    comparative_rank: float  # Compared to similar reviews


class EmotionDetector:
    """Detects emotions in review text"""
    
    def __init__(self):
        self.emotion_lexicon = self._load_emotion_lexicon()
        self.intensity_modifiers = self._load_intensity_modifiers()
        
    def _load_emotion_lexicon(self) -> Dict[EmotionType, Set[str]]:
        """Load emotion word lexicon"""
        return {
            EmotionType.JOY: {
                "happy", "joy", "delighted", "pleased", "satisfied", "wonderful",
                "excellent", "amazing", "fantastic", "great", "love", "perfect",
                "awesome", "brilliant", "superb", "enjoyable", "fun"
            },
            EmotionType.TRUST: {
                "reliable", "trustworthy", "dependable", "consistent", "stable",
                "secure", "confident", "faith", "believe", "honest", "genuine",
                "authentic", "solid", "proven", "credible"
            },
            EmotionType.FEAR: {
                "afraid", "scared", "worried", "anxious", "nervous", "concern",
                "risky", "dangerous", "uncertain", "hesitant", "doubt", "wary",
                "apprehensive", "fearful", "terrified"
            },
            EmotionType.SURPRISE: {
                "surprised", "unexpected", "shocking", "amazing", "astonished",
                "stunned", "wow", "unbelievable", "incredible", "remarkable",
                "extraordinary", "unprecedented", "mind-blowing"
            },
            EmotionType.SADNESS: {
                "sad", "disappointed", "unhappy", "depressed", "miserable",
                "unfortunate", "regret", "sorry", "awful", "terrible", "poor",
                "bad", "horrible", "dreadful", "pathetic"
            },
            EmotionType.DISGUST: {
                "disgusting", "gross", "revolting", "repulsive", "nasty",
                "awful", "terrible", "horrible", "unacceptable", "offensive",
                "appalling", "vile", "abhorrent", "repugnant"
            },
            EmotionType.ANGER: {
                "angry", "furious", "mad", "annoyed", "frustrated", "irritated",
                "outraged", "infuriated", "enraged", "upset", "agitated",
                "livid", "irate", "hostile", "aggressive"
            },
            EmotionType.ANTICIPATION: {
                "eager", "excited", "looking forward", "hopeful", "expecting",
                "anticipate", "await", "forthcoming", "upcoming", "future",
                "planned", "intend", "prepare", "ready"
            }
        }
    
    def _load_intensity_modifiers(self) -> Dict[str, float]:
        """Load intensity modifiers for emotion words"""
        return {
            "very": 1.5, "extremely": 2.0, "incredibly": 1.8,
            "absolutely": 1.7, "totally": 1.6, "completely": 1.6,
            "somewhat": 0.7, "slightly": 0.5, "a bit": 0.6,
            "kind of": 0.6, "sort of": 0.6, "barely": 0.3,
            "not": -1.0, "never": -1.2, "hardly": 0.2
        }
    
    def detect_emotions(self, text: str, context: Optional[Dict] = None) -> EmotionScore:
        """Detect emotions in review text"""
        text_lower = text.lower()
        words = text_lower.split()
        
        emotion_scores = defaultdict(float)
        emotion_words = []
        emotion_phrases = []
        
        # Score each emotion based on word matches
        for i, word in enumerate(words):
            for emotion, lexicon in self.emotion_lexicon.items():
                if word in lexicon:
                    # Check for intensity modifiers
                    intensity = 1.0
                    if i > 0 and words[i-1] in self.intensity_modifiers:
                        intensity *= self.intensity_modifiers[words[i-1]]
                    
                    emotion_scores[emotion] += intensity
                    emotion_words.append(word)
                    
                    # Extract phrase context
                    phrase_start = max(0, i-2)
                    phrase_end = min(len(words), i+3)
                    phrase = ' '.join(words[phrase_start:phrase_end])
                    emotion_phrases.append(phrase)
        
        # Normalize scores
        total_score = sum(emotion_scores.values())
        if total_score > 0:
            for emotion in emotion_scores:
                emotion_scores[emotion] /= total_score
        
        # Determine primary emotion
        if emotion_scores:
            primary_emotion = max(emotion_scores, key=emotion_scores.get)
            intensity = min(1.0, total_score / 10)  # Scale intensity
        else:
            primary_emotion = EmotionType.NEUTRAL
            intensity = 0.0
        
        # Calculate confidence based on evidence strength
        confidence = min(1.0, len(emotion_words) / 5) * 0.7
        if emotion_scores and max(emotion_scores.values()) > 0.4:
            confidence += 0.3
        
        return EmotionScore(
            primary_emotion=primary_emotion,
            emotion_scores=dict(emotion_scores),
            intensity=intensity,
            confidence=confidence,
            emotion_words=emotion_words,
            emotion_phrases=emotion_phrases
        )


class SarcasmDetector:
    """Detects sarcasm and irony in reviews"""
    
    def __init__(self):
        self.contradiction_patterns = self._load_contradiction_patterns()
        self.exaggeration_words = self._load_exaggeration_words()
        
    def _load_contradiction_patterns(self) -> List[Tuple[str, str]]:
        """Load patterns indicating contradictions"""
        return [
            (r"great.*but.*terrible", "positive-negative contrast"),
            (r"love.*except.*hate", "love-hate contrast"),
            (r"perfect.*if.*wasn't", "conditional contradiction"),
            (r"amazing.*NOT", "negation emphasis"),
            (r"best.*worst", "extreme contrast"),
            (r"fantastic.*useless", "quality contradiction")
        ]
    
    def _load_exaggeration_words(self) -> Set[str]:
        """Load words indicating exaggeration"""
        return {
            "absolutely", "totally", "completely", "utterly", "entirely",
            "definitely", "obviously", "clearly", "surely", "certainly",
            "perfect", "flawless", "incredible", "unbelievable", "revolutionary",
            "game-changer", "life-changing", "mind-blowing", "earth-shattering"
        }
    
    def detect_sarcasm(self, text: str, rating: Optional[float] = None,
                        context: Optional[Dict] = None) -> SarcasmDetection:
        """Detect sarcasm and irony in review"""
        indicators = []
        sarcastic_phrases = []
        context_clues = {}
        
        text_lower = text.lower()
        
        # Check for contradiction patterns
        for pattern, description in self.contradiction_patterns:
            if re.search(pattern, text_lower):
                indicators.append(SarcasmIndicator.CONTRADICTION)
                context_clues['contradiction'] = description
        
        # Check for exaggeration
        exaggeration_count = sum(1 for word in self.exaggeration_words
                                if word in text_lower)
        if exaggeration_count >= 3:
            indicators.append(SarcasmIndicator.EXAGGERATION)
            context_clues['exaggeration_level'] = exaggeration_count
        
        # Check for rhetorical questions
        if re.search(r'\?.*\?|\bwho\s+.*\?|\bwhat\s+.*\?|\breally\s*\?', text_lower):
            indicators.append(SarcasmIndicator.RHETORICAL_QUESTION)
        
        # Check for quotation marks around positive/negative words
        quote_pattern = r'"([^"]+)"|\'([^\']+)\''
        quotes = re.findall(quote_pattern, text)
        if quotes:
            indicators.append(SarcasmIndicator.QUOTATION_MARKS)
            sarcastic_phrases.extend([q[0] or q[1] for q in quotes])
        
        # Check for ellipsis
        if '...' in text or '…' in text:
            indicators.append(SarcasmIndicator.ELLIPSIS)
        
        # Check for rating mismatch
        if rating is not None:
            sentiment_words = len(re.findall(r'\b(good|great|excellent|perfect)\b', text_lower))
            negative_words = len(re.findall(r'\b(bad|terrible|awful|horrible)\b', text_lower))
            
            if rating >= 4 and negative_words > sentiment_words:
                indicators.append(SarcasmIndicator.RATING_MISMATCH)
                context_clues['rating_text_mismatch'] = True
            elif rating <= 2 and sentiment_words > negative_words:
                indicators.append(SarcasmIndicator.RATING_MISMATCH)
                context_clues['rating_text_mismatch'] = True
        
        # Calculate confidence
        confidence = min(1.0, len(indicators) * 0.25)
        is_sarcastic = len(indicators) >= 2
        
        # Determine irony type
        irony_type = None
        if is_sarcastic:
            if SarcasmIndicator.CONTRADICTION in indicators:
                irony_type = "verbal"
            elif SarcasmIndicator.RATING_MISMATCH in indicators:
                irony_type = "situational"
        
        return SarcasmDetection(
            is_sarcastic=is_sarcastic,
            confidence=confidence,
            indicators=indicators,
            sarcastic_phrases=sarcastic_phrases,
            context_clues=context_clues,
            irony_type=irony_type
        )


class CulturalSentimentAnalyzer:
    """Analyzes cultural variations in sentiment"""
    
    def __init__(self):
        self.cultural_markers = self._load_cultural_markers()
        self.regional_preferences = self._load_regional_preferences()
        
    def _load_cultural_markers(self) -> Dict[CulturalRegion, Set[str]]:
        """Load cultural and linguistic markers"""
        return {
            CulturalRegion.NORTH_AMERICA: {
                "awesome", "cool", "dude", "bucks", "color", "center",
                "cellphone", "elevator", "gas", "truck", "yard"
            },
            CulturalRegion.EUROPE: {
                "brilliant", "lovely", "cheers", "euro", "colour", "centre",
                "mobile", "lift", "petrol", "lorry", "garden", "queue"
            },
            CulturalRegion.ASIA_PACIFIC: {
                "kawaii", "convenient", "compact", "harmony", "quality",
                "respectful", "efficient", "modest", "practical"
            },
            CulturalRegion.LATIN_AMERICA: {
                "bueno", "excelente", "amigo", "familia", "calidad",
                "bonito", "caro", "barato", "rico"
            }
        }
    
    def _load_regional_preferences(self) -> Dict[CulturalRegion, Dict[str, float]]:
        """Load regional preference patterns"""
        return {
            CulturalRegion.NORTH_AMERICA: {
                "value_for_money": 0.8,
                "innovation": 0.9,
                "convenience": 0.85,
                "customer_service": 0.9
            },
            CulturalRegion.EUROPE: {
                "sustainability": 0.9,
                "quality": 0.85,
                "design": 0.8,
                "privacy": 0.95
            },
            CulturalRegion.ASIA_PACIFIC: {
                "technology": 0.9,
                "compactness": 0.85,
                "efficiency": 0.9,
                "brand_reputation": 0.8
            }
        }
    
    def analyze_cultural_sentiment(self, text: str,
                                  user_location: Optional[str] = None) -> CulturalSentiment:
        """Analyze cultural variations in sentiment"""
        text_lower = text.lower()
        
        # Detect cultural region based on markers
        region_scores = {}
        cultural_markers_found = []
        
        for region, markers in self.cultural_markers.items():
            matches = [marker for marker in markers if marker in text_lower]
            region_scores[region] = len(matches)
            if matches:
                cultural_markers_found.extend(matches)
        
        # Determine primary region
        if region_scores:
            detected_region = max(region_scores, key=region_scores.get)
        else:
            detected_region = CulturalRegion.GLOBAL
        
        # Calculate sentiment with cultural adjustments
        base_sentiment = self._calculate_base_sentiment(text)
        
        # Apply cultural adjustments
        cultural_adjustment = 0
        if detected_region in self.regional_preferences:
            preferences = self.regional_preferences[detected_region]
            for preference, weight in preferences.items():
                if preference.replace('_', ' ') in text_lower:
                    cultural_adjustment += weight * 0.1
        
        adjusted_sentiment = base_sentiment + cultural_adjustment
        adjusted_sentiment = max(-1, min(1, adjusted_sentiment))  # Clamp to [-1, 1]
        
        return CulturalSentiment(
            region=detected_region,
            sentiment_score=adjusted_sentiment,
            cultural_markers=cultural_markers_found,
            language_indicators=self._detect_language_indicators(text),
            regional_preferences=self.regional_preferences.get(detected_region, {}),
            cultural_context={
                'base_sentiment': base_sentiment,
                'cultural_adjustment': cultural_adjustment,
                'confidence': min(1.0, len(cultural_markers_found) / 3)
            }
        )
    
    def _calculate_base_sentiment(self, text: str) -> float:
        """Calculate base sentiment score"""
        positive_words = len(re.findall(r'\b(good|great|excellent|amazing|love)\b', text.lower()))
        negative_words = len(re.findall(r'\b(bad|terrible|awful|hate|poor)\b', text.lower()))
        
        total_words = positive_words + negative_words
        if total_words == 0:
            return 0
        
        return (positive_words - negative_words) / total_words
    
    def _detect_language_indicators(self, text: str) -> List[str]:
        """Detect language-specific indicators"""
        indicators = []
        
        # Check for British vs American English
        if re.search(r'\b(colour|favour|honour|centre|theatre)\b', text):
            indicators.append("british_english")
        elif re.search(r'\b(color|favor|honor|center|theater)\b', text):
            indicators.append("american_english")
        
        # Check for date formats
        if re.search(r'\d{1,2}/\d{1,2}/\d{4}', text):
            indicators.append("mm/dd/yyyy_format")
        elif re.search(r'\d{1,2}-\d{1,2}-\d{4}', text):
            indicators.append("dd-mm-yyyy_format")
        
        return indicators


class TemporalPatternAnalyzer:
    """Analyzes temporal patterns in reviews"""
    
    def analyze_temporal_patterns(self, reviews: List[Dict],
                                 time_window: int = 30) -> TemporalAnalysis:
        """Analyze temporal patterns in review data"""
        if not reviews:
            return self._empty_analysis()
        
        # Sort reviews by date
        sorted_reviews = sorted(reviews, key=lambda x: x.get('date', datetime.now()))
        
        # Extract time series data
        dates = [r['date'] for r in sorted_reviews]
        ratings = [r.get('rating', 0) for r in sorted_reviews]
        
        # Detect pattern type
        pattern_type = self._detect_pattern_type(ratings)
        
        # Calculate trend
        trend_direction, change_rate = self._calculate_trend(ratings)
        
        # Detect key events
        key_events = self._detect_key_events(sorted_reviews)
        
        # Calculate seasonality
        seasonality_score = self._calculate_seasonality(dates, ratings)
        
        # Predict future trends
        prediction_window = self._predict_future_trends(ratings, time_window)
        
        return TemporalAnalysis(
            pattern_type=pattern_type,
            trend_direction=trend_direction,
            change_rate=change_rate,
            key_events=key_events,
            seasonality_score=seasonality_score,
            prediction_window=prediction_window
        )
    
    def _detect_pattern_type(self, ratings: List[float]) -> TemporalPattern:
        """Detect the type of temporal pattern"""
        if len(ratings) < 3:
            return TemporalPattern.CONSISTENT
        
        # Calculate statistics
        mean_rating = np.mean(ratings)
        std_rating = np.std(ratings)
        
        # Check for honeymoon effect (high start, then decline)
        first_third = ratings[:len(ratings)//3]
        last_third = ratings[-len(ratings)//3:]
        
        if np.mean(first_third) > np.mean(last_third) + 0.5:
            return TemporalPattern.HONEYMOON
        elif np.mean(last_third) > np.mean(first_third) + 0.5:
            return TemporalPattern.GROWING_SATISFACTION
        elif std_rating > 1.5:
            return TemporalPattern.VOLATILE
        elif std_rating < 0.5:
            return TemporalPattern.CONSISTENT
        
        # Check for steady decline
        correlation = np.corrcoef(range(len(ratings)), ratings)[0, 1]
        if correlation < -0.5:
            return TemporalPattern.DECLINING
        
        return TemporalPattern.CONSISTENT
    
    def _calculate_trend(self, ratings: List[float]) -> Tuple[str, float]:
        """Calculate trend direction and rate of change"""
        if len(ratings) < 2:
            return "stable", 0.0
        
        # Calculate linear regression
        x = np.arange(len(ratings))
        slope, _ = np.polyfit(x, ratings, 1)
        
        if abs(slope) < 0.01:
            direction = "stable"
        elif slope > 0:
            direction = "increasing"
        else:
            direction = "decreasing"
        
        return direction, float(slope)
    
    def _detect_key_events(self, reviews: List[Dict]) -> List[Dict[str, Any]]:
        """Detect significant events in the review timeline"""
        events = []
        
        if len(reviews) < 5:
            return events
        
        # Detect sudden rating changes
        for i in range(1, len(reviews)):
            prev_rating = reviews[i-1].get('rating', 0)
            curr_rating = reviews[i].get('rating', 0)
            
            if abs(curr_rating - prev_rating) >= 2:
                events.append({
                    'type': 'rating_jump',
                    'date': reviews[i]['date'],
                    'change': curr_rating - prev_rating,
                    'description': f"Significant rating change from {prev_rating} to {curr_rating}"
                })
        
        # Detect review volume spikes
        # (This would need actual implementation based on review frequency)
        
        return events
    
    def _calculate_seasonality(self, dates: List[datetime],
                              ratings: List[float]) -> float:
        """Calculate seasonality score"""
        if len(dates) < 12:  # Need at least a year of data:
            return 0.0
        
        # Group by month
        monthly_ratings = defaultdict(list)
        for date, rating in zip(dates, ratings):
            monthly_ratings[date.month].append(rating)
        
        # Calculate variance between months
        monthly_means = [np.mean(ratings) for ratings in monthly_ratings.values()]
        
        if len(monthly_means) < 2:
            return 0.0
        
        seasonality = np.std(monthly_means) / (np.mean(monthly_means) + 0.001)
        return min(1.0, seasonality)
    
    def _predict_future_trends(self, ratings: List[float],
                              window: int) -> Dict[str, float]:
        """Predict future rating trends"""
        if len(ratings) < 3:
            return {'next_30_days': 0.0, 'confidence': 0.0}
        
        # Simple linear extrapolation
        x = np.arange(len(ratings))
        slope, intercept = np.polyfit(x, ratings, 1)
        
        # Predict next period
        future_x = len(ratings) + window // 2
        predicted_rating = slope * future_x + intercept
        predicted_rating = max(1, min(5, predicted_rating))  # Clamp to rating range
        
        # Calculate confidence based on historical consistency
        residuals = ratings - (slope * x + intercept)
        rmse = np.sqrt(np.mean(residuals**2))
        confidence = max(0, 1 - rmse / 2)
        
        return {
            'next_30_days': float(predicted_rating),
            'confidence': float(confidence),
            'trend_strength': abs(float(slope))
        }
    
    def _empty_analysis(self) -> TemporalAnalysis:
        """Return empty analysis when no data available"""
        return TemporalAnalysis(
            pattern_type=TemporalPattern.CONSISTENT,
            trend_direction="stable",
            change_rate=0.0,
            key_events=[],
            seasonality_score=0.0,
            prediction_window={'next_30_days': 0.0, 'confidence': 0.0}
        )


class ReviewHelpfulnessPredictor:
    """Predicts review helpfulness"""
    
    def predict_helpfulness(self, review: Dict) -> ReviewHelpfulness:
        """Predict how helpful a review will be to other users"""
        quality_indicators = {}
        missing_elements = []
        improvement_suggestions = []
        
        text = review.get('text', '')
        rating = review.get('rating', 0)
        
        # Length indicator
        word_count = len(text.split())
        if word_count < 50:
            quality_indicators['length'] = 0.3
            missing_elements.append("detailed_description")
            improvement_suggestions.append("Add more details about your experience")
        elif word_count > 500:
            quality_indicators['length'] = 0.7
            improvement_suggestions.append("Consider being more concise")
        else:
            quality_indicators['length'] = 1.0
        
        # Specificity indicator
        specific_features = self._count_specific_features(text)
        quality_indicators['specificity'] = min(1.0, specific_features / 5)
        if specific_features < 3:
            missing_elements.append("specific_features")
            improvement_suggestions.append("Mention specific product features")
        
        # Balance indicator (pros and cons)
        has_pros = any(word in text.lower() for word in ['pro', 'advantage', 'good', 'great'])
        has_cons = any(word in text.lower() for word in ['con', 'disadvantage', 'bad', 'issue'])
        
        if has_pros and has_cons:
            quality_indicators['balance'] = 1.0
        elif has_pros or has_cons:
            quality_indicators['balance'] = 0.6
            missing_elements.append("balanced_perspective")
            improvement_suggestions.append("Include both pros and cons")
        else:
            quality_indicators['balance'] = 0.3
        
        # Comparison indicator
        has_comparison = any(word in text.lower() for word in ['compared', 'versus', 'vs', 'better than', 'worse than'])
        quality_indicators['comparison'] = 1.0 if has_comparison else 0.4
        if not has_comparison:
            improvement_suggestions.append("Compare with similar products")
        
        # Use case indicator
        has_use_case = self._detect_use_case(text)
        quality_indicators['use_case'] = 1.0 if has_use_case else 0.5
        if not has_use_case:
            missing_elements.append("use_case_description")
            improvement_suggestions.append("Describe how you use the product")
        
        # Calculate overall helpfulness score
        helpfulness_score = np.mean(list(quality_indicators.values()))
        
        # Predict helpful votes (simplified model)
        base_votes = int(helpfulness_score * 100)
        length_bonus = min(20, word_count // 10)
        predicted_helpful_votes = base_votes + length_bonus
        
        # Calculate comparative rank
        comparative_rank = helpfulness_score  # Would compare against other reviews
        
        return ReviewHelpfulness(
            helpfulness_score=helpfulness_score,
            predicted_helpful_votes=predicted_helpful_votes,
            quality_indicators=quality_indicators,
            missing_elements=missing_elements,
            improvement_suggestions=improvement_suggestions,
            comparative_rank=comparative_rank
        )
    
    def _count_specific_features(self, text: str) -> int:
        """Count mentions of specific product features"""
        feature_keywords = [
            'battery', 'screen', 'camera', 'performance', 'speed',
            'storage', 'memory', 'display', 'sound', 'build',
            'design', 'size', 'weight', 'price', 'value'
        ]
        
        text_lower = text.lower()
        return sum(1 for keyword in feature_keywords if keyword in text_lower)
    
    def _detect_use_case(self, text: str) -> bool:
        """Detect if review describes actual use cases"""
        use_case_patterns = [
            r'I use[d]? (it|this|the)',
            r'for (my|work|gaming|travel)',
            r'when I',
            r'during (my|the)',
            r'everyday|daily|weekly',
            r'experience'
        ]
        
        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in use_case_patterns)


class DeeperInsightsEngine:
    """Main engine for deeper insights analysis"""
    
    def __init__(self):
        self.emotion_detector = EmotionDetector()
        self.sarcasm_detector = SarcasmDetector()
        self.cultural_analyzer = CulturalSentimentAnalyzer()
        self.temporal_analyzer = TemporalPatternAnalyzer()
        self.helpfulness_predictor = ReviewHelpfulnessPredictor()
        
        logger.info("Deeper Insights Engine initialized")
    
    def analyze_review(self, review: Dict, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Perform comprehensive deeper analysis on a single review"""
        text = review.get('text', '')
        rating = review.get('rating')
        
        # Perform all analyses
        emotion_score = self.emotion_detector.detect_emotions(text, context)
        sarcasm_detection = self.sarcasm_detector.detect_sarcasm(text, rating, context)
        cultural_sentiment = self.cultural_analyzer.analyze_cultural_sentiment(
            text, 
            review.get('user_location')
        )
        helpfulness = self.helpfulness_predictor.predict_helpfulness(review)
        
        return {
            'review_id': review.get('id'),
            'emotion_analysis': {
                'primary_emotion': emotion_score.primary_emotion.value,
                'emotion_scores': {k.value: v for k, v in emotion_score.emotion_scores.items()},
                'intensity': emotion_score.intensity,
                'confidence': emotion_score.confidence,
                'key_phrases': emotion_score.emotion_phrases[:5]
            },
            'sarcasm_detection': {
                'is_sarcastic': sarcasm_detection.is_sarcastic,
                'confidence': sarcasm_detection.confidence,
                'indicators': [i.value for i in sarcasm_detection.indicators],
                'irony_type': sarcasm_detection.irony_type
            },
            'cultural_analysis': {
                'detected_region': cultural_sentiment.region.value,
                'adjusted_sentiment': cultural_sentiment.sentiment_score,
                'cultural_markers': cultural_sentiment.cultural_markers,
                'regional_preferences': cultural_sentiment.regional_preferences
            },
            'helpfulness_prediction': {
                'score': helpfulness.helpfulness_score,
                'predicted_votes': helpfulness.predicted_helpful_votes,
                'quality_breakdown': helpfulness.quality_indicators,
                'improvements': helpfulness.improvement_suggestions[:3]
            }
        }
    
    def analyze_review_collection(self, reviews: List[Dict]) -> Dict[str, Any]:
        """Analyze a collection of reviews for patterns and insights"""
        if not reviews:
            return {'error': 'No reviews provided'}
        
        # Individual review analyses
        individual_analyses = [self.analyze_review(review) for review in reviews]
        
        # Temporal pattern analysis
        temporal_analysis = self.temporal_analyzer.analyze_temporal_patterns(reviews)
        
        # Aggregate emotion distribution
        emotion_distribution = self._aggregate_emotions(individual_analyses)
        
        # Sarcasm prevalence
        sarcasm_rate = self._calculate_sarcasm_rate(individual_analyses)
        
        # Cultural distribution
        cultural_distribution = self._aggregate_cultural_regions(individual_analyses)
        
        # Overall helpfulness metrics
        helpfulness_metrics = self._aggregate_helpfulness(individual_analyses)
        
        return {
            'total_reviews': len(reviews),
            'temporal_patterns': {
                'pattern_type': temporal_analysis.pattern_type.value,
                'trend': temporal_analysis.trend_direction,
                'change_rate': temporal_analysis.change_rate,
                'seasonality': temporal_analysis.seasonality_score,
                'prediction': temporal_analysis.prediction_window
            },
            'emotion_insights': {
                'distribution': emotion_distribution,
                'dominant_emotion': max(emotion_distribution, key=emotion_distribution.get) if emotion_distribution else None,
                'emotional_diversity': len([e for e in emotion_distribution.values() if e > 0.1])
            },
            'sarcasm_insights': {
                'prevalence_rate': sarcasm_rate,
                'risk_level': 'high' if sarcasm_rate > 0.3 else 'moderate' if sarcasm_rate > 0.1 else 'low'
            },
            'cultural_insights': {
                'distribution': cultural_distribution,
                'diversity_score': self._calculate_diversity_score(cultural_distribution)
            },
            'helpfulness_insights': helpfulness_metrics,
            'individual_analyses': individual_analyses[:10]  # Include sample of individual analyses
        }
    
    def generate_insights_report(self, analysis_results: Dict[str, Any]) -> str:
        """Generate human-readable insights report"""
        report = []
        report.append("=== DEEPER INSIGHTS ANALYSIS REPORT ===\n")
        
        # Temporal insights
        temporal = analysis_results.get('temporal_patterns', {})
        report.append(f"📈 Temporal Patterns:")
        report.append(f"  - Pattern Type: {temporal.get('pattern_type', 'Unknown')}")
        report.append(f"  - Trend Direction: {temporal.get('trend', 'stable')}")
        report.append(f"  - Seasonality Score: {temporal.get('seasonality', 0):.2f}")
        
        # Emotion insights
        emotions = analysis_results.get('emotion_insights', {})
        report.append(f"\n😊 Emotional Analysis:")
        report.append(f"  - Dominant Emotion: {emotions.get('dominant_emotion', 'neutral')}")
        report.append(f"  - Emotional Diversity: {emotions.get('emotional_diversity', 0)} distinct emotions")
        
        # Sarcasm insights
        sarcasm = analysis_results.get('sarcasm_insights', {})
        report.append(f"\n🎭 Sarcasm Detection:")
        report.append(f"  - Prevalence Rate: {sarcasm.get('prevalence_rate', 0):.1%}")
        report.append(f"  - Risk Level: {sarcasm.get('risk_level', 'low')}")
        
        # Cultural insights
        cultural = analysis_results.get('cultural_insights', {})
        report.append(f"\n🌍 Cultural Analysis:")
        report.append(f"  - Diversity Score: {cultural.get('diversity_score', 0):.2f}")
        
        # Helpfulness insights
        helpfulness = analysis_results.get('helpfulness_insights', {})
        report.append(f"\n⭐ Review Quality:")
        report.append(f"  - Average Helpfulness: {helpfulness.get('average_score', 0):.2f}")
        report.append(f"  - High Quality Reviews: {helpfulness.get('high_quality_percentage', 0):.1%}")
        
        # Key recommendations
        report.append(f"\n💡 Key Recommendations:")
        if temporal.get('pattern_type') == 'declining':
            report.append("  ⚠️ Declining satisfaction trend detected - investigate recent changes")
        if sarcasm.get('prevalence_rate', 0) > 0.2:
            report.append("  ⚠️ High sarcasm rate - reviews may not reflect true sentiment")
        if emotions.get('dominant_emotion') in ['anger', 'disgust', 'fear']:
            report.append("  ⚠️ Negative emotions dominate - address customer concerns")
        if helpfulness.get('average_score', 0) < 0.5:
            report.append("  ℹ️ Low review quality - encourage more detailed feedback")
        
        return '\n'.join(report)
    
    def _aggregate_emotions(self, analyses: List[Dict]) -> Dict[str, float]:
        """Aggregate emotion distribution across reviews"""
        emotion_counts = defaultdict(float)
        
        for analysis in analyses:
            emotion_data = analysis.get('emotion_analysis', {})
            primary = emotion_data.get('primary_emotion')
            if primary:
                emotion_counts[primary] += 1
        
        # Normalize
        total = sum(emotion_counts.values())
        if total > 0:
            return {emotion: count/total for emotion, count in emotion_counts.items()}
        return {}
    
    def _calculate_sarcasm_rate(self, analyses: List[Dict]) -> float:
        """Calculate rate of sarcastic reviews"""
        if not analyses:
            return 0.0
        
        sarcastic_count = sum(1 for a in analyses
                            if a.get('sarcasm_detection', {}).get('is_sarcastic', False))
        return sarcastic_count / len(analyses)
    
    def _aggregate_cultural_regions(self, analyses: List[Dict]) -> Dict[str, float]:
        """Aggregate cultural region distribution"""
        region_counts = defaultdict(int)
        
        for analysis in analyses:
            region = analysis.get('cultural_analysis', {}).get('detected_region')
            if region:
                region_counts[region] += 1
        
        # Normalize
        total = sum(region_counts.values())
        if total > 0:
            return {region: count/total for region, count in region_counts.items()}
        return {}
    
    def _aggregate_helpfulness(self, analyses: List[Dict]) -> Dict[str, Any]:
        """Aggregate helpfulness metrics"""
        scores = [a.get('helpfulness_prediction', {}).get('score', 0) for a in analyses]
        
        if not scores:
            return {'average_score': 0, 'high_quality_percentage': 0}
        
        return {
            'average_score': np.mean(scores),
            'min_score': min(scores),
            'max_score': max(scores),
            'high_quality_percentage': sum(1 for s in scores if s > 0.7) / len(scores),
            'low_quality_percentage': sum(1 for s in scores if s < 0.4) / len(scores)
        }
    
    def _calculate_diversity_score(self, distribution: Dict[str, float]) -> float:
        """Calculate diversity score using Shannon entropy"""
        if not distribution:
            return 0.0
        
        # Calculate entropy
        entropy = -sum(p * np.log(p) for p in distribution.values() if p > 0)
        
        # Normalize by maximum possible entropy
        max_entropy = np.log(len(distribution))
        if max_entropy > 0:
            return entropy / max_entropy
        return 0.0


# Example usage and testing
if __name__ == "__main__":
    # Initialize the engine
    insights_engine = DeeperInsightsEngine()
    
    # Sample review data
    sample_reviews = [
        {
            'id': 'r1',
            'text': "This phone is absolutely AMAZING!!! Best purchase ever... NOT! The battery dies in 2 hours, the camera is terrible, and it's slower than my 5-year-old phone. Save your money.",
            'rating': 5,
            'date': datetime.now() - timedelta(days=30),
            'user_location': 'USA'
        },
        {
            'id': 'r2',
            'text': "I'm extremely happy with this phone. The display is brilliant, performance is smooth, and the camera takes lovely photos. Battery life could be better but overall it's a great value.",
            'rating': 4,
            'date': datetime.now() - timedelta(days=20),
            'user_location': 'UK'
        },
        {
            'id': 'r3',
            'text': "Compared to my previous Samsung, this phone is more efficient and compact. The quality is excellent and it's very convenient for daily use. Highly recommended!",
            'rating': 5,
            'date': datetime.now() - timedelta(days=10),
            'user_location': 'Japan'
        }
    ]
    
    # Analyze individual review
    print("=== Individual Review Analysis ===")
    individual_result = insights_engine.analyze_review(sample_reviews[0])
    print(json.dumps(individual_result, indent=2))
    
    # Analyze collection
    print("\n=== Collection Analysis ===")
    collection_result = insights_engine.analyze_review_collection(sample_reviews)
    
    # Generate report
    print("\n" + insights_engine.generate_insights_report(collection_result))
    
    # Test specific components
    print("\n=== Component Tests ===")
    
    # Test emotion detection
    emotion_detector = EmotionDetector()
    emotion_result = emotion_detector.detect_emotions("I'm absolutely furious! This product is terrible and I hate it!")
    print(f"Emotion: {emotion_result.primary_emotion.value} (confidence: {emotion_result.confidence:.2f})")
    
    # Test sarcasm detection  
    sarcasm_detector = SarcasmDetector()
    sarcasm_result = sarcasm_detector.detect_sarcasm("Oh great, another 'perfect' product that breaks in a week...", rating=5)
    print(f"Sarcastic: {sarcasm_result.is_sarcastic} (confidence: {sarcasm_result.confidence:.2f})")
    
    # Test cultural analysis
    cultural_analyzer = CulturalSentimentAnalyzer()
    cultural_result = cultural_analyzer.analyze_cultural_sentiment("This mobile is brilliant! The colour display is lovely and it works perfectly in the lift.")
    print(f"Region: {cultural_result.region.value} (sentiment: {cultural_result.sentiment_score:.2f})")
