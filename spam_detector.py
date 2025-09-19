"""
Spam Detector Module for identifying fake reviews
"""

import logging
from typing import Dict, List, Any
import re

logger = logging.getLogger(__name__)


class SpamDetector:
    """
    Detect spam and fake reviews
    """
    
    def __init__(self):
        """Initialize spam detector"""
        self.spam_patterns = [
            r'click here',
            r'buy now',
            r'limited offer',
            r'act now',
            r'100% guarantee',
            r'risk free',
            r'order now',
            r'special promotion'
        ]
        
        self.repetitive_patterns = [
            r'(.)\1{4,}',  # Same character repeated 5+ times
            r'(\b\w+\b)( \1){3,}',  # Same word repeated 4+ times
        ]
        
        logger.info("Spam Detector initialized")
    
    def detect_spam(self, text: str) -> Dict[str, Any]:
        """
        Detect if a review is spam
        
        Args:
            text: Review text
            
        Returns:
            Spam detection results
        """
        text_lower = text.lower()
        spam_score = 0.0
        reasons = []
        
        # Check for spam patterns
        for pattern in self.spam_patterns:
            if re.search(pattern, text_lower):
                spam_score += 0.2
                reasons.append(f"Contains spam pattern: {pattern}")
        
        # Check for repetitive patterns
        for pattern in self.repetitive_patterns:
            if re.search(pattern, text):
                spam_score += 0.3
                reasons.append("Contains repetitive text")
        
        # Check text length
        if len(text) < 20:
            spam_score += 0.1
            reasons.append("Very short review")
        elif len(text) > 2000:
            spam_score += 0.1
            reasons.append("Unusually long review")
        
        # Check for all caps
        if text.isupper() and len(text) > 10:
            spam_score += 0.2
            reasons.append("Written in all caps")
        
        # Check for excessive exclamation marks
        exclamation_count = text.count('!')
        if exclamation_count > 5:
            spam_score += 0.1
            reasons.append("Excessive exclamation marks")
        
        # Check for URLs
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        if re.search(url_pattern, text):
            spam_score += 0.3
            reasons.append("Contains URLs")
        
        # Determine if spam
        is_spam = spam_score >= 0.5
        
        return {
            'is_spam': is_spam,
            'spam_score': min(spam_score, 1.0),
            'confidence': min(spam_score * 1.5, 1.0),
            'reasons': reasons,
            'risk_level': self._get_risk_level(spam_score)
        }
    
    def _get_risk_level(self, score: float) -> str:
        """
        Get risk level based on spam score
        
        Args:
            score: Spam score
            
        Returns:
            Risk level string
        """
        if score < 0.3:
            return 'low'
        elif score < 0.6:
            return 'medium'
        else:
            return 'high'
    
    def detect_fake_patterns(self, reviews: List[str]) -> Dict[str, Any]:
        """
        Detect fake review patterns across multiple reviews
        
        Args:
            reviews: List of review texts
            
        Returns:
            Fake pattern detection results
        """
        if not reviews:
            return {'fake_percentage': 0, 'patterns_found': []}
        
        fake_count = 0
        patterns_found = []
        
        # Check for similar reviews
        for i in range(len(reviews)):
            for j in range(i + 1, len(reviews)):
                similarity = self._calculate_similarity(reviews[i], reviews[j])
                if similarity > 0.8:
                    fake_count += 1
                    patterns_found.append("Similar reviews detected")
                    break
        
        fake_percentage = (fake_count / len(reviews)) * 100
        
        return {
            'fake_percentage': fake_percentage,
            'patterns_found': list(set(patterns_found)),
            'total_reviews': len(reviews),
            'suspicious_reviews': fake_count
        }
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        # Simple word-based similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
