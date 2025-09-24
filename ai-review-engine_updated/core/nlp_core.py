"""
NLP Core Module for text processing and analysis
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import re
from collections import Counter

logger = logging.getLogger(__name__)


class NLPCore:
    """
    Core NLP functionality for text processing
    """
    
    def __init__(self):
        """Initialize NLP core"""
        self.stopwords = {
            'the', 'is', 'at', 'which', 'on', 'a', 'an', 'as', 'are', 
            'was', 'were', 'be', 'have', 'had', 'do', 'does', 'did',
            'will', 'would', 'should', 'could', 'may', 'might', 'must',
            'can', 'this', 'that', 'these', 'those', 'i', 'you', 'we',
            'they', 'he', 'she', 'it', 'to', 'of', 'in', 'for', 'with'
        }
        logger.info("NLP Core initialized")
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Convert to lowercase and split
        text = text.lower()
        # Remove punctuation and split
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from tokens
        
        Args:
            tokens: List of tokens
            
        Returns:
            Filtered tokens
        """
        return [token for token in tokens if token not in self.stopwords]
    
    def extract_keywords(self, text: str, top_n: int = 10) -> List[Tuple[str, int]]:
        """
        Extract top keywords from text
        
        Args:
            text: Input text
            top_n: Number of keywords to extract
            
        Returns:
            List of (keyword, frequency) tuples
        """
        tokens = self.tokenize(text)
        filtered_tokens = self.remove_stopwords(tokens)
        
        # Count frequencies
        word_freq = Counter(filtered_tokens)
        
        # Get top keywords
        return word_freq.most_common(top_n)
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of entity types and entities
        """
        entities = {
            'phones': [],
            'brands': [],
            'features': []
        }
        
        # Simple pattern matching for phones
        phone_patterns = [
            r'iPhone \d+\w*',
            r'Galaxy S\d+\w*',
            r'Pixel \d+\w*',
            r'OnePlus \d+\w*'
        ]
        
        for pattern in phone_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['phones'].extend(matches)
        
        # Brand detection
        brands = ['Apple', 'Samsung', 'Google', 'OnePlus', 'Xiaomi', 'Huawei', 'Nokia']
        for brand in brands:
            if brand.lower() in text.lower():
                entities['brands'].append(brand)
        
        # Feature detection
        features = ['camera', 'battery', 'display', 'processor', '5G', 'storage']
        for feature in features:
            if feature.lower() in text.lower():
                entities['features'].append(feature)
        
        return entities
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        # Tokenize both texts
        tokens1 = set(self.tokenize(text1))
        tokens2 = set(self.tokenize(text2))
        
        # Calculate Jaccard similarity
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def detect_language(self, text: str) -> str:
        """
        Detect language of text (simplified)
        
        Args:
            text: Input text
            
        Returns:
            Language code
        """
        # Simple heuristic - check for common English words
        english_words = {'the', 'is', 'are', 'was', 'were', 'have', 'has', 'do', 'does'}
        tokens = set(self.tokenize(text.lower()))
        
        english_count = len(tokens.intersection(english_words))
        
        if english_count >= 2:
            return 'en'
        else:
            return 'unknown'
    
    def extract_sentiment_phrases(self, text: str) -> Dict[str, List[str]]:
        """
        Extract positive and negative sentiment phrases
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with positive and negative phrases
        """
        positive_words = {
            'excellent', 'amazing', 'great', 'good', 'fantastic', 'awesome',
            'perfect', 'love', 'best', 'wonderful', 'impressive', 'outstanding'
        }
        
        negative_words = {
            'bad', 'terrible', 'awful', 'poor', 'worst', 'hate', 'horrible',
            'disappointing', 'useless', 'broken', 'slow', 'expensive'
        }
        
        tokens = self.tokenize(text.lower())
        
        phrases = {
            'positive': [],
            'negative': []
        }
        
        for i, token in enumerate(tokens):
            if token in positive_words:
                # Get surrounding context
                start = max(0, i-2)
                end = min(len(tokens), i+3)
                phrase = ' '.join(tokens[start:end])
                phrases['positive'].append(phrase)
            elif token in negative_words:
                start = max(0, i-2)
                end = min(len(tokens), i+3)
                phrase = ' '.join(tokens[start:end])
                phrases['negative'].append(phrase)
        
        return phrases
    
    def get_text_statistics(self, text: str) -> Dict[str, Any]:
        """
        Get various statistics about the text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of statistics
        """
        tokens = self.tokenize(text)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return {
            'character_count': len(text),
            'word_count': len(tokens),
            'sentence_count': len(sentences),
            'avg_word_length': sum(len(token) for token in tokens) / len(tokens) if tokens else 0,
            'avg_sentence_length': len(tokens) / len(sentences) if sentences else 0,
            'unique_words': len(set(tokens)),
            'lexical_diversity': len(set(tokens)) / len(tokens) if tokens else 0
        }
