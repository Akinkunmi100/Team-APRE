"""
Data Preprocessing Module for Review Text
"""

import re
import string
import logging
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    pass

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Class for preprocessing review data"""
    
    def __init__(self, language='english'):
        """Initialize preprocessor with language settings"""
        self.language = language
        self.stop_words = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()
        
        # Try to load spaCy model
        try:
            logger.info("Attempting to load spaCy model 'en_core_web_sm'...")
            self.nlp = spacy.load('en_core_web_sm')
            logger.info("spaCy model loaded successfully")
            logger.info(f"spaCy model info: {self.nlp.meta}")
        except ImportError as e:
            logger.warning(f"spaCy ImportError: {e}")
            logger.warning("spaCy library may not be installed. Some features may be limited.")
            self.nlp = None
        except OSError as e:
            logger.warning(f"spaCy OSError: {e}")
            logger.warning("spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
            logger.warning("Some features may be limited.")
            self.nlp = None
        except Exception as e:
            logger.error(f"Unexpected error loading spaCy: {e}")
            logger.warning("Some features may be limited.")
            self.nlp = None
            
        # Phone-related aspects for extraction
        self.aspects = [
            'battery', 'camera', 'screen', 'display', 'performance',
            'storage', 'memory', 'design', 'build', 'quality',
            'price', 'value', 'software', 'speed', 'charging'
        ]
        
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s.,!?-]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from tokens
        
        Args:
            tokens: List of tokens
            
        Returns:
            Filtered tokens
        """
        return [token for token in tokens if token not in self.stop_words and len(token) > 2]
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens
        
        Args:
            tokens: List of tokens
            
        Returns:
            Lemmatized tokens
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess_review(self, review: str) -> Dict:
        """
        Complete preprocessing pipeline for a single review
        
        Args:
            review: Raw review text
            
        Returns:
            Dictionary with preprocessed data
        """
        # Clean text
        cleaned = self.clean_text(review)
        
        # Tokenize
        tokens = self.tokenize(cleaned)
        
        # Remove stopwords
        filtered_tokens = self.remove_stopwords(tokens)
        
        # Lemmatize
        lemmatized = self.lemmatize(filtered_tokens)
        
        # Rejoin for processed text
        processed_text = ' '.join(lemmatized)
        
        return {
            'original': review,
            'cleaned': cleaned,
            'tokens': tokens,
            'filtered_tokens': filtered_tokens,
            'lemmatized': lemmatized,
            'processed': processed_text
        }
    
    def extract_features(self, reviews: List[str]) -> np.ndarray:
        """
        Extract TF-IDF features from reviews
        
        Args:
            reviews: List of review texts
            
        Returns:
            Feature matrix
        """
        # Preprocess all reviews
        processed_reviews = [self.preprocess_review(r)['processed'] for r in reviews]
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        # Fit and transform
        features = vectorizer.fit_transform(processed_reviews)
        
        return features.toarray()
    
    def detect_spam(self, review: str) -> Dict:
        """
        Detect potential spam/fake reviews
        
        Args:
            review: Review text
            
        Returns:
            Dictionary with spam indicators
        """
        indicators = {
            'is_spam': False,
            'spam_score': 0.0,
            'reasons': []
        }
        
        # Check review length
        if len(review) < 10:
            indicators['spam_score'] += 0.3
            indicators['reasons'].append('Very short review')
        
        # Check for excessive capitalization
        caps_ratio = sum(1 for c in review if c.isupper()) / max(len(review), 1)
        if caps_ratio > 0.5:
            indicators['spam_score'] += 0.2
            indicators['reasons'].append('Excessive capitalization')
        
        # Check for repeated characters
        if re.search(r'(.)\1{3,}', review):
            indicators['spam_score'] += 0.2
            indicators['reasons'].append('Repeated characters')
        
        # Check for promotional language
        promo_keywords = ['buy now', 'click here', 'limited offer', 'discount', 'sale']
        review_lower = review.lower()
        for keyword in promo_keywords:
            if keyword in review_lower:
                indicators['spam_score'] += 0.3
                indicators['reasons'].append(f'Promotional keyword: {keyword}')
                break
        
        # Check for excessive exclamation marks
        exclamation_count = review.count('!')
        if exclamation_count > 3:
            indicators['spam_score'] += 0.1
            indicators['reasons'].append('Excessive exclamation marks')
        
        # Determine if spam based on score
        if indicators['spam_score'] >= 0.5:
            indicators['is_spam'] = True
        
        return indicators
    
    def extract_aspects(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract aspects mentioned in the review
        
        Args:
            text: Review text
            
        Returns:
            List of (aspect, context) tuples
        """
        aspects_found = []
        text_lower = text.lower()
        
        for aspect in self.aspects:
            if aspect in text_lower:
                # Find context around the aspect
                pattern = rf'.{{0,30}}{aspect}.{{0,30}}'
                matches = re.findall(pattern, text_lower)
                for match in matches:
                    aspects_found.append((aspect, match.strip()))
        
        return aspects_found
    
    def calculate_credibility_score(self, review: Dict) -> float:
        """
        Calculate credibility score for a review
        
        Args:
            review: Review dictionary with various fields
            
        Returns:
            Credibility score between 0 and 1
        """
        score = 1.0
        
        # Check if verified purchase
        if review.get('verified_purchase', False):
            score += 0.2
        
        # Check review length
        review_length = len(review.get('text', ''))
        if review_length < 20:
            score -= 0.3
        elif review_length > 100:
            score += 0.1
        
        # Check for spam indicators
        spam_check = self.detect_spam(review.get('text', ''))
        score -= spam_check['spam_score']
        
        # Check if review has both title and text
        if review.get('title') and review.get('text'):
            score += 0.1
        
        # Normalize score to 0-1 range
        score = max(0, min(1, score))
        
        return score
    
    def preprocess_dataset(self, reviews_df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess entire dataset of reviews
        
        Args:
            reviews_df: DataFrame with reviews
            
        Returns:
            Preprocessed DataFrame
        """
        # Clean text
        reviews_df['cleaned_text'] = reviews_df['text'].apply(self.clean_text)
        
        # Extract aspects
        reviews_df['aspects'] = reviews_df['text'].apply(self.extract_aspects)
        
        # Calculate credibility scores
        reviews_df['credibility_score'] = reviews_df.apply(
            lambda row: self.calculate_credibility_score(row.to_dict()), 
            axis=1
        )
        
        # Detect spam
        spam_checks = reviews_df['text'].apply(self.detect_spam)
        reviews_df['is_spam'] = spam_checks.apply(lambda x: x['is_spam'])
        reviews_df['spam_score'] = spam_checks.apply(lambda x: x['spam_score'])
        
        # Filter out likely spam
        reviews_df['use_for_analysis'] = (
            (reviews_df['credibility_score'] > 0.3) & 
            (~reviews_df['is_spam'])
        )
        
        # Add text statistics
        reviews_df['text_length'] = reviews_df['text'].str.len()
        reviews_df['word_count'] = reviews_df['text'].str.split().str.len()
        
        return reviews_df
    
    def deduplicate_reviews(self, reviews_df: pd.DataFrame,
                           similarity_threshold: float = 0.9) -> pd.DataFrame:
        """
        Remove duplicate or near-duplicate reviews
        
        Args:
            reviews_df: DataFrame with reviews
            similarity_threshold: Threshold for considering reviews as duplicates
            
        Returns:
            DataFrame without duplicates
        """
        # Remove exact duplicates
        reviews_df = reviews_df.drop_duplicates(subset=['text'])
        
        # Remove near-duplicates using TF-IDF similarity
        if len(reviews_df) > 1:
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(reviews_df['text'].fillna(''))
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Find duplicates
            duplicates = set()
            for i in range(len(similarity_matrix)):
                for j in range(i + 1, len(similarity_matrix)):
                    if similarity_matrix[i][j] > similarity_threshold:
                        duplicates.add(j)
            
            # Remove duplicates
            keep_indices = [i for i in range(len(reviews_df)) if i not in duplicates]
            reviews_df = reviews_df.iloc[keep_indices].reset_index(drop=True)
        
        logger.info(f"Removed {len(duplicates) if 'duplicates' in locals() else 0} duplicate reviews")
        
        return reviews_df
