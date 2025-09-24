"""
Unit tests for AI models
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.absa_model import ABSASentimentAnalyzer
from models.spam_detector import SpamDetector
from models.recommendation_engine import PhoneRecommendationEngine


class TestABSAModel(unittest.TestCase):
    """Test cases for ABSA Sentiment Analyzer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = ABSASentimentAnalyzer()
        self.sample_reviews = [
            "This phone has an amazing camera but terrible battery life",
            "Great performance and beautiful screen, highly recommend!",
            "Worst phone ever, everything is broken",
            "Average phone, nothing special but works fine"
        ]
    
    def test_initialization(self):
        """Test model initialization"""
        self.assertIsNotNone(self.analyzer)
        self.assertIsInstance(self.analyzer.aspects, list)
        self.assertIn('camera', self.analyzer.aspects)
        self.assertIn('battery', self.analyzer.aspects)
    
    def test_vader_sentiment_analysis(self):
        """Test VADER sentiment analysis"""
        positive_text = "This phone is absolutely amazing and perfect!"
        negative_text = "Terrible phone, worst purchase ever"
        neutral_text = "The phone is okay"
        
        # Test positive sentiment
        result = self.analyzer.analyze_sentiment_vader(positive_text)
        self.assertEqual(result['sentiment'], 'positive')
        self.assertGreater(result['confidence'], 0.5)
        
        # Test negative sentiment
        result = self.analyzer.analyze_sentiment_vader(negative_text)
        self.assertEqual(result['sentiment'], 'negative')
        
        # Test neutral sentiment
        result = self.analyzer.analyze_sentiment_vader(neutral_text)
        self.assertIn(result['sentiment'], ['neutral', 'positive'])
    
    def test_simple_sentiment_fallback(self):
        """Test simple sentiment analysis fallback"""
        text = "Good phone with great features"
        result = self.analyzer._simple_sentiment_analysis(text)
        
        self.assertIn('sentiment', result)
        self.assertIn('scores', result)
        self.assertIn('confidence', result)
        self.assertEqual(result['sentiment'], 'positive')
    
    def test_extract_aspects(self):
        """Test aspect extraction from reviews"""
        review = "The camera is excellent but the battery drains quickly"
        aspects = self.analyzer.extract_aspects(review)
        
        self.assertIsInstance(aspects, list)
        self.assertIn('camera', aspects)
        self.assertIn('battery', aspects)
    
    def test_analyze_batch(self):
        """Test batch analysis of reviews"""
        reviews = [
            {'text': 'Great phone!', 'rating': 5},
            {'text': 'Terrible experience', 'rating': 1}
        ]
        
        results = self.analyzer.analyze_batch(reviews)
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results, pd.DataFrame)
    
    @patch('models.absa_model.TRANSFORMERS_AVAILABLE', False)
    def test_no_transformers_fallback(self):
        """Test behavior when transformers are not available"""
        analyzer = ABSASentimentAnalyzer()
        self.assertIsNone(analyzer.model)
        
        # Should still work with fallback
        result = analyzer.analyze_sentiment_vader("Great phone!")
        self.assertIsNotNone(result)


class TestSpamDetector(unittest.TestCase):
    """Test cases for Spam Detector"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.detector = SpamDetector()
        self.spam_reviews = [
            "CLICK HERE TO WIN!!! BEST DEAL EVER!!!",
            "Buy now at www.scam-site.com cheap phones",
            "⭐⭐⭐⭐⭐ AMAZING DEAL ⭐⭐⭐⭐⭐"
        ]
        self.legitimate_reviews = [
            "I've been using this phone for 3 months and it's great",
            "Good value for money, camera could be better",
            "Battery life is impressive, lasts all day with heavy use"
        ]
    
    def test_spam_detection(self):
        """Test spam detection accuracy"""
        for spam in self.spam_reviews:
            result = self.detector.is_spam(spam)
            self.assertTrue(result['is_spam'] or result['confidence'] > 0.5,
                          f"Failed to detect spam: {spam}")
        
        for legit in self.legitimate_reviews:
            result = self.detector.is_spam(legit)
            self.assertFalse(result['is_spam'] and result['confidence'] > 0.7,
                           f"False positive for: {legit}")
    
    def test_spam_indicators(self):
        """Test detection of spam indicators"""
        text = "!!!! BEST DEAL !!!! Click here NOW"
        indicators = self.detector.get_spam_indicators(text)
        
        self.assertIn('excessive_punctuation', indicators)
        self.assertIn('all_caps_words', indicators)
        self.assertIn('urgency_words', indicators)


class TestRecommendationEngine(unittest.TestCase):
    """Test cases for Recommendation Engine"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = PhoneRecommendationEngine()
        self.user_preferences = {
            'budget': 800,
            'brand_preference': ['Apple', 'Samsung'],
            'features': ['camera', 'battery'],
            'use_case': 'photography'
        }
    
    def test_get_recommendations(self):
        """Test recommendation generation"""
        recommendations = self.engine.get_recommendations(
            user_preferences=self.user_preferences,
            num_recommendations=5
        )
        
        self.assertIsInstance(recommendations, list)
        self.assertLessEqual(len(recommendations), 5)
        
        if recommendations:
            rec = recommendations[0]
            self.assertIn('phone_id', rec)
            self.assertIn('score', rec)
            self.assertIn('reasons', rec)
    
    def test_collaborative_filtering(self):
        """Test collaborative filtering component"""
        user_id = "test_user_123"
        similar_users = self.engine.find_similar_users(user_id)
        
        self.assertIsInstance(similar_users, list)
    
    def test_content_based_filtering(self):
        """Test content-based filtering"""
        phone_features = {
            'brand': 'Apple',
            'price': 999,
            'camera_score': 95,
            'battery_score': 85
        }
        
        similar_phones = self.engine.find_similar_phones(phone_features)
        self.assertIsInstance(similar_phones, list)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def test_end_to_end_analysis(self):
        """Test complete analysis pipeline"""
        # Create sample data
        reviews = pd.DataFrame({
            'text': [
                'Amazing phone with great camera',
                'Battery life is disappointing',
                'Perfect for gaming and photography'
            ],
            'rating': [5, 2, 4],
            'product': ['iPhone 15', 'iPhone 15', 'iPhone 15']
        })
        
        # Initialize components
        analyzer = ABSASentimentAnalyzer()
        spam_detector = SpamDetector()
        
        # Process reviews
        for _, review in reviews.iterrows():
            # Check spam
            spam_result = spam_detector.is_spam(review['text'])
            self.assertIn('is_spam', spam_result)
            
            # Analyze sentiment
            sentiment = analyzer.analyze_sentiment_vader(review['text'])
            self.assertIn('sentiment', sentiment)
            
            # Extract aspects
            aspects = analyzer.extract_aspects(review['text'])
            self.assertIsInstance(aspects, list)


class TestPerformance(unittest.TestCase):
    """Performance benchmark tests"""
    
    def test_sentiment_analysis_speed(self):
        """Test sentiment analysis performance"""
        analyzer = ABSASentimentAnalyzer()
        text = "This is a sample review text for performance testing"
        
        import time
        start = time.time()
        for _ in range(100):
            analyzer.analyze_sentiment_vader(text)
        elapsed = time.time() - start
        
        # Should process 100 reviews in less than 5 seconds
        self.assertLess(elapsed, 5.0, f"Too slow: {elapsed:.2f}s for 100 reviews")
    
    def test_batch_processing_efficiency(self):
        """Test batch processing efficiency"""
        analyzer = ABSASentimentAnalyzer()
        reviews = [{'text': f'Review {i}', 'rating': 3} for i in range(50)]
        
        import time
        start = time.time()
        results = analyzer.analyze_batch(reviews)
        elapsed = time.time() - start
        
        self.assertLess(elapsed, 10.0, f"Batch processing too slow: {elapsed:.2f}s")


if __name__ == '__main__':
    unittest.main()
