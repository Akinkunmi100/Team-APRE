"""
Aspect-Based Sentiment Analysis (ABSA) Model
Using transformer models for aspect extraction and sentiment classification
"""

import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd

# Make torch optional - it's having installation issues on Windows
try:
    import torch
    from torch import nn
    TORCH_AVAILABLE = True
except (ImportError, OSError) as e:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    print(f"PyTorch not available (installation issue on Windows): {e}")

# Make transformers optional
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
os.environ['TRANSFORMERS_NO_TF'] = '1'  # Disable TensorFlow in transformers

try:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        pipeline,
        DebertaV2ForSequenceClassification,
        DebertaV2Tokenizer
    )
    TRANSFORMERS_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    TRANSFORMERS_AVAILABLE = False
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    pipeline = None
    print(f"Transformers not available (installation issue): {e}")

# Import VADER - should be installed
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
VADER_AVAILABLE = True

# Import TextBlob - should be installed
from textblob import TextBlob
TEXTBLOB_AVAILABLE = True

import yaml

logger = logging.getLogger(__name__)


class ABSASentimentAnalyzer:
    """
    Aspect-Based Sentiment Analysis for Phone Reviews
    Combines multiple approaches: VADER, TextBlob, and Transformer models
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize ABSA model with configuration"""
        logger.info("Initializing ABSASentimentAnalyzer...")

        try:
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)

            self.aspects = self.config['sentiment_analysis']['aspects']
            self.sentiment_labels = self.config['sentiment_analysis']['sentiment_labels']
            self.confidence_threshold = self.config['sentiment_analysis']['confidence_threshold']
            logger.info("Configuration loaded successfully")
        except Exception as e:
            logger.warning(f"Config file not found or invalid ({e}), using defaults")
            # Use defaults if config not found
            self.aspects = ['camera', 'battery', 'screen', 'performance', 'price', 'design', 'software']
            self.sentiment_labels = ['positive', 'negative', 'neutral']
            self.confidence_threshold = 0.7
        
        # Initialize VADER for rule-based sentiment if available
        if VADER_AVAILABLE:
            self.vader = SentimentIntensityAnalyzer()
        else:
            self.vader = None
            logger.warning("VADER not available - using fallback sentiment analysis")
        
        # Initialize transformer model if available
        if TORCH_AVAILABLE:
            logger.info("PyTorch available, initializing device...")
            try:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                logger.info(f"Device initialized: {self.device}")
                if torch.cuda.is_available():
                    logger.info(f"CUDA device count: {torch.cuda.device_count()}")
                    logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
            except Exception as e:
                logger.error(f"Error initializing PyTorch device: {e}")
                self.device = torch.device('cpu')
        else:
            logger.info("PyTorch not available, skipping device initialization")
            self.device = None

        logger.info("Calling _initialize_transformer...")
        self._initialize_transformer()
        logger.info("Transformer initialization completed")
        
    def _initialize_transformer(self):
        """Initialize transformer model for ABSA"""
        logger.info("Starting transformer initialization...")

        if not TRANSFORMERS_AVAILABLE or not TORCH_AVAILABLE:
            logger.warning("Transformers or PyTorch not available - using basic sentiment analysis")
            self.tokenizer = None
            self.model = None
            self.sentiment_pipeline = None
            return

        try:
            # Skip custom BERT model initialization to avoid random classifier weights warning
            # Instead, rely on the pre-trained sentiment pipeline which is already fine-tuned
            logger.info("Skipping custom BERT model initialization to avoid random classifier weights")
            self.tokenizer = None
            self.model = None

            # Create sentiment pipeline with pre-trained model
            logger.info("Creating sentiment analysis pipeline with pre-trained model...")
            logger.info(f"Device available: {self.device}")
            logger.info(f"CUDA available: {torch.cuda.is_available()}")

            # Log before pipeline creation - this is likely where the warning occurs
            logger.info("About to create pipeline - monitoring for PyTorch distributed warnings...")
            import warnings
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")

                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model="nlptown/bert-base-multilingual-uncased-sentiment",
                    device=0 if torch.cuda.is_available() else -1
                )

                # Check for warnings
                for warning in w:
                    if "torch.distributed.elastic.multiprocessing.redirects" in str(warning.message):
                        logger.warning(f"PyTorch distributed warning captured during pipeline creation: {warning.message}")
                        logger.warning(f"Warning category: {warning.category}")
                        logger.warning(f"Warning filename: {warning.filename}:{warning.lineno}")

            logger.info("Transformer model initialized successfully with pre-trained sentiment pipeline")

        except Exception as e:
            logger.error(f"Error initializing transformer: {e}")
            logger.error(f"Error type: {type(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            self.tokenizer = None
            self.model = None
            self.sentiment_pipeline = None
    
    def analyze_sentiment_vader(self, text: str) -> Dict:
        """
        Analyze sentiment using VADER
        
        Args:
            text: Review text
            
        Returns:
            Sentiment scores dictionary
        """
        if not self.vader:
            # Fallback to simple sentiment analysis
            return self._simple_sentiment_analysis(text)
        
        scores = self.vader.polarity_scores(text)
        
        # Determine overall sentiment
        if scores['compound'] >= 0.05:
            sentiment = 'positive'
        elif scores['compound'] <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'scores': scores,
            'confidence': abs(scores['compound'])
        }
    
    def _simple_sentiment_analysis(self, text: str) -> Dict:
        """
        Simple rule-based sentiment analysis fallback
        
        Args:
            text: Review text
            
        Returns:
            Sentiment scores dictionary
        """
        text_lower = text.lower()
        
        # Simple word lists for sentiment
        positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'best', 'love', 'perfect', 'awesome', 'fantastic'}
        negative_words = {'bad', 'terrible', 'awful', 'worst', 'hate', 'horrible', 'poor', 'disappointing', 'broken', 'useless'}
        
        # Count sentiment words
        words = text_lower.split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        total_words = len(words)
        
        # Calculate scores
        if total_words == 0:
            compound = 0
        else:
            compound = (positive_count - negative_count) / total_words
        
        # Determine sentiment
        if compound >= 0.05:
            sentiment = 'positive'
        elif compound <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'scores': {
                'pos': positive_count / max(total_words, 1),
                'neg': negative_count / max(total_words, 1),
                'neu': 1 - (positive_count + negative_count) / max(total_words, 1),
                'compound': compound
            },
            'confidence': min(abs(compound) * 2, 1.0)
        }
    
    def analyze_sentiment_textblob(self, text: str) -> Dict:
        """
        Analyze sentiment using TextBlob
        
        Args:
            text: Review text
            
        Returns:
            Sentiment scores dictionary
        """
        if not TEXTBLOB_AVAILABLE:
            # Fallback to simple sentiment analysis
            return self._simple_sentiment_analysis(text)
        
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Determine sentiment based on polarity
        if polarity > 0.1:
            sentiment = 'positive'
        elif polarity < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'polarity': polarity,
            'subjectivity': subjectivity,
            'confidence': abs(polarity)
        }
    
    def analyze_sentiment_transformer(self, text: str) -> Dict:
        """
        Analyze sentiment using transformer model
        
        Args:
            text: Review text
            
        Returns:
            Sentiment scores dictionary
        """
        if not self.sentiment_pipeline:
            return {'sentiment': 'neutral', 'confidence': 0.0}
        
        try:
            # Truncate text if too long
            max_length = 512
            if len(text) > max_length:
                text = text[:max_length]
            
            # Get prediction
            result = self.sentiment_pipeline(text)[0]
            
            # Map to our sentiment labels
            label_mapping = {
                '1 star': 'negative',
                '2 stars': 'negative',
                '3 stars': 'neutral',
                '4 stars': 'positive',
                '5 stars': 'positive'
            }
            
            sentiment = label_mapping.get(result['label'], 'neutral')
            
            return {
                'sentiment': sentiment,
                'confidence': result['score'],
                'raw_label': result['label']
            }
            
        except Exception as e:
            logger.error(f"Error in transformer sentiment analysis: {e}")
            return {'sentiment': 'neutral', 'confidence': 0.0}
    
    def extract_aspect_sentiments(self, text: str) -> List[Dict]:
        """
        Extract aspects and their associated sentiments from text
        
        Args:
            text: Review text
            
        Returns:
            List of aspect-sentiment pairs
        """
        aspect_sentiments = []
        text_lower = text.lower()
        
        for aspect in self.aspects:
            if aspect in text_lower:
                # Find sentences containing the aspect
                sentences = text.split('.')
                for sentence in sentences:
                    if aspect in sentence.lower():
                        # Analyze sentiment of this sentence
                        sentiment_result = self.analyze_combined_sentiment(sentence)
                        
                        aspect_sentiments.append({
                            'aspect': aspect,
                            'context': sentence.strip(),
                            'sentiment': sentiment_result['sentiment'],
                            'confidence': sentiment_result['confidence']
                        })
        
        return aspect_sentiments
    
    def analyze_combined_sentiment(self, text: str) -> Dict:
        """
        Combine multiple sentiment analysis approaches
        
        Args:
            text: Review text
            
        Returns:
            Combined sentiment analysis result
        """
        # Get results from different methods
        vader_result = self.analyze_sentiment_vader(text)
        textblob_result = self.analyze_sentiment_textblob(text)
        transformer_result = self.analyze_sentiment_transformer(text)
        
        # Combine results using weighted voting
        sentiment_votes = {
            'positive': 0,
            'negative': 0,
            'neutral': 0
        }
        
        # Weight based on confidence
        weights = {
            'vader': 0.3,
            'textblob': 0.2,
            'transformer': 0.5
        }
        
        # Add weighted votes
        sentiment_votes[vader_result['sentiment']] += weights['vader'] * vader_result['confidence']
        sentiment_votes[textblob_result['sentiment']] += weights['textblob'] * textblob_result['confidence']
        sentiment_votes[transformer_result['sentiment']] += weights['transformer'] * transformer_result['confidence']
        
        # Determine final sentiment
        final_sentiment = max(sentiment_votes, key=sentiment_votes.get)
        confidence = sentiment_votes[final_sentiment] / sum(weights.values())
        
        return {
            'sentiment': final_sentiment,
            'confidence': confidence,
            'scores': sentiment_votes,
            'methods': {
                'vader': vader_result,
                'textblob': textblob_result,
                'transformer': transformer_result
            }
        }
    
    def analyze_review(self, review: Dict) -> Dict:
        """
        Complete ABSA analysis for a single review
        
        Args:
            review: Review dictionary with text and metadata
            
        Returns:
            Analysis results
        """
        text = review.get('text', '')
        
        # Overall sentiment analysis
        overall_sentiment = self.analyze_combined_sentiment(text)
        
        # Aspect-based sentiment analysis
        aspect_sentiments = self.extract_aspect_sentiments(text)
        
        # Aggregate aspect sentiments
        aspect_summary = self._summarize_aspects(aspect_sentiments)
        
        return {
            'review_id': review.get('id'),
            'overall_sentiment': overall_sentiment,
            'aspect_sentiments': aspect_sentiments,
            'aspect_summary': aspect_summary,
            'text': text,
            'credibility_score': review.get('credibility_score', 1.0)
        }
    
    def _summarize_aspects(self, aspect_sentiments: List[Dict]) -> Dict:
        """
        Summarize sentiments for each aspect
        
        Args:
            aspect_sentiments: List of aspect-sentiment pairs
            
        Returns:
            Summary dictionary
        """
        summary = {}
        
        for item in aspect_sentiments:
            aspect = item['aspect']
            if aspect not in summary:
                summary[aspect] = {
                    'positive': 0,
                    'negative': 0,
                    'neutral': 0,
                    'mentions': 0
                }
            
            summary[aspect][item['sentiment']] += 1
            summary[aspect]['mentions'] += 1
        
        # Calculate dominant sentiment for each aspect
        for aspect in summary:
            mentions = summary[aspect]['mentions']
            if mentions > 0:
                summary[aspect]['dominant_sentiment'] = max(
                    ['positive', 'negative', 'neutral'],
                    key=lambda s: summary[aspect][s]
                )
                summary[aspect]['positive_ratio'] = summary[aspect]['positive'] / mentions
                summary[aspect]['negative_ratio'] = summary[aspect]['negative'] / mentions
        
        return summary

    def analyze_sentiment(self, text: str) -> Dict:
        """
        Analyze sentiment and return result in the expected format for run_app.py

        Args:
            text: Review text

        Returns:
            Dictionary with 'polarity' and 'aspects' keys as expected by run_app.py
        """
        # Get combined sentiment analysis
        combined_result = self.analyze_combined_sentiment(text)

        # Get aspect sentiments
        aspect_sentiments = self.extract_aspect_sentiments(text)

        # Convert to expected format
        return {
            'polarity': combined_result['confidence'] if combined_result['sentiment'] == 'positive'
                       else -combined_result['confidence'] if combined_result['sentiment'] == 'negative'
                       else 0.0,
            'aspects': [{'aspect': item['aspect'], 'sentiment': item['sentiment']}
                       for item in aspect_sentiments]
        }
    
    def analyze_batch(self, reviews: List[Dict]) -> pd.DataFrame:
        """
        Analyze a batch of reviews
        
        Args:
            reviews: List of review dictionaries
            
        Returns:
            DataFrame with analysis results
        """
        results = []
        
        for review in reviews:
            try:
                analysis = self.analyze_review(review)
                results.append(analysis)
            except Exception as e:
                logger.error(f"Error analyzing review: {e}")
                continue
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(results)
        
        return df
    
    def generate_summary(self, analysis_df: pd.DataFrame) -> Dict:
        """
        Generate overall summary from analysis results
        
        Args:
            analysis_df: DataFrame with analysis results
            
        Returns:
            Summary dictionary
        """
        summary = {
            'total_reviews': len(analysis_df),
            'overall_sentiment_distribution': {},
            'aspect_analysis': {},
            'key_insights': []
        }
        
        # Overall sentiment distribution
        if 'overall_sentiment' in analysis_df.columns:
            sentiments = analysis_df['overall_sentiment'].apply(lambda x: x['sentiment'])
            summary['overall_sentiment_distribution'] = sentiments.value_counts().to_dict()
        
        # Aggregate aspect analysis
        all_aspects = {}
        for _, row in analysis_df.iterrows():
            if 'aspect_summary' in row and row['aspect_summary']:
                for aspect, data in row['aspect_summary'].items():
                    if aspect not in all_aspects:
                        all_aspects[aspect] = {
                            'positive': 0,
                            'negative': 0,
                            'neutral': 0,
                            'total_mentions': 0
                        }
                    
                    all_aspects[aspect]['positive'] += data.get('positive', 0)
                    all_aspects[aspect]['negative'] += data.get('negative', 0)
                    all_aspects[aspect]['neutral'] += data.get('neutral', 0)
                    all_aspects[aspect]['total_mentions'] += data.get('mentions', 0)
        
        # Calculate percentages and identify key insights
        for aspect, data in all_aspects.items():
            total = data['total_mentions']
            if total > 0:
                data['positive_percentage'] = (data['positive'] / total) * 100
                data['negative_percentage'] = (data['negative'] / total) * 100
                
                # Identify strong points and weaknesses
                if data['positive_percentage'] > 70:
                    summary['key_insights'].append(f"Strong positive feedback on {aspect}")
                elif data['negative_percentage'] > 50:
                    summary['key_insights'].append(f"Significant concerns about {aspect}")
        
        summary['aspect_analysis'] = all_aspects
        
        return summary
