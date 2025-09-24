"""
Smart Review Summarization Engine
Intelligent review analysis and summarization with key theme extraction
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass
from collections import Counter, defaultdict
import re
from statistics import mode, median
import warnings

logger = logging.getLogger(__name__)

@dataclass
class ReviewSummary:
    """Represents a comprehensive review summary"""
    product_name: str
    total_reviews: int
    average_rating: float
    sentiment_distribution: Dict[str, float]
    key_themes: List[Dict[str, Any]]
    pros: List[str]
    cons: List[str]
    summary_text: str
    confidence_score: float
    recommendations: List[str]
    metadata: Dict[str, Any]
    timestamp: datetime

@dataclass
class ThemeAnalysis:
    """Represents analysis of a specific theme in reviews"""
    theme_name: str
    mentions: int
    sentiment_score: float
    importance_score: float
    representative_quotes: List[str]
    related_keywords: List[str]

class ReviewSummarizationEngine:
    """Advanced review summarization and theme extraction engine"""
    
    def __init__(self):
        """Initialize the review summarization engine"""
        
        # Theme keywords for phone reviews
        self.theme_keywords = self._initialize_theme_keywords()
        
        # Sentiment keywords
        self.sentiment_keywords = self._initialize_sentiment_keywords()
        
        # Summary templates
        self.summary_templates = self._initialize_summary_templates()
        
        # Analysis cache
        self.analysis_cache = {}
        
        logger.info("Review Summarization Engine initialized")
    
    def _initialize_theme_keywords(self) -> Dict[str, List[str]]:
        """Initialize theme keywords for different phone aspects"""
        return {
            'camera': [
                'camera', 'photo', 'picture', 'selfie', 'photography', 'lens', 'zoom', 
                'portrait', 'video', 'recording', 'quality', 'megapixel', 'flash',
                'night mode', 'macro', 'wide angle'
            ],
            'battery': [
                'battery', 'charge', 'charging', 'power', 'lasting', 'duration',
                'backup', 'drain', 'fast charge', 'wireless charging', 'standby',
                'usage', 'hours', 'day', 'life'
            ],
            'performance': [
                'performance', 'speed', 'fast', 'slow', 'lag', 'smooth', 'processor',
                'ram', 'memory', 'storage', 'app', 'gaming', 'multitask', 'responsive',
                'freeze', 'crash', 'hang'
            ],
            'display': [
                'screen', 'display', 'bright', 'brightness', 'color', 'resolution',
                'sharp', 'clear', 'size', 'inch', 'touch', 'responsive', 'refresh',
                'oled', 'lcd', 'hdr'
            ],
            'design': [
                'design', 'look', 'appearance', 'style', 'beautiful', 'ugly',
                'build', 'material', 'premium', 'cheap', 'weight', 'size',
                'grip', 'comfortable', 'sleek', 'elegant'
            ],
            'audio': [
                'audio', 'sound', 'speaker', 'volume', 'music', 'call', 'voice',
                'earphone', 'headphone', 'loud', 'clear', 'bass', 'quality'
            ],
            'connectivity': [
                'network', 'wifi', 'bluetooth', 'signal', 'connection', 'coverage',
                '5g', '4g', 'internet', 'data', 'hotspot', 'nfc'
            ],
            'usability': [
                'easy', 'difficult', 'user', 'interface', 'menu', 'setting',
                'intuitive', 'confusing', 'navigation', 'feature', 'function'
            ],
            'value': [
                'price', 'cost', 'expensive', 'cheap', 'worth', 'value', 'money',
                'affordable', 'budget', 'deal', 'overpriced', 'reasonable'
            ]
        }
    
    def _initialize_sentiment_keywords(self) -> Dict[str, List[str]]:
        """Initialize sentiment keywords for analysis"""
        return {
            'positive': [
                'excellent', 'amazing', 'fantastic', 'great', 'good', 'best',
                'love', 'perfect', 'awesome', 'outstanding', 'superb', 'wonderful',
                'impressive', 'satisfied', 'happy', 'pleased', 'recommend',
                'brilliant', 'exceptional', 'flawless', 'superior'
            ],
            'negative': [
                'terrible', 'awful', 'bad', 'worst', 'hate', 'horrible',
                'disappointing', 'poor', 'useless', 'broken', 'defective',
                'frustrated', 'annoying', 'regret', 'waste', 'fail', 'problem',
                'issue', 'bug', 'error', 'crash'
            ],
            'neutral': [
                'okay', 'average', 'normal', 'standard', 'typical', 'fine',
                'acceptable', 'decent', 'mediocre', 'ordinary', 'fair'
            ]
        }
    
    def _initialize_summary_templates(self) -> Dict[str, str]:
        """Initialize templates for different types of summaries"""
        return {
            'overall': "Based on {total_reviews} reviews, the {product} has an average rating of {avg_rating}/5.0. {sentiment_summary} {key_strengths} {main_concerns} {recommendation}",
            'positive': "Users particularly appreciate the {product}'s {top_pros}. {positive_highlights}",
            'negative': "Common complaints about the {product} include {top_cons}. {negative_concerns}",
            'balanced': "The {product} shows strong performance in {strengths} but has room for improvement in {weaknesses}."
        }
    
    def generate_summary(self, df: pd.DataFrame, product_name: str = None, 
                        summary_type: str = "comprehensive") -> ReviewSummary:
        """
        Generate comprehensive review summary
        
        Args:
            df: DataFrame with review data
            product_name: Name of the product to analyze
            summary_type: Type of summary to generate
            
        Returns:
            ReviewSummary object with comprehensive analysis
        """
        try:
            if df is None or df.empty:
                return self._create_empty_summary(product_name or "Unknown Product")
            
            # Filter by product if specified
            if product_name and 'product' in df.columns:
                df = df[df['product'].str.contains(product_name, case=False, na=False)]
                if df.empty:
                    return self._create_empty_summary(product_name)
            
            # Use most common product name if not specified
            if not product_name and 'product' in df.columns:
                product_name = df['product'].mode().iloc[0] if not df['product'].mode().empty else "Unknown Product"
            elif not product_name:
                product_name = "Product Reviews"
            
            logger.info(f"Generating summary for {product_name} with {len(df)} reviews")
            
            # Extract basic metrics
            total_reviews = len(df)
            average_rating = df['rating'].mean() if 'rating' in df.columns else 0.0
            
            # Analyze sentiment distribution
            sentiment_distribution = self._analyze_sentiment_distribution(df)
            
            # Extract key themes
            key_themes = self._extract_key_themes(df)
            
            # Identify pros and cons
            pros, cons = self._extract_pros_cons(df, key_themes)
            
            # Generate summary text
            summary_text = self._generate_summary_text(
                product_name, total_reviews, average_rating, 
                sentiment_distribution, pros, cons, key_themes
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(total_reviews, df)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                average_rating, sentiment_distribution, key_themes, pros, cons
            )
            
            # Create metadata
            metadata = self._create_metadata(df, key_themes)
            
            summary = ReviewSummary(
                product_name=product_name,
                total_reviews=total_reviews,
                average_rating=average_rating,
                sentiment_distribution=sentiment_distribution,
                key_themes=[theme.__dict__ for theme in key_themes],
                pros=pros,
                cons=cons,
                summary_text=summary_text,
                confidence_score=confidence_score,
                recommendations=recommendations,
                metadata=metadata,
                timestamp=datetime.now()
            )
            
            logger.info(f"Summary generated successfully for {product_name}")
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return self._create_empty_summary(product_name or "Unknown Product")
    
    def generate_comparative_summary(self, df: pd.DataFrame, 
                                   products: List[str]) -> Dict[str, ReviewSummary]:
        """Generate comparative summaries for multiple products"""
        
        comparative_summaries = {}
        
        for product in products:
            try:
                product_data = df[df['product'].str.contains(product, case=False, na=False)]
                if not product_data.empty:
                    summary = self.generate_summary(product_data, product)
                    comparative_summaries[product] = summary
                else:
                    comparative_summaries[product] = self._create_empty_summary(product)
            
            except Exception as e:
                logger.error(f"Error generating summary for {product}: {e}")
                comparative_summaries[product] = self._create_empty_summary(product)
        
        return comparative_summaries
    
    def _analyze_sentiment_distribution(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze sentiment distribution in reviews"""
        
        if 'sentiment_label' in df.columns:
            # Use existing sentiment labels
            sentiment_counts = df['sentiment_label'].value_counts()
            total = len(df)
            
            return {
                'positive': (sentiment_counts.get('positive', 0) / total) * 100,
                'negative': (sentiment_counts.get('negative', 0) / total) * 100,
                'neutral': (sentiment_counts.get('neutral', 0) / total) * 100
            }
        
        elif 'review_text' in df.columns:
            # Analyze text for sentiment keywords
            positive_count = 0
            negative_count = 0
            neutral_count = 0
            
            for text in df['review_text'].fillna(''):
                text_lower = str(text).lower()
                
                pos_matches = sum(1 for word in self.sentiment_keywords['positive'] if word in text_lower)
                neg_matches = sum(1 for word in self.sentiment_keywords['negative'] if word in text_lower)
                
                if pos_matches > neg_matches:
                    positive_count += 1
                elif neg_matches > pos_matches:
                    negative_count += 1
                else:
                    neutral_count += 1
            
            total = len(df)
            return {
                'positive': (positive_count / total) * 100 if total > 0 else 0,
                'negative': (negative_count / total) * 100 if total > 0 else 0,
                'neutral': (neutral_count / total) * 100 if total > 0 else 0
            }
        
        else:
            # Default distribution based on ratings
            if 'rating' in df.columns:
                ratings = df['rating'].dropna()
                positive = (ratings >= 4).sum()
                negative = (ratings <= 2).sum()
                neutral = len(ratings) - positive - negative
                total = len(ratings)
                
                return {
                    'positive': (positive / total) * 100 if total > 0 else 0,
                    'negative': (negative / total) * 100 if total > 0 else 0,
                    'neutral': (neutral / total) * 100 if total > 0 else 0
                }
            
            return {'positive': 50.0, 'negative': 25.0, 'neutral': 25.0}
    
    def _extract_key_themes(self, df: pd.DataFrame) -> List[ThemeAnalysis]:
        """Extract key themes from reviews"""
        
        themes = []
        
        if 'review_text' not in df.columns:
            return themes
        
        # Analyze each theme
        for theme_name, keywords in self.theme_keywords.items():
            mentions = 0
            sentiment_scores = []
            representative_quotes = []
            
            for idx, text in df['review_text'].fillna('').items():
                text_lower = str(text).lower()
                
                # Count keyword mentions
                theme_mentions = sum(1 for keyword in keywords if keyword in text_lower)
                
                if theme_mentions > 0:
                    mentions += 1
                    
                    # Extract sentiment for this mention
                    pos_words = sum(1 for word in self.sentiment_keywords['positive'] if word in text_lower)
                    neg_words = sum(1 for word in self.sentiment_keywords['negative'] if word in text_lower)
                    
                    if pos_words > neg_words:
                        sentiment_scores.append(1)
                    elif neg_words > pos_words:
                        sentiment_scores.append(-1)
                    else:
                        sentiment_scores.append(0)
                    
                    # Collect representative quotes (first few sentences mentioning theme)
                    if len(representative_quotes) < 3:
                        sentences = re.split(r'[.!?]+', str(text))
                        for sentence in sentences:
                            if any(keyword in sentence.lower() for keyword in keywords):
                                clean_sentence = sentence.strip()
                                if len(clean_sentence) > 20 and len(clean_sentence) < 150:
                                    representative_quotes.append(clean_sentence)
                                    break
            
            if mentions > 0:
                # Calculate theme metrics
                avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
                importance_score = (mentions / len(df)) * 100
                
                theme = ThemeAnalysis(
                    theme_name=theme_name,
                    mentions=mentions,
                    sentiment_score=avg_sentiment,
                    importance_score=importance_score,
                    representative_quotes=representative_quotes[:3],
                    related_keywords=keywords[:5]  # Top 5 keywords
                )
                themes.append(theme)
        
        # Sort by importance score
        themes.sort(key=lambda x: x.importance_score, reverse=True)
        
        return themes[:8]  # Return top 8 themes
    
    def _extract_pros_cons(self, df: pd.DataFrame, 
                          themes: List[ThemeAnalysis]) -> Tuple[List[str], List[str]]:
        """Extract pros and cons based on theme analysis"""
        
        pros = []
        cons = []
        
        # Analyze themes for positive/negative aspects
        for theme in themes:
            if theme.sentiment_score > 0.3 and theme.mentions >= 3:
                # Positive theme
                if theme.theme_name == 'camera':
                    pros.append(f"Excellent camera quality with {theme.mentions} positive mentions")
                elif theme.theme_name == 'battery':
                    pros.append(f"Good battery life satisfaction from users")
                elif theme.theme_name == 'performance':
                    pros.append(f"Strong performance praised by users")
                elif theme.theme_name == 'display':
                    pros.append(f"High-quality display appreciated by users")
                elif theme.theme_name == 'design':
                    pros.append(f"Attractive design and build quality")
                elif theme.theme_name == 'value':
                    pros.append(f"Good value for money according to reviews")
                else:
                    pros.append(f"Positive user experience with {theme.theme_name}")
            
            elif theme.sentiment_score < -0.3 and theme.mentions >= 3:
                # Negative theme
                if theme.theme_name == 'camera':
                    cons.append(f"Camera quality issues reported by users")
                elif theme.theme_name == 'battery':
                    cons.append(f"Battery life concerns from multiple users")
                elif theme.theme_name == 'performance':
                    cons.append(f"Performance issues mentioned in reviews")
                elif theme.theme_name == 'display':
                    cons.append(f"Display quality complaints from users")
                elif theme.theme_name == 'design':
                    cons.append(f"Design or build quality concerns")
                elif theme.theme_name == 'value':
                    cons.append(f"Price concerns - users find it expensive")
                else:
                    cons.append(f"User concerns with {theme.theme_name}")
        
        # Add general pros/cons based on sentiment analysis
        if 'rating' in df.columns:
            high_ratings = (df['rating'] >= 4.5).sum()
            low_ratings = (df['rating'] <= 2).sum()
            total_ratings = len(df['rating'].dropna())
            
            if high_ratings / total_ratings > 0.6:
                pros.insert(0, "High user satisfaction with majority of 4+ star ratings")
            
            if low_ratings / total_ratings > 0.2:
                cons.insert(0, "Significant number of users gave low ratings")
        
        # Limit to top 5 pros and cons
        return pros[:5], cons[:5]
    
    def _generate_summary_text(self, product_name: str, total_reviews: int, 
                              avg_rating: float, sentiment_dist: Dict[str, float],
                              pros: List[str], cons: List[str], 
                              themes: List[ThemeAnalysis]) -> str:
        """Generate comprehensive summary text"""
        
        # Start with overview
        summary_parts = []
        
        # Opening statement
        rating_desc = self._get_rating_description(avg_rating)
        summary_parts.append(
            f"Based on {total_reviews:,} user reviews, the {product_name} receives an average rating of "
            f"{avg_rating:.1f}/5.0, indicating {rating_desc} user satisfaction."
        )
        
        # Sentiment overview
        pos_pct = sentiment_dist.get('positive', 0)
        neg_pct = sentiment_dist.get('negative', 0)
        
        if pos_pct > 60:
            sentiment_summary = f"The majority of users ({pos_pct:.0f}%) express positive sentiments"
        elif neg_pct > 40:
            sentiment_summary = f"A significant portion of users ({neg_pct:.0f}%) express negative concerns"
        else:
            sentiment_summary = f"User opinions are mixed, with {pos_pct:.0f}% positive and {neg_pct:.0f}% negative feedback"
        
        summary_parts.append(sentiment_summary + ".")
        
        # Key strengths
        if pros:
            if len(pros) == 1:
                summary_parts.append(f"The main strength highlighted by users is: {pros[0].lower()}.")
            else:
                summary_parts.append(f"Key strengths include: {', '.join(pros[:3]).lower()}.")
        
        # Main concerns
        if cons:
            if len(cons) == 1:
                summary_parts.append(f"The primary concern raised by users is: {cons[0].lower()}.")
            else:
                summary_parts.append(f"Common concerns include: {', '.join(cons[:3]).lower()}.")
        
        # Top themes
        if themes:
            top_themes = [t.theme_name for t in themes[:3]]
            summary_parts.append(f"The most discussed aspects are {', '.join(top_themes)}.")
        
        # Overall recommendation
        if avg_rating >= 4.2 and pos_pct > 70:
            summary_parts.append("Overall, this phone is highly recommended by users.")
        elif avg_rating >= 3.5 and pos_pct > 50:
            summary_parts.append("This phone generally satisfies users despite some areas for improvement.")
        elif avg_rating < 3.0 or neg_pct > 50:
            summary_parts.append("Potential buyers should carefully consider the reported issues before purchasing.")
        else:
            summary_parts.append("This phone has mixed reviews and may suit specific user preferences.")
        
        return " ".join(summary_parts)
    
    def _get_rating_description(self, rating: float) -> str:
        """Get descriptive text for rating score"""
        if rating >= 4.5:
            return "excellent"
        elif rating >= 4.0:
            return "very good"
        elif rating >= 3.5:
            return "good"
        elif rating >= 3.0:
            return "fair"
        elif rating >= 2.5:
            return "below average"
        else:
            return "poor"
    
    def _calculate_confidence_score(self, total_reviews: int, df: pd.DataFrame) -> float:
        """Calculate confidence score for the summary"""
        
        confidence = 0.5  # Base confidence
        
        # More reviews = higher confidence
        if total_reviews >= 100:
            confidence += 0.3
        elif total_reviews >= 50:
            confidence += 0.2
        elif total_reviews >= 20:
            confidence += 0.1
        
        # Data quality factors
        if 'rating' in df.columns:
            rating_variance = df['rating'].var()
            if rating_variance < 1.0:  # Low variance = more consistent opinions
                confidence += 0.1
        
        if 'review_text' in df.columns:
            avg_text_length = df['review_text'].str.len().mean()
            if avg_text_length > 50:  # Longer reviews = more detailed feedback
                confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _generate_recommendations(self, avg_rating: float, sentiment_dist: Dict[str, float],
                                 themes: List[ThemeAnalysis], pros: List[str], 
                                 cons: List[str]) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        
        recommendations = []
        
        # Rating-based recommendations
        if avg_rating >= 4.5:
            recommendations.append("Highly recommended - excellent user satisfaction across reviews")
        elif avg_rating >= 4.0:
            recommendations.append("Recommended - strong user satisfaction with minor areas for improvement")
        elif avg_rating >= 3.0:
            recommendations.append("Consider carefully - mixed reviews suggest it may suit specific needs")
        else:
            recommendations.append("Exercise caution - multiple users report significant issues")
        
        # Sentiment-based recommendations
        pos_pct = sentiment_dist.get('positive', 0)
        neg_pct = sentiment_dist.get('negative', 0)
        
        if pos_pct > 80:
            recommendations.append("Strong positive consensus among users")
        elif neg_pct > 40:
            recommendations.append("Significant user concerns - investigate negative feedback carefully")
        
        # Theme-based recommendations
        positive_themes = [t for t in themes if t.sentiment_score > 0.3]
        negative_themes = [t for t in themes if t.sentiment_score < -0.3]
        
        if positive_themes:
            strong_points = [t.theme_name for t in positive_themes[:2]]
            recommendations.append(f"Particularly strong in: {', '.join(strong_points)}")
        
        if negative_themes:
            weak_points = [t.theme_name for t in negative_themes[:2]]
            recommendations.append(f"Consider alternatives if {', '.join(weak_points)} are important to you")
        
        # Specific use case recommendations
        camera_theme = next((t for t in themes if t.theme_name == 'camera'), None)
        if camera_theme and camera_theme.sentiment_score > 0.4:
            recommendations.append("Excellent choice for photography enthusiasts")
        
        battery_theme = next((t for t in themes if t.theme_name == 'battery'), None)
        if battery_theme and battery_theme.sentiment_score > 0.4:
            recommendations.append("Good option for heavy users who need long battery life")
        
        value_theme = next((t for t in themes if t.theme_name == 'value'), None)
        if value_theme and value_theme.sentiment_score > 0.3:
            recommendations.append("Offers good value for money according to users")
        
        return recommendations[:6]  # Limit to 6 recommendations
    
    def _create_metadata(self, df: pd.DataFrame, themes: List[ThemeAnalysis]) -> Dict[str, Any]:
        """Create metadata for the summary"""
        
        metadata = {
            'analysis_date': datetime.now().isoformat(),
            'total_reviews_analyzed': len(df),
            'themes_identified': len(themes),
            'data_sources': list(df.columns),
            'review_date_range': None,
            'top_themes': [t.theme_name for t in themes[:5]],
            'analysis_completeness': self._calculate_analysis_completeness(df)
        }
        
        # Add date range if available
        if 'timestamp' in df.columns:
            try:
                dates = pd.to_datetime(df['timestamp'], errors='coerce').dropna()
                if not dates.empty:
                    metadata['review_date_range'] = {
                        'earliest': dates.min().isoformat(),
                        'latest': dates.max().isoformat(),
                        'span_days': (dates.max() - dates.min()).days
                    }
            except:
                pass
        
        return metadata
    
    def _calculate_analysis_completeness(self, df: pd.DataFrame) -> float:
        """Calculate how complete the analysis is based on available data"""
        
        completeness = 0.0
        
        # Check for essential columns
        if 'review_text' in df.columns and df['review_text'].notna().sum() > 0:
            completeness += 0.4
        
        if 'rating' in df.columns and df['rating'].notna().sum() > 0:
            completeness += 0.3
        
        if 'product' in df.columns and df['product'].notna().sum() > 0:
            completeness += 0.1
        
        if 'sentiment_label' in df.columns and df['sentiment_label'].notna().sum() > 0:
            completeness += 0.1
        
        if 'timestamp' in df.columns and df['timestamp'].notna().sum() > 0:
            completeness += 0.1
        
        return completeness
    
    def _create_empty_summary(self, product_name: str) -> ReviewSummary:
        """Create empty summary for error cases"""
        
        return ReviewSummary(
            product_name=product_name,
            total_reviews=0,
            average_rating=0.0,
            sentiment_distribution={'positive': 0.0, 'negative': 0.0, 'neutral': 0.0},
            key_themes=[],
            pros=[],
            cons=[],
            summary_text=f"No review data available for {product_name}.",
            confidence_score=0.0,
            recommendations=["Insufficient data for recommendations"],
            metadata={'analysis_date': datetime.now().isoformat(), 'error': 'No data available'},
            timestamp=datetime.now()
        )
    
    def get_theme_details(self, theme_name: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Get detailed analysis for a specific theme"""
        
        if theme_name not in self.theme_keywords:
            return {'error': f'Theme {theme_name} not found'}
        
        keywords = self.theme_keywords[theme_name]
        theme_reviews = []
        sentiment_breakdown = {'positive': 0, 'negative': 0, 'neutral': 0}
        
        if 'review_text' not in df.columns:
            return {'error': 'No review text available for theme analysis'}
        
        # Find reviews mentioning this theme
        for idx, text in df['review_text'].fillna('').items():
            text_lower = str(text).lower()
            
            if any(keyword in text_lower for keyword in keywords):
                # Get review data
                review_data = df.loc[idx].to_dict()
                theme_reviews.append(review_data)
                
                # Analyze sentiment
                pos_words = sum(1 for word in self.sentiment_keywords['positive'] if word in text_lower)
                neg_words = sum(1 for word in self.sentiment_keywords['negative'] if word in text_lower)
                
                if pos_words > neg_words:
                    sentiment_breakdown['positive'] += 1
                elif neg_words > pos_words:
                    sentiment_breakdown['negative'] += 1
                else:
                    sentiment_breakdown['neutral'] += 1
        
        # Calculate statistics
        total_mentions = len(theme_reviews)
        if total_mentions == 0:
            return {'error': f'No reviews found mentioning {theme_name}'}
        
        avg_rating = np.mean([r.get('rating', 0) for r in theme_reviews if r.get('rating')])
        
        return {
            'theme_name': theme_name,
            'total_mentions': total_mentions,
            'percentage_of_reviews': (total_mentions / len(df)) * 100,
            'average_rating': avg_rating,
            'sentiment_breakdown': sentiment_breakdown,
            'sample_reviews': theme_reviews[:5],  # First 5 reviews
            'keywords': keywords
        }
    
    def generate_quick_summary(self, df: pd.DataFrame, max_length: int = 200) -> str:
        """Generate a quick, concise summary"""
        
        if df.empty:
            return "No review data available."
        
        # Basic stats
        total_reviews = len(df)
        avg_rating = df['rating'].mean() if 'rating' in df.columns else 0
        
        # Quick sentiment
        if 'sentiment_label' in df.columns:
            sentiment_counts = df['sentiment_label'].value_counts()
            pos_pct = (sentiment_counts.get('positive', 0) / total_reviews) * 100
        else:
            pos_pct = 50  # Default
        
        # Product name
        product = df['product'].mode().iloc[0] if 'product' in df.columns and not df['product'].mode().empty else "This product"
        
        # Generate concise summary
        if avg_rating >= 4.0 and pos_pct > 70:
            summary = f"{product} receives excellent reviews ({avg_rating:.1f}/5 from {total_reviews} users) with {pos_pct:.0f}% positive feedback."
        elif avg_rating >= 3.5:
            summary = f"{product} has good user satisfaction ({avg_rating:.1f}/5 from {total_reviews} reviews) with mostly positive feedback."
        else:
            summary = f"{product} has mixed reviews ({avg_rating:.1f}/5 from {total_reviews} users) with varied user experiences."
        
        # Truncate if needed
        if len(summary) > max_length:
            summary = summary[:max_length-3] + "..."
        
        return summary

# Usage example and testing
if __name__ == "__main__":
    # Create sample data for testing
    sample_data = pd.DataFrame({
        'product': ['iPhone 15 Pro'] * 100 + ['Galaxy S24'] * 50,
        'review_text': [
            'Amazing camera quality and great battery life',
            'Excellent performance but expensive price',
            'Love the design and display is fantastic',
            'Battery drains quickly, disappointing',
            'Camera is good but not worth the high cost',
            'Perfect phone with great features',
            'Screen is beautiful and responsive',
            'Poor battery life affects daily usage',
            'Great value for money, recommended',
            'Camera quality exceeded expectations'
        ] * 15,  # Repeat to get 150 reviews
        'rating': np.random.uniform(2.0, 5.0, 150),
        'sentiment_label': np.random.choice(['positive', 'negative', 'neutral'], 150, p=[0.6, 0.25, 0.15])
    })
    
    # Initialize engine
    engine = ReviewSummarizationEngine()
    
    # Generate summary
    summary = engine.generate_summary(sample_data, "iPhone 15 Pro")
    
    print("=== Review Summary ===")
    print(f"Product: {summary.product_name}")
    print(f"Total Reviews: {summary.total_reviews}")
    print(f"Average Rating: {summary.average_rating:.1f}/5.0")
    print(f"Confidence Score: {summary.confidence_score:.2f}")
    print()
    print("Summary:")
    print(summary.summary_text)
    print()
    print("Key Themes:")
    for theme in summary.key_themes[:5]:
        print(f"- {theme['theme_name']}: {theme['mentions']} mentions (sentiment: {theme['sentiment_score']:.2f})")
    print()
    print("Pros:")
    for pro in summary.pros:
        print(f"+ {pro}")
    print()
    print("Cons:")
    for con in summary.cons:
        print(f"- {con}")
    print()
    print("Recommendations:")
    for rec in summary.recommendations:
        print(f"• {rec}")