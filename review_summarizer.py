"""
Advanced Review Summarization Enhancement Module
Provides intelligent summaries, key insights, and structured analysis
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import re
import numpy as np
from collections import Counter, defaultdict
from transformers import (
    pipeline, 
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration,
    BartForConditionalGeneration
)
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from textblob import TextBlob

logger = logging.getLogger(__name__)


@dataclass
class ReviewSummary:
    """Structured summary output"""
    executive_summary: str
    tldr: str
    key_points: List[str]
    pros: List[str]
    cons: List[str]
    common_themes: List[Dict[str, Any]]
    unique_insights: List[str]
    technical_details: Dict[str, Any]
    user_recommendations: Dict[str, Any]
    consensus_score: float
    summary_metadata: Dict[str, Any]


class AdvancedReviewSummarizer:
    """
    Advanced review summarization with multiple techniques:
    - Executive summaries
    - Bullet point extraction
    - Theme clustering
    - Pros/cons extraction
    - Technical detail mining
    - Consensus building
    """
    
    def __init__(self, device: str = None):
        """Initialize summarization models and components"""
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Initializing Advanced Review Summarizer on {self.device}")
        
        # Load models
        self._load_models()
        
        # Load spaCy for NLP tasks
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            logger.warning("spaCy model not found. Some features will be limited.")
            self.nlp = None
        
        # Initialize configurations
        self.config = {
            'max_summary_length': 150,
            'min_summary_length': 50,
            'num_key_points': 5,
            'num_pros': 5,
            'num_cons': 5,
            'clustering_threshold': 0.7,
            'min_theme_frequency': 2
        }
    
    def _load_models(self):
        """Load all required models"""
        try:
            # Main summarization model (BART)
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=0 if self.device == 'cuda' else -1
            )
            
            # T5 for flexible text generation
            self.t5_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
            self.t5_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
            self.t5_model.to(self.device)
            
            # Sentence embeddings for clustering
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Sentiment analysis
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment",
                device=0 if self.device == 'cuda' else -1
            )
            
            logger.info("All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            # Fallback to lighter models if needed
            self._load_fallback_models()
    
    def _load_fallback_models(self):
        """Load lighter fallback models"""
        try:
            self.summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
            self.t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")
            self.t5_tokenizer = AutoTokenizer.from_pretrained("t5-small")
            self.sentiment_analyzer = pipeline("sentiment-analysis")
            logger.info("Fallback models loaded")
        except Exception as e:
            logger.error(f"Failed to load fallback models: {e}")
    
    def summarize_reviews(
        self,
        reviews: List[str],
        product_name: str = "product",
        max_reviews: int = 100
    ) -> ReviewSummary:
        """
        Generate comprehensive summary from reviews
        
        Args:
            reviews: List of review texts
            product_name: Name of the product
            max_reviews: Maximum reviews to process
            
        Returns:
            ReviewSummary object with all insights
        """
        logger.info(f"Starting summarize_reviews with {len(reviews)} reviews for product: {product_name}")
        if not reviews:
            return self._empty_summary()
        
        # Limit reviews if too many
        if len(reviews) > max_reviews:
            reviews = self._sample_representative_reviews(reviews, max_reviews)
        
        # Clean and preprocess reviews
        cleaned_reviews = [self._clean_text(r) for r in reviews]
        
        # Generate different summary components
        executive_summary = self._generate_executive_summary(cleaned_reviews, product_name)
        tldr = self._generate_tldr(executive_summary)
        key_points = self._extract_key_points(cleaned_reviews)
        pros, cons = self._extract_pros_cons(cleaned_reviews)
        themes = self._identify_themes(cleaned_reviews)
        unique_insights = self._find_unique_insights(cleaned_reviews)
        technical_details = self._extract_technical_details(cleaned_reviews)
        user_recommendations = self._analyze_recommendations(cleaned_reviews)
        consensus_score = self._calculate_consensus(cleaned_reviews)
        
        # Generate metadata
        metadata = self._generate_metadata(reviews, cleaned_reviews)
        
        return ReviewSummary(
            executive_summary=executive_summary,
            tldr=tldr,
            key_points=key_points,
            pros=pros,
            cons=cons,
            common_themes=themes,
            unique_insights=unique_insights,
            technical_details=technical_details,
            user_recommendations=user_recommendations,
            consensus_score=consensus_score,
            summary_metadata=metadata
        )
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize review text"""
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        # Remove special characters but keep sentence structure
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def _sample_representative_reviews(self, reviews: List[str], n: int) -> List[str]:
        """Sample representative reviews using clustering"""
        if len(reviews) <= n:
            return reviews
        
        # Create embeddings
        embeddings = self.sentence_model.encode(reviews)
        
        # Cluster reviews
        n_clusters = min(n, len(reviews))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        
        # Sample one review from each cluster (closest to centroid)
        sampled = []
        for i in range(n_clusters):
            cluster_indices = np.where(clusters == i)[0]
            cluster_embeddings = embeddings[cluster_indices]
            centroid = kmeans.cluster_centers_[i]
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            closest_idx = cluster_indices[np.argmin(distances)]
            sampled.append(reviews[closest_idx])
        
        return sampled
    
    def _generate_executive_summary(self, reviews: List[str], product_name: str) -> str:
        """Generate executive summary using BART"""
        # Combine reviews into chunks
        max_chunk_size = 1024
        chunks = []
        current_chunk = ""
        
        for review in reviews:
            if len(current_chunk) + len(review) < max_chunk_size:
                current_chunk += " " + review
            else:
                chunks.append(current_chunk)
                current_chunk = review
        
        if current_chunk:
            chunks.append(current_chunk)
        
        # Summarize each chunk
        chunk_summaries = []
        for chunk in chunks[:5]:  # Limit to 5 chunks:
            try:
                summary = self.summarizer(
                    chunk,
                    max_length=self.config['max_summary_length'],
                    min_length=self.config['min_summary_length'],
                    do_sample=False
                )
                chunk_summaries.append(summary[0]['summary_text'])
            except Exception as e:
                logger.error(f"Error summarizing chunk: {e}")
                continue
        
        # Combine chunk summaries
        if len(chunk_summaries) > 1:
            combined = " ".join(chunk_summaries)
            final_summary = self.summarizer(
                combined,
                max_length=self.config['max_summary_length'],
                min_length=self.config['min_summary_length'],
                do_sample=False
            )
            return final_summary[0]['summary_text']
        elif chunk_summaries:
            return chunk_summaries[0]
        else:
            return f"Based on user reviews, {product_name} has received mixed feedback."
    
    def _generate_tldr(self, executive_summary: str) -> str:
        """Generate TL;DR using T5"""
        prompt = f"Summarize in one sentence: {executive_summary}"
        
        inputs = self.t5_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.t5_model.generate(
                **inputs,
                max_length=50,
                min_length=10,
                temperature=0.7,
                num_beams=4,
                early_stopping=True
            )
        
        tldr = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return tldr
    
    def _extract_key_points(self, reviews: List[str]) -> List[str]:
        """Extract key points using T5"""
        combined_text = " ".join(reviews[:20])  # Use first 20 reviews
        prompt = f"List the 5 most important points from these reviews: {combined_text[:1500]}"
        
        inputs = self.t5_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.t5_model.generate(
                **inputs,
                max_length=200,
                min_length=50,
                temperature=0.8,
                num_beams=4,
                early_stopping=True
            )
        
        key_points_text = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parse the output into bullet points
        key_points = []
        for line in key_points_text.split('.'):
            line = line.strip()
            if line and len(line) > 10:
                # Clean up numbering if present
                line = re.sub(r'^\d+[\.\)]\s*', '', line)
                key_points.append(line)
        
        return key_points[:self.config['num_key_points']]
    
    def _extract_pros_cons(self, reviews: List[str]) -> Tuple[List[str], List[str]]:
        """Extract pros and cons from reviews"""
        pros = []
        cons = []
        
        # Keywords for identifying pros and cons
        pros_keywords = ['love', 'excellent', 'amazing', 'great', 'fantastic', 'perfect', 
                        'best', 'awesome', 'outstanding', 'impressive', 'good']
        cons_keywords = ['hate', 'terrible', 'awful', 'worst', 'disappointing', 'poor',
                        'bad', 'issue', 'problem', 'defect', 'broken', 'useless']
        
        for review in reviews:
            sentences = review.split('.')
            for sentence in sentences:
                sentence_lower = sentence.lower()
                
                # Check sentiment
                if any(word in sentence_lower for word in pros_keywords):
                    # Extract the key phrase
                    pros.append(self._extract_key_phrase(sentence))
                elif any(word in sentence_lower for word in cons_keywords):
                    cons.append(self._extract_key_phrase(sentence))
        
        # Use sentiment analysis for more nuanced extraction
        for review in reviews[:30]:  # Analyze first 30 reviews:
            try:
                sentiment = self.sentiment_analyzer(review[:512])[0]
                if sentiment['score'] > 0.7:
                    if sentiment['label'] in ['5 stars', '4 stars', 'POSITIVE']:
                        pros.append(self._extract_key_phrase(review))
                elif sentiment['score'] > 0.7 and sentiment['label'] in ['1 star', '2 stars', 'NEGATIVE']:
                    cons.append(self._extract_key_phrase(review))
            except:
                continue
        
        # Deduplicate and rank
        pros = self._deduplicate_and_rank(pros)[:self.config['num_pros']]
        cons = self._deduplicate_and_rank(cons)[:self.config['num_cons']]
        
        return pros, cons
    
    def _extract_key_phrase(self, text: str) -> str:
        """Extract the most important phrase from text"""
        if not text:
            return ""
        
        # Clean the text
        text = text.strip()
        if len(text) > 100:
            text = text[:100] + "..."
        
        # Remove common starting words
        text = re.sub(r'^(the |a |an |this |that |it |they |we |i )+', '', text.lower())
        
        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]
        
        return text
    
    def _deduplicate_and_rank(self, items: List[str]) -> List[str]:
        """Remove duplicates and rank by frequency/importance"""
        if not items:
            return []
        
        # Use embeddings to find similar items
        embeddings = self.sentence_model.encode(items)
        
        # Group similar items
        groups = []
        used = set()
        
        for i, item in enumerate(items):
            if i in used:
                continue
            
            group = [item]
            used.add(i)
            
            for j in range(i + 1, len(items)):
                if j in used:
                    continue
                
                similarity = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                if similarity > self.config['clustering_threshold']:
                    group.append(items[j])
                    used.add(j)
            
            groups.append(group)
        
        # Sort groups by size (frequency) and return representatives
        groups.sort(key=lambda x: len(x), reverse=True)
        return [group[0] for group in groups]
    
    def _identify_themes(self, reviews: List[str]) -> List[Dict[str, Any]]:
        """Identify common themes using clustering"""
        if not reviews:
            return []
        
        # Extract sentences
        sentences = []
        for review in reviews:
            sentences.extend(review.split('.'))
        
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if len(sentences) < 5:
            return []
        
        # Embed sentences
        embeddings = self.sentence_model.encode(sentences[:100])  # Limit for performance
        
        # Cluster sentences
        n_clusters = min(10, len(sentences) // 5)
        if n_clusters < 2:
            return []
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        
        # Analyze each cluster
        themes = []
        for i in range(n_clusters):
            cluster_sentences = [sentences[j] for j in range(len(sentences)) if clusters[j] == i]
            
            if len(cluster_sentences) < self.config['min_theme_frequency']:
                continue
            
            # Find representative sentence (closest to centroid)
            cluster_embeddings = embeddings[clusters == i]
            centroid = kmeans.cluster_centers_[i]
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            representative_idx = np.argmin(distances)
            representative = cluster_sentences[representative_idx]
            
            # Extract theme name using T5
            theme_name = self._extract_theme_name(cluster_sentences)
            
            themes.append({
                'name': theme_name,
                'description': representative,
                'frequency': len(cluster_sentences),
                'examples': cluster_sentences[:3]
            })
        
        # Sort by frequency
        themes.sort(key=lambda x: x['frequency'], reverse=True)
        return themes[:5]
    
    def _extract_theme_name(self, sentences: List[str]) -> str:
        """Extract a theme name from sentences"""
        combined = " ".join(sentences[:5])
        prompt = f"What is the main topic in one or two words: {combined[:200]}"
        
        try:
            inputs = self.t5_tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.t5_model.generate(
                    **inputs,
                    max_length=10,
                    temperature=0.7,
                    num_beams=2
                )
            
            theme = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
            return theme.title()
        except:
            # Fallback: extract most common noun phrase
            if self.nlp:
                doc = self.nlp(combined[:500])
                noun_phrases = [chunk.text for chunk in doc.noun_chunks]
                if noun_phrases:
                    return max(noun_phrases, key=len).title()
            return "General Feedback"
    
    def _find_unique_insights(self, reviews: List[str]) -> List[str]:
        """Find unique or surprising insights"""
        insights = []
        
        # Look for contrarian views
        sentiment_scores = []
        for review in reviews[:50]:
            try:
                blob = TextBlob(review)
                sentiment_scores.append(blob.sentiment.polarity)
            except:
                continue
        
        if sentiment_scores:
            mean_sentiment = np.mean(sentiment_scores)
            std_sentiment = np.std(sentiment_scores)
            
            # Find outlier reviews
            for i, review in enumerate(reviews[:50]):
                if i < len(sentiment_scores):
                    if abs(sentiment_scores[i] - mean_sentiment) > 2 * std_sentiment:
                        insight = self._extract_key_phrase(review)
                        if insight:
                            insights.append(f"Unique perspective: {insight}")
        
        # Look for specific patterns
        patterns = {
            r'compared to \w+': "Comparison insight",
            r'after \d+ (months?|weeks?|days?)': "Long-term usage insight",
            r'(tip|trick|hack)': "User tip",
            r'(warning|caution|careful)': "Important warning",
            r'(surprised|unexpected|didn\'t expect)': "Surprising discovery"
        }
        
        for review in reviews:
            for pattern, label in patterns.items():
                if re.search(pattern, review, re.IGNORECASE):
                    insight = self._extract_key_phrase(review)
                    if insight:
                        insights.append(f"{label}: {insight}")
                        break
        
        return list(set(insights))[:5]
    
    def _extract_technical_details(self, reviews: List[str]) -> Dict[str, Any]:
        """Extract technical specifications mentioned"""
        technical_details = defaultdict(list)
        
        # Patterns for technical specs
        patterns = {
            'battery': r'(\d+)\s*(mah|hours?|hrs?)',
            'storage': r'(\d+)\s*(gb|tb|gigabyte|terabyte)',
            'ram': r'(\d+)\s*(gb|gigabyte)\s*(ram|memory)',
            'camera': r'(\d+)\s*(mp|megapixel)',
            'screen': r'(\d+\.?\d*)\s*(inch|inches|")',
            'processor': r'(snapdragon|exynos|bionic|mediatek|helio)\s*\w*',
            'refresh_rate': r'(\d+)\s*(hz|hertz)',
            'charging': r'(\d+)\s*(w|watt|watts)'
        }
        
        for review in reviews:
            review_lower = review.lower()
            for spec, pattern in patterns.items():
                matches = re.findall(pattern, review_lower)
                if matches:
                    technical_details[spec].extend(matches)
        
        # Process and deduplicate
        processed_details = {}
        for spec, values in technical_details.items():
            if values:
                # Get most common value
                if isinstance(values[0], tuple):
                    values = [' '.join(v) for v in values]
                counter = Counter(values)
                most_common = counter.most_common(1)[0]
                processed_details[spec] = {
                    'value': most_common[0],
                    'mentions': most_common[1],
                    'variations': list(set(values))[:3]
                }
        
        return processed_details
    
    def _analyze_recommendations(self, reviews: List[str]) -> Dict[str, Any]:
        """Analyze user recommendations"""
        recommendations = {
            'would_recommend': 0,
            'would_not_recommend': 0,
            'conditional_recommend': 0,
            'target_users': [],
            'use_cases': []
        }
        
        recommend_patterns = [
            r'(recommend|recommended|recommending)',
            r'(worth it|worth buying|worth the price)',
            r'(must have|must buy)',
            r'(go for it|get it|buy it)'
        ]
        
        not_recommend_patterns = [
            r'(not recommend|don\'t recommend|cannot recommend)',
            r'(not worth|waste of money)',
            r'(avoid|stay away)',
            r'(regret|mistake)'
        ]
        
        conditional_patterns = [
            r'(if you|only if|provided that)',
            r'(depends on|depending on)',
            r'(for those who|for people who)'
        ]
        
        for review in reviews:
            review_lower = review.lower()
            
            if any(re.search(pattern, review_lower) for pattern in recommend_patterns):
                recommendations['would_recommend'] += 1
            elif any(re.search(pattern, review_lower) for pattern in not_recommend_patterns):
                recommendations['would_not_recommend'] += 1
            elif any(re.search(pattern, review_lower) for pattern in conditional_patterns):
                recommendations['conditional_recommend'] += 1
            
            # Extract target users
            user_match = re.search(r'(perfect for|great for|ideal for|good for)\s+([^.,]+)', review_lower)
            if user_match:
                recommendations['target_users'].append(user_match.group(2).strip())
            
            # Extract use cases
            use_match = re.search(r'(use it for|using for|used for)\s+([^.,]+)', review_lower)
            if use_match:
                recommendations['use_cases'].append(use_match.group(2).strip())
        
        # Calculate recommendation score
        total = (recommendations['would_recommend'] + 
                recommendations['would_not_recommend'] + 
                recommendations['conditional_recommend'])
        
        if total > 0:
            recommendations['recommendation_score'] = (
                (recommendations['would_recommend'] + 0.5 * recommendations['conditional_recommend']) / total
            ) * 100
        else:
            recommendations['recommendation_score'] = 50.0
        
        # Deduplicate lists
        recommendations['target_users'] = list(set(recommendations['target_users']))[:5]
        recommendations['use_cases'] = list(set(recommendations['use_cases']))[:5]
        
        return recommendations
    
    def _calculate_consensus(self, reviews: List[str]) -> float:
        """Calculate consensus score (0-100)"""
        if not reviews:
            return 0.0
        
        # Get sentiment scores
        sentiments = []
        for review in reviews[:50]:  # Sample for performance:
            try:
                blob = TextBlob(review)
                sentiments.append(blob.sentiment.polarity)
            except:
                continue
        
        if not sentiments:
            return 50.0
        
        # Calculate consensus based on standard deviation
        mean_sentiment = np.mean(sentiments)
        std_sentiment = np.std(sentiments)
        
        # Lower std means higher consensus
        # Normalize to 0-100 scale
        max_std = 1.0  # Maximum expected standard deviation
        consensus = max(0, (1 - (std_sentiment / max_std))) * 100
        
        # Adjust for overall sentiment
        if mean_sentiment > 0:
            consensus = min(100, consensus * 1.1)  # Boost positive consensus
        elif mean_sentiment < 0:
            consensus = max(0, consensus * 0.9)  # Reduce negative consensus
        
        return round(consensus, 1)
    
    def _generate_metadata(self, original_reviews: List[str], processed_reviews: List[str]) -> Dict[str, Any]:
        """Generate metadata about the summarization"""
        return {
            'total_reviews_analyzed': len(original_reviews),
            'total_words_processed': sum(len(r.split()) for r in original_reviews),
            'average_review_length': np.mean([len(r.split()) for r in original_reviews]),
            'processing_timestamp': datetime.now().isoformat(),
            'models_used': [
                'facebook/bart-large-cnn',
                'google/flan-t5-base',
                'all-MiniLM-L6-v2'
            ],
            'languages_detected': self._detect_languages(original_reviews[:10])
        }
    
    def _detect_languages(self, reviews: List[str]) -> List[str]:
        """Detect languages in reviews"""
        languages = set()
        for review in reviews:
            try:
                blob = TextBlob(review)
                lang = blob.detect_language()
                if lang:
                    languages.add(lang)
            except:
                languages.add('en')  # Default to English
        return list(languages)
    
    def _empty_summary(self) -> ReviewSummary:
        """Return empty summary when no reviews available"""
        return ReviewSummary(
            executive_summary="No reviews available for analysis.",
            tldr="No data available.",
            key_points=[],
            pros=[],
            cons=[],
            common_themes=[],
            unique_insights=[],
            technical_details={},
            user_recommendations={'recommendation_score': 0.0},
            consensus_score=0.0,
            summary_metadata={'error': 'No reviews provided'}
        )
    
    def generate_comparison_summary(
        self,
        product_reviews: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """Generate comparative summary for multiple products"""
        comparison = {
            'products': {},
            'winner': None,
            'comparison_table': [],
            'key_differences': [],
            'recommendation': ""
        }
        
        # Summarize each product
        for product_name, reviews in product_reviews.items():
            summary = self.summarize_reviews(reviews, product_name)
            comparison['products'][product_name] = {
                'summary': summary.tldr,
                'score': summary.consensus_score,
                'pros': summary.pros[:3],
                'cons': summary.cons[:3],
                'recommendation_score': summary.user_recommendations.get('recommendation_score', 0)
            }
        
        # Determine winner
        if comparison['products']:
            winner = max(
                comparison['products'].items(),
                key=lambda x: x[1]['recommendation_score']
            )
            comparison['winner'] = winner[0]
        
        # Create comparison table
        for product_name, data in comparison['products'].items():
            comparison['comparison_table'].append({
                'product': product_name,
                'consensus_score': data['score'],
                'recommendation_score': data['recommendation_score'],
                'top_pro': data['pros'][0] if data['pros'] else "N/A",
                'top_con': data['cons'][0] if data['cons'] else "N/A"
            })
        
        # Generate recommendation
        if comparison['winner']:
            comparison['recommendation'] = (
                f"Based on user reviews, {comparison['winner']} appears to be the best choice "
                f"with a recommendation score of {comparison['products'][comparison['winner']]['recommendation_score']:.1f}%"
            )
        
        return comparison


# Convenience function for easy integration
def create_smart_summary(reviews: List[str], product_name: str = "product") -> Dict[str, Any]:
    """
    Create a smart summary from reviews
    
    Args:
        reviews: List of review texts
        product_name: Name of the product
        
    Returns:
        Dictionary with summary components
    """
    summarizer = AdvancedReviewSummarizer()
    summary = summarizer.summarize_reviews(reviews, product_name)
    
    return {
        'executive_summary': summary.executive_summary,
        'tldr': summary.tldr,
        'key_points': summary.key_points,
        'pros': summary.pros,
        'cons': summary.cons,
        'themes': summary.common_themes,
        'unique_insights': summary.unique_insights,
        'technical_specs': summary.technical_details,
        'recommendation': summary.user_recommendations,
        'consensus_score': summary.consensus_score,
        'metadata': summary.summary_metadata
    }


# Example usage
if __name__ == "__main__":
    # Sample reviews for testing
    sample_reviews = [
        "This phone is amazing! The camera quality is outstanding and the battery lasts all day. "
        "The 120Hz display is smooth and responsive. Absolutely love it!",
        
        "Disappointing purchase. The phone overheats during gaming and the battery drains quickly. "
        "Camera is good but not worth the high price. Would not recommend.",
        
        "Great value for money! The performance is solid with the Snapdragon 888 processor. "
        "8GB RAM handles multitasking well. The 5000mAh battery easily lasts a full day.",
        
        "After 3 months of use, I've noticed some issues. The screen has minor burn-in and "
        "the charging port is loose. Customer service was unhelpful. Regret buying this.",
        
        "Perfect for photography enthusiasts! The 108MP main camera captures incredible detail. "
        "Night mode is impressive. Video stabilization could be better though.",
        
        "Compared to iPhone 13, this offers better value. Similar performance at a lower price. "
        "The 6.7 inch AMOLED display is gorgeous. Fast 65W charging is a game changer.",
        
        "Warning: This phone doesn't work well with certain carriers. Check compatibility first! "
        "Otherwise, it's a solid device with good build quality and features.",
        
        "Tip: Enable developer options and reduce animations for even snappier performance. "
        "This phone is already fast but this makes it feel even more responsive.",
        
        "Ideal for business users who need reliability and long battery life. "
        "The dual SIM support is convenient for travel. Security features are robust.",
        
        "Not recommended for gaming. Despite the specs, it struggles with demanding games. "
        "Casual use is fine but look elsewhere if gaming is a priority."
    ]
    
    # Create summary
    summarizer = AdvancedReviewSummarizer()
    summary = summarizer.summarize_reviews(sample_reviews, "Samsung Galaxy S24")
    
    # Display results
    print("=" * 80)
    print("REVIEW SUMMARY ANALYSIS")
    print("=" * 80)
    print(f"\n📝 EXECUTIVE SUMMARY:\n{summary.executive_summary}")
    print(f"\n⚡ TL;DR:\n{summary.tldr}")
    print(f"\n🎯 KEY POINTS:")
    for i, point in enumerate(summary.key_points, 1):
        print(f"  {i}. {point}")
    print(f"\n✅ PROS:")
    for i, pro in enumerate(summary.pros, 1):
        print(f"  {i}. {pro}")
    print(f"\n❌ CONS:")
    for i, con in enumerate(summary.cons, 1):
        print(f"  {i}. {con}")
    print(f"\n📊 COMMON THEMES:")
    for theme in summary.common_themes:
        print(f"  • {theme['name']} (mentioned {theme['frequency']} times)")
    print(f"\n💡 UNIQUE INSIGHTS:")
    for insight in summary.unique_insights:
        print(f"  • {insight}")
    print(f"\n🔧 TECHNICAL DETAILS:")
    for spec, details in summary.technical_details.items():
        print(f"  • {spec}: {details['value']} (mentioned {details['mentions']} times)")
    print(f"\n👥 RECOMMENDATION SCORE: {summary.user_recommendations.get('recommendation_score', 0):.1f}%")
    print(f"\n🤝 CONSENSUS SCORE: {summary.consensus_score}%")
    print(f"\n📈 METADATA:")
    print(f"  • Reviews analyzed: {summary.summary_metadata.get('total_reviews_analyzed', 0)}")
    print(f"  • Words processed: {summary.summary_metadata.get('total_words_processed', 0)}")
    print("=" * 80)
