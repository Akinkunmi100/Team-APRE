"""
AI-Powered Web Browsing System for AI Phone Review Engine
Implements intelligent web browsing with AI-driven content extraction and understanding
"""

import asyncio
import aiohttp
import logging
import json
import re
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse, quote_plus
from bs4 import BeautifulSoup
import hashlib
from pathlib import Path
import openai
import anthropic
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import requests
from readability import Document

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class WebPageAnalysis:
    """Structure for AI web page analysis results"""
    url: str
    title: str
    content_type: str
    phone_relevance_score: float
    content_quality_score: float
    information_density: float
    extracted_phones: List[Dict[str, Any]]
    key_information: Dict[str, Any]
    sentiment_analysis: Dict[str, float]
    summary: str
    extraction_confidence: float
    language: str
    page_structure: Dict[str, Any]
    analyzed_at: str

@dataclass
class PhoneInformation:
    """Structured phone information extracted by AI"""
    name: str
    brand: str
    model: str
    specifications: Dict[str, Any]
    reviews: List[Dict[str, Any]]
    ratings: List[Dict[str, float]]
    prices: List[Dict[str, Any]]
    pros_cons: Dict[str, List[str]]
    technical_details: Dict[str, Any]
    availability: Dict[str, Any]
    confidence_score: float
    extraction_source: str

class AIWebBrowser:
    """AI-powered web browsing system for intelligent content extraction"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize AI web browser"""
        
        self.config = config or {
            'ai_provider': 'openai',  # 'openai', 'anthropic', 'huggingface'
            'model_name': 'gpt-3.5-turbo',
            'max_tokens': 4000,
            'temperature': 0.1,
            'max_concurrent_requests': 3,
            'request_timeout': 60,
            'enable_content_cleaning': True,
            'enable_sentiment_analysis': True,
            'enable_summarization': True,
            'min_content_length': 500,
            'max_content_length': 50000,
            'cache_analyses': True,
            'cache_duration_hours': 24
        }
        
        # Initialize AI models
        self.ai_client = None
        self.sentence_transformer = None
        self.sentiment_analyzer = None
        self.summarizer = None
        
        # Content extraction patterns
        self.phone_patterns = {
            'phone_names': [
                r'(iPhone\s+\d+(?:\s+(?:Pro|Plus|Max|Mini))*)',
                r'(Galaxy\s+S\d+(?:\s+(?:Plus|Ultra|FE))*)',
                r'(Pixel\s+\d+(?:\s+(?:Pro|XL|a))*)',
                r'(OnePlus\s+\d+(?:\s+(?:Pro|T|RT))*)',
                r'(Xiaomi\s+\d+(?:\s+(?:Pro|Ultra|Lite))*)',
                r'(Huawei\s+P\d+(?:\s+(?:Pro|Lite))*)',
                r'(Samsung\s+Galaxy\s+[A-Z]\d+)',
                r'(iPhone\s+SE\s+\d+)',
                r'(Nothing\s+Phone\s+\d+)',
                r'(Realme\s+\d+(?:\s+Pro)*)'
            ],
            'specifications': {
                'display': r'(\d+\.?\d*)\s*(?:inch|"|\u201d)(?:\s+(?:OLED|LCD|AMOLED))?',
                'battery': r'(\d+)\s*mAh',
                'camera': r'(\d+)\s*MP(?:\s+(?:camera|sensor))?',
                'storage': r'(\d+)\s*GB(?:\s+(?:storage|ROM))?',
                'ram': r'(\d+)\s*GB(?:\s+(?:RAM|memory))?',
                'processor': r'(Snapdragon\s+\d+|A\d+\s+Bionic|Exynos\s+\d+|Tensor\s+G\d+)',
                'price': r'\$(\d+(?:,\d{3})*(?:\.\d{2})?)'
            },
            'ratings': [
                r'(\d+\.?\d*)\s*/\s*5(?:\s+stars?)?',
                r'(\d+\.?\d*)\s*/\s*10',
                r'(\d+)\s*%(?:\s+recommended)?',
                r'Rating:\s*(\d+\.?\d*)'
            ]
        }
        
        # Initialize components
        self._init_ai_components()
        
        # Session for web requests
        self.session = None
        
        # Cache for analyses
        self.analysis_cache = {}
    
    def _init_ai_components(self):
        """Initialize AI components based on configuration"""
        
        try:
            # Initialize AI client
            if self.config['ai_provider'] == 'openai':
                import openai
                self.ai_client = openai.AsyncOpenAI()
            elif self.config['ai_provider'] == 'anthropic':
                import anthropic
                self.ai_client = anthropic.AsyncAnthropic()
            
            # Initialize sentence transformer for embeddings
            if self.config.get('enable_semantic_analysis', True):
                self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize sentiment analyzer
            if self.config['enable_sentiment_analysis']:
                nltk.download('vader_lexicon', quiet=True)
                self.sentiment_analyzer = SentimentIntensityAnalyzer()
            
            # Initialize summarizer
            if self.config['enable_summarization']:
                try:
                    self.summarizer = pipeline(
                        'summarization',
                        model='facebook/bart-large-cnn',
                        device=0 if torch.cuda.is_available() else -1
                    )
                except:
                    logger.warning("Could not initialize BART summarizer, falling back to extractive summarization")
                    self.summarizer = None
            
            logger.info(f"Initialized AI components with provider: {self.config['ai_provider']}")
            
        except Exception as e:
            logger.error(f"Failed to initialize AI components: {e}")
    
    async def analyze_webpage(self, url: str, focus_query: str = None) -> WebPageAnalysis:
        """
        Analyze a webpage using AI to extract phone-related information
        
        Args:
            url: URL to analyze
            focus_query: Specific query to focus the analysis on
            
        Returns:
            WebPageAnalysis object with extracted information
        """
        
        logger.info(f"Starting AI analysis of webpage: {url}")
        
        # Check cache first
        cache_key = hashlib.md5(f"{url}_{focus_query}".encode()).hexdigest()
        if self.config['cache_analyses'] and cache_key in self.analysis_cache:
            cached_analysis = self.analysis_cache[cache_key]
            cache_age = datetime.now() - datetime.fromisoformat(cached_analysis.analyzed_at)
            if cache_age.total_seconds() < self.config['cache_duration_hours'] * 3600:
                logger.info("Returning cached analysis")
                return cached_analysis
        
        # Initialize session if needed
        await self._init_session()
        
        try:
            # Step 1: Fetch and clean webpage content
            raw_content = await self._fetch_webpage(url)
            cleaned_content = self._clean_content(raw_content)
            
            if len(cleaned_content) < self.config['min_content_length']:
                raise ValueError(f"Content too short: {len(cleaned_content)} characters")
            
            # Step 2: Extract basic information
            basic_info = self._extract_basic_info(cleaned_content, url)
            
            # Step 3: AI-powered content analysis
            ai_analysis = await self._ai_analyze_content(cleaned_content, focus_query, url)
            
            # Step 4: Extract structured phone information
            phone_info = await self._extract_phone_information(cleaned_content, ai_analysis)
            
            # Step 5: Sentiment analysis
            sentiment = self._analyze_sentiment(cleaned_content) if self.config['enable_sentiment_analysis'] else {}
            
            # Step 6: Generate summary
            summary = await self._generate_summary(cleaned_content, ai_analysis, focus_query)
            
            # Step 7: Calculate scores
            relevance_score = self._calculate_phone_relevance_score(cleaned_content, phone_info)
            quality_score = self._calculate_content_quality_score(cleaned_content, basic_info)
            density_score = self._calculate_information_density(cleaned_content, phone_info)
            confidence_score = ai_analysis.get('confidence', 0.7)
            
            # Create analysis result
            analysis = WebPageAnalysis(
                url=url,
                title=basic_info.get('title', 'Unknown'),
                content_type=basic_info.get('content_type', 'webpage'),
                phone_relevance_score=relevance_score,
                content_quality_score=quality_score,
                information_density=density_score,
                extracted_phones=phone_info,
                key_information=ai_analysis.get('key_information', {}),
                sentiment_analysis=sentiment,
                summary=summary,
                extraction_confidence=confidence_score,
                language=basic_info.get('language', 'en'),
                page_structure=basic_info.get('structure', {}),
                analyzed_at=datetime.now().isoformat()
            )
            
            # Cache the analysis
            if self.config['cache_analyses']:
                self.analysis_cache[cache_key] = analysis
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze webpage {url}: {e}")
            # Return minimal analysis
            return WebPageAnalysis(
                url=url,
                title="Analysis Failed",
                content_type="error",
                phone_relevance_score=0.0,
                content_quality_score=0.0,
                information_density=0.0,
                extracted_phones=[],
                key_information={'error': str(e)},
                sentiment_analysis={},
                summary="Analysis failed",
                extraction_confidence=0.0,
                language="en",
                page_structure={},
                analyzed_at=datetime.now().isoformat()
            )
    
    async def search_and_analyze(self, query: str, max_pages: int = 10) -> List[WebPageAnalysis]:
        """
        Search for phone information and analyze multiple webpages
        
        Args:
            query: Search query (e.g., "iPhone 15 Pro review")
            max_pages: Maximum number of pages to analyze
            
        Returns:
            List of WebPageAnalysis objects
        """
        
        logger.info(f"Starting search and analysis for query: {query}")
        
        # Use multiple search engines for broader coverage
        search_urls = await self._get_search_urls(query, max_pages)
        
        # Analyze pages concurrently
        semaphore = asyncio.Semaphore(self.config['max_concurrent_requests'])
        
        async def analyze_single_page(url):
            async with semaphore:
                try:
                    return await self.analyze_webpage(url, focus_query=query)
                except Exception as e:
                    logger.error(f"Failed to analyze {url}: {e}")
                    return None
        
        tasks = [analyze_single_page(url) for url in search_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful analyses
        analyses = [
            result for result in results 
            if result and not isinstance(result, Exception) and result.phone_relevance_score > 0.3
        ]
        
        # Sort by relevance and quality
        analyses.sort(key=lambda x: (x.phone_relevance_score + x.content_quality_score) / 2, reverse=True)
        
        logger.info(f"Completed analysis of {len(analyses)} pages for query: {query}")
        return analyses
    
    async def extract_phone_from_url(self, url: str, phone_name: str = None) -> Optional[PhoneInformation]:
        """
        Extract comprehensive phone information from a specific URL
        
        Args:
            url: URL to extract from
            phone_name: Specific phone to focus on (optional)
            
        Returns:
            PhoneInformation object or None
        """
        
        try:
            # Analyze the webpage
            analysis = await self.analyze_webpage(url, focus_query=phone_name)
            
            if not analysis.extracted_phones:
                return None
            
            # Find the most relevant phone
            target_phone = None
            if phone_name:
                # Look for specific phone
                for phone in analysis.extracted_phones:
                    if phone_name.lower() in phone.get('name', '').lower():
                        target_phone = phone
                        break
            
            if not target_phone and analysis.extracted_phones:
                # Use the most comprehensive phone data
                target_phone = max(analysis.extracted_phones, 
                                 key=lambda p: len(str(p.get('specifications', {}))))
            
            if not target_phone:
                return None
            
            # Create structured phone information
            phone_info = PhoneInformation(
                name=target_phone.get('name', 'Unknown'),
                brand=target_phone.get('brand', 'Unknown'),
                model=target_phone.get('model', 'Unknown'),
                specifications=target_phone.get('specifications', {}),
                reviews=target_phone.get('reviews', []),
                ratings=target_phone.get('ratings', []),
                prices=target_phone.get('prices', []),
                pros_cons=target_phone.get('pros_cons', {'pros': [], 'cons': []}),
                technical_details=target_phone.get('technical_details', {}),
                availability=target_phone.get('availability', {}),
                confidence_score=analysis.extraction_confidence,
                extraction_source=url
            )
            
            return phone_info
            
        except Exception as e:
            logger.error(f"Failed to extract phone information from {url}: {e}")
            return None
    
    async def _fetch_webpage(self, url: str) -> str:
        """Fetch webpage content"""
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        async with self.session.get(url, headers=headers, timeout=self.config['request_timeout']) as response:
            if response.status != 200:
                raise Exception(f"HTTP {response.status}")
            
            content = await response.text()
            return content
    
    def _clean_content(self, html_content: str) -> str:
        """Clean and extract relevant content from HTML"""
        
        if not self.config['enable_content_cleaning']:
            return html_content
        
        try:
            # Use readability library to extract main content
            doc = Document(html_content)
            main_content = doc.summary()
            
            # Parse with BeautifulSoup for further cleaning
            soup = BeautifulSoup(main_content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'advertisement']):
                element.decompose()
            
            # Extract text
            text = soup.get_text(separator=' ', strip=True)
            
            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Limit content length
            if len(text) > self.config['max_content_length']:
                text = text[:self.config['max_content_length']] + "..."
            
            return text
            
        except Exception as e:
            logger.warning(f"Content cleaning failed, using raw HTML: {e}")
            # Fallback to basic HTML parsing
            soup = BeautifulSoup(html_content, 'html.parser')
            return soup.get_text(separator=' ', strip=True)[:self.config['max_content_length']]
    
    def _extract_basic_info(self, content: str, url: str) -> Dict[str, Any]:
        """Extract basic information from content"""
        
        info = {
            'url': url,
            'content_length': len(content),
            'language': 'en',  # Could be enhanced with language detection
            'structure': {}
        }
        
        # Try to extract title
        title_patterns = [
            r'<title[^>]*>([^<]+)</title>',
            r'<h1[^>]*>([^<]+)</h1>',
            r'^([^\n\r]{10,100})'
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
            if match:
                info['title'] = match.group(1).strip()
                break
        
        # Determine content type
        if any(word in content.lower() for word in ['review', 'test', 'hands-on']):
            info['content_type'] = 'review'
        elif any(word in content.lower() for word in ['news', 'announcement', 'release']):
            info['content_type'] = 'news'
        elif any(word in content.lower() for word in ['specs', 'specifications', 'technical']):
            info['content_type'] = 'specifications'
        elif any(word in content.lower() for word in ['price', 'buy', 'purchase', 'cost']):
            info['content_type'] = 'shopping'
        else:
            info['content_type'] = 'general'
        
        # Basic structure analysis
        info['structure'] = {
            'paragraphs': content.count('\n\n'),
            'sentences': content.count('.') + content.count('!') + content.count('?'),
            'words': len(content.split())
        }
        
        return info
    
    async def _ai_analyze_content(self, content: str, focus_query: str, url: str) -> Dict[str, Any]:
        """Use AI to analyze content and extract key information"""
        
        try:
            # Prepare AI prompt
            prompt = self._create_analysis_prompt(content, focus_query, url)
            
            # Call AI service
            if self.config['ai_provider'] == 'openai':
                response = await self._call_openai(prompt)
            elif self.config['ai_provider'] == 'anthropic':
                response = await self._call_anthropic(prompt)
            else:
                response = await self._call_huggingface(prompt)
            
            # Parse AI response
            analysis = self._parse_ai_response(response)
            return analysis
            
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return {
                'key_information': {},
                'confidence': 0.1,
                'phone_mentions': [],
                'technical_details': {},
                'summary': 'AI analysis failed'
            }
    
    def _create_analysis_prompt(self, content: str, focus_query: str, url: str) -> str:
        """Create prompt for AI analysis"""
        
        base_prompt = f"""
Analyze this webpage content for phone/smartphone information. Extract structured data about phones mentioned.

URL: {url}
Focus Query: {focus_query or 'General phone information'}

Content:
{content[:3000]}...

Please extract and provide:
1. Phone models mentioned (name, brand, model)
2. Technical specifications (display, battery, camera, processor, storage, RAM, etc.)
3. Reviews and ratings
4. Prices and availability
5. Pros and cons
6. Key features and highlights
7. Overall assessment of content quality and relevance

Respond in JSON format with the following structure:
{{
    "phones_mentioned": [
        {{
            "name": "Phone name",
            "brand": "Brand",
            "model": "Model",
            "specifications": {{}},
            "ratings": [],
            "prices": [],
            "pros_cons": {{"pros": [], "cons": []}},
            "key_features": []
        }}
    ],
    "content_assessment": {{
        "phone_relevance": 0.0-1.0,
        "information_quality": 0.0-1.0,
        "technical_depth": 0.0-1.0
    }},
    "key_information": {{}},
    "confidence": 0.0-1.0,
    "summary": "Brief summary of content"
}}
"""
        
        return base_prompt
    
    async def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API"""
        
        try:
            response = await self.ai_client.chat.completions.create(
                model=self.config['model_name'],
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing web content for phone and smartphone information. Provide accurate, structured responses in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config['max_tokens'],
                temperature=self.config['temperature']
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise
    
    async def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic Claude API"""
        
        try:
            response = await self.ai_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=self.config['max_tokens'],
                temperature=self.config['temperature'],
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Anthropic API call failed: {e}")
            raise
    
    async def _call_huggingface(self, prompt: str) -> str:
        """Call Hugging Face models (local or API)"""
        
        try:
            # This would use local Hugging Face models or their API
            # For now, return a basic analysis
            return json.dumps({
                "phones_mentioned": [],
                "content_assessment": {
                    "phone_relevance": 0.5,
                    "information_quality": 0.5,
                    "technical_depth": 0.5
                },
                "key_information": {"note": "Hugging Face analysis not fully implemented"},
                "confidence": 0.3,
                "summary": "Basic content analysis"
            })
            
        except Exception as e:
            logger.error(f"Hugging Face analysis failed: {e}")
            raise
    
    def _parse_ai_response(self, response: str) -> Dict[str, Any]:
        """Parse AI response into structured data"""
        
        try:
            # Try to parse JSON response
            if response.startswith('{'):
                data = json.loads(response)
            else:
                # Extract JSON from response
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    raise ValueError("No JSON found in response")
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to parse AI response: {e}")
            # Return minimal structure
            return {
                'phones_mentioned': [],
                'content_assessment': {'phone_relevance': 0.1, 'information_quality': 0.1, 'technical_depth': 0.1},
                'key_information': {'raw_response': response[:500]},
                'confidence': 0.1,
                'summary': 'Failed to parse AI response'
            }
    
    async def _extract_phone_information(self, content: str, ai_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract and combine phone information from content and AI analysis"""
        
        phones = []
        
        # Start with AI-identified phones
        ai_phones = ai_analysis.get('phones_mentioned', [])
        
        # Enhance with pattern-based extraction
        for ai_phone in ai_phones:
            enhanced_phone = ai_phone.copy()
            
            # Extract additional specifications using patterns
            for spec_name, pattern in self.phone_patterns['specifications'].items():
                if spec_name not in enhanced_phone.get('specifications', {}):
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        enhanced_phone.setdefault('specifications', {})[spec_name] = matches[0]
            
            # Extract ratings
            if not enhanced_phone.get('ratings'):
                ratings = []
                for pattern in self.phone_patterns['ratings']:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    ratings.extend([{'rating': float(match), 'scale': 'detected'} for match in matches])
                enhanced_phone['ratings'] = ratings[:3]  # Limit to top 3
            
            phones.append(enhanced_phone)
        
        # If no AI phones found, use pattern-based extraction
        if not phones:
            for pattern in self.phone_patterns['phone_names']:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches[:5]:  # Limit to 5 phones max
                    phone = {
                        'name': match,
                        'brand': self._extract_brand(match),
                        'model': match,
                        'specifications': {},
                        'ratings': [],
                        'prices': [],
                        'pros_cons': {'pros': [], 'cons': []},
                        'source': 'pattern_extraction'
                    }
                    
                    # Extract specs for this phone
                    for spec_name, pattern in self.phone_patterns['specifications'].items():
                        matches = re.findall(pattern, content, re.IGNORECASE)
                        if matches:
                            phone['specifications'][spec_name] = matches[0]
                    
                    phones.append(phone)
        
        return phones
    
    def _extract_brand(self, phone_name: str) -> str:
        """Extract brand from phone name"""
        
        brand_mapping = {
            'iPhone': 'Apple',
            'Galaxy': 'Samsung',
            'Pixel': 'Google',
            'OnePlus': 'OnePlus',
            'Xiaomi': 'Xiaomi',
            'Huawei': 'Huawei',
            'Nothing': 'Nothing',
            'Realme': 'Realme'
        }
        
        for key, brand in brand_mapping.items():
            if key.lower() in phone_name.lower():
                return brand
        
        return 'Unknown'
    
    def _analyze_sentiment(self, content: str) -> Dict[str, float]:
        """Analyze sentiment of content"""
        
        if not self.sentiment_analyzer:
            return {}
        
        try:
            # Analyze overall sentiment
            scores = self.sentiment_analyzer.polarity_scores(content)
            
            # Focus on review-specific content
            review_sentences = [
                sentence for sentence in content.split('.')
                if any(word in sentence.lower() for word in ['good', 'bad', 'great', 'terrible', 'love', 'hate', 'recommend'])
            ]
            
            review_sentiment = {'compound': 0.0}
            if review_sentences:
                review_text = '. '.join(review_sentences[:10])  # Limit to 10 sentences
                review_sentiment = self.sentiment_analyzer.polarity_scores(review_text)
            
            return {
                'overall': scores,
                'review_focused': review_sentiment,
                'positive_indicators': scores['pos'],
                'negative_indicators': scores['neg'],
                'neutral_score': scores['neu']
            }
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {}
    
    async def _generate_summary(self, content: str, ai_analysis: Dict[str, Any], focus_query: str) -> str:
        """Generate content summary"""
        
        # First try AI-generated summary
        ai_summary = ai_analysis.get('summary', '')
        if ai_summary and len(ai_summary) > 50:
            return ai_summary
        
        # Try BART summarizer if available
        if self.summarizer:
            try:
                # Limit input length for BART
                input_text = content[:1024]
                summary_result = self.summarizer(input_text, max_length=150, min_length=50, do_sample=False)
                return summary_result[0]['summary_text']
            except Exception as e:
                logger.warning(f"BART summarization failed: {e}")
        
        # Fallback to extractive summarization
        sentences = content.split('. ')
        
        # Score sentences based on phone-related keywords
        scored_sentences = []
        phone_keywords = ['phone', 'smartphone', 'mobile', 'device', 'review', 'test', 'performance', 'camera', 'battery']
        
        for sentence in sentences[:50]:  # Limit to first 50 sentences
            score = sum(1 for keyword in phone_keywords if keyword in sentence.lower())
            if focus_query:
                score += sentence.lower().count(focus_query.lower()) * 2
            
            if score > 0:
                scored_sentences.append((score, sentence))
        
        # Get top sentences
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        top_sentences = [sentence for _, sentence in scored_sentences[:3]]
        
        return '. '.join(top_sentences) if top_sentences else "No clear summary available."
    
    def _calculate_phone_relevance_score(self, content: str, phone_info: List[Dict[str, Any]]) -> float:
        """Calculate how relevant the content is to phone information"""
        
        score = 0.0
        content_lower = content.lower()
        
        # Phone mention frequency
        phone_keywords = ['phone', 'smartphone', 'mobile', 'device', 'handset']
        phone_mentions = sum(content_lower.count(keyword) for keyword in phone_keywords)
        score += min(phone_mentions * 0.05, 0.3)  # Max 0.3 from mentions
        
        # Specific phone models mentioned
        if phone_info:
            score += min(len(phone_info) * 0.1, 0.3)  # Max 0.3 from phone count
        
        # Technical specifications presence
        spec_keywords = ['display', 'battery', 'camera', 'processor', 'storage', 'ram', 'specs']
        spec_mentions = sum(content_lower.count(keyword) for keyword in spec_keywords)
        score += min(spec_mentions * 0.03, 0.2)  # Max 0.2 from specs
        
        # Review/evaluation content
        review_keywords = ['review', 'test', 'performance', 'rating', 'recommend', 'pros', 'cons']
        review_mentions = sum(content_lower.count(keyword) for keyword in review_keywords)
        score += min(review_mentions * 0.02, 0.2)  # Max 0.2 from review content
        
        return min(score, 1.0)
    
    def _calculate_content_quality_score(self, content: str, basic_info: Dict[str, Any]) -> float:
        """Calculate content quality score"""
        
        score = 0.5  # Base score
        
        # Content length (comprehensive content is usually longer)
        length = len(content)
        if length > 5000:
            score += 0.2
        elif length > 2000:
            score += 0.1
        elif length < 500:
            score -= 0.2
        
        # Structure indicators
        structure = basic_info.get('structure', {})
        if structure.get('paragraphs', 0) > 5:
            score += 0.1
        if structure.get('sentences', 0) > 20:
            score += 0.1
        
        # Quality indicators
        quality_indicators = [
            'detailed', 'comprehensive', 'thorough', 'analysis', 'comparison',
            'technical', 'specifications', 'benchmark', 'test results'
        ]
        content_lower = content.lower()
        quality_mentions = sum(content_lower.count(indicator) for indicator in quality_indicators)
        score += min(quality_mentions * 0.02, 0.2)
        
        # Reduce for poor quality indicators
        poor_quality = ['click', 'advertisement', 'sponsored', 'buy now', 'limited time']
        poor_mentions = sum(content_lower.count(indicator) for indicator in poor_quality)
        score -= min(poor_mentions * 0.05, 0.3)
        
        return max(min(score, 1.0), 0.0)
    
    def _calculate_information_density(self, content: str, phone_info: List[Dict[str, Any]]) -> float:
        """Calculate information density score"""
        
        if not content:
            return 0.0
        
        total_info_items = 0
        
        # Count information items from extracted phones
        for phone in phone_info:
            total_info_items += len(phone.get('specifications', {}))
            total_info_items += len(phone.get('ratings', []))
            total_info_items += len(phone.get('prices', []))
            total_info_items += len(phone.get('pros_cons', {}).get('pros', []))
            total_info_items += len(phone.get('pros_cons', {}).get('cons', []))
        
        # Calculate density (info items per 1000 characters)
        density = (total_info_items * 1000) / len(content)
        
        # Normalize to 0-1 scale
        return min(density / 10, 1.0)  # Max density of 10 items per 1000 chars = 1.0
    
    async def _get_search_urls(self, query: str, max_results: int) -> List[str]:
        """Get search URLs for the query"""
        
        # This is a simplified implementation
        # In practice, you'd use Google Custom Search API or other search APIs
        search_engines = [
            f"https://duckduckgo.com/html/?q={quote_plus(query)}",
            f"https://www.bing.com/search?q={quote_plus(query)}",
        ]
        
        urls = []
        
        for search_url in search_engines:
            try:
                async with self.session.get(search_url, timeout=30) as response:
                    if response.status == 200:
                        content = await response.text()
                        soup = BeautifulSoup(content, 'html.parser')
                        
                        # Extract result URLs (this is search engine specific)
                        links = soup.find_all('a', href=True)
                        for link in links[:max_results//2]:
                            href = link.get('href')
                            if href and href.startswith('http') and 'duckduckgo' not in href:
                                urls.append(href)
                        
                await asyncio.sleep(2)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Search failed for {search_url}: {e}")
        
        return urls[:max_results]
    
    # Utility methods
    async def _init_session(self):
        """Initialize aiohttp session"""
        if not self.session:
            connector = aiohttp.TCPConnector(limit=20)
            timeout = aiohttp.ClientTimeout(total=self.config['request_timeout'])
            self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
            self.session = None

# Factory function
def create_ai_web_browser(config=None):
    """Create configured AI web browser"""
    return AIWebBrowser(config=config)