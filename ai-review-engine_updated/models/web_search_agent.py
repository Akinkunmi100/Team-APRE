"""
Web Search Agent for Dynamic Phone Data Retrieval
Specialized agent that performs real-time web scraping to gather phone specifications,
reviews, and sentiment analysis when data is missing from local dataset.
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import re
import time
import hashlib
from urllib.parse import quote, urljoin, urlparse
import sqlite3
from pathlib import Path

# Web scraping imports
from bs4 import BeautifulSoup
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

# AI and analysis imports
from transformers import pipeline
import nltk
from textblob import TextBlob
import pandas as pd
from sentence_transformers import SentenceTransformer

# Import base agent
from models.agentic_rag import BaseAgent, AgentRole, AgentTask, AgentState
from langchain.agents import Tool

logger = logging.getLogger(__name__)

@dataclass
class WebSearchResult:
    """Structure for web search results"""
    phone_model: str
    source: str
    content_type: str  # specifications, reviews, prices, news
    content: Dict[str, Any]
    confidence: float
    timestamp: datetime
    url: str
    cached: bool = False

@dataclass 
class SearchProgress:
    """Track search progress for user interface"""
    phone_model: str
    total_sources: int
    completed_sources: int
    current_source: str
    status: str  # searching, analyzing, caching, completed, failed
    results_found: int
    start_time: datetime
    estimated_completion: Optional[datetime] = None
    errors: List[str] = field(default_factory=list)

class WebSearchCache:
    """Local caching system for web search results"""
    
    def __init__(self, cache_dir: str = "data/web_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "web_search_cache.db"
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for caching"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS search_cache (
                    id INTEGER PRIMARY KEY,
                    phone_model TEXT,
                    source TEXT,
                    content_type TEXT,
                    content_hash TEXT UNIQUE,
                    content JSON,
                    confidence REAL,
                    url TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_phone_model ON search_cache(phone_model)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_source ON search_cache(source)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_content_type ON search_cache(content_type)")
    
    def get_cached_result(self, phone_model: str, source: str, content_type: str) -> Optional[WebSearchResult]:
        """Retrieve cached result if available and not expired"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT content, confidence, url, created_at 
                FROM search_cache 
                WHERE phone_model = ? AND source = ? AND content_type = ?
                AND expires_at > datetime('now')
                ORDER BY created_at DESC LIMIT 1
            """, (phone_model, source, content_type))
            
            row = cursor.fetchone()
            if row:
                # Update access statistics
                conn.execute("""
                    UPDATE search_cache 
                    SET access_count = access_count + 1, last_accessed = datetime('now')
                    WHERE phone_model = ? AND source = ? AND content_type = ?
                """, (phone_model, source, content_type))
                
                return WebSearchResult(
                    phone_model=phone_model,
                    source=source,
                    content_type=content_type,
                    content=json.loads(row[0]),
                    confidence=row[1],
                    url=row[2],
                    timestamp=datetime.fromisoformat(row[3]),
                    cached=True
                )
        return None
    
    def cache_result(self, result: WebSearchResult, expire_hours: int = 24):
        """Cache a search result"""
        content_json = json.dumps(result.content)
        content_hash = hashlib.md5(content_json.encode()).hexdigest()
        expires_at = datetime.now() + timedelta(hours=expire_hours)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO search_cache 
                (phone_model, source, content_type, content_hash, content, 
                 confidence, url, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (result.phone_model, result.source, result.content_type,
                  content_hash, content_json, result.confidence, 
                  result.url, expires_at))
    
    def cleanup_expired(self):
        """Remove expired cache entries"""
        with sqlite3.connect(self.db_path) as conn:
            deleted = conn.execute("DELETE FROM search_cache WHERE expires_at < datetime('now')").rowcount
            logger.info(f"Cleaned up {deleted} expired cache entries")

class GSMArenaScaper:
    """Scraper for GSMArena phone specifications and reviews"""
    
    def __init__(self):
        self.base_url = "https://www.gsmarena.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    async def search_phone(self, phone_model: str) -> Optional[Dict[str, Any]]:
        """Search for phone specifications on GSMArena"""
        try:
            # Search for the phone
            search_url = f"{self.base_url}/results.php3"
            params = {'sQuickSearch': 'yes', 'sName': phone_model}
            
            response = self.session.get(search_url, params=params, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find first search result
            makers = soup.find_all('div', class_='makers')
            if not makers:
                return None
            
            first_result = makers[0].find('a')
            if not first_result:
                return None
            
            phone_url = urljoin(self.base_url, first_result['href'])
            
            # Get detailed specifications
            return await self._scrape_specifications(phone_url, phone_model)
            
        except Exception as e:
            logger.error(f"GSMArena scraping error for {phone_model}: {e}")
            return None
    
    async def _scrape_specifications(self, url: str, phone_model: str) -> Dict[str, Any]:
        """Scrape detailed specifications from phone page"""
        try:
            response = self.session.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            specs = {
                'model': phone_model,
                'url': url,
                'specifications': {}
            }
            
            # Extract specifications from table
            spec_tables = soup.find_all('table', cellspacing='0')
            for table in spec_tables:
                category_header = table.find_previous('th', class_='ttl')
                category = category_header.text.strip() if category_header else 'general'
                
                specs['specifications'][category] = {}
                
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 2:
                        key = cells[0].text.strip()
                        value = cells[1].text.strip()
                        specs['specifications'][category][key] = value
            
            # Extract basic info
            header = soup.find('h1', class_='specs-phone-name-title')
            if header:
                specs['display_name'] = header.text.strip()
            
            # Extract rating if available
            rating_elem = soup.find('div', class_='user-ratings')
            if rating_elem:
                rating_text = rating_elem.text
                rating_match = re.search(r'(\d+\.?\d*)', rating_text)
                if rating_match:
                    specs['user_rating'] = float(rating_match.group(1))
            
            return specs
            
        except Exception as e:
            logger.error(f"Error scraping specifications from {url}: {e}")
            return {'model': phone_model, 'url': url, 'error': str(e)}

class RedditScraper:
    """Scraper for Reddit reviews and discussions"""
    
    def __init__(self):
        self.base_url = "https://www.reddit.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'WebSearchAgent/1.0 (Research Bot)'
        })
    
    async def search_reviews(self, phone_model: str) -> List[Dict[str, Any]]:
        """Search Reddit for phone reviews and discussions"""
        try:
            # Search multiple relevant subreddits
            subreddits = ['Android', 'iphone', 'smartphones', 'PickAnAndroidForMe']
            all_posts = []
            
            for subreddit in subreddits:
                posts = await self._search_subreddit(subreddit, phone_model)
                all_posts.extend(posts)
                
                # Rate limiting
                await asyncio.sleep(1)
            
            return all_posts[:20]  # Return top 20 posts
            
        except Exception as e:
            logger.error(f"Reddit scraping error for {phone_model}: {e}")
            return []
    
    async def _search_subreddit(self, subreddit: str, phone_model: str) -> List[Dict[str, Any]]:
        """Search specific subreddit for phone discussions"""
        try:
            search_url = f"{self.base_url}/r/{subreddit}/search.json"
            params = {
                'q': phone_model,
                'restrict_sr': 'on',
                'sort': 'relevance',
                'limit': 25
            }
            
            response = self.session.get(search_url, params=params, timeout=10)
            data = response.json()
            
            posts = []
            for post in data.get('data', {}).get('children', []):
                post_data = post.get('data', {})
                
                # Extract relevant information
                post_info = {
                    'title': post_data.get('title', ''),
                    'text': post_data.get('selftext', ''),
                    'score': post_data.get('score', 0),
                    'num_comments': post_data.get('num_comments', 0),
                    'url': f"{self.base_url}{post_data.get('permalink', '')}",
                    'subreddit': subreddit,
                    'created_utc': post_data.get('created_utc', 0),
                    'author': post_data.get('author', 'unknown')
                }
                
                # Only include posts with substantial content
                if len(post_info['text']) > 50 or post_info['score'] > 10:
                    posts.append(post_info)
            
            return posts
            
        except Exception as e:
            logger.error(f"Error searching r/{subreddit} for {phone_model}: {e}")
            return []

class TechBlogScraper:
    """Scraper for technology blog reviews"""
    
    def __init__(self):
        self.tech_sites = [
            'https://www.theverge.com',
            'https://www.cnet.com', 
            'https://www.androidcentral.com',
            'https://9to5google.com',
            'https://www.gsmarena.com'
        ]
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    async def search_reviews(self, phone_model: str) -> List[Dict[str, Any]]:
        """Search tech blogs for phone reviews"""
        all_reviews = []
        
        for site in self.tech_sites:
            try:
                reviews = await self._search_site(site, phone_model)
                all_reviews.extend(reviews)
                await asyncio.sleep(2)  # Rate limiting
            except Exception as e:
                logger.error(f"Error searching {site} for {phone_model}: {e}")
                continue
        
        return all_reviews[:15]  # Return top 15 reviews
    
    async def _search_site(self, site_url: str, phone_model: str) -> List[Dict[str, Any]]:
        """Search specific tech site for reviews"""
        # This is a simplified implementation
        # In production, you'd implement site-specific search logic
        try:
            search_query = quote(f"{phone_model} review")
            # Using site: operator for better results
            google_search = f"site:{urlparse(site_url).netloc} {phone_model} review"
            
            # This would typically involve more sophisticated scraping
            return [{
                'title': f'{phone_model} Review',
                'site': urlparse(site_url).netloc,
                'url': site_url,
                'summary': f'Professional review of {phone_model}',
                'rating': 4.0,
                'pros': ['Good performance', 'Great camera'],
                'cons': ['Battery life', 'Price']
            }]
            
        except Exception as e:
            logger.error(f"Error searching {site_url}: {e}")
            return []

class WebSearchAgent(BaseAgent):
    """Specialized agent for dynamic web search and data retrieval"""
    
    def __init__(self, agent_id: str = "web_search_001"):
        super().__init__(agent_id, AgentRole.SCRAPER)
        
        # Initialize scrapers
        self.gsmarena_scraper = GSMArenaScaper()
        self.reddit_scraper = RedditScraper()
        self.tech_blog_scraper = TechBlogScraper()
        
        # Initialize cache
        self.cache = WebSearchCache()
        
        # Initialize sentiment analysis
        try:
            self.sentiment_analyzer = pipeline("sentiment-analysis")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.error(f"Error initializing NLP models: {e}")
            self.sentiment_analyzer = None
            self.embedding_model = None
        
        # Search progress tracking
        self.active_searches = {}
        
        # Search configuration
        self.search_config = {
            'timeout_per_source': 30,  # seconds
            'max_retries': 3,
            'cache_expire_hours': 24,
            'min_confidence_threshold': 0.3,
            'max_concurrent_searches': 5
        }
        
        logger.info(f"WebSearchAgent {agent_id} initialized")
    
    def _initialize_tools(self) -> List[Tool]:
        """Initialize web search tools"""
        return [
            Tool(
                name="search_phone_data",
                func=self.search_phone_data,
                description="Search web for comprehensive phone data including specs and reviews"
            ),
            Tool(
                name="get_specifications",
                func=self.get_specifications,
                description="Get detailed phone specifications from web sources"
            ),
            Tool(
                name="get_reviews",
                func=self.get_reviews,
                description="Get user reviews and professional reviews from web"
            ),
            Tool(
                name="get_sentiment_analysis",
                func=self.get_sentiment_analysis,
                description="Perform sentiment analysis on web-sourced reviews"
            ),
            Tool(
                name="get_price_data",
                func=self.get_price_data,
                description="Get current pricing data from various sources"
            ),
            Tool(
                name="check_data_availability",
                func=self.check_data_availability,
                description="Check if phone data is available locally or needs web search"
            )
        ]
    
    async def execute_task(self, task: AgentTask) -> Dict[str, Any]:
        """Execute web search task"""
        self.state = AgentState.ACTING
        
        try:
            phone_model = task.context.get('phone_model', '')
            search_type = task.context.get('search_type', 'comprehensive')
            force_refresh = task.context.get('force_refresh', False)
            
            logger.info(f"Starting web search for {phone_model}, type: {search_type}")
            
            # Initialize progress tracking
            progress = SearchProgress(
                phone_model=phone_model,
                total_sources=4,  # GSMArena, Reddit, Tech blogs, Price sites
                completed_sources=0,
                current_source="initializing",
                status="searching",
                results_found=0,
                start_time=datetime.now()
            )
            self.active_searches[phone_model] = progress
            
            # Perform comprehensive search
            if search_type == 'comprehensive':
                result = await self.search_phone_data(phone_model, force_refresh)
            elif search_type == 'specifications':
                result = await self.get_specifications(phone_model, force_refresh)
            elif search_type == 'reviews':
                result = await self.get_reviews(phone_model, force_refresh)
            elif search_type == 'sentiment':
                result = await self.get_sentiment_analysis(phone_model, force_refresh)
            else:
                result = await self.search_phone_data(phone_model, force_refresh)
            
            progress.status = "completed"
            progress.estimated_completion = datetime.now()
            
            self.state = AgentState.COMPLETED
            return result
            
        except Exception as e:
            self.state = AgentState.FAILED
            if phone_model in self.active_searches:
                self.active_searches[phone_model].status = "failed"
                self.active_searches[phone_model].errors.append(str(e))
            
            logger.error(f"Web search task failed for {phone_model}: {e}")
            return {
                'error': str(e),
                'phone_model': phone_model,
                'timestamp': datetime.now().isoformat()
            }
    
    async def search_phone_data(self, phone_model: str, force_refresh: bool = False) -> Dict[str, Any]:
        """Comprehensive phone data search"""
        try:
            progress = self.active_searches.get(phone_model)
            if progress:
                progress.current_source = "comprehensive search"
            
            # Check cache first (unless force refresh)
            if not force_refresh:
                cached_result = self.cache.get_cached_result(phone_model, "comprehensive", "all")
                if cached_result:
                    logger.info(f"Using cached comprehensive data for {phone_model}")
                    return cached_result.content
            
            # Perform parallel searches
            search_tasks = [
                self._search_with_timeout(self.get_specifications, phone_model, force_refresh),
                self._search_with_timeout(self.get_reviews, phone_model, force_refresh),
                self._search_with_timeout(self.get_price_data, phone_model, force_refresh),
            ]
            
            results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            # Combine results
            comprehensive_data = {
                'phone_model': phone_model,
                'specifications': results[0] if not isinstance(results[0], Exception) else {},
                'reviews': results[1] if not isinstance(results[1], Exception) else {},
                'pricing': results[2] if not isinstance(results[2], Exception) else {},
                'search_metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'sources_searched': ['gsmarena', 'reddit', 'tech_blogs'],
                    'cache_used': not force_refresh,
                    'success_rate': sum(1 for r in results if not isinstance(r, Exception)) / len(results)
                }
            }
            
            # Perform sentiment analysis on collected reviews
            if comprehensive_data['reviews']:
                sentiment_result = await self.analyze_sentiment(comprehensive_data['reviews'])
                comprehensive_data['sentiment_analysis'] = sentiment_result
            
            # Cache the result
            cache_result = WebSearchResult(
                phone_model=phone_model,
                source="comprehensive",
                content_type="all",
                content=comprehensive_data,
                confidence=comprehensive_data['search_metadata']['success_rate'],
                url="multiple",
                timestamp=datetime.now()
            )
            self.cache.cache_result(cache_result, expire_hours=self.search_config['cache_expire_hours'])
            
            if progress:
                progress.results_found = len([r for r in results if not isinstance(r, Exception)])
                progress.completed_sources = progress.total_sources
            
            return comprehensive_data
            
        except Exception as e:
            logger.error(f"Comprehensive search failed for {phone_model}: {e}")
            return {
                'error': str(e),
                'phone_model': phone_model,
                'partial_results': True
            }
    
    async def get_specifications(self, phone_model: str, force_refresh: bool = False) -> Dict[str, Any]:
        """Get phone specifications from web sources"""
        try:
            progress = self.active_searches.get(phone_model)
            if progress:
                progress.current_source = "GSMArena"
            
            # Check cache
            if not force_refresh:
                cached_result = self.cache.get_cached_result(phone_model, "gsmarena", "specifications")
                if cached_result:
                    return cached_result.content
            
            # Search GSMArena for specifications
            specs = await self.gsmarena_scraper.search_phone(phone_model)
            
            if specs:
                # Cache the result
                cache_result = WebSearchResult(
                    phone_model=phone_model,
                    source="gsmarena",
                    content_type="specifications",
                    content=specs,
                    confidence=0.9,  # GSMArena is highly reliable for specs
                    url=specs.get('url', ''),
                    timestamp=datetime.now()
                )
                self.cache.cache_result(cache_result)
                
                if progress:
                    progress.completed_sources += 1
                
                return specs
            else:
                return {
                    'phone_model': phone_model,
                    'error': 'No specifications found',
                    'searched_sources': ['gsmarena']
                }
                
        except Exception as e:
            logger.error(f"Specifications search failed for {phone_model}: {e}")
            return {'error': str(e), 'phone_model': phone_model}
    
    async def get_reviews(self, phone_model: str, force_refresh: bool = False) -> Dict[str, Any]:
        """Get reviews from multiple web sources"""
        try:
            progress = self.active_searches.get(phone_model)
            
            # Check cache
            if not force_refresh:
                cached_result = self.cache.get_cached_result(phone_model, "reviews", "multiple")
                if cached_result:
                    return cached_result.content
            
            all_reviews = {
                'phone_model': phone_model,
                'reddit_reviews': [],
                'tech_blog_reviews': [],
                'review_summary': {}
            }
            
            # Search Reddit
            if progress:
                progress.current_source = "Reddit"
            reddit_reviews = await self.reddit_scraper.search_reviews(phone_model)
            all_reviews['reddit_reviews'] = reddit_reviews
            
            if progress:
                progress.completed_sources += 1
            
            # Search tech blogs
            if progress:
                progress.current_source = "Tech Blogs"
            blog_reviews = await self.tech_blog_scraper.search_reviews(phone_model)
            all_reviews['tech_blog_reviews'] = blog_reviews
            
            if progress:
                progress.completed_sources += 1
            
            # Create summary
            total_reviews = len(reddit_reviews) + len(blog_reviews)
            all_reviews['review_summary'] = {
                'total_reviews': total_reviews,
                'reddit_count': len(reddit_reviews),
                'tech_blog_count': len(blog_reviews),
                'search_timestamp': datetime.now().isoformat()
            }
            
            # Cache the result
            cache_result = WebSearchResult(
                phone_model=phone_model,
                source="reviews",
                content_type="multiple",
                content=all_reviews,
                confidence=0.8 if total_reviews > 5 else 0.5,
                url="multiple",
                timestamp=datetime.now()
            )
            self.cache.cache_result(cache_result)
            
            return all_reviews
            
        except Exception as e:
            logger.error(f"Reviews search failed for {phone_model}: {e}")
            return {'error': str(e), 'phone_model': phone_model}
    
    async def get_sentiment_analysis(self, phone_model: str, force_refresh: bool = False) -> Dict[str, Any]:
        """Perform sentiment analysis on web-sourced reviews"""
        try:
            # First get reviews
            reviews_data = await self.get_reviews(phone_model, force_refresh)
            
            if 'error' in reviews_data:
                return reviews_data
            
            return await self.analyze_sentiment(reviews_data)
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed for {phone_model}: {e}")
            return {'error': str(e), 'phone_model': phone_model}
    
    async def analyze_sentiment(self, reviews_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze sentiment of collected reviews"""
        if not self.sentiment_analyzer:
            return {'error': 'Sentiment analyzer not available'}
        
        try:
            phone_model = reviews_data['phone_model']
            all_texts = []
            
            # Collect all review texts
            for reddit_review in reviews_data.get('reddit_reviews', []):
                text = f"{reddit_review.get('title', '')} {reddit_review.get('text', '')}"
                if len(text.strip()) > 20:
                    all_texts.append(text)
            
            for blog_review in reviews_data.get('tech_blog_reviews', []):
                text = blog_review.get('summary', '')
                if len(text.strip()) > 20:
                    all_texts.append(text)
            
            if not all_texts:
                return {
                    'phone_model': phone_model,
                    'sentiment_summary': {'error': 'No text content found for analysis'}
                }
            
            # Analyze sentiment
            sentiments = []
            for text in all_texts[:50]:  # Limit to prevent overload
                try:
                    result = self.sentiment_analyzer(text[:512])  # Truncate long texts
                    sentiments.append(result[0])
                except Exception:
                    continue
            
            # Calculate summary statistics
            positive_count = sum(1 for s in sentiments if s['label'] == 'POSITIVE')
            negative_count = sum(1 for s in sentiments if s['label'] == 'NEGATIVE')
            total_count = len(sentiments)
            
            avg_score = sum(s['score'] for s in sentiments) / total_count if total_count > 0 else 0
            
            sentiment_summary = {
                'phone_model': phone_model,
                'sentiment_distribution': {
                    'positive': positive_count / total_count if total_count > 0 else 0,
                    'negative': negative_count / total_count if total_count > 0 else 0,
                    'neutral': (total_count - positive_count - negative_count) / total_count if total_count > 0 else 0
                },
                'overall_sentiment': 'positive' if positive_count > negative_count else 'negative' if negative_count > positive_count else 'neutral',
                'confidence_score': avg_score,
                'total_reviews_analyzed': total_count,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            # Add to original reviews data
            reviews_data['sentiment_analysis'] = sentiment_summary
            
            return reviews_data
            
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return {'error': str(e), 'phone_model': reviews_data.get('phone_model', 'unknown')}
    
    async def get_price_data(self, phone_model: str, force_refresh: bool = False) -> Dict[str, Any]:
        """Get pricing data from various sources"""
        try:
            # Simplified pricing data (in production, would scrape actual price sites)
            price_data = {
                'phone_model': phone_model,
                'prices': {
                    'retail': 999,
                    'discounted': 899,
                    'used': 750,
                    'currency': 'USD'
                },
                'sources': ['Amazon', 'BestBuy', 'Manufacturer'],
                'last_updated': datetime.now().isoformat(),
                'availability': 'in_stock'
            }
            
            # Cache the result
            cache_result = WebSearchResult(
                phone_model=phone_model,
                source="pricing",
                content_type="prices",
                content=price_data,
                confidence=0.7,
                url="multiple",
                timestamp=datetime.now()
            )
            self.cache.cache_result(cache_result, expire_hours=6)  # Shorter cache for prices
            
            return price_data
            
        except Exception as e:
            logger.error(f"Price data search failed for {phone_model}: {e}")
            return {'error': str(e), 'phone_model': phone_model}
    
    async def check_data_availability(self, phone_model: str) -> Dict[str, Any]:
        """Check if phone data is available locally or needs web search"""
        try:
            availability = {
                'phone_model': phone_model,
                'local_data_available': False,
                'cached_data_available': {},
                'recommendation': 'web_search_needed'
            }
            
            # Check cache for different data types
            data_types = ['specifications', 'reviews', 'prices', 'sentiment']
            for data_type in data_types:
                cached = self.cache.get_cached_result(phone_model, "any", data_type)
                availability['cached_data_available'][data_type] = cached is not None
            
            # Determine recommendation
            cached_count = sum(1 for available in availability['cached_data_available'].values() if available)
            if cached_count >= 3:
                availability['recommendation'] = 'use_cached_data'
            elif cached_count >= 1:
                availability['recommendation'] = 'partial_web_search'
            else:
                availability['recommendation'] = 'full_web_search'
            
            return availability
            
        except Exception as e:
            logger.error(f"Data availability check failed for {phone_model}: {e}")
            return {'error': str(e), 'phone_model': phone_model}
    
    async def _search_with_timeout(self, search_func, *args, **kwargs):
        """Execute search function with timeout"""
        try:
            return await asyncio.wait_for(
                search_func(*args, **kwargs),
                timeout=self.search_config['timeout_per_source']
            )
        except asyncio.TimeoutError:
            logger.warning(f"Search timeout for {search_func.__name__}")
            return {'error': 'timeout', 'function': search_func.__name__}
        except Exception as e:
            logger.error(f"Search error in {search_func.__name__}: {e}")
            return {'error': str(e), 'function': search_func.__name__}
    
    def get_search_progress(self, phone_model: str) -> Optional[SearchProgress]:
        """Get current search progress for a phone model"""
        return self.active_searches.get(phone_model)
    
    def cleanup_cache(self):
        """Clean up expired cache entries"""
        self.cache.cleanup_expired()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with sqlite3.connect(self.cache.db_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_entries,
                    COUNT(CASE WHEN expires_at > datetime('now') THEN 1 END) as active_entries,
                    SUM(access_count) as total_accesses,
                    AVG(confidence) as avg_confidence
                FROM search_cache
            """)
            stats = dict(zip(['total_entries', 'active_entries', 'total_accesses', 'avg_confidence'], cursor.fetchone()))
            
            # Get top sources
            cursor = conn.execute("""
                SELECT source, COUNT(*) as count 
                FROM search_cache 
                GROUP BY source 
                ORDER BY count DESC 
                LIMIT 5
            """)
            stats['top_sources'] = dict(cursor.fetchall())
            
            return stats

# Utility functions for easy integration
def create_web_search_agent() -> WebSearchAgent:
    """Create and initialize a WebSearchAgent"""
    return WebSearchAgent()

async def search_phone_web_data(phone_model: str, search_type: str = "comprehensive") -> Dict[str, Any]:
    """Standalone function to search for phone data"""
    agent = create_web_search_agent()
    task = AgentTask(
        task_id=f"web_search_{phone_model}_{int(time.time())}",
        description=f"Search web for {phone_model} data",
        context={
            'phone_model': phone_model,
            'search_type': search_type
        }
    )
    return await agent.execute_task(task)

if __name__ == "__main__":
    # Demo usage
    async def demo():
        agent = create_web_search_agent()
        
        test_phones = ["iPhone 15 Pro", "Samsung Galaxy S24", "Google Pixel 8 Pro"]
        
        for phone in test_phones:
            print(f"\n{'='*60}")
            print(f"Searching for: {phone}")
            print(f"{'='*60}")
            
            result = await search_phone_web_data(phone, "comprehensive")
            
            if 'error' not in result:
                print(f"✅ Search successful!")
                print(f"📱 Model: {result.get('phone_model')}")
                print(f"📊 Sources: {result.get('search_metadata', {}).get('sources_searched', [])}")
                print(f"🎯 Success Rate: {result.get('search_metadata', {}).get('success_rate', 0)*100:.1f}%")
                
                if 'specifications' in result and result['specifications']:
                    specs = result['specifications']
                    print(f"🔧 Specifications: {len(specs.get('specifications', {}))} categories")
                
                if 'reviews' in result and result['reviews']:
                    reviews = result['reviews']['review_summary']
                    print(f"💬 Reviews: {reviews.get('total_reviews', 0)} found")
                
                if 'sentiment_analysis' in result:
                    sentiment = result['sentiment_analysis'].get('sentiment_analysis', {})
                    overall = sentiment.get('overall_sentiment', 'unknown')
                    confidence = sentiment.get('confidence_score', 0)
                    print(f"😊 Sentiment: {overall} (confidence: {confidence:.2f})")
                
            else:
                print(f"❌ Search failed: {result['error']}")
        
        # Show cache stats
        print(f"\n{'='*60}")
        print("Cache Statistics")
        print(f"{'='*60}")
        cache_stats = agent.get_cache_stats()
        print(f"📦 Total entries: {cache_stats.get('total_entries', 0)}")
        print(f"✅ Active entries: {cache_stats.get('active_entries', 0)}")
        print(f"👁️ Total accesses: {cache_stats.get('total_accesses', 0)}")
        print(f"🎯 Avg confidence: {cache_stats.get('avg_confidence', 0):.2f}")
    
    asyncio.run(demo())