"""
Specialized Phone Review Sites Scraper for AI Phone Review Engine
Supports niche, regional, and specialized phone review websites worldwide
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
import yaml
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SpecializedSource:
    """Configuration for specialized review sources"""
    name: str
    base_url: str
    region: str
    language: str
    specialty: str  # reviews, news, specs, forums, regional, budget, flagship
    selectors: Dict[str, str]
    patterns: Dict[str, str]
    headers: Dict[str, str]
    rate_limit: float
    enabled: bool
    last_updated: str

@dataclass
class ExtractedReview:
    """Structured review data from specialized sources"""
    source_name: str
    phone_name: str
    review_title: str
    author: str
    rating: Optional[float]
    rating_scale: str
    content: str
    pros: List[str]
    cons: List[str]
    specifications: Dict[str, Any]
    images: List[str]
    published_date: str
    review_url: str
    language: str
    region: str
    confidence_score: float

class SpecializedReviewScraper:
    """Scraper for specialized and regional phone review sites"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize specialized review scraper"""
        
        self.config = config or {
            'max_concurrent_requests': 5,
            'request_timeout': 45,
            'enable_caching': True,
            'cache_duration_hours': 12,
            'user_agent_rotation': True,
            'respect_robots_txt': True,
            'sources_config_file': 'config/specialized_sources.yaml',
            'min_content_length': 300,
            'max_retries': 3,
            'retry_delay': 2
        }
        
        # Load specialized sources configuration
        self.sources = self._load_sources_config()
        
        # Session for web requests
        self.session = None
        
        # Cache for scraped data
        self.cache = {}
        
        # User agents for rotation
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0'
        ]
        self.current_ua_index = 0
    
    def _load_sources_config(self) -> Dict[str, SpecializedSource]:
        """Load specialized sources configuration"""
        
        # Default sources configuration
        default_sources = {
            # European Sources
            'gsmarena': {
                'name': 'GSMArena',
                'base_url': 'https://www.gsmarena.com',
                'region': 'Global',
                'language': 'en',
                'specialty': 'specifications',
                'selectors': {
                    'title': 'h1.specs-phone-name-title',
                    'rating': '.rating-score',
                    'specs': '#specs-list table tr',
                    'review_content': '.review-body',
                    'pros': '.pros li',
                    'cons': '.cons li',
                    'images': '.specs-photo-main img'
                },
                'patterns': {
                    'phone_name': r'([A-Za-z\s]+\d+[A-Za-z\s]*)',
                    'price': r'\$(\d+(?:,\d{3})*)',
                    'rating_scale': r'(\d+\.?\d*)\s*/\s*10'
                },
                'headers': {'User-Agent': 'Mozilla/5.0'},
                'rate_limit': 2.0,
                'enabled': True,
                'last_updated': datetime.now().isoformat()
            },
            
            'phonearena': {
                'name': 'PhoneArena',
                'base_url': 'https://www.phonearena.com',
                'region': 'Global',
                'language': 'en',
                'specialty': 'reviews',
                'selectors': {
                    'title': 'h1.s-article-title',
                    'rating': '.s-article-rating-score',
                    'review_content': '.s-article-body',
                    'pros': '.pros-cons .pros li',
                    'cons': '.pros-cons .cons li',
                    'author': '.s-article-author',
                    'date': '.s-article-date'
                },
                'patterns': {
                    'rating_scale': r'(\d+\.?\d*)\s*/\s*10',
                    'phone_model': r'(iPhone|Galaxy|Pixel|OnePlus|Xiaomi)\s+[\w\s]+'
                },
                'headers': {},
                'rate_limit': 3.0,
                'enabled': True,
                'last_updated': datetime.now().isoformat()
            },
            
            # Asian Sources
            'gizmochina': {
                'name': 'Gizmochina',
                'base_url': 'https://www.gizmochina.com',
                'region': 'Asia',
                'language': 'en',
                'specialty': 'news',
                'selectors': {
                    'title': 'h1.entry-title',
                    'content': '.entry-content',
                    'author': '.author-name',
                    'date': '.entry-date',
                    'images': '.wp-post-image'
                },
                'patterns': {
                    'phone_mention': r'(Xiaomi|Huawei|OnePlus|Oppo|Vivo|Realme)\s+[\w\s]+',
                    'specs': r'(\d+(?:GB|MP|mAh|inch))'
                },
                'headers': {},
                'rate_limit': 2.5,
                'enabled': True,
                'last_updated': datetime.now().isoformat()
            },
            
            'androidauthority': {
                'name': 'Android Authority',
                'base_url': 'https://www.androidauthority.com',
                'region': 'Global',
                'language': 'en',
                'specialty': 'reviews',
                'selectors': {
                    'title': 'h1.post-title',
                    'rating': '.score-number',
                    'content': '.post-content',
                    'pros': '.verdict-pros li',
                    'cons': '.verdict-cons li',
                    'author': '.author-name',
                    'specs': '.specs-table tr'
                },
                'patterns': {
                    'rating_scale': r'(\d+)\s*/\s*10',
                    'phone_model': r'(Samsung|Google|OnePlus|Xiaomi|Sony)\s+[\w\s]+'
                },
                'headers': {},
                'rate_limit': 2.0,
                'enabled': True,
                'last_updated': datetime.now().isoformat()
            },
            
            # Regional European Sources
            'nextpit_de': {
                'name': 'NextPit DE',
                'base_url': 'https://www.nextpit.de',
                'region': 'Germany',
                'language': 'de',
                'specialty': 'reviews',
                'selectors': {
                    'title': 'h1.article-title',
                    'rating': '.rating-stars',
                    'content': '.article-content',
                    'pros': '.pros-cons .pros li',
                    'cons': '.pros-cons .cons li'
                },
                'patterns': {
                    'phone_model': r'(iPhone|Samsung|Google|OnePlus)\s+[\w\s]+',
                    'rating': r'(\d+\.?\d*)\s*von\s*5'
                },
                'headers': {'Accept-Language': 'de-DE,de;q=0.9'},
                'rate_limit': 3.0,
                'enabled': True,
                'last_updated': datetime.now().isoformat()
            },
            
            'frandroid': {
                'name': 'Frandroid',
                'base_url': 'https://www.frandroid.com',
                'region': 'France',
                'language': 'fr',
                'specialty': 'reviews',
                'selectors': {
                    'title': 'h1.entry-title',
                    'rating': '.note-finale',
                    'content': '.entry-content',
                    'pros': '.avantages li',
                    'cons': '.inconvenients li'
                },
                'patterns': {
                    'phone_model': r'(iPhone|Samsung|OnePlus|Xiaomi)\s+[\w\s]+',
                    'rating': r'(\d+)/10'
                },
                'headers': {'Accept-Language': 'fr-FR,fr;q=0.9'},
                'rate_limit': 2.5,
                'enabled': True,
                'last_updated': datetime.now().isoformat()
            },
            
            # Specialized Budget/Value Sources
            'budgetphone_reviews': {
                'name': 'Budget Phone Reviews',
                'base_url': 'https://www.budgetphonereviews.com',
                'region': 'Global',
                'language': 'en',
                'specialty': 'budget',
                'selectors': {
                    'title': 'h1.review-title',
                    'rating': '.budget-rating',
                    'content': '.review-text',
                    'price_value': '.price-analysis',
                    'alternatives': '.budget-alternatives'
                },
                'patterns': {
                    'price_range': r'under\s*\$(\d+)',
                    'value_score': r'(\d+\.?\d*)\s*/\s*10\s*value'
                },
                'headers': {},
                'rate_limit': 2.0,
                'enabled': True,
                'last_updated': datetime.now().isoformat()
            },
            
            # Flagship Specialized Sources
            'flagship_central': {
                'name': 'Flagship Central',
                'base_url': 'https://www.flagshipcentral.com',
                'region': 'Global',
                'language': 'en',
                'specialty': 'flagship',
                'selectors': {
                    'title': 'h1.flagship-title',
                    'rating': '.premium-score',
                    'content': '.flagship-review',
                    'camera_test': '.camera-analysis',
                    'performance': '.benchmark-results'
                },
                'patterns': {
                    'flagship_models': r'(iPhone\s+\d+\s+Pro|Galaxy\s+S\d+\s+Ultra|Pixel\s+\d+\s+Pro)',
                    'premium_features': r'(120Hz|5G|wireless charging|IP68)'
                },
                'headers': {},
                'rate_limit': 3.0,
                'enabled': True,
                'last_updated': datetime.now().isoformat()
            },
            
            # Tech Forum Sources
            'xda_developers': {
                'name': 'XDA Developers',
                'base_url': 'https://www.xda-developers.com',
                'region': 'Global',
                'language': 'en',
                'specialty': 'forums',
                'selectors': {
                    'title': 'h1.p-title-value',
                    'content': '.bbWrapper',
                    'author': '.username',
                    'date': '.u-dt',
                    'tech_discussion': '.message-body'
                },
                'patterns': {
                    'phone_discussion': r'(rooting|custom ROM|bootloader|kernel)',
                    'device_mention': r'(OnePlus|Pixel|Galaxy|Xiaomi)\s+[\w\s]+'
                },
                'headers': {},
                'rate_limit': 2.0,
                'enabled': True,
                'last_updated': datetime.now().isoformat()
            }
        }
        
        sources = {}
        for source_id, config in default_sources.items():
            sources[source_id] = SpecializedSource(**config)
        
        # Try to load from config file if exists
        config_path = Path(self.config['sources_config_file'])
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    file_sources = yaml.safe_load(f)
                
                for source_id, config in file_sources.items():
                    sources[source_id] = SpecializedSource(**config)
                
                logger.info(f"Loaded {len(file_sources)} sources from config file")
                
            except Exception as e:
                logger.error(f"Failed to load sources config: {e}")
        
        return sources
    
    async def scrape_phone_reviews(self, phone_name: str, source_filter: Dict[str, Any] = None) -> List[ExtractedReview]:
        """
        Scrape phone reviews from specialized sources
        
        Args:
            phone_name: Name of the phone to search for
            source_filter: Filter criteria (region, language, specialty)
            
        Returns:
            List of extracted reviews
        """
        
        logger.info(f"Starting specialized review scrape for: {phone_name}")
        
        # Filter sources based on criteria
        active_sources = self._filter_sources(source_filter)
        
        # Initialize session
        await self._init_session()
        
        try:
            # Scrape from multiple sources concurrently
            semaphore = asyncio.Semaphore(self.config['max_concurrent_requests'])
            
            async def scrape_single_source(source_id, source):
                async with semaphore:
                    return await self._scrape_from_source(phone_name, source_id, source)
            
            tasks = [
                scrape_single_source(source_id, source)
                for source_id, source in active_sources.items()
                if source.enabled
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Collect successful results
            all_reviews = []
            for result in results:
                if result and not isinstance(result, Exception):
                    all_reviews.extend(result)
            
            # Sort by confidence score
            all_reviews.sort(key=lambda r: r.confidence_score, reverse=True)
            
            logger.info(f"Scraped {len(all_reviews)} reviews from {len(active_sources)} specialized sources")
            return all_reviews
            
        finally:
            await self._cleanup_session()
    
    async def discover_regional_reviews(self, phone_name: str, regions: List[str] = None) -> Dict[str, List[ExtractedReview]]:
        """
        Discover reviews from regional sources
        
        Args:
            phone_name: Phone to search for
            regions: List of regions to focus on
            
        Returns:
            Dictionary of reviews grouped by region
        """
        
        if not regions:
            regions = ['Global', 'Europe', 'Asia', 'North America']
        
        regional_reviews = {}
        
        for region in regions:
            source_filter = {'region': region}
            reviews = await self.scrape_phone_reviews(phone_name, source_filter)
            if reviews:
                regional_reviews[region] = reviews
        
        return regional_reviews
    
    async def scrape_by_specialty(self, phone_name: str, specialty: str) -> List[ExtractedReview]:
        """
        Scrape reviews by specialty (budget, flagship, gaming, etc.)
        
        Args:
            phone_name: Phone to search for
            specialty: Type of specialty content
            
        Returns:
            List of specialized reviews
        """
        
        source_filter = {'specialty': specialty}
        return await self.scrape_phone_reviews(phone_name, source_filter)
    
    def _filter_sources(self, filter_criteria: Dict[str, Any] = None) -> Dict[str, SpecializedSource]:
        """Filter sources based on criteria"""
        
        if not filter_criteria:
            return self.sources
        
        filtered = {}
        
        for source_id, source in self.sources.items():
            include = True
            
            if 'region' in filter_criteria:
                if source.region != filter_criteria['region'] and source.region != 'Global':
                    include = False
            
            if 'language' in filter_criteria:
                if source.language != filter_criteria['language']:
                    include = False
            
            if 'specialty' in filter_criteria:
                if source.specialty != filter_criteria['specialty']:
                    include = False
            
            if 'enabled' in filter_criteria:
                if source.enabled != filter_criteria['enabled']:
                    include = False
            
            if include:
                filtered[source_id] = source
        
        return filtered
    
    async def _scrape_from_source(self, phone_name: str, source_id: str, source: SpecializedSource) -> List[ExtractedReview]:
        """Scrape reviews from a single specialized source"""
        
        try:
            logger.info(f"Scraping {source.name} for {phone_name}")
            
            # Build search URLs
            search_urls = self._build_search_urls(phone_name, source)
            
            reviews = []
            
            for url in search_urls[:3]:  # Limit to 3 URLs per source
                try:
                    # Get headers with rotation
                    headers = self._get_headers_for_source(source)
                    
                    # Fetch page content
                    async with self.session.get(url, headers=headers, timeout=self.config['request_timeout']) as response:
                        if response.status != 200:
                            logger.warning(f"HTTP {response.status} for {url}")
                            continue
                        
                        content = await response.text()
                        
                        # Extract review data
                        review = self._extract_review_data(content, url, source_id, source, phone_name)
                        
                        if review and len(review.content) >= self.config['min_content_length']:
                            reviews.append(review)
                    
                    # Rate limiting
                    await asyncio.sleep(source.rate_limit)
                    
                except Exception as e:
                    logger.error(f"Failed to scrape {url}: {e}")
                    continue
            
            return reviews
            
        except Exception as e:
            logger.error(f"Source scraping failed for {source.name}: {e}")
            return []
    
    def _build_search_urls(self, phone_name: str, source: SpecializedSource) -> List[str]:
        """Build search URLs for a source"""
        
        urls = []
        
        # Clean phone name for URL
        clean_name = re.sub(r'[^\w\s]', '', phone_name).replace(' ', '+')
        
        # Common search patterns
        search_patterns = [
            f"/search?q={clean_name}+review",
            f"/search/?query={clean_name}",
            f"/{clean_name.replace('+', '-').lower()}-review",
            f"/reviews/{clean_name.replace('+', '-').lower()}",
            f"/phone/{clean_name.replace('+', '-').lower()}"
        ]
        
        for pattern in search_patterns:
            url = urljoin(source.base_url, pattern)
            urls.append(url)
        
        # Source-specific URL patterns
        if 'gsmarena' in source.base_url:
            model_url = f"{source.base_url}/{clean_name.replace('+', '_').lower()}.php"
            urls.append(model_url)
        
        elif 'phonearena' in source.base_url:
            review_url = f"{source.base_url}/reviews/{clean_name.replace('+', '-').lower()}"
            urls.append(review_url)
        
        elif 'xda-developers' in source.base_url:
            forum_url = f"{source.base_url}/search/10000000/?q={clean_name}&o=relevance"
            urls.append(forum_url)
        
        return urls[:5]  # Limit to 5 URLs max
    
    def _get_headers_for_source(self, source: SpecializedSource) -> Dict[str, str]:
        """Get appropriate headers for a source"""
        
        headers = source.headers.copy()
        
        # Add user agent rotation if enabled
        if self.config['user_agent_rotation']:
            headers['User-Agent'] = self.user_agents[self.current_ua_index % len(self.user_agents)]
            self.current_ua_index += 1
        
        # Add default headers if not present
        if 'User-Agent' not in headers:
            headers['User-Agent'] = self.user_agents[0]
        
        if 'Accept' not in headers:
            headers['Accept'] = 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
        
        return headers
    
    def _extract_review_data(self, html_content: str, url: str, source_id: str, 
                           source: SpecializedSource, phone_name: str) -> Optional[ExtractedReview]:
        """Extract review data from HTML content"""
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract basic information
            title = self._extract_with_selector(soup, source.selectors.get('title', ''))
            content = self._extract_with_selector(soup, source.selectors.get('content', ''))
            author = self._extract_with_selector(soup, source.selectors.get('author', ''))
            
            # Extract rating
            rating = None
            rating_scale = "unknown"
            
            rating_element = self._extract_with_selector(soup, source.selectors.get('rating', ''))
            if rating_element:
                # Try to parse rating with patterns
                for pattern_name, pattern in source.patterns.items():
                    if 'rating' in pattern_name:
                        match = re.search(pattern, rating_element)
                        if match:
                            rating = float(match.group(1))
                            if '/10' in pattern:
                                rating_scale = "10"
                            elif '/5' in pattern:
                                rating_scale = "5"
                            break
            
            # Extract pros and cons
            pros = self._extract_list_items(soup, source.selectors.get('pros', ''))
            cons = self._extract_list_items(soup, source.selectors.get('cons', ''))
            
            # Extract specifications
            specifications = self._extract_specifications(soup, source.selectors.get('specs', ''), source.patterns)
            
            # Extract images
            images = self._extract_images(soup, source.selectors.get('images', ''))
            
            # Extract date
            date_element = self._extract_with_selector(soup, source.selectors.get('date', ''))
            published_date = self._parse_date(date_element) if date_element else datetime.now().isoformat()
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence(title, content, pros, cons, specifications)
            
            # Validate minimum requirements
            if not title or not content or confidence_score < 0.3:
                return None
            
            return ExtractedReview(
                source_name=source.name,
                phone_name=phone_name,
                review_title=title,
                author=author or "Unknown",
                rating=rating,
                rating_scale=rating_scale,
                content=content,
                pros=pros,
                cons=cons,
                specifications=specifications,
                images=images,
                published_date=published_date,
                review_url=url,
                language=source.language,
                region=source.region,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            logger.error(f"Data extraction failed for {url}: {e}")
            return None
    
    def _extract_with_selector(self, soup: BeautifulSoup, selector: str) -> Optional[str]:
        """Extract text using CSS selector"""
        
        if not selector:
            return None
        
        try:
            elements = soup.select(selector)
            if elements:
                return ' '.join(elem.get_text(strip=True) for elem in elements)
        except Exception as e:
            logger.debug(f"Selector extraction failed: {e}")
        
        return None
    
    def _extract_list_items(self, soup: BeautifulSoup, selector: str) -> List[str]:
        """Extract list items using CSS selector"""
        
        if not selector:
            return []
        
        try:
            elements = soup.select(selector)
            return [elem.get_text(strip=True) for elem in elements if elem.get_text(strip=True)]
        except Exception as e:
            logger.debug(f"List extraction failed: {e}")
            return []
    
    def _extract_specifications(self, soup: BeautifulSoup, selector: str, patterns: Dict[str, str]) -> Dict[str, Any]:
        """Extract specifications from HTML"""
        
        specs = {}
        
        if not selector:
            return specs
        
        try:
            # Try table-based extraction first
            spec_elements = soup.select(selector)
            
            for elem in spec_elements:
                text = elem.get_text(strip=True)
                
                # Try to parse with patterns
                for spec_name, pattern in patterns.items():
                    if 'spec' in spec_name or spec_name in ['display', 'battery', 'camera', 'storage', 'ram']:
                        matches = re.findall(pattern, text, re.IGNORECASE)
                        if matches:
                            specs[spec_name] = matches[0] if len(matches) == 1 else matches
                
                # Generic specification parsing
                if ':' in text:
                    parts = text.split(':', 1)
                    if len(parts) == 2:
                        key = parts[0].strip().lower().replace(' ', '_')
                        value = parts[1].strip()
                        if key and value and len(key) < 50 and len(value) < 200:
                            specs[key] = value
        
        except Exception as e:
            logger.debug(f"Specs extraction failed: {e}")
        
        return specs
    
    def _extract_images(self, soup: BeautifulSoup, selector: str) -> List[str]:
        """Extract image URLs"""
        
        if not selector:
            return []
        
        try:
            img_elements = soup.select(selector)
            images = []
            
            for img in img_elements[:10]:  # Limit to 10 images
                src = img.get('src') or img.get('data-src') or img.get('data-lazy-src')
                if src:
                    if src.startswith('//'):
                        src = 'https:' + src
                    elif src.startswith('/'):
                        src = urljoin(soup.base_url if hasattr(soup, 'base_url') else '', src)
                    
                    if src.startswith('http'):
                        images.append(src)
            
            return images
            
        except Exception as e:
            logger.debug(f"Image extraction failed: {e}")
            return []
    
    def _parse_date(self, date_string: str) -> str:
        """Parse date string to ISO format"""
        
        if not date_string:
            return datetime.now().isoformat()
        
        try:
            # Try common date formats
            date_patterns = [
                r'(\d{4}-\d{2}-\d{2})',
                r'(\d{2}/\d{2}/\d{4})',
                r'(\d{1,2}\s+\w+\s+\d{4})',
                r'(\w+\s+\d{1,2},\s+\d{4})'
            ]
            
            for pattern in date_patterns:
                match = re.search(pattern, date_string)
                if match:
                    date_str = match.group(1)
                    # This would need more sophisticated date parsing
                    # For now, return current date
                    return datetime.now().isoformat()
            
        except Exception as e:
            logger.debug(f"Date parsing failed: {e}")
        
        return datetime.now().isoformat()
    
    def _calculate_confidence(self, title: str, content: str, pros: List[str], 
                            cons: List[str], specifications: Dict[str, Any]) -> float:
        """Calculate confidence score for extracted review"""
        
        score = 0.0
        
        # Title quality
        if title and len(title) > 10:
            score += 0.2
            if any(word in title.lower() for word in ['review', 'test', 'hands-on']):
                score += 0.1
        
        # Content quality
        if content:
            content_length = len(content)
            if content_length > 500:
                score += 0.3
            elif content_length > 200:
                score += 0.2
            else:
                score += 0.1
        
        # Structured data
        if pros:
            score += min(len(pros) * 0.05, 0.15)
        
        if cons:
            score += min(len(cons) * 0.05, 0.15)
        
        if specifications:
            score += min(len(specifications) * 0.02, 0.20)
        
        # Phone mention in content
        if content and any(brand in content.lower() for brand in ['iphone', 'samsung', 'google', 'oneplus', 'xiaomi']):
            score += 0.1
        
        return min(score, 1.0)
    
    async def get_source_statistics(self) -> Dict[str, Any]:
        """Get statistics about specialized sources"""
        
        stats = {
            'total_sources': len(self.sources),
            'enabled_sources': len([s for s in self.sources.values() if s.enabled]),
            'by_region': defaultdict(int),
            'by_language': defaultdict(int),
            'by_specialty': defaultdict(int)
        }
        
        for source in self.sources.values():
            stats['by_region'][source.region] += 1
            stats['by_language'][source.language] += 1
            stats['by_specialty'][source.specialty] += 1
        
        return dict(stats)
    
    def update_source_config(self, source_id: str, config_updates: Dict[str, Any]):
        """Update configuration for a specific source"""
        
        if source_id in self.sources:
            source = self.sources[source_id]
            
            for key, value in config_updates.items():
                if hasattr(source, key):
                    setattr(source, key, value)
            
            source.last_updated = datetime.now().isoformat()
            logger.info(f"Updated configuration for source: {source_id}")
    
    def add_custom_source(self, source_id: str, source_config: Dict[str, Any]):
        """Add a custom specialized source"""
        
        try:
            self.sources[source_id] = SpecializedSource(**source_config)
            logger.info(f"Added custom source: {source_id}")
        except Exception as e:
            logger.error(f"Failed to add custom source {source_id}: {e}")
    
    # Utility methods
    async def _init_session(self):
        """Initialize aiohttp session"""
        if not self.session:
            connector = aiohttp.TCPConnector(limit=20)
            timeout = aiohttp.ClientTimeout(total=self.config['request_timeout'])
            self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
    
    async def _cleanup_session(self):
        """Cleanup aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None

# Factory function
def create_specialized_review_scraper(config=None):
    """Create configured specialized review scraper"""
    return SpecializedReviewScraper(config=config)