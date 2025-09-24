"""
Enhanced Web Scraper for AI Phone Review Engine
Supports major review sites with robust error handling and rate limiting
"""

import asyncio
import aiohttp
import time
import logging
import json
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from urllib.parse import quote_plus, urljoin, urlparse
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
import random
from collections import defaultdict
import hashlib

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ScrapedReview:
    """Structure for scraped review data"""
    phone_model: str
    source: str
    title: str
    content: str
    rating: Optional[float]
    author: Optional[str]
    date: Optional[str]
    pros: List[str]
    cons: List[str]
    url: str
    confidence: float
    scraped_at: str

@dataclass
class ScrapedPhoneData:
    """Comprehensive scraped phone information"""
    model: str
    brand: str
    specifications: Dict[str, Any]
    reviews: List[ScrapedReview]
    overall_rating: Optional[float]
    review_count: int
    price_info: Dict[str, Any]
    availability: Dict[str, Any]
    images: List[str]
    source_urls: List[str]
    scrape_metadata: Dict[str, Any]

class RateLimiter:
    """Intelligent rate limiter with per-domain tracking"""
    
    def __init__(self):
        self.domain_limits = defaultdict(lambda: {'calls': 0, 'reset_time': time.time() + 60})
        self.default_delay = 2.0
        self.domain_delays = {
            'gsmarena.com': 3.0,
            'phonearena.com': 2.5,
            'cnet.com': 2.0,
            'techcrunch.com': 1.5,
            'techradar.com': 2.0,
            'androidauthority.com': 2.0
        }
    
    async def wait_if_needed(self, url: str):
        """Wait if rate limit would be exceeded"""
        domain = urlparse(url).netloc.lower()
        current_time = time.time()
        
        # Reset counter if minute has passed
        if current_time >= self.domain_limits[domain]['reset_time']:
            self.domain_limits[domain] = {'calls': 0, 'reset_time': current_time + 60}
        
        # Check if we need to wait
        if self.domain_limits[domain]['calls'] >= 20:  # 20 calls per minute max
            wait_time = self.domain_limits[domain]['reset_time'] - current_time
            if wait_time > 0:
                logger.info(f"Rate limiting: waiting {wait_time:.1f}s for {domain}")
                await asyncio.sleep(wait_time)
                self.domain_limits[domain] = {'calls': 0, 'reset_time': current_time + 60}
        
        # Apply domain-specific delay
        delay = self.domain_delays.get(domain, self.default_delay)
        await asyncio.sleep(delay + random.uniform(0.1, 0.5))  # Add jitter
        
        self.domain_limits[domain]['calls'] += 1

class CircuitBreaker:
    """Circuit breaker pattern for handling failing services"""
    
    def __init__(self, failure_threshold=5, timeout=300):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = defaultdict(int)
        self.last_failure_time = defaultdict(float)
        self.state = defaultdict(lambda: 'closed')  # closed, open, half-open
    
    def can_execute(self, service_name: str) -> bool:
        """Check if service can be called"""
        current_time = time.time()
        
        if self.state[service_name] == 'open':
            if current_time - self.last_failure_time[service_name] > self.timeout:
                self.state[service_name] = 'half-open'
                logger.info(f"Circuit breaker half-open for {service_name}")
                return True
            return False
        
        return True
    
    def record_success(self, service_name: str):
        """Record successful call"""
        self.failure_count[service_name] = 0
        self.state[service_name] = 'closed'
    
    def record_failure(self, service_name: str):
        """Record failed call"""
        self.failure_count[service_name] += 1
        self.last_failure_time[service_name] = time.time()
        
        if self.failure_count[service_name] >= self.failure_threshold:
            self.state[service_name] = 'open'
            logger.warning(f"Circuit breaker opened for {service_name}")

class EnhancedWebScraper:
    """Enhanced web scraper with support for major review sites"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the enhanced web scraper"""
        
        self.config = config or {
            'max_concurrent_requests': 5,
            'request_timeout': 30,
            'max_retries': 3,
            'use_selenium': True,
            'headless': True,
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'enable_javascript': True,
            'cache_results': True
        }
        
        # Initialize components
        self.rate_limiter = RateLimiter()
        self.circuit_breaker = CircuitBreaker()
        
        # Site configurations
        self.site_configs = {
            'gsmarena': {
                'base_url': 'https://www.gsmarena.com',
                'search_url': 'https://www.gsmarena.com/results.php3?sQuickSearch=yes&sName={query}',
                'enabled': True,
                'priority': 1,
                'requires_js': True,
                'selectors': {
                    'search_results': '.makers ul li a',
                    'phone_title': '.specs-phone-name-title',
                    'rating': '.rating-score',
                    'specs': '.specs-brief li',
                    'reviews': '.user-reviews .review-item'
                }
            },
            'phonearena': {
                'base_url': 'https://www.phonearena.com',
                'search_url': 'https://www.phonearena.com/search?term={query}',
                'enabled': True,
                'priority': 2,
                'requires_js': True,
                'selectors': {
                    'search_results': '.search-results .result-item a',
                    'phone_title': '.review-header h1',
                    'rating': '.rating-number',
                    'specs': '.specs-table tr',
                    'reviews': '.user-review'
                }
            },
            'cnet': {
                'base_url': 'https://www.cnet.com',
                'search_url': 'https://www.cnet.com/search/?query={query}',
                'enabled': True,
                'priority': 3,
                'requires_js': False,
                'selectors': {
                    'search_results': '.searchResults .result a',
                    'phone_title': '.review-title h1',
                    'rating': '.rating-badge',
                    'content': '.article-main-body',
                    'pros': '.pros-cons .pros li',
                    'cons': '.pros-cons .cons li'
                }
            },
            'techradar': {
                'base_url': 'https://www.techradar.com',
                'search_url': 'https://www.techradar.com/search?searchTerm={query}',
                'enabled': True,
                'priority': 4,
                'requires_js': False,
                'selectors': {
                    'search_results': '.search-results .search-result a',
                    'phone_title': '.review-header h1',
                    'rating': '.rating-score',
                    'content': '.text-copy',
                    'verdict': '.verdict'
                }
            }
        }
        
        # Initialize session
        self.session = None
        self.driver = None
        
    async def scrape_phone_reviews(self, phone_query: str, max_sources: int = 4) -> ScrapedPhoneData:
        """
        Scrape phone reviews from multiple sources
        
        Args:
            phone_query: Phone model to search for
            max_sources: Maximum number of sources to scrape
            
        Returns:
            ScrapedPhoneData with comprehensive information
        """
        
        logger.info(f"Starting enhanced scraping for: {phone_query}")
        
        # Initialize session
        await self._init_session()
        
        try:
            # Get enabled sources
            enabled_sources = [
                (name, config) for name, config in self.site_configs.items() 
                if config['enabled'] and self.circuit_breaker.can_execute(name)
            ]
            
            # Sort by priority and limit
            enabled_sources.sort(key=lambda x: x[1]['priority'])
            enabled_sources = enabled_sources[:max_sources]
            
            # Scrape concurrently with semaphore
            semaphore = asyncio.Semaphore(self.config['max_concurrent_requests'])
            tasks = [
                self._scrape_single_source(semaphore, source_name, source_config, phone_query)
                for source_name, source_config in enabled_sources
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            scraped_reviews = []
            source_urls = []
            scrape_errors = []
            
            for i, result in enumerate(results):
                source_name = enabled_sources[i][0]
                
                if isinstance(result, Exception):
                    logger.error(f"Scraping failed for {source_name}: {result}")
                    self.circuit_breaker.record_failure(source_name)
                    scrape_errors.append(f"{source_name}: {str(result)}")
                elif result:
                    scraped_reviews.extend(result.get('reviews', []))
                    source_urls.extend(result.get('urls', []))
                    self.circuit_breaker.record_success(source_name)
                    logger.info(f"Successfully scraped {len(result.get('reviews', []))} reviews from {source_name}")
            
            # Combine and process results
            phone_data = self._process_scraped_data(phone_query, scraped_reviews, source_urls, scrape_errors)
            
            return phone_data
            
        finally:
            await self._cleanup_session()
    
    async def _scrape_single_source(self, semaphore: asyncio.Semaphore, 
                                  source_name: str, source_config: Dict, 
                                  phone_query: str) -> Optional[Dict]:
        """Scrape a single source with proper error handling"""
        
        async with semaphore:
            try:
                # Rate limiting
                search_url = source_config['search_url'].format(query=quote_plus(phone_query))
                await self.rate_limiter.wait_if_needed(search_url)
                
                logger.info(f"Scraping {source_name} for: {phone_query}")
                
                if source_config.get('requires_js', False):
                    return await self._scrape_with_selenium(source_name, source_config, search_url, phone_query)
                else:
                    return await self._scrape_with_aiohttp(source_name, source_config, search_url, phone_query)
                    
            except Exception as e:
                logger.error(f"Error scraping {source_name}: {str(e)}")
                raise
    
    async def _scrape_with_selenium(self, source_name: str, source_config: Dict, 
                                  search_url: str, phone_query: str) -> Optional[Dict]:
        """Scrape using Selenium for JavaScript-heavy sites"""
        
        if not self.driver:
            self._init_selenium_driver()
        
        try:
            self.driver.get(search_url)
            
            # Wait for search results
            wait = WebDriverWait(self.driver, 10)
            search_results = wait.until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, source_config['selectors']['search_results']))
            )
            
            # Find best matching result
            best_match = self._find_best_phone_match(search_results, phone_query)
            if not best_match:
                return None
            
            # Navigate to phone page
            phone_url = best_match.get_attribute('href')
            if not phone_url.startswith('http'):
                phone_url = urljoin(source_config['base_url'], phone_url)
            
            self.driver.get(phone_url)
            
            # Scrape phone details
            return self._extract_phone_data_selenium(source_name, source_config, phone_url, phone_query)
            
        except TimeoutException:
            logger.warning(f"Timeout scraping {source_name}")
            return None
        except WebDriverException as e:
            logger.error(f"Selenium error for {source_name}: {str(e)}")
            return None
    
    async def _scrape_with_aiohttp(self, source_name: str, source_config: Dict, 
                                 search_url: str, phone_query: str) -> Optional[Dict]:
        """Scrape using aiohttp for static content"""
        
        try:
            headers = {
                'User-Agent': self.config['user_agent'],
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
            
            async with self.session.get(search_url, headers=headers, timeout=self.config['request_timeout']) as response:
                if response.status != 200:
                    logger.warning(f"HTTP {response.status} for {source_name}")
                    return None
                
                content = await response.text()
                soup = BeautifulSoup(content, 'html.parser')
                
                # Find search results
                search_results = soup.select(source_config['selectors']['search_results'])
                if not search_results:
                    logger.warning(f"No search results found for {source_name}")
                    return None
                
                # Find best matching result
                best_match = self._find_best_phone_match_soup(search_results, phone_query)
                if not best_match:
                    return None
                
                # Get phone URL
                phone_url = best_match.get('href')
                if not phone_url.startswith('http'):
                    phone_url = urljoin(source_config['base_url'], phone_url)
                
                # Scrape phone page
                await self.rate_limiter.wait_if_needed(phone_url)
                
                async with self.session.get(phone_url, headers=headers, timeout=self.config['request_timeout']) as phone_response:
                    if phone_response.status != 200:
                        return None
                    
                    phone_content = await phone_response.text()
                    return self._extract_phone_data_soup(source_name, source_config, phone_content, phone_url, phone_query)
                    
        except asyncio.TimeoutError:
            logger.warning(f"Timeout scraping {source_name}")
            return None
        except Exception as e:
            logger.error(f"HTTP error for {source_name}: {str(e)}")
            return None
    
    def _extract_phone_data_selenium(self, source_name: str, source_config: Dict, 
                                   phone_url: str, phone_query: str) -> Dict:
        """Extract phone data using Selenium"""
        
        reviews = []
        
        try:
            # Extract title
            title_element = self.driver.find_element(By.CSS_SELECTOR, source_config['selectors']['phone_title'])
            title = title_element.text.strip() if title_element else phone_query
            
            # Extract rating
            rating = None
            try:
                rating_element = self.driver.find_element(By.CSS_SELECTOR, source_config['selectors']['rating'])
                rating_text = rating_element.text.strip()
                rating = float(re.search(r'(\d+\.?\d*)', rating_text).group(1)) if rating_text else None
            except:
                pass
            
            # Extract specifications (source-specific logic)
            specs = self._extract_specs_selenium(source_name, source_config)
            
            # Extract reviews (if available)
            review_elements = self.driver.find_elements(By.CSS_SELECTOR, source_config['selectors'].get('reviews', ''))
            for review_elem in review_elements[:5]:  # Limit to 5 reviews per source
                review_data = self._parse_review_element_selenium(review_elem, source_name)
                if review_data:
                    reviews.append(ScrapedReview(
                        phone_model=phone_query,
                        source=source_name,
                        title=review_data.get('title', ''),
                        content=review_data.get('content', ''),
                        rating=review_data.get('rating'),
                        author=review_data.get('author'),
                        date=review_data.get('date'),
                        pros=review_data.get('pros', []),
                        cons=review_data.get('cons', []),
                        url=phone_url,
                        confidence=0.8,
                        scraped_at=datetime.now().isoformat()
                    ))
            
            return {
                'reviews': reviews,
                'urls': [phone_url],
                'title': title,
                'rating': rating,
                'specifications': specs
            }
            
        except Exception as e:
            logger.error(f"Error extracting data from {source_name}: {str(e)}")
            return {'reviews': [], 'urls': [phone_url]}
    
    def _extract_phone_data_soup(self, source_name: str, source_config: Dict, 
                               content: str, phone_url: str, phone_query: str) -> Dict:
        """Extract phone data using BeautifulSoup"""
        
        soup = BeautifulSoup(content, 'html.parser')
        reviews = []
        
        try:
            # Extract title
            title_element = soup.select_one(source_config['selectors']['phone_title'])
            title = title_element.get_text(strip=True) if title_element else phone_query
            
            # Extract rating
            rating = None
            rating_element = soup.select_one(source_config['selectors'].get('rating', ''))
            if rating_element:
                rating_text = rating_element.get_text(strip=True)
                match = re.search(r'(\d+\.?\d*)', rating_text)
                rating = float(match.group(1)) if match else None
            
            # Extract content
            content_element = soup.select_one(source_config['selectors'].get('content', ''))
            content_text = content_element.get_text(strip=True) if content_element else ''
            
            # Extract pros/cons if available
            pros = []
            cons = []
            
            pros_elements = soup.select(source_config['selectors'].get('pros', ''))
            for pro_elem in pros_elements:
                pros.append(pro_elem.get_text(strip=True))
            
            cons_elements = soup.select(source_config['selectors'].get('cons', ''))
            for con_elem in cons_elements:
                cons.append(con_elem.get_text(strip=True))
            
            # Create main review
            if content_text or pros or cons:
                reviews.append(ScrapedReview(
                    phone_model=phone_query,
                    source=source_name,
                    title=title,
                    content=content_text[:1000],  # Limit content length
                    rating=rating,
                    author=f"{source_name} Editorial",
                    date=datetime.now().strftime('%Y-%m-%d'),
                    pros=pros[:5],  # Limit pros
                    cons=cons[:5],  # Limit cons
                    url=phone_url,
                    confidence=0.7,
                    scraped_at=datetime.now().isoformat()
                ))
            
            return {
                'reviews': reviews,
                'urls': [phone_url],
                'title': title,
                'rating': rating
            }
            
        except Exception as e:
            logger.error(f"Error parsing {source_name}: {str(e)}")
            return {'reviews': [], 'urls': [phone_url]}
    
    def _process_scraped_data(self, phone_query: str, reviews: List[ScrapedReview], 
                            urls: List[str], errors: List[str]) -> ScrapedPhoneData:
        """Process and combine scraped data"""
        
        # Calculate overall rating
        ratings = [r.rating for r in reviews if r.rating is not None]
        overall_rating = sum(ratings) / len(ratings) if ratings else None
        
        # Extract phone brand
        brand = self._extract_brand_from_query(phone_query)
        
        # Aggregate pros and cons
        all_pros = []
        all_cons = []
        for review in reviews:
            all_pros.extend(review.pros)
            all_cons.extend(review.cons)
        
        # Deduplicate and limit
        unique_pros = list(dict.fromkeys(all_pros))[:10]
        unique_cons = list(dict.fromkeys(all_cons))[:10]
        
        return ScrapedPhoneData(
            model=phone_query,
            brand=brand or 'Unknown',
            specifications={},  # Will be enhanced in next step
            reviews=reviews,
            overall_rating=overall_rating,
            review_count=len(reviews),
            price_info={},  # Will be enhanced with pricing APIs
            availability={},
            images=[],
            source_urls=urls,
            scrape_metadata={
                'scrape_timestamp': datetime.now().isoformat(),
                'sources_scraped': len(set(r.source for r in reviews)),
                'scrape_errors': errors,
                'total_content_length': sum(len(r.content) for r in reviews)
            }
        )
    
    # Utility methods
    async def _init_session(self):
        """Initialize aiohttp session"""
        if not self.session:
            connector = aiohttp.TCPConnector(limit=10, ttl_dns_cache=300, use_dns_cache=True)
            timeout = aiohttp.ClientTimeout(total=self.config['request_timeout'])
            self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
    
    async def _cleanup_session(self):
        """Cleanup aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None
        
        if self.driver:
            self.driver.quit()
            self.driver = None
    
    def _init_selenium_driver(self):
        """Initialize Selenium WebDriver"""
        if not self.config.get('use_selenium', True):
            return
        
        try:
            options = Options()
            if self.config.get('headless', True):
                options.add_argument('--headless')
            
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-gpu')
            options.add_argument('--disable-web-security')
            options.add_argument('--allow-running-insecure-content')
            options.add_argument(f'--user-agent={self.config["user_agent"]}')
            
            self.driver = webdriver.Chrome(options=options)
            self.driver.set_page_load_timeout(self.config['request_timeout'])
            
        except Exception as e:
            logger.warning(f"Could not initialize Selenium driver: {e}")
            self.driver = None
    
    def _find_best_phone_match(self, results, phone_query: str):
        """Find best matching phone from Selenium results"""
        phone_query_lower = phone_query.lower()
        best_match = None
        best_score = 0
        
        for result in results:
            try:
                text = result.text.lower()
                score = self._calculate_match_score(text, phone_query_lower)
                if score > best_score:
                    best_score = score
                    best_match = result
            except:
                continue
        
        return best_match if best_score > 0.3 else None
    
    def _find_best_phone_match_soup(self, results, phone_query: str):
        """Find best matching phone from BeautifulSoup results"""
        phone_query_lower = phone_query.lower()
        best_match = None
        best_score = 0
        
        for result in results:
            try:
                text = result.get_text().lower()
                score = self._calculate_match_score(text, phone_query_lower)
                if score > best_score:
                    best_score = score
                    best_match = result
            except:
                continue
        
        return best_match if best_score > 0.3 else None
    
    def _calculate_match_score(self, text: str, query: str) -> float:
        """Calculate how well text matches the phone query"""
        query_words = set(query.split())
        text_words = set(text.split())
        
        # Exact match gets highest score
        if query in text:
            return 1.0
        
        # Word overlap score
        common_words = query_words.intersection(text_words)
        if not query_words:
            return 0.0
        
        overlap_score = len(common_words) / len(query_words)
        return overlap_score
    
    def _extract_brand_from_query(self, query: str) -> Optional[str]:
        """Extract brand from phone query"""
        brands = {
            'apple': ['iphone', 'apple'],
            'samsung': ['samsung', 'galaxy'],
            'google': ['google', 'pixel'],
            'oneplus': ['oneplus', 'one plus'],
            'xiaomi': ['xiaomi', 'mi', 'redmi'],
            'huawei': ['huawei', 'honor'],
            'oppo': ['oppo'],
            'vivo': ['vivo'],
            'nothing': ['nothing'],
            'motorola': ['motorola', 'moto'],
            'sony': ['sony', 'xperia']
        }
        
        query_lower = query.lower()
        for brand, keywords in brands.items():
            if any(keyword in query_lower for keyword in keywords):
                return brand.title()
        
        return None
    
    def _extract_specs_selenium(self, source_name: str, source_config: Dict) -> Dict:
        """Extract specifications using Selenium (source-specific logic)"""
        specs = {}
        
        try:
            spec_elements = self.driver.find_elements(By.CSS_SELECTOR, source_config['selectors'].get('specs', ''))
            
            for spec_elem in spec_elements:
                # This would need source-specific parsing logic
                # For now, just extract basic text
                spec_text = spec_elem.text.strip()
                if ':' in spec_text:
                    key, value = spec_text.split(':', 1)
                    specs[key.strip()] = value.strip()
        except:
            pass
        
        return specs
    
    def _parse_review_element_selenium(self, review_elem, source_name: str) -> Optional[Dict]:
        """Parse individual review element using Selenium"""
        try:
            review_data = {
                'title': '',
                'content': '',
                'rating': None,
                'author': '',
                'date': '',
                'pros': [],
                'cons': []
            }
            
            # Extract review text (simplified)
            review_data['content'] = review_elem.text.strip()[:500]
            
            return review_data
        except:
            return None

# Factory function
def create_enhanced_web_scraper(config=None):
    """Create configured enhanced web scraper"""
    return EnhancedWebScraper(config=config)