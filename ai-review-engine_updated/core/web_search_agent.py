"""
Web Search Agent for AI Phone Review Engine
Handles external web searches for phones not available in the local dataset
"""

import asyncio
import json
import re
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus, urljoin, urlparse
import concurrent.futures
from threading import Lock

# Import existing components
from .smart_search import SmartPhoneSearch, SearchQuery
from scrapers.base_scraper import BaseScraper

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class WebSearchResult:
    """Structure for web search results"""
    phone_model: str
    source: str
    url: str
    title: str
    snippet: str
    rating: Optional[float]
    review_count: Optional[int]
    price: Optional[str]
    sentiment_preview: str
    confidence: float
    scraped_at: str

@dataclass
class PhoneInfo:
    """Comprehensive phone information from web search"""
    model: str
    brand: str
    specifications: Dict[str, Any]
    reviews: List[Dict[str, Any]]
    overall_rating: Optional[float]
    price_range: Dict[str, str]
    availability: Dict[str, bool]
    key_features: List[str]
    pros: List[str]
    cons: List[str]
    similar_phones: List[str]
    sources: List[str]

class WebSearchAgent:
    """
    Advanced web search agent for phone research
    Integrates with the existing AI system to provide external data
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the web search agent"""
        self.smart_search = SmartPhoneSearch()
        self.search_lock = Lock()
        
        # Default configuration
        self.config = config or {
            'max_concurrent_searches': 3,
            'search_timeout': 30,
            'max_results_per_source': 5,
            'min_confidence_threshold': 0.6,
            'rate_limit_delay': 2.0,
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Search sources configuration
        self.search_sources = {
            'gsmarena': {
                'base_url': 'https://www.gsmarena.com/',
                'search_url': 'https://www.gsmarena.com/results.php3?sQuickSearch=yes&sName={query}',
                'enabled': True,
                'priority': 1,
                'parser': self._parse_gsmarena_results
            },
            'phonearena': {
                'base_url': 'https://www.phonearena.com/',
                'search_url': 'https://www.phonearena.com/search?term={query}',
                'enabled': True,
                'priority': 2,
                'parser': self._parse_phonearena_results
            },
            'cnet': {
                'base_url': 'https://www.cnet.com/',
                'search_url': 'https://www.cnet.com/search/?q={query}+phone+review',
                'enabled': True,
                'priority': 3,
                'parser': self._parse_cnet_results
            },
            'techcrunch': {
                'base_url': 'https://techcrunch.com/',
                'search_url': 'https://search.techcrunch.com/search?q={query}+phone+review',
                'enabled': True,
                'priority': 4,
                'parser': self._parse_techcrunch_results
            },
            'google_shopping': {
                'base_url': 'https://www.google.com/search',
                'search_url': 'https://www.google.com/search?q={query}+reviews+specs+price&tbm=shop',
                'enabled': True,
                'priority': 5,
                'parser': self._parse_google_shopping_results
            }
        }
        
        # Common headers for requests
        self.headers = {
            'User-Agent': self.config['user_agent'],
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        # Cache for search results
        self.search_cache = {}
        self.cache_expiry = 3600  # 1 hour
        
    def search_phone_external(self, query: str, max_sources: int = 3) -> Dict[str, Any]:
        """
        Main method to search for phone information externally
        
        Args:
            query: Search query (phone model or natural language)
            max_sources: Maximum number of sources to search
            
        Returns:
            Comprehensive phone information from web search
        """
        
        # Parse the query using existing smart search
        parsed_query = self.smart_search.parse_query(query)
        
        logger.info(f"External search initiated for: {parsed_query.phone_model}")
        logger.info(f"Search intent: {parsed_query.intent}, Confidence: {parsed_query.confidence}")
        
        # Check cache first
        cache_key = f"{parsed_query.phone_model}_{parsed_query.intent}"
        if self._is_cache_valid(cache_key):
            logger.info("Returning cached results")
            return self.search_cache[cache_key]['data']
        
        # Perform multi-source search
        search_results = self._perform_multi_source_search(
            parsed_query, 
            max_sources=max_sources
        )
        
        # Aggregate and analyze results
        phone_info = self._aggregate_search_results(search_results, parsed_query)
        
        # Cache results
        self._cache_results(cache_key, phone_info)
        
        # Format for integration with existing system
        return self._format_for_system_integration(phone_info, parsed_query)
    
    def _perform_multi_source_search(self, parsed_query: SearchQuery, max_sources: int) -> List[WebSearchResult]:
        """Perform concurrent searches across multiple sources"""
        
        results = []
        
        # Select top priority sources
        active_sources = sorted(
            [(name, config) for name, config in self.search_sources.items() if config['enabled']], 
            key=lambda x: x[1]['priority']
        )[:max_sources]
        
        # Concurrent search execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config['max_concurrent_searches']) as executor:
            
            # Submit search tasks
            future_to_source = {
                executor.submit(self._search_single_source, source_name, source_config, parsed_query): source_name
                for source_name, source_config in active_sources
            }
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_source, timeout=self.config['search_timeout']):
                source_name = future_to_source[future]
                
                try:
                    source_results = future.result()
                    if source_results:
                        results.extend(source_results)
                        logger.info(f"Retrieved {len(source_results)} results from {source_name}")
                    else:
                        logger.warning(f"No results from {source_name}")
                        
                except Exception as e:
                    logger.error(f"Error searching {source_name}: {str(e)}")
        
        return results
    
    def _search_single_source(self, source_name: str, source_config: Dict, parsed_query: SearchQuery) -> List[WebSearchResult]:
        """Search a single source for phone information"""
        
        try:
            # Rate limiting
            time.sleep(self.config['rate_limit_delay'])
            
            # Build search URL
            search_url = source_config['search_url'].format(
                query=quote_plus(f"{parsed_query.phone_model} {' '.join(parsed_query.aspects)}")
            )
            
            # Make request
            response = requests.get(search_url, headers=self.headers, timeout=15)
            response.raise_for_status()
            
            # Parse results using source-specific parser
            if source_config['parser']:
                results = source_config['parser'](response.text, parsed_query, source_name)
                return results[:self.config['max_results_per_source']]
            
        except Exception as e:
            logger.error(f"Error in {source_name} search: {str(e)}")
            
        return []
    
    # Source-specific parsers
    def _parse_gsmarena_results(self, html: str, query: SearchQuery, source: str) -> List[WebSearchResult]:
        """Parse GSMArena search results"""
        results = []
        soup = BeautifulSoup(html, 'html.parser')
        
        try:
            # GSMArena-specific parsing
            phone_divs = soup.find_all('div', class_='makers')
            
            for div in phone_divs[:3]:  # Limit to top 3 results
                try:
                    title_link = div.find('a')
                    if title_link:
                        title = title_link.get_text(strip=True)
                        url = urljoin('https://www.gsmarena.com/', title_link.get('href'))
                        
                        # Extract additional info if available
                        snippet = div.get_text(strip=True)[:200]
                        
                        results.append(WebSearchResult(
                            phone_model=query.phone_model,
                            source=source,
                            url=url,
                            title=title,
                            snippet=snippet,
                            rating=None,  # GSMArena doesn't show ratings in search
                            review_count=None,
                            price=None,
                            sentiment_preview="specs_focused",
                            confidence=0.9,  # High confidence for GSMArena
                            scraped_at=datetime.now().isoformat()
                        ))
                        
                except Exception as e:
                    logger.error(f"Error parsing GSMArena result: {str(e)}")
                    continue
        
        except Exception as e:
            logger.error(f"Error in GSMArena parser: {str(e)}")
        
        return results
    
    def _parse_phonearena_results(self, html: str, query: SearchQuery, source: str) -> List[WebSearchResult]:
        """Parse PhoneArena search results"""
        results = []
        soup = BeautifulSoup(html, 'html.parser')
        
        try:
            # PhoneArena-specific parsing
            articles = soup.find_all('article', class_='s-post')
            
            for article in articles[:3]:
                try:
                    title_link = article.find('h2', class_='s-post-title').find('a')
                    if title_link:
                        title = title_link.get_text(strip=True)
                        url = title_link.get('href')
                        
                        # Get snippet
                        snippet_elem = article.find('div', class_='s-post-summary')
                        snippet = snippet_elem.get_text(strip=True)[:200] if snippet_elem else ""
                        
                        # Try to extract rating if present
                        rating = None
                        rating_elem = article.find('div', class_='rating')
                        if rating_elem:
                            rating_text = rating_elem.get_text(strip=True)
                            rating = self._extract_rating(rating_text)
                        
                        results.append(WebSearchResult(
                            phone_model=query.phone_model,
                            source=source,
                            url=url,
                            title=title,
                            snippet=snippet,
                            rating=rating,
                            review_count=None,
                            price=None,
                            sentiment_preview="review_focused",
                            confidence=0.8,
                            scraped_at=datetime.now().isoformat()
                        ))
                        
                except Exception as e:
                    logger.error(f"Error parsing PhoneArena result: {str(e)}")
                    continue
        
        except Exception as e:
            logger.error(f"Error in PhoneArena parser: {str(e)}")
        
        return results
    
    def _parse_cnet_results(self, html: str, query: SearchQuery, source: str) -> List[WebSearchResult]:
        """Parse CNET search results"""
        results = []
        soup = BeautifulSoup(html, 'html.parser')
        
        try:
            # CNET-specific parsing
            search_results = soup.find_all('div', class_='searchResult')
            
            for result in search_results[:3]:
                try:
                    title_link = result.find('h3').find('a')
                    if title_link and 'phone' in title_link.get_text().lower():
                        title = title_link.get_text(strip=True)
                        url = title_link.get('href')
                        
                        snippet_elem = result.find('p', class_='dek')
                        snippet = snippet_elem.get_text(strip=True)[:200] if snippet_elem else ""
                        
                        # Try to extract rating
                        rating = None
                        rating_elem = result.find('div', class_='c-productRating')
                        if rating_elem:
                            rating_text = rating_elem.get_text(strip=True)
                            rating = self._extract_rating(rating_text)
                        
                        results.append(WebSearchResult(
                            phone_model=query.phone_model,
                            source=source,
                            url=url,
                            title=title,
                            snippet=snippet,
                            rating=rating,
                            review_count=None,
                            price=None,
                            sentiment_preview="professional_review",
                            confidence=0.85,
                            scraped_at=datetime.now().isoformat()
                        ))
                        
                except Exception as e:
                    logger.error(f"Error parsing CNET result: {str(e)}")
                    continue
        
        except Exception as e:
            logger.error(f"Error in CNET parser: {str(e)}")
        
        return results
    
    def _parse_techcrunch_results(self, html: str, query: SearchQuery, source: str) -> List[WebSearchResult]:
        """Parse TechCrunch search results"""
        results = []
        soup = BeautifulSoup(html, 'html.parser')
        
        try:
            articles = soup.find_all('article', class_='post-block')
            
            for article in articles[:2]:  # Fewer results from TechCrunch
                try:
                    title_link = article.find('h2', class_='post-block__title').find('a')
                    if title_link:
                        title = title_link.get_text(strip=True)
                        url = title_link.get('href')
                        
                        snippet_elem = article.find('div', class_='post-block__content')
                        snippet = snippet_elem.get_text(strip=True)[:200] if snippet_elem else ""
                        
                        results.append(WebSearchResult(
                            phone_model=query.phone_model,
                            source=source,
                            url=url,
                            title=title,
                            snippet=snippet,
                            rating=None,
                            review_count=None,
                            price=None,
                            sentiment_preview="tech_news",
                            confidence=0.75,
                            scraped_at=datetime.now().isoformat()
                        ))
                        
                except Exception as e:
                    logger.error(f"Error parsing TechCrunch result: {str(e)}")
                    continue
        
        except Exception as e:
            logger.error(f"Error in TechCrunch parser: {str(e)}")
        
        return results
    
    def _parse_google_shopping_results(self, html: str, query: SearchQuery, source: str) -> List[WebSearchResult]:
        """Parse Google Shopping search results for price information"""
        results = []
        soup = BeautifulSoup(html, 'html.parser')
        
        try:
            # Google Shopping specific parsing (simplified - actual implementation would be more complex)
            product_divs = soup.find_all('div', class_='sh-dgr__content')
            
            for div in product_divs[:2]:
                try:
                    price_elem = div.find('span', class_='a8Pemb')
                    title_elem = div.find('h3')
                    
                    if price_elem and title_elem:
                        price = price_elem.get_text(strip=True)
                        title = title_elem.get_text(strip=True)
                        
                        results.append(WebSearchResult(
                            phone_model=query.phone_model,
                            source=source,
                            url='#',  # Google Shopping results don't have direct URLs in this context
                            title=title,
                            snippet=f"Available for {price}",
                            rating=None,
                            review_count=None,
                            price=price,
                            sentiment_preview="price_info",
                            confidence=0.7,
                            scraped_at=datetime.now().isoformat()
                        ))
                        
                except Exception as e:
                    logger.error(f"Error parsing Google Shopping result: {str(e)}")
                    continue
        
        except Exception as e:
            logger.error(f"Error in Google Shopping parser: {str(e)}")
        
        return results
    
    def _aggregate_search_results(self, search_results: List[WebSearchResult], query: SearchQuery) -> PhoneInfo:
        """Aggregate results from multiple sources into comprehensive phone info"""
        
        if not search_results:
            return self._create_empty_phone_info(query.phone_model)
        
        # Extract information from results
        all_reviews = []
        all_sources = []
        ratings = []
        prices = []
        features = set()
        
        for result in search_results:
            all_sources.append(result.source)
            
            if result.rating:
                ratings.append(result.rating)
            
            if result.price:
                prices.append(result.price)
            
            # Convert search result to review format
            review = {
                'source': result.source,
                'title': result.title,
                'content': result.snippet,
                'url': result.url,
                'rating': result.rating,
                'scraped_at': result.scraped_at,
                'sentiment_preview': result.sentiment_preview
            }
            all_reviews.append(review)
            
            # Extract features from titles and snippets
            features.update(self._extract_features_from_text(f"{result.title} {result.snippet}"))
        
        # Calculate aggregated data
        overall_rating = sum(ratings) / len(ratings) if ratings else None
        price_range = self._analyze_price_range(prices)
        
        # Create comprehensive phone info
        phone_info = PhoneInfo(
            model=query.phone_model,
            brand=query.brand or 'Unknown',
            specifications={
                'extracted_from_web': True,
                'search_confidence': query.confidence,
                'features_mentioned': list(features)
            },
            reviews=all_reviews,
            overall_rating=overall_rating,
            price_range=price_range,
            availability={
                'web_available': True,
                'sources_count': len(set(all_sources))
            },
            key_features=list(features)[:10],  # Top 10 features
            pros=self._extract_pros_cons(search_results, 'pros'),
            cons=self._extract_pros_cons(search_results, 'cons'),
            similar_phones=[],  # Would require additional processing
            sources=list(set(all_sources))
        )
        
        return phone_info
    
    def _format_for_system_integration(self, phone_info: PhoneInfo, query: SearchQuery) -> Dict[str, Any]:
        """Format the phone info for integration with the existing AI system"""
        
        # Convert to format compatible with existing system
        formatted_data = {
            'phone_found': True,
            'source': 'web_search',
            'search_query': query.original_query,
            'confidence': query.confidence,
            'phone_data': {
                'product_name': phone_info.model,
                'brand': phone_info.brand,
                'overall_rating': phone_info.overall_rating,
                'review_count': len(phone_info.reviews),
                'web_sources': phone_info.sources,
                'key_features': phone_info.key_features,
                'pros': phone_info.pros,
                'cons': phone_info.cons,
                'price_info': phone_info.price_range,
                'availability': phone_info.availability
            },
            'reviews': phone_info.reviews,
            'analysis': {
                'sentiment_analysis': self._analyze_web_sentiment(phone_info.reviews),
                'feature_analysis': self._analyze_features(phone_info.key_features),
                'recommendation_score': self._calculate_recommendation_score(phone_info),
                'data_quality': self._assess_data_quality(phone_info)
            },
            'recommendations': {
                'should_buy': phone_info.overall_rating and phone_info.overall_rating >= 4.0,
                'confidence_level': 'medium',  # Web data has medium confidence
                'key_considerations': phone_info.pros[:3] + phone_info.cons[:3],
                'next_steps': [
                    'Check official website for latest pricing',
                    'Read detailed reviews from multiple sources',
                    'Compare with similar phones in database',
                    'Verify availability in your region'
                ]
            },
            'metadata': {
                'scraped_at': datetime.now().isoformat(),
                'sources_used': phone_info.sources,
                'search_method': 'multi_source_web_search',
                'data_freshness': 'real_time'
            }
        }
        
        return formatted_data
    
    # Helper methods
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached result is still valid"""
        if cache_key not in self.search_cache:
            return False
        
        cache_time = self.search_cache[cache_key]['timestamp']
        return (time.time() - cache_time) < self.cache_expiry
    
    def _cache_results(self, cache_key: str, data: Any) -> None:
        """Cache search results"""
        self.search_cache[cache_key] = {
            'data': data,
            'timestamp': time.time()
        }
    
    def _extract_rating(self, text: str) -> Optional[float]:
        """Extract numerical rating from text"""
        try:
            # Try to find rating patterns like "4.5/5", "8.5/10", "4.5 stars"
            rating_patterns = [
                r'(\d+\.?\d*)/5',
                r'(\d+\.?\d*)/10',
                r'(\d+\.?\d*)\s*(?:stars?|★)',
                r'(\d+\.?\d*)\s*out\s*of\s*(?:5|10)',
                r'rating:\s*(\d+\.?\d*)'
            ]
            
            for pattern in rating_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    rating = float(match.group(1))
                    # Normalize to 5-point scale
                    if '/10' in pattern or 'out of 10' in pattern.lower():
                        rating = rating / 2  # Convert 10-point to 5-point scale
                    return min(rating, 5.0)  # Cap at 5.0
        except:
            pass
        return None
    
    def _extract_features_from_text(self, text: str) -> set:
        """Extract phone features from text"""
        features = set()
        
        feature_keywords = [
            'camera', 'display', 'screen', 'battery', 'processor', 'memory', 'storage',
            'RAM', 'performance', 'design', '5G', 'wireless charging', 'fast charging',
            'waterproof', 'face recognition', 'fingerprint', 'dual sim', 'headphone jack'
        ]
        
        text_lower = text.lower()
        for keyword in feature_keywords:
            if keyword.lower() in text_lower:
                features.add(keyword)
        
        return features
    
    def _analyze_price_range(self, prices: List[str]) -> Dict[str, str]:
        """Analyze price information from search results"""
        if not prices:
            return {'status': 'not_found'}
        
        # Extract numerical values from price strings
        price_values = []
        for price_str in prices:
            # Extract numbers from price strings like "$599", "€799", "₹49,999"
            numbers = re.findall(r'\d+(?:,\d+)*(?:\.\d+)?', price_str.replace(',', ''))
            if numbers:
                try:
                    price_values.append(float(numbers[0]))
                except:
                    continue
        
        if price_values:
            min_price = min(price_values)
            max_price = max(price_values)
            avg_price = sum(price_values) / len(price_values)
            
            return {
                'status': 'found',
                'min_price': f"${min_price:.0f}",
                'max_price': f"${max_price:.0f}",
                'avg_price': f"${avg_price:.0f}",
                'price_sources': len(price_values)
            }
        
        return {'status': 'found_but_unclear', 'raw_prices': prices}
    
    def _extract_pros_cons(self, search_results: List[WebSearchResult], type: str) -> List[str]:
        """Extract pros or cons from search results"""
        items = []
        
        positive_indicators = ['excellent', 'great', 'amazing', 'outstanding', 'impressive', 'solid', 'good']
        negative_indicators = ['poor', 'bad', 'disappointing', 'weak', 'lacks', 'missing', 'slow']
        
        for result in search_results:
            text = f"{result.title} {result.snippet}".lower()
            
            if type == 'pros':
                for indicator in positive_indicators:
                    if indicator in text:
                        # Extract sentence containing the positive indicator
                        sentences = result.snippet.split('.')
                        for sentence in sentences:
                            if indicator in sentence.lower():
                                items.append(sentence.strip())
                                break
            else:  # cons
                for indicator in negative_indicators:
                    if indicator in text:
                        sentences = result.snippet.split('.')
                        for sentence in sentences:
                            if indicator in sentence.lower():
                                items.append(sentence.strip())
                                break
        
        # Remove duplicates and limit
        return list(set(items))[:5]
    
    def _analyze_web_sentiment(self, reviews: List[Dict]) -> Dict[str, Any]:
        """Analyze sentiment from web search results"""
        positive_count = 0
        neutral_count = 0
        negative_count = 0
        
        positive_words = ['excellent', 'amazing', 'great', 'outstanding', 'impressive', 'love', 'perfect']
        negative_words = ['poor', 'bad', 'disappointing', 'terrible', 'awful', 'hate', 'worst']
        
        for review in reviews:
            content = review.get('content', '').lower()
            
            pos_score = sum(1 for word in positive_words if word in content)
            neg_score = sum(1 for word in negative_words if word in content)
            
            if pos_score > neg_score:
                positive_count += 1
            elif neg_score > pos_score:
                negative_count += 1
            else:
                neutral_count += 1
        
        total = len(reviews)
        if total == 0:
            return {'status': 'no_data'}
        
        return {
            'positive_percentage': (positive_count / total) * 100,
            'negative_percentage': (negative_count / total) * 100,
            'neutral_percentage': (neutral_count / total) * 100,
            'overall_sentiment': 'positive' if positive_count > negative_count else 'negative' if negative_count > positive_count else 'neutral',
            'confidence': 'medium',  # Web sentiment analysis has medium confidence
            'total_reviews_analyzed': total
        }
    
    def _analyze_features(self, features: List[str]) -> Dict[str, Any]:
        """Analyze key features mentioned"""
        feature_categories = {
            'camera': ['camera', 'photo', 'video'],
            'performance': ['processor', 'RAM', 'performance', 'speed'],
            'display': ['display', 'screen', 'resolution'],
            'battery': ['battery', 'charging'],
            'connectivity': ['5G', '4G', 'wifi', 'bluetooth'],
            'design': ['design', 'build', 'waterproof']
        }
        
        categorized_features = {}
        for category, keywords in feature_categories.items():
            matching_features = [f for f in features if any(kw in f.lower() for kw in keywords)]
            if matching_features:
                categorized_features[category] = matching_features
        
        return {
            'total_features_found': len(features),
            'categorized_features': categorized_features,
            'most_mentioned_category': max(categorized_features.keys(), key=lambda k: len(categorized_features[k])) if categorized_features else None
        }
    
    def _calculate_recommendation_score(self, phone_info: PhoneInfo) -> float:
        """Calculate a recommendation score based on available data"""
        score = 0.5  # Base score
        
        # Adjust based on rating
        if phone_info.overall_rating:
            score += (phone_info.overall_rating - 3.0) * 0.2  # Scale rating impact
        
        # Adjust based on number of sources
        source_count = len(phone_info.sources)
        score += min(source_count * 0.1, 0.3)  # More sources = higher confidence
        
        # Adjust based on review count
        review_count = len(phone_info.reviews)
        score += min(review_count * 0.02, 0.2)  # More reviews = higher confidence
        
        return min(max(score, 0.0), 1.0)  # Clamp between 0 and 1
    
    def _assess_data_quality(self, phone_info: PhoneInfo) -> Dict[str, Any]:
        """Assess the quality of web-scraped data"""
        quality_score = 0.5  # Base quality
        
        factors = []
        
        # Check data completeness
        if phone_info.overall_rating:
            quality_score += 0.2
            factors.append("Rating available")
        
        if phone_info.price_range.get('status') == 'found':
            quality_score += 0.15
            factors.append("Price information found")
        
        if len(phone_info.reviews) >= 3:
            quality_score += 0.15
            factors.append("Multiple sources")
        
        if phone_info.key_features:
            quality_score += 0.1
            factors.append("Features identified")
        
        return {
            'quality_score': min(quality_score, 1.0),
            'quality_level': 'high' if quality_score > 0.8 else 'medium' if quality_score > 0.6 else 'low',
            'positive_factors': factors,
            'limitations': [
                "Web-scraped data may not be as comprehensive as database entries",
                "Real-time data may have inconsistencies",
                "Limited to public web sources"
            ]
        }
    
    def _create_empty_phone_info(self, phone_model: str) -> PhoneInfo:
        """Create empty phone info structure when no results found"""
        return PhoneInfo(
            model=phone_model,
            brand='Unknown',
            specifications={},
            reviews=[],
            overall_rating=None,
            price_range={'status': 'not_found'},
            availability={'web_available': False},
            key_features=[],
            pros=[],
            cons=[],
            similar_phones=[],
            sources=[]
        )

# Integration function for the main system
def integrate_web_search_with_system(user_query: str, local_search_result: Dict = None) -> Dict[str, Any]:
    """
    Integration function to be called from the main system
    
    Args:
        user_query: User's search query
        local_search_result: Result from local database search (None if not found)
        
    Returns:
        Combined result with web search data if needed
    """
    
    # If local data is sufficient, return it
    if local_search_result and local_search_result.get('confidence', 0) > 0.8:
        local_search_result['source'] = 'local_database'
        local_search_result['web_search_performed'] = False
        return local_search_result
    
    # Perform web search for additional/alternative data
    web_agent = WebSearchAgent()
    web_result = web_agent.search_phone_external(user_query)
    
    # If we have both local and web data, combine them
    if local_search_result and web_result.get('phone_found'):
        return {
            'combined_search': True,
            'local_data': local_search_result,
            'web_data': web_result,
            'recommendation': 'Combined data from local database and web search for comprehensive analysis',
            'confidence': max(local_search_result.get('confidence', 0), web_result.get('confidence', 0)),
            'source': 'hybrid'
        }
    
    # Return web results if local data is not available
    elif web_result.get('phone_found'):
        web_result['fallback_search'] = True
        web_result['local_data_available'] = False
        return web_result
    
    # No data found anywhere
    else:
        return {
            'phone_found': False,
            'search_query': user_query,
            'message': 'Phone not found in local database or web sources',
            'suggestions': [
                'Check phone model spelling',
                'Try searching for a similar phone',
                'Search for the brand name only',
                'Check if this is a very new or regional phone model'
            ],
            'source': 'no_source'
        }