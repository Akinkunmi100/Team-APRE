"""
API-based Web Search Agent for AI Phone Review Engine
Uses legitimate APIs instead of web scraping for better reliability
"""

import json
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import requests
from urllib.parse import quote_plus
import concurrent.futures
from threading import Lock
import random
import re

# Import existing components (with fallback)
try:
    from .smart_search import SmartPhoneSearch, SearchQuery
    SMART_SEARCH_AVAILABLE = True
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Smart search not available - using basic parsing")
    SMART_SEARCH_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APIWebSearchAgent:
    """
    API-based web search agent for phone research
    Uses legitimate APIs instead of web scraping
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the API-based web search agent"""
        if SMART_SEARCH_AVAILABLE:
            self.smart_search = SmartPhoneSearch()
        else:
            self.smart_search = None
        self.search_lock = Lock()
        
        # Default configuration
        self.config = config or {
            'max_concurrent_searches': 3,
            'search_timeout': 30,
            'max_results_per_source': 5,
            'min_confidence_threshold': 0.6,
            'rate_limit_delay': 1.0,
            'enable_fallback_search': True,
            'use_cached_data': True
        }
        
        # API endpoints and configurations
        self.api_sources = {
            'duckduckgo': {
                'base_url': 'https://api.duckduckgo.com/',
                'enabled': True,
                'priority': 1
            },
            'wikipedia_mobile': {
                'base_url': 'https://en.wikipedia.org/api/rest_v1/',
                'enabled': True,
                'priority': 2
            }
        }
        
        # Cache for search results
        self.search_cache = {}
        self.cache_expiry = 3600  # 1 hour
        
        # Common headers for requests
        self.headers = {
            'User-Agent': 'PhoneReviewEngine/1.0 (Educational Research)',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
        }
    
    def search_phone_external(self, query: str, max_sources: int = 3) -> Dict[str, Any]:
        """
        Main method to search for phone information using APIs
        
        Args:
            query: Search query (phone model or natural language)
            max_sources: Maximum number of API sources to query
            
        Returns:
            Comprehensive phone information from API search
        """
        
        # Parse the query using smart search or basic parsing
        if self.smart_search:
            parsed_query = self.smart_search.parse_query(query)
            logger.info(f"API-based external search for: {parsed_query.phone_model}")
            logger.info(f"Search intent: {parsed_query.intent}, Confidence: {parsed_query.confidence}")
        else:
            # Basic parsing fallback
            parsed_query = self._basic_query_parse(query)
            logger.info(f"API-based external search for: {parsed_query['phone_model']}")
        
        # Check cache first
        if self.smart_search:
            cache_key = f"{parsed_query.phone_model}_{parsed_query.intent}".lower()
        else:
            cache_key = f"{parsed_query['phone_model']}_search".lower()
            
        if self._is_cache_valid(cache_key):
            logger.info("Returning cached API results")
            return self.search_cache[cache_key]['data']
        
        # Perform API-based search
        try:
            search_results = self._perform_concurrent_api_search(parsed_query, max_sources)
            result = self._process_search_results(parsed_query, search_results)
            
            # Cache results
            self._cache_results(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"API search failed: {str(e)}")
            return self._create_fallback_result(parsed_query, str(e))
    
    def _basic_query_parse(self, query: str) -> Dict[str, Any]:
        """Basic query parsing fallback when smart search is unavailable"""
        
        return {
            'phone_model': query.strip(),
            'intent': 'reviews',
            'confidence': 0.7,
            'brand': self._extract_brand_from_query(query)
        }
    
    def _extract_brand_from_query(self, query: str) -> Optional[str]:
        """Extract brand from query"""
        query_lower = query.lower()
        brand_map = {
            'iphone': 'Apple', 'apple': 'Apple',
            'samsung': 'Samsung', 'galaxy': 'Samsung',
            'pixel': 'Google', 'google': 'Google',
            'oneplus': 'OnePlus', 'xiaomi': 'Xiaomi',
            'huawei': 'Huawei', 'nothing': 'Nothing'
        }
        
        for keyword, brand in brand_map.items():
            if keyword in query_lower:
                return brand
        return None
    
    def _perform_concurrent_api_search(self, parsed_query, max_sources: int) -> List[Dict[str, Any]]:
        """Perform concurrent API searches across multiple sources"""
        
        results = []
        
        # Get phone model
        if self.smart_search:
            phone_model = parsed_query.phone_model
        else:
            phone_model = parsed_query['phone_model']
        
        # Execute searches concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_sources) as executor:
            futures = []
            
            # DuckDuckGo search
            futures.append(executor.submit(self._search_duckduckgo, phone_model))
            
            # Wikipedia search
            futures.append(executor.submit(self._search_wikipedia, phone_model))
            
            # Phone Specs API search
            futures.append(executor.submit(self._search_phone_specs, phone_model))
            
            # Collect results
            for future in concurrent.futures.as_completed(futures, timeout=self.config['search_timeout']):
                try:
                    result = future.result(timeout=5)
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.warning(f"API search task failed: {str(e)}")
        
        return results
    
    def _search_duckduckgo(self, query: str) -> Optional[Dict[str, Any]]:
        """Search DuckDuckGo Instant Answer API"""
        
        try:
            params = {
                'q': f"{query} phone review specs",
                'format': 'json',
                'no_redirect': '1',
                'no_html': '1',
                'skip_disambig': '1'
            }
            
            response = requests.get(
                'https://api.duckduckgo.com/',
                params=params,
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('Abstract') or data.get('Answer'):
                return {
                    'source': 'duckduckgo',
                    'title': data.get('Heading', query),
                    'abstract': data.get('Abstract', ''),
                    'answer': data.get('Answer', ''),
                    'infobox': data.get('Infobox', {}),
                    'related_topics': data.get('RelatedTopics', [])[:3]
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"DuckDuckGo search failed: {str(e)}")
            return None
    
    def _search_wikipedia(self, query: str) -> Optional[Dict[str, Any]]:
        """Search Wikipedia API"""
        
        try:
            # Clean query for Wikipedia
            wiki_query = query.replace(' ', '_')
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote_plus(wiki_query)}"
            
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 404:
                # Try search API
                search_url = "https://en.wikipedia.org/w/api.php"
                search_params = {
                    'action': 'query',
                    'list': 'search',
                    'srsearch': query,
                    'format': 'json',
                    'srlimit': 1
                }
                
                search_response = requests.get(search_url, params=search_params, timeout=10)
                search_response.raise_for_status()
                search_data = search_response.json()
                
                if search_data.get('query', {}).get('search'):
                    page_title = search_data['query']['search'][0]['title']
                    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote_plus(page_title)}"
                    response = requests.get(url, headers=self.headers, timeout=10)
            
            response.raise_for_status()
            data = response.json()
            
            if data.get('extract'):
                return {
                    'source': 'wikipedia',
                    'title': data.get('title', query),
                    'extract': data.get('extract', ''),
                    'page_url': data.get('content_urls', {}).get('desktop', {}).get('page', ''),
                    'thumbnail': data.get('thumbnail', {}).get('source', ''),
                    'description': data.get('description', '')
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"Wikipedia search failed: {str(e)}")
            return None
    
    def _search_phone_specs(self, query: str) -> Optional[Dict[str, Any]]:
        """Search Phone Specs API"""
        
        try:
            # Use a public phone specs API
            search_params = {
                'search': query,
                'limit': 5
            }
            
            response = requests.get(
                'https://phone-specs-api.azharimm.dev/api/phones',
                params=search_params,
                headers=self.headers,
                timeout=15
            )
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('data') and len(data['data']) > 0:
                phone_data = data['data'][0]
                return {
                    'source': 'phone_specs',
                    'phone_name': phone_data.get('phone_name', query),
                    'brand': phone_data.get('brand', ''),
                    'release_date': phone_data.get('release_date', ''),
                    'dimension': phone_data.get('dimension', ''),
                    'os': phone_data.get('os', ''),
                    'storage': phone_data.get('storage', ''),
                    'display': phone_data.get('display', {}),
                    'camera': phone_data.get('camera', {}),
                    'battery': phone_data.get('battery', {})
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"Phone Specs API failed: {str(e)}")
            return None
    
    def _process_search_results(self, parsed_query, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process and combine search results"""
        
        if self.smart_search:
            phone_model = parsed_query.phone_model
        else:
            phone_model = parsed_query['phone_model']
        
        if not search_results:
            return self._create_fallback_result(parsed_query, "No API results found")
        
        # Combine information from all sources
        combined_specs = {}
        combined_features = []
        combined_pros = []
        combined_cons = []
        sources = []
        
        for result in search_results:
            source = result.get('source', 'unknown')
            sources.append(source)
            
            if source == 'duckduckgo':
                if result.get('abstract'):
                    combined_features.append(f"Overview: {result['abstract'][:100]}...")
                if result.get('infobox'):
                    combined_specs.update(result['infobox'])
            
            elif source == 'wikipedia':
                if result.get('extract'):
                    combined_features.append(f"Wikipedia: {result['extract'][:150]}...")
            
            elif source == 'phone_specs':
                if result.get('display'):
                    combined_specs['Display'] = result['display']
                if result.get('camera'):
                    combined_specs['Camera'] = result['camera']
                if result.get('battery'):
                    combined_specs['Battery'] = result['battery']
                if result.get('os'):
                    combined_specs['OS'] = result['os']
                
                # Generate pros/cons from specs
                if result.get('camera', {}):
                    combined_pros.append("Advanced camera system")
                if result.get('battery', {}):
                    combined_pros.append("Good battery life")
        
        # Generate rating based on available data
        base_rating = 4.0
        if len(sources) >= 3:
            base_rating += 0.3
        elif len(sources) >= 2:
            base_rating += 0.2
        
        overall_rating = min(base_rating, 5.0)
        
        # Calculate confidence
        confidence = 0.6 + (0.1 * len(sources))
        confidence = min(confidence, 0.9)
        
        # Add default cons if none found
        if not combined_cons:
            combined_cons = [
                "Price may vary by region",
                "Availability depends on market",
                "Accessories sold separately"
            ]
        
        return {
            'phone_found': True,
            'model': phone_model,
            'brand': self._extract_brand_from_query(phone_model),
            'confidence': confidence,
            'overall_rating': overall_rating,
            'review_count': len(sources) * 15,
            'key_features': combined_features[:6],
            'specifications': combined_specs,
            'pros': combined_pros[:5] if combined_pros else [
                "Reliable performance",
                "Good build quality",
                "Regular updates"
            ],
            'cons': combined_cons[:4],
            'price_range': {},
            'reviews_summary': search_results,
            'sources': sources,
            'search_metadata': {
                'api_sources_used': len(sources),
                'search_timestamp': datetime.now().isoformat(),
                'data_freshness': 'real_time'
            }
        }
    
    def _create_fallback_result(self, parsed_query, error_message: str) -> Dict[str, Any]:
        """Create fallback result when API search fails"""
        
        if self.smart_search:
            phone_model = parsed_query.phone_model
        else:
            phone_model = parsed_query['phone_model']
        
        return {
            'phone_found': False,
            'model': phone_model,
            'confidence': 0.0,
            'error_message': error_message,
            'sources': [],
            'search_metadata': {
                'api_search_failed': True,
                'error': error_message,
                'search_timestamp': datetime.now().isoformat()
            }
        }
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached result is still valid"""
        if cache_key not in self.search_cache:
            return False
        
        cache_entry = self.search_cache[cache_key]
        cache_age = datetime.now() - datetime.fromisoformat(cache_entry['timestamp'])
        
        return cache_age.total_seconds() < self.cache_expiry
    
    def _cache_results(self, cache_key: str, result: Dict[str, Any]):
        """Cache search results"""
        self.search_cache[cache_key] = {
            'data': result,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get search statistics"""
        return {
            'api_sources': list(self.api_sources.keys()),
            'cache_size': len(self.search_cache),
            'smart_search_available': self.smart_search is not None
        }


# Factory function
def create_api_web_search_agent(config=None):
    """Create configured API Web Search Agent"""
    return APIWebSearchAgent(config=config)