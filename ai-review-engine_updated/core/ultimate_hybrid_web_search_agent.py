"""
Ultimate Hybrid Web Search Agent for AI Phone Review Engine
Combines Google Custom Search API with enhanced offline capabilities
Ensures ANY phone can be found through multiple fallback layers
"""

import asyncio
import json
import time
import logging
import os
import re
import random
import aiohttp
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UniversalSearchResult:
    """Enhanced structure for universal search results"""
    phone_model: str
    source: str
    title: str
    snippet: str
    url: Optional[str]
    rating: Optional[float]
    review_count: Optional[int]
    price: Optional[str]
    sentiment_preview: str
    confidence: float
    retrieved_at: str
    additional_data: Dict[str, Any] = None
    search_layer: str = "unknown"  # google, api, mock, static, offline

@dataclass
class UniversalPhoneData:
    """Comprehensive phone data from all search layers"""
    phone_model: str
    phone_found: bool
    search_results: List[UniversalSearchResult]
    specifications: Dict[str, Any]
    reviews_summary: Dict[str, Any]
    price_info: Dict[str, Any]
    pros: List[str]
    cons: List[str]
    key_features: List[str]
    overall_rating: Optional[float]
    overall_sentiment: str
    confidence: float
    sources: List[str]
    search_layers_used: List[str]
    search_metadata: Dict[str, Any]

class UltimateHybridWebSearchAgent:
    """
    Ultimate hybrid search agent that combines:
    1. Google Custom Search API (universal web search)
    2. Original API sources (DuckDuckGo, Phone Specs, Wikipedia)
    3. Enhanced offline capabilities (static DB, mock data)
    4. Smart fallback system with confidence-based selection
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the ultimate hybrid search agent"""
        self.search_lock = Lock()
        
        # Default configuration
        self.config = config or {
            'google_api_key': os.getenv('GOOGLE_SEARCH_API_KEY'),
            'google_search_engine_id': os.getenv('GOOGLE_SEARCH_ENGINE_ID'),
            'enable_google_search': True,
            'enable_api_sources': True,
            'enable_offline_fallback': True,
            'enable_mock_data': False  # NO MOCK DATA,
            'max_concurrent_searches': 3,
            'search_timeout': 30,
            'max_results_per_source': 5,
            'min_confidence_threshold': 0.5,
            'rate_limit_delay': 1.0,
            'use_cached_data': True,
            'cache_expiry': 7200,  # 2 hours
            'prefer_google_over_apis': True,
            'google_search_confidence_boost': 0.2
        }
        
        # Check Google API availability
        self.google_available = (
            self.config.get('google_api_key') and 
            self.config.get('google_search_engine_id') and
            self.config.get('enable_google_search', True)
        )
        
        if self.google_available:
            logger.info("🌐 Google Custom Search API available - Universal search enabled")
        else:
            logger.info("📱 Google API not configured - Using API + offline fallback")
        
        # Initialize search layers
        self._init_google_search_layer()
        self._init_api_search_layer()
        self._init_offline_search_layer()
        
        # Cache for search results
        self.search_cache = {}
        
        # Search statistics
        self.search_stats = {
            'total_searches': 0,
            'google_searches': 0,
            'api_searches': 0,
            'offline_searches': 0,
            'mock_data_used': 0,
            'cache_hits': 0,
            'hybrid_results': 0,
            'success_rate': 0.0
        }
        
        # Import smart search for query parsing
        try:
            from .smart_search import SmartPhoneSearch
            self.smart_search = SmartPhoneSearch()
            self.smart_search_available = True
        except ImportError:
            logger.warning("Smart search not available - using basic query parsing")
            self.smart_search_available = False
    
    def _init_google_search_layer(self):
        """Initialize Google Custom Search layer"""
        if self.google_available:
            try:
                from .google_search_integration import GoogleCustomSearch
                self.google_search = GoogleCustomSearch({
                    'api_key': self.config['google_api_key'],
                    'search_engine_id': self.config['google_search_engine_id'],
                    'max_results_per_query': self.config['max_results_per_source'],
                    'request_timeout': self.config['search_timeout']
                })
                logger.info("✅ Google Custom Search layer initialized")
            except ImportError as e:
                logger.warning(f"Google search layer unavailable: {e}")
                self.google_available = False
        else:
            self.google_search = None
    
    def _init_api_search_layer(self):
        """Initialize API search layer (DuckDuckGo, Phone Specs, Wikipedia)"""
        if self.config.get('enable_api_sources', True):
            try:
                from .api_web_search_agent import APIWebSearchAgent
                self.api_search = APIWebSearchAgent({
                    'max_concurrent_searches': self.config.get('max_concurrent_searches', 3),
                    'search_timeout': self.config.get('search_timeout', 30),
                    'max_results_per_source': self.config.get('max_results_per_source', 5),
                    'rate_limit_delay': self.config.get('rate_limit_delay', 1.0)
                })
                self.api_search_available = True
                logger.info("✅ API search layer initialized (DuckDuckGo, Phone Specs, Wikipedia)")
            except ImportError as e:
                logger.warning(f"API search layer unavailable: {e}")
                self.api_search_available = False
        else:
            self.api_search_available = False
    
    def _init_offline_search_layer(self):
        """Initialize offline search layer (enhanced fallback)"""
        try:
            from .enhanced_api_web_search_agent import EnhancedAPIWebSearchAgent
            self.offline_search = EnhancedAPIWebSearchAgent({
                'enable_mock_data': self.config.get('enable_mock_data', True),
                'mock_data_confidence': 0.0  # NO MOCK DATA,
                'use_cached_data': True
            })
            self.offline_search_available = True
            logger.info("✅ Offline search layer initialized (Static DB + Mock Data)")
        except ImportError as e:
            logger.warning(f"Offline search layer unavailable: {e}")
            self.offline_search_available = False
    
    async def search_phone_universally(self, query: str, search_depth: str = 'comprehensive', max_retries: int = 3) -> UniversalPhoneData:
        """
        Ultimate phone search using all available layers with retry logic
        
        Search Priority:
        1. Google Custom Search API (if available) - Universal web coverage
        2. API Sources (DuckDuckGo, Phone Specs, Wikipedia) - Structured data
        3. Offline Fallback (Static DB + Mock Data) - Always available
        
        Args:
            query: Phone model or natural language query
            search_depth: 'basic', 'standard', or 'comprehensive'
            max_retries: Maximum number of retry attempts for failed searches
            
        Returns:
            UniversalPhoneData with comprehensive phone information
            
        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If all search attempts fail
        """
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")
            
        if search_depth not in ['basic', 'standard', 'comprehensive']:
            raise ValueError("Invalid search depth")
            
        retry_count = 0
        last_error = None
        
        while retry_count < max_retries:
            try:
                return await self._execute_search(query, search_depth)
            except aiohttp.ClientError as e:
                last_error = e
                logger.warning(f"Network error on attempt {retry_count + 1}: {e}")
                await asyncio.sleep(1 * (retry_count + 1))  # Exponential backoff
            except Exception as e:
                last_error = e
                logger.error(f"Search error on attempt {retry_count + 1}: {e}")
                if not isinstance(e, (aiohttp.ClientError, asyncio.TimeoutError)):
                    break  # Don't retry on non-network errors
            retry_count += 1
            
        # All retries failed or unretriable error
        logger.error(f"Search failed after {retry_count} attempts: {last_error}")
        raise RuntimeError(f"Search failed: {str(last_error)}")
        
    async def _execute_search(self, query: str, search_depth: str) -> UniversalPhoneData:
        """Execute the actual search with timeout protection and layer management"""
        try:
            async with asyncio.timeout(30):  # 30 second timeout
                # Initialize result container
                result = UniversalPhoneData(
                    phone_model=query,
                    phone_found=False,
                    search_results=[],
                    specifications={},
                    reviews_summary={},
                    price_info={},
                    pros=[],
                    cons=[],
                    key_features=[],
                    overall_rating=None,
                    overall_sentiment="unknown",
                    confidence=0.0,
                    sources=[],
                    search_layers_used=[],
                    search_metadata={}
                )
                
                # Track which search layers were used
                layers_used = []
                
                # 1. Try Google Custom Search first (if available)
                if self.google_available and self.google_search:
                    try:
                        google_results = await self.google_search.search_phone(query)
                        if google_results and google_results.success:
                            result.search_results.extend(google_results.results)
                            result.confidence = max(result.confidence, 
                                                  google_results.confidence + self.config.get('google_search_confidence_boost', 0.2))
                            layers_used.append('google')
                    except Exception as e:
                        logger.warning(f"Google search failed: {e}")
                
                # 2. Try API sources
                if self.api_search_available:
                    try:
                        api_results = await self.api_search.search_phone_async(query)
                        if api_results and api_results.success:
                            result.search_results.extend(api_results.results)
                            result.specifications.update(api_results.specifications or {})
                            result.confidence = max(result.confidence, api_results.confidence)
                            layers_used.append('api')
                    except Exception as e:
                        logger.warning(f"API search failed: {e}")
                
                # 3. Try offline search as fallback
                if not result.search_results and self.offline_search_available:
                    try:
                        offline_results = await self.offline_search.search_phone(query)
                        if offline_results and offline_results.success:
                            result.search_results.extend(offline_results.results)
                            result.confidence = max(result.confidence, offline_results.confidence)
                            layers_used.append('offline')
                    except Exception as e:
                        logger.warning(f"Offline search failed: {e}")
                
                # Update metadata
                result.search_layers_used = layers_used
                result.search_metadata.update({
                    'layers_attempted': layers_used,
                    'search_depth': search_depth,
                    'timestamp': datetime.now().isoformat(),
                    'total_results': len(result.search_results)
                })
                
                # Set overall success flag
                result.phone_found = len(result.search_results) > 0
                
                # Calculate overall rating if available
                if result.search_results:
                    ratings = [r.rating for r in result.search_results if r.rating is not None]
                    if ratings:
                        result.overall_rating = sum(ratings) / len(ratings)
                
                # Set overall sentiment
                if result.search_results:
                    positive = sum(1 for r in result.search_results if r.sentiment_preview == "positive")
                    negative = sum(1 for r in result.search_results if r.sentiment_preview == "negative")
                    total = len(result.search_results)
                    
                    if total > 0:
                        if positive / total > 0.6:
                            result.overall_sentiment = "positive"
                        elif negative / total > 0.6:
                            result.overall_sentiment = "negative"
                        else:
                            result.overall_sentiment = "neutral"
                
                return result
        Raises:
            ValueError: If query is invalid or search_depth is unknown
            RuntimeError: If all search layers fail
        """
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")
            
        if search_depth not in ['basic', 'standard', 'comprehensive']:
            raise ValueError("Invalid search depth. Must be 'basic', 'standard', or 'comprehensive'")
            
        try:
            # Initialize empty result container
            result = UniversalPhoneData(
                phone_model=query,
                phone_found=False,
                search_results=[],
                specifications={},
                reviews_summary={},
                price_info={},
                pros=[],
                cons=[],
                key_features=[],
                overall_rating=None,
                overall_sentiment="unknown",
                confidence=0.0,
                sources=[],
                search_layers_used=[],
                search_metadata={}
            )
        
        Args:
            query: Phone model or natural language query
            search_depth: 'basic', 'standard', or 'comprehensive'
            
        Returns:
            UniversalPhoneData with comprehensive phone information
        """
        
        self.search_stats['total_searches'] += 1
        search_start_time = datetime.now()
        
        # Parse query using smart search if available
        parsed_query = self._parse_query_universal(query)
        logger.info(f"🔍 Universal search for: {parsed_query['phone_model']} (Intent: {parsed_query['intent']})")
        
        # Check cache first
        cache_key = f"{parsed_query['phone_model']}_{search_depth}".lower()
        if self._is_cache_valid(cache_key):
            logger.info("⚡ Returning cached universal results")
            self.search_stats['cache_hits'] += 1
            return self.search_cache[cache_key]['data']
        
        # Execute search layers
        search_layers_used = []
        all_results = []
        
        try:
            # Layer 1: Google Custom Search (Universal Web Coverage)
            google_results = []
            if self.google_available and search_depth in ['standard', 'comprehensive']:
                try:
                    logger.info("🌐 Executing Google Custom Search...")
                    google_data = await self.google_search.search_phone_universally(
                        parsed_query['phone_model'], 
                        search_depth
                    )
                    
                    if google_data and google_data.search_results:
                        google_results = self._convert_google_results(google_data)
                        all_results.extend(google_results)
                        search_layers_used.append('google_custom_search')
                        self.search_stats['google_searches'] += 1
                        logger.info(f"✅ Google search: {len(google_results)} results")
                    
                except Exception as e:
                    logger.warning(f"Google search failed: {str(e)}")
            
            # Layer 2: API Sources (Structured Data)
            api_results = []
            if self.api_search_available and (not google_results or search_depth == 'comprehensive'):
                try:
                    logger.info("🔗 Executing API sources search...")
                    api_data = self.api_search.search_phone_external(parsed_query['phone_model'])
                    
                    if api_data and api_data.get('phone_found'):
                        api_results = self._convert_api_results(api_data)
                        all_results.extend(api_results)
                        search_layers_used.append('api_sources')
                        self.search_stats['api_searches'] += 1
                        logger.info(f"✅ API search: {len(api_results)} results")
                        
                except Exception as e:
                    logger.warning(f"API search failed: {str(e)}")
            
            # Layer 3: Offline Fallback (Always Available)
            offline_results = []
            if self.offline_search_available and (len(all_results) < 3 or search_depth == 'comprehensive'):
                try:
                    logger.info("💾 Executing offline fallback search...")
                    offline_data = self.offline_search.search_phone_external(parsed_query['phone_model'])
                    
                    if offline_data and offline_data.get('phone_found'):
                        offline_results = self._convert_offline_results(offline_data)
                        all_results.extend(offline_results)
                        search_layers_used.append('offline_fallback')
                        self.search_stats['offline_searches'] += 1
                        
                        if 'mock' in str(offline_data.get('sources', [])):
                            self.search_stats['mock_data_used'] += 1
                        
                        logger.info(f"✅ Offline search: {len(offline_results)} results")
                        
                except Exception as e:
                    logger.warning(f"Offline search failed: {str(e)}")
            
            # Combine and process all results
            universal_data = self._process_universal_results(
                parsed_query, all_results, search_layers_used, search_start_time
            )
            
            # Cache results
            self._cache_results(cache_key, universal_data)
            
            # Update success rate
            if universal_data.phone_found:
                success_count = self.search_stats['total_searches'] - self.search_stats.get('failed_searches', 0)
                self.search_stats['success_rate'] = (success_count / self.search_stats['total_searches']) * 100
            
            return universal_data
            
        except Exception as e:
            logger.error(f"Universal search failed: {str(e)}")
            return self._create_empty_universal_result(parsed_query, str(e))
    
    def _parse_query_universal(self, query: str) -> Dict[str, Any]:
        """Parse query using smart search or fallback method"""
        
        if self.smart_search_available:
            try:
                parsed = self.smart_search.parse_query(query)
                return {
                    'phone_model': parsed.phone_model,
                    'brand': parsed.brand,
                    'intent': parsed.intent,
                    'confidence': parsed.confidence,
                    'original_query': query
                }
            except Exception as e:
                logger.warning(f"Smart search parsing failed: {e}")
        
        # Fallback parsing
        return {
            'phone_model': query,
            'brand': self._extract_brand_simple(query),
            'intent': 'reviews',
            'confidence': 0.7,
            'original_query': query
        }
    
    def _extract_brand_simple(self, query: str) -> Optional[str]:
        """Simple brand extraction fallback"""
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
    
    def _convert_google_results(self, google_data) -> List[UniversalSearchResult]:
        """Convert Google Custom Search results to universal format"""
        results = []
        
        for result in google_data.search_results:
            # Apply confidence boost for Google results
            confidence = min(result.relevance_score + self.config.get('google_search_confidence_boost', 0.2), 1.0)
            
            universal_result = UniversalSearchResult(
                phone_model=google_data.phone_model,
                source=f"google_{result.source_type}",
                title=result.title,
                snippet=result.snippet,
                url=result.url,
                rating=None,  # Could be extracted from additional_data
                review_count=None,
                price=None,  # Could be extracted from additional_data
                sentiment_preview=google_data.overall_sentiment,
                confidence=confidence,
                retrieved_at=result.scraped_at,
                additional_data={
                    'extracted_data': result.extracted_data,
                    'page_content': result.page_content,
                    'google_specs': google_data.specifications
                },
                search_layer='google'
            )
            results.append(universal_result)
        
        return results
    
    def _convert_api_results(self, api_data: Dict) -> List[UniversalSearchResult]:
        """Convert API sources results to universal format"""
        results = []
        
        # Create results from API data
        for source in api_data.get('sources', []):
            universal_result = UniversalSearchResult(
                phone_model=api_data.get('model', 'Unknown'),
                source=f"api_{source}",
                title=f"{api_data.get('model')} - {source.title()}",
                snippet=f"API data from {source}",
                url=None,
                rating=api_data.get('overall_rating'),
                review_count=api_data.get('review_count'),
                price=api_data.get('price_range', {}).get('estimate'),
                sentiment_preview='positive' if api_data.get('overall_rating', 0) > 4.0 else 'neutral',
                confidence=api_data.get('confidence', 0.8),
                retrieved_at=datetime.now().isoformat(),
                additional_data={
                    'specifications': api_data.get('specifications', {}),
                    'pros': api_data.get('pros', []),
                    'cons': api_data.get('cons', []),
                    'key_features': api_data.get('key_features', [])
                },
                search_layer='api'
            )
            results.append(universal_result)
        
        return results
    
    def _convert_offline_results(self, offline_data: Dict) -> List[UniversalSearchResult]:
        """Convert offline search results to universal format"""
        results = []
        
        for source in offline_data.get('sources', []):
            universal_result = UniversalSearchResult(
                phone_model=offline_data.get('model', 'Unknown'),
                source=f"offline_{source}",
                title=f"{offline_data.get('model')} - {source.title()}",
                snippet=f"Offline data from {source}",
                url=None,
                rating=offline_data.get('overall_rating'),
                review_count=offline_data.get('review_count'),
                price=offline_data.get('price_range', {}).get('estimate'),
                sentiment_preview=offline_data.get('search_metadata', {}).get('data_quality', 'simulated'),
                confidence=offline_data.get('confidence', 0.6),
                retrieved_at=datetime.now().isoformat(),
                additional_data={
                    'specifications': offline_data.get('specifications', {}),
                    'pros': offline_data.get('pros', []),
                    'cons': offline_data.get('cons', []),
                    'key_features': offline_data.get('key_features', []),
                    'data_type': offline_data.get('search_metadata', {}).get('data_quality', 'simulated')
                },
                search_layer='offline'
            )
            results.append(universal_result)
        
        return results
    
    def _process_universal_results(self, parsed_query: Dict, all_results: List[UniversalSearchResult], 
                                 search_layers_used: List[str], search_start_time: datetime) -> UniversalPhoneData:
        """Process and combine all search results into universal phone data"""
        
        if not all_results:
            return self._create_empty_universal_result(parsed_query, "No results from any search layer")
        
        # Sort by confidence
        all_results.sort(key=lambda r: r.confidence, reverse=True)
        
        # Extract and combine specifications
        specifications = {}
        for result in all_results:
            if result.additional_data and 'specifications' in result.additional_data:
                specifications.update(result.additional_data['specifications'])
        
        # Extract pricing information
        price_info = {}
        price_mentions = [r for r in all_results if r.price]
        if price_mentions:
            prices = []
            for result in price_mentions:
                try:
                    price_str = result.price.replace('$', '').replace(',', '')
                    if price_str.replace('.', '').isdigit():
                        prices.append(float(price_str))
                except:
                    continue
            
            if prices:
                price_info = {
                    'min_price': f"${min(prices):,.0f}",
                    'max_price': f"${max(prices):,.0f}",
                    'avg_price': f"${sum(prices)/len(prices):,.0f}",
                    'status': 'found'
                }
        
        # Combine pros and cons
        all_pros = []
        all_cons = []
        for result in all_results:
            if result.additional_data:
                if 'pros' in result.additional_data:
                    all_pros.extend(result.additional_data['pros'])
                if 'cons' in result.additional_data:
                    all_cons.extend(result.additional_data['cons'])
        
        # Remove duplicates and limit
        unique_pros = list(dict.fromkeys(all_pros))[:5]
        unique_cons = list(dict.fromkeys(all_cons))[:4]
        
        # Extract key features
        all_features = []
        for result in all_results:
            if result.additional_data and 'key_features' in result.additional_data:
                all_features.extend(result.additional_data['key_features'])
        unique_features = list(dict.fromkeys(all_features))[:6]
        
        # Calculate overall rating
        ratings = [r.rating for r in all_results if r.rating]
        overall_rating = sum(ratings) / len(ratings) if ratings else None
        
        # Determine overall sentiment
        sentiments = [r.sentiment_preview for r in all_results if r.sentiment_preview]
        positive_count = sum(1 for s in sentiments if 'positive' in s)
        negative_count = sum(1 for s in sentiments if 'negative' in s)
        
        if positive_count > negative_count:
            overall_sentiment = 'positive'
        elif negative_count > positive_count:
            overall_sentiment = 'negative'
        else:
            overall_sentiment = 'neutral'
        
        # Calculate combined confidence
        if all_results:
            combined_confidence = sum(r.confidence for r in all_results) / len(all_results)
        else:
            combined_confidence = 0.0
        
        # Create reviews summary
        reviews_summary = {
            'total_sources': len(all_results),
            'average_sentiment': overall_sentiment,
            'average_rating': overall_rating,
            'confidence': combined_confidence
        }
        
        # Track if hybrid result (multiple layers)
        if len(search_layers_used) > 1:
            self.search_stats['hybrid_results'] += 1
        
        return UniversalPhoneData(
            phone_model=parsed_query['phone_model'],
            phone_found=True,
            search_results=all_results,
            specifications=specifications,
            reviews_summary=reviews_summary,
            price_info=price_info,
            pros=unique_pros,
            cons=unique_cons,
            key_features=unique_features,
            overall_rating=overall_rating,
            overall_sentiment=overall_sentiment,
            confidence=combined_confidence,
            sources=list(set(r.source for r in all_results)),
            search_layers_used=search_layers_used,
            search_metadata={
                'search_time': (datetime.now() - search_start_time).total_seconds(),
                'total_results': len(all_results),
                'layers_used': search_layers_used,
                'google_available': self.google_available,
                'api_available': self.api_search_available,
                'offline_available': self.offline_search_available,
                'search_timestamp': datetime.now().isoformat()
            }
        )
    
    def _create_empty_universal_result(self, parsed_query: Dict, error_message: str) -> UniversalPhoneData:
        """Create empty result when no data is found"""
        
        return UniversalPhoneData(
            phone_model=parsed_query['phone_model'],
            phone_found=False,
            search_results=[],
            specifications={},
            reviews_summary={'total_sources': 0, 'average_sentiment': 'neutral', 'confidence': 0.0},
            price_info={},
            pros=[],
            cons=[],
            key_features=[],
            overall_rating=None,
            overall_sentiment='neutral',
            confidence=0.0,
            sources=[],
            search_layers_used=[],
            search_metadata={
                'error': error_message,
                'search_timestamp': datetime.now().isoformat(),
                'universal_search_failed': True
            }
        )
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached result is still valid"""
        if cache_key not in self.search_cache:
            return False
        
        cache_entry = self.search_cache[cache_key]
        cache_age = datetime.now() - datetime.fromisoformat(cache_entry['timestamp'])
        
        return cache_age.total_seconds() < self.config['cache_expiry']
    
    def _cache_results(self, cache_key: str, result: UniversalPhoneData):
        """Cache universal search results"""
        self.search_cache[cache_key] = {
            'data': result,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get comprehensive search statistics"""
        
        total_searches = max(1, self.search_stats['total_searches'])
        
        return {
            **self.search_stats,
            'google_usage_rate': (self.search_stats['google_searches'] / total_searches) * 100,
            'api_usage_rate': (self.search_stats['api_searches'] / total_searches) * 100,
            'offline_usage_rate': (self.search_stats['offline_searches'] / total_searches) * 100,
            'cache_hit_rate': (self.search_stats['cache_hits'] / total_searches) * 100,
            'hybrid_result_rate': (self.search_stats['hybrid_results'] / total_searches) * 100,
            'capabilities': {
                'google_custom_search': self.google_available,
                'api_sources': self.api_search_available,
                'offline_fallback': self.offline_search_available,
                'smart_query_parsing': self.smart_search_available
            }
        }
    
    # Synchronous wrapper for compatibility
    def search_phone_external(self, query: str, max_sources: int = 3) -> Dict[str, Any]:
        """Synchronous wrapper for universal search"""
        
        try:
            # Run async search in event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Determine search depth based on max_sources
            if max_sources <= 2:
                search_depth = 'basic'
            elif max_sources <= 4:
                search_depth = 'standard'
            else:
                search_depth = 'comprehensive'
            
            universal_data = loop.run_until_complete(
                self.search_phone_universally(query, search_depth)
            )
            
            loop.close()
            
            # Convert to compatible format
            return self._convert_to_compatible_format(universal_data)
            
        except Exception as e:
            logger.error(f"Synchronous search wrapper failed: {str(e)}")
            return {
                'phone_found': False,
                'model': query,
                'confidence': 0.0,
                'error_message': str(e)
            }
    
    def _convert_to_compatible_format(self, universal_data: UniversalPhoneData) -> Dict[str, Any]:
        """Convert UniversalPhoneData to compatible dictionary format"""
        
        return {
            'phone_found': universal_data.phone_found,
            'model': universal_data.phone_model,
            'brand': self._extract_brand_simple(universal_data.phone_model),
            'confidence': universal_data.confidence,
            'overall_rating': universal_data.overall_rating,
            'review_count': universal_data.reviews_summary.get('total_sources', 0),
            'key_features': universal_data.key_features,
            'specifications': universal_data.specifications,
            'pros': universal_data.pros,
            'cons': universal_data.cons,
            'price_range': universal_data.price_info,
            'reviews_summary': universal_data.search_results,
            'sources': universal_data.sources,
            'search_metadata': {
                **universal_data.search_metadata,
                'search_layers_used': universal_data.search_layers_used,
                'universal_search': True
            },
            'recommendations': self._generate_recommendations(universal_data)
        }
    
    def _generate_recommendations(self, universal_data: UniversalPhoneData) -> Dict[str, Any]:
        """Generate recommendations based on universal search data"""
        
        confidence = universal_data.confidence
        rating = universal_data.overall_rating or 4.0
        sentiment = universal_data.overall_sentiment
        
        # Determine verdict
        if rating >= 4.3 and confidence >= 0.8 and sentiment == 'positive':
            verdict = "Highly Recommended"
            reason = f"Excellent rating ({rating:.1f}/5) with high confidence from multiple sources"
        elif rating >= 4.0 and confidence >= 0.6:
            verdict = "Recommended"
            reason = f"Good rating ({rating:.1f}/5) with reliable data"
        elif rating >= 3.5 and confidence >= 0.5:
            verdict = "Consider with Research"
            reason = f"Moderate rating ({rating:.1f}/5) - check detailed reviews"
        elif confidence < 0.5:
            verdict = "Insufficient Data"
            reason = "Limited reliable information available"
        else:
            verdict = "Not Recommended"
            reason = f"Below average rating ({rating:.1f}/5)"
        
        # Determine reliability
        if confidence >= 0.8:
            reliability = 'high'
        elif confidence >= 0.6:
            reliability = 'medium'
        else:
            reliability = 'low'
        
        # Create note based on search layers used
        layers_used = universal_data.search_layers_used
        if 'google' in str(layers_used):
            note = "Based on comprehensive web search and curated data sources"
        elif len(layers_used) > 1:
            note = f"Based on {len(layers_used)} data sources including APIs and offline data"
        else:
            note = f"Based on {layers_used[0]} data" if layers_used else "Based on available data"
        
        return {
            'verdict': verdict,
            'reason': reason,
            'reliability': reliability,
            'note': note,
            'confidence_score': confidence,
            'data_sources': universal_data.sources,
            'search_layers': universal_data.search_layers_used
        }


# Factory function for easy integration
def create_ultimate_hybrid_search_agent(config=None):
    """
    Create configured Ultimate Hybrid Web Search Agent
    
    This agent provides:
    1. Google Custom Search API (if configured)
    2. API sources (DuckDuckGo, Phone Specs, Wikipedia)
    3. Enhanced offline fallback (static DB + mock data)
    4. Smart fallback system with confidence-based selection
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured UltimateHybridWebSearchAgent
    """
    return UltimateHybridWebSearchAgent(config=config)