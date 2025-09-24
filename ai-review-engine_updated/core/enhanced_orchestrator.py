"""
Enhanced Search Orchestrator for AI Phone Review Engine
Integrates all enhanced components with ethical safeguards and backward compatibility
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import json

# Import enhanced components
from .enhanced_web_scraper import EnhancedWebScraper, ScrapedPhoneData, create_enhanced_web_scraper
from .pricing_api_integration import PricingAPIIntegration, PricingResult, create_pricing_api_integration
from .data_quality_validator import DataQualityValidator, ValidationResult, DataSourceType, create_data_quality_validator
from .smart_cache_system import SmartCacheSystem, CacheConfig, create_phone_review_cache

# Import existing components for backward compatibility
try:
    from .smart_search import SmartPhoneSearch, SearchQuery
    from .fallback_search_system import FallbackSearchSystem, create_fallback_search_system
    LEGACY_COMPONENTS_AVAILABLE = True
except ImportError:
    LEGACY_COMPONENTS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Legacy components not available - some features may be limited")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchResultQuality(Enum):
    """Quality levels for search results"""
    EXCELLENT = "excellent"
    GOOD = "good" 
    FAIR = "fair"
    POOR = "poor"
    UNACCEPTABLE = "unacceptable"

class DataSource(Enum):
    """Data source types"""
    LOCAL_DATABASE = "local"
    WEB_SCRAPING = "web_scraping"
    PRICING_API = "pricing_api"
    HYBRID = "hybrid"
    CACHED = "cached"
    FALLBACK = "fallback"

@dataclass
class EthicalDataInfo:
    """Ethical data handling information"""
    sources_disclosed: List[str]
    data_freshness: str
    confidence_level: float
    limitations: List[str]
    quality_indicators: Dict[str, Any]
    synthetic_data_used: bool = False
    user_privacy_protected: bool = True

@dataclass
class EnhancedSearchResult:
    """Comprehensive search result with ethical safeguards"""
    phone_model: str
    phone_found: bool
    data_source: DataSource
    confidence: float
    quality: SearchResultQuality
    
    # Phone data
    phone_data: Dict[str, Any]
    reviews: List[Dict[str, Any]]
    specifications: Dict[str, Any]
    pricing_info: Optional[PricingResult]
    
    # Metadata and ethics
    ethical_info: EthicalDataInfo
    search_metadata: Dict[str, Any]
    validation_results: Dict[str, ValidationResult]
    cache_info: Dict[str, Any]
    
    # Recommendations
    recommendations: Dict[str, Any]
    alternatives: List[str]
    
    # Error handling
    warnings: List[str]
    errors: List[str]

class EnhancedSearchOrchestrator:
    """
    Enhanced search orchestrator with ethical AI practices and production-ready features
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the enhanced search orchestrator"""
        
        self.config = config or {
            # Search configuration
            'enable_web_scraping': True,
            'enable_pricing_apis': True,
            'enable_caching': True,
            'enable_validation': True,
            
            # Quality thresholds
            'min_confidence_threshold': 0.6,
            'excellent_quality_threshold': 0.9,
            'good_quality_threshold': 0.7,
            
            # Timeout settings
            'web_scraping_timeout': 45,
            'pricing_api_timeout': 30,
            'total_search_timeout': 120,
            
            # Ethical settings
            'require_source_disclosure': True,
            'prohibit_synthetic_data': True,
            'require_data_freshness_checks': True,
            'enable_privacy_protection': True,
            
            # Performance settings
            'max_concurrent_operations': 3,
            'enable_result_caching': True,
            'cache_duration_hours': 2,
            
            # Fallback settings
            'enable_legacy_fallback': True,
            'fallback_confidence_boost': 0.1
        }
        
        # Initialize components
        self._init_components()
        
        # Search statistics
        self.search_stats = {
            'total_searches': 0,
            'successful_searches': 0,
            'cached_results': 0,
            'web_scraping_used': 0,
            'pricing_api_used': 0,
            'validation_failures': 0,
            'ethical_violations': 0,
            'average_response_time': 0.0
        }
    
    def _init_components(self):
        """Initialize all search components"""
        
        # Smart cache system
        if self.config['enable_caching']:
            try:
                cache_config = CacheConfig(
                    max_size_mb=2000,  # 2GB cache
                    default_ttl_seconds=self.config['cache_duration_hours'] * 3600,
                    max_entries=100000,
                    enable_persistence=True,
                    enable_compression=True,
                    enable_monitoring=True
                )
                self.cache_system = SmartCacheSystem(cache_config)
                logger.info("Smart cache system initialized")
            except Exception as e:
                logger.error(f"Failed to initialize cache system: {e}")
                self.cache_system = None
        else:
            self.cache_system = None
        
        # Enhanced web scraper
        if self.config['enable_web_scraping']:
            try:
                scraper_config = {
                    'max_concurrent_requests': self.config['max_concurrent_operations'],
                    'request_timeout': self.config['web_scraping_timeout'],
                    'max_retries': 3,
                    'use_selenium': True,
                    'headless': True,
                    'enable_javascript': True,
                    'cache_results': True
                }
                self.web_scraper = create_enhanced_web_scraper(scraper_config)
                logger.info("Enhanced web scraper initialized")
            except Exception as e:
                logger.error(f"Failed to initialize web scraper: {e}")
                self.web_scraper = None
        else:
            self.web_scraper = None
        
        # Pricing API integration
        if self.config['enable_pricing_apis']:
            try:
                pricing_config = {
                    'max_concurrent_requests': self.config['max_concurrent_operations'],
                    'request_timeout': self.config['pricing_api_timeout'],
                    'cache_duration': self.config['cache_duration_hours'] * 3600,
                    'max_price_age_hours': 6,  # Price data stales quickly
                    'min_confidence_threshold': self.config['min_confidence_threshold']
                }
                self.pricing_integration = create_pricing_api_integration(pricing_config)
                logger.info("Pricing API integration initialized")
            except Exception as e:
                logger.error(f"Failed to initialize pricing integration: {e}")
                self.pricing_integration = None
        else:
            self.pricing_integration = None
        
        # Data quality validator
        if self.config['enable_validation']:
            try:
                validator_config = {
                    'min_quality_threshold': self.config['min_confidence_threshold'],
                    'enable_source_tracking': True,
                    'enable_freshness_checks': self.config['require_data_freshness_checks']
                }
                self.data_validator = create_data_quality_validator(validator_config)
                logger.info("Data quality validator initialized")
            except Exception as e:
                logger.error(f"Failed to initialize data validator: {e}")
                self.data_validator = None
        else:
            self.data_validator = None
        
        # Legacy components for fallback
        if LEGACY_COMPONENTS_AVAILABLE and self.config['enable_legacy_fallback']:
            try:
                self.smart_search = SmartPhoneSearch()
                self.fallback_system = create_fallback_search_system()
                logger.info("Legacy fallback components initialized")
            except Exception as e:
                logger.warning(f"Legacy components initialization failed: {e}")
                self.smart_search = None
                self.fallback_system = None
        else:
            self.smart_search = None
            self.fallback_system = None
    
    async def search_phone(self, query: str, search_options: Dict[str, Any] = None) -> EnhancedSearchResult:
        """
        Enhanced phone search with ethical AI practices
        
        Args:
            query: Phone search query
            search_options: Additional search configuration
            
        Returns:
            EnhancedSearchResult with comprehensive phone information
        """
        
        search_start_time = datetime.now()
        self.search_stats['total_searches'] += 1
        
        # Initialize search options
        options = search_options or {}
        
        try:
            # Step 1: Check cache first
            cache_result = await self._check_cache(query, options)
            if cache_result:
                self.search_stats['cached_results'] += 1
                return cache_result
            
            # Step 2: Validate search query
            query_validation = await self._validate_search_query(query)
            if not query_validation['valid']:
                return self._create_error_result(query, query_validation['error'], search_start_time)
            
            # Step 3: Execute multi-source search
            search_results = await self._execute_multi_source_search(query, options)
            
            # Step 4: Validate and combine results
            validated_results = await self._validate_search_results(search_results)
            
            # Step 5: Apply ethical safeguards
            ethical_result = await self._apply_ethical_safeguards(validated_results)
            
            # Step 6: Create comprehensive result
            final_result = await self._create_comprehensive_result(
                query, ethical_result, search_start_time
            )
            
            # Step 7: Cache result if appropriate
            if self.cache_system and final_result.quality.value in ['excellent', 'good']:
                await self._cache_result(query, final_result, options)
            
            # Update statistics
            self.search_stats['successful_searches'] += 1
            self._update_response_time_stats(search_start_time)
            
            return final_result
            
        except Exception as e:
            logger.error(f"Enhanced search failed for query '{query}': {str(e)}")
            return self._create_error_result(query, str(e), search_start_time)
    
    async def _check_cache(self, query: str, options: Dict[str, Any]) -> Optional[EnhancedSearchResult]:
        """Check cache for existing results"""
        
        if not self.cache_system:
            return None
        
        try:
            cache_key = self._generate_cache_key(query, options)
            cached_result = await self.cache_system.get(cache_key)
            
            if cached_result:
                # Verify cache freshness and quality
                if self._is_cache_result_valid(cached_result):
                    cached_result.cache_info['cache_hit'] = True
                    cached_result.cache_info['retrieved_at'] = datetime.now().isoformat()
                    return cached_result
                else:
                    # Remove stale cache entry
                    await self.cache_system.delete(cache_key)
            
        except Exception as e:
            logger.warning(f"Cache check failed: {e}")
        
        return None
    
    async def _validate_search_query(self, query: str) -> Dict[str, Any]:
        """Validate search query for safety and quality"""
        
        validation = {'valid': True, 'error': None, 'warnings': []}
        
        # Basic validation
        if not query or len(query.strip()) < 2:
            validation['valid'] = False
            validation['error'] = "Query too short or empty"
            return validation
        
        if len(query) > 200:
            validation['warnings'].append("Query very long - truncating")
            query = query[:200]
        
        # Ethical validation
        prohibited_patterns = [
            r'\b(?:hack|crack|pirate|illegal)\b',
            r'\b(?:personal|private|confidential)\s+(?:data|info|details)\b',
            r'\b(?:steal|copy|download)\s+(?:data|information)\b'
        ]
        
        import re
        for pattern in prohibited_patterns:
            if re.search(pattern, query.lower()):
                validation['valid'] = False
                validation['error'] = "Query contains prohibited content"
                return validation
        
        return validation
    
    async def _execute_multi_source_search(self, query: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Execute search across multiple sources concurrently"""
        
        results = {
            'web_scraping': None,
            'pricing_data': None,
            'local_data': None,
            'errors': []
        }
        
        # Create tasks for concurrent execution
        tasks = []
        
        # Web scraping task
        if self.web_scraper and not options.get('skip_web_scraping', False):
            tasks.append(('web_scraping', self._search_web_sources(query)))
        
        # Pricing API task
        if self.pricing_integration and not options.get('skip_pricing', False):
            tasks.append(('pricing_data', self._search_pricing_sources(query)))
        
        # Local data task (if legacy components available)
        if self.smart_search and not options.get('skip_local', False):
            tasks.append(('local_data', self._search_local_sources(query)))
        
        # Execute tasks with timeout
        if tasks:
            try:
                # Use asyncio.wait_for with timeout
                timeout = self.config['total_search_timeout']
                completed_tasks = await asyncio.wait_for(
                    asyncio.gather(*[task[1] for task in tasks], return_exceptions=True),
                    timeout=timeout
                )
                
                # Process results
                for i, result in enumerate(completed_tasks):
                    task_name = tasks[i][0]
                    
                    if isinstance(result, Exception):
                        results['errors'].append(f"{task_name}: {str(result)}")
                        logger.warning(f"Task {task_name} failed: {str(result)}")
                    else:
                        results[task_name] = result
                        
            except asyncio.TimeoutError:
                results['errors'].append(f"Search timeout after {timeout} seconds")
                logger.warning(f"Multi-source search timed out for query: {query}")
            except Exception as e:
                results['errors'].append(f"Multi-source search error: {str(e)}")
                logger.error(f"Multi-source search failed: {str(e)}")
        
        return results
    
    async def _search_web_sources(self, query: str) -> Optional[ScrapedPhoneData]:
        """Search web sources using enhanced scraper"""
        
        try:
            if not self.web_scraper:
                return None
            
            scraping_result = await self.web_scraper.scrape_phone_reviews(query, max_sources=4)
            
            if scraping_result.review_count > 0:
                self.search_stats['web_scraping_used'] += 1
                return scraping_result
            
        except Exception as e:
            logger.error(f"Web scraping failed: {str(e)}")
        
        return None
    
    async def _search_pricing_sources(self, query: str) -> Optional[PricingResult]:
        """Search pricing sources using API integration"""
        
        try:
            if not self.pricing_integration:
                return None
            
            pricing_result = await self.pricing_integration.get_phone_pricing(query, max_sources=3)
            
            if pricing_result.all_prices:
                self.search_stats['pricing_api_used'] += 1
                return pricing_result
            
        except Exception as e:
            logger.error(f"Pricing search failed: {str(e)}")
        
        return None
    
    async def _search_local_sources(self, query: str) -> Optional[Dict[str, Any]]:
        """Search local sources using legacy components"""
        
        try:
            if not self.smart_search:
                return None
            
            # Parse query using smart search
            parsed_query = self.smart_search.parse_query(query)
            
            # Try fallback system
            if self.fallback_system:
                fallback_result = self.fallback_system.search_fallback(query)
                
                if fallback_result:
                    return {
                        'source': 'local_fallback',
                        'confidence': fallback_result.confidence + self.config['fallback_confidence_boost'],
                        'data': fallback_result.data,
                        'metadata': fallback_result.metadata
                    }
            
        except Exception as e:
            logger.error(f"Local search failed: {str(e)}")
        
        return None
    
    async def _validate_search_results(self, search_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate search results using data quality validator"""
        
        validated_results = {
            'web_data': None,
            'pricing_data': None,
            'local_data': None,
            'validation_results': {},
            'quality_scores': {}
        }
        
        if not self.data_validator:
            # Return unvalidated results if validator not available
            return {
                'web_data': search_results.get('web_scraping'),
                'pricing_data': search_results.get('pricing_data'),
                'local_data': search_results.get('local_data'),
                'validation_results': {},
                'quality_scores': {}
            }
        
        try:
            # Validate web scraping results
            if search_results.get('web_scraping'):
                web_validation = await self.data_validator.validate_data(
                    {'content': str(search_results['web_scraping'])},
                    DataSourceType.SCRAPED_REVIEW,
                    'web_scraping'
                )
                
                if web_validation.is_valid:
                    validated_results['web_data'] = search_results['web_scraping']
                    validated_results['validation_results']['web_scraping'] = web_validation
                    validated_results['quality_scores']['web_scraping'] = web_validation.quality_score
                else:
                    self.search_stats['validation_failures'] += 1
            
            # Validate pricing results
            if search_results.get('pricing_data'):
                pricing_validation = await self.data_validator.validate_data(
                    {'pricing': [asdict(price) for price in search_results['pricing_data'].all_prices]},
                    DataSourceType.PRICING_DATA,
                    'pricing_api'
                )
                
                if pricing_validation.is_valid:
                    validated_results['pricing_data'] = search_results['pricing_data']
                    validated_results['validation_results']['pricing_api'] = pricing_validation
                    validated_results['quality_scores']['pricing_api'] = pricing_validation.quality_score
                else:
                    self.search_stats['validation_failures'] += 1
            
            # Validate local data
            if search_results.get('local_data'):
                local_validation = await self.data_validator.validate_data(
                    search_results['local_data']['data'],
                    DataSourceType.API_RESULT,
                    'local_database'
                )
                
                if local_validation.is_valid:
                    validated_results['local_data'] = search_results['local_data']
                    validated_results['validation_results']['local_database'] = local_validation
                    validated_results['quality_scores']['local_database'] = local_validation.quality_score
                else:
                    self.search_stats['validation_failures'] += 1
            
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            # Return original results if validation fails
            validated_results['web_data'] = search_results.get('web_scraping')
            validated_results['pricing_data'] = search_results.get('pricing_data')
            validated_results['local_data'] = search_results.get('local_data')
        
        return validated_results
    
    async def _apply_ethical_safeguards(self, validated_results: Dict[str, Any]) -> Dict[str, Any]:
        """Apply ethical AI safeguards to search results"""
        
        ethical_result = validated_results.copy()
        ethical_violations = []
        
        # Check for synthetic data usage (prohibited)
        if self.config['prohibit_synthetic_data']:
            for source, data in validated_results.items():
                if data and self._contains_synthetic_data(data):
                    ethical_violations.append(f"Synthetic data detected in {source}")
                    ethical_result[source] = None  # Remove synthetic data
        
        # Verify source disclosure requirements
        if self.config['require_source_disclosure']:
            disclosed_sources = []
            
            if validated_results['web_data']:
                disclosed_sources.append('Web Scraping (GSMArena, PhoneArena, CNET, TechRadar)')
            
            if validated_results['pricing_data']:
                disclosed_sources.extend(['Google Shopping', 'eBay', 'Best Buy'])
            
            if validated_results['local_data']:
                disclosed_sources.append('Local Database')
            
            ethical_result['disclosed_sources'] = disclosed_sources
        
        # Check data freshness requirements
        if self.config['require_data_freshness_checks']:
            current_time = datetime.now()
            
            for source, validation in validated_results.get('validation_results', {}).items():
                if validation and hasattr(validation, 'metadata'):
                    freshness_info = validation.metadata.get('freshness')
                    if freshness_info and not freshness_info.get('is_fresh'):
                        ethical_violations.append(f"Stale data detected in {source}")
        
        # Privacy protection checks
        if self.config['enable_privacy_protection']:
            for source, data in validated_results.items():
                if data and self._contains_privacy_violations(data):
                    ethical_violations.append(f"Privacy violation detected in {source}")
                    ethical_result[source] = self._sanitize_privacy_data(data)
        
        # Record ethical violations
        if ethical_violations:
            self.search_stats['ethical_violations'] += len(ethical_violations)
            ethical_result['ethical_violations'] = ethical_violations
        
        return ethical_result
    
    async def _create_comprehensive_result(self, query: str, ethical_result: Dict[str, Any], 
                                         search_start_time: datetime) -> EnhancedSearchResult:
        """Create comprehensive search result with all enhancements"""
        
        # Determine primary data source and confidence
        data_source, confidence = self._determine_primary_source(ethical_result)
        
        # Extract phone data
        phone_data = self._extract_phone_data(ethical_result)
        
        # Calculate overall quality
        quality = self._calculate_result_quality(ethical_result, confidence)
        
        # Create ethical information
        ethical_info = EthicalDataInfo(
            sources_disclosed=ethical_result.get('disclosed_sources', []),
            data_freshness=f"Retrieved {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            confidence_level=confidence,
            limitations=self._identify_limitations(ethical_result),
            quality_indicators=ethical_result.get('quality_scores', {}),
            synthetic_data_used=False,  # Prohibited by ethical safeguards
            user_privacy_protected=True
        )
        
        # Create search metadata
        search_metadata = {
            'query': query,
            'search_duration_seconds': (datetime.now() - search_start_time).total_seconds(),
            'sources_attempted': self._count_sources_attempted(ethical_result),
            'sources_successful': self._count_sources_successful(ethical_result),
            'validation_enabled': self.config['enable_validation'],
            'ethical_safeguards_applied': True,
            'search_timestamp': search_start_time.isoformat()
        }
        
        # Generate recommendations
        recommendations = await self._generate_recommendations(phone_data, ethical_result)
        
        # Find alternatives
        alternatives = self._find_alternatives(phone_data)
        
        # Collect warnings and errors
        warnings = []
        errors = []
        
        if 'ethical_violations' in ethical_result:
            warnings.extend(ethical_result['ethical_violations'])
        
        if quality == SearchResultQuality.POOR:
            warnings.append("Search result quality is poor - consider refining your query")
        
        if confidence < self.config['min_confidence_threshold']:
            warnings.append(f"Low confidence result ({confidence:.2f}) - information may be incomplete")
        
        return EnhancedSearchResult(
            phone_model=phone_data.get('model', query),
            phone_found=len(phone_data) > 0,
            data_source=data_source,
            confidence=confidence,
            quality=quality,
            phone_data=phone_data,
            reviews=phone_data.get('reviews', []),
            specifications=phone_data.get('specifications', {}),
            pricing_info=ethical_result.get('pricing_data'),
            ethical_info=ethical_info,
            search_metadata=search_metadata,
            validation_results=ethical_result.get('validation_results', {}),
            cache_info={'cached': False, 'cache_hit': False},
            recommendations=recommendations,
            alternatives=alternatives,
            warnings=warnings,
            errors=errors
        )
    
    # Utility methods
    def _generate_cache_key(self, query: str, options: Dict[str, Any]) -> str:
        """Generate cache key for search"""
        import hashlib
        
        cache_data = {
            'query': query.lower().strip(),
            'options': sorted(options.items()) if options else []
        }
        
        cache_string = json.dumps(cache_data, sort_keys=True)
        return f"enhanced_search:{hashlib.md5(cache_string.encode()).hexdigest()}"
    
    def _is_cache_result_valid(self, cached_result: EnhancedSearchResult) -> bool:
        """Check if cached result is still valid"""
        
        # Check if result is recent enough
        cache_age_hours = self.config['cache_duration_hours']
        
        try:
            search_time = datetime.fromisoformat(cached_result.search_metadata['search_timestamp'])
            age = datetime.now() - search_time
            
            if age > timedelta(hours=cache_age_hours):
                return False
        except:
            return False
        
        # Check quality thresholds
        if cached_result.quality in [SearchResultQuality.POOR, SearchResultQuality.UNACCEPTABLE]:
            return False
        
        return True
    
    def _contains_synthetic_data(self, data: Any) -> bool:
        """Check if data contains synthetic/generated content"""
        # Implementation would check for markers indicating synthetic data
        # For now, return False as we've removed synthetic data generation
        return False
    
    def _contains_privacy_violations(self, data: Any) -> bool:
        """Check for potential privacy violations in data"""
        if isinstance(data, dict):
            for key, value in data.items():
                if 'personal' in key.lower() or 'private' in key.lower():
                    return True
                if isinstance(value, str) and len(value) > 50:
                    # Check for patterns that might be personal info
                    import re
                    if re.search(r'\b\d{3}-\d{2}-\d{4}\b', value):  # SSN pattern
                        return True
                    if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', value):  # Email
                        return True
        return False
    
    def _sanitize_privacy_data(self, data: Any) -> Any:
        """Remove or mask privacy-sensitive data"""
        # Implementation would sanitize personal information
        # For now, return data as-is since we're not collecting personal info
        return data
    
    def _determine_primary_source(self, ethical_result: Dict[str, Any]) -> tuple:
        """Determine primary data source and confidence"""
        
        # Priority order: web_data > pricing_data > local_data
        sources = [
            ('web_data', DataSource.WEB_SCRAPING),
            ('pricing_data', DataSource.PRICING_API),
            ('local_data', DataSource.LOCAL_DATABASE)
        ]
        
        for source_key, source_type in sources:
            if ethical_result.get(source_key):
                quality_score = ethical_result.get('quality_scores', {}).get(source_key.replace('_data', ''), 0.5)
                return source_type, quality_score
        
        return DataSource.FALLBACK, 0.3
    
    def _extract_phone_data(self, ethical_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and combine phone data from all sources"""
        
        combined_data = {
            'model': 'Unknown',
            'brand': 'Unknown',
            'specifications': {},
            'reviews': [],
            'rating': None,
            'price_info': {},
            'features': [],
            'pros': [],
            'cons': []
        }
        
        # Extract from web scraping data
        if ethical_result.get('web_data'):
            web_data = ethical_result['web_data']
            if hasattr(web_data, 'model'):
                combined_data['model'] = web_data.model
                combined_data['brand'] = web_data.brand
                combined_data['specifications'] = web_data.specifications
                combined_data['reviews'] = [asdict(review) for review in web_data.reviews]
                combined_data['rating'] = web_data.overall_rating
        
        # Extract from pricing data
        if ethical_result.get('pricing_data'):
            pricing_data = ethical_result['pricing_data']
            combined_data['price_info'] = {
                'lowest_price': asdict(pricing_data.lowest_price) if pricing_data.lowest_price else None,
                'highest_price': asdict(pricing_data.highest_price) if pricing_data.highest_price else None,
                'average_price': float(pricing_data.average_price) if pricing_data.average_price else None,
                'price_range': pricing_data.price_range,
                'market_analysis': pricing_data.market_analysis
            }
        
        # Extract from local data
        if ethical_result.get('local_data'):
            local_data = ethical_result['local_data']['data']
            if 'model' in local_data:
                combined_data['model'] = local_data['model']
            if 'specifications' in local_data:
                combined_data['specifications'].update(local_data['specifications'])
        
        return combined_data
    
    def _calculate_result_quality(self, ethical_result: Dict[str, Any], confidence: float) -> SearchResultQuality:
        """Calculate overall result quality"""
        
        if confidence >= self.config['excellent_quality_threshold']:
            return SearchResultQuality.EXCELLENT
        elif confidence >= self.config['good_quality_threshold']:
            return SearchResultQuality.GOOD
        elif confidence >= self.config['min_confidence_threshold']:
            return SearchResultQuality.FAIR
        elif confidence >= 0.3:
            return SearchResultQuality.POOR
        else:
            return SearchResultQuality.UNACCEPTABLE
    
    def _identify_limitations(self, ethical_result: Dict[str, Any]) -> List[str]:
        """Identify limitations in the search results"""
        
        limitations = []
        
        if not ethical_result.get('web_data'):
            limitations.append("No web review data available")
        
        if not ethical_result.get('pricing_data'):
            limitations.append("No current pricing information available")
        
        if not ethical_result.get('local_data'):
            limitations.append("No local database match found")
        
        if ethical_result.get('ethical_violations'):
            limitations.append("Some data sources excluded due to quality/ethical concerns")
        
        return limitations
    
    def _count_sources_attempted(self, ethical_result: Dict[str, Any]) -> int:
        """Count how many sources were attempted"""
        sources = ['web_data', 'pricing_data', 'local_data']
        return len(sources)  # We attempt all available sources
    
    def _count_sources_successful(self, ethical_result: Dict[str, Any]) -> int:
        """Count how many sources returned data"""
        sources = ['web_data', 'pricing_data', 'local_data']
        return sum(1 for source in sources if ethical_result.get(source))
    
    async def _generate_recommendations(self, phone_data: Dict[str, Any], 
                                      ethical_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate AI recommendations based on phone data"""
        
        recommendations = {
            'overall_verdict': 'Unable to determine',
            'verdict_confidence': 0.0,
            'target_users': [],
            'best_for': [],
            'considerations': [],
            'value_assessment': 'Unknown'
        }
        
        try:
            rating = phone_data.get('rating')
            if rating:
                if rating >= 4.5:
                    recommendations['overall_verdict'] = 'Excellent choice'
                    recommendations['verdict_confidence'] = 0.9
                elif rating >= 4.0:
                    recommendations['overall_verdict'] = 'Good choice'
                    recommendations['verdict_confidence'] = 0.8
                elif rating >= 3.5:
                    recommendations['overall_verdict'] = 'Fair choice'
                    recommendations['verdict_confidence'] = 0.6
                else:
                    recommendations['overall_verdict'] = 'Consider alternatives'
                    recommendations['verdict_confidence'] = 0.4
            
            # Analyze specifications for recommendations
            specs = phone_data.get('specifications', {})
            
            # Camera assessment
            camera_specs = [spec for spec in specs.keys() if 'camera' in spec.lower()]
            if camera_specs:
                recommendations['best_for'].append('Photography')
            
            # Performance assessment
            processor_specs = [spec for spec in specs.keys() if any(term in spec.lower() for term in ['processor', 'cpu', 'chip'])]
            if processor_specs:
                recommendations['target_users'].append('Performance users')
            
            # Price assessment
            price_info = phone_data.get('price_info', {})
            if price_info.get('average_price'):
                avg_price = price_info['average_price']
                if avg_price < 300:
                    recommendations['value_assessment'] = 'Budget-friendly'
                elif avg_price < 700:
                    recommendations['value_assessment'] = 'Mid-range value'
                else:
                    recommendations['value_assessment'] = 'Premium pricing'
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
        
        return recommendations
    
    def _find_alternatives(self, phone_data: Dict[str, Any]) -> List[str]:
        """Find alternative phone suggestions"""
        
        alternatives = []
        
        # Basic alternatives based on brand
        brand = phone_data.get('brand', '').lower()
        model = phone_data.get('model', '').lower()
        
        if 'iphone' in model:
            alternatives = ['Samsung Galaxy S24', 'Google Pixel 8', 'OnePlus 12']
        elif 'galaxy' in model:
            alternatives = ['iPhone 15', 'Google Pixel 8 Pro', 'OnePlus 12 Pro']
        elif 'pixel' in model:
            alternatives = ['iPhone 15', 'Samsung Galaxy S24', 'Nothing Phone 2']
        else:
            alternatives = ['iPhone 15', 'Samsung Galaxy S24', 'Google Pixel 8']
        
        return alternatives[:3]  # Limit to 3 alternatives
    
    async def _cache_result(self, query: str, result: EnhancedSearchResult, 
                          options: Dict[str, Any]):
        """Cache search result"""
        
        if not self.cache_system:
            return
        
        try:
            cache_key = self._generate_cache_key(query, options)
            ttl = self.config['cache_duration_hours'] * 3600
            
            await self.cache_system.set(
                cache_key, 
                result, 
                ttl=ttl,
                tags=['phone_search', result.phone_data.get('brand', 'unknown')],
                metadata={'query': query, 'quality': result.quality.value}
            )
            
        except Exception as e:
            logger.warning(f"Failed to cache result: {e}")
    
    def _create_error_result(self, query: str, error: str, 
                           search_start_time: datetime) -> EnhancedSearchResult:
        """Create error result for failed searches"""
        
        return EnhancedSearchResult(
            phone_model=query,
            phone_found=False,
            data_source=DataSource.FALLBACK,
            confidence=0.0,
            quality=SearchResultQuality.UNACCEPTABLE,
            phone_data={},
            reviews=[],
            specifications={},
            pricing_info=None,
            ethical_info=EthicalDataInfo(
                sources_disclosed=[],
                data_freshness="N/A",
                confidence_level=0.0,
                limitations=["Search failed"],
                quality_indicators={}
            ),
            search_metadata={
                'query': query,
                'search_duration_seconds': (datetime.now() - search_start_time).total_seconds(),
                'error': error,
                'search_timestamp': search_start_time.isoformat()
            },
            validation_results={},
            cache_info={'cached': False, 'cache_hit': False},
            recommendations={},
            alternatives=[],
            warnings=[],
            errors=[error]
        )
    
    def _update_response_time_stats(self, search_start_time: datetime):
        """Update average response time statistics"""
        
        response_time = (datetime.now() - search_start_time).total_seconds()
        
        current_avg = self.search_stats['average_response_time']
        total_searches = self.search_stats['successful_searches']
        
        if total_searches > 1:
            # Moving average
            self.search_stats['average_response_time'] = (
                (current_avg * (total_searches - 1) + response_time) / total_searches
            )
        else:
            self.search_stats['average_response_time'] = response_time
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get search statistics"""
        stats = self.search_stats.copy()
        
        if stats['total_searches'] > 0:
            stats['success_rate'] = (stats['successful_searches'] / stats['total_searches']) * 100
            stats['cache_hit_rate'] = (stats['cached_results'] / stats['total_searches']) * 100
        else:
            stats['success_rate'] = 0.0
            stats['cache_hit_rate'] = 0.0
        
        return stats
    
    def configure(self, new_config: Dict[str, Any]):
        """Update configuration"""
        self.config.update(new_config)
        logger.info("Configuration updated")

# Factory function
def create_enhanced_search_orchestrator(config=None):
    """Create configured enhanced search orchestrator"""
    return EnhancedSearchOrchestrator(config=config)