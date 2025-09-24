"""
Enhanced API Search Orchestrator - Works without external dependencies
Integrates with the Enhanced User-Friendly App
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import pandas as pd

# Import the enhanced web search agent
from .enhanced_api_web_search_agent import EnhancedAPIWebSearchAgent, APISearchResult

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class APISearchResult:
    """Standardized search result structure for enhanced API-based system"""
    phone_found: bool
    source: str  # 'local', 'api', 'hybrid', 'fallback', 'none'
    confidence: float
    phone_data: Dict[str, Any]
    search_metadata: Dict[str, Any]
    recommendations: Dict[str, Any]
    error_message: Optional[str] = None

class EnhancedAPISearchOrchestrator:
    """
    Enhanced API-based orchestrator for all phone search operations
    Works without external dependencies - uses local data + enhanced fallback
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the enhanced API-based search orchestrator"""
        
        # Default configuration optimized for enhanced user-friendly app
        self.config = config or {
            'local_confidence_threshold': 0.7,
            'api_confidence_threshold': 0.5,
            'enable_api_search': True,
            'enable_fallback_search': True,
            'enable_hybrid_search': True,
            'max_search_timeout': 15,  # Reduced timeout
            'cache_results': True,
            'log_searches': True,
            'prefer_api_over_local': False,
            'api_sources_limit': 2,  # Reduced for stability
            'enable_synthetic_generation': True,
            'enable_mock_data': False  # NO MOCK DATA  # Enable mock data when APIs fail
        }
        
        # Initialize enhanced web search agent
        if self.config['enable_api_search']:
            self.api_agent = EnhancedAPIWebSearchAgent({
                'max_concurrent_searches': self.config['api_sources_limit'],
                'search_timeout': self.config['max_search_timeout'],
                'min_confidence_threshold': self.config.get('api_confidence_threshold', 0.5),
                'enable_mock_data': self.config.get('enable_mock_data', True)
            })
        else:
            self.api_agent = None
        
        # Enhanced search statistics
        self.search_stats = {
            'total_searches': 0,
            'local_hits': 0,
            'api_searches': 0,
            'hybrid_results': 0,
            'fallback_searches': 0,
            'failed_searches': 0,
            'success_rate': 0.0,
            'average_confidence': 0.0,
            'api_success_rate': 0.0,
            'fallback_success_rate': 0.0,
            'mock_data_usage': 0,
            'cache_hits': 0
        }
        
        # Local data reference
        self.local_data = None
        self.local_search_function = None
        
        # Cache for orchestrator results
        self.orchestrator_cache = {}
        self.cache_expiry = 3600  # 1 hour
        
    def set_local_data_source(self, data: pd.DataFrame, search_function=None):
        """Set the local data source and search function"""
        self.local_data = data
        self.local_search_function = search_function or self._default_local_search
        logger.info(f"Enhanced orchestrator: Local data source set with {len(data) if data is not None else 0} records")
    
    def search_phone(self, query: str, search_options: Dict[str, Any] = None) -> APISearchResult:
        """
        Enhanced main search method - orchestrates local, API, and fallback search
        Works without external dependencies
        
        Args:
            query: User search query
            search_options: Additional search configuration
            
        Returns:
            APISearchResult with comprehensive phone information
        """
        
        search_start_time = datetime.now()
        self.search_stats['total_searches'] += 1
        
        logger.info(f"Enhanced orchestrator search for: {query}")
        
        # Check cache first
        cache_key = query.lower().strip()
        if self._is_orchestrator_cache_valid(cache_key):
            logger.info("Returning cached orchestrator results")
            self.search_stats['cache_hits'] += 1
            return self.orchestrator_cache[cache_key]['data']
        
        # Default search options
        options = search_options or {}
        force_api_search = options.get('force_api_search', False)
        skip_local_search = options.get('skip_local_search', False)
        enable_fallback = options.get('enable_fallback', True)
        
        try:
            # Step 1: Try local search first (if available and not skipped)
            local_result = None
            if not skip_local_search and not force_api_search and self.local_data is not None:
                local_result = self._search_local_database_enhanced(query)
                
                # If high confidence local result and not forcing API
                if (local_result and 
                    local_result.get('confidence', 0) >= self.config['local_confidence_threshold'] and
                    not force_api_search):
                    
                    self.search_stats['local_hits'] += 1
                    orchestrator_result = self._create_orchestrator_result(
                        phone_found=True,
                        source='local',
                        confidence=local_result['confidence'],
                        phone_data=local_result,
                        search_metadata={
                            'search_time': (datetime.now() - search_start_time).total_seconds(),
                            'local_search_used': True,
                            'api_search_used': False,
                            'fallback_used': False
                        }
                    )
                    
                    # Cache result
                    self._cache_orchestrator_results(cache_key, orchestrator_result)
                    return orchestrator_result
            
            # Step 2: Enhanced API search (if enabled and needed)
            api_result = None
            if (self.config['enable_api_search'] and self.api_agent and 
                (force_api_search or not local_result or 
                 local_result.get('confidence', 0) < self.config['local_confidence_threshold'])):
                
                api_result = self._search_enhanced_api_sources(query, local_result)
                
                if api_result and api_result.get('phone_found'):
                    self.search_stats['api_searches'] += 1
                    
                    # Update API success rate
                    self.search_stats['api_success_rate'] = (
                        self.search_stats['api_searches'] / max(1, self.search_stats['total_searches'])
                    )
                    
                    # Check if mock data was used
                    if 'mock' in str(api_result.get('sources', [])) or 'simulated' in str(api_result.get('search_metadata', {})):
                        self.search_stats['mock_data_usage'] += 1
                    
                    # Determine result type
                    if local_result and api_result.get('phone_found') and self.config['enable_hybrid_search']:
                        # Hybrid result
                        self.search_stats['hybrid_results'] += 1
                        orchestrator_result = self._create_hybrid_orchestrator_result(local_result, api_result, search_start_time)
                    else:
                        # API-only result
                        orchestrator_result = self._create_api_orchestrator_result(api_result, search_start_time)
                    
                    # Cache and return result
                    self._cache_orchestrator_results(cache_key, orchestrator_result)
                    return orchestrator_result
            
            # Step 3: Fallback to best available result
            if local_result and local_result.get('confidence', 0) > 0.4:
                # Use local result even with low confidence
                orchestrator_result = self._create_orchestrator_result(
                    phone_found=True,
                    source='local_low_confidence',
                    confidence=local_result['confidence'],
                    phone_data=local_result,
                    search_metadata={
                        'search_time': (datetime.now() - search_start_time).total_seconds(),
                        'local_search_used': True,
                        'api_search_used': False,
                        'fallback_used': True,
                        'warning': 'Low confidence local result used as fallback'
                    }
                )
                
                self.search_stats['fallback_searches'] += 1
                self._cache_orchestrator_results(cache_key, orchestrator_result)
                return orchestrator_result
            
            # Step 4: No results found
            self.search_stats['failed_searches'] += 1
            orchestrator_result = self._create_orchestrator_result(
                phone_found=False,
                source='none',
                confidence=0.0,
                phone_data={},
                search_metadata={
                    'search_time': (datetime.now() - search_start_time).total_seconds(),
                    'local_search_used': not skip_local_search,
                    'api_search_used': self.config['enable_api_search'],
                    'fallback_used': True,
                    'error': 'No phone found matching the query'
                },
                error_message=f"Phone '{query}' not found in local database or API sources"
            )
            
            self._cache_orchestrator_results(cache_key, orchestrator_result)
            return orchestrator_result
            
        except Exception as e:
            logger.error(f"Enhanced orchestrator search error: {str(e)}")
            self.search_stats['failed_searches'] += 1
            
            return self._create_orchestrator_result(
                phone_found=False,
                source='error',
                confidence=0.0,
                phone_data={},
                search_metadata={
                    'search_time': (datetime.now() - search_start_time).total_seconds(),
                    'error': str(e)
                },
                error_message=f"Search error: {str(e)}"
            )
    
    def _search_local_database_enhanced(self, query: str) -> Optional[Dict[str, Any]]:
        """Enhanced local database search"""
        
        if self.local_data is None or len(self.local_data) == 0:
            return None
        
        try:
            # Use the custom search function if provided
            if self.local_search_function:
                return self.local_search_function(query, self.local_data)
            else:
                return self._default_local_search(query, self.local_data)
                
        except Exception as e:
            logger.error(f"Local database search error: {str(e)}")
            return None
    
    def _default_local_search(self, query: str, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Default local search implementation"""
        
        query_lower = query.lower()
        
        # Search in product names
        if 'product' in data.columns:
            # Direct match
            direct_matches = data[data['product'].str.lower().str.contains(query_lower, na=False)]
            
            if len(direct_matches) > 0:
                # Get the most common phone from matches
                phone_counts = direct_matches['product'].value_counts()
                best_match = phone_counts.index[0]
                
                # Get data for this phone
                phone_data = direct_matches[direct_matches['product'] == best_match]
                
                # Calculate metrics
                avg_rating = phone_data['rating'].mean() if 'rating' in phone_data.columns else 4.0
                review_count = len(phone_data)
                
                # Sentiment distribution
                sentiment_dist = {}
                if 'sentiment_label' in phone_data.columns:
                    sentiment_counts = phone_data['sentiment_label'].value_counts(normalize=True) * 100
                    sentiment_dist = {
                        'positive': sentiment_counts.get('positive', 0),
                        'neutral': sentiment_counts.get('neutral', 0),
                        'negative': sentiment_counts.get('negative', 0)
                    }
                
                return {
                    'product_name': best_match,
                    'brand': phone_data['brand'].iloc[0] if 'brand' in phone_data.columns else 'Unknown',
                    'overall_rating': avg_rating,
                    'review_count': review_count,
                    'confidence': min(0.9, 0.6 + (review_count / 100)),  # Higher confidence for more reviews
                    'local_sentiment': sentiment_dist,
                    'source': 'local_database'
                }
        
        return None
    
    def _search_enhanced_api_sources(self, query: str, local_result: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """Search using enhanced API sources"""
        
        try:
            # Use the enhanced API web search agent
            api_result = self.api_agent.search_phone_external(query, max_sources=self.config['api_sources_limit'])
            
            if api_result and api_result.get('phone_found'):
                return api_result
                
        except Exception as e:
            logger.error(f"Enhanced API search error: {str(e)}")
        
        return None
    
    def _create_orchestrator_result(self, phone_found: bool, source: str, confidence: float, 
                                  phone_data: Dict[str, Any], search_metadata: Dict[str, Any],
                                  error_message: Optional[str] = None) -> APISearchResult:
        """Create standardized orchestrator result"""
        
        # Generate recommendations
        recommendations = self._generate_orchestrator_recommendations(phone_data, confidence, source)
        
        return APISearchResult(
            phone_found=phone_found,
            source=source,
            confidence=confidence,
            phone_data=phone_data,
            search_metadata=search_metadata,
            recommendations=recommendations,
            error_message=error_message
        )
    
    def _create_hybrid_orchestrator_result(self, local_result: Dict, api_result: Dict, search_start_time: datetime) -> APISearchResult:
        """Create hybrid result combining local and API data"""
        
        # Combine data from both sources
        combined_data = {
            'product_name': local_result.get('product_name', api_result.get('model', 'Unknown Phone')),
            'brand': local_result.get('brand', api_result.get('brand', 'Unknown')),
            
            # Ratings
            'local_rating': local_result.get('overall_rating'),
            'web_rating': api_result.get('overall_rating'),
            'combined_rating': (local_result.get('overall_rating', 0) + api_result.get('overall_rating', 0)) / 2,
            
            # Review counts
            'local_review_count': local_result.get('review_count', 0),
            'web_review_count': api_result.get('review_count', 0),
            'review_count': local_result.get('review_count', 0) + api_result.get('review_count', 0),
            
            # Features and specs
            'key_features': api_result.get('key_features', []),
            'specifications': api_result.get('specifications', {}),
            'pros': api_result.get('pros', []),
            'cons': api_result.get('cons', []),
            
            # Sentiment
            'local_sentiment': local_result.get('local_sentiment', {}),
            
            # Sources
            'total_sources': ['local_database'] + api_result.get('sources', []),
            
            # Price info
            'price_info': {
                'status': 'found' if api_result.get('price_range') else 'unavailable',
                'estimate': api_result.get('price_range', {}).get('estimate', 'N/A')
            }
        }
        
        # Calculate combined confidence
        local_confidence = local_result.get('confidence', 0.5)
        api_confidence = api_result.get('confidence', 0.5)
        combined_confidence = (local_confidence + api_confidence) / 2
        
        return self._create_orchestrator_result(
            phone_found=True,
            source='hybrid',
            confidence=combined_confidence,
            phone_data=combined_data,
            search_metadata={
                'search_time': (datetime.now() - search_start_time).total_seconds(),
                'local_search_used': True,
                'web_search_used': True,
                'sources_combined': len(combined_data['total_sources']),
                'web_sources': api_result.get('sources', []),
                'data_quality': api_result.get('search_metadata', {}).get('data_quality', 'mixed')
            }
        )
    
    def _create_api_orchestrator_result(self, api_result: Dict, search_start_time: datetime) -> APISearchResult:
        """Create API-only result"""
        
        return self._create_orchestrator_result(
            phone_found=api_result.get('phone_found', False),
            source='api',
            confidence=api_result.get('confidence', 0.5),
            phone_data=api_result,
            search_metadata={
                'search_time': (datetime.now() - search_start_time).total_seconds(),
                'local_search_used': False,
                'api_search_used': True,
                'api_sources': api_result.get('sources', []),
                'data_quality': api_result.get('search_metadata', {}).get('data_quality', 'api')
            }
        )
    
    def _generate_orchestrator_recommendations(self, phone_data: Dict, confidence: float, source: str) -> Dict[str, Any]:
        """Generate recommendations based on orchestrator analysis"""
        
        # Get rating
        rating = phone_data.get('combined_rating') or phone_data.get('overall_rating') or phone_data.get('rating', 0)
        
        # Generate verdict based on rating and confidence
        if rating >= 4.3 and confidence >= 0.8:
            verdict = "Highly Recommended"
            reason = f"Excellent rating ({rating:.1f}/5) with high data confidence"
        elif rating >= 4.0 and confidence >= 0.6:
            verdict = "Recommended"
            reason = f"Good rating ({rating:.1f}/5) with reliable data"
        elif rating >= 3.5 and confidence >= 0.5:
            verdict = "Consider with Research"
            reason = f"Moderate rating ({rating:.1f}/5) - review detailed feedback"
        elif confidence < 0.5:
            verdict = "Insufficient Data"
            reason = "Limited reliable information available"
        else:
            verdict = "Not Recommended"
            reason = f"Below average rating ({rating:.1f}/5)"
        
        # Add source-specific notes
        if source == 'hybrid':
            note = "Based on combined local database and web search data"
        elif source == 'local':
            note = "Based on curated local database"
        elif source == 'api':
            note = "Based on real-time web search data"
        else:
            note = f"Based on {source} data"
        
        # Reliability assessment
        if confidence >= 0.8:
            reliability = 'high'
        elif confidence >= 0.6:
            reliability = 'medium'
        else:
            reliability = 'low'
        
        return {
            'verdict': verdict,
            'reason': reason,
            'reliability': reliability,
            'note': note,
            'confidence_score': confidence,
            'data_source': source
        }
    
    def _is_orchestrator_cache_valid(self, cache_key: str) -> bool:
        """Check if cached orchestrator result is still valid"""
        if cache_key not in self.orchestrator_cache:
            return False
        
        cache_entry = self.orchestrator_cache[cache_key]
        cache_age = datetime.now() - datetime.fromisoformat(cache_entry['timestamp'])
        
        return cache_age.total_seconds() < self.cache_expiry
    
    def _cache_orchestrator_results(self, cache_key: str, result: APISearchResult):
        """Cache orchestrator search results"""
        self.orchestrator_cache[cache_key] = {
            'data': result,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get comprehensive search statistics"""
        
        total_searches = max(1, self.search_stats['total_searches'])  # Avoid division by zero
        
        stats = self.search_stats.copy()
        stats.update({
            'success_rate': ((total_searches - self.search_stats['failed_searches']) / total_searches) * 100,
            'local_hit_rate': (self.search_stats['local_hits'] / total_searches) * 100,
            'api_usage_rate': (self.search_stats['api_searches'] / total_searches) * 100,
            'hybrid_rate': (self.search_stats['hybrid_results'] / total_searches) * 100,
            'cache_hit_rate': (self.search_stats['cache_hits'] / total_searches) * 100,
            'mock_data_rate': (self.search_stats['mock_data_usage'] / total_searches) * 100
        })
        
        return stats


def create_api_search_orchestrator(local_data=None, config=None):
    """
    Factory function to create and configure the enhanced API search orchestrator
    
    Args:
        local_data: pandas DataFrame with local phone data
        config: Configuration dictionary
    
    Returns:
        Configured EnhancedAPISearchOrchestrator
    """
    
    orchestrator = EnhancedAPISearchOrchestrator(config=config)
    
    if local_data is not None:
        orchestrator.set_local_data_source(local_data)
    
    return orchestrator