"""
API-based Search Orchestrator for AI Phone Review Engine
Central coordination using API-based search instead of web scraping
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import pandas as pd

# Import API-based components
from .smart_search import SmartPhoneSearch, SearchQuery
from .api_web_search_agent import APIWebSearchAgent, integrate_api_search_with_system
from .fallback_search_system import FallbackSearchSystem, create_fallback_search_system

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class APISearchResult:
    """Standardized search result structure for API-based system"""
    phone_found: bool
    source: str  # 'local', 'api', 'hybrid', 'fallback', 'none'
    confidence: float
    phone_data: Dict[str, Any]
    search_metadata: Dict[str, Any]
    recommendations: Dict[str, Any]
    error_message: Optional[str] = None

class APISearchOrchestrator:
    """
    API-based orchestrator for all phone search operations
    Manages the decision flow between local, API, and fallback searches
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the API-based search orchestrator"""
        
        # Default configuration optimized for API usage
        self.config = config or {
            'local_confidence_threshold': 0.7,
            'api_confidence_threshold': 0.6,
            'enable_api_search': True,
            'enable_fallback_search': True,
            'enable_hybrid_search': True,
            'max_search_timeout': 30,
            'cache_results': True,
            'log_searches': True,
            'prefer_api_over_local': False,  # Whether to prefer API results over local
            'api_sources_limit': 3,
            'enable_synthetic_generation': True
        }
        
        # Initialize components
        self.smart_search = SmartPhoneSearch()
        
        # API-based web search agent
        if self.config['enable_api_search']:
            self.api_agent = APIWebSearchAgent({
                'max_concurrent_searches': self.config['api_sources_limit'],
                'search_timeout': self.config['max_search_timeout'],
                'min_confidence_threshold': self.config.get('api_confidence_threshold', 0.6)
            })
        else:
            self.api_agent = None
        
        # Fallback search system
        if self.config['enable_fallback_search']:
            self.fallback_system = FallbackSearchSystem({
                'min_match_confidence': 0.6,
                'use_fuzzy_matching': True,
                'generate_synthetic_data': self.config.get('enable_synthetic_generation', True),
                'offline_database_path': 'data/offline_phone_db.json'
            })
        else:
            self.fallback_system = None
        
        # Search statistics
        self.search_stats = {
            'total_searches': 0,
            'local_hits': 0,
            'api_searches': 0,
            'hybrid_results': 0,
            'fallback_searches': 0,
            'failed_searches': 0,
            'average_confidence': 0.0,
            'api_success_rate': 0.0,
            'fallback_success_rate': 0.0
        }
        
        # Local data reference
        self.local_data = None
        self.local_search_function = None
        
    def set_local_data_source(self, data: pd.DataFrame, search_function=None):
        """Set the local data source and search function"""
        self.local_data = data
        self.local_search_function = search_function or self._default_local_search
        logger.info(f"Local data source set with {len(data) if data is not None else 0} records")
    
    def search_phone(self, query: str, search_options: Dict[str, Any] = None) -> APISearchResult:
        """
        Main search method - orchestrates local, API, and fallback search
        
        Args:
            query: User search query
            search_options: Additional search configuration
            
        Returns:
            APISearchResult with comprehensive phone information
        """
        
        search_start_time = datetime.now()
        self.search_stats['total_searches'] += 1
        
        # Parse query
        parsed_query = self.smart_search.parse_query(query)
        logger.info(f"API-based search for: {parsed_query.phone_model} (Intent: {parsed_query.intent})")
        
        # Default search options
        options = search_options or {}
        force_api_search = options.get('force_api_search', False)
        skip_local_search = options.get('skip_local_search', False)
        enable_fallback = options.get('enable_fallback', True)
        
        try:
            # Step 1: Try local search first (unless skipped or API preferred)
            local_result = None
            if not skip_local_search and not (force_api_search or self.config.get('prefer_api_over_local', False)) and self.local_data is not None:
                local_result = self._search_local_database(parsed_query)
                
                # If high confidence local result and not forcing API
                if (local_result and 
                    local_result.get('confidence', 0) >= self.config['local_confidence_threshold'] and
                    not force_api_search):
                    
                    self.search_stats['local_hits'] += 1
                    return self._create_search_result(
                        phone_found=True,
                        source='local',
                        confidence=local_result['confidence'],
                        phone_data=local_result,
                        search_metadata={
                            'search_time': (datetime.now() - search_start_time).total_seconds(),
                            'query_parsed': parsed_query.__dict__,
                            'local_search_used': True,
                            'api_search_used': False,
                            'fallback_used': False
                        }
                    )
            
            # Step 2: API search (if enabled and needed)
            api_result = None
            if (self.config['enable_api_search'] and self.api_agent and 
                (force_api_search or not local_result or 
                 local_result.get('confidence', 0) < self.config['local_confidence_threshold'])):
                
                api_result = self._search_api_sources(query, local_result)
                
                if api_result and api_result.get('phone_found'):
                    self.search_stats['api_searches'] += 1
                    
                    # Update API success rate
                    self.search_stats['api_success_rate'] = (
                        self.search_stats['api_searches'] / max(1, self.search_stats['total_searches'])
                    )
                    
                    # Determine result type
                    if local_result and api_result.get('phone_found') and self.config['enable_hybrid_search']:
                        # Hybrid result
                        self.search_stats['hybrid_results'] += 1
                        return self._create_hybrid_result(local_result, api_result, search_start_time, parsed_query)
                    else:
                        # API-only result
                        return self._create_api_result(api_result, search_start_time, parsed_query)
            
            # Step 3: Fallback search (if enabled and no good results yet)
            if (enable_fallback and self.config['enable_fallback_search'] and 
                self.fallback_system and 
                (not api_result or not local_result or 
                 (api_result and api_result.get('confidence', 0) < 0.7) or
                 (local_result and local_result.get('confidence', 0) < 0.7))):
                
                fallback_result = self._search_fallback_system(parsed_query)
                
                if fallback_result:
                    self.search_stats['fallback_searches'] += 1
                    
                    # Update fallback success rate
                    self.search_stats['fallback_success_rate'] = (
                        self.search_stats['fallback_searches'] / max(1, self.search_stats['total_searches'])
                    )
                    
                    return self._create_fallback_result(fallback_result, search_start_time, parsed_query)
            
            # Step 4: Return best available result (local or API)
            if api_result and api_result.get('phone_found'):
                return self._create_api_result(api_result, search_start_time, parsed_query)
            elif local_result:
                self.search_stats['local_hits'] += 1
                return self._create_search_result(
                    phone_found=True,
                    source='local_low_confidence',
                    confidence=local_result['confidence'],
                    phone_data=local_result,
                    search_metadata={
                        'search_time': (datetime.now() - search_start_time).total_seconds(),
                        'query_parsed': parsed_query.__dict__,
                        'local_search_used': True,
                        'api_search_used': bool(api_result),
                        'fallback_used': False,
                        'note': 'Low confidence local result returned'
                    }
                )
            
            # Step 5: No results found anywhere
            self.search_stats['failed_searches'] += 1
            return self._create_no_results_found(query, search_start_time, parsed_query)
            
        except Exception as e:
            logger.error(f"API search orchestrator error: {str(e)}")
            self.search_stats['failed_searches'] += 1
            return self._create_error_result(query, str(e), search_start_time)
    
    def _search_local_database(self, parsed_query: SearchQuery) -> Optional[Dict[str, Any]]:
        """Search local database"""
        
        if self.local_data is None or not self.local_search_function:
            return None
            
        try:
            # Use the provided search function or default
            result = self.local_search_function(parsed_query, self.local_data)
            return result
            
        except Exception as e:
            logger.error(f"Local search error: {str(e)}")
            return None
    
    def _search_api_sources(self, query: str, local_result: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """Search using API-based sources"""
        
        if not self.api_agent:
            return None
            
        try:
            # Use the API agent to search external sources
            api_result = self.api_agent.search_phone_external(
                query, 
                max_sources=self.config['api_sources_limit']
            )
            
            if api_result and api_result.get('phone_found'):
                return api_result
                
        except Exception as e:
            logger.error(f"API search error: {str(e)}")
            
        return None
    
    def _search_fallback_system(self, parsed_query: SearchQuery) -> Optional[Any]:
        """Search using fallback system"""
        
        if not self.fallback_system:
            return None
            
        try:
            fallback_result = self.fallback_system.search_fallback(parsed_query.phone_model)
            return fallback_result
            
        except Exception as e:
            logger.error(f"Fallback search error: {str(e)}")
            
        return None
    
    def _default_local_search(self, parsed_query: SearchQuery, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Default local search implementation"""
        
        if data is None or len(data) == 0:
            return None
            
        # Simple fuzzy matching on product names
        phone_model = parsed_query.phone_model.lower()
        
        # Try exact matches first
        if 'product' in data.columns:
            exact_matches = data[data['product'].str.lower() == phone_model]
            if len(exact_matches) > 0:
                return self._format_local_result(exact_matches, 0.95)
        
        # Try partial matches
        if 'product' in data.columns:
            partial_matches = data[data['product'].str.lower().str.contains(phone_model, na=False)]
            if len(partial_matches) > 0:
                return self._format_local_result(partial_matches, 0.75)
        
        # Try brand matching
        if 'brand' in data.columns and parsed_query.brand:
            brand_matches = data[data['brand'].str.lower().str.contains(parsed_query.brand.lower(), na=False)]
            if len(brand_matches) > 0:
                return self._format_local_result(brand_matches, 0.6)
        
        return None
    
    def _format_local_result(self, matches: pd.DataFrame, confidence: float) -> Dict[str, Any]:
        """Format local search results"""
        
        # Aggregate data from matches
        avg_rating = matches['rating'].mean() if 'rating' in matches.columns else 4.0
        review_count = len(matches)
        
        # Get most common product name
        product_name = matches['product'].iloc[0] if len(matches) > 0 else "Unknown Phone"
        
        # Get brand if available
        brand = matches['brand'].iloc[0] if 'brand' in matches.columns and len(matches) > 0 else "Unknown"
        
        return {
            'model': product_name,
            'brand': brand,
            'confidence': confidence,
            'overall_rating': avg_rating,
            'review_count': review_count,
            'key_features': [],
            'specifications': {},
            'pros': [],
            'cons': [],
            'price_range': {},
            'reviews_summary': [],
            'sources': ['local_database'],
            'local_data': matches.to_dict('records')[:5]  # Include sample data
        }
    
    def _create_search_result(self, phone_found: bool, source: str, confidence: float, 
                            phone_data: Dict[str, Any], search_metadata: Dict[str, Any],
                            recommendations: Dict[str, Any] = None) -> APISearchResult:
        """Create standardized search result"""
        
        if not recommendations:
            recommendations = self._generate_basic_recommendations(phone_data)
        
        return APISearchResult(
            phone_found=phone_found,
            source=source,
            confidence=confidence,
            phone_data=phone_data,
            search_metadata=search_metadata,
            recommendations=recommendations
        )
    
    def _create_hybrid_result(self, local_result: Dict, api_result: Dict, 
                            search_start_time: datetime, parsed_query: SearchQuery) -> APISearchResult:
        """Create hybrid result combining local and API data"""
        
        # Merge the results intelligently
        hybrid_data = api_result.copy()  # Start with API data as base
        
        # Enhance with local data
        hybrid_data['local_rating'] = local_result.get('overall_rating')
        hybrid_data['local_review_count'] = local_result.get('review_count', 0)
        hybrid_data['confidence'] = max(local_result.get('confidence', 0), api_result.get('confidence', 0))
        
        # Combine sources
        api_sources = api_result.get('sources', [])
        hybrid_data['sources'] = ['local_database'] + api_sources
        
        return self._create_search_result(
            phone_found=True,
            source='hybrid',
            confidence=hybrid_data['confidence'],
            phone_data=hybrid_data,
            search_metadata={
                'search_time': (datetime.now() - search_start_time).total_seconds(),
                'query_parsed': parsed_query.__dict__,
                'local_search_used': True,
                'api_search_used': True,
                'fallback_used': False,
                'hybrid_merge': True,
                'local_confidence': local_result.get('confidence', 0),
                'api_confidence': api_result.get('confidence', 0)
            }
        )
    
    def _create_api_result(self, api_result: Dict, search_start_time: datetime, 
                          parsed_query: SearchQuery) -> APISearchResult:
        """Create result from API search"""
        
        return self._create_search_result(
            phone_found=True,
            source='api',
            confidence=api_result.get('confidence', 0.8),
            phone_data=api_result,
            search_metadata={
                'search_time': (datetime.now() - search_start_time).total_seconds(),
                'query_parsed': parsed_query.__dict__,
                'local_search_used': False,
                'api_search_used': True,
                'fallback_used': False,
                'api_sources': api_result.get('sources', [])
            }
        )
    
    def _create_fallback_result(self, fallback_result: Any, search_start_time: datetime, 
                               parsed_query: SearchQuery) -> APISearchResult:
        """Create result from fallback search"""
        
        fallback_data = fallback_result.data.copy()
        fallback_data['fallback_source'] = fallback_result.source
        fallback_data['confidence'] = fallback_result.confidence
        
        return self._create_search_result(
            phone_found=True,
            source='fallback',
            confidence=fallback_result.confidence,
            phone_data=fallback_data,
            search_metadata={
                'search_time': (datetime.now() - search_start_time).total_seconds(),
                'query_parsed': parsed_query.__dict__,
                'local_search_used': False,
                'api_search_used': False,
                'fallback_used': True,
                'fallback_type': fallback_result.source,
                'fallback_metadata': fallback_result.metadata
            }
        )
    
    def _create_no_results_found(self, query: str, search_start_time: datetime, 
                                parsed_query: SearchQuery) -> APISearchResult:
        """Create result when no phone is found"""
        
        return APISearchResult(
            phone_found=False,
            source='none',
            confidence=0.0,
            phone_data={
                'query': query,
                'parsed_query': parsed_query.phone_model,
                'suggestions': self._generate_search_suggestions(query)
            },
            search_metadata={
                'search_time': (datetime.now() - search_start_time).total_seconds(),
                'query_parsed': parsed_query.__dict__,
                'no_results_found': True
            },
            recommendations={'message': 'No phone found matching your query'}
        )
    
    def _create_error_result(self, query: str, error: str, search_start_time: datetime) -> APISearchResult:
        """Create error result"""
        
        return APISearchResult(
            phone_found=False,
            source='error',
            confidence=0.0,
            phone_data={'query': query, 'error_details': error},
            search_metadata={
                'search_time': (datetime.now() - search_start_time).total_seconds(),
                'error_occurred': True
            },
            recommendations={'message': 'Search error occurred'},
            error_message=error
        )
    
    def _generate_basic_recommendations(self, phone_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate basic recommendations for phone data"""
        
        rating = phone_data.get('overall_rating', 4.0)
        price_range = phone_data.get('price_range', {})
        
        if rating >= 4.5:
            verdict = "Excellent choice"
        elif rating >= 4.0:
            verdict = "Good choice"
        elif rating >= 3.5:
            verdict = "Fair choice"
        else:
            verdict = "Consider alternatives"
        
        return {
            'overall_verdict': verdict,
            'verdict_confidence': min(rating / 5.0, 1.0),
            'target_user': 'General users',
            'best_for': [],
            'considerations': []
        }
    
    def _generate_search_suggestions(self, query: str) -> List[str]:
        """Generate search suggestions when no results found"""
        
        suggestions = [
            "Try searching with just the phone model name",
            "Check spelling of the phone name",
            "Try searching for a similar phone model",
            "Browse popular phones in our database"
        ]
        
        return suggestions
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get current search statistics"""
        
        stats = self.search_stats.copy()
        
        # Calculate success rate
        if stats['total_searches'] > 0:
            successful_searches = (stats['local_hits'] + stats['api_searches'] + 
                                 stats['hybrid_results'] + stats['fallback_searches'])
            stats['success_rate'] = (successful_searches / stats['total_searches']) * 100
        else:
            stats['success_rate'] = 0.0
        
        return stats
    
    def clear_statistics(self):
        """Clear search statistics"""
        self.search_stats = {
            'total_searches': 0,
            'local_hits': 0,
            'api_searches': 0,
            'hybrid_results': 0,
            'fallback_searches': 0,
            'failed_searches': 0,
            'average_confidence': 0.0,
            'api_success_rate': 0.0,
            'fallback_success_rate': 0.0
        }


def create_api_search_orchestrator(local_data=None, config=None):
    """
    Factory function to create and configure the API-based search orchestrator
    
    Args:
        local_data: pandas DataFrame with local phone data
        config: configuration dictionary
        
    Returns:
        Configured APISearchOrchestrator instance
    """
    
    # Create orchestrator
    orchestrator = APISearchOrchestrator(config=config)
    
    # Set local data if provided
    if local_data is not None:
        orchestrator.set_local_data_source(local_data)
    
    return orchestrator