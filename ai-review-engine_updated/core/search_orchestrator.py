"""
Search Orchestrator for AI Phone Review Engine
Central coordination between local database search and web search agent
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import pandas as pd

# Import core components
from .smart_search import SmartPhoneSearch, SearchQuery
from .web_search_agent import WebSearchAgent, integrate_web_search_with_system

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Standardized search result structure"""
    phone_found: bool
    source: str  # 'local', 'web', 'hybrid', 'none'
    confidence: float
    phone_data: Dict[str, Any]
    search_metadata: Dict[str, Any]
    recommendations: Dict[str, Any]
    error_message: Optional[str] = None

class SearchOrchestrator:
    """
    Central orchestrator for all phone search operations
    Manages the decision flow between local and web searches
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the search orchestrator"""
        
        # Default configuration
        self.config = config or {
            'local_confidence_threshold': 0.8,
            'enable_web_fallback': True,
            'enable_hybrid_search': True,
            'max_search_timeout': 30,
            'cache_results': True,
            'log_searches': True
        }
        
        # Initialize components
        self.smart_search = SmartPhoneSearch()
        self.web_agent = WebSearchAgent() if self.config['enable_web_fallback'] else None
        
        # Search statistics
        self.search_stats = {
            'total_searches': 0,
            'local_hits': 0,
            'web_searches': 0,
            'hybrid_results': 0,
            'failed_searches': 0,
            'average_confidence': 0.0
        }
        
        # Local data reference
        self.local_data = None
        self.local_search_function = None
        
    def set_local_data_source(self, data: pd.DataFrame, search_function=None):
        """Set the local data source and search function"""
        self.local_data = data
        self.local_search_function = search_function or self._default_local_search
        logger.info(f"Local data source set with {len(data) if data is not None else 0} records")
    
    def search_phone(self, query: str, search_options: Dict[str, Any] = None) -> SearchResult:
        """
        Main search method - orchestrates local and web search
        
        Args:
            query: User search query
            search_options: Additional search configuration
            
        Returns:
            SearchResult with comprehensive phone information
        """
        
        search_start_time = datetime.now()
        self.search_stats['total_searches'] += 1
        
        # Parse query
        parsed_query = self.smart_search.parse_query(query)
        logger.info(f"Searching for: {parsed_query.phone_model} (Intent: {parsed_query.intent})")
        
        # Default search options
        options = search_options or {}
        force_web_search = options.get('force_web_search', False)
        skip_local_search = options.get('skip_local_search', False)
        
        try:
            # Step 1: Try local search first (unless skipped)
            local_result = None
            if not skip_local_search and self.local_data is not None:
                local_result = self._search_local_database(parsed_query)
                
                if local_result and local_result.get('confidence', 0) >= self.config['local_confidence_threshold']:
                    # High confidence local result - return it
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
                            'web_search_used': False
                        }
                    )
            
            # Step 2: Web search (if enabled and needed)
            if (self.config['enable_web_fallback'] and self.web_agent and 
                (force_web_search or not local_result or local_result.get('confidence', 0) < self.config['local_confidence_threshold'])):
                
                web_result = self._search_web_sources(query, local_result)
                
                if web_result and web_result.get('phone_found'):
                    self.search_stats['web_searches'] += 1
                    
                    # Determine result type
                    if local_result and web_result.get('phone_found'):
                        # Hybrid result
                        self.search_stats['hybrid_results'] += 1
                        return self._create_hybrid_result(local_result, web_result, search_start_time, parsed_query)
                    else:
                        # Web-only result
                        return self._create_web_result(web_result, search_start_time, parsed_query)
            
            # Step 3: Return local result if available (even with low confidence)
            if local_result:
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
                        'web_search_used': False,
                        'note': 'Low confidence local result returned'
                    }
                )
            
            # Step 4: No results found anywhere
            self.search_stats['failed_searches'] += 1
            return self._create_no_results_found(query, search_start_time, parsed_query)
            
        except Exception as e:
            logger.error(f"Search orchestrator error: {str(e)}")
            self.search_stats['failed_searches'] += 1
            return self._create_error_result(query, str(e), search_start_time)
    
    def _search_local_database(self, parsed_query: SearchQuery) -> Optional[Dict[str, Any]]:
        """Search local database"""
        
        if not self.local_data is not None or not self.local_search_function:
            return None
            
        try:
            # Use the provided search function or default
            result = self.local_search_function(parsed_query, self.local_data)
            return result
            
        except Exception as e:
            logger.error(f"Local search error: {str(e)}")
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
        if parsed_query.brand and 'brand' in data.columns:
            brand_matches = data[data['brand'].str.lower() == parsed_query.brand.lower()]
            if len(brand_matches) > 0:
                return self._format_local_result(brand_matches, 0.60)
        
        return None
    
    def _format_local_result(self, matches: pd.DataFrame, confidence: float) -> Dict[str, Any]:
        """Format local search results"""
        
        if len(matches) == 0:
            return None
            
        # Get the first match (or aggregate multiple matches)
        primary_match = matches.iloc[0]
        
        # Calculate statistics
        avg_rating = matches['rating'].mean() if 'rating' in matches.columns else None
        review_count = len(matches)
        sentiment_dist = None
        
        if 'sentiment_label' in matches.columns:
            sentiment_counts = matches['sentiment_label'].value_counts()
            total = len(matches)
            sentiment_dist = {
                'positive': (sentiment_counts.get('positive', 0) / total) * 100,
                'negative': (sentiment_counts.get('negative', 0) / total) * 100,
                'neutral': (sentiment_counts.get('neutral', 0) / total) * 100
            }
        
        return {
            'phone_found': True,
            'confidence': confidence,
            'product_name': primary_match.get('product', 'Unknown'),
            'brand': primary_match.get('brand', 'Unknown'),
            'overall_rating': avg_rating,
            'review_count': review_count,
            'sentiment_distribution': sentiment_dist,
            'local_data': True,
            'raw_matches': matches.to_dict('records')[:5]  # Limit to 5 records
        }
    
    def _search_web_sources(self, query: str, local_result: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """Search web sources using the web agent"""
        
        if not self.web_agent:
            return None
            
        try:
            return integrate_web_search_with_system(query, local_result)
        except Exception as e:
            logger.error(f"Web search error: {str(e)}")
            return None
    
    def _create_search_result(self, phone_found: bool, source: str, confidence: float, 
                            phone_data: Dict, search_metadata: Dict, 
                            recommendations: Dict = None, error_message: str = None) -> SearchResult:
        """Create standardized search result"""
        
        # Generate recommendations if not provided
        if not recommendations and phone_found:
            recommendations = self._generate_recommendations(phone_data, confidence, source)
        
        return SearchResult(
            phone_found=phone_found,
            source=source,
            confidence=confidence,
            phone_data=phone_data,
            search_metadata=search_metadata,
            recommendations=recommendations or {},
            error_message=error_message
        )
    
    def _create_hybrid_result(self, local_result: Dict, web_result: Dict, 
                            search_start_time: datetime, parsed_query: SearchQuery) -> SearchResult:
        """Create hybrid result combining local and web data"""
        
        # Combine data intelligently
        combined_data = {
            'product_name': local_result.get('product_name') or web_result['phone_data']['product_name'],
            'brand': local_result.get('brand') or web_result['phone_data']['brand'],
            'local_rating': local_result.get('overall_rating'),
            'web_rating': web_result['phone_data'].get('overall_rating'),
            'combined_rating': self._calculate_combined_rating(local_result, web_result),
            'local_review_count': local_result.get('review_count', 0),
            'web_review_count': web_result['phone_data'].get('review_count', 0),
            'total_sources': web_result['phone_data'].get('web_sources', []) + ['local_database'],
            'web_features': web_result['phone_data'].get('key_features', []),
            'web_pros': web_result['phone_data'].get('pros', []),
            'web_cons': web_result['phone_data'].get('cons', []),
            'price_info': web_result['phone_data'].get('price_info'),
            'local_sentiment': local_result.get('sentiment_distribution'),
            'web_sentiment': web_result.get('analysis', {}).get('sentiment_analysis')
        }
        
        # Calculate combined confidence
        local_conf = local_result.get('confidence', 0)
        web_conf = web_result.get('confidence', 0)
        combined_confidence = (local_conf + web_conf) / 2
        
        return self._create_search_result(
            phone_found=True,
            source='hybrid',
            confidence=combined_confidence,
            phone_data=combined_data,
            search_metadata={
                'search_time': (datetime.now() - search_start_time).total_seconds(),
                'query_parsed': parsed_query.__dict__,
                'local_search_used': True,
                'web_search_used': True,
                'local_confidence': local_conf,
                'web_confidence': web_conf,
                'sources_combined': len(combined_data['total_sources'])
            }
        )
    
    def _create_web_result(self, web_result: Dict, search_start_time: datetime, 
                          parsed_query: SearchQuery) -> SearchResult:
        """Create web-only result"""
        
        return self._create_search_result(
            phone_found=web_result['phone_found'],
            source='web',
            confidence=web_result['confidence'],
            phone_data=web_result['phone_data'],
            search_metadata={
                'search_time': (datetime.now() - search_start_time).total_seconds(),
                'query_parsed': parsed_query.__dict__,
                'local_search_used': False,
                'web_search_used': True,
                'web_sources': web_result['phone_data'].get('web_sources', []),
                'data_quality': web_result.get('analysis', {}).get('data_quality', {})
            },
            recommendations=web_result.get('recommendations', {})
        )
    
    def _create_no_results_found(self, query: str, search_start_time: datetime, 
                                parsed_query: SearchQuery) -> SearchResult:
        """Create no results found response"""
        
        suggestions = [
            f"Try searching for just the brand: '{parsed_query.brand}'" if parsed_query.brand else "Try searching for just the phone brand",
            "Check spelling of the phone model",
            "Try using a more general term",
            "Search for similar phones in the same category"
        ]
        
        return self._create_search_result(
            phone_found=False,
            source='none',
            confidence=0.0,
            phone_data={
                'search_query': query,
                'parsed_query': parsed_query.phone_model,
                'suggestions': suggestions,
                'message': 'Phone not found in local database or web sources'
            },
            search_metadata={
                'search_time': (datetime.now() - search_start_time).total_seconds(),
                'query_parsed': parsed_query.__dict__,
                'local_search_used': self.local_data is not None,
                'web_search_used': self.web_agent is not None
            }
        )
    
    def _create_error_result(self, query: str, error_message: str, 
                           search_start_time: datetime) -> SearchResult:
        """Create error result"""
        
        return self._create_search_result(
            phone_found=False,
            source='error',
            confidence=0.0,
            phone_data={'search_query': query},
            search_metadata={
                'search_time': (datetime.now() - search_start_time).total_seconds(),
                'error': True
            },
            error_message=error_message
        )
    
    def _calculate_combined_rating(self, local_result: Dict, web_result: Dict) -> float:
        """Calculate weighted combined rating"""
        
        local_rating = local_result.get('overall_rating')
        web_rating = web_result['phone_data'].get('overall_rating')
        
        if local_rating and web_rating:
            # Weight by review counts
            local_count = local_result.get('review_count', 1)
            web_count = web_result['phone_data'].get('review_count', 1)
            total_count = local_count + web_count
            
            return ((local_rating * local_count) + (web_rating * web_count)) / total_count
        elif local_rating:
            return local_rating
        elif web_rating:
            return web_rating
        else:
            return None
    
    def _generate_recommendations(self, phone_data: Dict, confidence: float, source: str) -> Dict[str, Any]:
        """Generate recommendations based on search results"""
        
        recommendations = {
            'confidence_level': 'high' if confidence > 0.8 else 'medium' if confidence > 0.6 else 'low',
            'data_source_note': f"Recommendations based on {source} data",
            'reliability': 'high' if source == 'local' else 'medium' if source == 'hybrid' else 'medium-low'
        }
        
        # Rating-based recommendations
        rating = phone_data.get('overall_rating') or phone_data.get('combined_rating')
        if rating:
            if rating >= 4.5:
                recommendations['verdict'] = 'Highly Recommended'
                recommendations['reason'] = f'Excellent rating of {rating:.1f}/5.0'
            elif rating >= 4.0:
                recommendations['verdict'] = 'Recommended'
                recommendations['reason'] = f'Good rating of {rating:.1f}/5.0'
            elif rating >= 3.5:
                recommendations['verdict'] = 'Consider with Caution'
                recommendations['reason'] = f'Average rating of {rating:.1f}/5.0'
            else:
                recommendations['verdict'] = 'Not Recommended'
                recommendations['reason'] = f'Below average rating of {rating:.1f}/5.0'
        
        # Add source-specific notes
        if source == 'web':
            recommendations['note'] = 'Based on real-time web data. Consider checking multiple sources.'
        elif source == 'hybrid':
            recommendations['note'] = 'Based on combined local and web data for comprehensive analysis.'
        elif source == 'local':
            recommendations['note'] = 'Based on curated local database with high confidence.'
        
        return recommendations
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get search statistics"""
        
        total = self.search_stats['total_searches']
        if total == 0:
            return {'message': 'No searches performed yet'}
        
        return {
            'total_searches': total,
            'success_rate': ((total - self.search_stats['failed_searches']) / total) * 100,
            'local_hit_rate': (self.search_stats['local_hits'] / total) * 100,
            'web_search_rate': (self.search_stats['web_searches'] / total) * 100,
            'hybrid_result_rate': (self.search_stats['hybrid_results'] / total) * 100,
            'failure_rate': (self.search_stats['failed_searches'] / total) * 100
        }
    
    def configure(self, config_updates: Dict[str, Any]):
        """Update configuration"""
        self.config.update(config_updates)
        
        # Reinitialize web agent if needed
        if 'enable_web_fallback' in config_updates:
            if config_updates['enable_web_fallback'] and not self.web_agent:
                self.web_agent = WebSearchAgent()
            elif not config_updates['enable_web_fallback']:
                self.web_agent = None
        
        logger.info(f"Configuration updated: {config_updates}")


# Convenience functions for easy integration
def create_search_orchestrator(local_data: pd.DataFrame = None, 
                             search_function=None, 
                             config: Dict[str, Any] = None) -> SearchOrchestrator:
    """Create and configure a search orchestrator"""
    
    orchestrator = SearchOrchestrator(config)
    
    if local_data is not None:
        orchestrator.set_local_data_source(local_data, search_function)
    
    return orchestrator

def quick_search(query: str, local_data: pd.DataFrame = None) -> SearchResult:
    """Quick search without configuration"""
    
    orchestrator = create_search_orchestrator(local_data)
    return orchestrator.search_phone(query)