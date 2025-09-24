"""
Universal Phone Search Orchestrator for AI Phone Review Engine
Coordinates all search methods for maximum phone information coverage
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import statistics
from collections import defaultdict, Counter
import hashlib

# Import all search modules
from google_custom_search import GoogleCustomSearchEngine
from dynamic_source_discovery import DynamicSourceDiscovery
from ai_web_browser import AIWebBrowser
from specialized_review_scraper import SpecializedReviewScraper
from user_feedback_system import UserFeedbackSystem
from social_media_search import SocialMediaSearchEngine

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UniversalSearchResult:
    """Comprehensive search result from all sources"""
    phone_name: str
    search_timestamp: str
    total_sources_searched: int
    results_by_source: Dict[str, Any]
    aggregated_data: Dict[str, Any]
    confidence_scores: Dict[str, float]
    coverage_metrics: Dict[str, Any]
    search_duration_seconds: float

@dataclass
class SearchConfiguration:
    """Configuration for universal search"""
    enable_google_search: bool = True
    enable_source_discovery: bool = True
    enable_ai_browsing: bool = True
    enable_specialized_scraping: bool = True
    enable_user_feedback: bool = True
    enable_social_media: bool = True
    max_concurrent_searches: int = 6
    search_timeout_minutes: int = 10
    min_confidence_threshold: float = 0.3
    include_historical_data: bool = True
    deep_search_mode: bool = False
    regional_focus: Optional[str] = None
    language_preference: str = 'en'

class UniversalSearchOrchestrator:
    """
    Master orchestrator that coordinates all phone search methods
    for comprehensive coverage and maximum information retrieval
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize universal search orchestrator"""
        
        self.config = config or {
            'cache_enabled': True,
            'cache_duration_hours': 6,
            'results_storage_path': 'data/universal_search_results',
            'enable_result_fusion': True,
            'enable_quality_filtering': True,
            'parallel_execution': True,
            'fallback_search_enabled': True,
            'search_result_deduplication': True,
            'confidence_weighting': True,
            'adaptive_search_strategy': True
        }
        
        # Initialize all search engines
        self.search_engines = {}
        self._init_search_engines()
        
        # Search result cache
        self.result_cache = {}
        
        # Performance metrics
        self.search_metrics = {
            'total_searches': 0,
            'successful_searches': 0,
            'average_response_time': 0.0,
            'source_success_rates': defaultdict(float),
            'coverage_statistics': defaultdict(int)
        }
        
        # Ensure results directory exists
        Path(self.config['results_storage_path']).mkdir(parents=True, exist_ok=True)
    
    def _init_search_engines(self):
        """Initialize all search engines"""
        
        try:
            # Google Custom Search
            self.search_engines['google_search'] = GoogleCustomSearchEngine({
                'max_results_per_query': 20,
                'enable_content_extraction': True,
                'quality_threshold': 0.4
            })
            
            # Dynamic Source Discovery
            self.search_engines['source_discovery'] = DynamicSourceDiscovery({
                'discovery_depth': 2,
                'validate_discovered_sources': True,
                'min_quality_threshold': 0.5
            })
            
            # AI Web Browser
            self.search_engines['ai_browser'] = AIWebBrowser({
                'ai_provider': 'openai',
                'enable_content_cleaning': True,
                'enable_sentiment_analysis': True,
                'max_concurrent_requests': 3
            })
            
            # Specialized Review Scraper
            self.search_engines['specialized_scraper'] = SpecializedReviewScraper({
                'max_concurrent_requests': 4,
                'user_agent_rotation': True,
                'respect_robots_txt': True
            })
            
            # User Feedback System
            self.search_engines['user_feedback'] = UserFeedbackSystem({
                'enable_gamification': True,
                'auto_moderation_threshold': 0.7,
                'cache_results': True
            })
            
            # Social Media Search
            self.search_engines['social_search'] = SocialMediaSearchEngine({
                'enable_twitter': True,
                'enable_reddit': True,
                'enable_xda': True,
                'max_posts_per_platform': 25,
                'sentiment_analysis_enabled': True
            })
            
            logger.info(f"Initialized {len(self.search_engines)} search engines")
            
        except Exception as e:
            logger.error(f"Failed to initialize search engines: {e}")
    
    async def search_phone_comprehensive(self, phone_name: str, 
                                       search_config: SearchConfiguration = None) -> UniversalSearchResult:
        """
        Perform comprehensive phone search across all sources
        
        Args:
            phone_name: Name of the phone to search for
            search_config: Search configuration options
            
        Returns:
            UniversalSearchResult with aggregated data from all sources
        """
        
        if not search_config:
            search_config = SearchConfiguration()
        
        logger.info(f"Starting comprehensive search for: {phone_name}")
        start_time = datetime.now()
        
        # Check cache first
        cache_key = self._generate_cache_key(phone_name, search_config)
        if self.config['cache_enabled'] and cache_key in self.result_cache:
            cached_result = self.result_cache[cache_key]
            cache_age = datetime.now() - datetime.fromisoformat(cached_result.search_timestamp)
            if cache_age.total_seconds() < self.config['cache_duration_hours'] * 3600:
                logger.info("Returning cached comprehensive search result")
                return cached_result
        
        # Prepare search tasks
        search_tasks = self._prepare_search_tasks(phone_name, search_config)
        
        # Execute searches
        if self.config['parallel_execution']:
            search_results = await self._execute_parallel_searches(search_tasks, search_config)
        else:
            search_results = await self._execute_sequential_searches(search_tasks, search_config)
        
        # Aggregate and process results
        aggregated_data = await self._aggregate_search_results(search_results, phone_name)
        
        # Calculate metrics
        confidence_scores = self._calculate_confidence_scores(search_results)
        coverage_metrics = self._calculate_coverage_metrics(search_results, phone_name)
        
        # Create comprehensive result
        end_time = datetime.now()
        search_duration = (end_time - start_time).total_seconds()
        
        universal_result = UniversalSearchResult(
            phone_name=phone_name,
            search_timestamp=start_time.isoformat(),
            total_sources_searched=len([r for r in search_results.values() if r is not None]),
            results_by_source=search_results,
            aggregated_data=aggregated_data,
            confidence_scores=confidence_scores,
            coverage_metrics=coverage_metrics,
            search_duration_seconds=search_duration
        )
        
        # Cache result
        if self.config['cache_enabled']:
            self.result_cache[cache_key] = universal_result
        
        # Update metrics
        self._update_search_metrics(universal_result)
        
        # Save detailed results
        await self._save_search_results(universal_result)
        
        logger.info(f"Comprehensive search completed in {search_duration:.2f}s with {len(search_results)} sources")
        return universal_result
    
    def _prepare_search_tasks(self, phone_name: str, search_config: SearchConfiguration) -> Dict[str, Any]:
        """Prepare search tasks based on configuration"""
        
        tasks = {}
        
        if search_config.enable_google_search and 'google_search' in self.search_engines:
            tasks['google_search'] = {
                'engine': self.search_engines['google_search'],
                'method': 'search_phone_comprehensive',
                'args': [phone_name],
                'kwargs': {'max_results': 20}
            }
        
        if search_config.enable_source_discovery and 'source_discovery' in self.search_engines:
            tasks['source_discovery'] = {
                'engine': self.search_engines['source_discovery'],
                'method': 'discover_new_sources',
                'args': [],
                'kwargs': {
                    'search_terms': [f"{phone_name} review", f"{phone_name} specs"],
                    'discovery_method': 'comprehensive' if search_config.deep_search_mode else 'standard'
                }
            }
        
        if search_config.enable_ai_browsing and 'ai_browser' in self.search_engines:
            tasks['ai_browser'] = {
                'engine': self.search_engines['ai_browser'],
                'method': 'search_and_analyze',
                'args': [f"{phone_name} review"],
                'kwargs': {'max_pages': 15 if search_config.deep_search_mode else 10}
            }
        
        if search_config.enable_specialized_scraping and 'specialized_scraper' in self.search_engines:
            filter_config = {}
            if search_config.regional_focus:
                filter_config['region'] = search_config.regional_focus
            if search_config.language_preference:
                filter_config['language'] = search_config.language_preference
            
            tasks['specialized_scraper'] = {
                'engine': self.search_engines['specialized_scraper'],
                'method': 'scrape_phone_reviews',
                'args': [phone_name],
                'kwargs': {'source_filter': filter_config}
            }
        
        if search_config.enable_user_feedback and 'user_feedback' in self.search_engines:
            tasks['user_feedback'] = {
                'engine': self.search_engines['user_feedback'],
                'method': 'get_crowdsourced_phone_data',
                'args': [phone_name],
                'kwargs': {}
            }
        
        if search_config.enable_social_media and 'social_search' in self.search_engines:
            platforms = []
            if search_config.deep_search_mode:
                platforms = ['twitter', 'reddit', 'xda_developers', 'android_forums']
            else:
                platforms = ['twitter', 'reddit']
            
            tasks['social_search'] = {
                'engine': self.search_engines['social_search'],
                'method': 'search_all_platforms',
                'args': [phone_name],
                'kwargs': {'platforms': platforms}
            }
        
        return tasks
    
    async def _execute_parallel_searches(self, search_tasks: Dict[str, Any], 
                                       search_config: SearchConfiguration) -> Dict[str, Any]:
        """Execute search tasks in parallel"""
        
        logger.info(f"Executing {len(search_tasks)} search tasks in parallel")
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(search_config.max_concurrent_searches)
        
        async def execute_single_search(task_name: str, task_config: Dict[str, Any]):
            async with semaphore:
                try:
                    logger.info(f"Starting {task_name} search")
                    start_time = datetime.now()
                    
                    # Get the method from the engine
                    engine = task_config['engine']
                    method_name = task_config['method']
                    args = task_config.get('args', [])
                    kwargs = task_config.get('kwargs', {})
                    
                    method = getattr(engine, method_name)
                    
                    # Execute search with timeout
                    result = await asyncio.wait_for(
                        method(*args, **kwargs),
                        timeout=search_config.search_timeout_minutes * 60
                    )
                    
                    duration = (datetime.now() - start_time).total_seconds()
                    logger.info(f"Completed {task_name} search in {duration:.2f}s")
                    
                    return {
                        'result': result,
                        'success': True,
                        'duration': duration,
                        'error': None
                    }
                    
                except asyncio.TimeoutError:
                    logger.warning(f"Search {task_name} timed out")
                    return {
                        'result': None,
                        'success': False,
                        'duration': search_config.search_timeout_minutes * 60,
                        'error': 'timeout'
                    }
                    
                except Exception as e:
                    logger.error(f"Search {task_name} failed: {e}")
                    return {
                        'result': None,
                        'success': False,
                        'duration': 0,
                        'error': str(e)
                    }
        
        # Execute all tasks
        task_futures = {
            task_name: execute_single_search(task_name, task_config)
            for task_name, task_config in search_tasks.items()
        }
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*task_futures.values(), return_exceptions=True)
        
        # Combine results with task names
        search_results = {}
        for task_name, result in zip(task_futures.keys(), results):
            if isinstance(result, Exception):
                search_results[task_name] = {
                    'result': None,
                    'success': False,
                    'duration': 0,
                    'error': str(result)
                }
            else:
                search_results[task_name] = result
        
        return search_results
    
    async def _execute_sequential_searches(self, search_tasks: Dict[str, Any], 
                                         search_config: SearchConfiguration) -> Dict[str, Any]:
        """Execute search tasks sequentially"""
        
        logger.info(f"Executing {len(search_tasks)} search tasks sequentially")
        
        search_results = {}
        
        for task_name, task_config in search_tasks.items():
            try:
                logger.info(f"Starting {task_name} search")
                start_time = datetime.now()
                
                # Get the method from the engine
                engine = task_config['engine']
                method_name = task_config['method']
                args = task_config.get('args', [])
                kwargs = task_config.get('kwargs', {})
                
                method = getattr(engine, method_name)
                
                # Execute search
                result = await method(*args, **kwargs)
                
                duration = (datetime.now() - start_time).total_seconds()
                logger.info(f"Completed {task_name} search in {duration:.2f}s")
                
                search_results[task_name] = {
                    'result': result,
                    'success': True,
                    'duration': duration,
                    'error': None
                }
                
            except Exception as e:
                logger.error(f"Search {task_name} failed: {e}")
                search_results[task_name] = {
                    'result': None,
                    'success': False,
                    'duration': 0,
                    'error': str(e)
                }
        
        return search_results
    
    async def _aggregate_search_results(self, search_results: Dict[str, Any], 
                                      phone_name: str) -> Dict[str, Any]:
        """Aggregate and process search results from all sources"""
        
        logger.info("Aggregating search results from all sources")
        
        aggregated = {
            'phone_name': phone_name,
            'specifications': {},
            'reviews': [],
            'ratings': [],
            'prices': [],
            'pros_cons': {'pros': [], 'cons': []},
            'user_opinions': [],
            'social_sentiment': {},
            'availability': {},
            'alternatives': [],
            'key_features': [],
            'expert_opinions': [],
            'community_feedback': {},
            'technical_details': {},
            'source_summary': {}
        }
        
        # Process results from each source
        for source_name, search_result in search_results.items():
            if not search_result['success'] or not search_result['result']:
                continue
            
            try:
                result_data = search_result['result']
                
                if source_name == 'google_search':
                    self._process_google_search_results(result_data, aggregated)
                
                elif source_name == 'source_discovery':
                    self._process_source_discovery_results(result_data, aggregated)
                
                elif source_name == 'ai_browser':
                    self._process_ai_browser_results(result_data, aggregated)
                
                elif source_name == 'specialized_scraper':
                    self._process_specialized_scraper_results(result_data, aggregated)
                
                elif source_name == 'user_feedback':
                    self._process_user_feedback_results(result_data, aggregated)
                
                elif source_name == 'social_search':
                    self._process_social_search_results(result_data, aggregated)
                
                # Add source summary
                aggregated['source_summary'][source_name] = {
                    'items_found': self._count_result_items(result_data),
                    'search_duration': search_result['duration'],
                    'success': search_result['success']
                }
                
            except Exception as e:
                logger.error(f"Failed to process results from {source_name}: {e}")
        
        # Post-process aggregated data
        aggregated = await self._post_process_aggregated_data(aggregated)
        
        return aggregated
    
    def _process_google_search_results(self, results: List[Any], aggregated: Dict[str, Any]):
        """Process Google Custom Search results"""
        
        for result in results:
            if hasattr(result, 'reviews') and result.reviews:
                aggregated['reviews'].extend(result.reviews)
            
            if hasattr(result, 'specifications') and result.specifications:
                for spec, value in result.specifications.items():
                    if spec not in aggregated['specifications']:
                        aggregated['specifications'][spec] = []
                    aggregated['specifications'][spec].append(value)
            
            if hasattr(result, 'rating') and result.rating:
                aggregated['ratings'].append({
                    'rating': result.rating,
                    'scale': getattr(result, 'rating_scale', 'unknown'),
                    'source': 'google_search'
                })
    
    def _process_source_discovery_results(self, results: Dict[str, Any], aggregated: Dict[str, Any]):
        """Process dynamic source discovery results"""
        
        # Add discovered sources information
        for category, sources in results.items():
            for source in sources:
                if hasattr(source, 'sample_urls'):
                    aggregated['expert_opinions'].extend([
                        {'source': source.name, 'url': url, 'type': 'discovered_source'}
                        for url in source.sample_urls[:3]
                    ])
    
    def _process_ai_browser_results(self, results: List[Any], aggregated: Dict[str, Any]):
        """Process AI web browser results"""
        
        for analysis in results:
            if hasattr(analysis, 'extracted_phones'):
                for phone_data in analysis.extracted_phones:
                    if phone_data.get('specifications'):
                        for spec, value in phone_data['specifications'].items():
                            if spec not in aggregated['specifications']:
                                aggregated['specifications'][spec] = []
                            aggregated['specifications'][spec].append(value)
                    
                    if phone_data.get('pros_cons'):
                        aggregated['pros_cons']['pros'].extend(phone_data['pros_cons'].get('pros', []))
                        aggregated['pros_cons']['cons'].extend(phone_data['pros_cons'].get('cons', []))
            
            # Add AI analysis summary
            if hasattr(analysis, 'summary'):
                aggregated['expert_opinions'].append({
                    'source': 'AI Analysis',
                    'content': analysis.summary,
                    'confidence': analysis.extraction_confidence,
                    'type': 'ai_analysis'
                })
    
    def _process_specialized_scraper_results(self, results: List[Any], aggregated: Dict[str, Any]):
        """Process specialized review scraper results"""
        
        for review in results:
            # Add review
            aggregated['reviews'].append({
                'title': review.review_title,
                'author': review.author,
                'content': review.content,
                'rating': review.rating,
                'rating_scale': review.rating_scale,
                'source': review.source_name,
                'url': review.review_url,
                'language': review.language,
                'region': review.region
            })
            
            # Add specifications
            for spec, value in review.specifications.items():
                if spec not in aggregated['specifications']:
                    aggregated['specifications'][spec] = []
                aggregated['specifications'][spec].append(value)
            
            # Add pros/cons
            aggregated['pros_cons']['pros'].extend(review.pros)
            aggregated['pros_cons']['cons'].extend(review.cons)
    
    def _process_user_feedback_results(self, result: Any, aggregated: Dict[str, Any]):
        """Process user feedback system results"""
        
        if not result:
            return
        
        # Add user reviews
        aggregated['user_opinions'].extend(result.user_reviews)
        
        # Add aggregated rating
        if result.aggregated_rating:
            aggregated['ratings'].append({
                'rating': result.aggregated_rating,
                'scale': '5.0',
                'source': 'user_community'
            })
        
        # Add community pros/cons
        aggregated['pros_cons']['pros'].extend(result.pros_cons.get('pros', []))
        aggregated['pros_cons']['cons'].extend(result.pros_cons.get('cons', []))
        
        # Add community feedback summary
        aggregated['community_feedback'] = {
            'contributor_count': result.contributor_count,
            'confidence': result.community_confidence,
            'last_updated': result.last_updated
        }
    
    def _process_social_search_results(self, results: Dict[str, Any], aggregated: Dict[str, Any]):
        """Process social media search results"""
        
        all_sentiments = []
        all_mentions = []
        
        for platform, result in results.items():
            # Collect sentiment data
            if result.sentiment_analysis:
                all_sentiments.append(result.sentiment_analysis.get('average_sentiment', 0))
            
            # Collect user opinions from posts
            for post in result.posts:
                aggregated['user_opinions'].append({
                    'platform': platform,
                    'author': post.author,
                    'content': post.content[:200] + '...' if len(post.content) > 200 else post.content,
                    'sentiment': post.sentiment_score,
                    'engagement': post.engagement_metrics,
                    'url': post.url,
                    'published_at': post.published_at
                })
                
                # Collect phone mentions
                all_mentions.extend(post.phone_mentions)
        
        # Calculate overall social sentiment
        if all_sentiments:
            aggregated['social_sentiment'] = {
                'overall_sentiment': statistics.mean(all_sentiments),
                'platforms_analyzed': len(results),
                'total_posts': sum(r.total_posts for r in results.values()),
                'top_mentions': [{'phone': phone, 'count': count} 
                               for phone, count in Counter(all_mentions).most_common(5)]
            }
    
    async def _post_process_aggregated_data(self, aggregated: Dict[str, Any]) -> Dict[str, Any]:
        """Post-process and clean aggregated data"""
        
        # Deduplicate specifications
        for spec_name, values in aggregated['specifications'].items():
            unique_values = list(set(str(v) for v in values if v))
            aggregated['specifications'][spec_name] = unique_values[:5]  # Limit to top 5
        
        # Deduplicate and rank pros/cons
        pro_counts = Counter(aggregated['pros_cons']['pros'])
        con_counts = Counter(aggregated['pros_cons']['cons'])
        
        aggregated['pros_cons']['pros'] = [
            pro for pro, count in pro_counts.most_common(10)
        ]
        aggregated['pros_cons']['cons'] = [
            con for con, count in con_counts.most_common(10)
        ]
        
        # Calculate average rating
        if aggregated['ratings']:
            # Normalize ratings to 5-point scale
            normalized_ratings = []
            for rating_data in aggregated['ratings']:
                rating = rating_data['rating']
                scale = rating_data.get('scale', '5')
                
                if scale == '10':
                    normalized_ratings.append(rating / 2)
                elif scale == '100':
                    normalized_ratings.append(rating / 20)
                else:
                    normalized_ratings.append(rating)
            
            aggregated['average_rating'] = statistics.mean(normalized_ratings)
            aggregated['rating_count'] = len(normalized_ratings)
        
        # Limit collections to reasonable sizes
        aggregated['reviews'] = aggregated['reviews'][:20]
        aggregated['user_opinions'] = aggregated['user_opinions'][:30]
        aggregated['expert_opinions'] = aggregated['expert_opinions'][:15]
        
        return aggregated
    
    def _calculate_confidence_scores(self, search_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate confidence scores for each search source"""
        
        confidence_scores = {}
        
        for source_name, search_result in search_results.items():
            if not search_result['success']:
                confidence_scores[source_name] = 0.0
                continue
            
            result_data = search_result['result']
            base_confidence = 0.5
            
            # Adjust confidence based on result data
            if result_data:
                if hasattr(result_data, '__len__') and len(result_data) > 0:
                    base_confidence += 0.2
                
                # Source-specific confidence adjustments
                if source_name == 'ai_browser':
                    if hasattr(result_data, '__iter__'):
                        avg_confidence = statistics.mean([
                            getattr(item, 'extraction_confidence', 0.5) 
                            for item in result_data
                        ])
                        base_confidence = avg_confidence
                
                elif source_name == 'user_feedback':
                    if hasattr(result_data, 'community_confidence'):
                        base_confidence = result_data.community_confidence
                
                elif source_name == 'specialized_scraper':
                    if hasattr(result_data, '__iter__'):
                        avg_confidence = statistics.mean([
                            getattr(item, 'confidence_score', 0.5) 
                            for item in result_data
                        ])
                        base_confidence = avg_confidence
            
            confidence_scores[source_name] = min(base_confidence, 1.0)
        
        return confidence_scores
    
    def _calculate_coverage_metrics(self, search_results: Dict[str, Any], phone_name: str) -> Dict[str, Any]:
        """Calculate coverage metrics for the search"""
        
        metrics = {
            'sources_attempted': len(search_results),
            'sources_successful': len([r for r in search_results.values() if r['success']]),
            'total_items_found': 0,
            'coverage_by_category': {},
            'geographic_coverage': set(),
            'language_coverage': set(),
            'data_freshness': {}
        }
        
        for source_name, search_result in search_results.items():
            if search_result['success'] and search_result['result']:
                result_data = search_result['result']
                items_count = self._count_result_items(result_data)
                metrics['total_items_found'] += items_count
                
                # Categorize coverage
                if source_name in ['google_search', 'ai_browser']:
                    metrics['coverage_by_category']['web_content'] = items_count
                elif source_name in ['specialized_scraper']:
                    metrics['coverage_by_category']['expert_reviews'] = items_count
                elif source_name in ['social_search']:
                    metrics['coverage_by_category']['social_media'] = items_count
                elif source_name in ['user_feedback']:
                    metrics['coverage_by_category']['community_data'] = items_count
        
        metrics['success_rate'] = metrics['sources_successful'] / metrics['sources_attempted'] if metrics['sources_attempted'] > 0 else 0
        
        return metrics
    
    def _count_result_items(self, result_data: Any) -> int:
        """Count the number of items in result data"""
        
        if not result_data:
            return 0
        
        if isinstance(result_data, list):
            return len(result_data)
        elif isinstance(result_data, dict):
            if 'total_posts' in result_data:
                return result_data['total_posts']
            elif 'user_reviews' in result_data:
                return len(result_data['user_reviews'])
            else:
                return sum(len(v) if isinstance(v, list) else 1 for v in result_data.values())
        
        return 1
    
    def _generate_cache_key(self, phone_name: str, search_config: SearchConfiguration) -> str:
        """Generate cache key for search results"""
        
        config_hash = hashlib.md5(str(asdict(search_config)).encode()).hexdigest()[:8]
        return f"{phone_name.lower().replace(' ', '_')}_{config_hash}"
    
    def _update_search_metrics(self, result: UniversalSearchResult):
        """Update search performance metrics"""
        
        self.search_metrics['total_searches'] += 1
        
        if result.coverage_metrics['success_rate'] > 0.5:
            self.search_metrics['successful_searches'] += 1
        
        # Update average response time
        current_avg = self.search_metrics['average_response_time']
        total_searches = self.search_metrics['total_searches']
        self.search_metrics['average_response_time'] = (
            (current_avg * (total_searches - 1) + result.search_duration_seconds) / total_searches
        )
        
        # Update source success rates
        for source, confidence in result.confidence_scores.items():
            self.search_metrics['source_success_rates'][source] = (
                (self.search_metrics['source_success_rates'][source] + confidence) / 2
            )
    
    async def _save_search_results(self, result: UniversalSearchResult):
        """Save detailed search results to file"""
        
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{result.phone_name.replace(' ', '_')}_{timestamp}.json"
            filepath = Path(self.config['results_storage_path']) / filename
            
            # Convert result to dict for JSON serialization
            result_dict = asdict(result)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Saved detailed search results to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save search results: {e}")
    
    async def get_search_suggestions(self, partial_phone_name: str) -> List[str]:
        """Get search suggestions based on partial phone name"""
        
        suggestions = []
        
        # Common phone models and brands
        phone_patterns = [
            "iPhone 14", "iPhone 14 Pro", "iPhone 15", "iPhone 15 Pro",
            "Galaxy S23", "Galaxy S24", "Galaxy S24 Ultra",
            "Pixel 7", "Pixel 8", "Pixel 8 Pro",
            "OnePlus 11", "OnePlus 12", "OnePlus Open",
            "Xiaomi 13", "Xiaomi 14", "Nothing Phone 2"
        ]
        
        partial_lower = partial_phone_name.lower()
        
        for phone in phone_patterns:
            if partial_lower in phone.lower():
                suggestions.append(phone)
        
        return suggestions[:10]
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get search performance statistics"""
        
        return {
            'performance_metrics': self.search_metrics,
            'cache_statistics': {
                'cached_results': len(self.result_cache),
                'cache_enabled': self.config['cache_enabled']
            },
            'engine_status': {
                engine_name: 'active' if engine else 'inactive'
                for engine_name, engine in self.search_engines.items()
            }
        }
    
    async def clear_cache(self):
        """Clear search result cache"""
        
        self.result_cache.clear()
        logger.info("Search result cache cleared")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all search engines"""
        
        health_status = {}
        
        for engine_name, engine in self.search_engines.items():
            try:
                # Basic availability check
                if hasattr(engine, 'health_check'):
                    status = await engine.health_check()
                else:
                    status = {'status': 'unknown', 'available': engine is not None}
                
                health_status[engine_name] = status
                
            except Exception as e:
                health_status[engine_name] = {
                    'status': 'error',
                    'error': str(e),
                    'available': False
                }
        
        return {
            'overall_health': 'healthy' if all(
                status.get('available', False) for status in health_status.values()
            ) else 'degraded',
            'engine_health': health_status,
            'timestamp': datetime.now().isoformat()
        }

# Factory function
def create_universal_search_orchestrator(config=None):
    """Create configured universal search orchestrator"""
    return UniversalSearchOrchestrator(config=config)