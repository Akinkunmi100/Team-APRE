"""
Google Custom Search API Integration for Universal Phone Search
Enables searching the entire web for phone information and reviews
"""

import asyncio
import aiohttp
import logging
import json
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from urllib.parse import quote_plus, urlencode, urlparse
from bs4 import BeautifulSoup
import hashlib
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GoogleSearchResult:
    """Structure for Google search results"""
    title: str
    url: str
    snippet: str
    page_content: Optional[str]
    extracted_data: Dict[str, Any]
    relevance_score: float
    source_type: str  # review, specification, news, forum, etc.
    scraped_at: str

@dataclass
class UniversalPhoneData:
    """Comprehensive phone data from universal web search"""
    phone_model: str
    search_results: List[GoogleSearchResult]
    specifications: Dict[str, Any]
    reviews_summary: Dict[str, Any]
    news_mentions: List[Dict[str, Any]]
    forum_discussions: List[Dict[str, Any]]
    price_mentions: List[Dict[str, Any]]
    images: List[str]
    videos: List[str]
    overall_sentiment: str
    confidence: float
    sources_count: int
    search_metadata: Dict[str, Any]

class GoogleCustomSearch:
    """Google Custom Search API integration for universal phone search"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize Google Custom Search"""
        
        self.config = config or {
            'api_key': os.getenv('GOOGLE_SEARCH_API_KEY'),
            'search_engine_id': os.getenv('GOOGLE_SEARCH_ENGINE_ID'),
            'max_results_per_query': 10,
            'max_concurrent_requests': 3,
            'enable_content_extraction': True,
            'enable_smart_filtering': True,
            'request_timeout': 30,
            'rate_limit_delay': 1.0
        }
        
        # Validate configuration
        if not self.config['api_key'] or not self.config['search_engine_id']:
            logger.warning("Google Search API credentials not found. Universal search will be limited.")
            self.enabled = False
        else:
            self.enabled = True
        
        # Search query templates for different types of information
        self.search_templates = {
            'general': '{phone_model} phone review specifications',
            'specifications': '{phone_model} phone specs technical specifications',
            'reviews': '{phone_model} phone review 2024 2023',
            'pricing': '{phone_model} phone price buy cost',
            'comparison': '{phone_model} vs comparison alternative',
            'news': '{phone_model} phone news announcement launch',
            'forums': '{phone_model} phone discussion forum reddit',
            'problems': '{phone_model} phone problems issues complaints',
            'accessories': '{phone_model} phone case accessories',
            'tutorials': '{phone_model} phone setup tips tricks'
        }
        
        # Content extraction patterns
        self.extraction_patterns = {
            'specifications': {
                'display': r'(?:display|screen).*?(\d+\.?\d*\s*(?:inch|"))',
                'battery': r'(?:battery|mah).*?(\d+\.?\d*\s*mah)',
                'camera': r'(?:camera|mp).*?(\d+\.?\d*\s*mp)',
                'ram': r'(?:ram|memory).*?(\d+\.?\d*\s*gb)',
                'storage': r'(?:storage|rom).*?(\d+\.?\d*\s*gb)',
                'processor': r'(?:processor|cpu|chipset).*?([a-z0-9\s]+)',
                'os': r'(?:android|ios|operating system).*?(android\s+\d+|ios\s+\d+)',
                'price': r'(?:price|cost|\$).*?(\$?\d+[,\d]*)'
            },
            'sentiment_indicators': {
                'positive': ['excellent', 'great', 'amazing', 'fantastic', 'love', 'best', 'perfect', 'outstanding'],
                'negative': ['terrible', 'awful', 'worst', 'hate', 'bad', 'poor', 'disappointing', 'issues'],
                'neutral': ['okay', 'decent', 'average', 'fair', 'reasonable', 'standard']
            }
        }
        
        self.session = None
    
    async def search_phone_universally(self, phone_query: str, search_depth: str = 'comprehensive') -> UniversalPhoneData:
        """
        Perform universal web search for phone information
        
        Args:
            phone_query: Phone model to search for
            search_depth: 'basic', 'standard', or 'comprehensive'
            
        Returns:
            UniversalPhoneData with comprehensive information from across the web
        """
        
        if not self.enabled:
            return self._create_empty_result(phone_query, "Google Search API not configured")
        
        logger.info(f"Starting universal web search for: {phone_query}")
        
        # Initialize session
        await self._init_session()
        
        try:
            # Determine search queries based on depth
            search_queries = self._get_search_queries(phone_query, search_depth)
            
            # Execute searches concurrently
            search_results = await self._execute_concurrent_searches(search_queries)
            
            # Extract content from top results
            enhanced_results = await self._extract_content_from_results(search_results)
            
            # Process and analyze all data
            universal_data = await self._process_universal_data(phone_query, enhanced_results)
            
            return universal_data
            
        finally:
            await self._cleanup_session()
    
    def _get_search_queries(self, phone_query: str, search_depth: str) -> List[Tuple[str, str]]:
        """Generate search queries based on depth setting"""
        
        queries = []
        
        if search_depth == 'basic':
            # Just general and specifications
            templates = ['general', 'specifications']
        elif search_depth == 'standard':
            # Most important categories
            templates = ['general', 'specifications', 'reviews', 'pricing']
        else:  # comprehensive
            # All available templates
            templates = list(self.search_templates.keys())
        
        for template_name in templates:
            template = self.search_templates[template_name]
            query = template.format(phone_model=phone_query)
            queries.append((template_name, query))
        
        return queries
    
    async def _execute_concurrent_searches(self, search_queries: List[Tuple[str, str]]) -> Dict[str, List[GoogleSearchResult]]:
        """Execute multiple search queries concurrently"""
        
        results = {}
        
        # Create tasks for concurrent execution
        semaphore = asyncio.Semaphore(self.config['max_concurrent_requests'])
        
        async def search_single_query(query_type: str, query: str):
            async with semaphore:
                try:
                    # Add delay for rate limiting
                    await asyncio.sleep(self.config['rate_limit_delay'])
                    return await self._perform_google_search(query_type, query)
                except Exception as e:
                    logger.error(f"Search failed for {query_type}: {str(e)}")
                    return []
        
        # Execute all searches
        tasks = [search_single_query(qtype, query) for qtype, query in search_queries]
        completed_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(completed_results):
            query_type = search_queries[i][0]
            
            if isinstance(result, Exception):
                logger.error(f"Search error for {query_type}: {result}")
                results[query_type] = []
            else:
                results[query_type] = result or []
        
        return results
    
    async def _perform_google_search(self, query_type: str, query: str) -> List[GoogleSearchResult]:
        """Perform a single Google Custom Search"""
        
        try:
            # Prepare search parameters
            params = {
                'key': self.config['api_key'],
                'cx': self.config['search_engine_id'],
                'q': query,
                'num': min(self.config['max_results_per_query'], 10),
                'safe': 'active',
                'lr': 'lang_en'
            }
            
            # Execute search
            search_url = f"https://www.googleapis.com/customsearch/v1?{urlencode(params)}"
            
            async with self.session.get(search_url, timeout=self.config['request_timeout']) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_google_search_results(data, query_type)
                else:
                    logger.warning(f"Google Search API returned status {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Google Search error: {str(e)}")
            return []
    
    def _parse_google_search_results(self, data: Dict, query_type: str) -> List[GoogleSearchResult]:
        """Parse Google Search API response"""
        
        results = []
        
        for item in data.get('items', []):
            try:
                # Calculate relevance score
                relevance_score = self._calculate_relevance_score(item, query_type)
                
                # Determine source type
                source_type = self._determine_source_type(item['link'])
                
                result = GoogleSearchResult(
                    title=item.get('title', ''),
                    url=item.get('link', ''),
                    snippet=item.get('snippet', ''),
                    page_content=None,  # Will be populated later if needed
                    extracted_data={},  # Will be populated during content extraction
                    relevance_score=relevance_score,
                    source_type=source_type,
                    scraped_at=datetime.now().isoformat()
                )
                
                results.append(result)
                
            except Exception as e:
                logger.warning(f"Error parsing search result: {e}")
                continue
        
        # Sort by relevance score
        results.sort(key=lambda r: r.relevance_score, reverse=True)
        
        return results
    
    async def _extract_content_from_results(self, search_results: Dict[str, List[GoogleSearchResult]]) -> Dict[str, List[GoogleSearchResult]]:
        """Extract content from the most relevant search results"""
        
        if not self.config['enable_content_extraction']:
            return search_results
        
        enhanced_results = {}
        
        for query_type, results in search_results.items():
            enhanced_results[query_type] = []
            
            # Extract content from top results only
            top_results = results[:5]  # Limit to top 5 per query type
            
            for result in top_results:
                try:
                    # Extract content from the page
                    page_content, extracted_data = await self._extract_page_content(result.url, query_type)
                    
                    # Update result with extracted data
                    enhanced_result = GoogleSearchResult(
                        title=result.title,
                        url=result.url,
                        snippet=result.snippet,
                        page_content=page_content[:2000] if page_content else None,  # Limit content length
                        extracted_data=extracted_data,
                        relevance_score=result.relevance_score,
                        source_type=result.source_type,
                        scraped_at=result.scraped_at
                    )
                    
                    enhanced_results[query_type].append(enhanced_result)
                    
                except Exception as e:
                    logger.warning(f"Content extraction failed for {result.url}: {e}")
                    # Add original result without content
                    enhanced_results[query_type].append(result)
        
        return enhanced_results
    
    async def _extract_page_content(self, url: str, query_type: str) -> Tuple[Optional[str], Dict[str, Any]]:
        """Extract content and structured data from a web page"""
        
        try:
            # Get page content
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            async with self.session.get(url, headers=headers, timeout=15) as response:
                if response.status != 200:
                    return None, {}
                
                content = await response.text()
                
                # Parse with BeautifulSoup
                soup = BeautifulSoup(content, 'html.parser')
                
                # Remove unwanted elements
                for element in soup(['script', 'style', 'nav', 'footer', 'header', 'ads']):
                    element.decompose()
                
                # Extract text content
                page_text = soup.get_text(separator=' ', strip=True)
                
                # Extract structured data based on query type
                extracted_data = self._extract_structured_data(page_text, soup, query_type)
                
                return page_text, extracted_data
                
        except Exception as e:
            logger.warning(f"Page content extraction failed for {url}: {e}")
            return None, {}
    
    def _extract_structured_data(self, page_text: str, soup: BeautifulSoup, query_type: str) -> Dict[str, Any]:
        """Extract structured data from page content"""
        
        extracted = {}
        
        try:
            # Extract specifications
            if query_type in ['general', 'specifications']:
                extracted['specifications'] = self._extract_specifications(page_text)
            
            # Extract pricing information
            if query_type in ['pricing', 'general']:
                extracted['pricing'] = self._extract_pricing_info(page_text)
            
            # Extract review scores and sentiment
            if query_type in ['reviews', 'general']:
                extracted['review_sentiment'] = self._analyze_sentiment(page_text)
                extracted['ratings'] = self._extract_ratings(page_text, soup)
            
            # Extract pros and cons
            if query_type == 'reviews':
                extracted['pros_cons'] = self._extract_pros_cons(page_text, soup)
            
            # Extract images
            extracted['images'] = self._extract_images(soup)
            
        except Exception as e:
            logger.warning(f"Structured data extraction failed: {e}")
        
        return extracted
    
    def _extract_specifications(self, text: str) -> Dict[str, str]:
        """Extract phone specifications from text"""
        
        specs = {}
        text_lower = text.lower()
        
        for spec_type, pattern in self.extraction_patterns['specifications'].items():
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                # Take the most common match
                from collections import Counter
                most_common = Counter(matches).most_common(1)
                if most_common:
                    specs[spec_type] = most_common[0][0]
        
        return specs
    
    def _extract_pricing_info(self, text: str) -> Dict[str, Any]:
        """Extract pricing information from text"""
        
        pricing = {}
        
        # Extract price mentions
        price_pattern = r'\$(\d+(?:,\d{3})*(?:\.\d{2})?)'
        prices = re.findall(price_pattern, text)
        
        if prices:
            # Convert to float and analyze
            float_prices = []
            for price in prices:
                try:
                    float_prices.append(float(price.replace(',', '')))
                except:
                    continue
            
            if float_prices:
                pricing['min_price'] = min(float_prices)
                pricing['max_price'] = max(float_prices)
                pricing['avg_price'] = sum(float_prices) / len(float_prices)
                pricing['price_mentions'] = len(float_prices)
        
        return pricing
    
    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of the text"""
        
        text_lower = text.lower()
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        
        for sentiment, words in self.extraction_patterns['sentiment_indicators'].items():
            for word in words:
                sentiment_counts[sentiment] += text_lower.count(word)
        
        total_sentiment = sum(sentiment_counts.values())
        
        if total_sentiment == 0:
            return {'overall': 'neutral', 'confidence': 0.0}
        
        # Calculate percentages
        sentiment_percentages = {
            k: v / total_sentiment for k, v in sentiment_counts.items()
        }
        
        # Determine overall sentiment
        max_sentiment = max(sentiment_percentages.items(), key=lambda x: x[1])
        
        return {
            'overall': max_sentiment[0],
            'confidence': max_sentiment[1],
            'breakdown': sentiment_percentages,
            'total_indicators': total_sentiment
        }
    
    def _extract_ratings(self, text: str, soup: BeautifulSoup) -> List[float]:
        """Extract rating scores from text and HTML"""
        
        ratings = []
        
        # Common rating patterns
        rating_patterns = [
            r'(\d+\.?\d*)/10',
            r'(\d+\.?\d*)/5',
            r'(\d+\.?\d*)\s*(?:star|stars)',
            r'rating[:\s]*(\d+\.?\d*)',
            r'score[:\s]*(\d+\.?\d*)'
        ]
        
        for pattern in rating_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    rating = float(match)
                    if 0 <= rating <= 10:  # Reasonable rating range
                        ratings.append(rating)
                except:
                    continue
        
        return ratings
    
    def _extract_pros_cons(self, text: str, soup: BeautifulSoup) -> Dict[str, List[str]]:
        """Extract pros and cons from text and HTML"""
        
        pros_cons = {'pros': [], 'cons': []}
        
        # Look for structured pros/cons in HTML
        pros_sections = soup.find_all(text=re.compile(r'pro[s]?[:.]?', re.IGNORECASE))
        cons_sections = soup.find_all(text=re.compile(r'con[s]?[:.]?', re.IGNORECASE))
        
        # Extract nearby list items
        for section in pros_sections:
            parent = section.parent if section.parent else section
            lists = parent.find_all_next(['ul', 'ol'], limit=2)
            for lst in lists:
                items = lst.find_all('li')
                pros_cons['pros'].extend([item.get_text(strip=True) for item in items[:5]])
        
        for section in cons_sections:
            parent = section.parent if section.parent else section
            lists = parent.find_all_next(['ul', 'ol'], limit=2)
            for lst in lists:
                items = lst.find_all('li')
                pros_cons['cons'].extend([item.get_text(strip=True) for item in items[:5]])
        
        return pros_cons
    
    def _extract_images(self, soup: BeautifulSoup) -> List[str]:
        """Extract phone images from HTML"""
        
        images = []
        
        # Look for phone images
        img_tags = soup.find_all('img')
        
        for img in img_tags:
            src = img.get('src') or img.get('data-src')
            alt = img.get('alt', '').lower()
            
            if src and any(keyword in alt for keyword in ['phone', 'mobile', 'device', 'review']):
                if src.startswith('//'):
                    src = 'https:' + src
                elif src.startswith('/'):
                    continue  # Skip relative URLs without domain
                
                if src.startswith('http'):
                    images.append(src)
        
        return list(set(images))[:10]  # Deduplicate and limit
    
    async def _process_universal_data(self, phone_query: str, enhanced_results: Dict[str, List[GoogleSearchResult]]) -> UniversalPhoneData:
        """Process and combine all search results into comprehensive phone data"""
        
        # Combine all results
        all_results = []
        for results_list in enhanced_results.values():
            all_results.extend(results_list)
        
        # Extract and combine specifications
        specifications = {}
        for result in all_results:
            if 'specifications' in result.extracted_data:
                specifications.update(result.extracted_data['specifications'])
        
        # Analyze reviews
        reviews_summary = self._analyze_all_reviews(all_results)
        
        # Extract news mentions
        news_mentions = [
            {'title': r.title, 'url': r.url, 'snippet': r.snippet}
            for r in all_results if r.source_type in ['news', 'blog']
        ]
        
        # Extract forum discussions
        forum_discussions = [
            {'title': r.title, 'url': r.url, 'snippet': r.snippet}
            for r in all_results if r.source_type in ['forum', 'social']
        ]
        
        # Extract price mentions
        price_mentions = []
        for result in all_results:
            if 'pricing' in result.extracted_data:
                pricing_data = result.extracted_data['pricing']
                if pricing_data:
                    price_mentions.append({
                        'source': result.title,
                        'url': result.url,
                        **pricing_data
                    })
        
        # Collect images and videos
        images = []
        for result in all_results:
            if 'images' in result.extracted_data:
                images.extend(result.extracted_data['images'])
        
        # Calculate overall sentiment
        overall_sentiment = self._calculate_overall_sentiment(all_results)
        
        # Calculate confidence based on results quality
        confidence = self._calculate_confidence(all_results, specifications)
        
        return UniversalPhoneData(
            phone_model=phone_query,
            search_results=all_results,
            specifications=specifications,
            reviews_summary=reviews_summary,
            news_mentions=news_mentions,
            forum_discussions=forum_discussions,
            price_mentions=price_mentions,
            images=list(set(images))[:20],  # Deduplicate and limit
            videos=[],  # Could be expanded to extract video links
            overall_sentiment=overall_sentiment['overall'],
            confidence=confidence,
            sources_count=len(set(r.url for r in all_results)),
            search_metadata={
                'search_timestamp': datetime.now().isoformat(),
                'total_results': len(all_results),
                'query_types_used': list(enhanced_results.keys()),
                'extraction_enabled': self.config['enable_content_extraction']
            }
        )
    
    def _analyze_all_reviews(self, results: List[GoogleSearchResult]) -> Dict[str, Any]:
        """Analyze all review results to create summary"""
        
        review_results = [r for r in results if r.source_type == 'review' or 'review' in r.title.lower()]
        
        if not review_results:
            return {'count': 0, 'average_sentiment': 'neutral', 'confidence': 0.0}
        
        # Combine all sentiment analyses
        sentiments = []
        all_ratings = []
        
        for result in review_results:
            if 'review_sentiment' in result.extracted_data:
                sentiment_data = result.extracted_data['review_sentiment']
                sentiments.append(sentiment_data)
            
            if 'ratings' in result.extracted_data:
                all_ratings.extend(result.extracted_data['ratings'])
        
        # Calculate average sentiment
        if sentiments:
            positive_count = sum(1 for s in sentiments if s.get('overall') == 'positive')
            negative_count = sum(1 for s in sentiments if s.get('overall') == 'negative')
            neutral_count = len(sentiments) - positive_count - negative_count
            
            if positive_count > negative_count:
                avg_sentiment = 'positive'
            elif negative_count > positive_count:
                avg_sentiment = 'negative'
            else:
                avg_sentiment = 'neutral'
        else:
            avg_sentiment = 'neutral'
        
        # Calculate average rating
        avg_rating = sum(all_ratings) / len(all_ratings) if all_ratings else None
        
        return {
            'count': len(review_results),
            'average_sentiment': avg_sentiment,
            'average_rating': avg_rating,
            'rating_count': len(all_ratings),
            'confidence': min(len(review_results) / 5.0, 1.0)  # Higher confidence with more reviews
        }
    
    def _calculate_overall_sentiment(self, results: List[GoogleSearchResult]) -> Dict[str, Any]:
        """Calculate overall sentiment from all results"""
        
        all_sentiments = []
        
        for result in results:
            if 'review_sentiment' in result.extracted_data:
                all_sentiments.append(result.extracted_data['review_sentiment'])
        
        if not all_sentiments:
            return {'overall': 'neutral', 'confidence': 0.0}
        
        # Weight sentiments by their confidence
        weighted_positive = sum(s.get('breakdown', {}).get('positive', 0) * s.get('confidence', 1) for s in all_sentiments)
        weighted_negative = sum(s.get('breakdown', {}).get('negative', 0) * s.get('confidence', 1) for s in all_sentiments)
        weighted_neutral = sum(s.get('breakdown', {}).get('neutral', 0) * s.get('confidence', 1) for s in all_sentiments)
        
        total_weight = weighted_positive + weighted_negative + weighted_neutral
        
        if total_weight == 0:
            return {'overall': 'neutral', 'confidence': 0.0}
        
        positive_ratio = weighted_positive / total_weight
        negative_ratio = weighted_negative / total_weight
        
        if positive_ratio > negative_ratio:
            overall = 'positive'
            confidence = positive_ratio
        elif negative_ratio > positive_ratio:
            overall = 'negative'
            confidence = negative_ratio
        else:
            overall = 'neutral'
            confidence = weighted_neutral / total_weight
        
        return {'overall': overall, 'confidence': confidence}
    
    def _calculate_confidence(self, results: List[GoogleSearchResult], specifications: Dict[str, Any]) -> float:
        """Calculate overall confidence in the search results"""
        
        confidence_factors = []
        
        # Factor 1: Number of results
        result_count_score = min(len(results) / 20.0, 1.0)
        confidence_factors.append(result_count_score)
        
        # Factor 2: Source diversity
        source_types = set(r.source_type for r in results)
        source_diversity_score = min(len(source_types) / 5.0, 1.0)
        confidence_factors.append(source_diversity_score)
        
        # Factor 3: Specification completeness
        spec_completeness_score = min(len(specifications) / 8.0, 1.0)
        confidence_factors.append(spec_completeness_score)
        
        # Factor 4: Average relevance score
        if results:
            avg_relevance = sum(r.relevance_score for r in results) / len(results)
            confidence_factors.append(avg_relevance)
        
        # Calculate weighted average
        return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.0
    
    def _calculate_relevance_score(self, item: Dict, query_type: str) -> float:
        """Calculate relevance score for a search result"""
        
        score = 0.5  # Base score
        
        title = item.get('title', '').lower()
        snippet = item.get('snippet', '').lower()
        url = item.get('link', '').lower()
        
        # Boost for phone-related keywords
        phone_keywords = ['phone', 'mobile', 'smartphone', 'device', 'review', 'specs', 'specification']
        for keyword in phone_keywords:
            if keyword in title:
                score += 0.1
            if keyword in snippet:
                score += 0.05
        
        # Boost for trusted domains
        trusted_domains = ['gsmarena.com', 'phonearena.com', 'cnet.com', 'techradar.com', 
                          'androidauthority.com', 'androidcentral.com', 'theverge.com']
        for domain in trusted_domains:
            if domain in url:
                score += 0.2
                break
        
        # Query type specific boosts
        if query_type == 'specifications' and any(word in title + snippet for word in ['specs', 'specification', 'technical']):
            score += 0.15
        elif query_type == 'reviews' and any(word in title + snippet for word in ['review', 'hands-on', 'test']):
            score += 0.15
        elif query_type == 'pricing' and any(word in title + snippet for word in ['price', 'buy', 'cost', 'deal']):
            score += 0.15
        
        return min(score, 1.0)
    
    def _determine_source_type(self, url: str) -> str:
        """Determine the type of source based on URL"""
        
        url_lower = url.lower()
        
        # Review sites
        if any(domain in url_lower for domain in ['gsmarena.com', 'phonearena.com', 'cnet.com', 'techradar.com', 'androidauthority.com']):
            return 'review'
        
        # News sites
        elif any(domain in url_lower for domain in ['techcrunch.com', 'theverge.com', 'engadget.com', 'arstechnica.com']):
            return 'news'
        
        # Forums and social
        elif any(domain in url_lower for domain in ['reddit.com', 'xda-developers.com', 'forum.', 'community.']):
            return 'forum'
        
        # Shopping sites
        elif any(domain in url_lower for domain in ['amazon.com', 'ebay.com', 'bestbuy.com', 'shop']):
            return 'shopping'
        
        # Blogs
        elif any(word in url_lower for word in ['blog', 'wordpress', 'medium.com']):
            return 'blog'
        
        # Manufacturer sites
        elif any(domain in url_lower for domain in ['apple.com', 'samsung.com', 'google.com', 'oneplus.com']):
            return 'manufacturer'
        
        else:
            return 'general'
    
    def _create_empty_result(self, phone_query: str, reason: str) -> UniversalPhoneData:
        """Create empty result when search is not possible"""
        
        return UniversalPhoneData(
            phone_model=phone_query,
            search_results=[],
            specifications={},
            reviews_summary={'count': 0, 'average_sentiment': 'neutral', 'confidence': 0.0},
            news_mentions=[],
            forum_discussions=[],
            price_mentions=[],
            images=[],
            videos=[],
            overall_sentiment='neutral',
            confidence=0.0,
            sources_count=0,
            search_metadata={
                'search_timestamp': datetime.now().isoformat(),
                'error': reason,
                'universal_search_available': False
            }
        )
    
    # Utility methods
    async def _init_session(self):
        """Initialize aiohttp session"""
        if not self.session:
            connector = aiohttp.TCPConnector(limit=10)
            timeout = aiohttp.ClientTimeout(total=self.config['request_timeout'])
            self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
    
    async def _cleanup_session(self):
        """Cleanup aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None

# Factory function
def create_google_search_integration(config=None):
    """Create configured Google Custom Search integration"""
    return GoogleCustomSearch(config=config)