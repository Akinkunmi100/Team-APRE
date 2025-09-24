"""
Dynamic Source Discovery System for AI Phone Review Engine
Automatically discovers and integrates new phone review sources and websites
"""

import asyncio
import aiohttp
import logging
import json
import re
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse, quote_plus
from bs4 import BeautifulSoup
import hashlib
import pickle
from pathlib import Path
from collections import defaultdict, Counter
import random

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DiscoveredSource:
    """Structure for discovered phone information sources"""
    domain: str
    base_url: str
    source_type: str  # review, news, forum, blog, shopping, etc.
    phone_coverage_score: float
    content_quality_score: float
    update_frequency_score: float
    technical_relevance_score: float
    overall_score: float
    language: str
    region: str
    sample_urls: List[str]
    extraction_patterns: Dict[str, str]
    discovered_at: str
    last_validated: str
    validation_status: str

@dataclass
class SourceValidationResult:
    """Result of source validation"""
    is_valid: bool
    phone_content_found: bool
    content_quality: float
    extraction_success: bool
    response_time: float
    accessibility: bool
    robots_allowed: bool
    issues: List[str]
    extracted_sample: Dict[str, Any]

class DynamicSourceDiscovery:
    """System for automatically discovering new phone information sources"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize dynamic source discovery system"""
        
        self.config = config or {
            'max_sources_per_discovery': 20,
            'discovery_depth': 3,
            'min_quality_threshold': 0.6,
            'max_concurrent_requests': 5,
            'request_timeout': 30,
            'enable_deep_crawling': True,
            'validate_discovered_sources': True,
            'save_discoveries': True,
            'discoveries_file': 'data/discovered_sources.json'
        }
        
        # Known seed sources to start discovery from
        self.seed_sources = [
            'https://www.google.com/search?q=phone+review+sites',
            'https://www.reddit.com/r/Android/wiki/phones',
            'https://en.wikipedia.org/wiki/List_of_technology_websites',
            'https://www.similarweb.com/top-websites/computers-electronics-and-technology/',
        ]
        
        # Patterns to identify phone-related content
        self.phone_indicators = {
            'domain_keywords': [
                'phone', 'mobile', 'smartphone', 'android', 'iphone', 'galaxy', 'pixel',
                'tech', 'review', 'gsm', 'cellular', 'wireless', 'telecom'
            ],
            'content_keywords': [
                'phone review', 'smartphone', 'mobile phone', 'cell phone', 'android',
                'ios', 'iphone', 'galaxy', 'pixel', 'specifications', 'specs',
                'camera test', 'battery life', 'performance', 'benchmark'
            ],
            'title_patterns': [
                r'.*phone.*review.*',
                r'.*smartphone.*test.*',
                r'.*mobile.*news.*',
                r'.*\b(?:iphone|galaxy|pixel|oneplus|xiaomi)\b.*',
                r'.*tech.*review.*'
            ]
        }
        
        # Regional patterns for international coverage
        self.regional_patterns = {
            'international': ['.com', '.org', '.net'],
            'europe': ['.uk', '.de', '.fr', '.it', '.es', '.nl'],
            'asia': ['.jp', '.kr', '.cn', '.in', '.sg', '.tw'],
            'others': ['.au', '.ca', '.br', '.ru']
        }
        
        # Content quality indicators
        self.quality_indicators = {
            'high_quality': [
                'detailed review', 'comprehensive', 'thorough analysis', 'in-depth',
                'technical specifications', 'benchmark', 'comparison', 'expert'
            ],
            'medium_quality': [
                'review', 'test', 'hands-on', 'first impressions', 'overview'
            ],
            'low_quality': [
                'rumor', 'leak', 'unconfirmed', 'speculation', 'clickbait'
            ]
        }
        
        # Discovered sources cache
        self.discovered_sources = {}
        self.validated_sources = {}
        self.session = None
        
        # Load existing discoveries
        self._load_existing_discoveries()
    
    async def discover_new_sources(self, search_terms: List[str] = None, 
                                 discovery_method: str = 'comprehensive') -> Dict[str, List[DiscoveredSource]]:
        """
        Discover new phone information sources
        
        Args:
            search_terms: Specific terms to search for sources
            discovery_method: 'basic', 'standard', or 'comprehensive'
            
        Returns:
            Dictionary of discovered sources by category
        """
        
        logger.info(f"Starting dynamic source discovery with method: {discovery_method}")
        
        # Initialize session
        await self._init_session()
        
        try:
            # Get search terms
            if not search_terms:
                search_terms = self._get_default_search_terms(discovery_method)
            
            # Discover sources using multiple methods
            discoveries = {}
            
            # Method 1: Search engine discovery
            search_discoveries = await self._discover_via_search_engines(search_terms)
            discoveries['search_engine'] = search_discoveries
            
            # Method 2: Link following from known sources
            if discovery_method in ['standard', 'comprehensive']:
                link_discoveries = await self._discover_via_link_following()
                discoveries['link_following'] = link_discoveries
            
            # Method 3: Pattern-based domain discovery
            if discovery_method == 'comprehensive':
                pattern_discoveries = await self._discover_via_domain_patterns()
                discoveries['pattern_based'] = pattern_discoveries
            
            # Method 4: Social media and forum discovery
            if discovery_method == 'comprehensive':
                social_discoveries = await self._discover_via_social_sources()
                discoveries['social_media'] = social_discoveries
            
            # Validate discovered sources
            if self.config['validate_discovered_sources']:
                validated_discoveries = await self._validate_all_discoveries(discoveries)
                discoveries = validated_discoveries
            
            # Save discoveries
            if self.config['save_discoveries']:
                await self._save_discoveries(discoveries)
            
            return discoveries
            
        finally:
            await self._cleanup_session()
    
    def _get_default_search_terms(self, discovery_method: str) -> List[str]:
        """Get default search terms based on discovery method"""
        
        basic_terms = [
            'phone review sites',
            'smartphone review websites',
            'mobile phone news'
        ]
        
        standard_terms = basic_terms + [
            'android review sites',
            'iphone review websites',
            'tech review platforms',
            'mobile phone specifications',
            'phone comparison sites'
        ]
        
        comprehensive_terms = standard_terms + [
            'phone review blogs',
            'mobile tech forums',
            'smartphone news sites',
            'phone specs databases',
            'mobile device reviews',
            'phone price comparison',
            'tech review channels',
            'mobile phone communities'
        ]
        
        if discovery_method == 'basic':
            return basic_terms
        elif discovery_method == 'standard':
            return standard_terms
        else:
            return comprehensive_terms
    
    async def _discover_via_search_engines(self, search_terms: List[str]) -> List[DiscoveredSource]:
        """Discover sources using search engines"""
        
        discovered = []
        
        # Use DuckDuckGo as it doesn't require API keys
        for term in search_terms[:5]:  # Limit to prevent overload
            try:
                # Construct search URL
                search_url = f"https://duckduckgo.com/html/?q={quote_plus(term)}"
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                
                async with self.session.get(search_url, headers=headers, timeout=self.config['request_timeout']) as response:
                    if response.status == 200:
                        content = await response.text()
                        sources = self._extract_sources_from_search_results(content, term)
                        discovered.extend(sources)
                
                # Rate limiting
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Search engine discovery failed for term '{term}': {e}")
        
        return discovered
    
    def _extract_sources_from_search_results(self, html_content: str, search_term: str) -> List[DiscoveredSource]:
        """Extract potential sources from search engine results"""
        
        sources = []
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Find result links (DuckDuckGo specific)
            result_links = soup.find_all('a', href=True)
            
            for link in result_links[:20]:  # Limit processing
                href = link.get('href')
                if not href or not href.startswith('http'):
                    continue
                
                # Skip known non-relevant domains
                if any(skip in href for skip in ['facebook.com', 'twitter.com', 'youtube.com', 'linkedin.com']):
                    continue
                
                # Extract domain
                parsed_url = urlparse(href)
                domain = parsed_url.netloc
                
                if domain and self._is_potential_phone_source(href, link.get_text()):
                    source = self._create_discovered_source(
                        domain=domain,
                        base_url=f"{parsed_url.scheme}://{domain}",
                        sample_url=href,
                        context=f"Search: {search_term}",
                        link_text=link.get_text()
                    )
                    sources.append(source)
            
        except Exception as e:
            logger.error(f"Error extracting sources from search results: {e}")
        
        return sources
    
    async def _discover_via_link_following(self) -> List[DiscoveredSource]:
        """Discover sources by following links from known sources"""
        
        discovered = []
        
        # Start from known high-quality sources
        seed_sources = [
            'https://www.gsmarena.com',
            'https://www.phonearena.com',
            'https://www.androidauthority.com',
            'https://www.cnet.com/tech/mobile',
        ]
        
        for seed_url in seed_sources:
            try:
                # Get the seed page
                headers = {'User-Agent': 'Mozilla/5.0 (compatible; SourceDiscovery/1.0)'}
                
                async with self.session.get(seed_url, headers=headers, timeout=self.config['request_timeout']) as response:
                    if response.status == 200:
                        content = await response.text()
                        
                        # Extract outbound links
                        soup = BeautifulSoup(content, 'html.parser')
                        links = soup.find_all('a', href=True)
                        
                        for link in links[:50]:  # Limit to prevent overload
                            href = link.get('href')
                            if href:
                                # Convert relative URLs to absolute
                                full_url = urljoin(seed_url, href)
                                
                                # Check if it's a potential phone source
                                if self._is_potential_phone_source(full_url, link.get_text()):
                                    parsed_url = urlparse(full_url)
                                    domain = parsed_url.netloc
                                    
                                    if domain and domain not in [urlparse(s).netloc for s in seed_sources]:
                                        source = self._create_discovered_source(
                                            domain=domain,
                                            base_url=f"{parsed_url.scheme}://{domain}",
                                            sample_url=full_url,
                                            context=f"Link from: {seed_url}",
                                            link_text=link.get_text()
                                        )
                                        discovered.append(source)
                
                # Rate limiting
                await asyncio.sleep(3)
                
            except Exception as e:
                logger.error(f"Link following failed for {seed_url}: {e}")
        
        return discovered
    
    async def _discover_via_domain_patterns(self) -> List[DiscoveredSource]:
        """Discover sources using common domain patterns"""
        
        discovered = []
        
        # Common patterns for phone/tech websites
        base_patterns = [
            'phone', 'mobile', 'smartphone', 'android', 'tech', 'review', 
            'gadget', 'device', 'cellular', 'wireless'
        ]
        
        suffixes = ['review', 'news', 'blog', 'site', 'world', 'central', 'authority']
        tlds = ['.com', '.net', '.org']
        
        # Generate potential domains
        potential_domains = []
        for base in base_patterns[:5]:  # Limit to prevent too many requests
            for suffix in suffixes[:3]:
                for tld in tlds[:2]:
                    domain = f"{base}{suffix}{tld}"
                    potential_domains.append(f"https://{domain}")
        
        # Test domains for validity
        semaphore = asyncio.Semaphore(self.config['max_concurrent_requests'])
        
        async def test_domain(url):
            async with semaphore:
                try:
                    headers = {'User-Agent': 'Mozilla/5.0 (compatible; SourceDiscovery/1.0)'}
                    async with self.session.get(url, headers=headers, timeout=10) as response:
                        if response.status == 200:
                            content = await response.text()
                            if self._has_phone_content(content):
                                parsed_url = urlparse(url)
                                return self._create_discovered_source(
                                    domain=parsed_url.netloc,
                                    base_url=url,
                                    sample_url=url,
                                    context="Domain pattern matching",
                                    link_text="Pattern-based discovery"
                                )
                except:
                    pass
                return None
        
        # Test domains concurrently
        tasks = [test_domain(domain) for domain in potential_domains[:20]]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect valid sources
        for result in results:
            if result and not isinstance(result, Exception):
                discovered.append(result)
        
        return discovered
    
    async def _discover_via_social_sources(self) -> List[DiscoveredSource]:
        """Discover sources from social media and forums"""
        
        discovered = []
        
        # Reddit sources (publicly accessible)
        reddit_sources = [
            'https://www.reddit.com/r/Android/wiki/phones',
            'https://www.reddit.com/r/iphone/wiki/index',
            'https://www.reddit.com/r/PickAnAndroidForMe/wiki/index'
        ]
        
        for reddit_url in reddit_sources:
            try:
                headers = {'User-Agent': 'Mozilla/5.0 (compatible; SourceDiscovery/1.0)'}
                
                async with self.session.get(reddit_url, headers=headers, timeout=self.config['request_timeout']) as response:
                    if response.status == 200:
                        content = await response.text()
                        
                        # Extract links from Reddit wiki pages
                        soup = BeautifulSoup(content, 'html.parser')
                        links = soup.find_all('a', href=True)
                        
                        for link in links:
                            href = link.get('href')
                            if href and href.startswith('http'):
                                if self._is_potential_phone_source(href, link.get_text()):
                                    parsed_url = urlparse(href)
                                    domain = parsed_url.netloc
                                    
                                    if domain:
                                        source = self._create_discovered_source(
                                            domain=domain,
                                            base_url=f"{parsed_url.scheme}://{domain}",
                                            sample_url=href,
                                            context=f"Reddit: {reddit_url}",
                                            link_text=link.get_text()
                                        )
                                        discovered.append(source)
                
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Social discovery failed for {reddit_url}: {e}")
        
        return discovered
    
    def _is_potential_phone_source(self, url: str, link_text: str) -> bool:
        """Check if a URL/link is potentially a phone information source"""
        
        url_lower = url.lower()
        text_lower = link_text.lower() if link_text else ""
        
        # Domain keyword check
        domain_match = any(keyword in url_lower for keyword in self.phone_indicators['domain_keywords'])
        
        # Content keyword check
        content_match = any(keyword in text_lower for keyword in self.phone_indicators['content_keywords'])
        
        # Title pattern check
        title_match = any(re.search(pattern, text_lower) for pattern in self.phone_indicators['title_patterns'])
        
        # Exclude obviously irrelevant sources
        exclude_patterns = ['facebook.com', 'twitter.com', 'linkedin.com', 'pinterest.com', 'instagram.com']
        is_excluded = any(pattern in url_lower for pattern in exclude_patterns)
        
        return (domain_match or content_match or title_match) and not is_excluded
    
    def _has_phone_content(self, html_content: str) -> bool:
        """Check if HTML content contains phone-related information"""
        
        content_lower = html_content.lower()
        
        # Count phone-related keywords
        keyword_count = sum(content_lower.count(keyword) for keyword in self.phone_indicators['content_keywords'])
        
        # Look for phone model mentions
        phone_models = ['iphone', 'galaxy', 'pixel', 'oneplus', 'xiaomi', 'huawei', 'samsung']
        model_count = sum(content_lower.count(model) for model in phone_models)
        
        # Check for review/specification structures
        has_structure = any(pattern in content_lower for pattern in [
            'specifications', 'pros and cons', 'rating', 'review', 'camera test', 'battery life'
        ])
        
        return keyword_count >= 5 or model_count >= 3 or has_structure
    
    def _create_discovered_source(self, domain: str, base_url: str, sample_url: str, 
                                context: str, link_text: str) -> DiscoveredSource:
        """Create a discovered source object"""
        
        # Determine source type
        source_type = self._determine_source_type(domain, sample_url, link_text)
        
        # Calculate initial scores
        coverage_score = self._calculate_coverage_score(domain, sample_url, link_text)
        quality_score = self._calculate_quality_score(domain, link_text)
        relevance_score = self._calculate_relevance_score(domain, sample_url, link_text)
        
        # Determine language and region
        language, region = self._determine_language_region(domain)
        
        overall_score = (coverage_score + quality_score + relevance_score) / 3
        
        return DiscoveredSource(
            domain=domain,
            base_url=base_url,
            source_type=source_type,
            phone_coverage_score=coverage_score,
            content_quality_score=quality_score,
            update_frequency_score=0.5,  # Will be updated during validation
            technical_relevance_score=relevance_score,
            overall_score=overall_score,
            language=language,
            region=region,
            sample_urls=[sample_url],
            extraction_patterns={},  # Will be populated during validation
            discovered_at=datetime.now().isoformat(),
            last_validated="",
            validation_status="pending"
        )
    
    def _determine_source_type(self, domain: str, url: str, link_text: str) -> str:
        """Determine the type of source"""
        
        domain_lower = domain.lower()
        url_lower = url.lower()
        text_lower = link_text.lower()
        
        if any(word in domain_lower for word in ['review', 'test']):
            return 'review'
        elif any(word in domain_lower for word in ['news', 'blog']):
            return 'news'
        elif any(word in domain_lower for word in ['forum', 'community']):
            return 'forum'
        elif any(word in domain_lower for word in ['shop', 'buy', 'price']):
            return 'shopping'
        elif any(word in domain_lower for word in ['spec', 'database']):
            return 'specification'
        else:
            return 'general'
    
    def _calculate_coverage_score(self, domain: str, url: str, link_text: str) -> float:
        """Calculate how comprehensive the phone coverage might be"""
        
        score = 0.5  # Base score
        
        # Domain indicators
        coverage_indicators = ['gsm', 'phone', 'mobile', 'smartphone', 'android', 'tech']
        for indicator in coverage_indicators:
            if indicator in domain.lower():
                score += 0.1
        
        # URL path indicators
        if any(word in url.lower() for word in ['phone', 'mobile', 'review', 'specs']):
            score += 0.1
        
        # Link text indicators
        if any(word in link_text.lower() for word in ['comprehensive', 'complete', 'all', 'database']):
            score += 0.15
        
        return min(score, 1.0)
    
    def _calculate_quality_score(self, domain: str, link_text: str) -> float:
        """Calculate expected content quality"""
        
        score = 0.5  # Base score
        
        # High-quality indicators
        for indicator in self.quality_indicators['high_quality']:
            if indicator in link_text.lower():
                score += 0.1
        
        # Trusted domain patterns
        if any(pattern in domain.lower() for pattern in ['authority', 'central', 'arena', 'expert']):
            score += 0.15
        
        # Reduce score for low-quality indicators
        for indicator in self.quality_indicators['low_quality']:
            if indicator in link_text.lower():
                score -= 0.1
        
        return max(min(score, 1.0), 0.0)
    
    def _calculate_relevance_score(self, domain: str, url: str, link_text: str) -> float:
        """Calculate technical relevance for phone information"""
        
        score = 0.5  # Base score
        
        # Technical indicators
        technical_terms = ['specs', 'benchmark', 'test', 'analysis', 'comparison', 'technical']
        for term in technical_terms:
            if term in (domain + url + link_text).lower():
                score += 0.08
        
        # Phone-specific terms
        phone_terms = ['iphone', 'android', 'galaxy', 'pixel', 'smartphone']
        for term in phone_terms:
            if term in (domain + url + link_text).lower():
                score += 0.05
        
        return min(score, 1.0)
    
    def _determine_language_region(self, domain: str) -> Tuple[str, str]:
        """Determine language and region from domain"""
        
        domain_lower = domain.lower()
        
        # Check TLD for region
        region = 'international'
        for region_name, tlds in self.regional_patterns.items():
            if any(tld in domain_lower for tld in tlds):
                region = region_name
                break
        
        # Language is mostly English for tech content
        language = 'en'
        
        # Could be expanded to detect other languages
        if any(pattern in domain_lower for pattern in ['.de', 'deutsch', 'german']):
            language = 'de'
        elif any(pattern in domain_lower for pattern in ['.fr', 'francais', 'french']):
            language = 'fr'
        elif any(pattern in domain_lower for pattern in ['.es', 'espanol', 'spanish']):
            language = 'es'
        
        return language, region
    
    async def _validate_all_discoveries(self, discoveries: Dict[str, List[DiscoveredSource]]) -> Dict[str, List[DiscoveredSource]]:
        """Validate all discovered sources"""
        
        validated_discoveries = {}
        
        for category, sources in discoveries.items():
            validated_sources = []
            
            # Validate sources concurrently
            semaphore = asyncio.Semaphore(self.config['max_concurrent_requests'])
            
            async def validate_source(source):
                async with semaphore:
                    validation = await self._validate_single_source(source)
                    if validation.is_valid and validation.phone_content_found:
                        # Update source with validation data
                        source.validation_status = "validated"
                        source.last_validated = datetime.now().isoformat()
                        return source
                    return None
            
            # Validate top sources only (to save time)
            top_sources = sorted(sources, key=lambda s: s.overall_score, reverse=True)[:10]
            
            tasks = [validate_source(source) for source in top_sources]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Collect validated sources
            for result in results:
                if result and not isinstance(result, Exception):
                    validated_sources.append(result)
            
            validated_discoveries[category] = validated_sources
        
        return validated_discoveries
    
    async def _validate_single_source(self, source: DiscoveredSource) -> SourceValidationResult:
        """Validate a single discovered source"""
        
        try:
            start_time = datetime.now()
            
            # Test accessibility
            headers = {'User-Agent': 'Mozilla/5.0 (compatible; SourceDiscovery/1.0)'}
            
            async with self.session.get(source.base_url, headers=headers, timeout=15) as response:
                response_time = (datetime.now() - start_time).total_seconds()
                
                if response.status != 200:
                    return SourceValidationResult(
                        is_valid=False,
                        phone_content_found=False,
                        content_quality=0.0,
                        extraction_success=False,
                        response_time=response_time,
                        accessibility=False,
                        robots_allowed=True,  # Assume true if no robots.txt
                        issues=[f"HTTP {response.status}"],
                        extracted_sample={}
                    )
                
                content = await response.text()
                
                # Check for phone content
                has_phone_content = self._has_phone_content(content)
                
                # Assess content quality
                quality_score = self._assess_content_quality(content)
                
                # Try to extract sample data
                extracted_sample = self._extract_sample_data(content)
                extraction_success = len(extracted_sample) > 0
                
                return SourceValidationResult(
                    is_valid=True,
                    phone_content_found=has_phone_content,
                    content_quality=quality_score,
                    extraction_success=extraction_success,
                    response_time=response_time,
                    accessibility=True,
                    robots_allowed=True,
                    issues=[],
                    extracted_sample=extracted_sample
                )
                
        except Exception as e:
            return SourceValidationResult(
                is_valid=False,
                phone_content_found=False,
                content_quality=0.0,
                extraction_success=False,
                response_time=0.0,
                accessibility=False,
                robots_allowed=True,
                issues=[str(e)],
                extracted_sample={}
            )
    
    def _assess_content_quality(self, content: str) -> float:
        """Assess the quality of content on a page"""
        
        quality_score = 0.5
        content_lower = content.lower()
        
        # Check for high-quality indicators
        for indicator in self.quality_indicators['high_quality']:
            if indicator in content_lower:
                quality_score += 0.1
        
        # Check for structured content
        if any(tag in content_lower for tag in ['<table', '<ul', '<ol', 'specifications', 'pros', 'cons']):
            quality_score += 0.15
        
        # Check content length (longer usually means more comprehensive)
        if len(content) > 10000:
            quality_score += 0.1
        elif len(content) > 5000:
            quality_score += 0.05
        
        # Reduce for low-quality indicators
        for indicator in self.quality_indicators['low_quality']:
            if indicator in content_lower:
                quality_score -= 0.1
        
        return max(min(quality_score, 1.0), 0.0)
    
    def _extract_sample_data(self, content: str) -> Dict[str, Any]:
        """Extract sample phone data from content"""
        
        sample = {}
        
        try:
            soup = BeautifulSoup(content, 'html.parser')
            
            # Try to find phone names
            phone_patterns = [
                r'(iPhone \d+[^\s]*)',
                r'(Galaxy S\d+[^\s]*)',
                r'(Pixel \d+[^\s]*)',
                r'(OnePlus \d+[^\s]*)'
            ]
            
            text_content = soup.get_text()
            phones_found = []
            
            for pattern in phone_patterns:
                matches = re.findall(pattern, text_content, re.IGNORECASE)
                phones_found.extend(matches[:3])  # Limit matches
            
            if phones_found:
                sample['phones_mentioned'] = list(set(phones_found))
            
            # Try to find ratings
            rating_patterns = [r'(\d+\.?\d*)/10', r'(\d+\.?\d*)/5', r'(\d+\.?\d*) stars?']
            ratings_found = []
            
            for pattern in rating_patterns:
                matches = re.findall(pattern, text_content)
                ratings_found.extend(matches[:3])
            
            if ratings_found:
                sample['ratings_found'] = ratings_found
            
            # Try to find specifications
            spec_patterns = [
                r'(\d+\.?\d*)\s*inch',
                r'(\d+\.?\d*)\s*mah',
                r'(\d+\.?\d*)\s*mp',
                r'(\d+\.?\d*)\s*gb'
            ]
            
            specs_found = []
            for pattern in spec_patterns:
                matches = re.findall(pattern, text_content.lower())
                specs_found.extend(matches[:2])
            
            if specs_found:
                sample['specifications_found'] = specs_found
            
        except Exception as e:
            logger.warning(f"Sample data extraction failed: {e}")
        
        return sample
    
    async def _save_discoveries(self, discoveries: Dict[str, List[DiscoveredSource]]):
        """Save discovered sources to file"""
        
        try:
            # Prepare data for saving
            save_data = {
                'discovery_timestamp': datetime.now().isoformat(),
                'total_sources': sum(len(sources) for sources in discoveries.values()),
                'categories': {}
            }
            
            for category, sources in discoveries.items():
                save_data['categories'][category] = [asdict(source) for source in sources]
            
            # Ensure directory exists
            Path(self.config['discoveries_file']).parent.mkdir(parents=True, exist_ok=True)
            
            # Save to file
            with open(self.config['discoveries_file'], 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {save_data['total_sources']} discovered sources to {self.config['discoveries_file']}")
            
        except Exception as e:
            logger.error(f"Failed to save discoveries: {e}")
    
    def _load_existing_discoveries(self):
        """Load existing discovered sources from file"""
        
        try:
            if Path(self.config['discoveries_file']).exists():
                with open(self.config['discoveries_file'], 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Load discovered sources into memory
                for category, sources_data in data.get('categories', {}).items():
                    sources = [DiscoveredSource(**source_data) for source_data in sources_data]
                    self.discovered_sources[category] = sources
                
                logger.info(f"Loaded {len(self.discovered_sources)} categories of discovered sources")
            
        except Exception as e:
            logger.warning(f"Failed to load existing discoveries: {e}")
    
    def get_validated_sources(self, category: str = None, min_quality: float = 0.6) -> List[DiscoveredSource]:
        """Get validated sources optionally filtered by category and quality"""
        
        sources = []
        
        categories_to_check = [category] if category else self.discovered_sources.keys()
        
        for cat in categories_to_check:
            if cat in self.discovered_sources:
                category_sources = self.discovered_sources[cat]
                validated_sources = [
                    source for source in category_sources 
                    if source.validation_status == "validated" and source.overall_score >= min_quality
                ]
                sources.extend(validated_sources)
        
        return sorted(sources, key=lambda s: s.overall_score, reverse=True)
    
    # Utility methods
    async def _init_session(self):
        """Initialize aiohttp session"""
        if not self.session:
            connector = aiohttp.TCPConnector(limit=20)
            timeout = aiohttp.ClientTimeout(total=self.config['request_timeout'])
            self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
    
    async def _cleanup_session(self):
        """Cleanup aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None

# Factory function
def create_dynamic_source_discovery(config=None):
    """Create configured dynamic source discovery system"""
    return DynamicSourceDiscovery(config=config)