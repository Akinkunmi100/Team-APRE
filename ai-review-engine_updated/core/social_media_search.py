"""
Social Media and Forum Search Integration for AI Phone Review Engine
Searches Twitter, Reddit, tech forums, and social platforms for phone discussions and reviews
"""

import asyncio
import aiohttp
import logging
import json
import re
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse, quote_plus
from bs4 import BeautifulSoup
import hashlib
from pathlib import Path
import praw
import tweepy
import time
from collections import defaultdict, Counter
import statistics

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SocialMediaPost:
    """Structure for social media posts about phones"""
    platform: str
    post_id: str
    author: str
    content: str
    phone_mentions: List[str]
    sentiment_score: float
    engagement_metrics: Dict[str, int]
    published_at: str
    url: str
    hashtags: List[str]
    mentions: List[str]
    media_urls: List[str]
    confidence_score: float

@dataclass
class ForumPost:
    """Structure for forum posts about phones"""
    forum_name: str
    thread_title: str
    post_id: str
    author: str
    content: str
    phone_mentions: List[str]
    post_type: str  # question, review, discussion, tech_support
    replies_count: int
    views_count: int
    upvotes: int
    published_at: str
    thread_url: str
    tags: List[str]
    is_solution: bool
    confidence_score: float

@dataclass
class SocialMediaSearchResult:
    """Aggregated search results from social media"""
    query: str
    platform: str
    total_posts: int
    posts: List[Union[SocialMediaPost, ForumPost]]
    sentiment_analysis: Dict[str, float]
    trending_topics: List[str]
    top_influencers: List[str]
    search_timestamp: str
    query_confidence: float

class SocialMediaSearchEngine:
    """Engine for searching social media and forums for phone information"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize social media search engine"""
        
        self.config = config or {
            'twitter_api_key': '',
            'twitter_api_secret': '',
            'twitter_access_token': '',
            'twitter_access_token_secret': '',
            'reddit_client_id': '',
            'reddit_client_secret': '',
            'reddit_user_agent': 'PhoneReviewBot/1.0',
            'enable_twitter': True,
            'enable_reddit': True,
            'enable_xda': True,
            'enable_android_forums': True,
            'max_posts_per_platform': 50,
            'search_timeframe_days': 7,
            'min_engagement_threshold': 5,
            'sentiment_analysis_enabled': True,
            'rate_limit_delay': 2,
            'cache_results': True,
            'cache_duration_hours': 4
        }
        
        # Initialize API clients
        self.twitter_client = None
        self.reddit_client = None
        self.session = None
        
        # Initialize clients
        self._init_api_clients()
        
        # Phone mention patterns - Updated with global brands
        self.phone_patterns = [
            # Premium brands
            r'(iPhone\s+\d+(?:\s+(?:Pro|Plus|Max|Mini))?)',
            r'(Galaxy\s+S\d+(?:\s+(?:Plus|Ultra|FE))?)',
            r'(Galaxy\s+[A-Z]\d+(?:\s+(?:Plus|Ultra|FE))?)',
            r'(Pixel\s+\d+(?:\s+(?:Pro|XL|a))?)',
            r'(OnePlus\s+\d+(?:\s+(?:Pro|T|RT|R))?)',
            
            # Chinese brands
            r'(Xiaomi\s+\d+(?:\s+(?:Pro|Ultra|Lite|T))?)',
            r'(Redmi\s+\w+\s*\d*(?:\s+(?:Pro|Max|Plus))?)',
            r'(Mi\s+\d+(?:\s+(?:Pro|Ultra|Lite|T))?)',
            r'(POCO\s+[A-Z]\d+(?:\s+(?:Pro|GT))?)',
            r'(Huawei\s+P\d+(?:\s+(?:Pro|Lite))?)',
            r'(Honor\s+\d+(?:\s+(?:Pro|Lite|X))?)',
            r'(Oppo\s+[A-Z]\d+(?:\s+(?:Pro|Plus))?)',
            r'(Vivo\s+[A-Z]\d+(?:\s+(?:Pro|Plus))?)',
            r'(Realme\s+\w+\s*\d*(?:\s+(?:Pro|GT|Neo))?)',
            
            # African/Emerging market brands
            r'(Tecno\s+\w+\s*\d*(?:\s+(?:Pro|Plus|Air))?)',
            r'(Infinix\s+\w+\s*\d*(?:\s+(?:Pro|Plus|Hot|Smart|Note))?)',
            r'(Itel\s+\w+\s*\d*(?:\s+(?:Pro|Plus|Smart))?)',
            
            # Other global brands
            r'(Nothing\s+Phone\s+\d+)',
            r'(Motorola\s+\w+\s*\d*(?:\s+(?:Pro|Plus|Edge))?)',
            r'(Nokia\s+\d+(?:\.\d+)?(?:\s+(?:Pro|Plus))?)',
            r'(Sony\s+Xperia\s+\w+\s*\d*)',
            r'(Fairphone\s+\d+)',
            r'(BlackBerry\s+\w+\s*\d*)',
            
            # Generic patterns for any brand
            r'([A-Z][a-z]+\s+[A-Z]?\d+[a-zA-Z]*(?:\s+(?:Pro|Plus|Max|Ultra|Lite|Mini|Air|Neo|GT))?)',
        ]
        
        # Forum configurations
        self.forum_configs = {
            'xda_developers': {
                'base_url': 'https://www.xda-developers.com',
                'search_endpoint': '/search/',
                'selectors': {
                    'title': 'h3.contentRow-title a',
                    'author': '.username',
                    'content': '.message-body',
                    'replies': '.pairs dt:contains("Replies") + dd',
                    'views': '.pairs dt:contains("Views") + dd'
                }
            },
            'android_forums': {
                'base_url': 'https://androidforums.com',
                'search_endpoint': '/search/',
                'selectors': {
                    'title': '.searchResult h3 a',
                    'author': '.username',
                    'content': '.messageContent',
                    'replies': '.stats .major',
                    'date': '.DateTime'
                }
            },
            'reddit_android': {
                'subreddits': ['Android', 'AndroidQuestions', 'PickAnAndroidForMe', 
                              'galaxys21', 'GooglePixel', 'oneplus', 'Xiaomi', 'iphone']
            }
        }
        
        # Cache for search results
        self.results_cache = {}
        
        # Sentiment keywords
        self.sentiment_keywords = {
            'positive': [
                'love', 'amazing', 'great', 'excellent', 'fantastic', 'awesome', 'perfect',
                'recommend', 'impressed', 'satisfied', 'happy', 'glad', 'pleased'
            ],
            'negative': [
                'hate', 'terrible', 'awful', 'horrible', 'disappointed', 'frustrated',
                'annoying', 'problem', 'issue', 'bug', 'broken', 'useless', 'regret'
            ]
        }
    
    def _init_api_clients(self):
        """Initialize API clients for social platforms"""
        
        try:
            # Initialize Twitter client
            if self.config['enable_twitter'] and self.config.get('twitter_api_key'):
                auth = tweepy.OAuthHandler(
                    self.config['twitter_api_key'],
                    self.config['twitter_api_secret']
                )
                auth.set_access_token(
                    self.config['twitter_access_token'],
                    self.config['twitter_access_token_secret']
                )
                self.twitter_client = tweepy.API(auth, wait_on_rate_limit=True)
                logger.info("Twitter API client initialized")
            
            # Initialize Reddit client
            if self.config['enable_reddit'] and self.config.get('reddit_client_id'):
                self.reddit_client = praw.Reddit(
                    client_id=self.config['reddit_client_id'],
                    client_secret=self.config['reddit_client_secret'],
                    user_agent=self.config['reddit_user_agent'],
                    read_only=True
                )
                logger.info("Reddit API client initialized")
        
        except Exception as e:
            logger.error(f"Failed to initialize API clients: {e}")
    
    async def search_all_platforms(self, phone_query: str, 
                                 platforms: List[str] = None) -> Dict[str, SocialMediaSearchResult]:
        """
        Search all enabled platforms for phone information
        
        Args:
            phone_query: Phone model to search for
            platforms: Specific platforms to search (optional)
            
        Returns:
            Dictionary of search results by platform
        """
        
        logger.info(f"Starting multi-platform search for: {phone_query}")
        
        # Check cache first
        cache_key = hashlib.md5(f"{phone_query}_{datetime.now().strftime('%Y%m%d%H')}".encode()).hexdigest()
        if self.config['cache_results'] and cache_key in self.results_cache:
            cache_age = datetime.now() - self.results_cache[cache_key]['timestamp']
            if cache_age.total_seconds() < self.config['cache_duration_hours'] * 3600:
                return self.results_cache[cache_key]['results']
        
        # Determine platforms to search
        if not platforms:
            platforms = []
            if self.config['enable_twitter']:
                platforms.append('twitter')
            if self.config['enable_reddit']:
                platforms.append('reddit')
            if self.config['enable_xda']:
                platforms.append('xda_developers')
            if self.config['enable_android_forums']:
                platforms.append('android_forums')
        
        # Initialize session
        await self._init_session()
        
        try:
            # Search each platform
            results = {}
            
            for platform in platforms:
                try:
                    if platform == 'twitter':
                        results['twitter'] = await self._search_twitter(phone_query)
                    elif platform == 'reddit':
                        results['reddit'] = await self._search_reddit(phone_query)
                    elif platform == 'xda_developers':
                        results['xda_developers'] = await self._search_xda(phone_query)
                    elif platform == 'android_forums':
                        results['android_forums'] = await self._search_android_forums(phone_query)
                    
                    # Rate limiting between platforms
                    await asyncio.sleep(self.config['rate_limit_delay'])
                    
                except Exception as e:
                    logger.error(f"Search failed for platform {platform}: {e}")
                    # Continue with other platforms
                    continue
            
            # Cache results
            if self.config['cache_results']:
                self.results_cache[cache_key] = {
                    'results': results,
                    'timestamp': datetime.now()
                }
            
            logger.info(f"Completed search across {len(results)} platforms")
            return results
            
        finally:
            await self._cleanup_session()
    
    async def _search_twitter(self, phone_query: str) -> SocialMediaSearchResult:
        """Search Twitter for phone discussions"""
        
        if not self.twitter_client:
            logger.warning("Twitter client not initialized")
            return SocialMediaSearchResult(
                query=phone_query,
                platform='twitter',
                total_posts=0,
                posts=[],
                sentiment_analysis={},
                trending_topics=[],
                top_influencers=[],
                search_timestamp=datetime.now().isoformat(),
                query_confidence=0.0
            )
        
        try:
            # Search tweets
            search_query = f"{phone_query} -RT"  # Exclude retweets
            tweets = tweepy.Cursor(
                self.twitter_client.search_tweets,
                q=search_query,
                result_type='mixed',
                lang='en',
                tweet_mode='extended'
            ).items(self.config['max_posts_per_platform'])
            
            posts = []
            all_hashtags = []
            authors = []
            sentiment_scores = []
            
            for tweet in tweets:
                # Extract phone mentions
                phone_mentions = self._extract_phone_mentions(tweet.full_text)
                
                # Calculate sentiment
                sentiment = self._calculate_sentiment(tweet.full_text)
                sentiment_scores.append(sentiment)
                
                # Extract hashtags
                hashtags = [tag['text'] for tag in tweet.entities.get('hashtags', [])]
                all_hashtags.extend(hashtags)
                
                # Extract mentions
                mentions = [mention['screen_name'] for mention in tweet.entities.get('user_mentions', [])]
                
                # Extract media URLs
                media_urls = []
                if 'media' in tweet.entities:
                    media_urls = [media['media_url_https'] for media in tweet.entities['media']]
                
                # Engagement metrics
                engagement = {
                    'retweets': tweet.retweet_count,
                    'favorites': tweet.favorite_count,
                    'replies': getattr(tweet, 'reply_count', 0)
                }
                
                # Skip low engagement posts
                total_engagement = sum(engagement.values())
                if total_engagement < self.config['min_engagement_threshold']:
                    continue
                
                authors.append(tweet.user.screen_name)
                
                post = SocialMediaPost(
                    platform='twitter',
                    post_id=tweet.id_str,
                    author=tweet.user.screen_name,
                    content=tweet.full_text,
                    phone_mentions=phone_mentions,
                    sentiment_score=sentiment,
                    engagement_metrics=engagement,
                    published_at=tweet.created_at.isoformat(),
                    url=f"https://twitter.com/{tweet.user.screen_name}/status/{tweet.id_str}",
                    hashtags=hashtags,
                    mentions=mentions,
                    media_urls=media_urls,
                    confidence_score=self._calculate_post_confidence(tweet.full_text, phone_mentions, engagement)
                )
                
                posts.append(post)
            
            # Analyze results
            sentiment_analysis = {
                'average_sentiment': statistics.mean(sentiment_scores) if sentiment_scores else 0.0,
                'positive_ratio': len([s for s in sentiment_scores if s > 0.1]) / len(sentiment_scores) if sentiment_scores else 0,
                'negative_ratio': len([s for s in sentiment_scores if s < -0.1]) / len(sentiment_scores) if sentiment_scores else 0
            }
            
            # Top hashtags
            hashtag_counts = Counter(all_hashtags)
            trending_topics = [tag for tag, count in hashtag_counts.most_common(10)]
            
            # Top influencers (by frequency)
            author_counts = Counter(authors)
            top_influencers = [author for author, count in author_counts.most_common(5)]
            
            return SocialMediaSearchResult(
                query=phone_query,
                platform='twitter',
                total_posts=len(posts),
                posts=posts,
                sentiment_analysis=sentiment_analysis,
                trending_topics=trending_topics,
                top_influencers=top_influencers,
                search_timestamp=datetime.now().isoformat(),
                query_confidence=self._calculate_query_confidence(posts, phone_query)
            )
            
        except Exception as e:
            logger.error(f"Twitter search failed: {e}")
            return SocialMediaSearchResult(
                query=phone_query,
                platform='twitter',
                total_posts=0,
                posts=[],
                sentiment_analysis={},
                trending_topics=[],
                top_influencers=[],
                search_timestamp=datetime.now().isoformat(),
                query_confidence=0.0
            )
    
    async def _search_reddit(self, phone_query: str) -> SocialMediaSearchResult:
        """Search Reddit for phone discussions"""
        
        if not self.reddit_client:
            logger.warning("Reddit client not initialized")
            return SocialMediaSearchResult(
                query=phone_query,
                platform='reddit',
                total_posts=0,
                posts=[],
                sentiment_analysis={},
                trending_topics=[],
                top_influencers=[],
                search_timestamp=datetime.now().isoformat(),
                query_confidence=0.0
            )
        
        try:
            posts = []
            all_topics = []
            authors = []
            sentiment_scores = []
            
            # Search relevant subreddits
            subreddits = self.forum_configs['reddit_android']['subreddits']
            
            for subreddit_name in subreddits:
                try:
                    subreddit = self.reddit_client.subreddit(subreddit_name)
                    
                    # Search recent posts
                    search_results = list(subreddit.search(
                        phone_query,
                        sort='new',
                        time_filter='week',
                        limit=self.config['max_posts_per_platform'] // len(subreddits)
                    ))
                    
                    for submission in search_results:
                        # Extract phone mentions
                        content = f"{submission.title} {submission.selftext}"
                        phone_mentions = self._extract_phone_mentions(content)
                        
                        # Calculate sentiment
                        sentiment = self._calculate_sentiment(content)
                        sentiment_scores.append(sentiment)
                        
                        # Engagement metrics
                        engagement = {
                            'upvotes': submission.score,
                            'comments': submission.num_comments,
                            'upvote_ratio': getattr(submission, 'upvote_ratio', 0.5)
                        }
                        
                        # Skip low engagement posts
                        if submission.score < self.config['min_engagement_threshold']:
                            continue
                        
                        authors.append(submission.author.name if submission.author else 'deleted')
                        
                        # Extract topics (from title keywords)
                        topics = re.findall(r'\b\w{4,}\b', submission.title.lower())
                        all_topics.extend(topics)
                        
                        post = SocialMediaPost(
                            platform='reddit',
                            post_id=submission.id,
                            author=submission.author.name if submission.author else 'deleted',
                            content=content,
                            phone_mentions=phone_mentions,
                            sentiment_score=sentiment,
                            engagement_metrics=engagement,
                            published_at=datetime.fromtimestamp(submission.created_utc).isoformat(),
                            url=f"https://reddit.com{submission.permalink}",
                            hashtags=[],
                            mentions=[],
                            media_urls=[submission.url] if submission.url and submission.url.startswith('http') else [],
                            confidence_score=self._calculate_post_confidence(content, phone_mentions, engagement)
                        )
                        
                        posts.append(post)
                    
                    # Rate limiting
                    time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error searching subreddit {subreddit_name}: {e}")
                    continue
            
            # Analyze results
            sentiment_analysis = {
                'average_sentiment': statistics.mean(sentiment_scores) if sentiment_scores else 0.0,
                'positive_ratio': len([s for s in sentiment_scores if s > 0.1]) / len(sentiment_scores) if sentiment_scores else 0,
                'negative_ratio': len([s for s in sentiment_scores if s < -0.1]) / len(sentiment_scores) if sentiment_scores else 0
            }
            
            # Top topics
            topic_counts = Counter(all_topics)
            trending_topics = [topic for topic, count in topic_counts.most_common(10)]
            
            # Top contributors
            author_counts = Counter(authors)
            top_influencers = [author for author, count in author_counts.most_common(5)]
            
            return SocialMediaSearchResult(
                query=phone_query,
                platform='reddit',
                total_posts=len(posts),
                posts=posts,
                sentiment_analysis=sentiment_analysis,
                trending_topics=trending_topics,
                top_influencers=top_influencers,
                search_timestamp=datetime.now().isoformat(),
                query_confidence=self._calculate_query_confidence(posts, phone_query)
            )
            
        except Exception as e:
            logger.error(f"Reddit search failed: {e}")
            return SocialMediaSearchResult(
                query=phone_query,
                platform='reddit',
                total_posts=0,
                posts=[],
                sentiment_analysis={},
                trending_topics=[],
                top_influencers=[],
                search_timestamp=datetime.now().isoformat(),
                query_confidence=0.0
            )
    
    async def _search_xda(self, phone_query: str) -> SocialMediaSearchResult:
        """Search XDA Developers forum"""
        
        try:
            config = self.forum_configs['xda_developers']
            search_url = f"{config['base_url']}{config['search_endpoint']}"
            
            # Build search parameters
            params = {
                'q': phone_query,
                'o': 'relevance',
                'c[node]': '2',  # Android section
            }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            async with self.session.get(search_url, params=params, headers=headers) as response:
                if response.status != 200:
                    logger.warning(f"XDA search returned {response.status}")
                    return self._empty_search_result(phone_query, 'xda_developers')
                
                content = await response.text()
                soup = BeautifulSoup(content, 'html.parser')
                
                posts = []
                sentiment_scores = []
                
                # Extract search results
                result_elements = soup.find_all('div', class_='contentRow')
                
                for element in result_elements[:self.config['max_posts_per_platform']]:
                    try:
                        # Extract basic information
                        title_elem = element.select_one(config['selectors']['title'])
                        author_elem = element.select_one(config['selectors']['author'])
                        
                        if not title_elem:
                            continue
                        
                        title = title_elem.get_text(strip=True)
                        author = author_elem.get_text(strip=True) if author_elem else 'Unknown'
                        thread_url = urljoin(config['base_url'], title_elem.get('href', ''))
                        
                        # Extract phone mentions
                        phone_mentions = self._extract_phone_mentions(title)
                        
                        # Calculate sentiment
                        sentiment = self._calculate_sentiment(title)
                        sentiment_scores.append(sentiment)
                        
                        # Try to get engagement metrics
                        replies_elem = element.select_one(config['selectors'].get('replies', ''))
                        views_elem = element.select_one(config['selectors'].get('views', ''))
                        
                        replies = self._extract_number(replies_elem.get_text() if replies_elem else '0')
                        views = self._extract_number(views_elem.get_text() if views_elem else '0')
                        
                        engagement = {
                            'replies': replies,
                            'views': views,
                            'total': replies + (views // 10)  # Weight views less
                        }
                        
                        forum_post = ForumPost(
                            forum_name='XDA Developers',
                            thread_title=title,
                            post_id=hashlib.md5(thread_url.encode()).hexdigest()[:8],
                            author=author,
                            content=title,  # Only title available in search results
                            phone_mentions=phone_mentions,
                            post_type='discussion',
                            replies_count=replies,
                            views_count=views,
                            upvotes=0,
                            published_at=datetime.now().isoformat(),  # Not available in search
                            thread_url=thread_url,
                            tags=[],
                            is_solution=False,
                            confidence_score=self._calculate_post_confidence(title, phone_mentions, engagement)
                        )
                        
                        posts.append(forum_post)
                        
                    except Exception as e:
                        logger.debug(f"Error parsing XDA result: {e}")
                        continue
                
                # Analyze results
                sentiment_analysis = {
                    'average_sentiment': statistics.mean(sentiment_scores) if sentiment_scores else 0.0,
                    'positive_ratio': len([s for s in sentiment_scores if s > 0.1]) / len(sentiment_scores) if sentiment_scores else 0,
                    'negative_ratio': len([s for s in sentiment_scores if s < -0.1]) / len(sentiment_scores) if sentiment_scores else 0
                }
                
                return SocialMediaSearchResult(
                    query=phone_query,
                    platform='xda_developers',
                    total_posts=len(posts),
                    posts=posts,
                    sentiment_analysis=sentiment_analysis,
                    trending_topics=[],
                    top_influencers=[],
                    search_timestamp=datetime.now().isoformat(),
                    query_confidence=self._calculate_query_confidence(posts, phone_query)
                )
        
        except Exception as e:
            logger.error(f"XDA search failed: {e}")
            return self._empty_search_result(phone_query, 'xda_developers')
    
    async def _search_android_forums(self, phone_query: str) -> SocialMediaSearchResult:
        """Search AndroidForums.com"""
        
        try:
            config = self.forum_configs['android_forums']
            search_url = f"{config['base_url']}{config['search_endpoint']}"
            
            params = {
                'keywords': phone_query,
                'sortby': 'relevancy'
            }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            async with self.session.get(search_url, params=params, headers=headers) as response:
                if response.status != 200:
                    logger.warning(f"Android Forums search returned {response.status}")
                    return self._empty_search_result(phone_query, 'android_forums')
                
                content = await response.text()
                soup = BeautifulSoup(content, 'html.parser')
                
                posts = []
                sentiment_scores = []
                
                # Extract search results
                result_elements = soup.find_all('li', class_='searchResult')
                
                for element in result_elements[:self.config['max_posts_per_platform']]:
                    try:
                        # Extract basic information
                        title_elem = element.select_one(config['selectors']['title'])
                        author_elem = element.select_one(config['selectors']['author'])
                        content_elem = element.select_one(config['selectors']['content'])
                        
                        if not title_elem:
                            continue
                        
                        title = title_elem.get_text(strip=True)
                        author = author_elem.get_text(strip=True) if author_elem else 'Unknown'
                        content_text = content_elem.get_text(strip=True) if content_elem else title
                        thread_url = urljoin(config['base_url'], title_elem.get('href', ''))
                        
                        # Extract phone mentions
                        full_text = f"{title} {content_text}"
                        phone_mentions = self._extract_phone_mentions(full_text)
                        
                        # Calculate sentiment
                        sentiment = self._calculate_sentiment(full_text)
                        sentiment_scores.append(sentiment)
                        
                        # Try to get engagement metrics
                        replies_elem = element.select_one(config['selectors'].get('replies', ''))
                        replies = self._extract_number(replies_elem.get_text() if replies_elem else '0')
                        
                        engagement = {
                            'replies': replies,
                            'views': 0,  # Not easily available
                            'total': replies
                        }
                        
                        forum_post = ForumPost(
                            forum_name='Android Forums',
                            thread_title=title,
                            post_id=hashlib.md5(thread_url.encode()).hexdigest()[:8],
                            author=author,
                            content=content_text,
                            phone_mentions=phone_mentions,
                            post_type='discussion',
                            replies_count=replies,
                            views_count=0,
                            upvotes=0,
                            published_at=datetime.now().isoformat(),
                            thread_url=thread_url,
                            tags=[],
                            is_solution=False,
                            confidence_score=self._calculate_post_confidence(full_text, phone_mentions, engagement)
                        )
                        
                        posts.append(forum_post)
                        
                    except Exception as e:
                        logger.debug(f"Error parsing Android Forums result: {e}")
                        continue
                
                # Analyze results
                sentiment_analysis = {
                    'average_sentiment': statistics.mean(sentiment_scores) if sentiment_scores else 0.0,
                    'positive_ratio': len([s for s in sentiment_scores if s > 0.1]) / len(sentiment_scores) if sentiment_scores else 0,
                    'negative_ratio': len([s for s in sentiment_scores if s < -0.1]) / len(sentiment_scores) if sentiment_scores else 0
                }
                
                return SocialMediaSearchResult(
                    query=phone_query,
                    platform='android_forums',
                    total_posts=len(posts),
                    posts=posts,
                    sentiment_analysis=sentiment_analysis,
                    trending_topics=[],
                    top_influencers=[],
                    search_timestamp=datetime.now().isoformat(),
                    query_confidence=self._calculate_query_confidence(posts, phone_query)
                )
        
        except Exception as e:
            logger.error(f"Android Forums search failed: {e}")
            return self._empty_search_result(phone_query, 'android_forums')
    
    def _extract_phone_mentions(self, text: str) -> List[str]:
        """Extract phone model mentions from text"""
        
        mentions = []
        text_lower = text.lower()
        
        for pattern in self.phone_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            mentions.extend(matches)
        
        # Remove duplicates and normalize
        unique_mentions = []
        for mention in mentions:
            normalized = mention.strip()
            if normalized and normalized not in unique_mentions:
                unique_mentions.append(normalized)
        
        return unique_mentions[:5]  # Limit to 5 mentions
    
    def _calculate_sentiment(self, text: str) -> float:
        """Calculate sentiment score for text"""
        
        if not self.config['sentiment_analysis_enabled']:
            return 0.0
        
        text_lower = text.lower()
        
        positive_count = sum(text_lower.count(word) for word in self.sentiment_keywords['positive'])
        negative_count = sum(text_lower.count(word) for word in self.sentiment_keywords['negative'])
        
        total_words = len(text.split())
        if total_words == 0:
            return 0.0
        
        # Normalize by text length
        positive_score = positive_count / total_words
        negative_score = negative_count / total_words
        
        # Return score between -1 and 1
        net_score = positive_score - negative_score
        return max(min(net_score * 10, 1.0), -1.0)
    
    def _calculate_post_confidence(self, content: str, phone_mentions: List[str], 
                                 engagement: Dict[str, int]) -> float:
        """Calculate confidence score for a post"""
        
        score = 0.5  # Base score
        
        # Phone mentions
        if phone_mentions:
            score += min(len(phone_mentions) * 0.1, 0.3)
        
        # Content length
        if len(content) > 50:
            score += 0.1
        if len(content) > 200:
            score += 0.1
        
        # Engagement
        total_engagement = sum(engagement.values())
        if total_engagement > 5:
            score += 0.1
        if total_engagement > 50:
            score += 0.1
        
        # Phone-specific keywords
        phone_keywords = ['review', 'camera', 'battery', 'performance', 'specs', 'price']
        keyword_count = sum(1 for keyword in phone_keywords if keyword in content.lower())
        score += min(keyword_count * 0.05, 0.2)
        
        return min(score, 1.0)
    
    def _calculate_query_confidence(self, posts: List[Any], query: str) -> float:
        """Calculate overall confidence for query results"""
        
        if not posts:
            return 0.0
        
        # Average post confidence
        avg_confidence = statistics.mean([post.confidence_score for post in posts])
        
        # Query relevance
        query_mentions = sum(1 for post in posts if query.lower() in post.content.lower())
        query_relevance = query_mentions / len(posts)
        
        # Combine factors
        overall_confidence = (avg_confidence + query_relevance) / 2
        
        return min(overall_confidence, 1.0)
    
    def _extract_number(self, text: str) -> int:
        """Extract number from text string"""
        
        match = re.search(r'\d+', text)
        return int(match.group()) if match else 0
    
    def _empty_search_result(self, query: str, platform: str) -> SocialMediaSearchResult:
        """Create empty search result"""
        
        return SocialMediaSearchResult(
            query=query,
            platform=platform,
            total_posts=0,
            posts=[],
            sentiment_analysis={},
            trending_topics=[],
            top_influencers=[],
            search_timestamp=datetime.now().isoformat(),
            query_confidence=0.0
        )
    
    def get_aggregated_insights(self, results: Dict[str, SocialMediaSearchResult]) -> Dict[str, Any]:
        """Get aggregated insights from all platform results"""
        
        insights = {
            'total_posts': 0,
            'platforms_searched': len(results),
            'overall_sentiment': 0.0,
            'top_mentions': [],
            'platform_breakdown': {},
            'trending_topics': [],
            'key_influencers': [],
            'confidence_score': 0.0
        }
        
        all_posts = []
        all_sentiments = []
        all_mentions = []
        all_topics = []
        all_influencers = []
        
        for platform, result in results.items():
            insights['platform_breakdown'][platform] = {
                'posts': result.total_posts,
                'sentiment': result.sentiment_analysis.get('average_sentiment', 0.0)
            }
            
            all_posts.extend(result.posts)
            if result.sentiment_analysis.get('average_sentiment') is not None:
                all_sentiments.append(result.sentiment_analysis['average_sentiment'])
            
            # Collect mentions from all posts
            for post in result.posts:
                if hasattr(post, 'phone_mentions'):
                    all_mentions.extend(post.phone_mentions)
            
            all_topics.extend(result.trending_topics)
            all_influencers.extend(result.top_influencers)
        
        insights['total_posts'] = len(all_posts)
        
        # Overall sentiment
        if all_sentiments:
            insights['overall_sentiment'] = statistics.mean(all_sentiments)
        
        # Top phone mentions
        mention_counts = Counter(all_mentions)
        insights['top_mentions'] = [
            {'phone': mention, 'count': count}
            for mention, count in mention_counts.most_common(10)
        ]
        
        # Trending topics
        topic_counts = Counter(all_topics)
        insights['trending_topics'] = [
            topic for topic, count in topic_counts.most_common(15)
        ]
        
        # Key influencers
        influencer_counts = Counter(all_influencers)
        insights['key_influencers'] = [
            influencer for influencer, count in influencer_counts.most_common(10)
        ]
        
        # Overall confidence
        if all_posts:
            confidences = [post.confidence_score for post in all_posts if hasattr(post, 'confidence_score')]
            insights['confidence_score'] = statistics.mean(confidences) if confidences else 0.0
        
        return insights
    
    # Utility methods
    async def _init_session(self):
        """Initialize aiohttp session"""
        if not self.session:
            connector = aiohttp.TCPConnector(limit=10)
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
    
    async def _cleanup_session(self):
        """Cleanup aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None

# Factory function
def create_social_media_search_engine(config=None):
    """Create configured social media search engine"""
    return SocialMediaSearchEngine(config=config)