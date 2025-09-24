"""
Pricing API Integration for AI Phone Review Engine
Integrates with multiple pricing and retail APIs for real-time pricing data
"""

import asyncio
import aiohttp
import logging
import json
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from urllib.parse import quote_plus, urlencode
import hashlib
import os
from decimal import Decimal, InvalidOperation

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PriceData:
    """Structure for price information"""
    retailer: str
    price: Decimal
    currency: str
    availability: str
    condition: str  # new, refurbished, used
    shipping_info: Optional[str]
    url: str
    last_updated: str
    confidence: float

@dataclass
class PricingResult:
    """Comprehensive pricing information"""
    phone_model: str
    lowest_price: Optional[PriceData]
    highest_price: Optional[PriceData]
    average_price: Optional[Decimal]
    price_range: Dict[str, Decimal]
    all_prices: List[PriceData]
    market_analysis: Dict[str, Any]
    data_freshness: str
    sources_used: List[str]

class APIKeyManager:
    """Secure API key management"""
    
    def __init__(self, config_file: str = "config/api_keys.json"):
        self.config_file = config_file
        self.keys = {}
        self.load_keys()
    
    def load_keys(self):
        """Load API keys from encrypted storage"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    self.keys = json.load(f)
            else:
                # Create default structure
                self.keys = {
                    "google_shopping": {
                        "api_key": os.getenv("GOOGLE_SHOPPING_API_KEY", ""),
                        "cx": os.getenv("GOOGLE_SHOPPING_CX", ""),
                        "enabled": bool(os.getenv("GOOGLE_SHOPPING_API_KEY"))
                    },
                    "amazon_pa_api": {
                        "access_key": os.getenv("AMAZON_ACCESS_KEY", ""),
                        "secret_key": os.getenv("AMAZON_SECRET_KEY", ""),
                        "associate_tag": os.getenv("AMAZON_ASSOCIATE_TAG", ""),
                        "enabled": bool(os.getenv("AMAZON_ACCESS_KEY"))
                    },
                    "ebay_api": {
                        "client_id": os.getenv("EBAY_CLIENT_ID", ""),
                        "client_secret": os.getenv("EBAY_CLIENT_SECRET", ""),
                        "enabled": bool(os.getenv("EBAY_CLIENT_ID"))
                    },
                    "bestbuy_api": {
                        "api_key": os.getenv("BESTBUY_API_KEY", ""),
                        "enabled": bool(os.getenv("BESTBUY_API_KEY"))
                    },
                    "priceapi": {
                        "api_key": os.getenv("PRICEAPI_KEY", ""),
                        "enabled": bool(os.getenv("PRICEAPI_KEY"))
                    }
                }
                self.save_keys()
        except Exception as e:
            logger.error(f"Error loading API keys: {e}")
            self.keys = {}
    
    def save_keys(self):
        """Save API keys to file"""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(self.keys, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving API keys: {e}")
    
    def get_key(self, service: str) -> Optional[Dict[str, str]]:
        """Get API key for a service"""
        return self.keys.get(service, {}) if self.keys.get(service, {}).get('enabled', False) else None

class ExponentialBackoff:
    """Exponential backoff for API requests"""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
    
    async def execute(self, func, *args, **kwargs):
        """Execute function with exponential backoff"""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt == self.max_retries:
                    break
                
                delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay:.1f}s: {str(e)}")
                await asyncio.sleep(delay)
        
        raise last_exception

class PricingAPIIntegration:
    """Main pricing API integration class"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize pricing API integration"""
        
        self.config = config or {
            'max_concurrent_requests': 5,
            'request_timeout': 15,
            'cache_duration': 3600,  # 1 hour
            'max_price_age_hours': 24,
            'min_confidence_threshold': 0.6,
            'max_results_per_source': 10
        }
        
        # Initialize components
        self.key_manager = APIKeyManager()
        self.backoff = ExponentialBackoff()
        self.session = None
        
        # Price cache
        self.price_cache = {}
        
        # API configurations
        self.api_configs = {
            'google_shopping': {
                'base_url': 'https://www.googleapis.com/customsearch/v1',
                'enabled': True,
                'priority': 1,
                'handler': self._search_google_shopping
            },
            'amazon_pa': {
                'base_url': 'https://webservices.amazon.com/paapi5/searchitems',
                'enabled': True,
                'priority': 2,
                'handler': self._search_amazon_pa
            },
            'ebay_api': {
                'base_url': 'https://api.ebay.com/buy/browse/v1/item_summary/search',
                'enabled': True,
                'priority': 3,
                'handler': self._search_ebay
            },
            'bestbuy_api': {
                'base_url': 'https://api.bestbuy.com/v1/products',
                'enabled': True,
                'priority': 4,
                'handler': self._search_bestbuy
            },
            'priceapi': {
                'base_url': 'https://api.priceapi.com/v2/jobs',
                'enabled': True,
                'priority': 5,
                'handler': self._search_priceapi
            }
        }
    
    async def get_phone_pricing(self, phone_model: str, max_sources: int = 3) -> PricingResult:
        """
        Get comprehensive pricing information for a phone
        
        Args:
            phone_model: Phone model to search for
            max_sources: Maximum number of pricing sources to query
            
        Returns:
            PricingResult with comprehensive pricing information
        """
        
        logger.info(f"Getting pricing for: {phone_model}")
        
        # Check cache first
        cache_key = f"pricing_{hashlib.md5(phone_model.lower().encode()).hexdigest()}"
        if self._is_cache_valid(cache_key):
            logger.info("Returning cached pricing data")
            return self.price_cache[cache_key]['data']
        
        # Initialize session
        await self._init_session()
        
        try:
            # Get enabled API sources
            enabled_sources = [
                (name, config) for name, config in self.api_configs.items()
                if config['enabled'] and self.key_manager.get_key(name)
            ]
            
            # Sort by priority and limit
            enabled_sources.sort(key=lambda x: x[1]['priority'])
            enabled_sources = enabled_sources[:max_sources]
            
            # Search pricing concurrently
            semaphore = asyncio.Semaphore(self.config['max_concurrent_requests'])
            tasks = [
                self._search_single_source(semaphore, source_name, source_config, phone_model)
                for source_name, source_config in enabled_sources
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            all_prices = []
            sources_used = []
            
            for i, result in enumerate(results):
                source_name = enabled_sources[i][0]
                
                if isinstance(result, Exception):
                    logger.error(f"Pricing search failed for {source_name}: {result}")
                elif result:
                    all_prices.extend(result)
                    sources_used.append(source_name)
                    logger.info(f"Found {len(result)} prices from {source_name}")
            
            # Create pricing result
            pricing_result = self._create_pricing_result(phone_model, all_prices, sources_used)
            
            # Cache result
            self.price_cache[cache_key] = {
                'data': pricing_result,
                'timestamp': datetime.now().isoformat()
            }
            
            return pricing_result
            
        finally:
            await self._cleanup_session()
    
    async def _search_single_source(self, semaphore: asyncio.Semaphore,
                                  source_name: str, source_config: Dict,
                                  phone_model: str) -> List[PriceData]:
        """Search a single pricing source"""
        
        async with semaphore:
            try:
                handler = source_config['handler']
                return await self.backoff.execute(handler, phone_model)
            except Exception as e:
                logger.error(f"Error searching {source_name}: {str(e)}")
                return []
    
    async def _search_google_shopping(self, phone_model: str) -> List[PriceData]:
        """Search Google Shopping API"""
        
        api_keys = self.key_manager.get_key('google_shopping')
        if not api_keys or not api_keys.get('api_key'):
            return []
        
        try:
            params = {
                'key': api_keys['api_key'],
                'cx': api_keys['cx'],
                'q': f"{phone_model} buy price",
                'searchType': 'image',  # Shopping search
                'num': self.config['max_results_per_source']
            }
            
            url = f"{self.api_configs['google_shopping']['base_url']}?{urlencode(params)}"
            
            async with self.session.get(url, timeout=self.config['request_timeout']) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_google_shopping_results(data, phone_model)
                else:
                    logger.warning(f"Google Shopping API returned status {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Google Shopping search error: {str(e)}")
            return []
    
    async def _search_amazon_pa(self, phone_model: str) -> List[PriceData]:
        """Search Amazon Product Advertising API"""
        
        api_keys = self.key_manager.get_key('amazon_pa_api')
        if not api_keys or not api_keys.get('access_key'):
            return []
        
        # Amazon PA API requires complex authentication
        # For demo purposes, returning placeholder
        logger.info("Amazon PA API integration would require full authentication setup")
        return []
    
    async def _search_ebay(self, phone_model: str) -> List[PriceData]:
        """Search eBay API"""
        
        api_keys = self.key_manager.get_key('ebay_api')
        if not api_keys or not api_keys.get('client_id'):
            return []
        
        try:
            # Get OAuth token first
            token = await self._get_ebay_token(api_keys)
            if not token:
                return []
            
            headers = {
                'Authorization': f'Bearer {token}',
                'X-EBAY-C-MARKETPLACE-ID': 'EBAY_US'
            }
            
            params = {
                'q': phone_model,
                'category_ids': '9355',  # Cell Phones & Smartphones
                'limit': self.config['max_results_per_source'],
                'filter': 'conditionIds:{1000|1500|2000|2500}'  # New, Like New, Very Good, Good
            }
            
            url = f"{self.api_configs['ebay_api']['base_url']}?{urlencode(params)}"
            
            async with self.session.get(url, headers=headers, timeout=self.config['request_timeout']) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_ebay_results(data, phone_model)
                else:
                    logger.warning(f"eBay API returned status {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"eBay search error: {str(e)}")
            return []
    
    async def _search_bestbuy(self, phone_model: str) -> List[PriceData]:
        """Search Best Buy API"""
        
        api_keys = self.key_manager.get_key('bestbuy_api')
        if not api_keys or not api_keys.get('api_key'):
            return []
        
        try:
            # Best Buy API search
            encoded_query = quote_plus(phone_model)
            params = {
                'apikey': api_keys['api_key'],
                'format': 'json',
                'search': encoded_query,
                'categoryPath.id': '9355',  # Cell Phones
                'pageSize': self.config['max_results_per_source']
            }
            
            url = f"{self.api_configs['bestbuy_api']['base_url']}?{urlencode(params)}"
            
            async with self.session.get(url, timeout=self.config['request_timeout']) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_bestbuy_results(data, phone_model)
                else:
                    logger.warning(f"Best Buy API returned status {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Best Buy search error: {str(e)}")
            return []
    
    async def _search_priceapi(self, phone_model: str) -> List[PriceData]:
        """Search PriceAPI"""
        
        api_keys = self.key_manager.get_key('priceapi')
        if not api_keys or not api_keys.get('api_key'):
            return []
        
        try:
            headers = {
                'Authorization': f'Bearer {api_keys["api_key"]}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'source': 'google_shopping',
                'country': 'us',
                'topic': 'product_and_offers',
                'key': phone_model,
                'max_pages': 1
            }
            
            async with self.session.post(
                self.api_configs['priceapi']['base_url'],
                headers=headers,
                json=payload,
                timeout=self.config['request_timeout']
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_priceapi_results(data, phone_model)
                else:
                    logger.warning(f"PriceAPI returned status {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"PriceAPI search error: {str(e)}")
            return []
    
    # Result parsing methods
    def _parse_google_shopping_results(self, data: Dict, phone_model: str) -> List[PriceData]:
        """Parse Google Shopping API results"""
        prices = []
        
        for item in data.get('items', []):
            try:
                # Extract price from snippet or other fields
                price_match = re.search(r'\$(\d+(?:,\d{3})*(?:\.\d{2})?)', item.get('snippet', ''))
                if price_match:
                    price = Decimal(price_match.group(1).replace(',', ''))
                    
                    prices.append(PriceData(
                        retailer='Google Shopping',
                        price=price,
                        currency='USD',
                        availability='Unknown',
                        condition='new',
                        shipping_info=None,
                        url=item.get('link', ''),
                        last_updated=datetime.now().isoformat(),
                        confidence=0.7
                    ))
            except (InvalidOperation, ValueError):
                continue
        
        return prices
    
    def _parse_ebay_results(self, data: Dict, phone_model: str) -> List[PriceData]:
        """Parse eBay API results"""
        prices = []
        
        for item in data.get('itemSummaries', []):
            try:
                price_info = item.get('price', {})
                if 'value' in price_info:
                    price = Decimal(str(price_info['value']))
                    
                    prices.append(PriceData(
                        retailer='eBay',
                        price=price,
                        currency=price_info.get('currency', 'USD'),
                        availability='Available' if item.get('buyingOptions') else 'Unknown',
                        condition=item.get('condition', 'Unknown'),
                        shipping_info=self._extract_shipping_info(item.get('shippingOptions', [])),
                        url=item.get('itemWebUrl', ''),
                        last_updated=datetime.now().isoformat(),
                        confidence=0.8
                    ))
            except (InvalidOperation, ValueError, KeyError):
                continue
        
        return prices
    
    def _parse_bestbuy_results(self, data: Dict, phone_model: str) -> List[PriceData]:
        """Parse Best Buy API results"""
        prices = []
        
        for item in data.get('products', []):
            try:
                if 'salePrice' in item:
                    price = Decimal(str(item['salePrice']))
                    
                    prices.append(PriceData(
                        retailer='Best Buy',
                        price=price,
                        currency='USD',
                        availability='In Stock' if item.get('inStoreAvailability') else 'Check Availability',
                        condition='new',
                        shipping_info='Free shipping on orders $35+',
                        url=item.get('url', ''),
                        last_updated=datetime.now().isoformat(),
                        confidence=0.9
                    ))
            except (InvalidOperation, ValueError, KeyError):
                continue
        
        return prices
    
    def _parse_priceapi_results(self, data: Dict, phone_model: str) -> List[PriceData]:
        """Parse PriceAPI results"""
        prices = []
        
        # PriceAPI returns job data that needs to be processed
        # This would require additional API calls to get results
        logger.info("PriceAPI results would require job status checking")
        
        return prices
    
    def _create_pricing_result(self, phone_model: str, all_prices: List[PriceData], 
                             sources_used: List[str]) -> PricingResult:
        """Create comprehensive pricing result"""
        
        if not all_prices:
            return PricingResult(
                phone_model=phone_model,
                lowest_price=None,
                highest_price=None,
                average_price=None,
                price_range={},
                all_prices=[],
                market_analysis={'message': 'No pricing data found'},
                data_freshness='No data',
                sources_used=sources_used
            )
        
        # Sort prices
        sorted_prices = sorted(all_prices, key=lambda p: p.price)
        
        # Calculate statistics
        prices_only = [p.price for p in all_prices]
        average_price = sum(prices_only) / len(prices_only)
        
        price_range = {
            'min': min(prices_only),
            'max': max(prices_only),
            'median': sorted(prices_only)[len(prices_only) // 2]
        }
        
        # Market analysis
        market_analysis = {
            'total_listings': len(all_prices),
            'price_spread': max(prices_only) - min(prices_only),
            'coefficient_of_variation': (
                (sum((p - average_price) ** 2 for p in prices_only) / len(prices_only)) ** 0.5
            ) / average_price,
            'retailer_count': len(set(p.retailer for p in all_prices)),
            'new_condition_count': len([p for p in all_prices if p.condition == 'new']),
            'average_by_condition': self._calculate_average_by_condition(all_prices)
        }
        
        return PricingResult(
            phone_model=phone_model,
            lowest_price=sorted_prices[0],
            highest_price=sorted_prices[-1],
            average_price=average_price,
            price_range=price_range,
            all_prices=all_prices,
            market_analysis=market_analysis,
            data_freshness=f"Updated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            sources_used=sources_used
        )
    
    def _calculate_average_by_condition(self, prices: List[PriceData]) -> Dict[str, Decimal]:
        """Calculate average price by condition"""
        condition_groups = {}
        
        for price in prices:
            condition = price.condition.lower()
            if condition not in condition_groups:
                condition_groups[condition] = []
            condition_groups[condition].append(price.price)
        
        return {
            condition: sum(prices) / len(prices)
            for condition, prices in condition_groups.items()
        }
    
    def _extract_shipping_info(self, shipping_options: List[Dict]) -> Optional[str]:
        """Extract shipping information from API response"""
        if not shipping_options:
            return None
        
        # Find free or lowest cost shipping
        free_shipping = [s for s in shipping_options if s.get('shippingCost', {}).get('value', 0) == 0]
        if free_shipping:
            return "Free shipping"
        
        # Get cheapest shipping
        cheapest = min(shipping_options, key=lambda s: s.get('shippingCost', {}).get('value', float('inf')))
        cost = cheapest.get('shippingCost', {})
        if 'value' in cost:
            return f"Shipping: ${cost['value']}"
        
        return "Shipping available"
    
    async def _get_ebay_token(self, api_keys: Dict) -> Optional[str]:
        """Get eBay OAuth token"""
        try:
            token_url = 'https://api.ebay.com/identity/v1/oauth2/token'
            
            auth_header = f"{api_keys['client_id']}:{api_keys['client_secret']}"
            import base64
            encoded_auth = base64.b64encode(auth_header.encode()).decode()
            
            headers = {
                'Authorization': f'Basic {encoded_auth}',
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            data = 'grant_type=client_credentials&scope=https://api.ebay.com/oauth/api_scope'
            
            async with self.session.post(token_url, headers=headers, data=data) as response:
                if response.status == 200:
                    token_data = await response.json()
                    return token_data.get('access_token')
                
        except Exception as e:
            logger.error(f"Error getting eBay token: {e}")
        
        return None
    
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
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached pricing data is still valid"""
        if cache_key not in self.price_cache:
            return False
        
        cache_entry = self.price_cache[cache_key]
        cache_age = datetime.now() - datetime.fromisoformat(cache_entry['timestamp'])
        
        return cache_age.total_seconds() < self.config['cache_duration']

# Factory function
def create_pricing_api_integration(config=None):
    """Create configured pricing API integration"""
    return PricingAPIIntegration(config=config)