"""
Enhanced Fallback Search System for AI Phone Review Engine
Provides robust search capabilities even when APIs are unavailable
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import re
import difflib
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FallbackSearchResult:
    """Structure for fallback search results"""
    phone_model: str
    source: str
    confidence: float
    data: Dict[str, Any]
    metadata: Dict[str, Any]

class FallbackSearchSystem:
    """
    Fallback search system with two main purposes:
    
    1. **Web Search**: For phones NOT found in your primary dataset
       - Searches the web for real phone information
       - Returns actual phone data from external sources
       - NO synthetic data generation
    
    2. **Alternative Suggestions**: When phones in your dataset have negative sentiment
       - Suggests real alternative phones from your dataset
       - Only triggers for phones with poor sentiment in your data
       - Helps users find better alternatives
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the fallback search system"""
        
        # Default configuration
        default_config = {
            # Web search for missing phones
            'enable_web_search': True,
            'web_search_timeout': 30,
            'web_search_max_sources': 5,
            
            # Alternative suggestions for negative sentiment
            'enable_sentiment_alternatives': True,
            'negative_sentiment_threshold': 0.4,  # When to suggest alternatives
            'max_alternatives': 3,
            
            # Offline database (for backup real phone data)
            'offline_database_path': 'data/offline_phone_db.json',
            
            # General settings
            'cache_results': True,
            'log_searches': True
        }
        
        # Merge with user config
        self.config = default_config
        if config:
            self.config.update(config)
        
        # Load offline phone database
        self.offline_db = self._load_offline_database()
        
        # Phone specifications templates for synthetic generation
        self.spec_templates = self._load_specification_templates()
        
        # Brand and model patterns
        self.brand_patterns = self._compile_brand_patterns()
        
    def search_web_for_missing_phone(self, query: str) -> Optional[FallbackSearchResult]:
        """
        Search the web for phones NOT found in your primary dataset.
        This is the main fallback when a phone is completely missing from your data.
        
        Args:
            query: Search query for the missing phone
            
        Returns:
            FallbackSearchResult with web search data, None if not found
        """
        
        logger.info(f"Web search initiated for missing phone: {query}")
        
        try:
            # Import web search agent (use enhanced version that's available)
            from .enhanced_api_web_search_agent import EnhancedAPIWebSearchAgent
            
            # Initialize web search agent
            web_agent = EnhancedAPIWebSearchAgent()
            
            # Perform web search for the missing phone
            web_result = web_agent.search_phone_external(query)
            
            if web_result and web_result.get('phone_found'):
                # Convert web result to FallbackSearchResult
                return FallbackSearchResult(
                    phone_model=web_result['phone_data']['product_name'],
                    source='web_search',
                    confidence=web_result.get('confidence', 0.8),
                    data=web_result['phone_data'],
                    metadata={
                        'search_type': 'web_search',
                        'web_sources': web_result['phone_data'].get('web_sources', []),
                        'original_query': query,
                        'found_via': 'external_web_search'
                    }
                )
            
            logger.info(f"No web search results found for: {query}")
            return None
            
        except Exception as e:
            logger.error(f"Web search failed for {query}: {e}")
            return None
    
    def suggest_alternatives_for_negative_sentiment(self, phone_data: Dict[str, Any], 
                                                  user_dataset: Any = None) -> List[FallbackSearchResult]:
        """
        Suggest alternative phones when a phone in your dataset has negative sentiment.
        Only suggests phones from your actual dataset, not synthetic data.
        
        Args:
            phone_data: Data of the phone with negative sentiment
            user_dataset: Your primary dataset to find alternatives from
            
        Returns:
            List of alternative phone suggestions from your dataset
        """
        
        logger.info(f"Finding alternatives for phone with negative sentiment: {phone_data.get('product', 'Unknown')}")
        
        alternatives = []
        
        if user_dataset is None:
            logger.warning("No dataset provided for finding alternatives")
            return alternatives
        
        try:
            import pandas as pd
            
            # Get phone details
            original_brand = phone_data.get('brand', '').lower()
            original_price_range = self._estimate_price_range_from_data(phone_data)
            
            # Filter dataset for alternatives
            df = user_dataset
            
            # Find phones from same brand with better sentiment
            if original_brand and 'brand' in df.columns:
                brand_phones = df[df['brand'].str.lower() == original_brand]
                
                # Filter by sentiment if available
                if 'sentiment_label' in brand_phones.columns:
                    positive_phones = brand_phones[brand_phones['sentiment_label'] == 'positive']
                    
                    # Get top rated alternatives
                    if 'rating' in positive_phones.columns and len(positive_phones) > 0:
                        top_alternatives = positive_phones.nlargest(3, 'rating')
                        
                        for _, alt_phone in top_alternatives.iterrows():
                            alternatives.append(FallbackSearchResult(
                                phone_model=alt_phone['product'],
                                source='dataset_alternative',
                                confidence=0.9,
                                data=alt_phone.to_dict(),
                                metadata={
                                    'search_type': 'sentiment_alternative',
                                    'reason': 'Better sentiment alternative from same brand',
                                    'original_phone': phone_data.get('product', 'Unknown'),
                                    'alternative_type': 'same_brand_better_sentiment'
                                }
                            ))
            
            # If no same-brand alternatives, find similar price range with better sentiment
            if len(alternatives) == 0:
                # Find phones in similar price range with positive sentiment
                if 'sentiment_label' in df.columns and 'rating' in df.columns:
                    positive_phones = df[df['sentiment_label'] == 'positive']
                    
                    if len(positive_phones) > 0:
                        # Get top 2 highest rated as alternatives
                        top_alternatives = positive_phones.nlargest(2, 'rating')
                        
                        for _, alt_phone in top_alternatives.iterrows():
                            alternatives.append(FallbackSearchResult(
                                phone_model=alt_phone['product'],
                                source='dataset_alternative',
                                confidence=0.7,
                                data=alt_phone.to_dict(),
                                metadata={
                                    'search_type': 'sentiment_alternative',
                                    'reason': 'Better sentiment alternative',
                                    'original_phone': phone_data.get('product', 'Unknown'),
                                    'alternative_type': 'better_sentiment_general'
                                }
                            ))
            
            logger.info(f"Found {len(alternatives)} alternatives for negative sentiment phone")
            return alternatives[:3]  # Limit to 3 alternatives
            
        except Exception as e:
            logger.error(f"Error finding alternatives: {e}")
            return []
    
    def _load_offline_database(self) -> Dict[str, Dict[str, Any]]:
        """Load the offline phone database"""
        
        # Try to load from file first
        db_path = Path(self.config['offline_database_path'])
        if db_path.exists():
            try:
                with open(db_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load offline database: {e}")
        
        # Return comprehensive default database
        return self._create_comprehensive_phone_database()
    
    def _create_comprehensive_phone_database(self) -> Dict[str, Dict[str, Any]]:
        """Create a comprehensive offline phone database"""
        
        phones_db = {
            # Apple iPhones
            'iphone_15_pro': {
                'model': 'iPhone 15 Pro',
                'brand': 'Apple',
                'variants': ['iPhone 15 Pro', 'iPhone 15 Pro Max'],
                'launch_year': 2023,
                'rating': 4.5,
                'price_range': {'min': 999, 'max': 1199, 'currency': 'USD'},
                'key_features': [
                    'A17 Pro chip', 'Titanium design', 'Action Button', 
                    'USB-C', '48MP Main camera', 'ProRes video'
                ],
                'pros': [
                    'Exceptional performance', 'Premium titanium build',
                    'Excellent camera system', 'Long software support',
                    'USB-C connectivity'
                ],
                'cons': [
                    'Very expensive', 'Limited customization',
                    'No charger included', 'Small battery'
                ],
                'specifications': {
                    'display': '6.1-inch Super Retina XDR OLED',
                    'resolution': '2556×1179 pixels',
                    'processor': 'A17 Pro',
                    'ram': '8GB',
                    'storage_options': ['128GB', '256GB', '512GB', '1TB'],
                    'main_camera': '48MP f/1.78',
                    'front_camera': '12MP f/1.9',
                    'battery': '3274mAh',
                    'os': 'iOS 17',
                    'connectivity': ['5G', 'Wi-Fi 6E', 'Bluetooth 5.3'],
                    'water_resistance': 'IP68'
                },
                'similar_phones': ['iPhone 15', 'iPhone 14 Pro', 'Samsung Galaxy S24'],
                'target_users': ['Professional photographers', 'Power users', 'iOS enthusiasts']
            },
            
            'iphone_14_pro': {
                'model': 'iPhone 14 Pro',
                'brand': 'Apple',
                'variants': ['iPhone 14 Pro', 'iPhone 14 Pro Max'],
                'launch_year': 2022,
                'rating': 4.4,
                'price_range': {'min': 899, 'max': 1099, 'currency': 'USD'},
                'key_features': [
                    'A16 Bionic chip', 'Dynamic Island', '48MP Main camera',
                    'Always-On display', 'ProMotion 120Hz'
                ],
                'pros': [
                    'Great performance', 'Innovative Dynamic Island',
                    'Excellent cameras', 'Premium build quality'
                ],
                'cons': [
                    'Expensive', 'Lightning connector', 'No significant battery improvement'
                ],
                'specifications': {
                    'display': '6.1-inch Super Retina XDR OLED',
                    'resolution': '2556×1179 pixels',
                    'processor': 'A16 Bionic',
                    'ram': '6GB',
                    'storage_options': ['128GB', '256GB', '512GB', '1TB'],
                    'main_camera': '48MP f/1.78',
                    'front_camera': '12MP f/1.9',
                    'battery': '3200mAh',
                    'os': 'iOS 16',
                    'connectivity': ['5G', 'Wi-Fi 6', 'Bluetooth 5.3']
                },
                'similar_phones': ['iPhone 15 Pro', 'iPhone 13 Pro', 'Samsung Galaxy S23'],
                'target_users': ['Professional users', 'Content creators', 'Apple ecosystem users']
            },
            
            # Samsung Galaxy
            'galaxy_s24_ultra': {
                'model': 'Samsung Galaxy S24 Ultra',
                'brand': 'Samsung',
                'variants': ['Galaxy S24 Ultra'],
                'launch_year': 2024,
                'rating': 4.4,
                'price_range': {'min': 1199, 'max': 1419, 'currency': 'USD'},
                'key_features': [
                    'S Pen included', '200MP camera', 'AI features',
                    'Titanium frame', '100x Space Zoom', 'Galaxy AI'
                ],
                'pros': [
                    'Incredible zoom capabilities', 'S Pen functionality',
                    'Large beautiful display', 'Excellent performance',
                    'Long battery life'
                ],
                'cons': [
                    'Very expensive', 'Large and heavy', 'Complex UI',
                    'Bloatware', 'S Pen easy to lose'
                ],
                'specifications': {
                    'display': '6.8-inch Dynamic AMOLED 2X',
                    'resolution': '3120×1440 pixels',
                    'processor': 'Snapdragon 8 Gen 3',
                    'ram': '12GB',
                    'storage_options': ['256GB', '512GB', '1TB'],
                    'main_camera': '200MP f/1.7',
                    'front_camera': '12MP f/2.2',
                    'battery': '5000mAh',
                    'os': 'Android 14 with One UI 6.1',
                    'connectivity': ['5G', 'Wi-Fi 7', 'Bluetooth 5.3']
                },
                'similar_phones': ['iPhone 15 Pro Max', 'Galaxy S23 Ultra', 'Google Pixel 8 Pro'],
                'target_users': ['Business professionals', 'Content creators', 'Power users']
            },
            
            'galaxy_s23_ultra': {
                'model': 'Samsung Galaxy S23 Ultra',
                'brand': 'Samsung',
                'variants': ['Galaxy S23 Ultra'],
                'launch_year': 2023,
                'rating': 4.3,
                'price_range': {'min': 1099, 'max': 1379, 'currency': 'USD'},
                'key_features': [
                    'S Pen', '200MP camera', 'Snapdragon 8 Gen 2',
                    'Large AMOLED display', '100x Space Zoom'
                ],
                'pros': [
                    'Excellent camera system', 'S Pen productivity',
                    'Great performance', 'Good battery life'
                ],
                'cons': [
                    'Expensive', 'Heavy phone', 'UI complexity'
                ],
                'specifications': {
                    'display': '6.8-inch Dynamic AMOLED 2X',
                    'resolution': '3088×1440 pixels',
                    'processor': 'Snapdragon 8 Gen 2',
                    'ram': '12GB',
                    'storage_options': ['256GB', '512GB', '1TB'],
                    'main_camera': '200MP f/1.7',
                    'front_camera': '12MP f/2.2',
                    'battery': '5000mAh',
                    'os': 'Android 13 with One UI 5.1'
                },
                'similar_phones': ['Galaxy S24 Ultra', 'iPhone 14 Pro Max', 'Google Pixel 7 Pro'],
                'target_users': ['Professionals', 'Note series users', 'Camera enthusiasts']
            },
            
            # Google Pixel
            'pixel_8_pro': {
                'model': 'Google Pixel 8 Pro',
                'brand': 'Google',
                'variants': ['Pixel 8 Pro'],
                'launch_year': 2023,
                'rating': 4.3,
                'price_range': {'min': 999, 'max': 1099, 'currency': 'USD'},
                'key_features': [
                    'Google Tensor G3', 'AI photography features', 'Pure Android',
                    'Magic Eraser', 'Call Screen', 'Fast Android updates'
                ],
                'pros': [
                    'Excellent computational photography', 'Clean Android experience',
                    'Fast and consistent updates', 'Great AI features',
                    'Good value for flagship'
                ],
                'cons': [
                    'Limited availability', 'Battery life inconsistent',
                    'Tensor chip efficiency', 'No headphone jack'
                ],
                'specifications': {
                    'display': '6.7-inch LTPO OLED',
                    'resolution': '2992×1344 pixels',
                    'processor': 'Google Tensor G3',
                    'ram': '12GB',
                    'storage_options': ['128GB', '256GB', '512GB'],
                    'main_camera': '50MP f/1.68',
                    'front_camera': '10.5MP f/2.2',
                    'battery': '5050mAh',
                    'os': 'Android 14'
                },
                'similar_phones': ['Pixel 7 Pro', 'iPhone 15 Pro', 'OnePlus 12'],
                'target_users': ['Photography enthusiasts', 'Android purists', 'Google services users']
            },
            
            # OnePlus
            'oneplus_12': {
                'model': 'OnePlus 12',
                'brand': 'OnePlus',
                'variants': ['OnePlus 12'],
                'launch_year': 2024,
                'rating': 4.2,
                'price_range': {'min': 799, 'max': 899, 'currency': 'USD'},
                'key_features': [
                    'Snapdragon 8 Gen 3', 'Hasselblad cameras', 'Fast charging',
                    '100W SuperVOOC', 'Alert Slider', 'OxygenOS'
                ],
                'pros': [
                    'Excellent performance', 'Fast charging', 'Good value',
                    'Clean software experience', 'Premium design'
                ],
                'cons': [
                    'Camera inconsistency', 'Limited availability',
                    'No wireless charging', 'Software update concerns'
                ],
                'specifications': {
                    'display': '6.82-inch AMOLED',
                    'resolution': '3168×1440 pixels',
                    'processor': 'Snapdragon 8 Gen 3',
                    'ram': '16GB',
                    'storage_options': ['256GB', '512GB'],
                    'main_camera': '50MP f/1.6',
                    'front_camera': '32MP f/2.4',
                    'battery': '5400mAh',
                    'os': 'OxygenOS 14 (Android 14)'
                },
                'similar_phones': ['OnePlus 11', 'Galaxy S24', 'Pixel 8 Pro'],
                'target_users': ['Performance enthusiasts', 'Fast charging users', 'Value seekers']
            }
        }
        
        return phones_db
    
    def _load_specification_templates(self) -> Dict[str, Any]:
        """Load specification templates for synthetic data generation"""
        
        return {
            'display_sizes': {
                'compact': ['5.4"', '5.8"', '6.0"'],
                'standard': ['6.1"', '6.2"', '6.3"'],
                'large': ['6.5"', '6.7"', '6.8"']
            },
            'processors': {
                'apple': ['A17 Pro', 'A16 Bionic', 'A15 Bionic'],
                'snapdragon': ['Snapdragon 8 Gen 3', 'Snapdragon 8 Gen 2', 'Snapdragon 7+ Gen 3'],
                'mediatek': ['Dimensity 9300', 'Dimensity 8200', 'Dimensity 7200'],
                'google': ['Google Tensor G3', 'Google Tensor G2']
            },
            'ram_options': ['6GB', '8GB', '12GB', '16GB'],
            'storage_options': ['128GB', '256GB', '512GB', '1TB'],
            'camera_configs': {
                'basic': '12MP main',
                'standard': '48MP main + 12MP ultrawide',
                'advanced': '108MP main + 12MP ultrawide + 12MP telephoto',
                'pro': '200MP main + 50MP ultrawide + 12MP telephoto + ToF'
            }
        }
    
    def _compile_brand_patterns(self) -> Dict[str, List[str]]:
        """Compile brand recognition patterns"""
        
        return {
            'Apple': ['iphone', 'apple', 'ios'],
            'Samsung': ['galaxy', 'samsung', 'note'],
            'Google': ['pixel', 'google', 'nexus'],
            'OnePlus': ['oneplus', 'one plus', 'never settle'],
            'Xiaomi': ['xiaomi', 'mi', 'redmi', 'poco'],
            'Huawei': ['huawei', 'mate', 'p series'],
            'Oppo': ['oppo', 'find', 'reno'],
            'Vivo': ['vivo', 'iqoo'],
            'Nothing': ['nothing', 'nothing phone'],
            'Motorola': ['motorola', 'moto', 'edge'],
            'Sony': ['sony', 'xperia']
        }
    
    def _clean_query(self, query: str) -> str:
        """Clean and normalize search query"""
        
        # Convert to lowercase
        cleaned = query.lower().strip()
        
        # Remove common noise words
        noise_words = ['phone', 'smartphone', 'mobile', 'review', 'specs', 'price']
        for noise in noise_words:
            cleaned = cleaned.replace(noise, ' ')
        
        # Normalize spacing
        cleaned = ' '.join(cleaned.split())
        
        return cleaned
    
    def _parse_phone_query(self, query: str) -> Dict[str, Any]:
        """Parse phone query to extract brand, model, and other info"""
        
        parsed = {
            'brand': None,
            'model': query,
            'year': None,
            'variant': None,
            'original_query': query
        }
        
        # Extract brand
        for brand, patterns in self.brand_patterns.items():
            for pattern in patterns:
                if pattern in query:
                    parsed['brand'] = brand
                    break
            if parsed['brand']:
                break
        
        # Extract year if present
        year_match = re.search(r'\b(20\d{2})\b', query)
        if year_match:
            parsed['year'] = int(year_match.group(1))
        
        # Extract variant information
        if 'pro' in query:
            parsed['variant'] = 'pro'
        elif 'plus' in query or 'max' in query:
            parsed['variant'] = 'plus'
        elif 'ultra' in query:
            parsed['variant'] = 'ultra'
        elif 'mini' in query:
            parsed['variant'] = 'mini'
        
        return parsed
    
    def _find_exact_match(self, parsed_info: Dict[str, Any]) -> Optional[FallbackSearchResult]:
        """Find exact match in offline database"""
        
        query_key = self._generate_search_key(parsed_info['model'])
        
        for phone_key, phone_data in self.offline_db.items():
            if query_key == phone_key or query_key in phone_data.get('variants', []):
                return FallbackSearchResult(
                    phone_model=phone_data['model'],
                    source='offline_exact_match',
                    confidence=0.95,
                    data=phone_data,
                    metadata={'match_type': 'exact', 'search_key': query_key}
                )
        
        return None
    
    def _find_fuzzy_match(self, parsed_info: Dict[str, Any]) -> Optional[FallbackSearchResult]:
        """Find fuzzy match using string similarity"""
        
        query = parsed_info['model']
        best_match = None
        best_confidence = 0.0
        
        for phone_key, phone_data in self.offline_db.items():
            # Check against model name
            model_similarity = difflib.SequenceMatcher(None, query, phone_data['model'].lower()).ratio()
            
            # Check against variants
            max_variant_similarity = 0.0
            for variant in phone_data.get('variants', []):
                variant_similarity = difflib.SequenceMatcher(None, query, variant.lower()).ratio()
                max_variant_similarity = max(max_variant_similarity, variant_similarity)
            
            # Use best similarity score
            confidence = max(model_similarity, max_variant_similarity)
            
            # Boost confidence if brand matches
            if parsed_info['brand'] and parsed_info['brand'] == phone_data.get('brand'):
                confidence += 0.1
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_match = FallbackSearchResult(
                    phone_model=phone_data['model'],
                    source='offline_fuzzy_match',
                    confidence=confidence,
                    data=phone_data,
                    metadata={'match_type': 'fuzzy', 'similarity_score': confidence}
                )
        
        return best_match
    
    def _search_alternative_sources(self, parsed_info: Dict[str, Any]) -> Optional[FallbackSearchResult]:
        """Search alternative real data sources when primary search fails"""
        
        phone_model = parsed_info.get('model') or ''
        brand = parsed_info.get('brand') or ''
        
        # Search in cached/secondary databases (if available)
        cached_result = self._search_cached_data(phone_model, brand)
        if cached_result:
            return cached_result
        
        # Search for partial matches with higher tolerance
        partial_matches = self._find_partial_matches(parsed_info, min_confidence=0.4)
        if partial_matches:
            # Return the best partial match as an alternative
            return partial_matches[0]
        
        return None
    
    def _find_brand_alternatives(self, parsed_info: Dict[str, Any]) -> Optional[FallbackSearchResult]:
        """Find alternative real phones from the same brand"""
        
        brand = (parsed_info.get('brand') or '').lower()
        if not brand:
            return None
        
        # Find all phones from the same brand in our database
        brand_phones = []
        for phone_key, phone_data in self.offline_db.items():
            if phone_data.get('brand', '').lower() == brand:
                brand_phones.append((phone_key, phone_data))
        
        if not brand_phones:
            return None
        
        # Sort by rating or popularity (use rating as proxy)
        brand_phones.sort(key=lambda x: x[1].get('rating', 0), reverse=True)
        
        # Return the highest-rated phone from the same brand as an alternative
        best_phone = brand_phones[0]
        phone_data = best_phone[1]
        
        return FallbackSearchResult(
            phone_model=phone_data['model'],
            source='brand_alternative',
            confidence=0.7,  # Medium confidence since it's an alternative
            data=phone_data,
            metadata={
                'search_type': 'brand_alternative',
                'original_query': parsed_info.get('model', ''),
                'alternative_reason': f'Found popular {brand} phone as alternative'
            }
        )
    
    def _search_cached_data(self, phone_model: str, brand: str) -> Optional[FallbackSearchResult]:
        """Search for phone data in cached/secondary sources"""
        
        # In a full implementation, this would search:
        # - Recently cached API results
        # - Secondary database files
        # - User-contributed phone data
        # - Imported data from other sources
        
        # For now, return None - can be extended to include actual cache search
        return None
    
    def _find_partial_matches(self, parsed_info: Dict[str, Any], min_confidence: float = 0.4) -> List[FallbackSearchResult]:
        """Find partial matches with lower confidence threshold for alternatives"""
        
        phone_model = (parsed_info.get('model') or '').lower()
        brand = (parsed_info.get('brand') or '').lower()
        
        partial_matches = []
        
        for phone_key, phone_data in self.offline_db.items():
            # Calculate similarity for partial matching
            db_model = phone_data.get('model', '').lower()
            db_brand = phone_data.get('brand', '').lower()
            
            # Check for partial model name matches
            model_similarity = 0
            if phone_model and db_model:
                # Check if any significant words match
                query_words = set(phone_model.split())
                db_words = set(db_model.split())
                common_words = query_words.intersection(db_words)
                if common_words:
                    model_similarity = len(common_words) / max(len(query_words), len(db_words))
            
            # Brand match bonus
            brand_match = (brand and db_brand and brand in db_brand) or (db_brand and brand and db_brand in brand)
            
            # Calculate total confidence
            confidence = model_similarity
            if brand_match:
                confidence += 0.3  # Boost for brand match
            
            # Only include if meets minimum confidence
            if confidence >= min_confidence:
                result = FallbackSearchResult(
                    phone_model=phone_data['model'],
                    source='partial_match',
                    confidence=confidence,
                    data=phone_data,
                    metadata={
                        'match_type': 'partial',
                        'model_similarity': model_similarity,
                        'brand_match': brand_match,
                        'original_query': parsed_info.get('model', '')
                    }
                )
                partial_matches.append(result)
        
        # Sort by confidence, return best matches
        partial_matches.sort(key=lambda x: x.confidence, reverse=True)
        return partial_matches[:3]  # Return top 3 partial matches
    
    def _estimate_price_range_from_data(self, phone_data: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate price range from existing phone data (not synthetic generation)"""
        
        # Extract price information if available
        price_info = {
            'min': 0,
            'max': 1000,
            'currency': 'USD',
            'estimated': True
        }
        
        # Try to get actual price data
        if 'price' in phone_data:
            try:
                price = float(phone_data['price'])
                price_info = {
                    'min': price * 0.8,  # 20% range
                    'max': price * 1.2,
                    'currency': 'USD',
                    'estimated': False
                }
            except (ValueError, TypeError):
                pass
        
        return price_info
    
    def _generate_search_key(self, query: str) -> str:
        """Generate standardized search key"""
        
        # Convert to lowercase and replace spaces with underscores
        key = query.lower().replace(' ', '_')
        
        # Remove special characters
        key = re.sub(r'[^a-z0-9_]', '', key)
        
        return key
    
    def save_offline_database(self, filepath: str = None):
        """Save the current offline database to file"""
        
        if not filepath:
            filepath = self.config['offline_database_path']
        
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.offline_db, f, indent=2, ensure_ascii=False)
            logger.info(f"Offline database saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save offline database: {e}")


def create_fallback_search_system(config: Dict[str, Any] = None) -> FallbackSearchSystem:
    """
    Factory function to create a fallback search system
    """
    
    return FallbackSearchSystem(config=config)