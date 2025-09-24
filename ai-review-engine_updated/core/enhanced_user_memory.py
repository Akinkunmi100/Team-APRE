"""
Enhanced User Memory and Profiling System
Tracks user behavior, preferences, and provides intelligent recommendations
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict, Counter
import logging
import hashlib

logger = logging.getLogger(__name__)

class EnhancedUserMemory:
    """Advanced user memory system for tracking behavior and preferences"""
    
    def __init__(self, storage_path: str = "user_profiles", user_id: str = None):
        """Initialize the user memory system with optional user ID"""
        self.storage_path = storage_path
        self.ensure_storage_directory()
        
        # Use provided user_id or generate new one
        session_user_id = user_id if user_id else self._generate_session_id()
        
        # In-memory cache for current session
        self.current_session = {
            'user_id': session_user_id,
            'session_start': datetime.now(),
            'searches': [],
            'preferences': {},
            'behavioral_signals': defaultdict(list)
        }
        
        # Load existing user profile if available
        self.user_profile = self._load_or_create_profile()
        
    def ensure_storage_directory(self):
        """Create storage directory if it doesn't exist"""
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)
            
    def _generate_session_id(self) -> str:
        """Generate a unique session ID based on browser fingerprint"""
        # Use browser session info + timestamp to create unique ID
        # This will be overridden by Streamlit integration
        timestamp = datetime.now().isoformat()
        import random
        random_part = str(random.randint(100000, 999999))
        return hashlib.md5(f"{timestamp}_{random_part}".encode()).hexdigest()[:12]
        
    def _get_profile_path(self) -> str:
        """Get the path for the user profile file"""
        return os.path.join(self.storage_path, f"profile_{self.current_session['user_id']}.json")
        
    def _load_or_create_profile(self) -> Dict:
        """Load existing profile or create new one"""
        profile_path = self._get_profile_path()
        
        default_profile = {
            'user_id': self.current_session['user_id'],
            'created_at': datetime.now().isoformat(),
            'last_active': datetime.now().isoformat(),
            'total_sessions': 1,
            'search_history': [],
            'preferences': {
                'preferred_brands': [],
                'price_ranges_searched': [],
                'feature_interests': [],
                'search_patterns': []
            },
            'behavioral_patterns': {
                'avg_searches_per_session': 0.0,
                'total_searches': 0,
                'successful_searches': 0,
                'brand_affinity': {},
                'price_sensitivity': 'unknown',
                'feature_priority': {}
            },
            'smart_insights': {
                'likely_budget': None,
                'preferred_category': None,
                'purchase_intent': 'unknown',
                'expertise_level': 'beginner'
            }
        }
        
        if os.path.exists(profile_path):
            try:
                with open(profile_path, 'r') as f:
                    profile = json.load(f)
                    # Update last active
                    profile['last_active'] = datetime.now().isoformat()
                    profile['total_sessions'] = profile.get('total_sessions', 0) + 1
                    return profile
            except Exception as e:
                logger.warning(f"Error loading profile: {e}")
                
        return default_profile
        
    def track_search(self, query: str, result: Dict, user_interaction: Dict = None):
        """Track a user search with comprehensive analytics"""
        search_data = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'phone_model': result.get('phone_model', ''),
            'success': result.get('success', False),
            'confidence': result.get('confidence', 0.0),
            'sources_used': result.get('search_sources', []),
            'found_in_database': result.get('found_in_database', False),
            'found_in_web': result.get('found_in_web', False),
            'ai_recommendations': result.get('has_ai_recommendations', False)
        }
        
        # Add to current session
        self.current_session['searches'].append(search_data)
        
        # Add to persistent profile
        self.user_profile['search_history'].append(search_data)
        
        # Keep only last 100 searches to avoid huge files
        if len(self.user_profile['search_history']) > 100:
            self.user_profile['search_history'] = self.user_profile['search_history'][-100:]
            
        # Update behavioral patterns
        self._update_behavioral_patterns(search_data)
        
        # Learn preferences from search
        self._learn_preferences_from_search(query, result)
        
        # Update smart insights
        self._update_smart_insights()
        
        # Save profile
        self._save_profile()
        
    def _update_behavioral_patterns(self, search_data: Dict):
        """Update behavioral patterns based on search"""
        patterns = self.user_profile['behavioral_patterns']
        
        # Update search counts
        patterns['total_searches'] = patterns.get('total_searches', 0) + 1
        if search_data['success']:
            patterns['successful_searches'] = patterns.get('successful_searches', 0) + 1
            
        # Update success rate
        if patterns['total_searches'] > 0:
            patterns['success_rate'] = patterns['successful_searches'] / patterns['total_searches']
            
        # Update average searches per session
        total_sessions = self.user_profile.get('total_sessions', 1)
        patterns['avg_searches_per_session'] = patterns['total_searches'] / total_sessions
        
        # Track brand affinity
        phone_model = search_data.get('phone_model', '').lower()
        if phone_model:
            brand = self._extract_brand(phone_model)
            if brand:
                if 'brand_affinity' not in patterns:
                    patterns['brand_affinity'] = {}
                patterns['brand_affinity'][brand] = patterns['brand_affinity'].get(brand, 0) + 1
                
    def _learn_preferences_from_search(self, query: str, result: Dict):
        """Learn user preferences from search behavior"""
        preferences = self.user_profile['preferences']
        
        # Extract brand preference
        phone_model = result.get('phone_model', '').lower()
        if phone_model:
            brand = self._extract_brand(phone_model)
            if brand and brand not in preferences['preferred_brands']:
                preferences['preferred_brands'].append(brand)
                
        # Detect feature interests from query
        feature_keywords = {
            'camera': ['camera', 'photo', 'photography', 'selfie'],
            'battery': ['battery', 'power', 'charge', 'lasting'],
            'gaming': ['gaming', 'game', 'performance', 'processor'],
            'display': ['display', 'screen', 'oled', 'amoled'],
            'price': ['cheap', 'budget', 'affordable', 'expensive', 'premium'],
            'storage': ['storage', 'memory', 'gb', 'tb'],
            'design': ['design', 'look', 'beautiful', 'slim', 'light']
        }
        
        query_lower = query.lower()
        for feature, keywords in feature_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                if feature not in preferences['feature_interests']:
                    preferences['feature_interests'].append(feature)
                    
        # Extract price range from query
        price_indicators = self._extract_price_indicators(query)
        if price_indicators:
            preferences['price_ranges_searched'].extend(price_indicators)
            
    def _extract_brand(self, phone_model: str) -> Optional[str]:
        """Extract brand from phone model"""
        brands = {
            'iphone': 'Apple',
            'apple': 'Apple',
            'samsung': 'Samsung',
            'galaxy': 'Samsung',
            'pixel': 'Google',
            'google': 'Google',
            'oneplus': 'OnePlus',
            'xiaomi': 'Xiaomi',
            'huawei': 'Huawei',
            'oppo': 'Oppo',
            'vivo': 'Vivo',
            'nothing': 'Nothing',
            'motorola': 'Motorola'
        }
        
        phone_lower = phone_model.lower()
        for key, brand in brands.items():
            if key in phone_lower:
                return brand
                
        return None
        
    def _extract_price_indicators(self, query: str) -> List[str]:
        """Extract price range indicators from query"""
        price_ranges = []
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['budget', 'cheap', 'affordable']):
            price_ranges.append('budget')
        elif any(word in query_lower for word in ['premium', 'expensive', 'flagship']):
            price_ranges.append('premium')
        elif 'mid' in query_lower or 'middle' in query_lower:
            price_ranges.append('mid-range')
            
        # Extract specific price mentions (e.g., "under $500")
        import re
        price_matches = re.findall(r'\$?(\d+)', query)
        if price_matches:
            price_ranges.extend([f"${price}" for price in price_matches])
            
        return price_ranges
        
    def _update_smart_insights(self):
        """Update smart insights based on accumulated data"""
        insights = self.user_profile['smart_insights']
        patterns = self.user_profile['behavioral_patterns']
        preferences = self.user_profile['preferences']
        
        # Determine likely budget
        price_searches = preferences.get('price_ranges_searched', [])
        if price_searches:
            budget_counts = Counter(price_searches)
            most_common = budget_counts.most_common(1)
            if most_common:
                insights['likely_budget'] = most_common[0][0]
                
        # Determine preferred category
        feature_interests = preferences.get('feature_interests', [])
        if feature_interests:
            feature_counts = Counter(feature_interests)
            most_common = feature_counts.most_common(1)
            if most_common:
                insights['preferred_category'] = most_common[0][0]
                
        # Assess purchase intent
        total_searches = patterns.get('total_searches', 0)
        if total_searches > 5:
            insights['purchase_intent'] = 'high'
        elif total_searches > 2:
            insights['purchase_intent'] = 'medium'
        else:
            insights['purchase_intent'] = 'low'
            
        # Assess expertise level
        unique_brands = len(set(preferences.get('preferred_brands', [])))
        if unique_brands > 3 and total_searches > 10:
            insights['expertise_level'] = 'expert'
        elif unique_brands > 1 and total_searches > 5:
            insights['expertise_level'] = 'intermediate'
        else:
            insights['expertise_level'] = 'beginner'
            
    def get_personalized_recommendations(self, current_search: str = None) -> Dict:
        """Generate personalized recommendations based on user history"""
        insights = self.user_profile['smart_insights']
        preferences = self.user_profile['preferences']
        patterns = self.user_profile['behavioral_patterns']
        
        recommendations = {
            'personalized_phones': [],
            'suggested_searches': [],
            'insights': {},
            'tips': []
        }
        
        # Brand-based recommendations
        brand_affinity = patterns.get('brand_affinity', {})
        if brand_affinity:
            top_brands = sorted(brand_affinity.items(), key=lambda x: x[1], reverse=True)[:2]
            for brand, count in top_brands:
                recommendations['suggested_searches'].append(f"Latest {brand} phones")
                
        # Feature-based recommendations
        feature_interests = preferences.get('feature_interests', [])
        if feature_interests:
            top_feature = Counter(feature_interests).most_common(1)
            if top_feature:
                feature = top_feature[0][0]
                recommendations['suggested_searches'].append(f"Best {feature} phones 2024")
                recommendations['tips'].append(f"You seem interested in {feature} - consider phones with top-rated {feature} features")
                
        # Budget-based recommendations
        if insights.get('likely_budget'):
            budget = insights['likely_budget']
            recommendations['suggested_searches'].append(f"Best phones {budget}")
            
        # Expertise-based tips
        expertise = insights.get('expertise_level', 'beginner')
        if expertise == 'beginner':
            recommendations['tips'].append("💡 Try searching for specific brands like 'iPhone' or 'Samsung' to get started")
        elif expertise == 'intermediate':
            recommendations['tips'].append("💡 You might want to compare phones with similar features")
        else:
            recommendations['tips'].append("💡 As an expert, you might be interested in detailed spec comparisons")
            
        # Purchase intent insights
        intent = insights.get('purchase_intent', 'unknown')
        if intent == 'high':
            recommendations['insights']['purchase_readiness'] = "You seem ready to make a purchase decision!"
            recommendations['tips'].append("🛒 Consider checking recent reviews and price comparisons")
        elif intent == 'medium':
            recommendations['insights']['purchase_readiness'] = "You're actively researching - great approach!"
        
        return recommendations
        
    def get_user_profile_summary(self) -> Dict:
        """Get a summary of the user profile for display"""
        patterns = self.user_profile['behavioral_patterns']
        insights = self.user_profile['smart_insights']
        preferences = self.user_profile['preferences']
        
        # Get top brands
        brand_affinity = patterns.get('brand_affinity', {})
        top_brands = sorted(brand_affinity.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Get top features
        feature_interests = preferences.get('feature_interests', [])
        top_features = Counter(feature_interests).most_common(3)
        
        return {
            'total_searches': patterns.get('total_searches', 0),
            'success_rate': f"{patterns.get('success_rate', 0) * 100:.1f}%",
            'expertise_level': insights.get('expertise_level', 'beginner').title(),
            'purchase_intent': insights.get('purchase_intent', 'unknown').title(),
            'top_brands': [brand for brand, count in top_brands],
            'top_features': [feature for feature, count in top_features],
            'likely_budget': insights.get('likely_budget', 'Unknown'),
            'preferred_category': insights.get('preferred_category', 'General')
        }
        
    def _save_profile(self):
        """Save user profile to disk"""
        try:
            profile_path = self._get_profile_path()
            with open(profile_path, 'w') as f:
                json.dump(self.user_profile, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving user profile: {e}")
            
    def get_session_stats(self) -> Dict:
        """Get current session statistics"""
        searches = self.current_session['searches']
        
        return {
            'session_id': self.current_session['user_id'],
            'session_duration': (datetime.now() - self.current_session['session_start']).total_seconds() / 60,
            'searches_this_session': len(searches),
            'successful_searches': sum(1 for s in searches if s['success']),
            'unique_phones_searched': len(set(s['phone_model'] for s in searches if s['phone_model']))
        }
        
    def clear_user_data(self):
        """Clear all user data (for privacy)"""
        try:
            profile_path = self._get_profile_path()
            if os.path.exists(profile_path):
                os.remove(profile_path)
            
            # Reset current session
            self.current_session = {
                'user_id': self._generate_session_id(),
                'session_start': datetime.now(),
                'searches': [],
                'preferences': {},
                'behavioral_signals': defaultdict(list)
            }
            
            # Create new profile
            self.user_profile = self._load_or_create_profile()
            
            return True
        except Exception as e:
            logger.error(f"Error clearing user data: {e}")
            return False