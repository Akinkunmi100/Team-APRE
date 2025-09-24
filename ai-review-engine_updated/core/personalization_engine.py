"""
Advanced Personalization Engine
Optional module for user profiling, preference learning, and personalized experiences
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import numpy as np
from collections import defaultdict, Counter
import hashlib
import pickle
from abc import ABC, abstractmethod

# Machine Learning imports
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available. Basic personalization only.")

# Deep Learning imports (optional)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available. Deep learning features disabled.")

# Database imports
try:
    from sqlalchemy import create_engine, Column, String, Float, DateTime, JSON, Integer, Boolean
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker
    SQLALCHEMY_AVAILABLE = True
    Base = declarative_base()
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    Base = None
    logging.warning("SQLAlchemy not available. Using in-memory storage.")

logger = logging.getLogger(__name__)


class InteractionType(Enum):
    """Types of user interactions"""
    VIEW = "view"
    CLICK = "click"
    SEARCH = "search"
    COMPARE = "compare"
    REVIEW = "review"
    PURCHASE = "purchase"
    BOOKMARK = "bookmark"
    SHARE = "share"
    LIKE = "like"
    DISLIKE = "dislike"
    DWELL_TIME = "dwell_time"
    SCROLL = "scroll"


class PreferenceCategory(Enum):
    """Categories of user preferences"""
    BRAND = "brand"
    PRICE_RANGE = "price_range"
    FEATURES = "features"
    DESIGN = "design"
    PERFORMANCE = "performance"
    CAMERA = "camera"
    BATTERY = "battery"
    DISPLAY = "display"
    STORAGE = "storage"
    USE_CASE = "use_case"


@dataclass
class UserInteraction:
    """Represents a user interaction"""
    user_id: str
    interaction_type: InteractionType
    item_id: str
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)
    value: float = 1.0  # Interaction strength/importance
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserProfile:
    """Comprehensive user profile"""
    user_id: str
    created_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)
    
    # Preference vectors
    brand_preferences: Dict[str, float] = field(default_factory=dict)
    feature_preferences: Dict[str, float] = field(default_factory=dict)
    price_preferences: Dict[str, float] = field(default_factory=dict)
    
    # Behavioral patterns
    interaction_history: List[UserInteraction] = field(default_factory=list)
    search_history: List[str] = field(default_factory=list)
    view_history: List[str] = field(default_factory=list)
    
    # Derived insights
    user_segment: Optional[str] = None
    predicted_budget: Optional[Tuple[float, float]] = None
    interests: List[str] = field(default_factory=list)
    
    # Personalization settings
    explicit_preferences: Dict[str, Any] = field(default_factory=dict)
    privacy_settings: Dict[str, bool] = field(default_factory=dict)
    
    # Statistics
    total_interactions: int = 0
    avg_session_duration: float = 0.0
    conversion_rate: float = 0.0
    
    # Embeddings
    embedding: Optional[np.ndarray] = None
    cluster_id: Optional[int] = None


@dataclass
class PersonalizationContext:
    """Context for personalization decisions"""
    user_profile: UserProfile
    current_session: Dict[str, Any]
    device_info: Optional[Dict[str, Any]] = None
    location: Optional[str] = None
    time_of_day: Optional[str] = None
    referrer: Optional[str] = None
    active_filters: Dict[str, Any] = field(default_factory=dict)


class UserProfileStore:
    """Storage and retrieval of user profiles"""
    
    def __init__(self, storage_type: str = "memory"):
        self.storage_type = storage_type
        
        if storage_type == "memory":
            self.profiles = {}
        elif storage_type == "database" and SQLALCHEMY_AVAILABLE:
            self.engine = create_engine('sqlite:///user_profiles.db')
            Base.metadata.create_all(self.engine)
            self.Session = sessionmaker(bind=self.engine)
        else:
            self.profiles = {}  # Fallback to memory
            
    def save_profile(self, profile: UserProfile):
        """Save user profile"""
        if self.storage_type == "memory":
            self.profiles[profile.user_id] = profile
        elif self.storage_type == "database" and SQLALCHEMY_AVAILABLE:
            # Serialize and save to database
            session = self.Session()
            # Implementation would go here
            session.close()
    
    def load_profile(self, user_id: str) -> Optional[UserProfile]:
        """Load user profile"""
        if self.storage_type == "memory":
            return self.profiles.get(user_id)
        elif self.storage_type == "database" and SQLALCHEMY_AVAILABLE:
            # Load from database
            session = self.Session()
            # Implementation would go here
            session.close()
        return None
    
    def get_or_create_profile(self, user_id: str) -> UserProfile:
        """Get existing profile or create new one"""
        profile = self.load_profile(user_id)
        if profile is None:
            profile = UserProfile(user_id=user_id)
            self.save_profile(profile)
        return profile


class PreferenceLearner:
    """Learns user preferences from interactions"""
    
    def __init__(self):
        self.interaction_weights = {
            InteractionType.PURCHASE: 10.0,
            InteractionType.BOOKMARK: 5.0,
            InteractionType.REVIEW: 8.0,
            InteractionType.COMPARE: 3.0,
            InteractionType.CLICK: 2.0,
            InteractionType.VIEW: 1.0,
            InteractionType.SHARE: 6.0,
            InteractionType.LIKE: 4.0,
            InteractionType.DISLIKE: -3.0
        }
        
        # Initialize models if available
        if SKLEARN_AVAILABLE:
            self.brand_predictor = RandomForestClassifier(n_estimators=100)
            self.price_predictor = GradientBoostingRegressor(n_estimators=100)
            self.feature_extractor = PCA(n_components=50)
    
    def learn_from_interaction(
        self,
        profile: UserProfile,
        interaction: UserInteraction
    ) -> UserProfile:
        """Update user profile based on new interaction"""
        
        # Add to interaction history
        profile.interaction_history.append(interaction)
        profile.total_interactions += 1
        profile.last_active = datetime.now()
        
        # Update preferences based on interaction type
        weight = self.interaction_weights.get(interaction.interaction_type, 1.0)
        
        # Extract item attributes from context
        item_data = interaction.context.get('item_data', {})
        
        # Update brand preferences
        if 'brand' in item_data:
            brand = item_data['brand']
            profile.brand_preferences[brand] = profile.brand_preferences.get(brand, 0) + weight
        
        # Update feature preferences
        if 'features' in item_data:
            for feature in item_data['features']:
                profile.feature_preferences[feature] = profile.feature_preferences.get(feature, 0) + weight
        
        # Update price preferences
        if 'price' in item_data:
            price = item_data['price']
            price_range = self._get_price_range(price)
            profile.price_preferences[price_range] = profile.price_preferences.get(price_range, 0) + weight
        
        # Update search history
        if interaction.interaction_type == InteractionType.SEARCH:
            query = interaction.context.get('query', '')
            if query:
                profile.search_history.append(query)
        
        # Update view history
        if interaction.interaction_type in [InteractionType.VIEW, InteractionType.CLICK]:
            profile.view_history.append(interaction.item_id)
        
        # Recalculate derived insights
        self._update_insights(profile)
        
        return profile
    
    def _get_price_range(self, price: float) -> str:
        """Categorize price into ranges"""
        if price < 300:
            return "budget"
        elif price < 600:
            return "mid-range"
        elif price < 1000:
            return "premium"
        else:
            return "flagship"
    
    def _update_insights(self, profile: UserProfile):
        """Update derived insights from interaction history"""
        
        # Identify interests from feature preferences
        top_features = sorted(
            profile.feature_preferences.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        profile.interests = [f[0] for f in top_features]
        
        # Predict budget range
        if profile.price_preferences:
            weighted_prices = []
            for range_name, weight in profile.price_preferences.items():
                if range_name == "budget":
                    weighted_prices.extend([200] * int(weight))
                elif range_name == "mid-range":
                    weighted_prices.extend([450] * int(weight))
                elif range_name == "premium":
                    weighted_prices.extend([800] * int(weight))
                else:  # flagship
                    weighted_prices.extend([1200] * int(weight))
            
            if weighted_prices:
                avg_price = np.mean(weighted_prices)
                std_price = np.std(weighted_prices)
                profile.predicted_budget = (
                    max(0, avg_price - std_price),
                    avg_price + std_price
                )
        
        # Calculate conversion rate
        purchases = sum(1 for i in profile.interaction_history
                        if i.interaction_type == InteractionType.PURCHASE)
        views = sum(1 for i in profile.interaction_history
                    if i.interaction_type == InteractionType.VIEW)
        if views > 0:
            profile.conversion_rate = purchases / views
    
    def extract_features(self, profile: UserProfile) -> np.ndarray:
        """Extract feature vector from user profile"""
        features = []
        
        # Brand preference features
        top_brands = ['Apple', 'Samsung', 'Google', 'OnePlus', 'Xiaomi']
        for brand in top_brands:
            features.append(profile.brand_preferences.get(brand, 0))
        
        # Feature preference features
        top_features = ['camera', 'battery', 'performance', 'display', 'price']
        for feature in top_features:
            features.append(profile.feature_preferences.get(feature, 0))
        
        # Price preference features
        price_ranges = ['budget', 'mid-range', 'premium', 'flagship']
        for range_name in price_ranges:
            features.append(profile.price_preferences.get(range_name, 0))
        
        # Behavioral features
        features.append(profile.total_interactions)
        features.append(profile.conversion_rate)
        features.append(len(profile.search_history))
        features.append(len(profile.view_history))
        
        # Time-based features
        days_since_created = (datetime.now() - profile.created_at).days
        days_since_active = (datetime.now() - profile.last_active).days
        features.append(days_since_created)
        features.append(days_since_active)
        
        return np.array(features)


class UserSegmentation:
    """Segment users into meaningful groups"""
    
    def __init__(self, n_segments: int = 5):
        self.n_segments = n_segments
        self.segments = {
            'tech_enthusiast': {
                'description': 'Early adopters who love latest technology',
                'traits': ['high_engagement', 'flagship_preference', 'feature_focused']
            },
            'value_seeker': {
                'description': 'Price-conscious users seeking best value',
                'traits': ['price_sensitive', 'comparison_heavy', 'mid_range_preference']
            },
            'brand_loyal': {
                'description': 'Stick to preferred brands',
                'traits': ['single_brand_preference', 'premium_willing', 'low_comparison']
            },
            'casual_user': {
                'description': 'Basic needs, occasional upgrades',
                'traits': ['low_engagement', 'budget_preference', 'simple_features']
            },
            'professional': {
                'description': 'Business users with specific requirements',
                'traits': ['productivity_focused', 'reliability_important', 'moderate_price']
            }
        }
        
        if SKLEARN_AVAILABLE:
            self.kmeans = KMeans(n_clusters=n_segments, random_state=42)
            self.scaler = StandardScaler()
    
    def segment_users(self, profiles: List[UserProfile]) -> Dict[str, List[str]]:
        """Segment users based on their profiles"""
        if not profiles or not SKLEARN_AVAILABLE:
            return {}
        
        # Extract features for all users
        learner = PreferenceLearner()
        features = np.array([learner.extract_features(p) for p in profiles])
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Perform clustering
        cluster_labels = self.kmeans.fit_predict(features_scaled)
        
        # Assign segments
        segments = defaultdict(list)
        for profile, label in zip(profiles, cluster_labels):
            profile.cluster_id = label
            segment = self._identify_segment(profile)
            profile.user_segment = segment
            segments[segment].append(profile.user_id)
        
        return dict(segments)
    
    def _identify_segment(self, profile: UserProfile) -> str:
        """Identify which segment a user belongs to"""
        
        # Calculate segment scores
        scores = {}
        
        # Tech enthusiast score
        tech_score = 0
        if profile.total_interactions > 50:
            tech_score += 2
        if any(p in profile.price_preferences for p in ['premium', 'flagship']):
            tech_score += 2
        if len(profile.interests) > 3:
            tech_score += 1
        scores['tech_enthusiast'] = tech_score
        
        # Value seeker score
        value_score = 0
        if 'mid-range' in profile.price_preferences:
            value_score += 2
        comparisons = sum(1 for i in profile.interaction_history
                          if i.interaction_type == InteractionType.COMPARE)
        if comparisons > 10:
            value_score += 2
        scores['value_seeker'] = value_score
        
        # Brand loyal score
        brand_score = 0
        if profile.brand_preferences:
            top_brand_weight = max(profile.brand_preferences.values())
            total_brand_weight = sum(profile.brand_preferences.values())
            if total_brand_weight > 0 and top_brand_weight / total_brand_weight > 0.7:
                brand_score += 3
        scores['brand_loyal'] = brand_score
        
        # Casual user score
        casual_score = 0
        if profile.total_interactions < 20:
            casual_score += 2
        if 'budget' in profile.price_preferences:
            casual_score += 1
        scores['casual_user'] = casual_score
        
        # Professional score
        prof_score = 0
        if 'productivity' in ' '.join(profile.search_history).lower():
            prof_score += 2
        if 'battery' in profile.interests or 'performance' in profile.interests:
            prof_score += 1
        scores['professional'] = prof_score
        
        # Return segment with highest score
        return max(scores.items(), key=lambda x: x[1])[0]


class RecommendationPersonalizer:
    """Personalize recommendations based on user profile"""
    
    def __init__(self):
        self.collaborative_weight = 0.4
        self.content_weight = 0.4
        self.trending_weight = 0.2
    
    def personalize_recommendations(
        self,
        profile: UserProfile,
        candidates: List[Dict[str, Any]],
        context: Optional[PersonalizationContext] = None
    ) -> List[Dict[str, Any]]:
        """
        Personalize recommendations for a user
        
        Args:
            profile: User profile
            candidates: List of candidate items
            context: Additional context
            
        Returns:
            Personalized and ranked recommendations
        """
        if not candidates:
            return []
        
        # Score each candidate
        scored_candidates = []
        for candidate in candidates:
            score = self._calculate_personalization_score(profile, candidate, context)
            scored_candidates.append({
                **candidate,
                'personalization_score': score,
                'reasons': self._generate_reasons(profile, candidate, score)
            })
        
        # Sort by personalization score
        scored_candidates.sort(key=lambda x: x['personalization_score'], reverse=True)
        
        # Add diversity if needed
        if len(scored_candidates) > 10:
            scored_candidates = self._ensure_diversity(scored_candidates, profile)
        
        return scored_candidates
    
    def _calculate_personalization_score(
        self,
        profile: UserProfile,
        item: Dict[str, Any],
        context: Optional[PersonalizationContext]
    ) -> float:
        """Calculate personalization score for an item"""
        score = 0.0
        
        # Brand preference score
        brand = item.get('brand', '')
        if brand in profile.brand_preferences:
            score += profile.brand_preferences[brand] * 0.3
        
        # Feature match score
        item_features = set(item.get('features', []))
        user_interests = set(profile.interests)
        if user_interests:
            feature_overlap = len(item_features & user_interests) / len(user_interests)
            score += feature_overlap * 30
        
        # Price match score
        price = item.get('price', 0)
        if profile.predicted_budget:
            min_budget, max_budget = profile.predicted_budget
            if min_budget <= price <= max_budget:
                score += 20
            else:
                # Penalize based on distance from budget
                distance = min(abs(price - min_budget), abs(price - max_budget))
                score -= distance / 100
        
        # Previous interaction bonus
        if item.get('id') in profile.view_history:
            score += 5
        
        # Segment-specific adjustments
        if profile.user_segment:
            score += self._apply_segment_adjustments(profile.user_segment, item)
        
        # Context-based adjustments
        if context:
            score += self._apply_context_adjustments(context, item)
        
        return max(0, min(100, score))  # Normalize to 0-100
    
    def _apply_segment_adjustments(self, segment: str, item: Dict[str, Any]) -> float:
        """Apply segment-specific score adjustments"""
        adjustment = 0.0
        
        if segment == 'tech_enthusiast':
            if item.get('is_flagship', False):
                adjustment += 10
            if item.get('release_date', '') > (datetime.now() - timedelta(days=180)).isoformat():
                adjustment += 5
        
        elif segment == 'value_seeker':
            value_score = item.get('value_score', 0)
            adjustment += value_score * 2
        
        elif segment == 'brand_loyal':
            # Boost items from preferred brand
            adjustment += 5
        
        elif segment == 'casual_user':
            if item.get('is_simple', False):
                adjustment += 8
        
        elif segment == 'professional':
            if 'productivity' in item.get('tags', []):
                adjustment += 10
        
        return adjustment
    
    def _apply_context_adjustments(
        self,
        context: PersonalizationContext,
        item: Dict[str, Any]
    ) -> float:
        """Apply context-based adjustments"""
        adjustment = 0.0
        
        # Time of day adjustments
        if context.time_of_day == 'evening':
            if 'entertainment' in item.get('tags', []):
                adjustment += 3
        elif context.time_of_day == 'morning':
            if 'productivity' in item.get('tags', []):
                adjustment += 3
        
        # Device adjustments
        if context.device_info:
            if context.device_info.get('type') == 'mobile':
                # Prefer items optimized for mobile viewing
                adjustment += 2
        
        # Active filter boosts
        for filter_key, filter_value in context.active_filters.items():
            if item.get(filter_key) == filter_value:
                adjustment += 5
        
        return adjustment
    
    def _generate_reasons(
        self,
        profile: UserProfile,
        item: Dict[str, Any],
        score: float
    ) -> List[str]:
        """Generate reasons for recommendation"""
        reasons = []
        
        # Brand match
        if item.get('brand') in profile.brand_preferences:
            reasons.append(f"From your preferred brand {item['brand']}")
        
        # Feature match
        matching_features = set(item.get('features', [])) & set(profile.interests)
        if matching_features:
            reasons.append(f"Has {', '.join(list(matching_features)[:2])} you're interested in")
        
        # Price match
        if profile.predicted_budget:
            min_b, max_b = profile.predicted_budget
            if min_b <= item.get('price', 0) <= max_b:
                reasons.append("Within your typical budget")
        
        # Segment-based reason
        if profile.user_segment == 'tech_enthusiast' and item.get('is_flagship'):
            reasons.append("Latest flagship technology")
        elif profile.user_segment == 'value_seeker':
            reasons.append("Great value for money")
        
        # Trending
        if item.get('is_trending', False):
            reasons.append("Currently trending")
        
        return reasons if reasons else ["Matches your preferences"]
    
    def _ensure_diversity(
        self,
        candidates: List[Dict[str, Any]],
        profile: UserProfile
    ) -> List[Dict[str, Any]]:
        """Ensure diversity in recommendations"""
        diverse_list = []
        seen_brands = set()
        seen_categories = set()
        
        for candidate in candidates:
            brand = candidate.get('brand', '')
            category = candidate.get('category', '')
            
            # Include if adds diversity or is highly scored
            if (len(diverse_list) < 3 or
                brand not in seen_brands or
                category not in seen_categories or
                candidate['personalization_score'] > 80):
                
                diverse_list.append(candidate)
                seen_brands.add(brand)
                seen_categories.add(category)
                
                if len(diverse_list) >= 20:
                    break
        
        return diverse_list


class ABTestingFramework:
    """A/B testing for personalization strategies"""
    
    def __init__(self):
        self.experiments = {}
        self.results = defaultdict(lambda: {'conversions': 0, 'impressions': 0})
    
    def create_experiment(
        self,
        experiment_id: str,
        variants: List[Dict[str, Any]],
        traffic_split: Optional[List[float]] = None
    ):
        """Create an A/B test experiment"""
        if traffic_split is None:
            traffic_split = [1.0 / len(variants)] * len(variants)
        
        self.experiments[experiment_id] = {
            'variants': variants,
            'traffic_split': traffic_split,
            'created_at': datetime.now(),
            'status': 'active'
        }
    
    def get_variant(self, experiment_id: str, user_id: str) -> Dict[str, Any]:
        """Get variant for a user"""
        if experiment_id not in self.experiments:
            return {}
        
        experiment = self.experiments[experiment_id]
        
        # Use user_id hash for consistent assignment
        hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        bucket = (hash_val % 100) / 100
        
        # Assign to variant based on traffic split
        cumulative = 0
        for i, split in enumerate(experiment['traffic_split']):
            cumulative += split
            if bucket < cumulative:
                return experiment['variants'][i]
        
        return experiment['variants'][-1]
    
    def track_impression(self, experiment_id: str, variant_id: str):
        """Track an impression"""
        key = f"{experiment_id}:{variant_id}"
        self.results[key]['impressions'] += 1
    
    def track_conversion(self, experiment_id: str, variant_id: str):
        """Track a conversion"""
        key = f"{experiment_id}:{variant_id}"
        self.results[key]['conversions'] += 1
    
    def get_results(self, experiment_id: str) -> Dict[str, Any]:
        """Get experiment results"""
        results = {}
        for key, data in self.results.items():
            if key.startswith(f"{experiment_id}:"):
                variant_id = key.split(':')[1]
                conversion_rate = (
                    data['conversions'] / data['impressions']
                    if data['impressions'] > 0 else 0
                )
                results[variant_id] = {
                    'impressions': data['impressions'],
                    'conversions': data['conversions'],
                    'conversion_rate': conversion_rate
                }
        return results


class PersonalizationEngine:
    """Main personalization engine orchestrator"""
    
    def __init__(self, enabled: bool = True, storage_type: str = "memory"):
        """
        Initialize personalization engine
        
        Args:
            enabled: Whether personalization is enabled
            storage_type: Type of storage ('memory' or 'database')
        """
        self.enabled = enabled
        
        if not self.enabled:
            logger.info("Personalization engine disabled")
            return
        
        # Initialize components
        self.profile_store = UserProfileStore(storage_type)
        self.preference_learner = PreferenceLearner()
        self.segmentation = UserSegmentation()
        self.personalizer = RecommendationPersonalizer()
        self.ab_testing = ABTestingFramework()
        
        # Cache for performance
        self.profile_cache = {}
        self.segment_cache = {}
        
        # Statistics
        self.stats = {
            'profiles_created': 0,
            'interactions_processed': 0,
            'recommendations_generated': 0,
            'start_time': datetime.now()
        }
        
        logger.info("Personalization engine initialized")
    
    def track_interaction(
        self,
        user_id: str,
        interaction_type: InteractionType,
        item_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Track a user interaction
        
        Args:
            user_id: User identifier
            interaction_type: Type of interaction
            item_id: Item interacted with
            context: Additional context
            
        Returns:
            Success status
        """
        if not self.enabled:
            return True  # Silently succeed if disabled
        
        try:
            # Get or create user profile
            profile = self.profile_store.get_or_create_profile(user_id)
            
            # Create interaction
            interaction = UserInteraction(
                user_id=user_id,
                interaction_type=interaction_type,
                item_id=item_id,
                timestamp=datetime.now(),
                context=context or {}
            )
            
            # Learn from interaction
            updated_profile = self.preference_learner.learn_from_interaction(
                profile, interaction
            )
            
            # Save updated profile
            self.profile_store.save_profile(updated_profile)
            
            # Update cache
            self.profile_cache[user_id] = updated_profile
            
            # Update statistics
            self.stats['interactions_processed'] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Error tracking interaction: {e}")
            return False
    
    def get_personalized_recommendations(
        self,
        user_id: str,
        candidates: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get personalized recommendations for a user
        
        Args:
            user_id: User identifier
            candidates: List of candidate items
            context: Additional context
            
        Returns:
            Personalized recommendations
        """
        if not self.enabled or not candidates:
            return candidates  # Return unmodified if disabled
        
        try:
            # Get user profile
            profile = self._get_profile(user_id)
            
            if profile is None:
                # New user - return popular items
                return self._get_cold_start_recommendations(candidates)
            
            # Create context
            personalization_context = PersonalizationContext(
                user_profile=profile,
                current_session=context or {}
            )
            
            # Get personalized recommendations
            recommendations = self.personalizer.personalize_recommendations(
                profile,
                candidates,
                personalization_context
            )
            
            # Update statistics
            self.stats['recommendations_generated'] += 1
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return candidates
    
    def _get_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile with caching"""
        # Check cache first
        if user_id in self.profile_cache:
            return self.profile_cache[user_id]
        
        # Load from store
        profile = self.profile_store.load_profile(user_id)
        
        # Cache if found
        if profile:
            self.profile_cache[user_id] = profile
        
        return profile
    
    def _get_cold_start_recommendations(
        self,
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Get recommendations for new users"""
        # Sort by popularity/rating
        sorted_candidates = sorted(
            candidates,
            key=lambda x: x.get('popularity_score', 0) + x.get('rating', 0),
            reverse=True
        )
        
        # Add cold start indicator
        for candidate in sorted_candidates:
            candidate['is_cold_start'] = True
            candidate['personalization_score'] = candidate.get('popularity_score', 50)
        
        return sorted_candidates
    
    def segment_users(self, user_ids: Optional[List[str]] = None) -> Dict[str, List[str]]:
        """
        Segment users into groups
        
        Args:
            user_ids: Specific users to segment (None for all)
            
        Returns:
            Mapping of segment to user IDs
        """
        if not self.enabled:
            return {}
        
        # Get profiles
        if user_ids:
            profiles = [self._get_profile(uid) for uid in user_ids]
            profiles = [p for p in profiles if p is not None]
        else:
            profiles = list(self.profile_cache.values())
        
        if not profiles:
            return {}
        
        # Perform segmentation
        segments = self.segmentation.segment_users(profiles)
        
        # Cache segments
        for segment, users in segments.items():
            self.segment_cache[segment] = users
        
        return segments
    
    def get_user_insights(self, user_id: str) -> Dict[str, Any]:
        """
        Get insights about a user
        
        Args:
            user_id: User identifier
            
        Returns:
            User insights dictionary
        """
        if not self.enabled:
            return {}
        
        profile = self._get_profile(user_id)
        
        if profile is None:
            return {'status': 'new_user'}
        
        return {
            'user_id': user_id,
            'segment': profile.user_segment,
            'interests': profile.interests,
            'predicted_budget': profile.predicted_budget,
            'brand_preferences': dict(sorted(
                profile.brand_preferences.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]),
            'engagement_level': self._calculate_engagement_level(profile),
            'conversion_rate': profile.conversion_rate,
            'total_interactions': profile.total_interactions,
            'last_active': profile.last_active.isoformat()
        }
    
    def _calculate_engagement_level(self, profile: UserProfile) -> str:
        """Calculate user engagement level"""
        if profile.total_interactions > 100:
            return "high"
        elif profile.total_interactions > 30:
            return "medium"
        else:
            return "low"
    
    def run_ab_test(
        self,
        user_id: str,
        experiment_id: str,
        control: Dict[str, Any],
        variants: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Run A/B test for personalization
        
        Args:
            user_id: User identifier
            experiment_id: Experiment ID
            control: Control configuration
            variants: Variant configurations
            
        Returns:
            Selected variant for user
        """
        if not self.enabled:
            return control
        
        # Create experiment if doesn't exist
        if experiment_id not in self.ab_testing.experiments:
            all_variants = [control] + variants
            self.ab_testing.create_experiment(experiment_id, all_variants)
        
        # Get variant for user
        variant = self.ab_testing.get_variant(experiment_id, user_id)
        
        # Track impression
        self.ab_testing.track_impression(experiment_id, variant.get('id', 'control'))
        
        return variant
    
    def get_status(self) -> Dict[str, Any]:
        """Get engine status"""
        if not self.enabled:
            return {'status': 'disabled'}
        
        uptime = (datetime.now() - self.stats['start_time']).total_seconds()
        
        return {
            'status': 'enabled',
            'stats': self.stats,
            'uptime_seconds': uptime,
            'cached_profiles': len(self.profile_cache),
            'segments': list(self.segment_cache.keys()),
            'active_experiments': len(self.ab_testing.experiments),
            'storage_type': self.profile_store.storage_type
        }


# Singleton instance (optional, can be disabled)
_engine_instance = None


def get_personalization_engine(
    enabled: bool = True,
    storage_type: str = "memory"
) -> PersonalizationEngine:
    """Get or create personalization engine instance"""
    global _engine_instance
    
    if _engine_instance is None:
        _engine_instance = PersonalizationEngine(enabled, storage_type)
    
    return _engine_instance


# Example usage
if __name__ == "__main__":
    # Create engine (can be disabled)
    engine = PersonalizationEngine(enabled=True)
    
    # Track some interactions
    engine.track_interaction(
        user_id="user123",
        interaction_type=InteractionType.VIEW,
        item_id="iphone_15_pro",
        context={
            'item_data': {
                'brand': 'Apple',
                'price': 1199,
                'features': ['camera', 'performance', 'display']
            }
        }
    )
    
    engine.track_interaction(
        user_id="user123",
        interaction_type=InteractionType.COMPARE,
        item_id="galaxy_s24",
        context={
            'item_data': {
                'brand': 'Samsung',
                'price': 999,
                'features': ['camera', 'battery', 'display']
            }
        }
    )
    
    # Get personalized recommendations
    candidates = [
        {'id': '1', 'brand': 'Apple', 'price': 1199, 'features': ['camera', 'performance']},
        {'id': '2', 'brand': 'Samsung', 'price': 999, 'features': ['battery', 'display']},
        {'id': '3', 'brand': 'Google', 'price': 799, 'features': ['camera', 'ai']},
    ]
    
    recommendations = engine.get_personalized_recommendations(
        user_id="user123",
        candidates=candidates
    )
    
    print("Personalized Recommendations:")
    for rec in recommendations:
        print(f"- {rec['brand']} (Score: {rec.get('personalization_score', 0):.1f})")
        if 'reasons' in rec:
            print(f"  Reasons: {', '.join(rec['reasons'])}")
    
    # Get user insights
    insights = engine.get_user_insights("user123")
    print(f"\nUser Insights: {json.dumps(insights, indent=2, default=str)}")
    
    # Get engine status
    status = engine.get_status()
    print(f"\nEngine Status: {json.dumps(status, indent=2, default=str)}")
