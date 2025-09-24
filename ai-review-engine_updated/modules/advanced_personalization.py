"""
Advanced Personalization Features
Extends the base personalization engine with sophisticated user learning and recommendations
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
from abc import ABC, abstractmethod

# ML/Statistical imports
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.ensemble import RandomForestRegressor
    import pandas as pd
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Deep learning (optional)
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


# ==================== USER PREFERENCE LEARNING ====================

@dataclass
class ImplicitSignal:
    """Implicit user behavior signal"""
    signal_type: str  # 'dwell_time', 'scroll_depth', 'click_through', 'bounce'
    strength: float  # 0.0 to 1.0
    context: Dict[str, Any]
    timestamp: datetime


@dataclass
class ExplicitPreference:
    """Explicit user preference"""
    preference_type: str  # 'budget', 'brand', 'feature', 'use_case'
    value: Any
    confidence: float  # User's confidence in preference
    timestamp: datetime


class UserPreferenceLearner:
    """Advanced user preference learning from implicit and explicit signals"""
    
    def __init__(self):
        self.implicit_weights = {
            'dwell_time': 0.3,
            'scroll_depth': 0.2,
            'click_through': 0.4,
            'add_to_cart': 0.6,
            'purchase': 1.0,
            'bounce': -0.3,
            'quick_back': -0.2
        }
        
        self.preference_decay_rate = 0.95  # Weekly decay
        self.min_confidence_threshold = 0.3
        
    def learn_from_implicit(
        self,
        user_id: str,
        signals: List[ImplicitSignal]
    ) -> Dict[str, float]:
        """
        Learn preferences from implicit user behavior
        
        Args:
            user_id: User identifier
            signals: List of implicit signals
            
        Returns:
            Learned preferences with confidence scores
        """
        preferences = defaultdict(float)
        signal_counts = defaultdict(int)
        
        for signal in signals:
            weight = self.implicit_weights.get(signal.signal_type, 0.1)
            
            # Extract preferences from context
            if 'product_features' in signal.context:
                for feature in signal.context['product_features']:
                    preferences[f'feature_{feature}'] += weight * signal.strength
                    signal_counts[f'feature_{feature}'] += 1
            
            if 'price_range' in signal.context:
                price_range = signal.context['price_range']
                preferences[f'budget_{price_range}'] += weight * signal.strength
                signal_counts[f'budget_{price_range}'] += 1
            
            if 'brand' in signal.context:
                brand = signal.context['brand']
                preferences[f'brand_{brand}'] += weight * signal.strength
                signal_counts[f'brand_{brand}'] += 1
        
        # Normalize by signal count
        normalized_preferences = {}
        for pref, score in preferences.items():
            if signal_counts[pref] > 0:
                normalized_score = score / signal_counts[pref]
                if normalized_score >= self.min_confidence_threshold:
                    normalized_preferences[pref] = min(1.0, normalized_score)
        
        return normalized_preferences
    
    def learn_from_explicit(
        self,
        user_id: str,
        preferences: List[ExplicitPreference]
    ) -> Dict[str, Any]:
        """
        Process explicit user preferences
        
        Args:
            user_id: User identifier
            preferences: List of explicit preferences
            
        Returns:
            Structured preference profile
        """
        profile = {
            'budget': None,
            'brands': [],
            'features': [],
            'use_cases': [],
            'constraints': []
        }
        
        for pref in preferences:
            if pref.preference_type == 'budget':
                if isinstance(pref.value, dict):
                    profile['budget'] = pref.value  # {'min': x, 'max': y}
                elif isinstance(pref.value, (int, float)):
                    profile['budget'] = {'max': pref.value}
            
            elif pref.preference_type == 'brand':
                profile['brands'].append({
                    'name': pref.value,
                    'preference': pref.confidence  # positive or negative
                })
            
            elif pref.preference_type == 'feature':
                profile['features'].append({
                    'name': pref.value,
                    'importance': pref.confidence
                })
            
            elif pref.preference_type == 'use_case':
                profile['use_cases'].append(pref.value)
            
            elif pref.preference_type == 'constraint':
                profile['constraints'].append(pref.value)
        
        return profile
    
    def combine_preferences(
        self,
        implicit_prefs: Dict[str, float],
        explicit_prefs: Dict[str, Any],
        implicit_weight: float = 0.4
    ) -> Dict[str, Any]:
        """
        Combine implicit and explicit preferences
        
        Args:
            implicit_prefs: Learned implicit preferences
            explicit_prefs: Explicit user preferences
            implicit_weight: Weight for implicit preferences (0-1)
            
        Returns:
            Combined preference profile
        """
        combined = explicit_prefs.copy()
        
        # Enhance explicit preferences with implicit learning
        for key, value in implicit_prefs.items():
            if key.startswith('feature_'):
                feature = key.replace('feature_', '')
                # Check if feature already in explicit
                existing = next((f for f in combined['features'] 
                               if f['name'] == feature), None)
                if existing:
                    # Blend scores
                    existing['importance'] = (
                        existing['importance'] * (1 - implicit_weight) +
                        value * implicit_weight
                    )
                else:
                    # Add from implicit
                    combined['features'].append({
                        'name': feature,
                        'importance': value * implicit_weight,
                        'source': 'implicit'
                    })
            
            elif key.startswith('brand_'):
                brand = key.replace('brand_', '')
                existing = next((b for b in combined['brands'] 
                               if b['name'] == brand), None)
                if not existing:
                    combined['brands'].append({
                        'name': brand,
                        'preference': value * implicit_weight,
                        'source': 'implicit'
                    })
        
        return combined


# ==================== CONTEXTUAL RECOMMENDATIONS ====================

@dataclass
class RecommendationContext:
    """Context for generating recommendations"""
    user_id: str
    budget_range: Optional[Tuple[float, float]] = None
    use_case: Optional[str] = None  # 'gaming', 'photography', 'business', 'casual'
    urgency: Optional[str] = None  # 'immediate', 'researching', 'future'
    comparison_products: List[str] = field(default_factory=list)
    excluded_brands: List[str] = field(default_factory=list)
    required_features: List[str] = field(default_factory=list)
    time_of_day: Optional[str] = None
    device_type: Optional[str] = None
    location: Optional[str] = None
    season: Optional[str] = None


class ContextualRecommender:
    """Generate context-aware recommendations"""
    
    def __init__(self):
        self.use_case_features = {
            'gaming': ['performance', 'display', 'cooling', 'battery', 'refresh_rate'],
            'photography': ['camera', 'storage', 'display', 'editing', 'raw_support'],
            'business': ['battery', 'security', 'productivity', 'durability', '5g'],
            'casual': ['ease_of_use', 'battery', 'camera', 'price', 'brand'],
            'content_creation': ['camera', 'storage', 'performance', 'display', 'stabilization'],
            'travel': ['battery', 'camera', 'durability', 'dual_sim', 'maps']
        }
        
        self.budget_categories = {
            'ultra_budget': (0, 200),
            'budget': (200, 400),
            'mid_range': (400, 700),
            'premium': (700, 1000),
            'flagship': (1000, 2000),
            'luxury': (2000, 5000)
        }
    
    def generate_contextual_recommendations(
        self,
        products: List[Dict[str, Any]],
        context: RecommendationContext,
        user_preferences: Optional[Dict[str, Any]] = None,
        n_recommendations: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Generate recommendations based on context
        
        Args:
            products: Available products
            context: Recommendation context
            user_preferences: User preference profile
            n_recommendations: Number of recommendations
            
        Returns:
            Contextual recommendations with explanations
        """
        scored_products = []
        
        for product in products:
            score = 0.0
            reasons = []
            
            # Budget filtering
            if context.budget_range:
                price = product.get('price', 0)
                min_budget, max_budget = context.budget_range
                if min_budget <= price <= max_budget:
                    score += 20
                    reasons.append(f"Within budget ${min_budget}-${max_budget}")
                else:
                    continue  # Skip products outside budget
            
            # Use case matching
            if context.use_case:
                use_case_features = self.use_case_features.get(context.use_case, [])
                product_features = set(product.get('features', []))
                
                feature_match = len(set(use_case_features) & product_features)
                feature_score = (feature_match / len(use_case_features)) * 30
                score += feature_score
                
                if feature_match > 0:
                    reasons.append(f"Good for {context.use_case}")
            
            # Required features check
            if context.required_features:
                product_features = set(product.get('features', []))
                has_all = all(f in product_features for f in context.required_features)
                if has_all:
                    score += 25
                    reasons.append("Has all required features")
                else:
                    continue  # Skip products missing required features
            
            # Brand exclusion
            if context.excluded_brands:
                if product.get('brand') in context.excluded_brands:
                    continue  # Skip excluded brands
            
            # User preference alignment
            if user_preferences:
                pref_score = self._calculate_preference_alignment(
                    product, user_preferences
                )
                score += pref_score * 0.3
                if pref_score > 50:
                    reasons.append("Matches your preferences")
            
            # Urgency adjustment
            if context.urgency == 'immediate':
                if product.get('in_stock', False):
                    score += 10
                    reasons.append("Available now")
                else:
                    score -= 20
            
            # Time and seasonal adjustments
            if context.season == 'holiday':
                if product.get('has_deals', False):
                    score += 15
                    reasons.append("Holiday deal available")
            
            # Add contextual metadata
            scored_products.append({
                **product,
                'context_score': score,
                'reasons': reasons,
                'match_percentage': min(100, score),
                'context_type': context.use_case or 'general'
            })
        
        # Sort by score and return top N
        scored_products.sort(key=lambda x: x['context_score'], reverse=True)
        return scored_products[:n_recommendations]
    
    def _calculate_preference_alignment(
        self,
        product: Dict[str, Any],
        preferences: Dict[str, Any]
    ) -> float:
        """Calculate how well a product aligns with user preferences"""
        score = 0.0
        
        # Brand preference
        if preferences.get('brands'):
            for brand_pref in preferences['brands']:
                if product.get('brand') == brand_pref['name']:
                    score += brand_pref.get('preference', 0.5) * 20
        
        # Feature preferences
        if preferences.get('features'):
            product_features = set(product.get('features', []))
            for feature_pref in preferences['features']:
                if feature_pref['name'] in product_features:
                    score += feature_pref.get('importance', 0.5) * 10
        
        return score


# ==================== A/B TESTING FRAMEWORK ====================

class ABTestVariant:
    """Represents an A/B test variant"""
    
    def __init__(self, variant_id: str, config: Dict[str, Any]):
        self.variant_id = variant_id
        self.config = config
        self.impressions = 0
        self.conversions = 0
        self.revenue = 0.0
        self.engagement_events = []


class ABTestingFramework:
    """Advanced A/B testing for recommendations"""
    
    def __init__(self):
        self.active_tests = {}
        self.test_results = defaultdict(dict)
        self.user_assignments = {}  # user_id -> test_id -> variant_id
        
    def create_test(
        self,
        test_id: str,
        test_name: str,
        variants: Dict[str, Dict[str, Any]],
        traffic_allocation: Optional[Dict[str, float]] = None,
        success_metrics: List[str] = None
    ):
        """
        Create a new A/B test
        
        Args:
            test_id: Unique test identifier
            test_name: Human-readable test name
            variants: Dictionary of variant_id -> configuration
            traffic_allocation: Traffic split between variants
            success_metrics: Metrics to track
        """
        if traffic_allocation is None:
            # Equal split by default
            n_variants = len(variants)
            traffic_allocation = {v: 1.0/n_variants for v in variants}
        
        self.active_tests[test_id] = {
            'name': test_name,
            'variants': {
                vid: ABTestVariant(vid, config) 
                for vid, config in variants.items()
            },
            'traffic_allocation': traffic_allocation,
            'success_metrics': success_metrics or ['conversion_rate', 'engagement'],
            'created_at': datetime.now(),
            'status': 'active'
        }
    
    def get_variant(
        self,
        test_id: str,
        user_id: str,
        user_attributes: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Get variant assignment for a user
        
        Args:
            test_id: Test identifier
            user_id: User identifier
            user_attributes: Optional user attributes for targeting
            
        Returns:
            Tuple of (variant_id, variant_config)
        """
        if test_id not in self.active_tests:
            return 'control', {}
        
        test = self.active_tests[test_id]
        
        # Check if user already assigned
        if user_id in self.user_assignments:
            if test_id in self.user_assignments[user_id]:
                variant_id = self.user_assignments[user_id][test_id]
                return variant_id, test['variants'][variant_id].config
        
        # Assign user to variant
        hash_input = f"{test_id}:{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        bucket = (hash_value % 100) / 100.0
        
        cumulative = 0.0
        for variant_id, allocation in test['traffic_allocation'].items():
            cumulative += allocation
            if bucket < cumulative:
                # Store assignment
                if user_id not in self.user_assignments:
                    self.user_assignments[user_id] = {}
                self.user_assignments[user_id][test_id] = variant_id
                
                # Track impression
                test['variants'][variant_id].impressions += 1
                
                return variant_id, test['variants'][variant_id].config
        
        # Fallback to last variant
        variant_id = list(test['variants'].keys())[-1]
        return variant_id, test['variants'][variant_id].config
    
    def track_event(
        self,
        test_id: str,
        user_id: str,
        event_type: str,
        event_value: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Track user event for A/B test
        
        Args:
            test_id: Test identifier
            user_id: User identifier
            event_type: Type of event (conversion, engagement, etc.)
            event_value: Optional event value (revenue, score, etc.)
            metadata: Additional event metadata
        """
        if test_id not in self.active_tests:
            return
        
        if user_id not in self.user_assignments:
            return
        
        if test_id not in self.user_assignments[user_id]:
            return
        
        variant_id = self.user_assignments[user_id][test_id]
        variant = self.active_tests[test_id]['variants'][variant_id]
        
        # Track event
        if event_type == 'conversion':
            variant.conversions += 1
            if event_value:
                variant.revenue += event_value
        
        variant.engagement_events.append({
            'type': event_type,
            'value': event_value,
            'metadata': metadata,
            'timestamp': datetime.now()
        })
    
    def get_test_results(
        self,
        test_id: str,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Get A/B test results with statistical significance
        
        Args:
            test_id: Test identifier
            confidence_level: Statistical confidence level
            
        Returns:
            Test results with winner determination
        """
        if test_id not in self.active_tests:
            return {}
        
        test = self.active_tests[test_id]
        results = {
            'test_name': test['name'],
            'status': test['status'],
            'created_at': test['created_at'],
            'variants': {}
        }
        
        for variant_id, variant in test['variants'].items():
            conversion_rate = (
                variant.conversions / variant.impressions 
                if variant.impressions > 0 else 0
            )
            
            avg_revenue = (
                variant.revenue / variant.conversions
                if variant.conversions > 0 else 0
            )
            
            results['variants'][variant_id] = {
                'impressions': variant.impressions,
                'conversions': variant.conversions,
                'conversion_rate': conversion_rate,
                'total_revenue': variant.revenue,
                'avg_revenue': avg_revenue,
                'engagement_events': len(variant.engagement_events)
            }
        
        # Determine winner (simplified - should use proper statistical testing)
        if len(results['variants']) > 1:
            best_variant = max(
                results['variants'].items(),
                key=lambda x: x[1]['conversion_rate']
            )
            results['winner'] = best_variant[0]
            results['confidence'] = self._calculate_confidence(test)
        
        return results
    
    def _calculate_confidence(self, test: Dict[str, Any]) -> float:
        """Calculate statistical confidence (simplified)"""
        # This should use proper statistical tests (chi-square, t-test, etc.)
        # Simplified version for demonstration
        total_conversions = sum(
            v.conversions for v in test['variants'].values()
        )
        if total_conversions < 100:
            return 0.0  # Not enough data
        elif total_conversions < 500:
            return 0.8
        else:
            return 0.95


# ==================== BEHAVIORAL ANALYTICS ====================

class BehavioralAnalytics:
    """Track and analyze user behavior patterns"""
    
    def __init__(self):
        self.user_sessions = defaultdict(list)
        self.behavioral_patterns = {}
        self.anomaly_threshold = 3.0  # Standard deviations
        
    def track_behavior(
        self,
        user_id: str,
        action: str,
        context: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ):
        """
        Track user behavior
        
        Args:
            user_id: User identifier
            action: Action type
            context: Action context
            timestamp: When action occurred
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        behavior = {
            'action': action,
            'context': context,
            'timestamp': timestamp
        }
        
        self.user_sessions[user_id].append(behavior)
        
        # Update patterns
        self._update_patterns(user_id, behavior)
    
    def _update_patterns(self, user_id: str, behavior: Dict[str, Any]):
        """Update behavioral patterns for user"""
        if user_id not in self.behavioral_patterns:
            self.behavioral_patterns[user_id] = {
                'action_frequency': defaultdict(int),
                'time_patterns': defaultdict(list),
                'sequence_patterns': [],
                'anomalies': []
            }
        
        patterns = self.behavioral_patterns[user_id]
        
        # Track action frequency
        patterns['action_frequency'][behavior['action']] += 1
        
        # Track time patterns
        hour = behavior['timestamp'].hour
        patterns['time_patterns'][behavior['action']].append(hour)
        
        # Track sequences
        recent_actions = [
            b['action'] for b in self.user_sessions[user_id][-5:]
        ]
        if len(recent_actions) >= 3:
            sequence = '->'.join(recent_actions[-3:])
            patterns['sequence_patterns'].append(sequence)
    
    def get_user_insights(self, user_id: str) -> Dict[str, Any]:
        """
        Get behavioral insights for a user
        
        Args:
            user_id: User identifier
            
        Returns:
            Behavioral insights
        """
        if user_id not in self.behavioral_patterns:
            return {'status': 'no_data'}
        
        patterns = self.behavioral_patterns[user_id]
        sessions = self.user_sessions[user_id]
        
        # Calculate metrics
        total_actions = sum(patterns['action_frequency'].values())
        most_common_action = max(
            patterns['action_frequency'].items(),
            key=lambda x: x[1]
        )[0] if patterns['action_frequency'] else None
        
        # Session analysis
        session_durations = []
        if len(sessions) > 1:
            for i in range(1, len(sessions)):
                duration = (sessions[i]['timestamp'] - sessions[i-1]['timestamp']).seconds
                session_durations.append(duration)
        
        avg_session_duration = np.mean(session_durations) if session_durations else 0
        
        # Time pattern analysis
        peak_hours = {}
        for action, hours in patterns['time_patterns'].items():
            if hours:
                most_common_hour = Counter(hours).most_common(1)[0][0]
                peak_hours[action] = most_common_hour
        
        # Sequence analysis
        common_sequences = Counter(patterns['sequence_patterns']).most_common(3)
        
        return {
            'user_id': user_id,
            'total_actions': total_actions,
            'most_common_action': most_common_action,
            'avg_session_duration': avg_session_duration,
            'peak_activity_hours': peak_hours,
            'common_sequences': common_sequences,
            'engagement_level': self._calculate_engagement_level(patterns),
            'user_type': self._classify_user_type(patterns)
        }
    
    def _calculate_engagement_level(self, patterns: Dict[str, Any]) -> str:
        """Calculate user engagement level"""
        total_actions = sum(patterns['action_frequency'].values())
        
        if total_actions > 100:
            return 'highly_engaged'
        elif total_actions > 30:
            return 'engaged'
        elif total_actions > 10:
            return 'moderate'
        else:
            return 'low'
    
    def _classify_user_type(self, patterns: Dict[str, Any]) -> str:
        """Classify user type based on behavior"""
        actions = patterns['action_frequency']
        
        if actions.get('purchase', 0) > 2:
            return 'buyer'
        elif actions.get('compare', 0) > actions.get('view', 0):
            return 'researcher'
        elif actions.get('search', 0) > 5:
            return 'explorer'
        elif actions.get('view', 0) > 10:
            return 'browser'
        else:
            return 'casual'


# ==================== PERSONALIZED ALERT SYSTEM ====================

@dataclass
class Alert:
    """Personalized alert"""
    alert_id: str
    user_id: str
    type: str  # 'price_drop', 'back_in_stock', 'new_review', 'recommendation'
    title: str
    message: str
    priority: str  # 'high', 'medium', 'low'
    data: Dict[str, Any]
    created_at: datetime
    expires_at: Optional[datetime] = None
    delivered: bool = False
    read: bool = False


class PersonalizedAlertSystem:
    """Generate and manage personalized alerts"""
    
    def __init__(self):
        self.alert_queue = defaultdict(list)
        self.alert_preferences = {}
        self.delivery_channels = ['email', 'push', 'in_app', 'sms']
        
    def set_user_preferences(
        self,
        user_id: str,
        alert_types: List[str],
        channels: List[str],
        frequency: str = 'immediate',  # 'immediate', 'daily', 'weekly'
        quiet_hours: Optional[Tuple[int, int]] = None
    ):
        """
        Set user alert preferences
        
        Args:
            user_id: User identifier
            alert_types: Types of alerts user wants
            channels: Delivery channels
            frequency: Alert frequency
            quiet_hours: Hours when not to send alerts (start, end)
        """
        self.alert_preferences[user_id] = {
            'alert_types': alert_types,
            'channels': channels,
            'frequency': frequency,
            'quiet_hours': quiet_hours
        }
    
    def create_alert(
        self,
        user_id: str,
        alert_type: str,
        title: str,
        message: str,
        data: Dict[str, Any],
        priority: str = 'medium',
        expires_in_hours: Optional[int] = None
    ) -> Optional[Alert]:
        """
        Create a personalized alert
        
        Args:
            user_id: User identifier
            alert_type: Type of alert
            title: Alert title
            message: Alert message
            data: Additional alert data
            priority: Alert priority
            expires_in_hours: Hours until alert expires
            
        Returns:
            Created alert or None if user doesn't want this type
        """
        # Check user preferences
        if user_id in self.alert_preferences:
            prefs = self.alert_preferences[user_id]
            if alert_type not in prefs['alert_types']:
                return None  # User doesn't want this alert type
            
            # Check quiet hours
            if prefs.get('quiet_hours'):
                current_hour = datetime.now().hour
                quiet_start, quiet_end = prefs['quiet_hours']
                if quiet_start <= current_hour < quiet_end:
                    # Delay alert until after quiet hours
                    return None
        
        # Create alert
        alert = Alert(
            alert_id=f"{user_id}_{alert_type}_{datetime.now().timestamp()}",
            user_id=user_id,
            type=alert_type,
            title=title,
            message=message,
            priority=priority,
            data=data,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=expires_in_hours) if expires_in_hours else None
        )
        
        # Add to queue
        self.alert_queue[user_id].append(alert)
        
        return alert
    
    def generate_price_drop_alert(
        self,
        user_id: str,
        product: Dict[str, Any],
        old_price: float,
        new_price: float
    ):
        """Generate alert for price drop"""
        discount_percent = ((old_price - new_price) / old_price) * 100
        
        alert = self.create_alert(
            user_id=user_id,
            alert_type='price_drop',
            title=f"Price Drop: {product['name']}",
            message=f"${old_price:.2f} → ${new_price:.2f} (-{discount_percent:.0f}%)",
            data={
                'product_id': product['id'],
                'product_name': product['name'],
                'old_price': old_price,
                'new_price': new_price,
                'discount_percent': discount_percent
            },
            priority='high' if discount_percent > 20 else 'medium',
            expires_in_hours=48
        )
        
        return alert
    
    def generate_recommendation_alert(
        self,
        user_id: str,
        recommendations: List[Dict[str, Any]],
        reason: str
    ):
        """Generate personalized recommendation alert"""
        alert = self.create_alert(
            user_id=user_id,
            alert_type='recommendation',
            title="New Recommendations for You",
            message=f"We found {len(recommendations)} phones matching your preferences: {reason}",
            data={
                'recommendations': recommendations[:3],  # Top 3
                'reason': reason,
                'total_count': len(recommendations)
            },
            priority='medium',
            expires_in_hours=72
        )
        
        return alert
    
    def get_pending_alerts(
        self,
        user_id: str,
        include_expired: bool = False
    ) -> List[Alert]:
        """
        Get pending alerts for user
        
        Args:
            user_id: User identifier
            include_expired: Include expired alerts
            
        Returns:
            List of pending alerts
        """
        if user_id not in self.alert_queue:
            return []
        
        alerts = self.alert_queue[user_id]
        
        if not include_expired:
            # Filter out expired alerts
            current_time = datetime.now()
            logger.info(f"Filtering expired alerts for user {user_id}, current_time: {current_time}")
            alerts = [
                a for a in alerts
                if not a.expires_at or a.expires_at > current_time
            ]
        
        # Filter undelivered/unread
        pending = [a for a in alerts if not a.read]
        
        # Sort by priority and creation time
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        pending.sort(
            key=lambda a: (priority_order.get(a.priority, 3), a.created_at)
        )
        
        return pending


# ==================== MAIN ORCHESTRATOR ====================

class AdvancedPersonalizationEngine:
    """
    Orchestrates all advanced personalization features
    """
    
    def __init__(self, enabled: bool = True):
        """
        Initialize advanced personalization engine
        
        Args:
            enabled: Whether engine is enabled
        """
        self.enabled = enabled
        
        if not self.enabled:
            logger.info("Advanced personalization disabled")
            return
        
        # Initialize components
        self.preference_learner = UserPreferenceLearner()
        self.contextual_recommender = ContextualRecommender()
        self.ab_testing = ABTestingFramework()
        self.behavioral_analytics = BehavioralAnalytics()
        self.alert_system = PersonalizedAlertSystem()
        
        # Storage
        self.user_profiles = {}
        
        logger.info("Advanced personalization engine initialized")
    
    def process_user_action(
        self,
        user_id: str,
        action_type: str,
        context: Dict[str, Any]
    ):
        """
        Process a user action through all systems
        
        Args:
            user_id: User identifier
            action_type: Type of action
            context: Action context
        """
        if not self.enabled:
            return
        
        # Track in behavioral analytics
        self.behavioral_analytics.track_behavior(
            user_id=user_id,
            action=action_type,
            context=context
        )
        
        # Learn preferences if applicable
        if action_type in ['view', 'click', 'purchase']:
            signal = ImplicitSignal(
                signal_type=action_type,
                strength=self._calculate_signal_strength(action_type, context),
                context=context,
                timestamp=datetime.now()
            )
            
            # Update preferences
            implicit_prefs = self.preference_learner.learn_from_implicit(
                user_id=user_id,
                signals=[signal]
            )
            
            # Store in profile
            if user_id not in self.user_profiles:
                self.user_profiles[user_id] = {}
            
            self.user_profiles[user_id]['implicit_preferences'] = implicit_prefs
        
        # Track A/B test events if applicable
        if 'test_id' in context:
            self.ab_testing.track_event(
                test_id=context['test_id'],
                user_id=user_id,
                event_type=action_type,
                event_value=context.get('value'),
                metadata=context
            )
    
    def _calculate_signal_strength(
        self,
        action_type: str,
        context: Dict[str, Any]
    ) -> float:
        """Calculate signal strength from action"""
        base_strength = {
            'view': 0.3,
            'click': 0.5,
            'add_to_cart': 0.7,
            'purchase': 1.0,
            'review': 0.8
        }.get(action_type, 0.1)
        
        # Adjust based on context
        if context.get('dwell_time', 0) > 30:
            base_strength *= 1.2
        
        return min(1.0, base_strength)
    
    def get_status(self) -> Dict[str, Any]:
        """Get engine status"""
        if not self.enabled:
            return {'status': 'disabled'}
        
        return {
            'status': 'enabled',
            'components': {
                'preference_learning': 'active',
                'contextual_recommendations': 'active',
                'ab_testing': f"{len(self.ab_testing.active_tests)} active tests",
                'behavioral_analytics': f"{len(self.behavioral_analytics.user_sessions)} users tracked",
                'alert_system': 'active'
            },
            'user_profiles': len(self.user_profiles)
        }


# Example usage
if __name__ == "__main__":
    # Initialize engine
    engine = AdvancedPersonalizationEngine(enabled=True)
    
    # Track user behavior
    engine.process_user_action(
        user_id="user123",
        action_type="view",
        context={
            'product_id': 'iphone_15',
            'product_features': ['camera', 'performance'],
            'price_range': 'premium',
            'brand': 'Apple',
            'dwell_time': 45
        }
    )
    
    # Get status
    print(json.dumps(engine.get_status(), indent=2))
