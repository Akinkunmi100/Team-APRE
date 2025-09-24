"""
User Role Management System
Extends the existing EnhancedUserMemory system to support user roles and permissions
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Literal
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class UserRole(Enum):
    """User role enumeration"""
    GUEST = "guest"          # Anonymous/trial users
    REGULAR = "regular"      # Regular consumers  
    BUSINESS = "business"    # Business users
    PREMIUM = "premium"      # Premium business users
    ADMIN = "admin"         # System administrators

class UserPermissions:
    """Define permissions for each user role"""
    
    ROLE_PERMISSIONS = {
        UserRole.GUEST: {
            'max_searches_per_day': 5,
            'advanced_analytics': False,
            'bulk_search': False,
            'api_access': False,
            'export_data': False,
            'custom_reports': False,
            'priority_support': False,
            'search_history_days': 1
        },
        UserRole.REGULAR: {
            'max_searches_per_day': 50,
            'advanced_analytics': True,
            'bulk_search': False,
            'api_access': False,
            'export_data': True,
            'custom_reports': False,
            'priority_support': False,
            'search_history_days': 30
        },
        UserRole.BUSINESS: {
            'max_searches_per_day': 200,
            'advanced_analytics': True,
            'bulk_search': True,
            'api_access': True,
            'export_data': True,
            'custom_reports': True,
            'priority_support': True,
            'search_history_days': 90,
            'competitor_analysis': True,
            'market_insights': True,
            'white_label': False
        },
        UserRole.PREMIUM: {
            'max_searches_per_day': 1000,
            'advanced_analytics': True,
            'bulk_search': True,
            'api_access': True,
            'export_data': True,
            'custom_reports': True,
            'priority_support': True,
            'search_history_days': 365,
            'competitor_analysis': True,
            'market_insights': True,
            'white_label': True,
            'custom_integrations': True
        },
        UserRole.ADMIN: {
            'unlimited': True
        }
    }
    
    @classmethod
    def get_permissions(cls, role: UserRole) -> Dict[str, Any]:
        """Get permissions for a specific role"""
        return cls.ROLE_PERMISSIONS.get(role, cls.ROLE_PERMISSIONS[UserRole.GUEST])
    
    @classmethod
    def can_perform_action(cls, role: UserRole, action: str) -> bool:
        """Check if a role can perform a specific action"""
        permissions = cls.get_permissions(role)
        return permissions.get(action, False)

class EnhancedUserRoleManager:
    """Extended user management with role-based access control"""
    
    def __init__(self, storage_path: str = "user_profiles", user_id: str = None):
        self.storage_path = storage_path
        self.user_id = user_id
        self.ensure_storage_directory()
        
        # Load or create user profile with role information
        self.user_profile = self._load_or_create_role_profile()
        
    def ensure_storage_directory(self):
        """Create storage directory if it doesn't exist"""
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)
    
    def _get_role_profile_path(self) -> str:
        """Get the path for the user role profile file"""
        return os.path.join(self.storage_path, f"role_profile_{self.user_id}.json")
    
    def _load_or_create_role_profile(self) -> Dict:
        """Load existing role profile or create new one"""
        profile_path = self._get_role_profile_path()
        
        default_profile = {
            'user_id': self.user_id,
            'role': UserRole.GUEST.value,
            'created_at': datetime.now().isoformat(),
            'last_active': datetime.now().isoformat(),
            'subscription_info': {
                'plan': 'free',
                'expires_at': None,
                'auto_renew': False,
                'payment_method': None
            },
            'usage_stats': {
                'searches_today': 0,
                'searches_this_month': 0,
                'last_reset_date': datetime.now().date().isoformat(),
                'total_api_calls': 0,
                'reports_generated': 0
            },
            'business_info': {
                'company_name': None,
                'industry': None,
                'company_size': None,
                'use_cases': []
            },
            'preferences': {
                'dashboard_layout': 'standard',
                'notification_settings': {},
                'export_format': 'csv'
            }
        }
        
        if os.path.exists(profile_path):
            try:
                with open(profile_path, 'r') as f:
                    profile = json.load(f)
                    profile['last_active'] = datetime.now().isoformat()
                    return profile
            except Exception as e:
                logger.warning(f"Error loading role profile: {e}")
        
        return default_profile
    
    def get_user_role(self) -> UserRole:
        """Get current user role"""
        role_str = self.user_profile.get('role', UserRole.GUEST.value)
        return UserRole(role_str)
    
    def upgrade_user_role(self, new_role: UserRole, subscription_info: Dict = None):
        """Upgrade user to a new role"""
        old_role = self.get_user_role()
        self.user_profile['role'] = new_role.value
        self.user_profile['role_upgrade_date'] = datetime.now().isoformat()
        
        if subscription_info:
            self.user_profile['subscription_info'].update(subscription_info)
        
        self._save_role_profile()
        logger.info(f"User {self.user_id} upgraded from {old_role} to {new_role}")
    
    def can_perform_search(self):
        """Check if user can perform another search"""
        role = self.get_user_role()
        permissions = UserPermissions.get_permissions(role)
        
        if permissions.get('unlimited'):
            return True, "Unlimited searches"
        
        # Check daily limit
        usage = self.user_profile['usage_stats']
        today = datetime.now().date().isoformat()
        
        # Reset daily counter if new day
        if usage.get('last_reset_date') != today:
            usage['searches_today'] = 0
            usage['last_reset_date'] = today
            self._save_role_profile()
        
        max_searches = permissions.get('max_searches_per_day', 0)
        current_searches = usage.get('searches_today', 0)
        
        if current_searches >= max_searches:
            return False, f"Daily limit reached ({max_searches} searches)"
        
        return True, f"{max_searches - current_searches} searches remaining today"
    
    def increment_search_count(self):
        """Increment search count for the user"""
        usage = self.user_profile['usage_stats']
        usage['searches_today'] = usage.get('searches_today', 0) + 1
        usage['searches_this_month'] = usage.get('searches_this_month', 0) + 1
        self._save_role_profile()
    
    def get_user_dashboard_config(self) -> Dict:
        """Get dashboard configuration based on user role"""
        role = self.get_user_role()
        permissions = UserPermissions.get_permissions(role)
        
        config = {
            'role': role.value,
            'permissions': permissions,
            'features': {
                'basic_search': True,
                'advanced_analytics': permissions.get('advanced_analytics', False),
                'bulk_search': permissions.get('bulk_search', False),
                'api_access': permissions.get('api_access', False),
                'export_data': permissions.get('export_data', False),
                'custom_reports': permissions.get('custom_reports', False),
                'competitor_analysis': permissions.get('competitor_analysis', False),
                'market_insights': permissions.get('market_insights', False),
            },
            'limits': {
                'max_searches_per_day': permissions.get('max_searches_per_day', 0),
                'search_history_days': permissions.get('search_history_days', 1)
            },
            'ui_elements': self._get_ui_elements_for_role(role)
        }
        
        return config
    
    def _get_ui_elements_for_role(self, role: UserRole) -> Dict:
        """Get UI elements that should be visible for each role"""
        if role in [UserRole.BUSINESS, UserRole.PREMIUM]:
            return {
                'show_advanced_filters': True,
                'show_bulk_search': True,
                'show_api_section': True,
                'show_reports_section': True,
                'show_competitor_analysis': True,
                'show_market_insights': True,
                'show_usage_dashboard': True,
                'show_team_management': role == UserRole.PREMIUM
            }
        elif role == UserRole.REGULAR:
            return {
                'show_advanced_filters': True,
                'show_bulk_search': False,
                'show_api_section': False,
                'show_reports_section': False,
                'show_competitor_analysis': False,
                'show_market_insights': False,
                'show_usage_dashboard': False,
                'show_team_management': False
            }
        else:  # GUEST
            return {
                'show_advanced_filters': False,
                'show_bulk_search': False,
                'show_api_section': False,
                'show_reports_section': False,
                'show_competitor_analysis': False,
                'show_market_insights': False,
                'show_usage_dashboard': False,
                'show_team_management': False
            }
    
    def update_business_info(self, business_info: Dict):
        """Update business information for business users"""
        self.user_profile['business_info'].update(business_info)
        self._save_role_profile()
    
    def get_usage_summary(self) -> Dict:
        """Get usage summary for the user"""
        usage = self.user_profile['usage_stats']
        role = self.get_user_role()
        permissions = UserPermissions.get_permissions(role)
        
        return {
            'role': role.value.title(),
            'searches_today': usage.get('searches_today', 0),
            'searches_this_month': usage.get('searches_this_month', 0),
            'daily_limit': permissions.get('max_searches_per_day', 0),
            'api_calls': usage.get('total_api_calls', 0),
            'reports_generated': usage.get('reports_generated', 0),
            'subscription_status': self.user_profile['subscription_info']['plan']
        }
    
    def _save_role_profile(self):
        """Save user role profile to disk"""
        try:
            profile_path = self._get_role_profile_path()
            with open(profile_path, 'w') as f:
                json.dump(self.user_profile, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving role profile: {e}")
    
    def get_onboarding_flow(self) -> Dict:
        """Get appropriate onboarding flow based on intended user type"""
        return {
            'regular_user': {
                'steps': [
                    'Welcome and introduction',
                    'Basic search tutorial',
                    'Understanding results',
                    'Setting preferences'
                ],
                'estimated_time': '5 minutes'
            },
            'business_user': {
                'steps': [
                    'Business use case identification',
                    'Company information setup',
                    'Advanced features overview',
                    'API integration guide',
                    'Team setup (if applicable)'
                ],
                'estimated_time': '15 minutes'
            }
        }