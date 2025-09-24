"""
Dynamic UI Adapter
Adapts the user interface based on user roles and permissions
"""

import streamlit as st
from typing import Dict, List, Optional, Any
from core.user_role_manager import UserRole, UserPermissions, EnhancedUserRoleManager
from utils.business_ui_components import (
    display_business_dashboard, display_bulk_search_interface,
    display_competitor_analysis, display_api_access_panel,
    display_export_options, display_custom_reports_builder,
    display_usage_dashboard
)

class DynamicUIAdapter:
    """Adapts UI based on user role and permissions"""
    
    def __init__(self, role_manager: EnhancedUserRoleManager):
        self.role_manager = role_manager
        self.user_role = role_manager.get_user_role()
        self.permissions = UserPermissions.get_permissions(self.user_role)
        self.ui_config = role_manager.get_user_dashboard_config()
    
    def render_header(self):
        """Render role-appropriate header"""
        if self.user_role in [UserRole.BUSINESS, UserRole.PREMIUM]:
            self.render_business_header()
        else:
            self.render_regular_header()
    
    def render_regular_header(self):
        """Header for regular users"""
        st.markdown("""
        <div class="ultimate-header">
            <h1 style="color: white; margin: 0;">🚀 Ultimate AI Phone Review Engine</h1>
            <p style="color: rgba(255,255,255,0.9); margin: 0; font-size: 1.1rem;">
                Find your perfect phone with AI-powered insights
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_business_header(self):
        """Header for business users"""
        plan_name = "Premium" if self.user_role == UserRole.PREMIUM else "Business"
        st.markdown(f"""
        <div class="ultimate-header">
            <h1 style="color: white; margin: 0;">🚀 Ultimate AI Phone Review Engine</h1>
            <p style="color: rgba(255,255,255,0.9); margin: 0; font-size: 1.1rem;">
                Professional phone market intelligence & analytics
            </p>
            <div style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; display: inline-block; margin-top: 1rem;">
                <span style="color: white; font-weight: bold;">🏢 {plan_name} Plan Active</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render role-appropriate sidebar"""
        st.sidebar.markdown(f"### 👤 User: {self.user_role.value.title()}")
        
        # Usage info
        can_search, search_info = self.role_manager.can_perform_search()
        if can_search:
            st.sidebar.success(f"✅ {search_info}")
        else:
            st.sidebar.error(f"❌ {search_info}")
        
        # Role-specific sidebar content
        if self.user_role in [UserRole.BUSINESS, UserRole.PREMIUM]:
            self.render_business_sidebar()
        else:
            self.render_regular_sidebar()
    
    def render_regular_sidebar(self):
        """Sidebar for regular users"""
        st.sidebar.markdown("### 🎯 Quick Actions")
        
        # Popular searches
        st.sidebar.markdown("#### 🔥 Trending Searches")
        trending_phones = [
            "iPhone 15 Pro", "Samsung Galaxy S24", "Google Pixel 8",
            "OnePlus 12", "Nothing Phone 2a"
        ]
        
        for phone in trending_phones:
            if st.sidebar.button(f"🔍 {phone}", key=f"trending_{phone}"):
                st.session_state.quick_search_query = phone
                st.rerun()
        
        # User preferences reminder
        if hasattr(st.session_state, 'user_preferences'):
            prefs = st.session_state.user_preferences
            if prefs.get('preferred_brands'):
                st.sidebar.markdown("#### 🎯 Your Preferred Brands")
                st.sidebar.info(" • ".join(prefs['preferred_brands']))
        
        # Upgrade prompt
        st.sidebar.markdown("---")
        st.sidebar.markdown("### ⬆️ Upgrade to Business")
        st.sidebar.markdown("Unlock:")
        st.sidebar.markdown("• Bulk search")
        st.sidebar.markdown("• Advanced analytics") 
        st.sidebar.markdown("• API access")
        
        if st.sidebar.button("🚀 Upgrade Now"):
            st.session_state.show_upgrade_modal = True
    
    def render_business_sidebar(self):
        """Sidebar for business users"""
        st.sidebar.markdown("### 📊 Business Tools")
        
        # Quick access to business features
        business_features = []
        if self.ui_config['ui_elements']['show_bulk_search']:
            business_features.append(("🔍 Bulk Search", "bulk_search"))
        if self.ui_config['ui_elements']['show_competitor_analysis']:
            business_features.append(("🏆 Competitor Analysis", "competitor"))
        if self.ui_config['ui_elements']['show_api_section']:
            business_features.append(("🔌 API Access", "api"))
        if self.ui_config['ui_elements']['show_reports_section']:
            business_features.append(("📋 Custom Reports", "reports"))
        if self.ui_config['ui_elements']['show_usage_dashboard']:
            business_features.append(("📈 Usage Analytics", "usage"))
        
        for feature_name, feature_key in business_features:
            if st.sidebar.button(feature_name, key=f"nav_{feature_key}"):
                st.session_state.active_business_tab = feature_key
                st.rerun()
        
        # Current usage
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Usage Today")
        usage = self.role_manager.get_usage_summary()
        st.sidebar.metric("Searches", usage['searches_today'], 
                         delta=f"{usage['daily_limit'] - usage['searches_today']} left")
        st.sidebar.metric("API Calls", usage['api_calls'])
    
    def render_main_interface(self, search_results: List[Dict] = None):
        """Render main interface based on user role"""
        
        # Handle business feature navigation
        if self.user_role in [UserRole.BUSINESS, UserRole.PREMIUM]:
            active_tab = st.session_state.get('active_business_tab', 'search')
            
            if active_tab == 'bulk_search':
                return self.render_bulk_search_tab()
            elif active_tab == 'competitor':
                return self.render_competitor_analysis_tab(search_results)
            elif active_tab == 'api':
                return self.render_api_tab()
            elif active_tab == 'reports':
                return self.render_reports_tab()
            elif active_tab == 'usage':
                return self.render_usage_tab()
        
        # Default: render search interface
        return self.render_search_interface()
    
    def render_search_interface(self):
        """Main search interface"""
        
        # Business dashboard at top for business users
        if self.user_role in [UserRole.BUSINESS, UserRole.PREMIUM]:
            display_business_dashboard(self.role_manager)
            st.markdown("---")
        
        # Main search section
        st.markdown("<div class='main-search'>", unsafe_allow_html=True)
        
        # Search input
        search_query = st.text_input(
            "Search for phones...",
            placeholder="e.g., iPhone 15 Pro, Samsung Galaxy S24, best camera phone",
            key="main_search_input",
            label_visibility="collapsed"
        )
        
        # Handle quick search from sidebar
        if hasattr(st.session_state, 'quick_search_query'):
            search_query = st.session_state.quick_search_query
            del st.session_state.quick_search_query
        
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            search_clicked = st.button("🔍 Search", type="primary", use_container_width=True)
        with col2:
            if self.permissions.get('export_data', False):
                st.button("📤 Export", disabled=not search_clicked)
        with col3:
            if self.user_role in [UserRole.BUSINESS, UserRole.PREMIUM]:
                if st.button("🏆 Compare"):
                    st.session_state.active_business_tab = 'competitor'
                    st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        return search_query, search_clicked
    
    def render_bulk_search_tab(self):
        """Bulk search interface for business users"""
        if not self.ui_config['ui_elements']['show_bulk_search']:
            st.error("⛔ Bulk search not available in your plan")
            return None, False
        
        # Back button
        if st.button("← Back to Search"):
            st.session_state.active_business_tab = 'search'
            st.rerun()
        
        phone_list = display_bulk_search_interface()
        return phone_list, phone_list is not None
    
    def render_competitor_analysis_tab(self, search_results):
        """Competitor analysis tab"""
        if not self.ui_config['ui_elements']['show_competitor_analysis']:
            st.error("⛔ Competitor analysis not available in your plan")
            return None, False
        
        if st.button("← Back to Search"):
            st.session_state.active_business_tab = 'search'
            st.rerun()
        
        display_competitor_analysis(search_results or [])
        return None, False
    
    def render_api_tab(self):
        """API access tab"""
        if not self.ui_config['ui_elements']['show_api_section']:
            st.error("⛔ API access not available in your plan")
            return None, False
        
        if st.button("← Back to Search"):
            st.session_state.active_business_tab = 'search'
            st.rerun()
        
        display_api_access_panel()
        return None, False
    
    def render_reports_tab(self):
        """Custom reports tab"""
        if not self.ui_config['ui_elements']['show_reports_section']:
            st.error("⛔ Custom reports not available in your plan")
            return None, False
        
        if st.button("← Back to Search"):
            st.session_state.active_business_tab = 'search'
            st.rerun()
        
        display_custom_reports_builder()
        return None, False
    
    def render_usage_tab(self):
        """Usage analytics tab"""
        if not self.ui_config['ui_elements']['show_usage_dashboard']:
            st.error("⛔ Usage dashboard not available in your plan")
            return None, False
        
        if st.button("← Back to Search"):
            st.session_state.active_business_tab = 'search'
            st.rerun()
        
        display_usage_dashboard(self.role_manager)
        return None, False
    
    def render_search_results(self, search_results: List[Dict]):
        """Render search results with role-appropriate features"""
        if not search_results:
            return
        
        for i, result in enumerate(search_results):
            with st.container():
                # Result card with role-specific styling
                if self.user_role in [UserRole.BUSINESS, UserRole.PREMIUM]:
                    st.markdown('<div class="web-result-card">', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                
                # Basic result info
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"### {result.get('phone_model', 'Unknown Phone')}")
                    st.markdown(f"**Rating:** {result.get('overall_rating', 'N/A')} ⭐")
                    
                    if result.get('key_features'):
                        st.markdown("**Key Features:** " + ", ".join(result['key_features'][:3]))
                
                with col2:
                    # Role-specific actions
                    if self.user_role in [UserRole.BUSINESS, UserRole.PREMIUM]:
                        if st.button(f"📊 Analyze", key=f"analyze_{i}"):
                            st.session_state.selected_phone_analysis = result
                        if st.button(f"🔍 Deep Dive", key=f"deep_{i}"):
                            st.session_state.deep_dive_phone = result
                    else:
                        if st.button(f"❤️ Save", key=f"save_{i}"):
                            st.success("Saved to your favorites!")
                
                st.markdown("</div>", unsafe_allow_html=True)
        
        # Export options for business users
        if self.user_role in [UserRole.BUSINESS, UserRole.PREMIUM]:
            st.markdown("---")
            display_export_options(search_results, self.user_role)
    
    def render_personalized_recommendations(self, user_memory):
        """Render personalized recommendations based on user role"""
        if not user_memory:
            return
        
        recommendations = user_memory.get_personalized_recommendations()
        
        if recommendations['suggested_searches']:
            if self.user_role in [UserRole.BUSINESS, UserRole.PREMIUM]:
                st.markdown("### 🎯 Business Intelligence Suggestions")
                st.markdown("*Based on your search patterns and industry trends*")
            else:
                st.markdown("### 🎯 Personalized Recommendations")
                st.markdown("*Based on your preferences and search history*")
            
            # Display suggestions differently based on role
            cols = st.columns(3)
            for i, suggestion in enumerate(recommendations['suggested_searches'][:3]):
                with cols[i % 3]:
                    if st.button(f"🔍 {suggestion}", key=f"rec_{i}"):
                        st.session_state.quick_search_query = suggestion
                        st.rerun()
        
        # Show insights
        if recommendations['insights']:
            if self.user_role in [UserRole.BUSINESS, UserRole.PREMIUM]:
                st.markdown("#### 📈 Market Intelligence")
            else:
                st.markdown("#### 💡 Personal Insights")
            
            for key, insight in recommendations['insights'].items():
                st.info(f"**{key.replace('_', ' ').title()}:** {insight}")
    
    def show_upgrade_prompts(self):
        """Show contextual upgrade prompts for regular users"""
        if self.user_role != UserRole.REGULAR:
            return
        
        # Show upgrade prompts in specific contexts
        if st.session_state.get('show_upgrade_modal'):
            st.markdown("### 🚀 Unlock Business Features")
            
            st.info("""
            **Get 4x more searches plus:**
            - 🔍 **Bulk Search** - Compare 50+ phones at once
            - 📊 **Advanced Analytics** - Market insights & trends  
            - 🔌 **API Access** - Integrate with your tools
            - 📋 **Custom Reports** - Professional presentations
            - 📤 **Full Data Export** - CSV, JSON, Excel formats
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🆓 Start Free Trial", type="primary"):
                    st.session_state.show_business_onboarding = True
                    st.session_state.show_upgrade_modal = False
                    st.rerun()
            
            with col2:
                if st.button("Cancel"):
                    st.session_state.show_upgrade_modal = False
                    st.rerun()

def create_ui_adapter(role_manager: EnhancedUserRoleManager) -> DynamicUIAdapter:
    """Factory function to create UI adapter"""
    return DynamicUIAdapter(role_manager)