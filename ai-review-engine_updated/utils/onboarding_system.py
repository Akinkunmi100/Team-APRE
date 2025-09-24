"""
User Onboarding and Subscription Management System
Handles user registration, role selection, and subscription management
"""

import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from core.user_role_manager import UserRole, UserPermissions

def display_welcome_screen():
    """Initial welcome screen for new users"""
    st.markdown("""
    <div style="text-align: center; padding: 2rem;">
        <h1>🚀 Welcome to Ultimate AI Phone Review Engine</h1>
        <p style="font-size: 1.2rem; color: #666;">
            Discover the perfect phone with AI-powered insights and comprehensive reviews
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # User type selection
    st.markdown("### 👤 What best describes you?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📱 Regular User", type="primary", use_container_width=True):
            st.session_state.selected_user_type = "regular"
            st.session_state.onboarding_step = "regular_setup"
            st.rerun()
            
        st.markdown("""
        **Perfect for:**
        - Finding your next phone
        - Comparing phone features
        - Reading authentic reviews
        - Personal use
        """)
    
    with col2:
        if st.button("🏢 Business User", type="secondary", use_container_width=True):
            st.session_state.selected_user_type = "business"
            st.session_state.onboarding_step = "business_setup"
            st.rerun()
            
        st.markdown("""
        **Perfect for:**
        - Market research
        - Competitive analysis  
        - Bulk phone comparisons
        - API integration
        - Custom reports
        """)

def display_regular_user_onboarding():
    """Onboarding flow for regular users"""
    st.markdown("## 📱 Regular User Setup")
    
    progress_bar = st.progress(0)
    step = st.session_state.get('onboarding_substep', 1)
    
    if step == 1:
        progress_bar.progress(25)
        st.markdown("### 🎯 Tell us about your phone preferences")
        
        col1, col2 = st.columns(2)
        with col1:
            preferred_brands = st.multiselect(
                "Preferred brands (optional):",
                ["Apple", "Samsung", "Google", "OnePlus", "Xiaomi", "Nothing", "Motorola"],
                help="We'll personalize recommendations based on your preferences"
            )
            
            budget_range = st.selectbox(
                "Typical budget range:",
                ["Under ₦500,000", "₦500,000-₦800,000", "₦800,000-₦1,200,000", "₦1,200,000-₦1,800,000", "Over ₦1,800,000", "No preference"]
            )
        
        with col2:
            important_features = st.multiselect(
                "Most important features:",
                ["Camera quality", "Battery life", "Performance", "Display", "Storage", "Price", "Design"],
                help="Help us understand what matters most to you"
            )
            
            usage_type = st.selectbox(
                "Primary phone usage:",
                ["Daily personal use", "Photography", "Gaming", "Business", "Basic calls/texts"]
            )
        
        if st.button("Continue →", type="primary"):
            # Save preferences
            st.session_state.user_preferences = {
                'preferred_brands': preferred_brands,
                'budget_range': budget_range,
                'important_features': important_features,
                'usage_type': usage_type
            }
            st.session_state.onboarding_substep = 2
            st.rerun()
    
    elif step == 2:
        progress_bar.progress(50)
        st.markdown("### 📊 Choose your experience level")
        
        experience_level = st.radio(
            "How familiar are you with phone specifications?",
            ["Beginner - I just want good recommendations",
             "Intermediate - I know some specs but want guidance", 
             "Expert - I want detailed technical information"]
        )
        
        notification_prefs = st.multiselect(
            "Get notified about (optional):",
            ["New phone releases", "Price drops", "Review updates", "Feature comparisons"]
        )
        
        if st.button("Continue →", type="primary"):
            st.session_state.user_preferences.update({
                'experience_level': experience_level,
                'notifications': notification_prefs
            })
            st.session_state.onboarding_substep = 3
            st.rerun()
    
    elif step == 3:
        progress_bar.progress(75)
        st.markdown("### 🚀 Ready to get started!")
        
        st.success("✅ Your preferences have been saved!")
        
        # Show what they'll get
        st.markdown("#### 🎁 Your personalized experience includes:")
        st.markdown("""
        - **🎯 Smart Recommendations** - Phones tailored to your preferences
        - **📈 50 Daily Searches** - More than enough for personal use  
        - **💾 Search History** - Keep track of phones you've researched
        - **📊 Basic Analytics** - See your search patterns and preferences
        - **📤 Export Results** - Download your findings as CSV/JSON
        """)
        
        if st.button("🎉 Start Exploring Phones!", type="primary", use_container_width=True):
            st.session_state.user_role = UserRole.REGULAR.value
            st.session_state.onboarding_complete = True
            st.session_state.show_welcome_message = True
            st.rerun()

def display_business_user_onboarding():
    """Onboarding flow for business users"""
    st.markdown("## 🏢 Business User Setup")
    
    progress_bar = st.progress(0)
    step = st.session_state.get('onboarding_substep', 1)
    
    if step == 1:
        progress_bar.progress(20)
        st.markdown("### 🏢 Company Information")
        
        col1, col2 = st.columns(2)
        with col1:
            company_name = st.text_input("Company Name*")
            industry = st.selectbox(
                "Industry*",
                ["Technology", "Retail", "Telecommunications", "Manufacturing", 
                 "Marketing/Advertising", "Consulting", "Research", "Other"]
            )
        
        with col2:
            company_size = st.selectbox(
                "Company Size*",
                ["1-10 employees", "11-50 employees", "51-200 employees", 
                 "201-1000 employees", "1000+ employees"]
            )
            role = st.text_input("Your Role*", placeholder="e.g., Product Manager, Analyst")
        
        if st.button("Continue →", type="primary") and company_name and industry and company_size and role:
            st.session_state.business_info = {
                'company_name': company_name,
                'industry': industry, 
                'company_size': company_size,
                'user_role': role
            }
            st.session_state.onboarding_substep = 2
            st.rerun()
        elif st.button("Continue →", type="primary"):
            st.error("Please fill in all required fields marked with *")
    
    elif step == 2:
        progress_bar.progress(40)
        st.markdown("### 🎯 Use Cases & Requirements")
        
        use_cases = st.multiselect(
            "Primary use cases (select all that apply):",
            ["Market research", "Competitive analysis", "Product comparison", 
             "Consumer sentiment analysis", "Price monitoring", "Feature benchmarking",
             "Supply chain analysis", "Customer support", "Sales enablement"]
        )
        
        data_requirements = st.multiselect(
            "Data requirements:",
            ["Real-time search results", "Historical data", "API access", 
             "Bulk search capabilities", "Custom reporting", "Data exports",
             "Integration with existing tools"]
        )
        
        team_size = st.selectbox(
            "How many team members will use this?",
            ["Just me", "2-5 people", "6-15 people", "16+ people"]
        )
        
        if st.button("Continue →", type="primary") and use_cases:
            st.session_state.business_info.update({
                'use_cases': use_cases,
                'data_requirements': data_requirements,
                'team_size': team_size
            })
            st.session_state.onboarding_substep = 3
            st.rerun()
        elif st.button("Continue →", type="primary"):
            st.error("Please select at least one use case")
    
    elif step == 3:
        progress_bar.progress(60)
        st.markdown("### 💼 Choose Your Business Plan")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style="border: 2px solid #4CAF50; border-radius: 10px; padding: 20px; margin: 10px 0;">
                <h3>🏢 Business Plan</h3>
                <h2 style="color: #4CAF50;">₦78,000/month</h2>
                <ul>
                    <li>✅ 200 searches/day</li>
                    <li>✅ Advanced analytics</li>
                    <li>✅ Bulk search (up to 50 phones)</li>
                    <li>✅ API access</li>
                    <li>✅ Custom reports</li>
                    <li>✅ Data export (CSV, JSON)</li>
                    <li>✅ Email support</li>
                    <li>✅ 90-day search history</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Choose Business Plan", type="primary", use_container_width=True):
                st.session_state.selected_plan = "business"
                st.session_state.onboarding_substep = 4
                st.experimental_rerun()
        
        with col2:
            st.markdown("""
            <div style="border: 2px solid #FF9800; border-radius: 10px; padding: 20px; margin: 10px 0;">
                <h3>🚀 Premium Plan</h3>
                <h2 style="color: #FF9800;">₦158,000/month</h2>
                <ul>
                    <li>✅ 1000 searches/day</li>
                    <li>✅ All Business features</li>
                    <li>✅ Unlimited bulk search</li>
                    <li>✅ White-label options</li>
                    <li>✅ Custom integrations</li>
                    <li>✅ Priority support</li>
                    <li>✅ Dedicated account manager</li>
                    <li>✅ 1-year search history</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Choose Premium Plan", type="secondary", use_container_width=True):
                st.session_state.selected_plan = "premium"
                st.session_state.onboarding_substep = 4
                st.experimental_rerun()
        
        # Free trial option
        st.markdown("---")
        st.markdown("### 🆓 Start with Free Trial")
        st.info("Try Business features free for 14 days - no credit card required!")
        
        if st.button("Start Free Trial", use_container_width=True):
            st.session_state.selected_plan = "trial"
            st.session_state.onboarding_substep = 4
            st.rerun()
    
    elif step == 4:
        progress_bar.progress(100)
        st.markdown("### 🎉 Setup Complete!")
        
        plan = st.session_state.get('selected_plan', 'trial')
        
        if plan == "trial":
            st.success("✅ Your 14-day free trial has started!")
            st.markdown("#### 🎁 Trial includes:")
            st.markdown("""
            - **📊 Full Business Features** - Try everything risk-free
            - **📈 200 Daily Searches** - Perfect for evaluation
            - **🔍 Bulk Search** - Test with up to 50 phones at once
            - **📤 Data Export** - Download results in multiple formats
            - **📞 Email Support** - Get help when you need it
            """)
            
            role_to_set = UserRole.BUSINESS
            
        elif plan == "business":
            st.success("✅ Business Plan activated!")
            role_to_set = UserRole.BUSINESS
            
        elif plan == "premium":
            st.success("✅ Premium Plan activated!")
            role_to_set = UserRole.PREMIUM
            
        if st.button("🚀 Access Business Dashboard", type="primary", use_container_width=True):
            st.session_state.user_role = role_to_set.value
            st.session_state.onboarding_complete = True
            st.session_state.show_welcome_message = True
            st.rerun()

def display_subscription_management():
    """Subscription management interface"""
    st.markdown("## 💳 Subscription Management")
    
    # Current plan info
    current_role = st.session_state.get('user_role', 'guest')
    
    if current_role == UserRole.REGULAR.value:
        st.info("📱 **Current Plan:** Regular User (Free)")
        
        st.markdown("### ⬆️ Upgrade to Business")
        st.markdown("Unlock powerful business features:")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            - 🔍 **Bulk Search** - Compare multiple phones at once
            - 📊 **Advanced Analytics** - Deep market insights  
            - 🔌 **API Access** - Integrate with your tools
            - 📋 **Custom Reports** - Professional analysis reports
            - 📈 **200 Daily Searches** - 4x more than regular plan
            """)
        
        with col2:
            if st.button("Upgrade Now", type="primary"):
                st.session_state.show_upgrade_modal = True
                
    elif current_role in [UserRole.BUSINESS.value, UserRole.PREMIUM.value]:
        plan_name = "Business" if current_role == UserRole.BUSINESS.value else "Premium"
        st.success(f"✅ **Current Plan:** {plan_name}")
        
        # Usage summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Searches This Month", "142", "58 remaining today")
        with col2:
            st.metric("API Calls", "89", "1,911 remaining")
        with col3:
            st.metric("Reports Generated", "7", "Unlimited")
        
        # Billing info
        st.markdown("### 💳 Billing Information")
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Next Billing:** {(datetime.now() + timedelta(days=23)).strftime('%B %d, %Y')}")
            st.info("**Payment Method:** •••• 4242")
        with col2:
            if st.button("Update Payment Method"):
                st.info("Payment update feature coming soon!")
            if st.button("Download Invoice"):
                st.info("Invoice download feature coming soon!")
    
    # Upgrade/downgrade options
    if current_role != UserRole.PREMIUM.value:
        st.markdown("---")
        st.markdown("### 🚀 Available Upgrades")
        display_plan_comparison()

def display_plan_comparison():
    """Display plan comparison table"""
    
    plans_data = {
        "Feature": [
            "Daily Searches", "Advanced Analytics", "Bulk Search", 
            "API Access", "Custom Reports", "Data Export", 
            "Search History", "Support", "Price"
        ],
        "Regular": [
            "50", "✅", "❌", "❌", "❌", "Basic", "30 days", "Community", "Free"
        ],
        "Business": [
            "200", "✅", "✅", "✅", "✅", "Full", "90 days", "Email", "$49/mo"
        ],
        "Premium": [
            "1000", "✅", "✅", "✅", "✅", "Full", "1 year", "Priority", "$99/mo"
        ]
    }
    
    import pandas as pd
    df = pd.DataFrame(plans_data)
    
    # Style the dataframe
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

def handle_onboarding_flow():
    """Main onboarding flow controller"""
    
    # Initialize onboarding state
    if 'onboarding_complete' not in st.session_state:
        st.session_state.onboarding_complete = False
    
    if 'onboarding_step' not in st.session_state:
        st.session_state.onboarding_step = "welcome"
        
    if 'onboarding_substep' not in st.session_state:
        st.session_state.onboarding_substep = 1
    
    # Skip onboarding if already complete
    if st.session_state.onboarding_complete:
        return True
    
    # Route to appropriate onboarding screen
    if st.session_state.onboarding_step == "welcome":
        display_welcome_screen()
        return False
        
    elif st.session_state.onboarding_step == "regular_setup":
        display_regular_user_onboarding()
        return False
        
    elif st.session_state.onboarding_step == "business_setup":
        display_business_user_onboarding() 
        return False
    
    return st.session_state.onboarding_complete