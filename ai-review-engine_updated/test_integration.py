#!/usr/bin/env python3
"""
Test script to verify the role-based integration works without errors
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported"""
    try:
        print("Testing imports...")
        
        # Test core role manager
        from core.user_role_manager import UserRole, UserPermissions, EnhancedUserRoleManager
        print("✅ Core role manager imported successfully")
        
        # Test onboarding system
        from utils.onboarding_system import handle_onboarding_flow
        print("✅ Onboarding system imported successfully")
        
        # Test business UI components
        from utils.business_ui_components import display_business_dashboard
        print("✅ Business UI components imported successfully")
        
        # Test dynamic UI adapter
        from utils.dynamic_ui_adapter import create_ui_adapter
        print("✅ Dynamic UI adapter imported successfully")
        
        print("✅ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_role_system():
    """Test basic role system functionality"""
    try:
        print("\nTesting role system...")
        
        from core.user_role_manager import UserRole, UserPermissions, EnhancedUserRoleManager
        
        # Test user roles
        assert UserRole.GUEST.value == "guest"
        assert UserRole.REGULAR.value == "regular"
        assert UserRole.BUSINESS.value == "business"
        print("✅ User roles work correctly")
        
        # Test permissions
        guest_perms = UserPermissions.get_permissions(UserRole.GUEST)
        business_perms = UserPermissions.get_permissions(UserRole.BUSINESS)
        
        assert guest_perms['max_searches_per_day'] == 5
        assert business_perms['max_searches_per_day'] == 200
        assert business_perms['bulk_search'] == True
        print("✅ Permissions work correctly")
        
        # Test role manager
        role_manager = EnhancedUserRoleManager(user_id="test_user_123")
        initial_role = role_manager.get_user_role()
        assert initial_role == UserRole.GUEST
        print("✅ Role manager works correctly")
        
        print("✅ Role system tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Role system test error: {e}")
        return False

def test_main_app_import():
    """Test that main app can be imported"""
    try:
        print("\nTesting main app import...")
        
        # This will test the entire import chain
        import user_friendly_app_ultimate
        print("✅ Main app imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Main app import error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Running integration tests...\n")
    
    tests = [
        test_imports,
        test_role_system,
        test_main_app_import
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            results.append(False)
    
    print(f"\n📊 Test Results: {sum(results)}/{len(results)} passed")
    
    if all(results):
        print("🎉 All tests passed! Integration is ready.")
        return 0
    else:
        print("❌ Some tests failed. Check errors above.")
        return 1

if __name__ == "__main__":
    exit(main())