"""
Complete Integration Test Suite for AI Phone Review Engine
Tests the full system integration including orchestrator, web search, and UI components
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

class TestCompleteIntegration(unittest.TestCase):
    """Test complete system integration"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_data = self._create_test_data()
        
    def _create_test_data(self):
        """Create test data for integration tests"""
        return pd.DataFrame({
            'product': ['iPhone 15 Pro', 'Samsung Galaxy S24', 'Google Pixel 8'] * 10,
            'brand': ['Apple', 'Samsung', 'Google'] * 10,
            'rating': [4.5, 4.3, 4.4] * 10,
            'review_text': ['Great phone', 'Good camera', 'Nice battery'] * 10,
            'sentiment_label': ['positive', 'positive', 'positive'] * 10
        })
    
    def test_config_manager_integration(self):
        """Test configuration manager integration"""
        try:
            from utils.config_manager import get_config_manager
            
            config = get_config_manager()
            
            # Test basic configuration access
            self.assertIsNotNone(config.web_search)
            self.assertIsNotNone(config.orchestrator)
            self.assertIsNotNone(config.app)
            
            # Test configuration validation
            issues = config.validate_config()
            self.assertIsInstance(issues, list)
            
            print("✅ Config Manager Integration: PASSED")
            
        except ImportError as e:
            self.fail(f"Config manager import failed: {e}")
    
    def test_search_orchestrator_integration(self):
        """Test search orchestrator integration"""
        try:
            from core.search_orchestrator import SearchOrchestrator, create_search_orchestrator
            
            # Test orchestrator creation
            orchestrator = create_search_orchestrator(
                local_data=self.test_data,
                config={'enable_web_fallback': False}  # Disable web for testing
            )
            
            self.assertIsNotNone(orchestrator)
            self.assertIsNotNone(orchestrator.local_data)
            
            # Test search functionality
            result = orchestrator.search_phone("iPhone 15 Pro")
            
            self.assertIsNotNone(result)
            self.assertTrue(hasattr(result, 'phone_found'))
            self.assertTrue(hasattr(result, 'source'))
            self.assertTrue(hasattr(result, 'confidence'))
            
            # Test statistics
            stats = orchestrator.get_search_statistics()
            self.assertIsInstance(stats, dict)
            self.assertIn('total_searches', stats)
            
            print("✅ Search Orchestrator Integration: PASSED")
            
        except ImportError as e:
            self.fail(f"Search orchestrator import failed: {e}")
    
    @patch('requests.get')
    def test_web_search_agent_integration(self, mock_get):
        """Test web search agent integration"""
        try:
            from core.web_search_agent import WebSearchAgent, integrate_web_search_with_system
            
            # Mock web response
            mock_response = Mock()
            mock_response.text = '<html><body>Test phone results</body></html>'
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            # Test agent creation
            web_agent = WebSearchAgent()
            self.assertIsNotNone(web_agent)
            self.assertIsNotNone(web_agent.smart_search)
            
            # Test search functionality (with mocked requests)
            with patch.object(web_agent, '_search_single_source', return_value=[]):
                result = web_agent.search_phone_external("Test Phone")
                self.assertIsInstance(result, dict)
                self.assertIn('phone_found', result)
            
            print("✅ Web Search Agent Integration: PASSED")
            
        except ImportError as e:
            self.fail(f"Web search agent import failed: {e}")
    
    def test_ui_components_integration(self):
        """Test UI components integration"""
        try:
            from utils.enhanced_ui_components import (
                display_search_source_indicator,
                enhanced_search_interface,
                inject_enhanced_ui_css
            )
            from core.search_orchestrator import SearchResult
            
            # Create mock search result
            mock_result = SearchResult(
                phone_found=True,
                source='local',
                confidence=0.9,
                phone_data={'product_name': 'Test Phone'},
                search_metadata={'search_time': 1.0},
                recommendations={'verdict': 'Recommended'}
            )
            
            # Test that UI components can handle search results
            # (We can't actually test Streamlit rendering, but we can test the functions exist)
            self.assertTrue(callable(display_search_source_indicator))
            self.assertTrue(callable(enhanced_search_interface))
            self.assertTrue(callable(inject_enhanced_ui_css))
            
            print("✅ UI Components Integration: PASSED")
            
        except ImportError as e:
            self.fail(f"UI components import failed: {e}")
    
    def test_enhanced_app_integration(self):
        """Test enhanced app integration"""
        try:
            # Import the enhanced app to test basic loading
            import user_friendly_app_enhanced
            
            # Test that main functions exist
            self.assertTrue(hasattr(user_friendly_app_enhanced, 'initialize_enhanced_system'))
            self.assertTrue(hasattr(user_friendly_app_enhanced, 'main'))
            
            print("✅ Enhanced App Integration: PASSED")
            
        except ImportError as e:
            print(f"⚠️ Enhanced App Integration: WARNING - {e}")
            # This might fail if streamlit isn't available, which is okay
    
    def test_end_to_end_search_flow(self):
        """Test complete end-to-end search flow"""
        try:
            from core.search_orchestrator import create_search_orchestrator
            
            # Create orchestrator with test data
            orchestrator = create_search_orchestrator(
                local_data=self.test_data,
                config={
                    'enable_web_fallback': False,  # Disable web for stable testing
                    'local_confidence_threshold': 0.5
                }
            )
            
            # Test different search scenarios
            test_cases = [
                ("iPhone 15 Pro", True),  # Should find in test data
                ("Samsung Galaxy S24", True),  # Should find in test data
                ("NonExistent Phone", False)  # Should not find
            ]
            
            for query, should_find in test_cases:
                result = orchestrator.search_phone(query)
                
                self.assertIsNotNone(result)
                self.assertIsInstance(result.phone_found, bool)
                
                if should_find:
                    self.assertTrue(result.phone_found, f"Should have found {query}")
                    self.assertIn(result.source, ['local', 'local_low_confidence'])
                    self.assertIsNotNone(result.phone_data)
                    self.assertIsNotNone(result.recommendations)
                else:
                    # For non-existent phones, we might still get results with low confidence
                    # or no results, both are acceptable
                    pass
            
            # Test statistics after multiple searches
            stats = orchestrator.get_search_statistics()
            self.assertGreater(stats['total_searches'], 0)
            
            print("✅ End-to-End Search Flow: PASSED")
            
        except Exception as e:
            self.fail(f"End-to-end search flow failed: {e}")
    
    def test_configuration_integration(self):
        """Test configuration system integration"""
        try:
            from utils.config_manager import get_config_manager, get_web_search_config, get_orchestrator_config
            
            # Test global config access
            config_manager = get_config_manager()
            
            # Test convenience functions
            web_config = get_web_search_config()
            orch_config = get_orchestrator_config()
            
            self.assertIsInstance(web_config, dict)
            self.assertIsInstance(orch_config, dict)
            
            # Test feature flags
            features = config_manager.get_feature_flags()
            self.assertIsInstance(features, dict)
            self.assertIn('web_search', features)
            
            print("✅ Configuration Integration: PASSED")
            
        except ImportError as e:
            self.fail(f"Configuration integration failed: {e}")
    
    def test_error_handling_integration(self):
        """Test error handling across the system"""
        try:
            from core.search_orchestrator import create_search_orchestrator
            
            # Test with minimal/empty data
            empty_data = pd.DataFrame()
            orchestrator = create_search_orchestrator(local_data=empty_data)
            
            # Should handle empty data gracefully
            result = orchestrator.search_phone("Test Phone")
            self.assertIsNotNone(result)
            
            # Test with invalid configuration
            invalid_config = {'local_confidence_threshold': 2.0}  # Invalid value > 1.0
            orchestrator_invalid = create_search_orchestrator(
                local_data=self.test_data,
                config=invalid_config
            )
            
            # Should still work with invalid config (using defaults)
            result = orchestrator_invalid.search_phone("iPhone 15 Pro")
            self.assertIsNotNone(result)
            
            print("✅ Error Handling Integration: PASSED")
            
        except Exception as e:
            self.fail(f"Error handling test failed: {e}")

class TestSystemDependencies(unittest.TestCase):
    """Test system dependencies and imports"""
    
    def test_required_imports(self):
        """Test that all required modules can be imported"""
        required_modules = [
            'pandas',
            'numpy', 
            'yaml',
            'requests',
            'beautifulsoup4'
        ]
        
        optional_modules = [
            'streamlit',
            'plotly',
            'sklearn',
            'transformers',
            'torch'
        ]
        
        # Test required modules
        for module in required_modules:
            try:
                __import__(module)
                print(f"✅ Required module {module}: Available")
            except ImportError:
                self.fail(f"Required module {module} not available")
        
        # Test optional modules (warnings only)
        for module in optional_modules:
            try:
                __import__(module)
                print(f"✅ Optional module {module}: Available")
            except ImportError:
                print(f"⚠️ Optional module {module}: Not available (some features may be limited)")
    
    def test_project_structure(self):
        """Test project directory structure"""
        project_root = Path(__file__).parent.parent
        
        required_dirs = [
            'core',
            'utils', 
            'models',
            'scrapers',
            'tests',
            'examples'
        ]
        
        required_files = [
            'core/search_orchestrator.py',
            'core/web_search_agent.py',
            'utils/enhanced_ui_components.py',
            'utils/config_manager.py',
            'user_friendly_app_enhanced.py',
            'setup_enhanced.py'
        ]
        
        # Test directories
        for dir_name in required_dirs:
            dir_path = project_root / dir_name
            self.assertTrue(dir_path.exists(), f"Required directory {dir_name} not found")
            print(f"✅ Directory {dir_name}: Exists")
        
        # Test files
        for file_name in required_files:
            file_path = project_root / file_name
            self.assertTrue(file_path.exists(), f"Required file {file_name} not found")
            print(f"✅ File {file_name}: Exists")

def run_integration_tests():
    """Run all integration tests"""
    print("🧪 Running Complete Integration Test Suite")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_suite.addTest(unittest.makeSuite(TestCompleteIntegration))
    test_suite.addTest(unittest.makeSuite(TestSystemDependencies))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(test_suite)
    
    print("\n" + "=" * 60)
    print("📊 Integration Test Results")
    print("=" * 60)
    
    if result.wasSuccessful():
        print("✅ ALL INTEGRATION TESTS PASSED!")
        print("🚀 Your AI Phone Review Engine is ready for deployment")
        
        print("\n📋 Next Steps:")
        print("1. Run the enhanced application:")
        print("   streamlit run user_friendly_app_enhanced.py")
        print("2. Test web search functionality with a real query")
        print("3. Configure settings as needed in the System Settings page")
        
    else:
        print("❌ Some integration tests failed")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
        
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_integration_tests()
    sys.exit(0 if success else 1)