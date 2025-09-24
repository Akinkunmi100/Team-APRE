"""
Hybrid Integration Module for AI Phone Review Engine
Updates the enhanced user-friendly app to use the Ultimate Hybrid Web Search Agent
"""

import logging
from typing import Dict, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def update_enhanced_app_with_hybrid_search():
    """
    Update the enhanced user-friendly app to use the Ultimate Hybrid Web Search Agent
    
    This function:
    1. Imports the ultimate hybrid search agent
    2. Updates the enhanced app configuration
    3. Replaces the search orchestrator with the hybrid agent
    4. Maintains backward compatibility
    
    Returns:
        Dict with updated configuration and agent instances
    """
    
    try:
        # Import the ultimate hybrid search agent
        from .ultimate_hybrid_web_search_agent import create_ultimate_hybrid_search_agent
        
        # Configuration for the hybrid agent
        hybrid_config = {
            # Google Custom Search API (if available)
            'google_api_key': None,  # Set from environment or config
            'google_search_engine_id': None,  # Set from environment or config
            'enable_google_search': True,
            
            # API Sources configuration
            'enable_api_sources': True,
            'max_concurrent_searches': 3,
            'search_timeout': 30,
            'max_results_per_source': 5,
            'rate_limit_delay': 1.0,
            
            # Offline fallback configuration
            'enable_offline_fallback': True,
            'enable_mock_data': False  # NO MOCK DATA,
            
            # Performance and reliability
            'use_cached_data': True,
            'cache_expiry': 7200,  # 2 hours
            'min_confidence_threshold': 0.5,
            'google_search_confidence_boost': 0.2,
            'prefer_google_over_apis': True
        }
        
        # Create the ultimate hybrid search agent
        ultimate_agent = create_ultimate_hybrid_search_agent(config=hybrid_config)
        
        logger.info("✅ Ultimate Hybrid Web Search Agent created successfully")
        
        # Return configuration for integration
        return {
            'status': 'success',
            'ultimate_agent': ultimate_agent,
            'hybrid_config': hybrid_config,
            'capabilities': ultimate_agent.get_search_statistics()['capabilities'],
            'integration_notes': {
                'google_available': ultimate_agent.google_available,
                'api_sources_available': ultimate_agent.api_search_available,
                'offline_fallback_available': ultimate_agent.offline_search_available,
                'smart_parsing_available': ultimate_agent.smart_search_available
            }
        }
        
    except ImportError as e:
        logger.error(f"Failed to import ultimate hybrid search agent: {e}")
        return {
            'status': 'error',
            'error_type': 'import_error',
            'message': str(e),
            'fallback': 'Use enhanced API search agent instead'
        }
    
    except Exception as e:
        logger.error(f"Failed to create hybrid integration: {e}")
        return {
            'status': 'error',
            'error_type': 'integration_error',
            'message': str(e),
            'fallback': 'Use existing search configuration'
        }

def create_app_integration_config():
    """
    Create configuration specifically for integrating with user-friendly apps
    
    Returns:
        Configuration dict optimized for user-friendly app integration
    """
    
    return {
        # Search configuration
        'search_agent': 'ultimate_hybrid',
        'enable_universal_search': True,
        'fallback_to_offline': True,
        'enable_mock_data_fallback': True,
        
        # UI configuration
        'show_search_statistics': True,
        'show_data_sources': True,
        'show_confidence_scores': True,
        'enable_search_layer_breakdown': True,
        
        # Performance configuration
        'cache_search_results': True,
        'preload_popular_phones': False,
        'enable_background_updates': False,
        
        # Feature flags
        'enable_google_search': True,
        'enable_api_sources': True,
        'enable_offline_fallback': True,
        'enable_smart_recommendations': True,
        
        # Display preferences
        'max_search_results': 10,
        'show_technical_specs': True,
        'show_price_comparisons': True,
        'show_pros_and_cons': True,
        'enable_export_options': True
    }

def get_integration_instructions():
    """
    Get step-by-step integration instructions for developers
    
    Returns:
        Dict with integration instructions and code examples
    """
    
    return {
        'integration_steps': [
            "1. Import the hybrid integration module",
            "2. Call update_enhanced_app_with_hybrid_search()",
            "3. Replace existing search agent with ultimate_agent",
            "4. Update UI components to show search statistics",
            "5. Test all search layers (Google, API, offline)"
        ],
        
        'code_example': '''
# Integration example for enhanced user-friendly app
from core.hybrid_integration import update_enhanced_app_with_hybrid_search

# Update the app with hybrid search
integration_result = update_enhanced_app_with_hybrid_search()

if integration_result['status'] == 'success':
    # Use the ultimate hybrid agent
    search_agent = integration_result['ultimate_agent']
    
    # Perform searches
    phone_data = search_agent.search_phone_external("iPhone 15 Pro")
    
    # Get statistics
    stats = search_agent.get_search_statistics()
    
else:
    # Handle integration errors
    print(f"Integration failed: {integration_result['message']}")
''',
        
        'ui_updates': [
            "Add search layer indicators (Google, API, Offline)",
            "Show confidence scores for search results",
            "Display data source information",
            "Add search statistics dashboard",
            "Include reliability indicators"
        ],
        
        'testing_checklist': [
            "✓ Test search without Google API configured",
            "✓ Test search with all APIs available",
            "✓ Test offline fallback functionality",
            "✓ Test mock data generation",
            "✓ Test caching and performance",
            "✓ Test error handling and graceful degradation"
        ]
    }

def validate_integration_environment():
    """
    Validate the environment for hybrid integration
    
    Returns:
        Dict with validation results and recommendations
    """
    
    validation_results = {
        'environment_ready': True,
        'issues': [],
        'recommendations': [],
        'dependency_status': {},
        'configuration_status': {}
    }
    
    try:
        # Check core dependencies
        try:
            from .ultimate_hybrid_web_search_agent import UltimateHybridWebSearchAgent
            validation_results['dependency_status']['ultimate_hybrid_agent'] = 'available'
        except ImportError:
            validation_results['dependency_status']['ultimate_hybrid_agent'] = 'missing'
            validation_results['issues'].append('Ultimate Hybrid Web Search Agent not found')
            validation_results['environment_ready'] = False
        
        # Check enhanced API search agent
        try:
            from .enhanced_api_web_search_agent import EnhancedAPIWebSearchAgent
            validation_results['dependency_status']['enhanced_api_agent'] = 'available'
        except ImportError:
            validation_results['dependency_status']['enhanced_api_agent'] = 'missing'
            validation_results['recommendations'].append('Enhanced API agent recommended for offline fallback')
        
        # Check original API search agent
        try:
            from .api_web_search_agent import APIWebSearchAgent
            validation_results['dependency_status']['api_agent'] = 'available'
        except ImportError:
            validation_results['dependency_status']['api_agent'] = 'missing'
            validation_results['recommendations'].append('Original API agent recommended for API sources')
        
        # Check Google search integration
        try:
            from .google_search_integration import GoogleCustomSearch
            validation_results['dependency_status']['google_search'] = 'available'
        except ImportError:
            validation_results['dependency_status']['google_search'] = 'missing'
            validation_results['recommendations'].append('Google Custom Search integration available but not found')
        
        # Check smart search
        try:
            from .smart_search import SmartPhoneSearch
            validation_results['dependency_status']['smart_search'] = 'available'
        except ImportError:
            validation_results['dependency_status']['smart_search'] = 'missing'
            validation_results['recommendations'].append('Smart search parsing would improve query understanding')
        
        # Check environment variables
        import os
        
        google_api_key = os.getenv('GOOGLE_SEARCH_API_KEY')
        google_engine_id = os.getenv('GOOGLE_SEARCH_ENGINE_ID')
        
        if google_api_key and google_engine_id:
            validation_results['configuration_status']['google_search_api'] = 'configured'
        else:
            validation_results['configuration_status']['google_search_api'] = 'not_configured'
            validation_results['recommendations'].append('Configure Google Custom Search API for universal web search')
        
        # Generate final recommendations
        if validation_results['environment_ready']:
            if len(validation_results['recommendations']) == 0:
                validation_results['overall_status'] = 'excellent'
                validation_results['message'] = 'Environment fully ready for hybrid integration with all features'
            else:
                validation_results['overall_status'] = 'good'
                validation_results['message'] = 'Environment ready with optional enhancements available'
        else:
            validation_results['overall_status'] = 'needs_attention'
            validation_results['message'] = 'Some critical dependencies missing - check issues list'
        
        return validation_results
        
    except Exception as e:
        logger.error(f"Environment validation failed: {e}")
        return {
            'environment_ready': False,
            'overall_status': 'error',
            'message': f'Validation failed: {str(e)}',
            'issues': [str(e)],
            'recommendations': ['Check system configuration and dependencies']
        }

# Convenience function for quick integration
def quick_hybrid_integration():
    """
    Quick integration function for immediate setup
    
    Returns:
        Ready-to-use hybrid search agent or None if failed
    """
    
    logger.info("🚀 Starting quick hybrid integration...")
    
    # Validate environment first
    validation = validate_integration_environment()
    logger.info(f"Environment validation: {validation['overall_status']}")
    
    if validation['overall_status'] == 'error':
        logger.error("Environment validation failed - aborting integration")
        return None
    
    # Perform integration
    integration_result = update_enhanced_app_with_hybrid_search()
    
    if integration_result['status'] == 'success':
        logger.info("✅ Quick hybrid integration successful!")
        logger.info(f"Available capabilities: {list(integration_result['capabilities'].keys())}")
        return integration_result['ultimate_agent']
    
    else:
        logger.error(f"Quick integration failed: {integration_result['message']}")
        return None

if __name__ == "__main__":
    # Test the integration
    print("Testing Hybrid Integration...")
    
    # Validate environment
    validation = validate_integration_environment()
    print(f"Environment Status: {validation['overall_status']}")
    print(f"Message: {validation['message']}")
    
    if validation.get('issues'):
        print("Issues found:")
        for issue in validation['issues']:
            print(f"  - {issue}")
    
    if validation.get('recommendations'):
        print("Recommendations:")
        for rec in validation['recommendations']:
            print(f"  - {rec}")
    
    # Try quick integration
    agent = quick_hybrid_integration()
    if agent:
        print("✅ Hybrid integration test successful!")
        stats = agent.get_search_statistics()
        print(f"Agent capabilities: {stats['capabilities']}")
    else:
        print("❌ Hybrid integration test failed")