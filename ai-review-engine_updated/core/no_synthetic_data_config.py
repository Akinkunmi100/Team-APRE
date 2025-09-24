"""
No Synthetic Data Configuration

System-wide configuration to ensure no synthetic, mock, or fake data is generated.
All components should use this configuration to disable synthetic data generation.
"""

from typing import Dict, Any


# Global configuration to disable all synthetic data generation
NO_SYNTHETIC_DATA_CONFIG = {
    # Fallback search system
    'generate_synthetic_data': False,
    'enable_synthetic_generation': False,
    'synthetic_data_enabled': False,
    
    # API search agents
    'enable_mock_data': False,
    'enable_mock_api_data': False,
    'mock_data_enabled': False,
    'mock_data_confidence': 0.0,  # NO MOCK DATA
    
    # Data generation systems
    'enable_data_generation': False,
    'enable_fake_data': False,
    'generate_test_data': False,
    
    # Realtime data simulation
    'enable_fake_realtime': False,
    'simulate_realtime_data': False,
    'fake_events_enabled': False,
    
    # Dataset generation
    'auto_generate_data': False,
    'create_synthetic_datasets': False,
    
    # Error handling preferences
    'prefer_error_over_synthetic': True,
    'require_real_data_only': True,
    'strict_data_validation': True,
    
    # Alternative behaviors when data is unavailable
    'show_data_unavailable_message': True,
    'suggest_alternatives': True,
    'log_data_unavailability': True,
    'cache_availability_status': True,
}


def get_no_synthetic_config() -> Dict[str, Any]:
    """
    Get the configuration that disables all synthetic data generation.
    
    Returns:
        dict: Configuration with all synthetic data generation disabled
    """
    return NO_SYNTHETIC_DATA_CONFIG.copy()


def apply_no_synthetic_config(existing_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply no-synthetic-data settings to an existing configuration.
    
    Args:
        existing_config: Existing configuration dictionary
        
    Returns:
        dict: Updated configuration with synthetic data generation disabled
    """
    updated_config = existing_config.copy()
    updated_config.update(NO_SYNTHETIC_DATA_CONFIG)
    return updated_config


def validate_no_synthetic_config(config: Dict[str, Any]) -> bool:
    """
    Validate that a configuration has synthetic data generation properly disabled.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        bool: True if no synthetic data generation is enabled
    """
    synthetic_flags = [
        'generate_synthetic_data',
        'enable_synthetic_generation', 
        'synthetic_data_enabled',
        'enable_mock_data',
        'enable_mock_api_data',
        'mock_data_enabled',
        'enable_data_generation',
        'enable_fake_data',
        'generate_test_data',
        'enable_fake_realtime',
        'simulate_realtime_data',
        'fake_events_enabled',
        'auto_generate_data',
        'create_synthetic_datasets'
    ]
    
    # Check that all synthetic data flags are disabled
    for flag in synthetic_flags:
        if config.get(flag, False):
            return False
    
    # Check that mock data confidence is zero
    if config.get('mock_data_confidence', 1.0) > 0.0:
        return False
    
    return True


def get_error_handling_config() -> Dict[str, Any]:
    """
    Get configuration for proper error handling instead of synthetic data.
    
    Returns:
        dict: Error handling configuration
    """
    return {
        'handle_missing_data_gracefully': True,
        'return_none_on_missing_data': True,
        'log_data_unavailability': True,
        'provide_alternative_suggestions': True,
        'cache_availability_checks': True,
        'timeout_on_unavailable_apis': 5,
        'max_retries_before_error': 2,
        'user_friendly_error_messages': True
    }


# Component-specific configurations
COMPONENT_CONFIGS = {
    'fallback_search_system': {
        'generate_synthetic_data': False,
        'enable_synthetic_generation': False,
        'return_none_on_no_match': True,
        'log_failed_searches': True
    },
    
    'api_search_agents': {
        'enable_mock_data': False,
        'enable_mock_api_data': False,
        'mock_data_confidence': 0.0,  # NO MOCK DATA
        'return_none_on_api_failure': True,
        'timeout_on_api_failure': 5
    },
    
    'realtime_systems': {
        'enable_fake_realtime': False,
        'simulate_realtime_data': False,
        'use_actual_data_only': True,
        'return_empty_on_no_activity': True
    },
    
    'dataset_systems': {
        'auto_generate_data': False,
        'create_synthetic_datasets': False,
        'require_real_datasets': True,
        'validate_data_authenticity': True
    }
}


def get_component_config(component_name: str) -> Dict[str, Any]:
    """
    Get configuration for a specific component with synthetic data disabled.
    
    Args:
        component_name: Name of the component
        
    Returns:
        dict: Component-specific configuration
    """
    base_config = get_no_synthetic_config()
    component_config = COMPONENT_CONFIGS.get(component_name, {})
    
    base_config.update(component_config)
    return base_config