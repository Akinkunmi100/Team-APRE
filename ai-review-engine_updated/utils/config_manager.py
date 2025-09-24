"""
Configuration Management System for AI Phone Review Engine
Handles loading, validation, and management of system configuration
"""

import os
import yaml
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path
import logging

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class WebSearchConfig:
    """Web search configuration"""
    enabled: bool = True
    max_concurrent_searches: int = 3
    search_timeout: int = 30
    max_results_per_source: int = 5
    min_confidence_threshold: float = 0.6
    rate_limit_delay: float = 2.0
    cache_expiry: int = 3600

@dataclass
class OrchestratorConfig:
    """Search orchestrator configuration"""
    local_confidence_threshold: float = 0.7
    enable_web_fallback: bool = True
    enable_hybrid_search: bool = True
    max_search_timeout: int = 30
    cache_results: bool = True
    log_searches: bool = True

@dataclass
class DatabaseConfig:
    """Database configuration"""
    local_db_path: str = "data/local_reviews.db"
    use_postgresql: bool = False
    postgresql_url: str = ""
    use_redis: bool = False
    redis_url: str = "redis://localhost:6379/0"

@dataclass
class AppConfig:
    """Main application configuration"""
    name: str = "AI Phone Review Engine"
    version: str = "2.1.0"
    debug: bool = False
    log_level: str = "INFO"
    environment: str = "development"

class ConfigManager:
    """Central configuration manager"""
    
    def __init__(self, config_file: str = None):
        """Initialize configuration manager"""
        self.config_file = config_file or self._find_config_file()
        self.config_data = {}
        self._load_config()
        
        # Initialize configuration objects
        self.web_search = self._create_web_search_config()
        self.orchestrator = self._create_orchestrator_config()
        self.database = self._create_database_config()
        self.app = self._create_app_config()
    
    def _find_config_file(self) -> str:
        """Find configuration file in project"""
        possible_paths = [
            "config/config.yaml",
            "config/config.yml",
            "config.yaml",
            "config.yml",
            "settings.yaml",
            "settings.yml"
        ]
        
        project_root = Path(__file__).parent.parent
        
        for path in possible_paths:
            full_path = project_root / path
            if full_path.exists():
                return str(full_path)
        
        # If no config file found, create default
        return self._create_default_config()
    
    def _create_default_config(self) -> str:
        """Create default configuration file"""
        
        default_config = {
            'app': {
                'name': 'AI Phone Review Engine',
                'version': '2.1.0',
                'debug': False,
                'log_level': 'INFO',
                'environment': 'development'
            },
            
            'web_search': {
                'enabled': True,
                'max_concurrent_searches': 3,
                'search_timeout': 30,
                'max_results_per_source': 5,
                'min_confidence_threshold': 0.6,
                'rate_limit_delay': 2.0,
                'cache_expiry': 3600,
                
                'sources': {
                    'gsmarena': {
                        'enabled': True,
                        'priority': 1,
                        'timeout': 15
                    },
                    'phonearena': {
                        'enabled': True,
                        'priority': 2,
                        'timeout': 15
                    },
                    'cnet': {
                        'enabled': True,
                        'priority': 3,
                        'timeout': 15
                    },
                    'techcrunch': {
                        'enabled': True,
                        'priority': 4,
                        'timeout': 15
                    },
                    'google_shopping': {
                        'enabled': False,
                        'priority': 5,
                        'timeout': 20
                    }
                }
            },
            
            'orchestrator': {
                'local_confidence_threshold': 0.7,
                'enable_web_fallback': True,
                'enable_hybrid_search': True,
                'max_search_timeout': 30,
                'cache_results': True,
                'log_searches': True
            },
            
            'database': {
                'local_db_path': 'data/local_reviews.db',
                'use_postgresql': False,
                'postgresql_url': '',
                'use_redis': False,
                'redis_url': 'redis://localhost:6379/0'
            },
            
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file_enabled': True,
                'file_path': 'logs/app.log'
            },
            
            'features': {
                'basic_search': True,
                'web_search': True,
                'hybrid_search': True,
                'emotion_detection': True,
                'sarcasm_detection': True,
                'personalization': True,
                'real_time_updates': False
            }
        }
        
        # Create config directory if it doesn't exist
        config_dir = Path(__file__).parent.parent / 'config'
        config_dir.mkdir(exist_ok=True)
        
        config_path = config_dir / 'config.yaml'
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(default_config, f, default_flow_style=False, indent=2)
            logger.info(f"Created default configuration at {config_path}")
        except Exception as e:
            logger.warning(f"Could not create config file: {e}")
            # Return empty string to use in-memory config
            config_path = ""
        
        return str(config_path)
    
    def _load_config(self):
        """Load configuration from file"""
        
        if not self.config_file or not os.path.exists(self.config_file):
            logger.warning("No configuration file found, using defaults")
            self.config_data = {}
            return
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                if self.config_file.endswith('.json'):
                    self.config_data = json.load(f)
                else:
                    self.config_data = yaml.safe_load(f) or {}
            
            logger.info(f"Loaded configuration from {self.config_file}")
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            self.config_data = {}
    
    def _create_web_search_config(self) -> WebSearchConfig:
        """Create web search configuration"""
        web_config = self.config_data.get('web_search', {})
        
        return WebSearchConfig(
            enabled=web_config.get('enabled', True),
            max_concurrent_searches=web_config.get('max_concurrent_searches', 3),
            search_timeout=web_config.get('search_timeout', 30),
            max_results_per_source=web_config.get('max_results_per_source', 5),
            min_confidence_threshold=web_config.get('min_confidence_threshold', 0.6),
            rate_limit_delay=web_config.get('rate_limit_delay', 2.0),
            cache_expiry=web_config.get('cache_expiry', 3600)
        )
    
    def _create_orchestrator_config(self) -> OrchestratorConfig:
        """Create orchestrator configuration"""
        orch_config = self.config_data.get('orchestrator', {})
        
        return OrchestratorConfig(
            local_confidence_threshold=orch_config.get('local_confidence_threshold', 0.7),
            enable_web_fallback=orch_config.get('enable_web_fallback', True),
            enable_hybrid_search=orch_config.get('enable_hybrid_search', True),
            max_search_timeout=orch_config.get('max_search_timeout', 30),
            cache_results=orch_config.get('cache_results', True),
            log_searches=orch_config.get('log_searches', True)
        )
    
    def _create_database_config(self) -> DatabaseConfig:
        """Create database configuration"""
        db_config = self.config_data.get('database', {})
        
        return DatabaseConfig(
            local_db_path=db_config.get('local_db_path', 'data/local_reviews.db'),
            use_postgresql=db_config.get('use_postgresql', False),
            postgresql_url=db_config.get('postgresql_url', ''),
            use_redis=db_config.get('use_redis', False),
            redis_url=db_config.get('redis_url', 'redis://localhost:6379/0')
        )
    
    def _create_app_config(self) -> AppConfig:
        """Create application configuration"""
        app_config = self.config_data.get('app', {})
        
        return AppConfig(
            name=app_config.get('name', 'AI Phone Review Engine'),
            version=app_config.get('version', '2.1.0'),
            debug=app_config.get('debug', False),
            log_level=app_config.get('log_level', 'INFO'),
            environment=app_config.get('environment', 'development')
        )
    
    def get_web_search_sources_config(self) -> Dict[str, Dict]:
        """Get web search sources configuration"""
        web_config = self.config_data.get('web_search', {})
        return web_config.get('sources', {})
    
    def get_feature_flags(self) -> Dict[str, bool]:
        """Get feature flags configuration"""
        return self.config_data.get('features', {
            'basic_search': True,
            'web_search': True,
            'hybrid_search': True,
            'emotion_detection': True,
            'sarcasm_detection': True,
            'personalization': True,
            'real_time_updates': False
        })
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return self.config_data.get('logging', {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file_enabled': True,
            'file_path': 'logs/app.log'
        })
    
    def update_config(self, section: str, updates: Dict[str, Any]):
        """Update configuration section"""
        if section not in self.config_data:
            self.config_data[section] = {}
        
        self.config_data[section].update(updates)
        
        # Update corresponding config objects
        if section == 'web_search':
            self.web_search = self._create_web_search_config()
        elif section == 'orchestrator':
            self.orchestrator = self._create_orchestrator_config()
        elif section == 'database':
            self.database = self._create_database_config()
        elif section == 'app':
            self.app = self._create_app_config()
    
    def save_config(self):
        """Save current configuration to file"""
        if not self.config_file:
            logger.warning("No config file specified, cannot save")
            return
        
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(self.config_data, f, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Validate web search config
        web_config = self.config_data.get('web_search', {})
        if web_config.get('enabled'):
            if web_config.get('max_concurrent_searches', 0) < 1:
                issues.append("web_search.max_concurrent_searches must be at least 1")
            if web_config.get('search_timeout', 0) < 5:
                issues.append("web_search.search_timeout must be at least 5 seconds")
        
        # Validate orchestrator config
        orch_config = self.config_data.get('orchestrator', {})
        threshold = orch_config.get('local_confidence_threshold', 0.7)
        if not 0.0 <= threshold <= 1.0:
            issues.append("orchestrator.local_confidence_threshold must be between 0.0 and 1.0")
        
        # Validate database config
        db_config = self.config_data.get('database', {})
        if db_config.get('use_postgresql') and not db_config.get('postgresql_url'):
            issues.append("database.postgresql_url required when use_postgresql is true")
        
        return issues
    
    def get_environment_config(self, environment: str = None) -> Dict[str, Any]:
        """Get environment-specific configuration"""
        env = environment or self.app.environment
        env_config = self.config_data.get('environments', {}).get(env, {})
        
        # Merge with base config
        merged_config = self.config_data.copy()
        
        def deep_merge(base: Dict, override: Dict):
            for key, value in override.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_merge(base[key], value)
                else:
                    base[key] = value
        
        deep_merge(merged_config, env_config)
        return merged_config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'app': {
                'name': self.app.name,
                'version': self.app.version,
                'debug': self.app.debug,
                'log_level': self.app.log_level,
                'environment': self.app.environment
            },
            'web_search': {
                'enabled': self.web_search.enabled,
                'max_concurrent_searches': self.web_search.max_concurrent_searches,
                'search_timeout': self.web_search.search_timeout,
                'max_results_per_source': self.web_search.max_results_per_source,
                'min_confidence_threshold': self.web_search.min_confidence_threshold,
                'rate_limit_delay': self.web_search.rate_limit_delay,
                'cache_expiry': self.web_search.cache_expiry
            },
            'orchestrator': {
                'local_confidence_threshold': self.orchestrator.local_confidence_threshold,
                'enable_web_fallback': self.orchestrator.enable_web_fallback,
                'enable_hybrid_search': self.orchestrator.enable_hybrid_search,
                'max_search_timeout': self.orchestrator.max_search_timeout,
                'cache_results': self.orchestrator.cache_results,
                'log_searches': self.orchestrator.log_searches
            },
            'database': {
                'local_db_path': self.database.local_db_path,
                'use_postgresql': self.database.use_postgresql,
                'postgresql_url': self.database.postgresql_url,
                'use_redis': self.database.use_redis,
                'redis_url': self.database.redis_url
            }
        }


# Global configuration manager instance
_config_manager = None

def get_config_manager() -> ConfigManager:
    """Get global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def get_config() -> ConfigManager:
    """Alias for get_config_manager"""
    return get_config_manager()

# Convenience functions for common configurations
def get_web_search_config() -> Dict[str, Any]:
    """Get web search agent configuration"""
    config = get_config_manager()
    sources_config = config.get_web_search_sources_config()
    
    return {
        'max_concurrent_searches': config.web_search.max_concurrent_searches,
        'search_timeout': config.web_search.search_timeout,
        'max_results_per_source': config.web_search.max_results_per_source,
        'min_confidence_threshold': config.web_search.min_confidence_threshold,
        'rate_limit_delay': config.web_search.rate_limit_delay,
        'cache_expiry': config.web_search.cache_expiry,
        'sources': sources_config
    }

def get_orchestrator_config() -> Dict[str, Any]:
    """Get search orchestrator configuration"""
    config = get_config_manager()
    
    return {
        'local_confidence_threshold': config.orchestrator.local_confidence_threshold,
        'enable_web_fallback': config.orchestrator.enable_web_fallback,
        'enable_hybrid_search': config.orchestrator.enable_hybrid_search,
        'max_search_timeout': config.orchestrator.max_search_timeout,
        'cache_results': config.orchestrator.cache_results,
        'log_searches': config.orchestrator.log_searches
    }

def is_feature_enabled(feature_name: str) -> bool:
    """Check if a feature is enabled"""
    config = get_config_manager()
    features = config.get_feature_flags()
    return features.get(feature_name, False)