"""
Enhanced System Orchestrator
Centralized initialization and management for the Ultimate AI Phone Review Engine
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from functools import lru_cache
from dataclasses import dataclass

# Import components
from .preprocessed_data_loader import PreprocessedDataLoader
from .data_preprocessing import DataPreprocessor
from .data_quality_validator import DataQualityValidator

logger = logging.getLogger(__name__)

@dataclass
class ComponentStatus:
    """Status tracking for system components"""
    name: str
    status: str  # 'initializing', 'ready', 'error', 'degraded'
    message: str
    last_updated: datetime
    health_score: float = 1.0
    metrics: Dict = None

class SystemOrchestrator:
    """
    Enhanced system orchestrator with proper initialization pattern
    Manages all components, data loading, and system health
    """
    
    def __init__(self, config: Dict = None):
        """Initialize the enhanced system orchestrator"""
        self.config = config or {}
        self.components = {}
        self.system_status = {
            'initialization_time': datetime.now(),
            'status': 'initializing',
            'components': {},
            'data_sources': {},
            'health_metrics': {},
            'error_log': []
        }
        
        # Component registry
        self.component_registry = {
            'data_loader': None,
            'data_preprocessor': None,
            'data_validator': None,
            'sentiment_analyzer': None,
            'recommendation_engine': None,
            'review_analyzer': None
        }
        
        logger.info("🚀 Starting Enhanced System Initialization...")
        self._initialize_enhanced_system()
    
    def _initialize_enhanced_system(self):
        """Enhanced system initialization with comprehensive status tracking"""
        try:
            initialization_steps = [
                ("🔍 Detecting Available Components", self._detect_available_components),
                ("📊 Loading Data Sources", self._initialize_data_sources),
                ("🛠️ Setting Up Preprocessors", self._initialize_preprocessors),
                ("✅ Validating Data Quality", self._validate_system_data),
                ("🤖 Loading AI Components", self._initialize_ai_components),
                ("🔧 Running System Health Check", self._perform_system_health_check),
                ("🎯 Finalizing System Setup", self._finalize_system_setup)
            ]
            
            for step_name, step_function in initialization_steps:
                logger.info(f"Executing: {step_name}")
                start_time = time.time()
                
                try:
                    result = step_function()
                    execution_time = time.time() - start_time
                    
                    self.system_status['components'][step_name] = {
                        'status': 'completed',
                        'execution_time': execution_time,
                        'result': result,
                        'timestamp': datetime.now()
                    }
                    logger.info(f"✅ {step_name} completed in {execution_time:.2f}s")
                    
                except Exception as e:
                    execution_time = time.time() - start_time
                    error_msg = f"❌ {step_name} failed: {str(e)}"
                    logger.error(error_msg)
                    
                    self.system_status['components'][step_name] = {
                        'status': 'error',
                        'execution_time': execution_time,
                        'error': str(e),
                        'timestamp': datetime.now()
                    }
                    self.system_status['error_log'].append({
                        'step': step_name,
                        'error': str(e),
                        'timestamp': datetime.now()
                    })
            
            # Determine overall system status
            self._determine_system_status()
            
        except Exception as e:
            logger.error(f"Critical error during system initialization: {e}")
            self.system_status['status'] = 'critical_error'
            self.system_status['error'] = str(e)
    
    def _detect_available_components(self) -> Dict[str, bool]:
        """Detect which components are available for loading"""
        component_availability = {
            'preprocessed_data_available': False,
            'data_modules_available': False,
            'ai_models_available': False,
            'spacy_model_available': False,
            'nltk_resources_available': False
        }
        
        # Check for preprocessed data
        try:
            data_loader = PreprocessedDataLoader()
            if data_loader.data is not None and not data_loader.data.empty:
                component_availability['preprocessed_data_available'] = True
                self.system_status['data_sources']['preprocessed'] = {
                    'path': str(data_loader.dataset_path),
                    'records': len(data_loader.data),
                    'status': 'available'
                }
        except Exception as e:
            logger.warning(f"Preprocessed data not available: {e}")
        
        # Check for unified data access
        try:
            from .unified_data_access import get_primary_dataset
            test_df = get_primary_dataset()
            if test_df is not None and not test_df.empty:
                component_availability['data_modules_available'] = True
                self.system_status['data_sources']['unified'] = {
                    'records': len(test_df),
                    'status': 'available'
                }
        except Exception as e:
            logger.warning(f"Unified data access not available: {e}")
        
        # Check for CSV fallback
        csv_path = Path('final_dataset_streamlined_clean.csv')
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path, nrows=5)  # Test read
                self.system_status['data_sources']['csv_fallback'] = {
                    'path': str(csv_path),
                    'size': csv_path.stat().st_size,
                    'status': 'available'
                }
            except Exception as e:
                logger.warning(f"CSV fallback not readable: {e}")
        
        # Check AI components
        try:
            from models.absa_model import ABSASentimentAnalyzer
            from models.recommendation_engine_simple import RecommendationEngine
            component_availability['ai_models_available'] = True
        except ImportError as e:
            logger.warning(f"AI models not available: {e}")
        
        # Check spaCy
        try:
            import spacy
            nlp = spacy.load('en_core_web_sm')
            component_availability['spacy_model_available'] = True
        except Exception as e:
            logger.warning(f"spaCy model not available: {e}")
        
        # Check NLTK resources
        try:
            import nltk
            from nltk.corpus import stopwords
            stopwords.words('english')
            component_availability['nltk_resources_available'] = True
        except Exception as e:
            logger.warning(f"NLTK resources not available: {e}")
        
        return component_availability
    
    def _initialize_data_sources(self) -> Dict[str, Any]:
        """Initialize data sources with fallback hierarchy"""
        data_info = {
            'primary_source': None,
            'fallback_sources': [],
            'total_records': 0,
            'data_quality_score': 0.0
        }
        
        # Try preprocessed data first (highest priority)
        if self.system_status['data_sources'].get('preprocessed', {}).get('status') == 'available':
            try:
                data_loader = PreprocessedDataLoader()
                self.component_registry['data_loader'] = data_loader
                
                df = data_loader.get_full_dataset()
                data_info['primary_source'] = 'preprocessed'
                data_info['total_records'] = len(df)
                data_info['columns'] = list(df.columns)
                data_info['data_quality_score'] = self._calculate_data_quality_score(df)
                
                logger.info(f"✅ Loaded {len(df)} records from preprocessed data")
                return data_info
                
            except Exception as e:
                logger.warning(f"Failed to load preprocessed data: {e}")
                data_info['fallback_sources'].append(f"preprocessed_failed: {e}")
        
        # Try unified data access (medium priority)
        if self.system_status['data_sources'].get('unified', {}).get('status') == 'available':
            try:
                from .unified_data_access import get_primary_dataset
                df = get_primary_dataset()
                
                data_info['primary_source'] = 'unified'
                data_info['total_records'] = len(df)
                data_info['columns'] = list(df.columns)
                data_info['data_quality_score'] = self._calculate_data_quality_score(df)
                
                logger.info(f"✅ Loaded {len(df)} records from unified data access")
                return data_info
                
            except Exception as e:
                logger.warning(f"Failed to load unified data: {e}")
                data_info['fallback_sources'].append(f"unified_failed: {e}")
        
        # Try CSV fallback (lowest priority)
        if self.system_status['data_sources'].get('csv_fallback', {}).get('status') == 'available':
            try:
                df = pd.read_csv('final_dataset_streamlined_clean.csv')
                
                data_info['primary_source'] = 'csv_fallback'
                data_info['total_records'] = len(df)
                data_info['columns'] = list(df.columns)
                data_info['data_quality_score'] = self._calculate_data_quality_score(df)
                
                # Store in component registry
                class SimpleDataLoader:
                    def __init__(self, data):
                        self.data = data
                    def get_full_dataset(self):
                        return self.data
                
                self.component_registry['data_loader'] = SimpleDataLoader(df)
                
                logger.info(f"✅ Loaded {len(df)} records from CSV fallback")
                return data_info
                
            except Exception as e:
                logger.error(f"Failed to load CSV fallback: {e}")
                data_info['fallback_sources'].append(f"csv_failed: {e}")
        
        # No data sources available
        logger.error("❌ No data sources available")
        data_info['primary_source'] = 'none'
        return data_info
    
    def _initialize_preprocessors(self) -> Dict[str, Any]:
        """Initialize data preprocessing components"""
        preprocessor_info = {
            'text_preprocessor': None,
            'data_validator': None,
            'features_available': [],
            'preprocessing_capabilities': {}
        }
        
        # Initialize text preprocessor
        try:
            preprocessor = DataPreprocessor()
            self.component_registry['data_preprocessor'] = preprocessor
            
            preprocessor_info['text_preprocessor'] = 'ready'
            preprocessor_info['preprocessing_capabilities'] = {
                'text_cleaning': True,
                'tokenization': True,
                'stopword_removal': True,
                'lemmatization': True,
                'tfidf_extraction': True,
                'spam_detection': True
            }
            
            logger.info("✅ Text preprocessor initialized")
            
        except Exception as e:
            logger.warning(f"Text preprocessor failed: {e}")
            preprocessor_info['text_preprocessor'] = f'error: {e}'
        
        # Initialize data validator
        try:
            validator = DataQualityValidator()
            self.component_registry['data_validator'] = validator
            
            preprocessor_info['data_validator'] = 'ready'
            preprocessor_info['features_available'].append('data_validation')
            
            logger.info("✅ Data validator initialized")
            
        except Exception as e:
            logger.warning(f"Data validator failed: {e}")
            preprocessor_info['data_validator'] = f'error: {e}'
        
        return preprocessor_info
    
    def _validate_system_data(self) -> Dict[str, Any]:
        """Validate system data quality and integrity"""
        validation_results = {
            'overall_score': 0.0,
            'validations_passed': [],
            'validations_failed': [],
            'data_health': 'unknown',
            'recommendations': []
        }
        
        if not self.component_registry.get('data_loader'):
            validation_results['validations_failed'].append('no_data_loader')
            validation_results['data_health'] = 'critical'
            return validation_results
        
        try:
            df = self.component_registry['data_loader'].get_full_dataset()
            
            # Basic validation checks
            validations = [
                ('data_not_empty', len(df) > 0),
                ('required_columns_present', self._check_required_columns(df)),
                ('no_excessive_nulls', self._check_null_percentage(df) < 50),
                ('rating_values_valid', self._check_rating_validity(df)),
                ('text_content_available', self._check_text_content(df))
            ]
            
            passed_validations = 0
            for validation_name, validation_result in validations:
                if validation_result:
                    validation_results['validations_passed'].append(validation_name)
                    passed_validations += 1
                else:
                    validation_results['validations_failed'].append(validation_name)
            
            # Calculate overall score
            validation_results['overall_score'] = (passed_validations / len(validations)) * 100
            
            # Determine data health
            if validation_results['overall_score'] >= 80:
                validation_results['data_health'] = 'excellent'
            elif validation_results['overall_score'] >= 60:
                validation_results['data_health'] = 'good'
            elif validation_results['overall_score'] >= 40:
                validation_results['data_health'] = 'fair'
            else:
                validation_results['data_health'] = 'poor'
            
            # Generate recommendations
            if 'no_excessive_nulls' in validation_results['validations_failed']:
                validation_results['recommendations'].append('Consider data cleaning to reduce null values')
            
            logger.info(f"✅ Data validation completed - Health: {validation_results['data_health']}")
            
        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            validation_results['validations_failed'].append(f'validation_error: {e}')
            validation_results['data_health'] = 'error'
        
        return validation_results
    
    def _initialize_ai_components(self) -> Dict[str, Any]:
        """Initialize AI/ML components with graceful degradation"""
        ai_info = {
            'sentiment_analyzer': None,
            'recommendation_engine': None,
            'capabilities': [],
            'performance_level': 'basic'
        }
        
        # Try to load advanced AI components
        try:
            from models.absa_model import ABSASentimentAnalyzer
            sentiment_analyzer = ABSASentimentAnalyzer()
            self.component_registry['sentiment_analyzer'] = sentiment_analyzer
            
            ai_info['sentiment_analyzer'] = 'advanced'
            ai_info['capabilities'].append('aspect_based_sentiment_analysis')
            ai_info['performance_level'] = 'advanced'
            
            logger.info("✅ Advanced sentiment analyzer loaded")
            
        except Exception as e:
            logger.warning(f"Advanced sentiment analyzer not available: {e}")
            ai_info['sentiment_analyzer'] = f'fallback: {e}'
            ai_info['capabilities'].append('basic_sentiment_analysis')
        
        # Try to load recommendation engine
        try:
            from models.recommendation_engine_simple import RecommendationEngine
            recommendation_engine = RecommendationEngine()
            self.component_registry['recommendation_engine'] = recommendation_engine
            
            ai_info['recommendation_engine'] = 'ready'
            ai_info['capabilities'].append('ml_recommendations')
            
            logger.info("✅ Recommendation engine loaded")
            
        except Exception as e:
            logger.warning(f"Recommendation engine not available: {e}")
            ai_info['recommendation_engine'] = f'unavailable: {e}'
            ai_info['capabilities'].append('rule_based_recommendations')
        
        return ai_info
    
    def _perform_system_health_check(self) -> Dict[str, Any]:
        """Comprehensive system health check"""
        health_metrics = {
            'overall_health': 0.0,
            'component_health': {},
            'performance_metrics': {},
            'system_readiness': False
        }
        
        # Check each component
        component_scores = {}
        
        # Data loader health
        if self.component_registry.get('data_loader'):
            try:
                df = self.component_registry['data_loader'].get_full_dataset()
                component_scores['data_loader'] = 1.0 if not df.empty else 0.5
            except:
                component_scores['data_loader'] = 0.0
        else:
            component_scores['data_loader'] = 0.0
        
        # Preprocessor health
        component_scores['preprocessor'] = 1.0 if self.component_registry.get('data_preprocessor') else 0.0
        
        # AI components health
        component_scores['sentiment_analyzer'] = 1.0 if self.component_registry.get('sentiment_analyzer') else 0.5
        component_scores['recommendation_engine'] = 1.0 if self.component_registry.get('recommendation_engine') else 0.5
        
        # Calculate overall health
        health_metrics['component_health'] = component_scores
        health_metrics['overall_health'] = sum(component_scores.values()) / len(component_scores) * 100
        
        # System readiness
        critical_components = ['data_loader', 'preprocessor']
        health_metrics['system_readiness'] = all(
            component_scores.get(comp, 0) > 0 for comp in critical_components
        )
        
        logger.info(f"✅ System health check completed - Overall: {health_metrics['overall_health']:.1f}%")
        
        return health_metrics
    
    def _finalize_system_setup(self) -> Dict[str, Any]:
        """Finalize system setup and create main analyzer"""
        setup_info = {
            'analyzer_created': False,
            'system_mode': 'unknown',
            'capabilities_summary': [],
            'initialization_complete': False
        }
        
        try:
            # Create the main review analyzer with initialized components
            from ultimate_web_app import UltimateReviewAnalyzer
            
            # Override the analyzer's components with our initialized ones
            analyzer = UltimateReviewAnalyzer()
            
            # Replace its components with our properly initialized ones
            if self.component_registry.get('data_loader'):
                analyzer.df = self.component_registry['data_loader'].get_full_dataset()
            
            if self.component_registry.get('sentiment_analyzer'):
                analyzer.sentiment_analyzer = self.component_registry['sentiment_analyzer']
            
            if self.component_registry.get('recommendation_engine'):
                analyzer.recommendation_engine = self.component_registry['recommendation_engine']
            
            self.component_registry['review_analyzer'] = analyzer
            
            setup_info['analyzer_created'] = True
            setup_info['system_mode'] = self._determine_system_mode()
            setup_info['capabilities_summary'] = self._generate_capabilities_summary()
            setup_info['initialization_complete'] = True
            
            logger.info("✅ Enhanced system initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to finalize system setup: {e}")
            setup_info['analyzer_created'] = False
            setup_info['error'] = str(e)
        
        return setup_info
    
    def _determine_system_status(self):
        """Determine overall system status based on component states"""
        error_count = sum(1 for comp in self.system_status['components'].values() 
                         if comp.get('status') == 'error')
        total_components = len(self.system_status['components'])
        
        if error_count == 0:
            self.system_status['status'] = 'fully_operational'
        elif error_count < total_components / 2:
            self.system_status['status'] = 'operational_with_degradation'
        else:
            self.system_status['status'] = 'limited_functionality'
        
        completion_time = datetime.now() - self.system_status['initialization_time']
        self.system_status['initialization_duration'] = completion_time.total_seconds()
    
    def _calculate_data_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate data quality score"""
        if df.empty:
            return 0.0
        
        # Basic quality metrics
        completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        consistency = 100  # Simplified - could add more checks
        
        return (completeness + consistency) / 2
    
    def _check_required_columns(self, df: pd.DataFrame) -> bool:
        """Check if required columns are present"""
        required_columns = ['product', 'brand', 'review_text', 'rating']
        return all(col in df.columns for col in required_columns)
    
    def _check_null_percentage(self, df: pd.DataFrame) -> float:
        """Check percentage of null values"""
        return (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    
    def _check_rating_validity(self, df: pd.DataFrame) -> bool:
        """Check if rating values are valid"""
        if 'rating' not in df.columns:
            return False
        
        ratings = df['rating'].dropna()
        if ratings.empty:
            return False
        
        return ratings.between(1, 5).all() or ratings.between(0, 10).all()
    
    def _check_text_content(self, df: pd.DataFrame) -> bool:
        """Check if text content is available"""
        if 'review_text' not in df.columns:
            return False
        
        text_content = df['review_text'].dropna()
        return len(text_content) > 0 and text_content.str.len().mean() > 10
    
    def _determine_system_mode(self) -> str:
        """Determine the operational mode of the system"""
        if (self.component_registry.get('sentiment_analyzer') and 
            self.component_registry.get('recommendation_engine')):
            return 'advanced_ai_mode'
        elif self.component_registry.get('data_preprocessor'):
            return 'standard_mode'
        else:
            return 'basic_mode'
    
    def _generate_capabilities_summary(self) -> List[str]:
        """Generate a summary of system capabilities"""
        capabilities = []
        
        if self.component_registry.get('data_loader'):
            capabilities.append('phone_search_and_analytics')
        
        if self.component_registry.get('data_preprocessor'):
            capabilities.append('text_preprocessing')
        
        if self.component_registry.get('sentiment_analyzer'):
            capabilities.append('advanced_sentiment_analysis')
        else:
            capabilities.append('basic_sentiment_analysis')
        
        if self.component_registry.get('recommendation_engine'):
            capabilities.append('ml_recommendations')
        else:
            capabilities.append('rule_based_recommendations')
        
        return capabilities
    
    # Public API Methods
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return self.system_status.copy()
    
    def get_component_status(self, component_name: str) -> Optional[ComponentStatus]:
        """Get status of a specific component"""
        return self.components.get(component_name)
    
    def get_analyzer(self):
        """Get the initialized review analyzer"""
        return self.component_registry.get('review_analyzer')
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get information about loaded data"""
        return self.system_status.get('data_sources', {})
    
    def is_system_ready(self) -> bool:
        """Check if system is ready for operation"""
        return (self.system_status.get('status') in ['fully_operational', 'operational_with_degradation'] and
                self.component_registry.get('review_analyzer') is not None)
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """Get system health metrics"""
        health_info = {}
        for step_name, step_info in self.system_status.get('components', {}).items():
            if 'result' in step_info and isinstance(step_info['result'], dict):
                if 'overall_health' in step_info['result']:
                    health_info = step_info['result']
                    break
        return health_info
    
    def restart_component(self, component_name: str) -> bool:
        """Restart a specific component"""
        logger.info(f"Restarting component: {component_name}")
        # Implementation would depend on the specific component
        return False