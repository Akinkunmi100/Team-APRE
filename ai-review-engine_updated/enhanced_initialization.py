"""
Enhanced System Initialization
Integration layer between the System Orchestrator and Flask Application
"""

import logging
from typing import Dict, Any, Optional
from utils.system_orchestrator import SystemOrchestrator

logger = logging.getLogger(__name__)

# Global orchestrator instance
_orchestrator = None
_initialization_status = {
    'initialized': False,
    'status': 'not_started',
    'error': None,
    'orchestrator_available': False
}

def initialize_enhanced_system(config: Dict = None) -> Dict[str, Any]:
    """
    Initialize the enhanced system with orchestrator
    Similar to your expected pattern but adapted for Flask
    """
    global _orchestrator, _initialization_status
    
    logger.info("🚀 Initializing Enhanced AI Phone Review System...")
    
    try:
        # Check if already initialized
        if _initialization_status.get('initialized', False):
            logger.info("✅ System already initialized")
            return {
                'success': True,
                'status': 'already_initialized',
                'orchestrator': _orchestrator,
                'system_info': _orchestrator.get_system_status() if _orchestrator else {}
            }
        
        # Start initialization process
        _initialization_status['status'] = 'initializing'
        
        # Create the system orchestrator (this will run all initialization steps)
        _orchestrator = SystemOrchestrator(config)
        
        # Check if initialization was successful
        if _orchestrator.is_system_ready():
            _initialization_status.update({
                'initialized': True,
                'status': 'fully_operational',
                'orchestrator_available': True,
                'error': None
            })
            
            system_status = _orchestrator.get_system_status()
            data_info = _orchestrator.get_data_info()
            health_metrics = _orchestrator.get_health_metrics()
            
            logger.info("✅ Enhanced system initialization completed successfully!")
            logger.info(f"📊 Data Source: {data_info.get('primary_source', 'unknown')}")
            logger.info(f"📈 Total Records: {data_info.get('total_records', 0):,}")
            logger.info(f"🔧 System Health: {health_metrics.get('overall_health', 0):.1f}%")
            logger.info(f"⚡ System Mode: {system_status.get('status', 'unknown')}")
            
            return {
                'success': True,
                'status': 'initialization_complete',
                'orchestrator': _orchestrator,
                'system_info': {
                    'data_source': data_info.get('primary_source'),
                    'data_info': f"{data_info.get('total_records', 0):,} phone reviews loaded",
                    'preprocessing_status': 'completed',
                    'ai_components': 'loaded' if health_metrics.get('overall_health', 0) > 80 else 'partial',
                    'system_health': health_metrics.get('overall_health', 0),
                    'capabilities': system_status.get('components', {}).get('🎯 Finalizing System Setup', {}).get('result', {}).get('capabilities_summary', [])
                }
            }
        
        else:
            # System initialized with degradation
            _initialization_status.update({
                'initialized': True,
                'status': 'operational_with_degradation',
                'orchestrator_available': True,
                'error': None
            })
            
            system_status = _orchestrator.get_system_status()
            data_info = _orchestrator.get_data_info()
            
            logger.warning("⚠️ System initialized with limited functionality")
            logger.info(f"📊 Data Source: {data_info.get('primary_source', 'unknown')}")
            logger.info(f"📈 Total Records: {data_info.get('total_records', 0):,}")
            
            return {
                'success': True,
                'status': 'limited_functionality',
                'orchestrator': _orchestrator,
                'system_info': {
                    'data_source': data_info.get('primary_source'),
                    'data_info': f"{data_info.get('total_records', 0):,} phone reviews loaded",
                    'preprocessing_status': 'partial',
                    'ai_components': 'limited',
                    'warnings': ['Some components failed to initialize']
                },
                'warnings': ['System running with reduced capabilities']
            }
    
    except Exception as e:
        # Critical initialization failure
        error_msg = f"Critical error during system initialization: {str(e)}"
        logger.error(error_msg)
        
        _initialization_status.update({
            'initialized': False,
            'status': 'critical_error',
            'orchestrator_available': False,
            'error': str(e)
        })
        
        return {
            'success': False,
            'status': 'initialization_failed',
            'error': error_msg,
            'fallback_available': True,
            'system_info': {
                'data_source': 'fallback_data',
                'data_info': 'Using minimal fallback system',
                'preprocessing_status': 'disabled',
                'ai_components': 'disabled'
            }
        }

def get_orchestrator() -> Optional[SystemOrchestrator]:
    """Get the global orchestrator instance"""
    return _orchestrator

def get_analyzer():
    """Get the initialized analyzer from orchestrator"""
    if _orchestrator and _orchestrator.is_system_ready():
        return _orchestrator.get_analyzer()
    return None

def get_system_status() -> Dict[str, Any]:
    """Get current system status"""
    if _orchestrator:
        return {
            'orchestrator_status': _orchestrator.get_system_status(),
            'initialization_status': _initialization_status,
            'system_ready': _orchestrator.is_system_ready(),
            'health_metrics': _orchestrator.get_health_metrics(),
            'data_info': _orchestrator.get_data_info()
        }
    
    return {
        'orchestrator_status': None,
        'initialization_status': _initialization_status,
        'system_ready': False,
        'health_metrics': {},
        'data_info': {}
    }

def is_system_ready() -> bool:
    """Check if system is ready for operation"""
    return (_orchestrator is not None and 
            _orchestrator.is_system_ready() and 
            _initialization_status.get('initialized', False))

def create_fallback_analyzer():
    """Create a fallback analyzer if orchestrator fails"""
    logger.warning("Creating fallback analyzer - limited functionality")
    
    try:
        # Import the basic analyzer
        from ultimate_web_app import UltimateReviewAnalyzer
        
        # Create basic analyzer without orchestrator
        analyzer = UltimateReviewAnalyzer()
        
        logger.info("✅ Fallback analyzer created")
        return analyzer
        
    except Exception as e:
        logger.error(f"Failed to create fallback analyzer: {e}")
        return None

def get_initialization_summary() -> Dict[str, Any]:
    """Get a summary of the initialization process"""
    if not _orchestrator:
        return {
            'status': 'not_initialized',
            'summary': 'System has not been initialized'
        }
    
    system_status = _orchestrator.get_system_status()
    data_info = _orchestrator.get_data_info()
    health_metrics = _orchestrator.get_health_metrics()
    
    # Count successful vs failed components
    components = system_status.get('components', {})
    successful = len([c for c in components.values() if c.get('status') == 'completed'])
    failed = len([c for c in components.values() if c.get('status') == 'error'])
    total = len(components)
    
    return {
        'status': system_status.get('status'),
        'initialization_time': system_status.get('initialization_duration', 0),
        'components': {
            'total': total,
            'successful': successful,
            'failed': failed,
            'success_rate': (successful / total * 100) if total > 0 else 0
        },
        'data': {
            'source': data_info.get('primary_source'),
            'records': data_info.get('total_records', 0),
            'quality_score': data_info.get('data_quality_score', 0)
        },
        'health': {
            'overall_health': health_metrics.get('overall_health', 0),
            'system_ready': _orchestrator.is_system_ready(),
            'component_health': health_metrics.get('component_health', {})
        },
        'capabilities': system_status.get('components', {}).get('🎯 Finalizing System Setup', {}).get('result', {}).get('capabilities_summary', []),
        'errors': system_status.get('error_log', [])
    }

# Integration with Flask app logging
def setup_enhanced_logging():
    """Setup enhanced logging for the system"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Add specific loggers for different components
    loggers = [
        'utils.system_orchestrator',
        'utils.data_preprocessing', 
        'utils.preprocessed_data_loader',
        'utils.data_quality_validator',
        'enhanced_initialization',
        'ultimate_web_app'
    ]
    
    for logger_name in loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)