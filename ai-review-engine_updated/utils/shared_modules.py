"""
Shared Production Module Initializer
Provides centralized initialization of production modules for all apps
"""

import streamlit as st
import logging
from typing import Dict, Any, Optional
from pathlib import Path

# Import production modules
from utils.exceptions import ErrorHandler, ReviewEngineException
from utils.logging_config import LoggingManager
from core.model_manager import ModelManager
from core.robust_analyzer import RobustReviewAnalyzer
from models.review_summarizer import AdvancedReviewSummarizer
from core.smart_search import SmartSearchEngine
from models.market_analyzer import MarketAnalyzer

# Optional modules - import with error handling
try:
    from core.nlp_core import NLPCore
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False
    NLPCore = None

# Configure module availability
PRODUCTION_MODULES = {
    'error_handler': True,
    'logging_manager': True,
    'model_manager': True,
    'robust_analyzer': True,
    'review_summarizer': True,
    'smart_search': True,
    'market_analyzer': True,
    'nlp_core': NLP_AVAILABLE
}


@st.cache_resource
def initialize_production_modules(
    app_name: str = "AI_Review_App",
    log_level: str = "INFO",
    enable_console_logging: bool = False
) -> Dict[str, Any]:
    """
    Initialize all production modules for any app
    
    Args:
        app_name: Name of the application for logging
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        enable_console_logging: Whether to enable console output
    
    Returns:
        Dictionary containing all initialized modules
    """
    modules = {}
    
    # Setup logging first
    logging_manager = LoggingManager()
    logging_manager.setup_logging(
        log_level=log_level,
        console_output=enable_console_logging,
        file_output=True,
        json_format=True,
        colored_console=False  # Disable for Streamlit
    )
    modules['logging_manager'] = logging_manager
    
    # Log initialization
    logger = logging.getLogger(app_name)
    logger.info(f"Initializing production modules for {app_name}")
    
    # Initialize error handler
    try:
        modules['error_handler'] = ErrorHandler(log_to_file=True)
        logger.info("✅ Error handler initialized")
    except Exception as e:
        logger.error(f"Failed to initialize error handler: {e}")
        modules['error_handler'] = None
    
    # Initialize model manager (singleton)
    try:
        modules['model_manager'] = ModelManager()
        logger.info("✅ Model manager initialized")
    except Exception as e:
        logger.error(f"Failed to initialize model manager: {e}")
        modules['model_manager'] = None
    
    # Initialize robust analyzer
    try:
        modules['robust_analyzer'] = RobustReviewAnalyzer()
        logger.info("✅ Robust analyzer initialized")
    except Exception as e:
        logger.error(f"Failed to initialize robust analyzer: {e}")
        modules['robust_analyzer'] = None
    
    # Initialize review summarizer
    try:
        modules['review_summarizer'] = AdvancedReviewSummarizer()
        logger.info("✅ Review summarizer initialized")
    except Exception as e:
        logger.error(f"Failed to initialize review summarizer: {e}")
        modules['review_summarizer'] = None
    
    # Initialize smart search
    try:
        modules['smart_search'] = SmartSearchEngine()
        logger.info("✅ Smart search engine initialized")
    except Exception as e:
        logger.error(f"Failed to initialize smart search: {e}")
        modules['smart_search'] = None
    
    # Initialize market analyzer
    try:
        modules['market_analyzer'] = MarketAnalyzer()
        logger.info("✅ Market analyzer initialized")
    except Exception as e:
        logger.error(f"Failed to initialize market analyzer: {e}")
        modules['market_analyzer'] = None
    
    # Initialize NLP core if available
    if NLP_AVAILABLE:
        try:
            modules['nlp_core'] = NLPCore()
            logger.info("✅ NLP core initialized")
        except Exception as e:
            logger.error(f"Failed to initialize NLP core: {e}")
            modules['nlp_core'] = None
    else:
        modules['nlp_core'] = None
        logger.warning("NLP core not available")
    
    logger.info(f"Production modules initialization complete for {app_name}")
    return modules


def handle_error(error: Exception, context: Optional[Dict[str, Any]] = None) -> str:
    """
    Universal error handler for all apps
    
    Args:
        error: The exception to handle
        context: Optional context information
    
    Returns:
        User-friendly error message
    """
    if 'prod_modules' in st.session_state and st.session_state.prod_modules.get('error_handler'):
        error_handler = st.session_state.prod_modules['error_handler']
        error_response = error_handler.handle_error(error, context)
        return error_handler.get_user_friendly_message(error)
    else:
        # Fallback if error handler not initialized
        logging.error(f"Error: {error}", exc_info=True)
        return "An error occurred. Please try again later."


def safe_execute(func, *args, **kwargs):
    """
    Safely execute a function with error handling
    
    Args:
        func: Function to execute
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function
    
    Returns:
        Function result or None if error
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        error_msg = handle_error(e, {'function': func.__name__})
        st.error(f"⚠️ {error_msg}")
        return None


def get_module(module_name: str) -> Optional[Any]:
    """
    Get a specific module from session state
    
    Args:
        module_name: Name of the module to retrieve
    
    Returns:
        Module instance or None if not available
    """
    if 'prod_modules' in st.session_state:
        return st.session_state.prod_modules.get(module_name)
    return None


def check_module_status() -> Dict[str, bool]:
    """
    Check the status of all production modules
    
    Returns:
        Dictionary with module availability status
    """
    status = {}
    if 'prod_modules' in st.session_state:
        for module_name, expected in PRODUCTION_MODULES.items():
            module = st.session_state.prod_modules.get(module_name)
            status[module_name] = module is not None
    else:
        status = {name: False for name in PRODUCTION_MODULES.keys()}
    
    return status


def display_module_status():
    """
    Display module status in Streamlit sidebar
    """
    st.sidebar.header("🔧 System Status")
    status = check_module_status()
    
    for module_name, is_active in status.items():
        display_name = module_name.replace('_', ' ').title()
        if is_active:
            st.sidebar.success(f"✅ {display_name}")
        else:
            st.sidebar.warning(f"⚠️ {display_name}")


# Decorator for automatic error handling
def with_error_handling(func):
    """
    Decorator to add automatic error handling to functions
    
    Usage:
        @with_error_handling
        def my_function():
            # function code
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = handle_error(e, {'function': func.__name__})
            st.error(f"⚠️ {error_msg}")
            return None
    return wrapper


# Quick initialization for apps
def quick_init(app_name: str = "AI_App"):
    """
    Quick initialization of production modules for any app
    
    Usage:
        from utils.shared_modules import quick_init
        quick_init("My App Name")
    """
    if 'prod_modules' not in st.session_state:
        with st.spinner("Initializing production modules..."):
            st.session_state.prod_modules = initialize_production_modules(app_name)
            st.success("✅ Production modules ready!")
    return st.session_state.prod_modules