"""
Custom exceptions and error handling for AI Phone Review Engine
"""

from typing import Optional, Dict, Any
import logging
import traceback
from datetime import datetime

logger = logging.getLogger(__name__)


class ReviewEngineException(Exception):
    """Base exception for all Review Engine errors"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.now()
        super().__init__(self.message)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/API responses"""
        return {
            'error': self.__class__.__name__,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp.isoformat()
        }


class DataNotFoundException(ReviewEngineException):
    """Raised when requested data is not found"""
    pass


class ModelLoadException(ReviewEngineException):
    """Raised when a model fails to load"""
    pass


class InvalidInputException(ReviewEngineException):
    """Raised when input validation fails"""
    pass


class ScrapingException(ReviewEngineException):
    """Raised when web scraping fails"""
    pass


class DatabaseException(ReviewEngineException):
    """Raised when database operations fail"""
    pass


class APIException(ReviewEngineException):
    """Raised when API calls fail"""
    
    def __init__(self, message: str, status_code: Optional[int] = None, 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)
        self.status_code = status_code


class ConfigurationException(ReviewEngineException):
    """Raised when configuration is invalid or missing"""
    pass


class AuthenticationException(ReviewEngineException):
    """Raised when authentication fails"""
    pass


class RateLimitException(ReviewEngineException):
    """Raised when rate limit is exceeded"""
    
    def __init__(self, message: str, retry_after: Optional[int] = None,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)
        self.retry_after = retry_after


class ValidationException(ReviewEngineException):
    """Raised when data validation fails"""
    
    def __init__(self, message: str, field: Optional[str] = None, 
                 value: Any = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)
        self.field = field
        self.value = value
        if field:
            self.details['field'] = field
        if value is not None:
            self.details['value'] = str(value)


class ProcessingException(ReviewEngineException):
    """Raised when data processing fails"""
    pass


class ErrorHandler:
    """Centralized error handler for the application"""
    
    def __init__(self, log_to_file: bool = True, log_file: str = "logs/errors.log"):
        self.log_to_file = log_to_file
        self.log_file = log_file
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup error logging configuration"""
        if self.log_to_file:
            import os
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(logging.ERROR)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Handle an error and return a standardized response
        
        Args:
            error: The exception that occurred
            context: Additional context about the error
            
        Returns:
            Dictionary with error information
        """
        error_id = self._generate_error_id()
        
        # Build error response
        if isinstance(error, ReviewEngineException):
            response = error.to_dict()
            response['error_id'] = error_id
            severity = 'warning'
        else:
            # Handle unexpected errors
            response = {
                'error': error.__class__.__name__,
                'message': str(error),
                'error_id': error_id,
                'timestamp': datetime.now().isoformat()
            }
            severity = 'error'
        
        # Add context if provided
        if context:
            response['context'] = context
        
        # Add traceback for debugging (only in debug mode)
        if logger.isEnabledFor(logging.DEBUG):
            response['traceback'] = traceback.format_exc()
        
        # Log the error
        self._log_error(error, response, severity)
        
        return response
    
    def _log_error(self, error: Exception, response: Dict[str, Any], severity: str):
        """Log error details"""
        log_message = f"Error {response.get('error_id', 'UNKNOWN')}: {error}"
        
        if severity == 'error':
            logger.error(log_message, exc_info=True, extra=response)
        else:
            logger.warning(log_message, extra=response)
    
    def _generate_error_id(self) -> str:
        """Generate a unique error ID for tracking"""
        import uuid
        return f"ERR-{uuid.uuid4().hex[:8].upper()}"
    
    def get_user_friendly_message(self, error: Exception) -> str:
        """
        Get a user-friendly error message
        
        Args:
            error: The exception
            
        Returns:
            User-friendly error message
        """
        error_messages = {
            DataNotFoundException: "The requested data could not be found. Please try again later.",
            ModelLoadException: "There was an issue loading the AI model. Please try again.",
            InvalidInputException: "The provided input is invalid. Please check and try again.",
            ScrapingException: "Unable to fetch data from the web. Please try again later.",
            DatabaseException: "A database error occurred. Please contact support if this persists.",
            APIException: "There was an issue with the API. Please try again later.",
            ConfigurationException: "System configuration error. Please contact support.",
            AuthenticationException: "Authentication failed. Please check your credentials.",
            RateLimitException: "Too many requests. Please wait a moment and try again.",
            ValidationException: "The provided data is invalid. Please check your input.",
            ProcessingException: "An error occurred while processing your request."
        }
        
        for error_type, message in error_messages.items():
            if isinstance(error, error_type):
                return message
        
        # Generic message for unknown errors
        return "An unexpected error occurred. Please try again later."


# Global error handler instance
error_handler = ErrorHandler()


def safe_execute(func):
    """
    Decorator for safe function execution with error handling
    
    Usage:
        @safe_execute
        def my_function():
            # function code
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            context = {
                'function': func.__name__,
                'args': str(args)[:100],  # Limit size
                'kwargs': str(kwargs)[:100]
            }
            error_response = error_handler.handle_error(e, context)
            logger.error(f"Error in {func.__name__}: {error_response}")
            raise
    return wrapper


class InputValidator:
    """Validate and sanitize user inputs"""
    
    @staticmethod
    def validate_review_text(text: str, min_length: int = 10, max_length: int = 10000) -> str:
        """
        Validate and sanitize review text
        
        Args:
            text: Review text to validate
            min_length: Minimum allowed length
            max_length: Maximum allowed length
            
        Returns:
            Sanitized text
            
        Raises:
            ValidationException: If validation fails
        """
        if not text or not isinstance(text, str):
            raise ValidationException("Review text must be a non-empty string", field="review_text")
        
        text = text.strip()
        
        if len(text) < min_length:
            raise ValidationException(
                f"Review text must be at least {min_length} characters",
                field="review_text",
                value=text
            )
        
        if len(text) > max_length:
            raise ValidationException(
                f"Review text must not exceed {max_length} characters",
                field="review_text",
                value=text[:50] + "..."
            )
        
        # Remove potentially harmful content
        import re
        text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
        text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)  # Remove javascript
        
        return text
    
    @staticmethod
    def validate_rating(rating: Any) -> float:
        """
        Validate rating value
        
        Args:
            rating: Rating to validate
            
        Returns:
            Valid rating as float
            
        Raises:
            ValidationException: If validation fails
        """
        try:
            rating = float(rating)
        except (TypeError, ValueError):
            raise ValidationException(
                "Rating must be a number",
                field="rating",
                value=rating
            )
        
        if not 1 <= rating <= 5:
            raise ValidationException(
                "Rating must be between 1 and 5",
                field="rating",
                value=rating
            )
        
        return rating
    
    @staticmethod
    def validate_phone_name(name: str) -> str:
        """
        Validate phone name
        
        Args:
            name: Phone name to validate
            
        Returns:
            Validated phone name
            
        Raises:
            ValidationException: If validation fails
        """
        if not name or not isinstance(name, str):
            raise ValidationException("Phone name must be a non-empty string", field="phone_name")
        
        name = name.strip()
        
        if len(name) < 2:
            raise ValidationException(
                "Phone name too short",
                field="phone_name",
                value=name
            )
        
        if len(name) > 100:
            raise ValidationException(
                "Phone name too long",
                field="phone_name",
                value=name
            )
        
        # Check for suspicious patterns
        import re
        if re.search(r'[<>\"\'%;()&+]', name):
            raise ValidationException(
                "Phone name contains invalid characters",
                field="phone_name",
                value=name
            )
        
        return name
