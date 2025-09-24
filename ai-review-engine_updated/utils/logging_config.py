"""
Comprehensive logging configuration for AI Phone Review Engine
Provides structured logging with multiple handlers and formatters
"""

import logging
import logging.handlers
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import os


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'filename', 'funcName',
                          'levelname', 'levelno', 'lineno', 'module', 'msecs',
                          'message', 'pathname', 'process', 'processName',
                          'relativeCreated', 'stack_info', 'thread', 'threadName',
                          'exc_info', 'exc_text']:
                log_data[key] = value
        
        return json.dumps(log_data)


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        """Format with colors for console"""
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


class LoggingManager:
    """Centralized logging manager for the application"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize logging manager
        
        Args:
            config_path: Path to logging configuration file
        """
        self.config_path = config_path
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        self.loggers: Dict[str, logging.Logger] = {}
        
    def setup_logging(self, 
                     log_level: str = "INFO",
                     console_output: bool = True,
                     file_output: bool = True,
                     json_format: bool = False,
                     colored_console: bool = True) -> None:
        """
        Setup comprehensive logging configuration
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            console_output: Enable console logging
            file_output: Enable file logging
            json_format: Use JSON format for file logging
            colored_console: Use colored output for console
        """
        # Create root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper()))
        
        # Remove existing handlers
        root_logger.handlers = []
        
        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, log_level.upper()))
            
            if colored_console and not json_format:
                console_formatter = ColoredFormatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
            else:
                console_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
            
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)
        
        # File handlers
        if file_output:
            # Main application log
            self._setup_file_handler(
                root_logger,
                self.log_dir / "app.log",
                log_level,
                json_format
            )
            
            # Error log
            self._setup_file_handler(
                root_logger,
                self.log_dir / "errors.log",
                "ERROR",
                json_format
            )
            
            # Debug log (if in debug mode)
            if log_level == "DEBUG":
                self._setup_file_handler(
                    root_logger,
                    self.log_dir / "debug.log",
                    "DEBUG",
                    json_format
                )
        
        # Setup specialized loggers
        self._setup_specialized_loggers(log_level, json_format)
        
        logging.info("Logging system initialized")
    
    def _setup_file_handler(self,
                           logger: logging.Logger,
                           filename: Path,
                           level: str,
                           json_format: bool,
                           max_bytes: int = 10485760,  # 10MB
                           backup_count: int = 5) -> None:
        """
        Setup rotating file handler
        
        Args:
            logger: Logger to add handler to
            filename: Log file path
            level: Logging level
            json_format: Use JSON formatting
            max_bytes: Maximum file size before rotation
            backup_count: Number of backup files to keep
        """
        file_handler = logging.handlers.RotatingFileHandler(
            filename,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(getattr(logging, level.upper()))
        
        if json_format:
            formatter = JSONFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    def _setup_specialized_loggers(self, log_level: str, json_format: bool) -> None:
        """Setup specialized loggers for different components"""
        
        # Performance logger
        perf_logger = self.get_logger("performance")
        self._setup_file_handler(
            perf_logger,
            self.log_dir / "performance.log",
            log_level,
            json_format
        )
        
        # Security logger
        security_logger = self.get_logger("security")
        self._setup_file_handler(
            security_logger,
            self.log_dir / "security.log",
            "INFO",
            json_format
        )
        
        # API logger
        api_logger = self.get_logger("api")
        self._setup_file_handler(
            api_logger,
            self.log_dir / "api.log",
            log_level,
            json_format
        )
        
        # Database logger
        db_logger = self.get_logger("database")
        self._setup_file_handler(
            db_logger,
            self.log_dir / "database.log",
            log_level,
            json_format
        )
        
        # Model logger
        model_logger = self.get_logger("models")
        self._setup_file_handler(
            model_logger,
            self.log_dir / "models.log",
            log_level,
            json_format
        )
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        Get or create a logger
        
        Args:
            name: Logger name
            
        Returns:
            Logger instance
        """
        if name not in self.loggers:
            self.loggers[name] = logging.getLogger(name)
        return self.loggers[name]
    
    def log_performance(self, 
                       operation: str,
                       duration: float,
                       details: Optional[Dict[str, Any]] = None) -> None:
        """
        Log performance metrics
        
        Args:
            operation: Operation name
            duration: Duration in seconds
            details: Additional details
        """
        perf_logger = self.get_logger("performance")
        perf_data = {
            'operation': operation,
            'duration_seconds': duration,
            'timestamp': datetime.utcnow().isoformat()
        }
        if details:
            perf_data.update(details)
        
        perf_logger.info(f"Performance: {operation}", extra=perf_data)
    
    def log_api_request(self,
                       method: str,
                       path: str,
                       status_code: int,
                       duration: float,
                       user_id: Optional[str] = None) -> None:
        """
        Log API request
        
        Args:
            method: HTTP method
            path: Request path
            status_code: Response status code
            duration: Request duration
            user_id: User identifier
        """
        api_logger = self.get_logger("api")
        api_logger.info(
            f"{method} {path} - {status_code}",
            extra={
                'method': method,
                'path': path,
                'status_code': status_code,
                'duration': duration,
                'user_id': user_id
            }
        )
    
    def log_security_event(self,
                          event_type: str,
                          message: str,
                          user_id: Optional[str] = None,
                          ip_address: Optional[str] = None,
                          details: Optional[Dict[str, Any]] = None) -> None:
        """
        Log security event
        
        Args:
            event_type: Type of security event
            message: Event message
            user_id: User identifier
            ip_address: IP address
            details: Additional details
        """
        security_logger = self.get_logger("security")
        event_data = {
            'event_type': event_type,
            'user_id': user_id,
            'ip_address': ip_address,
            'timestamp': datetime.utcnow().isoformat()
        }
        if details:
            event_data.update(details)
        
        security_logger.warning(f"Security Event: {message}", extra=event_data)
    
    def get_log_stats(self) -> Dict[str, Any]:
        """
        Get logging statistics
        
        Returns:
            Dictionary with log file statistics
        """
        stats = {}
        for log_file in self.log_dir.glob("*.log"):
            stats[log_file.name] = {
                'size_bytes': log_file.stat().st_size,
                'size_mb': log_file.stat().st_size / (1024 * 1024),
                'modified': datetime.fromtimestamp(log_file.stat().st_mtime).isoformat()
            }
        return stats
    
    def cleanup_old_logs(self, days: int = 30) -> None:
        """
        Clean up old log files
        
        Args:
            days: Number of days to keep logs
        """
        cutoff_time = datetime.now().timestamp() - (days * 24 * 60 * 60)
        
        for log_file in self.log_dir.glob("*.log.*"):  # Rotated logs
            if log_file.stat().st_mtime < cutoff_time:
                try:
                    log_file.unlink()
                    logging.info(f"Deleted old log file: {log_file}")
                except Exception as e:
                    logging.error(f"Failed to delete {log_file}: {e}")


# Global logging manager instance
logging_manager = LoggingManager()


def setup_default_logging():
    """Setup default logging configuration"""
    # Check for environment variables
    log_level = os.getenv('LOG_LEVEL', 'INFO')
    log_file = os.getenv('LOG_FILE', 'logs/app.log')
    
    logging_manager.setup_logging(
        log_level=log_level,
        console_output=True,
        file_output=True,
        json_format=False,
        colored_console=True
    )


# Performance monitoring decorator
def log_execution_time(func):
    """
    Decorator to log function execution time
    
    Usage:
        @log_execution_time
        def my_function():
            # function code
    """
    import time
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logging_manager.log_performance(
                operation=f"{func.__module__}.{func.__name__}",
                duration=duration,
                details={'status': 'success'}
            )
            return result
        except Exception as e:
            duration = time.time() - start_time
            logging_manager.log_performance(
                operation=f"{func.__module__}.{func.__name__}",
                duration=duration,
                details={'status': 'error', 'error': str(e)}
            )
            raise
    return wrapper


# Initialize default logging on import
if __name__ != "__main__":
    setup_default_logging()
