"""
Logging configuration for the Zoom Meeting Transcript API
"""

import logging
import logging.handlers
import sys
from typing import Dict, Any
from pathlib import Path
import json
from datetime import datetime

from config import get_settings


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add extra fields if present
        if hasattr(record, "request_id"):
            log_entry["request_id"] = record.request_id
        
        if hasattr(record, "user_id"):
            log_entry["user_id"] = record.user_id
        
        if hasattr(record, "duration"):
            log_entry["duration"] = record.duration
        
        if hasattr(record, "file_size"):
            log_entry["file_size"] = record.file_size
        
        if hasattr(record, "file_type"):
            log_entry["file_type"] = record.file_type
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)


def setup_logging() -> None:
    """Setup application logging configuration"""
    settings = get_settings()
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.log_level))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, settings.log_level))
    
    # Use JSON formatter for production, simple formatter for development
    if settings.debug:
        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    else:
        console_formatter = JSONFormatter()
    
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if log file is specified
    if settings.log_file:
        log_file_path = Path(settings.log_file)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            settings.log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(getattr(logging, settings.log_level))
        file_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(file_handler)
    
    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    
    # Suppress noisy third-party loggers in production
    if not settings.debug:
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("PIL").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance"""
    return logging.getLogger(name)


class LoggerMixin:
    """Mixin class to add logging capabilities"""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class"""
        return logging.getLogger(self.__class__.__name__)


def log_request_start(logger: logging.Logger, request_id: str, endpoint: str, 
                     method: str, file_size: int = None, file_type: str = None) -> None:
    """Log request start"""
    extra = {"request_id": request_id}
    if file_size:
        extra["file_size"] = file_size
    if file_type:
        extra["file_type"] = file_type
    
    logger.info(f"Request started: {method} {endpoint}", extra=extra)


def log_request_end(logger: logging.Logger, request_id: str, endpoint: str, 
                   method: str, status_code: int, duration: float) -> None:
    """Log request completion"""
    extra = {
        "request_id": request_id,
        "duration": duration,
        "status_code": status_code
    }
    
    logger.info(f"Request completed: {method} {endpoint} - {status_code}", extra=extra)


def log_error(logger: logging.Logger, request_id: str, error: Exception, 
             context: str = None) -> None:
    """Log error with context"""
    extra = {"request_id": request_id}
    
    message = f"Error occurred: {str(error)}"
    if context:
        message = f"Error in {context}: {str(error)}"
    
    logger.error(message, extra=extra, exc_info=True)


def log_performance_metric(logger: logging.Logger, metric_name: str, 
                          value: float, unit: str = "seconds", 
                          request_id: str = None) -> None:
    """Log performance metrics"""
    extra = {"metric_name": metric_name, "metric_value": value, "unit": unit}
    if request_id:
        extra["request_id"] = request_id
    
    logger.info(f"Performance metric: {metric_name} = {value} {unit}", extra=extra)


# Application-specific loggers
app_logger = get_logger("transcript_api")
audio_logger = get_logger("audio_processing")
whisper_logger = get_logger("whisper_transcription")
security_logger = get_logger("security")
performance_logger = get_logger("performance")
