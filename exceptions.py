"""
Custom exceptions for the Zoom Meeting Transcript API
"""

from typing import Any, Dict, Optional
from fastapi import HTTPException, status


class TranscriptAPIException(Exception):
    """Base exception for the transcript API"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class AudioProcessingError(TranscriptAPIException):
    """Exception raised during audio processing"""
    pass


class TranscriptionError(TranscriptAPIException):
    """Exception raised during transcription"""
    pass


class FileValidationError(TranscriptAPIException):
    """Exception raised during file validation"""
    pass


class ModelLoadError(TranscriptAPIException):
    """Exception raised when model fails to load"""
    pass


class ResourceExhaustionError(TranscriptAPIException):
    """Exception raised when system resources are exhausted"""
    pass


# HTTP Exception classes for FastAPI
class BadRequestException(HTTPException):
    """400 Bad Request"""
    
    def __init__(self, detail: str, headers: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
            headers=headers
        )


class UnauthorizedException(HTTPException):
    """401 Unauthorized"""
    
    def __init__(self, detail: str = "Authentication required", 
                 headers: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers=headers
        )


class ForbiddenException(HTTPException):
    """403 Forbidden"""
    
    def __init__(self, detail: str = "Access forbidden", 
                 headers: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail,
            headers=headers
        )


class NotFoundException(HTTPException):
    """404 Not Found"""
    
    def __init__(self, detail: str = "Resource not found", 
                 headers: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=detail,
            headers=headers
        )


class UnprocessableEntityException(HTTPException):
    """422 Unprocessable Entity"""
    
    def __init__(self, detail: str, headers: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=detail,
            headers=headers
        )


class TooManyRequestsException(HTTPException):
    """429 Too Many Requests"""
    
    def __init__(self, detail: str = "Rate limit exceeded", 
                 headers: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=detail,
            headers=headers
        )


class InternalServerErrorException(HTTPException):
    """500 Internal Server Error"""
    
    def __init__(self, detail: str = "Internal server error", 
                 headers: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
            headers=headers
        )


class ServiceUnavailableException(HTTPException):
    """503 Service Unavailable"""
    
    def __init__(self, detail: str = "Service temporarily unavailable", 
                 headers: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
            headers=headers
        )


# Error response models
def create_error_response(
    error_code: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """Create standardized error response"""
    response = {
        "error": {
            "code": error_code,
            "message": message,
            "timestamp": None,  # Will be set by middleware
        }
    }
    
    if details:
        response["error"]["details"] = details
    
    if request_id:
        response["error"]["request_id"] = request_id
    
    return response


# Error code constants
class ErrorCodes:
    """Standard error codes"""
    
    # File validation errors
    INVALID_FILE_TYPE = "INVALID_FILE_TYPE"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    FILE_CORRUPTED = "FILE_CORRUPTED"
    FILE_EMPTY = "FILE_EMPTY"
    
    # Audio processing errors
    AUDIO_PROCESSING_FAILED = "AUDIO_PROCESSING_FAILED"
    AUDIO_FORMAT_UNSUPPORTED = "AUDIO_FORMAT_UNSUPPORTED"
    AUDIO_TOO_LONG = "AUDIO_TOO_LONG"
    AUDIO_TOO_SHORT = "AUDIO_TOO_SHORT"
    
    # Transcription errors
    TRANSCRIPTION_FAILED = "TRANSCRIPTION_FAILED"
    MODEL_NOT_LOADED = "MODEL_NOT_LOADED"
    LANGUAGE_NOT_SUPPORTED = "LANGUAGE_NOT_SUPPORTED"
    
    # System errors
    RESOURCE_EXHAUSTED = "RESOURCE_EXHAUSTED"
    DISK_SPACE_FULL = "DISK_SPACE_FULL"
    MEMORY_LIMIT_EXCEEDED = "MEMORY_LIMIT_EXCEEDED"
    
    # Rate limiting
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    
    # Generic errors
    INTERNAL_ERROR = "INTERNAL_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
