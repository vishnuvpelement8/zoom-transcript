"""
Pydantic models for request/response validation
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum


class LanguageCode(str, Enum):
    """Supported language codes for transcription"""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    JAPANESE = "ja"
    KOREAN = "ko"
    CHINESE = "zh"
    DUTCH = "nl"
    ARABIC = "ar"
    HINDI = "hi"


class TranscriptionRequest(BaseModel):
    """Request model for transcription endpoint"""
    language: LanguageCode = Field(
        default=LanguageCode.ENGLISH,
        description="Language code for transcription"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "language": "en"
            }
        }


class HealthStatus(str, Enum):
    """Health check status values"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"


class HealthCheckResponse(BaseModel):
    """Health check response model"""
    status: HealthStatus
    message: str
    timestamp: datetime
    version: str
    model_loaded: bool
    uptime_seconds: Optional[float] = None
    checks: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "message": "Zoom Meeting Transcript API is running",
                "timestamp": "2023-12-07T10:30:00Z",
                "version": "1.0.0",
                "model_loaded": True,
                "uptime_seconds": 3600.5,
                "checks": {
                    "whisper_model": "loaded",
                    "disk_space": "sufficient",
                    "memory": "normal"
                }
            }
        }


class TranscriptionSegment(BaseModel):
    """Individual transcription segment"""
    start: float = Field(description="Start time in seconds")
    end: float = Field(description="End time in seconds")
    text: str = Field(description="Transcribed text")
    speaker: str = Field(description="Speaker identifier")
    confidence: Optional[float] = Field(description="Confidence score")
    
    @validator("start", "end")
    def validate_timestamps(cls, v):
        if v < 0:
            raise ValueError("Timestamps must be non-negative")
        return v
    
    @validator("text")
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError("Text cannot be empty")
        return v.strip()


class TranscriptionStats(BaseModel):
    """Statistics about the transcription"""
    total_duration: float = Field(description="Total audio duration in seconds")
    total_segments: int = Field(description="Number of transcription segments")
    speakers_detected: int = Field(description="Number of speakers detected")
    processing_time: float = Field(description="Processing time in seconds")
    audio_file_size: int = Field(description="Original audio file size in bytes")
    language_detected: Optional[str] = Field(description="Detected language code")
    confidence_avg: Optional[float] = Field(description="Average confidence score")


class TranscriptionResponse(BaseModel):
    """Response model for transcription endpoint"""
    request_id: str = Field(description="Unique request identifier")
    filename: str = Field(description="Original filename")
    status: str = Field(description="Processing status")
    created_at: datetime = Field(description="Processing timestamp")
    stats: TranscriptionStats
    segments: List[TranscriptionSegment] = Field(description="Transcription segments")
    
    class Config:
        schema_extra = {
            "example": {
                "request_id": "550e8400-e29b-41d4-a716-446655440000",
                "filename": "meeting_recording.m4a",
                "status": "completed",
                "created_at": "2023-12-07T10:30:00Z",
                "stats": {
                    "total_duration": 1800.5,
                    "total_segments": 45,
                    "speakers_detected": 2,
                    "processing_time": 120.3,
                    "audio_file_size": 52428800,
                    "language_detected": "en",
                    "confidence_avg": 0.85
                },
                "segments": [
                    {
                        "start": 0.0,
                        "end": 5.2,
                        "text": "Welcome to today's meeting.",
                        "speaker": "Speaker 1",
                        "confidence": 0.92
                    }
                ]
            }
        }


class ErrorDetail(BaseModel):
    """Error detail model"""
    code: str = Field(description="Error code")
    message: str = Field(description="Error message")
    field: Optional[str] = Field(description="Field that caused the error")
    value: Optional[Any] = Field(description="Invalid value")


class ErrorResponse(BaseModel):
    """Standard error response model"""
    error: Dict[str, Any] = Field(description="Error information")
    request_id: Optional[str] = Field(description="Request identifier")
    timestamp: datetime = Field(description="Error timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "error": {
                    "code": "INVALID_FILE_TYPE",
                    "message": "Unsupported file type. Allowed: .m4a, .mp4, .wav, .mp3, .flac, .ogg",
                    "details": {
                        "file_extension": ".txt",
                        "allowed_extensions": [".m4a", ".mp4", ".wav", ".mp3", ".flac", ".ogg"]
                    }
                },
                "request_id": "550e8400-e29b-41d4-a716-446655440000",
                "timestamp": "2023-12-07T10:30:00Z"
            }
        }


class FileUploadInfo(BaseModel):
    """File upload information"""
    filename: str
    size: int
    content_type: str
    extension: str
    
    @validator("size")
    def validate_size(cls, v):
        if v <= 0:
            raise ValueError("File size must be positive")
        return v
    
    @validator("filename")
    def validate_filename(cls, v):
        if not v or not v.strip():
            raise ValueError("Filename cannot be empty")
        return v.strip()


class ProcessingStatus(str, Enum):
    """Processing status values"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobStatus(BaseModel):
    """Job status model for async processing"""
    job_id: str
    status: ProcessingStatus
    created_at: datetime
    updated_at: datetime
    progress: Optional[float] = Field(ge=0, le=100, description="Progress percentage")
    message: Optional[str] = Field(description="Status message")
    result: Optional[Dict[str, Any]] = Field(description="Processing result")
    error: Optional[ErrorDetail] = Field(description="Error information if failed")
    
    class Config:
        schema_extra = {
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "processing",
                "created_at": "2023-12-07T10:30:00Z",
                "updated_at": "2023-12-07T10:32:00Z",
                "progress": 65.5,
                "message": "Transcribing audio segments...",
                "result": None,
                "error": None
            }
        }
