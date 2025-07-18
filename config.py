"""
Configuration management for the Zoom Meeting Transcript API
"""

import os
from typing import List, Optional
try:
    from pydantic_settings import BaseSettings
    from pydantic import Field, validator
except ImportError:
    # Fallback for older pydantic versions
    from pydantic import BaseSettings, Field, validator
from pathlib import Path


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # API Configuration
    app_name: str = Field(default="Zoom Meeting Transcript API", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    app_description: str = Field(
        default="Upload audio files to get meeting transcripts", 
        env="APP_DESCRIPTION"
    )
    debug: bool = Field(default=False, env="DEBUG")
    
    # Server Configuration
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=1, env="WORKERS")
    
    # Security Configuration
    allowed_hosts: List[str] = Field(default=["*"], env="ALLOWED_HOSTS")
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    max_file_size: int = Field(default=100 * 1024 * 1024, env="MAX_FILE_SIZE")  # 100MB
    
    # Audio Processing Configuration
    target_sample_rate: int = Field(default=16000, env="TARGET_SAMPLE_RATE")
    silence_threshold: int = Field(default=-40, env="SILENCE_THRESHOLD")
    min_silence_duration: int = Field(default=1000, env="MIN_SILENCE_DURATION")
    chunk_size: int = Field(default=300, env="CHUNK_SIZE")
    max_audio_length: int = Field(default=1800, env="MAX_AUDIO_LENGTH")
    
    # Whisper Model Configuration
    whisper_model_name: str = Field(default="base", env="WHISPER_MODEL_NAME")
    whisper_device: str = Field(default="cpu", env="WHISPER_DEVICE")
    
    # File Storage Configuration
    upload_dir: str = Field(default="uploads", env="UPLOAD_DIR")
    temp_dir: Optional[str] = Field(default=None, env="TEMP_DIR")
    
    # Supported file extensions
    allowed_extensions: List[str] = Field(
        default=[".m4a", ".mp4", ".wav", ".mp3", ".flac", ".ogg"],
        env="ALLOWED_EXTENSIONS"
    )
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )
    log_file: Optional[str] = Field(default=None, env="LOG_FILE")
    
    # Rate Limiting
    rate_limit_requests: int = Field(default=10, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=60, env="RATE_LIMIT_WINDOW")  # seconds
    
    # Monitoring and Health Checks
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    health_check_timeout: int = Field(default=30, env="HEALTH_CHECK_TIMEOUT")
    
    @validator("allowed_extensions", pre=True)
    def parse_extensions(cls, v):
        """Parse comma-separated extensions from environment variable"""
        if isinstance(v, str):
            return [ext.strip() for ext in v.split(",")]
        return v
    
    @validator("allowed_hosts", "cors_origins", pre=True)
    def parse_list_strings(cls, v):
        """Parse comma-separated strings from environment variables"""
        if isinstance(v, str):
            return [item.strip() for item in v.split(",")]
        return v
    
    @validator("upload_dir", "temp_dir")
    def create_directories(cls, v):
        """Ensure directories exist"""
        if v:
            Path(v).mkdir(parents=True, exist_ok=True)
        return v
    
    @validator("log_level")
    def validate_log_level(cls, v):
        """Validate log level"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()
    
    @validator("max_file_size")
    def validate_file_size(cls, v):
        """Validate max file size (minimum 1MB, maximum 1GB)"""
        min_size = 1024 * 1024  # 1MB
        max_size = 1024 * 1024 * 1024  # 1GB
        if v < min_size or v > max_size:
            raise ValueError(f"Max file size must be between {min_size} and {max_size} bytes")
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings"""
    return settings


def get_temp_dir() -> str:
    """Get temporary directory path"""
    if settings.temp_dir:
        return settings.temp_dir
    return os.path.join(settings.upload_dir, "temp")


def is_allowed_file_extension(filename: str) -> bool:
    """Check if file extension is allowed"""
    if not filename:
        return False
    
    file_ext = Path(filename).suffix.lower()
    return file_ext in settings.allowed_extensions


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"
