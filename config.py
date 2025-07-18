"""
Configuration management for the Zoom Meeting Transcript API
"""

import os
from typing import List, Optional
from pathlib import Path

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, skip loading .env file
    pass

# Simple configuration class without BaseSettings for compatibility
class Settings:
    """Application settings with environment variable support"""

    def __init__(self):
        # Load from environment variables with defaults

        # API Configuration
        self.app_name = os.getenv("APP_NAME", "Zoom Meeting Transcript API")
        self.app_version = os.getenv("APP_VERSION", "1.0.0")
        self.app_description = os.getenv("APP_DESCRIPTION", "Upload audio files to get meeting transcripts")
        self.debug = os.getenv("DEBUG", "true").lower() == "true"  # Enable debug by default for development

        # Server Configuration
        self.host = os.getenv("HOST", "0.0.0.0")
        self.port = int(os.getenv("PORT", "8000"))
        self.workers = int(os.getenv("WORKERS", "1"))

        # Security Configuration
        self.allowed_hosts = self._parse_list(os.getenv("ALLOWED_HOSTS", "*"))
        self.cors_origins = self._parse_list(os.getenv("CORS_ORIGINS", "*"))
        self.max_file_size = int(os.getenv("MAX_FILE_SIZE", str(100 * 1024 * 1024)))  # 100MB

        # Audio Processing Configuration
        self.target_sample_rate = int(os.getenv("TARGET_SAMPLE_RATE", "16000"))
        self.silence_threshold = int(os.getenv("SILENCE_THRESHOLD", "-40"))
        self.min_silence_duration = int(os.getenv("MIN_SILENCE_DURATION", "1000"))
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "300"))
        self.max_audio_length = int(os.getenv("MAX_AUDIO_LENGTH", "1800"))

        # Whisper Model Configuration
        self.whisper_model_name = os.getenv("WHISPER_MODEL_NAME", "base")
        self.whisper_device = os.getenv("WHISPER_DEVICE", "cpu")

        # File Storage Configuration
        self.upload_dir = os.getenv("UPLOAD_DIR", "uploads")
        self.temp_dir = os.getenv("TEMP_DIR", None)

        # Supported file extensions
        self.allowed_extensions = self._parse_list(
            os.getenv("ALLOWED_EXTENSIONS", ".m4a,.mp4,.wav,.mp3,.flac,.ogg")
        )

        # Logging Configuration
        self.log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        self.log_format = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        self.log_file = os.getenv("LOG_FILE", None)

        # Rate Limiting
        self.rate_limit_requests = int(os.getenv("RATE_LIMIT_REQUESTS", "10"))
        self.rate_limit_window = int(os.getenv("RATE_LIMIT_WINDOW", "60"))

        # Monitoring and Health Checks
        self.enable_metrics = os.getenv("ENABLE_METRICS", "true").lower() == "true"
        self.health_check_timeout = int(os.getenv("HEALTH_CHECK_TIMEOUT", "30"))

        # Create directories
        self._create_directories()

        # Validate settings
        self._validate_settings()

    def _parse_list(self, value: str) -> List[str]:
        """Parse comma-separated string into list"""
        if not value:
            return []
        if value == "*":
            return ["*"]
        return [item.strip() for item in value.split(",") if item.strip()]

    def _create_directories(self):
        """Create necessary directories"""
        if self.upload_dir:
            Path(self.upload_dir).mkdir(parents=True, exist_ok=True)
        if self.temp_dir:
            Path(self.temp_dir).mkdir(parents=True, exist_ok=True)

    def _validate_settings(self):
        """Validate configuration settings"""
        # Validate log level
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")

        # Validate file size
        min_size = 1024 * 1024  # 1MB
        max_size = 1024 * 1024 * 1024  # 1GB
        if self.max_file_size < min_size or self.max_file_size > max_size:
            raise ValueError(f"Max file size must be between {min_size} and {max_size} bytes")



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
