"""
Tests for the Zoom Meeting Transcript API
"""

import pytest
import tempfile
import os
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from main import app
from config import get_settings


@pytest.fixture
def client():
    """Test client fixture"""
    return TestClient(app)


@pytest.fixture
def test_settings():
    """Test settings fixture"""
    settings = get_settings()
    settings.debug = True
    settings.upload_dir = tempfile.mkdtemp()
    settings.max_file_size = 1024 * 1024  # 1MB for testing
    return settings


@pytest.fixture
def sample_audio_file():
    """Create a sample audio file for testing"""
    # Create a minimal WAV file (44 bytes header + silence)
    wav_header = b'RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00'
    
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_file.write(wav_header)
    temp_file.close()
    
    yield temp_file.name
    
    # Cleanup
    try:
        os.unlink(temp_file.name)
    except:
        pass


class TestHealthCheck:
    """Test health check endpoint"""
    
    def test_health_check_success(self, client):
        """Test successful health check"""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "message" in data
        assert "timestamp" in data
        assert "version" in data
        assert "model_loaded" in data
    
    def test_health_check_headers(self, client):
        """Test health check response headers"""
        response = client.get("/")
        
        # Check security headers
        assert "X-Content-Type-Options" in response.headers
        assert "X-Frame-Options" in response.headers
        assert "X-XSS-Protection" in response.headers
        
        # Check rate limit headers
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers


class TestFileValidation:
    """Test file validation"""
    
    def test_upload_without_file(self, client):
        """Test upload without file"""
        response = client.post("/transcribe")
        assert response.status_code == 422
    
    def test_upload_invalid_file_type(self, client):
        """Test upload with invalid file type"""
        with tempfile.NamedTemporaryFile(suffix='.txt') as temp_file:
            temp_file.write(b"test content")
            temp_file.seek(0)
            
            response = client.post(
                "/transcribe",
                files={"file": ("test.txt", temp_file, "text/plain")}
            )
            assert response.status_code == 400
            
            data = response.json()
            assert "error" in data
            assert "INVALID_FILE_TYPE" in data["error"]["code"] or "Unsupported file type" in data["error"]["message"]
    
    def test_upload_empty_file(self, client):
        """Test upload with empty file"""
        with tempfile.NamedTemporaryFile(suffix='.wav') as temp_file:
            response = client.post(
                "/transcribe",
                files={"file": ("empty.wav", temp_file, "audio/wav")}
            )
            assert response.status_code == 400
    
    @patch('main.whisper_model', None)
    def test_upload_model_not_loaded(self, client, sample_audio_file):
        """Test upload when model is not loaded"""
        with open(sample_audio_file, 'rb') as f:
            response = client.post(
                "/transcribe",
                files={"file": ("test.wav", f, "audio/wav")}
            )
            assert response.status_code == 503


class TestRateLimiting:
    """Test rate limiting"""
    
    def test_rate_limit_health_check_excluded(self, client):
        """Test that health check is excluded from rate limiting"""
        # Make multiple requests to health check
        for _ in range(20):
            response = client.get("/")
            assert response.status_code == 200
    
    def test_rate_limit_transcribe_endpoint(self, client, sample_audio_file):
        """Test rate limiting on transcribe endpoint"""
        # This test would need to be adjusted based on actual rate limits
        # and might require mocking the rate limiter for faster testing
        pass


class TestSecurity:
    """Test security features"""
    
    def test_security_headers_present(self, client):
        """Test that security headers are present"""
        response = client.get("/")
        
        expected_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options", 
            "X-XSS-Protection",
            "Referrer-Policy",
            "Content-Security-Policy"
        ]
        
        for header in expected_headers:
            assert header in response.headers
    
    def test_server_header_hidden(self, client):
        """Test that server information is hidden"""
        response = client.get("/")
        assert "Server" not in response.headers or "uvicorn" not in response.headers.get("Server", "").lower()


class TestConfiguration:
    """Test configuration management"""
    
    def test_settings_loading(self):
        """Test that settings load correctly"""
        settings = get_settings()
        
        assert settings.app_name
        assert settings.app_version
        assert settings.target_sample_rate > 0
        assert settings.max_file_size > 0
        assert len(settings.allowed_extensions) > 0
    
    def test_environment_override(self):
        """Test environment variable override"""
        with patch.dict(os.environ, {"APP_NAME": "Test API"}):
            # This would require reloading settings or using a fresh instance
            pass


class TestErrorHandling:
    """Test error handling"""
    
    def test_404_error(self, client):
        """Test 404 error handling"""
        response = client.get("/nonexistent")
        assert response.status_code == 404
    
    def test_method_not_allowed(self, client):
        """Test method not allowed error"""
        response = client.put("/")
        assert response.status_code == 405


@pytest.mark.integration
class TestTranscriptionIntegration:
    """Integration tests for transcription (requires model)"""
    
    @patch('main.whisper_model')
    def test_transcription_success(self, mock_model, client, sample_audio_file):
        """Test successful transcription"""
        # Mock whisper model response
        mock_result = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 5.0,
                    "text": "Hello world",
                    "avg_logprob": -0.5
                }
            ]
        }
        mock_model.transcribe.return_value = mock_result
        
        with open(sample_audio_file, 'rb') as f:
            response = client.post(
                "/transcribe",
                files={"file": ("test.wav", f, "audio/wav")},
                data={"language": "en"}
            )
        
        # Should return a file download
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    
    @patch('main.whisper_model')
    def test_transcription_with_language(self, mock_model, client, sample_audio_file):
        """Test transcription with different language"""
        mock_result = {"segments": []}
        mock_model.transcribe.return_value = mock_result
        
        with open(sample_audio_file, 'rb') as f:
            response = client.post(
                "/transcribe",
                files={"file": ("test.wav", f, "audio/wav")},
                data={"language": "es"}
            )
        
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
