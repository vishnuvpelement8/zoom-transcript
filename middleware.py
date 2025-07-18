"""
Middleware for the Zoom Meeting Transcript API
"""

import time
from typing import Dict, Tuple
from collections import defaultdict, deque
from datetime import datetime, timezone

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from config import get_settings
from exceptions import TooManyRequestsException, create_error_response, ErrorCodes
from logging_config import get_logger, security_logger


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware using sliding window algorithm"""
    
    def __init__(self, app, requests_per_window: int = None, window_seconds: int = None):
        super().__init__(app)
        settings = get_settings()
        self.requests_per_window = requests_per_window or settings.rate_limit_requests
        self.window_seconds = window_seconds or settings.rate_limit_window
        self.client_requests: Dict[str, deque] = defaultdict(deque)
        self.logger = get_logger(__name__)
    
    def get_client_ip(self, request: Request) -> str:
        """Get client IP address, considering proxy headers"""
        # Check for forwarded headers (common in production behind proxies)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # X-Forwarded-For can contain multiple IPs, take the first one
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to direct client IP
        return request.client.host if request.client else "unknown"
    
    def is_rate_limited(self, client_ip: str) -> Tuple[bool, int]:
        """Check if client is rate limited and return remaining requests"""
        current_time = time.time()
        window_start = current_time - self.window_seconds
        
        # Get client's request history
        requests = self.client_requests[client_ip]
        
        # Remove old requests outside the window
        while requests and requests[0] < window_start:
            requests.popleft()
        
        # Check if limit exceeded
        if len(requests) >= self.requests_per_window:
            return True, 0
        
        # Add current request
        requests.append(current_time)
        
        # Return remaining requests
        remaining = self.requests_per_window - len(requests)
        return False, remaining
    
    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting"""
        client_ip = self.get_client_ip(request)
        
        # Skip rate limiting for health check endpoint
        if request.url.path == "/":
            return await call_next(request)
        
        # Check rate limit
        is_limited, remaining = self.is_rate_limited(client_ip)
        
        if is_limited:
            # Log rate limit violation
            security_logger.warning(
                f"Rate limit exceeded for client {client_ip}",
                extra={
                    "client_ip": client_ip,
                    "endpoint": request.url.path,
                    "method": request.method,
                    "user_agent": request.headers.get("User-Agent", "unknown")
                }
            )
            
            # Return rate limit error
            error_response = create_error_response(
                ErrorCodes.RATE_LIMIT_EXCEEDED,
                f"Rate limit exceeded. Maximum {self.requests_per_window} requests per {self.window_seconds} seconds.",
                details={
                    "limit": self.requests_per_window,
                    "window_seconds": self.window_seconds,
                    "retry_after": self.window_seconds
                }
            )
            
            return JSONResponse(
                status_code=429,
                content=error_response,
                headers={
                    "X-RateLimit-Limit": str(self.requests_per_window),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(time.time() + self.window_seconds)),
                    "Retry-After": str(self.window_seconds)
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers to response
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_window)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(time.time() + self.window_seconds))
        
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to responses"""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "font-src 'self'; "
            "connect-src 'self'; "
            "frame-ancestors 'none';"
        )
        
        return response


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Limit request body size"""
    
    def __init__(self, app, max_size: int = None):
        super().__init__(app)
        settings = get_settings()
        self.max_size = max_size or settings.max_file_size
        self.logger = get_logger(__name__)
    
    async def dispatch(self, request: Request, call_next):
        # Check Content-Length header
        content_length = request.headers.get("Content-Length")
        if content_length:
            try:
                size = int(content_length)
                if size > self.max_size:
                    self.logger.warning(
                        f"Request body too large: {size} bytes (max: {self.max_size})",
                        extra={"client_ip": request.client.host if request.client else "unknown"}
                    )
                    
                    error_response = create_error_response(
                        ErrorCodes.FILE_TOO_LARGE,
                        f"Request body too large. Maximum size: {self.max_size} bytes",
                        details={"max_size": self.max_size, "received_size": size}
                    )
                    
                    return JSONResponse(
                        status_code=413,
                        content=error_response
                    )
            except ValueError:
                pass  # Invalid Content-Length header, let it through
        
        return await call_next(request)


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Global error handling middleware"""
    
    def __init__(self, app):
        super().__init__(app)
        self.logger = get_logger(__name__)
    
    async def dispatch(self, request: Request, call_next):
        try:
            return await call_next(request)
        except Exception as e:
            # Log the error
            request_id = getattr(request.state, 'request_id', 'unknown')
            self.logger.error(
                f"Unhandled exception in request {request_id}: {str(e)}",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "url": str(request.url),
                    "client_ip": request.client.host if request.client else "unknown"
                },
                exc_info=True
            )
            
            # Return generic error response
            error_response = create_error_response(
                ErrorCodes.INTERNAL_ERROR,
                "An internal server error occurred",
                request_id=request_id
            )
            error_response["error"]["timestamp"] = datetime.now(timezone.utc).isoformat()
            
            return JSONResponse(
                status_code=500,
                content=error_response
            )
