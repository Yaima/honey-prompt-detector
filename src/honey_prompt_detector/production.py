"""
Production-ready wrapper for honey-prompt detector.

Adds caching, rate limiting, health checks, and graceful degradation.
"""

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from src.honey_prompt_detector.main import HoneyPromptSystem
from src.honey_prompt_detector.utils.cache import DetectionCache, RedisCache
from src.honey_prompt_detector.utils.config import Config
from src.honey_prompt_detector.utils.rate_limiter import GlobalRateLimiter, RateLimiter

logger = logging.getLogger("honey_prompt")


class ProductionHoneyPromptSystem:
    """
    Production-ready honey-prompt detection system.

    Features:
    - Request caching (in-memory or Redis)
    - Per-client and global rate limiting
    - Health monitoring
    - Graceful degradation on failures
    - Aggressive timeouts
    - Performance metrics
    """

    def __init__(
        self,
        env_path: Optional[Path] = None,
        enable_cache: bool = True,
        cache_ttl_seconds: int = 3600,
        cache_max_size: int = 1000,
        redis_url: Optional[str] = None,
        enable_rate_limiting: bool = True,
        requests_per_minute: int = 60,
        global_requests_per_minute: int = 1000,
        detection_timeout_seconds: int = 10,
        enable_fallback_mode: bool = True,
    ):
        """
        Initialize production system.

        Args:
            env_path: Path to .env file
            enable_cache: Enable result caching
            cache_ttl_seconds: Cache TTL in seconds
            cache_max_size: Maximum cache entries
            redis_url: Redis URL for distributed caching (optional)
            enable_rate_limiting: Enable rate limiting
            requests_per_minute: Per-client rate limit
            global_requests_per_minute: Global rate limit
            detection_timeout_seconds: Timeout for detection operations
            enable_fallback_mode: Enable graceful degradation on errors
        """
        self.config = Config(env_path)

        # Core system
        self.core_system = HoneyPromptSystem(env_path)

        # Caching
        self.enable_cache = enable_cache
        if enable_cache:
            if redis_url:
                self.cache = RedisCache(redis_url, ttl_seconds=cache_ttl_seconds)
                logger.info("Using Redis cache")
            else:
                self.cache = DetectionCache(max_size=cache_max_size, ttl_seconds=cache_ttl_seconds)
                logger.info("Using in-memory cache")
        else:
            self.cache = None

        # Rate limiting
        self.enable_rate_limiting = enable_rate_limiting
        if enable_rate_limiting:
            self.rate_limiter = RateLimiter(requests_per_minute=requests_per_minute)
            self.global_rate_limiter = GlobalRateLimiter(requests_per_minute=global_requests_per_minute)
            logger.info(
                f"Rate limiting enabled: {requests_per_minute} req/min per client, {global_requests_per_minute} global"
            )
        else:
            self.rate_limiter = None
            self.global_rate_limiter = None

        # Configuration
        self.detection_timeout = detection_timeout_seconds
        self.enable_fallback = enable_fallback_mode

        # Metrics
        self._total_requests = 0
        self._cache_hits = 0
        self._rate_limited = 0
        self._timeouts = 0
        self._errors = 0
        self._fallback_responses = 0

        # Health status
        self._is_healthy = True
        self._last_error_time = None
        self._consecutive_errors = 0

        logger.info("ProductionHoneyPromptSystem initialized")

    async def start(self) -> bool:
        """Start the production system."""
        success = await self.core_system.start()
        if success:
            self._is_healthy = True
            logger.info("Production system started successfully")
        else:
            self._is_healthy = False
            logger.error("Production system failed to start")
        return success

    async def monitor_text(
        self, text: str, client_id: str = "default", expected_detection: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Monitor text with production features.

        Args:
            text: Input text to analyze
            client_id: Unique client identifier for rate limiting
            expected_detection: Expected result for self-tuning (optional)

        Returns:
            Detection result dictionary with additional metadata
        """
        start_time = time.time()
        self._total_requests += 1

        try:
            # Check health status
            if not self._is_healthy and not self.enable_fallback:
                return self._create_error_response("System unhealthy", reason="SYSTEM_UNHEALTHY")

            # Global rate limiting
            if self.global_rate_limiter and not self.global_rate_limiter.is_allowed():
                self._rate_limited += 1
                return self._create_error_response(
                    "Global rate limit exceeded", reason="GLOBAL_RATE_LIMIT", status_code=429
                )

            # Per-client rate limiting
            if self.rate_limiter and not self.rate_limiter.is_allowed(client_id):
                self._rate_limited += 1
                return self._create_error_response(
                    f"Rate limit exceeded for client: {client_id}", reason="CLIENT_RATE_LIMIT", status_code=429
                )

            # Check cache
            if self.cache:
                cached_result = self.cache.get(text)
                if cached_result:
                    self._cache_hits += 1
                    logger.debug("Cache hit - returning cached result")
                    cached_result["cache_hit"] = True
                    cached_result["response_time_ms"] = round((time.time() - start_time) * 1000, 2)
                    return cached_result

            # Perform detection with timeout
            try:
                result = await asyncio.wait_for(
                    self.core_system.monitor_text(text, expected_detection), timeout=self.detection_timeout
                )

                # Add metadata
                result["cache_hit"] = False
                result["response_time_ms"] = round((time.time() - start_time) * 1000, 2)
                result["client_id"] = client_id

                # Cache successful result
                if self.cache and "error" not in result:
                    self.cache.put(text, result)

                # Reset error tracking on success
                self._consecutive_errors = 0
                self._is_healthy = True

                return result

            except asyncio.TimeoutError:
                self._timeouts += 1
                logger.error(f"Detection timeout after {self.detection_timeout}s")

                if self.enable_fallback:
                    return self._create_fallback_response(text, reason="TIMEOUT", elapsed=time.time() - start_time)
                else:
                    return self._create_error_response(
                        f"Detection timeout after {self.detection_timeout}s", reason="TIMEOUT", status_code=504
                    )

        except Exception as e:
            self._errors += 1
            self._consecutive_errors += 1
            self._last_error_time = datetime.now()

            # Mark unhealthy after 5 consecutive errors
            if self._consecutive_errors >= 5:
                self._is_healthy = False
                logger.error("System marked unhealthy after 5 consecutive errors")

            logger.error(f"Error in monitor_text: {str(e)}", exc_info=True)

            if self.enable_fallback:
                return self._create_fallback_response(
                    text, reason="ERROR", error=str(e), elapsed=time.time() - start_time
                )
            else:
                return self._create_error_response(str(e), reason="ERROR", status_code=500)

    def _create_error_response(self, error: str, reason: str, status_code: int = 500) -> Dict[str, Any]:
        """Create standardized error response."""
        return {
            "detection": False,
            "confidence": 0.0,
            "error": error,
            "error_reason": reason,
            "status_code": status_code,
            "timestamp": datetime.now().isoformat(),
        }

    def _create_fallback_response(
        self, text: str, reason: str, error: Optional[str] = None, elapsed: float = 0.0
    ) -> Dict[str, Any]:
        """
        Create fallback response using simple heuristics.

        When main detection fails, use basic pattern matching as fallback.
        """
        self._fallback_responses += 1

        # Simple heuristic detection
        suspicious_patterns = [
            "ignore",
            "disregard",
            "override",
            "bypass",
            "system prompt",
            "hidden",
            "secret",
            "reveal",
            "previous instructions",
            "developer mode",
        ]

        text_lower = text.lower()
        detected = any(pattern in text_lower for pattern in suspicious_patterns)

        return {
            "detection": detected,
            "confidence": 0.6 if detected else 0.1,  # Lower confidence for fallback
            "explanation": f"Fallback mode - {reason}",
            "risk_level": "medium" if detected else "low",
            "fallback": True,
            "fallback_reason": reason,
            "error": error,
            "response_time_ms": round(elapsed * 1000, 2),
            "timestamp": datetime.now().isoformat(),
        }

    async def get_health_status(self) -> Dict[str, Any]:
        """
        Get detailed health status.

        Returns comprehensive health check information.
        """
        core_status = await self.core_system.get_system_status()

        health_status = {
            "healthy": self._is_healthy,
            "core_system_status": core_status.get("status", "unknown"),
            "last_error": self._last_error_time.isoformat() if self._last_error_time else None,
            "consecutive_errors": self._consecutive_errors,
            "metrics": {
                "total_requests": self._total_requests,
                "cache_hits": self._cache_hits,
                "cache_hit_rate_percent": round(
                    (self._cache_hits / self._total_requests * 100) if self._total_requests > 0 else 0, 2
                ),
                "rate_limited_requests": self._rate_limited,
                "timeouts": self._timeouts,
                "errors": self._errors,
                "fallback_responses": self._fallback_responses,
                "error_rate_percent": round(
                    (self._errors / self._total_requests * 100) if self._total_requests > 0 else 0, 2
                ),
            },
        }

        # Add cache stats
        if self.cache:
            health_status["cache_stats"] = self.cache.get_stats()

        # Add rate limiter stats
        if self.rate_limiter:
            health_status["rate_limiter_stats"] = {
                "per_client": self.rate_limiter.get_stats(),
                "global": self.global_rate_limiter.get_stats(),
            }

        # Add core system metrics
        health_status["detection_metrics"] = core_status.get("metrics_summary", {})

        return health_status

    async def reset_health(self) -> None:
        """Reset health status."""
        self._is_healthy = True
        self._consecutive_errors = 0
        self._last_error_time = None
        logger.info("Health status reset")

    async def stop(self) -> None:
        """Stop the production system."""
        logger.info("Stopping production system")

        # Clear cache if needed
        if self.cache:
            logger.info("Clearing cache before shutdown")

        # Stop core system
        await self.core_system.stop()

        logger.info("Production system stopped")

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report."""
        return {
            "total_requests": self._total_requests,
            "cache_performance": {
                "hits": self._cache_hits,
                "hit_rate_percent": round(
                    (self._cache_hits / self._total_requests * 100) if self._total_requests > 0 else 0, 2
                ),
                "enabled": self.enable_cache,
            },
            "rate_limiting": {
                "blocked_requests": self._rate_limited,
                "block_rate_percent": round(
                    (self._rate_limited / self._total_requests * 100) if self._total_requests > 0 else 0, 2
                ),
                "enabled": self.enable_rate_limiting,
            },
            "reliability": {
                "timeouts": self._timeouts,
                "errors": self._errors,
                "fallback_responses": self._fallback_responses,
                "error_rate_percent": round(
                    (self._errors / self._total_requests * 100) if self._total_requests > 0 else 0, 2
                ),
                "healthy": self._is_healthy,
            },
        }
