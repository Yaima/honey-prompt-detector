"""
Production-ready rate limiting for honey-prompt detector.

Implements token bucket algorithm for rate limiting requests.
"""

import logging
import time
from collections import defaultdict
from threading import Lock
from typing import Any, Dict, Optional

logger = logging.getLogger("honey_prompt")


class RateLimiter:
    """
    Token bucket rate limiter.

    Limits requests per client to prevent abuse and manage API costs.
    """

    def __init__(self, requests_per_minute: int = 60, burst_size: Optional[int] = None):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests allowed per minute per client
            burst_size: Maximum burst size (defaults to requests_per_minute)
        """
        self.rate = requests_per_minute / 60.0  # requests per second
        self.burst_size = burst_size or requests_per_minute
        self._buckets: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"tokens": self.burst_size, "last_update": time.time()}
        )
        self._lock = Lock()
        self._blocked_count = 0
        self._allowed_count = 0

        logger.info(f"Initialized rate limiter: {requests_per_minute} req/min, " f"burst={self.burst_size}")

    def _refill_bucket(self, bucket: Dict[str, float]) -> None:
        """Refill tokens in bucket based on elapsed time."""
        now = time.time()
        elapsed = now - bucket["last_update"]
        bucket["tokens"] = min(self.burst_size, bucket["tokens"] + (elapsed * self.rate))
        bucket["last_update"] = now

    def is_allowed(self, client_id: str) -> bool:
        """
        Check if request is allowed for client.

        Args:
            client_id: Unique identifier for client (e.g., IP, user ID, API key)

        Returns:
            True if request is allowed, False if rate limit exceeded
        """
        with self._lock:
            bucket = self._buckets[client_id]
            self._refill_bucket(bucket)

            if bucket["tokens"] >= 1.0:
                bucket["tokens"] -= 1.0
                self._allowed_count += 1
                return True
            else:
                self._blocked_count += 1
                logger.warning(f"Rate limit exceeded for client: {client_id}")
                return False

    def reset_client(self, client_id: str) -> None:
        """Reset rate limit for specific client."""
        with self._lock:
            if client_id in self._buckets:
                del self._buckets[client_id]
                logger.info(f"Rate limit reset for client: {client_id}")

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        with self._lock:
            return {
                "active_clients": len(self._buckets),
                "requests_allowed": self._allowed_count,
                "requests_blocked": self._blocked_count,
                "block_rate_percent": round(
                    (
                        (self._blocked_count / (self._allowed_count + self._blocked_count) * 100)
                        if (self._allowed_count + self._blocked_count) > 0
                        else 0
                    ),
                    2,
                ),
            }

    def cleanup_stale_clients(self, stale_seconds: int = 3600) -> int:
        """
        Remove stale client buckets.

        Args:
            stale_seconds: Remove clients inactive for this many seconds

        Returns:
            Number of clients removed
        """
        with self._lock:
            now = time.time()
            stale_clients = [
                client_id
                for client_id, bucket in self._buckets.items()
                if (now - bucket["last_update"]) > stale_seconds
            ]

            for client_id in stale_clients:
                del self._buckets[client_id]

            if stale_clients:
                logger.info(f"Cleaned up {len(stale_clients)} stale clients")

            return len(stale_clients)


class GlobalRateLimiter:
    """
    Global rate limiter for entire system.

    Limits total requests across all clients to manage API costs.
    """

    def __init__(self, requests_per_minute: int = 1000):
        """
        Initialize global rate limiter.

        Args:
            requests_per_minute: Maximum total requests per minute
        """
        self.rate = requests_per_minute / 60.0
        self.burst_size = requests_per_minute
        self._tokens = float(self.burst_size)
        self._last_update = time.time()
        self._lock = Lock()
        self._blocked_count = 0
        self._allowed_count = 0

        logger.info(f"Initialized global rate limiter: {requests_per_minute} req/min")

    def is_allowed(self) -> bool:
        """Check if request is allowed globally."""
        with self._lock:
            now = time.time()
            elapsed = now - self._last_update
            self._tokens = min(self.burst_size, self._tokens + (elapsed * self.rate))
            self._last_update = now

            if self._tokens >= 1.0:
                self._tokens -= 1.0
                self._allowed_count += 1
                return True
            else:
                self._blocked_count += 1
                logger.warning("Global rate limit exceeded")
                return False

    def get_stats(self) -> Dict[str, Any]:
        """Get global rate limiter statistics."""
        with self._lock:
            return {
                "current_tokens": round(self._tokens, 2),
                "max_tokens": self.burst_size,
                "requests_allowed": self._allowed_count,
                "requests_blocked": self._blocked_count,
                "block_rate_percent": round(
                    (
                        (self._blocked_count / (self._allowed_count + self._blocked_count) * 100)
                        if (self._allowed_count + self._blocked_count) > 0
                        else 0
                    ),
                    2,
                ),
            }
