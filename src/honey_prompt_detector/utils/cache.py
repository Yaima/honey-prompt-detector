"""
Production-ready caching layer for honey-prompt detector.

Provides in-memory LRU cache with optional Redis backend for distributed deployments.
"""

import hashlib
import logging
import time
from collections import OrderedDict
from threading import Lock
from typing import Any, Dict, Optional

logger = logging.getLogger("honey_prompt")


class DetectionCache:
    """
    Thread-safe LRU cache for detection results.

    Caches detection results for identical inputs to reduce API calls
    and improve response time. Uses LRU eviction policy.
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """
        Initialize the cache.

        Args:
            max_size: Maximum number of cached entries
            ttl_seconds: Time-to-live for cached entries (default 1 hour)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = Lock()
        self._hits = 0
        self._misses = 0
        logger.info(f"Initialized cache: max_size={max_size}, ttl={ttl_seconds}s")

    def _compute_key(self, text: str) -> str:
        """Compute cache key from input text."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached detection result.

        Args:
            text: Input text to check

        Returns:
            Cached result dict or None if not found/expired
        """
        key = self._compute_key(text)

        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            entry = self._cache[key]

            # Check if expired
            if time.time() - entry["timestamp"] > self.ttl_seconds:
                del self._cache[key]
                self._misses += 1
                logger.debug(f"Cache expired for key: {key[:8]}")
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            logger.debug(f"Cache HIT for key: {key[:8]}")
            return entry["result"]

    def put(self, text: str, result: Dict[str, Any]) -> None:
        """
        Store detection result in cache.

        Args:
            text: Input text
            result: Detection result to cache
        """
        key = self._compute_key(text)

        with self._lock:
            # If key exists, move to end
            if key in self._cache:
                self._cache.move_to_end(key)

            # Add new entry
            self._cache[key] = {"result": result, "timestamp": time.time()}

            # Evict oldest if over capacity
            if len(self._cache) > self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                logger.debug(f"Cache evicted key: {oldest_key[:8]}")

            logger.debug(f"Cache PUT for key: {key[:8]}")

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0

            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate_percent": round(hit_rate, 2),
                "total_requests": total_requests,
            }


class RedisCache:
    """
    Redis-backed cache for distributed deployments.

    Optional Redis backend for sharing cache across multiple instances.
    Falls back to in-memory cache if Redis is unavailable.
    """

    def __init__(self, redis_url: Optional[str] = None, ttl_seconds: int = 3600):
        """
        Initialize Redis cache.

        Args:
            redis_url: Redis connection URL (e.g., "redis://localhost:6379/0")
            ttl_seconds: Time-to-live for cached entries
        """
        self.ttl_seconds = ttl_seconds
        self.redis_client = None
        self._fallback_cache = DetectionCache(max_size=1000, ttl_seconds=ttl_seconds)

        if redis_url:
            try:
                import redis

                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                self.redis_client.ping()
                logger.info(f"Connected to Redis: {redis_url}")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}. Using in-memory cache.")
                self.redis_client = None
        else:
            logger.info("Redis URL not provided. Using in-memory cache.")

    def get(self, text: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached result from Redis or fallback."""
        if self.redis_client:
            try:
                import json

                key = f"detection:{hashlib.sha256(text.encode()).hexdigest()}"
                cached = self.redis_client.get(key)
                if cached:
                    logger.debug(f"Redis HIT for key: {key[:20]}")
                    return json.loads(cached)
                logger.debug(f"Redis MISS for key: {key[:20]}")
                return None
            except Exception as e:
                logger.error(f"Redis get error: {e}. Using fallback cache.")
                return self._fallback_cache.get(text)
        else:
            return self._fallback_cache.get(text)

    def put(self, text: str, result: Dict[str, Any]) -> None:
        """Store result in Redis or fallback."""
        if self.redis_client:
            try:
                import json

                key = f"detection:{hashlib.sha256(text.encode()).hexdigest()}"
                self.redis_client.setex(key, self.ttl_seconds, json.dumps(result))
                logger.debug(f"Redis PUT for key: {key[:20]}")
            except Exception as e:
                logger.error(f"Redis put error: {e}. Using fallback cache.")
                self._fallback_cache.put(text, result)
        else:
            self._fallback_cache.put(text, result)

    def clear(self) -> None:
        """Clear cache."""
        if self.redis_client:
            try:
                # Clear all detection keys
                for key in self.redis_client.scan_iter("detection:*"):
                    self.redis_client.delete(key)
                logger.info("Redis cache cleared")
            except Exception as e:
                logger.error(f"Redis clear error: {e}")
        self._fallback_cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if self.redis_client:
            try:
                info = self.redis_client.info("stats")
                return {
                    "backend": "redis",
                    "keyspace_hits": info.get("keyspace_hits", 0),
                    "keyspace_misses": info.get("keyspace_misses", 0),
                    "connected": True,
                }
            except Exception as e:
                logger.error(f"Redis stats error: {e}")
                return {"backend": "redis", "connected": False, "error": str(e)}
        else:
            stats = self._fallback_cache.get_stats()
            stats["backend"] = "in-memory"
            return stats
