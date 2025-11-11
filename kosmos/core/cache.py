"""
Core caching infrastructure for Kosmos.

Provides abstract base classes and implementations for multi-layer caching
with LRU eviction, TTL support, and statistics tracking.
"""

import hashlib
import json
import pickle
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)


class CacheError(Exception):
    """Exception raised for cache-related errors."""
    pass


class CacheStats:
    """Thread-safe cache statistics tracker."""

    def __init__(self):
        """Initialize cache statistics."""
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._sets = 0
        self._evictions = 0
        self._errors = 0
        self._invalidations = 0

    def record_hit(self):
        """Record a cache hit."""
        with self._lock:
            self._hits += 1

    def record_miss(self):
        """Record a cache miss."""
        with self._lock:
            self._misses += 1

    def record_set(self):
        """Record a cache set operation."""
        with self._lock:
            self._sets += 1

    def record_eviction(self):
        """Record a cache eviction."""
        with self._lock:
            self._evictions += 1

    def record_error(self):
        """Record a cache error."""
        with self._lock:
            self._errors += 1

    def record_invalidation(self, count: int = 1):
        """Record cache invalidation(s)."""
        with self._lock:
            self._invalidations += count

    def get_stats(self) -> Dict[str, Any]:
        """
        Get current statistics.

        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0.0

            return {
                "hits": self._hits,
                "misses": self._misses,
                "sets": self._sets,
                "evictions": self._evictions,
                "errors": self._errors,
                "invalidations": self._invalidations,
                "total_requests": total_requests,
                "hit_rate_percent": round(hit_rate, 2)
            }

    def reset(self):
        """Reset all statistics."""
        with self._lock:
            self._hits = 0
            self._misses = 0
            self._sets = 0
            self._evictions = 0
            self._errors = 0
            self._invalidations = 0


class BaseCache(ABC):
    """
    Abstract base class for cache implementations.

    Defines the interface that all cache implementations must follow.
    """

    def __init__(self, ttl_seconds: int = 172800):  # 48 hours default
        """
        Initialize the base cache.

        Args:
            ttl_seconds: Time-to-live for cached items in seconds (default: 48 hours)
        """
        self.ttl_seconds = ttl_seconds
        self.stats = CacheStats()

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a cached value.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        pass

    @abstractmethod
    def set(self, key: str, value: Any) -> bool:
        """
        Store a value in the cache.

        Args:
            key: Cache key
            value: Value to cache

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """
        Delete a cached value.

        Args:
            key: Cache key

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    def clear(self) -> int:
        """
        Clear all cached values.

        Returns:
            Number of entries removed
        """
        pass

    @abstractmethod
    def size(self) -> int:
        """
        Get the number of items in the cache.

        Returns:
            Number of cached items
        """
        pass

    @staticmethod
    def generate_key(*args, **kwargs) -> str:
        """
        Generate a cache key from arguments.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Hexadecimal cache key
        """
        # Sort kwargs for consistent key generation
        key_data = {
            'args': args,
            'kwargs': sorted(kwargs.items())
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        base_stats = self.stats.get_stats()
        base_stats.update({
            "ttl_seconds": self.ttl_seconds,
            "size": self.size(),
        })
        return base_stats


class InMemoryCache(BaseCache):
    """
    Thread-safe in-memory LRU cache with TTL support.

    Uses OrderedDict for LRU eviction and stores expiration times
    for each entry.
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: int = 172800  # 48 hours
    ):
        """
        Initialize the in-memory cache.

        Args:
            max_size: Maximum number of items to cache
            ttl_seconds: Time-to-live for cached items in seconds
        """
        super().__init__(ttl_seconds=ttl_seconds)
        self.max_size = max_size
        self._cache: OrderedDict[str, Tuple[Any, datetime]] = OrderedDict()
        self._lock = threading.RLock()

        logger.info(f"Initialized InMemoryCache: max_size={max_size}, ttl={ttl_seconds}s")

    def _is_expired(self, expires_at: datetime) -> bool:
        """Check if a cached item has expired."""
        return datetime.utcnow() > expires_at

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a cached value.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            if key not in self._cache:
                self.stats.record_miss()
                return None

            value, expires_at = self._cache[key]

            # Check expiration
            if self._is_expired(expires_at):
                logger.debug(f"Cache expired: {key[:8]}...")
                del self._cache[key]
                self.stats.record_miss()
                return None

            # Move to end (LRU)
            self._cache.move_to_end(key)
            self.stats.record_hit()
            logger.debug(f"Cache hit: {key[:8]}...")
            return value

    def set(self, key: str, value: Any) -> bool:
        """
        Store a value in the cache.

        Args:
            key: Cache key
            value: Value to cache

        Returns:
            True if successful
        """
        with self._lock:
            try:
                # Calculate expiration
                expires_at = datetime.utcnow() + timedelta(seconds=self.ttl_seconds)

                # Remove oldest if at capacity
                if len(self._cache) >= self.max_size and key not in self._cache:
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]
                    self.stats.record_eviction()
                    logger.debug(f"Evicted: {oldest_key[:8]}...")

                # Store value
                self._cache[key] = (value, expires_at)
                self._cache.move_to_end(key)  # Mark as most recently used
                self.stats.record_set()
                logger.debug(f"Cached: {key[:8]}...")
                return True

            except Exception as e:
                logger.error(f"Error caching value: {e}")
                self.stats.record_error()
                return False

    def delete(self, key: str) -> bool:
        """
        Delete a cached value.

        Args:
            key: Cache key

        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self.stats.record_invalidation()
                return True
            return False

    def clear(self) -> int:
        """
        Clear all cached values.

        Returns:
            Number of entries removed
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self.stats.record_invalidation(count)
            logger.info(f"Cleared {count} cache entries")
            return count

    def size(self) -> int:
        """Get the number of items in the cache."""
        with self._lock:
            return len(self._cache)

    def cleanup_expired(self) -> int:
        """
        Remove expired cache entries.

        Returns:
            Number of entries removed
        """
        with self._lock:
            now = datetime.utcnow()
            expired_keys = [
                key for key, (_, expires_at) in self._cache.items()
                if expires_at < now
            ]

            for key in expired_keys:
                del self._cache[key]

            if expired_keys:
                self.stats.record_invalidation(len(expired_keys))
                logger.info(f"Cleaned up {len(expired_keys)} expired entries")

            return len(expired_keys)


class DiskCache(BaseCache):
    """
    Thread-safe disk-based cache with TTL support.

    Stores cached values as pickle files with metadata.
    Uses subdirectory distribution for better filesystem performance.
    """

    def __init__(
        self,
        cache_dir: str = ".kosmos_cache",
        ttl_seconds: int = 172800,  # 48 hours
        max_size_mb: int = 5000  # 5GB default
    ):
        """
        Initialize the disk cache.

        Args:
            cache_dir: Directory to store cache files
            ttl_seconds: Time-to-live for cached items in seconds
            max_size_mb: Maximum cache directory size in MB
        """
        super().__init__(ttl_seconds=ttl_seconds)
        self.cache_dir = Path(cache_dir)
        self.max_size_mb = max_size_mb
        self._lock = threading.RLock()

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized DiskCache: dir={cache_dir}, ttl={ttl_seconds}s")

    def _get_cache_path(self, key: str) -> Path:
        """Get the file path for a cache key."""
        # Use first 2 chars for subdirectory distribution
        subdir = self.cache_dir / key[:2]
        subdir.mkdir(exist_ok=True)
        return subdir / f"{key}.pkl"

    def _is_expired(self, cached_at: datetime) -> bool:
        """Check if a cached item has expired."""
        expiry = cached_at + timedelta(seconds=self.ttl_seconds)
        return datetime.utcnow() > expiry

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a cached value.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        cache_path = self._get_cache_path(key)

        if not cache_path.exists():
            self.stats.record_miss()
            return None

        with self._lock:
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)

                # Check expiration
                if self._is_expired(cached_data['cached_at']):
                    logger.debug(f"Cache expired: {key[:8]}...")
                    cache_path.unlink()
                    self.stats.record_miss()
                    return None

                self.stats.record_hit()
                logger.debug(f"Cache hit: {key[:8]}...")
                return cached_data['value']

            except Exception as e:
                logger.warning(f"Error reading cache: {e}")
                # Delete corrupted cache
                if cache_path.exists():
                    cache_path.unlink()
                self.stats.record_error()
                return None

    def set(self, key: str, value: Any) -> bool:
        """
        Store a value in the cache.

        Args:
            key: Cache key
            value: Value to cache

        Returns:
            True if successful
        """
        cache_path = self._get_cache_path(key)

        with self._lock:
            try:
                cached_data = {
                    'key': key,
                    'value': value,
                    'cached_at': datetime.utcnow()
                }

                with open(cache_path, 'wb') as f:
                    pickle.dump(cached_data, f)

                self.stats.record_set()
                logger.debug(f"Cached: {key[:8]}...")

                # Check cache size
                self._check_cache_size()
                return True

            except Exception as e:
                logger.error(f"Error writing cache: {e}")
                self.stats.record_error()
                return False

    def delete(self, key: str) -> bool:
        """
        Delete a cached value.

        Args:
            key: Cache key

        Returns:
            True if deleted, False if not found
        """
        cache_path = self._get_cache_path(key)

        with self._lock:
            if cache_path.exists():
                cache_path.unlink()
                self.stats.record_invalidation()
                return True
            return False

    def clear(self) -> int:
        """
        Clear all cached values.

        Returns:
            Number of entries removed
        """
        with self._lock:
            count = 0
            for cache_file in self.cache_dir.rglob("*.pkl"):
                cache_file.unlink()
                count += 1

            self.stats.record_invalidation(count)
            logger.info(f"Cleared {count} cache entries")
            return count

    def size(self) -> int:
        """Get the number of items in the cache."""
        return len(list(self.cache_dir.rglob("*.pkl")))

    def cleanup_expired(self) -> int:
        """
        Remove expired cache entries.

        Returns:
            Number of entries removed
        """
        with self._lock:
            count = 0

            for cache_file in self.cache_dir.rglob("*.pkl"):
                try:
                    with open(cache_file, 'rb') as f:
                        cached_data = pickle.load(f)

                    if self._is_expired(cached_data['cached_at']):
                        cache_file.unlink()
                        count += 1

                except Exception:
                    # Delete corrupted files
                    cache_file.unlink()
                    count += 1

            if count > 0:
                self.stats.record_invalidation(count)
                logger.info(f"Cleaned up {count} expired entries")

            return count

    def _check_cache_size(self):
        """Check cache size and cleanup if exceeds limit."""
        cache_files = list(self.cache_dir.rglob("*.pkl"))
        total_size_mb = sum(f.stat().st_size for f in cache_files) / (1024 * 1024)

        if total_size_mb > self.max_size_mb:
            logger.warning(
                f"Cache size ({total_size_mb:.1f} MB) exceeds "
                f"limit ({self.max_size_mb} MB)"
            )

            # Delete oldest files first (LRU)
            cache_files.sort(key=lambda f: f.stat().st_mtime)

            deleted_mb = 0
            target_mb = self.max_size_mb * 0.8  # Clean to 80% of max
            evicted_count = 0

            for cache_file in cache_files:
                if total_size_mb - deleted_mb <= target_mb:
                    break

                file_size_mb = cache_file.stat().st_size / (1024 * 1024)
                cache_file.unlink()
                deleted_mb += file_size_mb
                evicted_count += 1

            self.stats.record_eviction()
            logger.info(f"Evicted {evicted_count} files ({deleted_mb:.1f} MB)")


class HybridCache(BaseCache):
    """
    Two-tier hybrid cache: fast in-memory LRU + persistent disk cache.

    Hot items are cached in memory for fast access, while all items
    are persisted to disk for durability and overflow.
    """

    def __init__(
        self,
        memory_size: int = 1000,
        cache_dir: str = ".kosmos_cache",
        ttl_seconds: int = 172800,
        max_size_mb: int = 5000
    ):
        """
        Initialize the hybrid cache.

        Args:
            memory_size: Maximum items in memory cache
            cache_dir: Directory for disk cache
            ttl_seconds: Time-to-live in seconds
            max_size_mb: Maximum disk cache size in MB
        """
        super().__init__(ttl_seconds=ttl_seconds)
        self.memory_cache = InMemoryCache(max_size=memory_size, ttl_seconds=ttl_seconds)
        self.disk_cache = DiskCache(
            cache_dir=cache_dir,
            ttl_seconds=ttl_seconds,
            max_size_mb=max_size_mb
        )

        logger.info(f"Initialized HybridCache: memory={memory_size}, disk={cache_dir}")

    def get(self, key: str) -> Optional[Any]:
        """Retrieve from memory first, then disk."""
        # Try memory cache first
        value = self.memory_cache.get(key)
        if value is not None:
            self.stats.record_hit()
            return value

        # Try disk cache
        value = self.disk_cache.get(key)
        if value is not None:
            # Promote to memory cache
            self.memory_cache.set(key, value)
            self.stats.record_hit()
            return value

        self.stats.record_miss()
        return None

    def set(self, key: str, value: Any) -> bool:
        """Store in both memory and disk."""
        memory_success = self.memory_cache.set(key, value)
        disk_success = self.disk_cache.set(key, value)

        if memory_success or disk_success:
            self.stats.record_set()
            return True

        self.stats.record_error()
        return False

    def delete(self, key: str) -> bool:
        """Delete from both caches."""
        memory_deleted = self.memory_cache.delete(key)
        disk_deleted = self.disk_cache.delete(key)

        if memory_deleted or disk_deleted:
            self.stats.record_invalidation()
            return True
        return False

    def clear(self) -> int:
        """Clear both caches."""
        memory_count = self.memory_cache.clear()
        disk_count = self.disk_cache.clear()
        total = memory_count + disk_count

        self.stats.record_invalidation(total)
        return total

    def size(self) -> int:
        """Get total items across both caches."""
        # Use disk cache size (it has everything)
        return self.disk_cache.size()

    def cleanup_expired(self) -> int:
        """Remove expired entries from both caches."""
        memory_cleaned = self.memory_cache.cleanup_expired()
        disk_cleaned = self.disk_cache.cleanup_expired()
        total = memory_cleaned + disk_cleaned

        if total > 0:
            self.stats.record_invalidation(total)

        return total

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics from both caches."""
        base_stats = super().get_stats()
        base_stats.update({
            "memory_cache": self.memory_cache.get_stats(),
            "disk_cache": self.disk_cache.get_stats(),
        })
        return base_stats
