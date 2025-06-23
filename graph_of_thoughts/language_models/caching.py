# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Derek Vitrano

"""
Intelligent Caching Layer for Language Models.

This module provides sophisticated caching capabilities for language model responses,
configurations, and metadata. The caching system is designed to improve performance
while maintaining data consistency and providing configurable cache policies.

Key Features:
    - Multi-level caching (response, configuration, metadata)
    - TTL (Time To Live) support for automatic expiration
    - LRU (Least Recently Used) eviction for memory management
    - Intelligent cache key generation considering all parameters
    - Cache statistics and monitoring
    - Thread-safe operations for concurrent access
    - Configurable cache policies and limits

The caching layer is designed to be transparent to the language model implementations
while providing significant performance improvements for repeated operations.
"""

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class CachePolicy(Enum):
    """Cache eviction policies."""

    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In, First Out
    TTL_ONLY = "ttl_only"  # Only TTL-based expiration


@dataclass
class CacheEntry:
    """Represents a single cache entry with metadata."""

    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    ttl: Optional[float] = None

    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl

    def touch(self) -> None:
        """Update access metadata."""
        self.last_accessed = time.time()
        self.access_count += 1


@dataclass
class CacheConfig:
    """Configuration for cache behavior."""

    max_size: int = 1000
    default_ttl: Optional[float] = 3600.0  # 1 hour default
    policy: CachePolicy = CachePolicy.LRU
    enable_statistics: bool = True
    cleanup_interval: float = 300.0  # 5 minutes

    # Specific cache configurations
    response_cache_size: int = 500
    config_cache_size: int = 50
    metadata_cache_size: int = 200

    # TTL configurations
    response_ttl: float = 1800.0  # 30 minutes
    config_ttl: float = 7200.0  # 2 hours
    metadata_ttl: float = 3600.0  # 1 hour


@dataclass
class CacheStats:
    """Cache statistics for monitoring and debugging."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    size: int = 0
    max_size: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> "dict[str, Any]":
        """Convert stats to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "expirations": self.expirations,
            "size": self.size,
            "max_size": self.max_size,
            "hit_rate": self.hit_rate,
        }


class IntelligentCache:
    """
    Intelligent caching system with TTL, LRU eviction, and statistics.

    This cache provides sophisticated caching capabilities including:
    - Automatic expiration based on TTL
    - LRU eviction when cache is full
    - Thread-safe operations
    - Detailed statistics tracking
    - Configurable policies and limits
    """

    def __init__(self, config: CacheConfig):
        """
        Initialize the intelligent cache.

        :param config: Cache configuration
        :type config: CacheConfig
        """
        self.config: CacheConfig = config
        self._cache: "OrderedDict[str, CacheEntry]" = OrderedDict()
        self._lock: threading.RLock = threading.RLock()
        self._stats: CacheStats = CacheStats(max_size=config.max_size)
        self._last_cleanup: float = time.time()

        logger.debug(
            f"Initialized cache with max_size={config.max_size}, policy={config.policy.value}"
        )

    def _generate_key(self, *args: Any, **kwargs: Any) -> str:
        """
        Generate a cache key from arguments and keyword arguments.

        :param args: Positional arguments
        :param kwargs: Keyword arguments
        :return: Cache key
        :rtype: str
        """
        # Create a deterministic representation of the arguments
        key_data = {"args": args, "kwargs": sorted(kwargs.items()) if kwargs else {}}

        # Convert to JSON and hash for consistent key generation
        key_json = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_json.encode()).hexdigest()[:16]  # Use first 16 chars

    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.

        :param key: Cache key
        :type key: str
        :return: Cached value or None if not found/expired
        :rtype: Optional[Any]
        """
        with self._lock:
            self._maybe_cleanup()

            if key not in self._cache:
                self._stats.misses += 1
                return None

            entry = self._cache[key]

            # Check if expired
            if entry.is_expired():
                del self._cache[key]
                self._stats.expirations += 1
                self._stats.misses += 1
                self._stats.size = len(self._cache)
                return None

            # Update access metadata
            entry.touch()

            # Move to end for LRU
            if self.config.policy == CachePolicy.LRU:
                self._cache.move_to_end(key)

            self._stats.hits += 1
            return entry.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Put a value in the cache.

        :param key: Cache key
        :type key: str
        :param value: Value to cache
        :type value: Any
        :param ttl: Time to live in seconds (optional)
        :type ttl: Optional[float]
        """
        with self._lock:
            current_time = time.time()

            # Use provided TTL or default
            effective_ttl = ttl if ttl is not None else self.config.default_ttl

            # Create cache entry
            entry = CacheEntry(
                value=value,
                created_at=current_time,
                last_accessed=current_time,
                ttl=effective_ttl,
            )

            # If key already exists, update it
            if key in self._cache:
                self._cache[key] = entry
                if self.config.policy == CachePolicy.LRU:
                    self._cache.move_to_end(key)
                return

            # Check if we need to evict
            if len(self._cache) >= self.config.max_size:
                self._evict()

            # Add new entry
            self._cache[key] = entry
            self._stats.size = len(self._cache)

    def _evict(self) -> None:
        """Evict entries based on the configured policy."""
        if not self._cache:
            return

        if self.config.policy == CachePolicy.LRU:
            # Remove least recently used (first item in Ordered)
            self._cache.popitem(last=False)
        elif self.config.policy == CachePolicy.LFU:
            # Remove least frequently used
            lfu_key = min(self._cache.keys(), key=lambda k: self._cache[k].access_count)
            del self._cache[lfu_key]
        elif self.config.policy == CachePolicy.FIFO:
            # Remove first in (first item in Ordered)
            self._cache.popitem(last=False)

        self._stats.evictions += 1
        self._stats.size = len(self._cache)

    def _maybe_cleanup(self) -> None:
        """Perform cleanup if enough time has passed."""
        current_time = time.time()
        if current_time - self._last_cleanup > self.config.cleanup_interval:
            self._cleanup_expired()
            self._last_cleanup = current_time

    def _cleanup_expired(self) -> None:
        """Remove all expired entries."""
        expired_keys = [key for key, entry in self._cache.items() if entry.is_expired()]

        for key in expired_keys:
            del self._cache[key]
            self._stats.expirations += 1

        self._stats.size = len(self._cache)

        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._stats = CacheStats(max_size=self.config.max_size)

    def get_stats(self) -> CacheStats:
        """Get current cache statistics."""
        with self._lock:
            self._stats.size = len(self._cache)
            return self._stats

    def cache_key(self, *args: Any, **kwargs: Any) -> str:
        """
        Generate a cache key for the given arguments.

        :param args: Positional arguments
        :param kwargs: Keyword arguments
        :return: Cache key
        :rtype: str
        """
        return self._generate_key(*args, **kwargs)


class MultiLevelCacheManager:
    """
    Multi-level cache manager for different types of data.

    Manages separate caches for:
    - Response cache: Language model responses
    - Configuration cache: Loaded configurations
    - Metadata cache: Token usage, costs, and other metadata
    """

    def __init__(self, config: CacheConfig):
        """
        Initialize the multi-level cache manager.

        :param config: Cache configuration
        :type config: CacheConfig
        """
        self.config: CacheConfig = config

        # Create specialized caches
        response_config = CacheConfig(
            max_size=config.response_cache_size,
            default_ttl=config.response_ttl,
            policy=config.policy,
            enable_statistics=config.enable_statistics,
        )

        config_cache_config = CacheConfig(
            max_size=config.config_cache_size,
            default_ttl=config.config_ttl,
            policy=CachePolicy.LRU,  # Always use LRU for configs
            enable_statistics=config.enable_statistics,
        )

        metadata_config = CacheConfig(
            max_size=config.metadata_cache_size,
            default_ttl=config.metadata_ttl,
            policy=config.policy,
            enable_statistics=config.enable_statistics,
        )

        self.response_cache: IntelligentCache = IntelligentCache(response_config)
        self.config_cache: IntelligentCache = IntelligentCache(config_cache_config)
        self.metadata_cache: IntelligentCache = IntelligentCache(metadata_config)

        logger.info(
            "Initialized multi-level cache manager with %d response, "
            "%d config, %d metadata entries",
            config.response_cache_size,
            config.config_cache_size,
            config.metadata_cache_size
        )

    def get_response(self, query: str, **params: Any) -> Optional[Any]:
        """
        Get a cached response for the given query and parameters.

        :param query: Query string
        :type query: str
        :param params: Additional parameters that affect the response
        :return: Cached response or None
        :rtype: Optional[Any]
        """
        key = self.response_cache.cache_key(query, **params)
        return self.response_cache.get(key)

    def put_response(
        self, query: str, response: Any, ttl: Optional[float] = None, **params: Any
    ) -> None:
        """
        Cache a response for the given query and parameters.

        :param query: Query string
        :type query: str
        :param response: Response to cache
        :type response: Any
        :param ttl: Time to live in seconds
        :type ttl: Optional[float]
        :param params: Additional parameters that affect the response
        """
        key = self.response_cache.cache_key(query, **params)
        self.response_cache.put(key, response, ttl)

    def get_config(self, config_path: str, model_name: str) -> Optional[Any]:
        """
        Get a cached configuration.

        :param config_path: Path to configuration file
        :type config_path: str
        :param model_name: Model name
        :type model_name: str
        :return: Cached configuration or None
        :rtype: Optional[Any]
        """
        key = self.config_cache.cache_key(config_path, model_name)
        return self.config_cache.get(key)

    def put_config(
        self,
        config_path: str,
        model_name: str,
        config: Any,
        ttl: Optional[float] = None,
    ) -> None:
        """
        Cache a configuration.

        :param config_path: Path to configuration file
        :type config_path: str
        :param model_name: Model name
        :type model_name: str
        :param config: Configuration to cache
        :type config: Any
        :param ttl: Time to live in seconds
        :type ttl: Optional[float]
        """
        key = self.config_cache.cache_key(config_path, model_name)
        self.config_cache.put(key, config, ttl)

    def get_metadata(self, key: str) -> Optional[Any]:
        """
        Get cached metadata.

        :param key: Metadata key
        :type key: str
        :return: Cached metadata or None
        :rtype: Optional[Any]
        """
        return self.metadata_cache.get(key)

    def put_metadata(
        self, key: str, metadata: Any, ttl: Optional[float] = None
    ) -> None:
        """
        Cache metadata.

        :param key: Metadata key
        :type key: str
        :param metadata: Metadata to cache
        :type metadata: Any
        :param ttl: Time to live in seconds
        :type ttl: Optional[float]
        """
        self.metadata_cache.put(key, metadata, ttl)

    def clear_all(self) -> None:
        """Clear all caches."""
        self.response_cache.clear()
        self.config_cache.clear()
        self.metadata_cache.clear()

    def get_all_stats(self) -> "dict[str, dict[str, Any]]":
        """
        Get statistics for all caches.

        :return: Dictionary containing stats for each cache
        :rtype: dict[str, dict[str, Any]]
        """
        return {
            "response_cache": self.response_cache.get_stats().to_dict(),
            "config_cache": self.config_cache.get_stats().to_dict(),
            "metadata_cache": self.metadata_cache.get_stats().to_dict(),
        }


# Global cache manager instance
_global_cache_manager: Optional[MultiLevelCacheManager] = None


def get_cache_manager(config: Optional[CacheConfig] = None) -> MultiLevelCacheManager:
    """
    Get the global cache manager instance.

    :param config: Cache configuration (used only for first initialization)
    :type config: Optional[CacheConfig]
    :return: Cache manager instance
    :rtype: MultiLevelCacheManager
    """
    global _global_cache_manager

    if _global_cache_manager is None:
        if config is None:
            config = CacheConfig()  # Use default configuration
        _global_cache_manager = MultiLevelCacheManager(config)

    return _global_cache_manager


def clear_global_cache() -> None:
    """Clear the global cache manager."""
    global _global_cache_manager
    if _global_cache_manager is not None:
        _global_cache_manager.clear_all()


def get_global_cache_stats() -> "dict[str, dict[str, Any]]":
    """
    Get statistics for the global cache manager.

    :return: Cache statistics
    :rtype: dict[str, dict[str, Any]]
    """
    global _global_cache_manager
    if _global_cache_manager is None:
        return {}
    return _global_cache_manager.get_all_stats()