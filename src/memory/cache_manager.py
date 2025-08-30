"""
Cache manager for working memory and temporary data storage.

This module provides in-memory caching capabilities for frequently accessed
data, analysis results, and temporary computations to improve performance.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import OrderedDict
import threading

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Cache manager for working memory and temporary data storage.
    
    This class provides TTL-based caching with LRU eviction for managing
    working memory and improving performance of frequently accessed data.
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        default_ttl_seconds: int = 3600,
        cleanup_interval_seconds: int = 300
    ):
        """
        Initialize the cache manager.
        
        Args:
            max_size: Maximum number of items in cache
            default_ttl_seconds: Default TTL for cached items
            cleanup_interval_seconds: Interval for cleanup of expired items
        """
        self.max_size = max_size
        self.default_ttl_seconds = default_ttl_seconds
        self.cleanup_interval_seconds = cleanup_interval_seconds
        
        # Cache storage: key -> (value, expiry_time, access_time)
        self.cache: OrderedDict[str, Tuple[Any, float, float]] = OrderedDict()
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expired_cleanups": 0
        }
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
        
        logger.info(f"CacheManager initialized: max_size={max_size}, ttl={default_ttl_seconds}s")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        with self.lock:
            if key not in self.cache:
                self.stats["misses"] += 1
                return None
            
            value, expiry_time, access_time = self.cache[key]
            
            # Check if expired
            if time.time() > expiry_time:
                del self.cache[key]
                self.stats["misses"] += 1
                self.stats["expired_cleanups"] += 1
                return None
            
            # Update access time and move to end (LRU)
            self.cache[key] = (value, expiry_time, time.time())
            self.cache.move_to_end(key)
            
            self.stats["hits"] += 1
            return value
    
    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None
    ) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: TTL in seconds (uses default if None)
        """
        with self.lock:
            ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds
            expiry_time = time.time() + ttl
            access_time = time.time()
            
            # Remove existing entry if present
            if key in self.cache:
                del self.cache[key]
            
            # Add new entry
            self.cache[key] = (value, expiry_time, access_time)
            self.cache.move_to_end(key)
            
            # Evict if over capacity
            if len(self.cache) > self.max_size:
                self._evict_lru()
    
    def delete(self, key: str) -> bool:
        """
        Delete a value from the cache.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if key was deleted, False if not found
        """
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    def exists(self, key: str) -> bool:
        """
        Check if a key exists in the cache and is not expired.
        
        Args:
            key: Cache key to check
            
        Returns:
            True if key exists and is not expired
        """
        with self.lock:
            if key not in self.cache:
                return False
            
            _, expiry_time, _ = self.cache[key]
            if time.time() > expiry_time:
                del self.cache[key]
                self.stats["expired_cleanups"] += 1
                return False
            
            return True
    
    def clear(self) -> None:
        """Clear all items from the cache."""
        with self.lock:
            self.cache.clear()
            logger.info("Cache cleared")
    
    def get_or_set(
        self,
        key: str,
        factory_func,
        ttl_seconds: Optional[int] = None
    ) -> Any:
        """
        Get a value from cache or set it using a factory function.
        
        Args:
            key: Cache key
            factory_func: Function to call if key not in cache
            ttl_seconds: TTL in seconds (uses default if None)
            
        Returns:
            Cached or newly created value
        """
        value = self.get(key)
        if value is not None:
            return value
        
        # Create new value using factory function
        value = factory_func()
        self.set(key, value, ttl_seconds)
        return value
    
    def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """
        Get multiple values from the cache.
        
        Args:
            keys: List of cache keys
            
        Returns:
            Dictionary mapping keys to values (only for found keys)
        """
        result = {}
        for key in keys:
            value = self.get(key)
            if value is not None:
                result[key] = value
        return result
    
    def set_many(
        self,
        items: Dict[str, Any],
        ttl_seconds: Optional[int] = None
    ) -> None:
        """
        Set multiple values in the cache.
        
        Args:
            items: Dictionary of key-value pairs to cache
            ttl_seconds: TTL in seconds (uses default if None)
        """
        for key, value in items.items():
            self.set(key, value, ttl_seconds)
    
    def _evict_lru(self) -> None:
        """Evict the least recently used item."""
        if self.cache:
            key, _ = self.cache.popitem(last=False)  # Remove first (oldest) item
            self.stats["evictions"] += 1
            logger.debug(f"Evicted LRU item: {key}")
    
    def _cleanup_loop(self) -> None:
        """Background cleanup loop for expired items."""
        while True:
            try:
                time.sleep(self.cleanup_interval_seconds)
                self._cleanup_expired()
            except Exception as e:
                logger.error(f"Error in cache cleanup loop: {e}")
    
    def _cleanup_expired(self) -> None:
        """Remove expired items from the cache."""
        with self.lock:
            current_time = time.time()
            expired_keys = []
            
            for key, (_, expiry_time, _) in self.cache.items():
                if current_time > expiry_time:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.cache[key]
                self.stats["expired_cleanups"] += 1
            
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired items")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self.lock:
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0.0
            
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.stats["hits"],
                "misses": self.stats["misses"],
                "hit_rate": hit_rate,
                "evictions": self.stats["evictions"],
                "expired_cleanups": self.stats["expired_cleanups"],
                "default_ttl_seconds": self.default_ttl_seconds
            }
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get estimated memory usage of the cache.
        
        Returns:
            Dictionary with memory usage information
        """
        with self.lock:
            # Rough estimation of memory usage
            total_items = len(self.cache)
            avg_item_size = 1024  # Rough estimate in bytes
            
            return {
                "total_items": total_items,
                "estimated_size_bytes": total_items * avg_item_size,
                "estimated_size_mb": (total_items * avg_item_size) / (1024 * 1024)
            }
    
    def get_ttl(self, key: str) -> Optional[float]:
        """
        Get the TTL for a key.
        
        Args:
            key: Cache key
            
        Returns:
            TTL in seconds or None if key not found/expired
        """
        with self.lock:
            if key not in self.cache:
                return None
            
            _, expiry_time, _ = self.cache[key]
            current_time = time.time()
            
            if current_time > expiry_time:
                del self.cache[key]
                return None
            
            return expiry_time - current_time
    
    def extend_ttl(self, key: str, additional_seconds: int) -> bool:
        """
        Extend the TTL for a key.
        
        Args:
            key: Cache key
            additional_seconds: Additional seconds to add to TTL
            
        Returns:
            True if TTL was extended, False if key not found/expired
        """
        with self.lock:
            if key not in self.cache:
                return False
            
            value, expiry_time, access_time = self.cache[key]
            current_time = time.time()
            
            if current_time > expiry_time:
                del self.cache[key]
                return False
            
            # Extend TTL
            new_expiry_time = expiry_time + additional_seconds
            self.cache[key] = (value, new_expiry_time, access_time)
            return True
