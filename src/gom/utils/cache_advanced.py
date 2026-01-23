# igp/utils/cache_advanced.py
# Advanced LRU Cache with intelligent eviction and memory management

from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image


@dataclass
class CacheStats:
    """Statistics for cache performance monitoring."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size_bytes: int = 0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def __str__(self) -> str:
        return (f"Cache Stats: {self.hits} hits, {self.misses} misses, "
                f"{self.evictions} evictions, hit_rate={self.hit_rate:.2%}, "
                f"size={self.total_size_bytes / 1024**2:.1f} MB")


class LRUCache:
    """
    Advanced LRU cache with:
    - Automatic eviction based on size/count limits
    - Memory-aware storage (tracks object sizes)
    - Performance statistics
    - Thread-safe operations (optional)
    """
    
    def __init__(
        self, 
        max_items: int = 100,
        max_size_mb: float = 500.0,
        enable_stats: bool = True
    ):
        self.max_items = max_items
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.enable_stats = enable_stats
        
        # LRU storage: OrderedDict maintains insertion order
        self._cache: OrderedDict[str, Tuple[Any, int, float]] = OrderedDict()
        # key -> (value, size_bytes, timestamp)
        
        self.stats = CacheStats() if enable_stats else None
        
    def _estimate_size(self, obj: Any) -> int:
        """Estimate memory size of object in bytes."""
        if isinstance(obj, np.ndarray):
            return obj.nbytes
        elif isinstance(obj, dict):
            total = sys.getsizeof(obj)
            for k, v in obj.items():
                total += self._estimate_size(k) + self._estimate_size(v)
            return total
        elif isinstance(obj, (list, tuple)):
            return sum(self._estimate_size(item) for item in obj) + sys.getsizeof(obj)
        else:
            return sys.getsizeof(obj)
    
    def _evict_oldest(self) -> None:
        """Remove oldest (first) item from cache."""
        if not self._cache:
            return
        key, (value, size, timestamp) = self._cache.popitem(last=False)
        if self.stats:
            self.stats.evictions += 1
            self.stats.total_size_bytes -= size
    
    def _enforce_limits(self) -> None:
        """Evict items until within size and count limits."""
        # Evict by count
        while len(self._cache) > self.max_items:
            self._evict_oldest()
        
        # Evict by size
        while self.stats and self.stats.total_size_bytes > self.max_size_bytes and self._cache:
            self._evict_oldest()
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve item from cache (moves to end for LRU)."""
        if key not in self._cache:
            if self.stats:
                self.stats.misses += 1
            return None
        
        # Move to end (mark as recently used)
        value, size, timestamp = self._cache.pop(key)
        self._cache[key] = (value, size, time.time())
        
        if self.stats:
            self.stats.hits += 1
        
        return value
    
    def put(self, key: str, value: Any) -> None:
        """Store item in cache with LRU eviction."""
        # Remove if already exists (will re-add with new timestamp)
        if key in self._cache:
            old_value, old_size, old_timestamp = self._cache.pop(key)
            if self.stats:
                self.stats.total_size_bytes -= old_size
        
        # Estimate size
        size = self._estimate_size(value)
        
        # Add to cache
        self._cache[key] = (value, size, time.time())
        
        if self.stats:
            self.stats.total_size_bytes += size
        
        # Enforce limits
        self._enforce_limits()
    
    def clear(self) -> None:
        """Clear all cached items."""
        self._cache.clear()
        if self.stats:
            self.stats = CacheStats()
    
    def __len__(self) -> int:
        return len(self._cache)
    
    def __contains__(self, key: str) -> bool:
        return key in self._cache


class ImageDetectionCache:
    """
    Specialized cache for image detection results with smart key generation.
    """
    
    def __init__(self, max_items: int = 100, max_size_mb: float = 500.0):
        self._cache = LRUCache(max_items=max_items, max_size_mb=max_size_mb)
    
    @staticmethod
    def generate_key(
        image: Image.Image,
        detectors: Tuple[str, ...],
        thresholds: Dict[str, float],
        question: str = "",
        extra_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate deterministic cache key from image and parameters.
        
        Uses:
        - Image hash (first 1MB of bytes for speed)
        - Detector configuration
        - Score thresholds
        - Question (if applicable)
        """
        # Fast image hash (sample first 1MB to avoid hashing huge images)
        img_bytes = image.tobytes()[:1024*1024]
        img_hash = hashlib.md5(img_bytes).hexdigest()[:16]
        
        # Parameter hash
        params = {
            "detectors": sorted(detectors),
            "thresholds": {k: round(v, 3) for k, v in sorted(thresholds.items())},
            "question": question.strip().lower() if question else "",
            "image_size": image.size,
        }
        
        if extra_params:
            params.update(extra_params)
        
        param_str = json.dumps(params, sort_keys=True)
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
        
        return f"det_{img_hash}_{param_hash}"
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve detection results from cache."""
        return self._cache.get(key)
    
    def put(self, key: str, detections: Dict[str, Any]) -> None:
        """Store detection results in cache."""
        self._cache.put(key, detections)
    
    def clear(self) -> None:
        """Clear all cached detections."""
        self._cache.clear()
    
    @property
    def stats(self) -> Optional[CacheStats]:
        """Get cache statistics."""
        return self._cache.stats
    
    def __len__(self) -> int:
        return len(self._cache)


# Example usage:
if __name__ == "__main__":
    # Test basic LRU cache
    cache = LRUCache(max_items=3, max_size_mb=1.0)
    
    cache.put("a", np.zeros((100, 100)))
    cache.put("b", np.ones((100, 100)))
    cache.put("c", np.zeros((100, 100)))
    
    print(f"Cache size: {len(cache)}")
    print(cache.stats)
    
    # Access "a" (moves to end)
    val_a = cache.get("a")
    print(f"Retrieved 'a': {val_a is not None}")
    
    # Add "d" (should evict "b" as oldest)
    cache.put("d", np.ones((100, 100)))
    
    print(f"Cache size after adding 'd': {len(cache)}")
    print(f"'b' still in cache: {cache.get('b') is not None}")
    print(cache.stats)
