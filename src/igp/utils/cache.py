# igp/utils/cache.py
# Minimal LRU cache built on top of OrderedDict:
# - O(1) get/put/update
# - Evicts the least-recently used item when capacity is exceeded

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Generic, Hashable, Iterable, Iterator, MutableMapping, Optional, Tuple, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


@dataclass
class CacheStats:
    """Lightweight stats about the cache."""
    size: int
    max_size: int


class LRUCache(Generic[K, V]):
    """
    Simple LRU cache (hashable keys). Uses OrderedDict to maintain access order.
    The most recently accessed item is moved to the end; eviction is O(1).
    """
    def __init__(self, max_size: int = 128) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be > 0")
        self._store: "OrderedDict[K, V]" = OrderedDict()
        self._max = int(max_size)

    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """
        Return the value for 'key' and mark it as recently used.
        If the key is absent, return 'default'.
        """
        if key in self._store:
            self._store.move_to_end(key, last=True)
            return self._store[key]
        return default

    def __contains__(self, key: K) -> bool:
        """Membership test without altering LRU order."""
        return key in self._store

    def put(self, key: K, value: V) -> None:
        """
        Insert or update 'key' with 'value' and mark as recently used.
        Evicts the least-recently used item if capacity is exceeded.
        """
        if key in self._store:
            self._store.move_to_end(key, last=True)
            self._store[key] = value
            return
        self._store[key] = value
        if len(self._store) > self._max:
            self._store.popitem(last=False)  # evict least-recent

    def pop(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """Remove 'key' and return its value; return 'default' if missing."""
        return self._store.pop(key, default)  # type: ignore[return-value]

    def clear(self) -> None:
        """Remove all items from the cache."""
        self._store.clear()

    def stats(self) -> CacheStats:
        """Return current size and capacity."""
        return CacheStats(size=len(self._store), max_size=self._max)

    def __len__(self) -> int:
        """Number of items currently stored."""
        return len(self._store)

    def items(self):
        """View of (key, value) pairs in LRU order (oldest → newest)."""
        return self._store.items()

    def keys(self):
        """View of keys in LRU order (oldest → newest)."""
        return self._store.keys()

    def values(self):
        """View of values in LRU order (oldest → newest)."""
        return self._store.values()
