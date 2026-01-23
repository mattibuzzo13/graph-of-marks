# igp/utils/cache.py
# Thread-safe LRU cache with optional TTL and lightweight stats.
# - O(1) get/put/update via OrderedDict
# - Evicts least-recently used when capacity exceeded
# - Optional TTL invalidation on access
# - Hit/miss stats

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from typing import Callable, Generic, Hashable, Iterable, Optional, Tuple, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


@dataclass
class CacheStats:
    size: int
    max_size: int
    hits: int = 0
    misses: int = 0


class LRUCache(Generic[K, V]):
    def __init__(self, max_size: int = 128, ttl_seconds: Optional[float] = None) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be > 0")
        self._store: "OrderedDict[K, Tuple[V, float]]" = OrderedDict()
        self._max = int(max_size)
        self._ttl = float(ttl_seconds) if ttl_seconds is not None else None
        self._hits = 0
        self._misses = 0
        self._lock = RLock()

    def _expired(self, ts: float) -> bool:
        return self._ttl is not None and (time.time() - ts) > self._ttl

    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        with self._lock:
            if key in self._store:
                val, ts = self._store[key]
                if self._expired(ts):
                    self._store.pop(key, None)
                    self._misses += 1
                    return default
                self._store.move_to_end(key, last=True)
                self._hits += 1
                return val
            self._misses += 1
            return default

    def __contains__(self, key: K) -> bool:
        with self._lock:
            if key not in self._store:
                return False
            _, ts = self._store[key]
            if self._expired(ts):
                self._store.pop(key, None)
                return False
            return True

    def put(self, key: K, value: V) -> None:
        with self._lock:
            now = time.time()
            if key in self._store:
                self._store.move_to_end(key, last=True)
                self._store[key] = (value, now)
            else:
                self._store[key] = (value, now)
                if len(self._store) > self._max:
                    self._store.popitem(last=False)

    def get_or_put(self, key: K, factory: Callable[[], V]) -> V:
        val = self.get(key, None)
        if val is not None:
            return val
        new_val = factory()
        self.put(key, new_val)
        return new_val

    def pop(self, key: K, default: Optional[V] = None) -> Optional[V]:
        with self._lock:
            item = self._store.pop(key, None)
            if item is None:
                return default
            return item[0]

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> CacheStats:
        with self._lock:
            return CacheStats(size=len(self._store), max_size=self._max, hits=self._hits, misses=self._misses)

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def items(self) -> Iterable[Tuple[K, V]]:
        with self._lock:
            for k, (v, _) in self._store.items():
                yield k, v

    def keys(self) -> Iterable[K]:
        with self._lock:
            return list(self._store.keys())

    def values(self) -> Iterable[V]:
        with self._lock:
            return [v for (v, _) in self._store.values()]