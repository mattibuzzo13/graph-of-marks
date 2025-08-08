# igp/utils/cache.py
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Generic, Hashable, Iterable, Iterator, MutableMapping, Optional, Tuple, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


@dataclass
class CacheStats:
    size: int
    max_size: int


class LRUCache(Generic[K, V]):
    """
    Semplice LRU cache (chiavi hashable). Evict O(1) usando OrderedDict.
    """
    def __init__(self, max_size: int = 128) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be > 0")
        self._store: "OrderedDict[K, V]" = OrderedDict()
        self._max = int(max_size)

    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        if key in self._store:
            self._store.move_to_end(key, last=True)
            return self._store[key]
        return default

    def __contains__(self, key: K) -> bool:
        return key in self._store

    def put(self, key: K, value: V) -> None:
        if key in self._store:
            self._store.move_to_end(key, last=True)
            self._store[key] = value
            return
        self._store[key] = value
        if len(self._store) > self._max:
            self._store.popitem(last=False)  # evict least-recent

    def pop(self, key: K, default: Optional[V] = None) -> Optional[V]:
        return self._store.pop(key, default)  # type: ignore[return-value]

    def clear(self) -> None:
        self._store.clear()

    def stats(self) -> CacheStats:
        return CacheStats(size=len(self._store), max_size=self._max)

    def __len__(self) -> int:
        return len(self._store)

    def items(self):
        return self._store.items()

    def keys(self):
        return self._store.keys()

    def values(self):
        return self._store.values()
