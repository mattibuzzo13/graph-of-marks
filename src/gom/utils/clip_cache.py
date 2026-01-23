"""Simple persistent disk-backed cache for CLIP scores.

This uses sqlite3 with a tiny table mapping keys -> pickled values.
Keys should be deterministic strings (e.g. SHA256 of image crop bytes + coords + prompt).
Values are pickled Python objects (floats or small lists).

This cache is intentionally small and simple: it doesn't implement advanced
eviction. It's intended for repeated dataset preprocessing runs where hits
substantially reduce expensive model calls.
"""
from __future__ import annotations

import os
import pickle
import sqlite3
from typing import Optional


class ClipDiskCache:
    def __init__(self, path: Optional[str] = None, max_entries: int = 100000, max_age_days: Optional[float] = None) -> None:
        if path is None:
            # default to repo-local .cache file
            repo_root = os.getcwd()
            path = os.path.join(repo_root, ".igp_clip_cache.db")
        self.path = os.path.abspath(os.path.expanduser(path))
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self.max_entries = int(max_entries) if max_entries is not None else 100000
        # max_age_days: if set, entries older than this will be removed on writes/reads
        self.max_age_days = float(max_age_days) if max_age_days is not None else None
        self._ensure_table()

    def _ensure_table(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS clip_cache (
                key TEXT PRIMARY KEY,
                value BLOB,
                ts INTEGER
            )
            """
        )
        # index on ts to help eviction queries
        cur.execute("CREATE INDEX IF NOT EXISTS idx_clip_cache_ts ON clip_cache(ts)")
        self._conn.commit()

    def get(self, key: str):
        # cleanup old entries if TTL configured
        try:
            if self.max_age_days is not None:
                self._cleanup_old()
        except Exception:
            pass

        cur = self._conn.cursor()
        cur.execute("SELECT value FROM clip_cache WHERE key=?", (key,))
        row = cur.fetchone()
        if not row:
            return None
        try:
            return pickle.loads(row[0])
        except Exception:
            return None

    def set(self, key: str, value) -> None:
        try:
            blob = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            cur = self._conn.cursor()
            ts = int(__import__("time").time())
            cur.execute(
                "INSERT OR REPLACE INTO clip_cache (key, value, ts) VALUES (?, ?, ?)",
                (key, blob, ts),
            )
            self._conn.commit()
            # cleanup old entries by age if configured
            try:
                if self.max_age_days is not None:
                    self._cleanup_old()
            except Exception:
                pass
            # Evict oldest rows if we exceeded max_entries
            try:
                cur.execute("SELECT COUNT(*) FROM clip_cache")
                total = cur.fetchone()[0]
                if total > self.max_entries:
                    # delete all keys not in the newest max_entries
                    cur.execute(
                        "DELETE FROM clip_cache WHERE key NOT IN (SELECT key FROM clip_cache ORDER BY ts DESC LIMIT ?)",
                        (self.max_entries,)
                    )
                    self._conn.commit()
            except Exception:
                # best-effort, ignore eviction errors
                pass
        except Exception:
            # best-effort: do not crash preprocessing because of cache IO
            pass

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def _cleanup_old(self) -> None:
        """Remove entries older than max_age_days (best-effort)."""
        if self.max_age_days is None:
            return
        try:
            import time as _time
            cutoff = int(_time.time() - float(self.max_age_days) * 24 * 3600)
            cur = self._conn.cursor()
            cur.execute("DELETE FROM clip_cache WHERE ts < ?", (cutoff,))
            self._conn.commit()
        except Exception:
            pass
