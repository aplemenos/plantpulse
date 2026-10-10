"""
Pluggable cache: Redis when REDIS_URL is set and reachable, in-process
TTL cache otherwise. Prediction results are cached per (site_id, as_of)
since the agent frequently re-queries the same tool while reasoning
through a question.
"""

from __future__ import annotations

import json
import os
import time
from threading import Lock
from typing import Any, Protocol, cast

Payload = dict[str, Any]


class Cache(Protocol):
    def get(self, key: str) -> Payload | None: ...
    def set(self, key: str, value: Payload, ttl_seconds: float = 30.0) -> None: ...
    def stats(self) -> dict[str, Any]: ...


class InMemoryTTLCache:
    def __init__(self) -> None:
        self._store: dict[str, tuple[float, Payload]] = {}
        self._lock = Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Payload | None:
        with self._lock:
            item = self._store.get(key)
            if item is None:
                self.misses += 1
                return None
            expires_at, value = item
            if expires_at < time.time():
                del self._store[key]
                self.misses += 1
                return None
            self.hits += 1
            return value

    def set(self, key: str, value: Payload, ttl_seconds: float = 30.0) -> None:
        with self._lock:
            self._store[key] = (time.time() + ttl_seconds, value)

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "backend": "in_memory",
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else None,
            "entries": len(self._store),
        }


class RedisCache:
    def __init__(self, url: str) -> None:
        import redis  # imported lazily so redis-py is only required when actually used

        self._client = redis.Redis.from_url(url, decode_responses=True, socket_connect_timeout=1)
        self._client.ping()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Payload | None:
        # decode_responses=True on a synchronous client: values come back as str
        raw = cast(str | None, self._client.get(key))
        if raw is None:
            self.misses += 1
            return None
        self.hits += 1
        return cast(Payload, json.loads(raw))

    def set(self, key: str, value: Payload, ttl_seconds: float = 30.0) -> None:
        self._client.set(key, json.dumps(value), ex=int(ttl_seconds))

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "backend": "redis",
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else None,
        }


def build_cache() -> Cache:
    url = os.environ.get("REDIS_URL")
    if url:
        import redis

        try:
            return RedisCache(url)
        except (redis.RedisError, ValueError) as exc:  # unreachable server or malformed URL
            print(f"[cache] REDIS_URL set but unreachable ({exc!r}); falling back to in-memory cache")
    return InMemoryTTLCache()
