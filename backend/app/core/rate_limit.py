"""In-process fixed-window rate limiter for login attempts.

Deliberately simple: a dict of deques guarded by a lock. Single-process
deployments (the target here) get brute-force protection with no extra
infrastructure. If this app is ever run behind multiple workers, swap this for
a Redis-backed counter -- the interface is small on purpose.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from app.core.config import settings


class RateLimiter:
    def __init__(self, max_attempts: int, window_seconds: int) -> None:
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def _prune(self, bucket: deque[float], now: float) -> None:
        cutoff = now - self.window_seconds
        while bucket and bucket[0] < cutoff:
            bucket.popleft()

    def hit(self, key: str) -> bool:
        """Record an attempt. Returns ``True`` if it is allowed."""
        now = time.monotonic()
        with self._lock:
            bucket = self._hits[key]
            self._prune(bucket, now)
            if len(bucket) >= self.max_attempts:
                return False
            bucket.append(now)
            return True

    def retry_after(self, key: str) -> int:
        """Seconds until the oldest attempt in the window expires."""
        now = time.monotonic()
        with self._lock:
            bucket = self._hits[key]
            self._prune(bucket, now)
            if not bucket:
                return 0
            return max(1, int(self.window_seconds - (now - bucket[0])) + 1)

    def reset(self, key: str | None = None) -> None:
        """Clear one key (after a successful login) or everything (tests)."""
        with self._lock:
            if key is None:
                self._hits.clear()
            else:
                self._hits.pop(key, None)


#: Shared limiter used by the login route.
login_limiter = RateLimiter(
    max_attempts=settings.login_rate_limit_attempts,
    window_seconds=settings.login_rate_limit_window_seconds,
)

