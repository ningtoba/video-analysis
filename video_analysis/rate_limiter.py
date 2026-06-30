"""In-memory token bucket rate limiter for the REST API.

Provides a per-client-key token bucket implementation for rate-limiting
FastAPI endpoints.  Uses ``asyncio.Lock`` for thread safety so it works
correctly with FastAPI async endpoint handlers.

Exports:
    TokenBucketLimiter: Singleton-compatible rate limiter with default
        config of 100 requests / minute per IP address.

Usage::

    from video_analysis.rate_limiter import TokenBucketLimiter

    limiter = TokenBucketLimiter()

    if not await limiter.consume(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CAPACITY: int = 100  # maximum bucket size (burst allowance)
DEFAULT_RATE: float = 100.0 / 60  # tokens per second (≈ 100 requests / minute)


# ---------------------------------------------------------------------------
# TokenBucketLimiter
# ---------------------------------------------------------------------------


class TokenBucketLimiter:
    """Per-client token bucket rate limiter.

    Each client (identified by a key — typically the IP address) gets its
    own token bucket with a configurable capacity and refill rate.  Tokens
    are added continuously at *rate* per second, up to *capacity*.

    Thread safety is provided by an ``asyncio.Lock`` so that concurrent
    async endpoint handlers do not race on bucket state.

    Args:
        capacity: Maximum number of tokens a bucket can hold (burst limit).
        rate: Token refill rate per second.

    Example::

        limiter = TokenBucketLimiter(capacity=100, rate=100.0 / 60)

        # In an endpoint:
        if not await limiter.consume(request.client.host):
            raise HTTPException(status_code=429, detail="Too many requests")
    """

    def __init__(
        self,
        capacity: int = DEFAULT_CAPACITY,
        rate: float = DEFAULT_RATE,
        max_clients: int = 10000,
    ) -> None:
        self._capacity = capacity
        self._rate = rate
        self._max_clients = max_clients
        self._buckets: Dict[str, _TokenBucket] = {}
        self._lock = asyncio.Lock()

    async def consume(self, key: str, tokens: int = 1) -> bool:
        """Try to consume *tokens* from the bucket for *key*.

        Args:
            key: Client identifier (e.g. IP address or API key).
            tokens: Number of tokens to consume (default 1).

        Returns:
            ``True`` if the request is allowed, ``False`` if rate limited.
        """
        async with self._lock:
            now = time.monotonic()
            bucket = self._buckets.get(key)

            if bucket is None:
                # Evict LRU bucket if at capacity.
                if len(self._buckets) >= self._max_clients:
                    self._evict_lru()

                # First request from this client — create a full bucket.
                bucket = _TokenBucket(
                    tokens=float(self._capacity),
                    last_refill=now,
                )
                self._buckets[key] = bucket
            else:
                # Refill tokens based on elapsed time.
                elapsed = now - bucket.last_refill
                bucket.tokens = min(
                    float(self._capacity),
                    bucket.tokens + elapsed * self._rate,
                )
                bucket.last_refill = now

            bucket.last_access = now

            if bucket.tokens >= tokens:
                bucket.tokens -= tokens
                return True

            logger.debug(
                "Rate limit hit for %r (%.2f tokens available)",
                key,
                bucket.tokens,
            )
            return False

    async def reset(self, key: str | None = None) -> None:
        """Reset the bucket(s) for *key*, or all buckets if *key* is ``None``.

        Args:
            key: Optional client key to reset.  If ``None``, all buckets
                are cleared.
        """
        async with self._lock:
            if key is not None:
                self._buckets.pop(key, None)
            else:
                self._buckets.clear()

    def _evict_lru(self) -> None:
        """Evict the least recently accessed bucket.

        Called under ``self._lock`` when the number of buckets reaches
        ``self._max_clients`` and a new client arrives.
        """
        oldest_key: str | None = None
        oldest_access = float("inf")
        for k, b in self._buckets.items():
            if b.last_access < oldest_access:
                oldest_access = b.last_access
                oldest_key = k
        if oldest_key is not None:
            del self._buckets[oldest_key]

    @property
    def capacity(self) -> int:
        """Maximum bucket size (burst allowance)."""
        return self._capacity

    @property
    def rate(self) -> float:
        """Token refill rate per second."""
        return self._rate


# ---------------------------------------------------------------------------
# Internal bucket state
# ---------------------------------------------------------------------------


class _TokenBucket:
    """Internal mutable state for a single client bucket.

    Not exported — managed exclusively by ``TokenBucketLimiter``.
    """

    __slots__ = ("tokens", "last_refill", "last_access")

    def __init__(self, tokens: float, last_refill: float) -> None:
        self.tokens = tokens
        self.last_refill = last_refill
        self.last_access = last_refill
