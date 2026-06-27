"""
Tests for the TokenBucketLimiter rate limiter (v0.49.0).

Tests core rate-limiting logic: consumption, refill, reset, concurrency.
Note: consume() and reset() are async methods.
"""

from __future__ import annotations

import asyncio

import pytest

from video_analysis.rate_limiter import (
    TokenBucketLimiter,
    DEFAULT_CAPACITY,
    DEFAULT_RATE,
)


class TestTokenBucketLimiter:
    """Unit tests for TokenBucketLimiter."""

    def test_init_defaults(self):
        limiter = TokenBucketLimiter()
        assert limiter.capacity == DEFAULT_CAPACITY
        assert limiter.rate == DEFAULT_RATE

    def test_init_custom_params(self):
        limiter = TokenBucketLimiter(capacity=50, rate=10.0)
        assert limiter.capacity == 50
        assert limiter.rate == 10.0

    @pytest.mark.asyncio
    async def test_consume_under_capacity(self):
        limiter = TokenBucketLimiter(capacity=5, rate=1.0)
        for _ in range(5):
            assert await limiter.consume("client_a")

    @pytest.mark.asyncio
    async def test_consume_exceeds_capacity(self):
        limiter = TokenBucketLimiter(capacity=3, rate=1.0)
        for _ in range(3):
            assert await limiter.consume("client_a")
        assert not await limiter.consume("client_a")

    @pytest.mark.asyncio
    async def test_reset_clears_all_buckets(self):
        """Reset restores full capacity."""
        limiter = TokenBucketLimiter(capacity=5, rate=1.0)
        for _ in range(5):
            await limiter.consume("client_a")
        assert not await limiter.consume("client_a")

        await limiter.reset()
        # After reset, should be able to consume again
        assert await limiter.consume("client_a")

    @pytest.mark.asyncio
    async def test_reset_single_client(self):
        limiter = TokenBucketLimiter(capacity=3, rate=1.0)
        await limiter.consume("a")
        await limiter.consume("a")
        await limiter.consume("a")
        assert not await limiter.consume("a")
        await limiter.reset("a")
        assert await limiter.consume("a")

    @pytest.mark.asyncio
    async def test_multiple_clients_independent(self):
        limiter = TokenBucketLimiter(capacity=2, rate=1.0)
        assert await limiter.consume("client_a")
        assert await limiter.consume("client_a")
        assert not await limiter.consume("client_a")
        # Client B has its own bucket
        assert await limiter.consume("client_b")
        assert await limiter.consume("client_b")
        assert not await limiter.consume("client_b")

    @pytest.mark.asyncio
    async def test_custom_key_abstraction(self):
        limiter = TokenBucketLimiter(capacity=3, rate=1.0)
        assert await limiter.consume("192.168.1.1")
        assert await limiter.consume("192.168.1.1")
        assert await limiter.consume("192.168.1.1")
        assert not await limiter.consume("192.168.1.1")
        # Different IP
        assert await limiter.consume("10.0.0.1")

    @pytest.mark.asyncio
    async def test_refill_over_time(self):
        """Wait for token refill."""
        limiter = TokenBucketLimiter(capacity=5, rate=5.0)
        for _ in range(5):
            await limiter.consume("client_a")
        assert not await limiter.consume("client_a")
        # Wait for ~1 token at 5/s rate
        await asyncio.sleep(0.25)
        assert await limiter.consume("client_a")

    @pytest.mark.asyncio
    async def test_negative_capacity(self):
        """Negative capacity should be treated as zero."""
        limiter = TokenBucketLimiter(capacity=-1)
        assert not await limiter.consume("key")

    @pytest.mark.asyncio
    async def test_consume_multiple_tokens(self):
        limiter = TokenBucketLimiter(capacity=10, rate=1.0)
        assert await limiter.consume("key", tokens=5)
        assert not await limiter.consume("key", tokens=10)  # only 5 left
        assert await limiter.consume("key", tokens=5)

    def test_string_representation(self):
        limiter = TokenBucketLimiter(capacity=5, rate=2.0)
        s = repr(limiter)
        assert "TokenBucketLimiter" in s

    @pytest.mark.asyncio
    async def test_concurrent_safety(self):
        """Multiple concurrent consume calls should not corrupt state."""
        limiter = TokenBucketLimiter(capacity=50, rate=10.0)

        async def consume_many(n: int):
            for _ in range(n):
                await limiter.consume("shared")

        await asyncio.gather(*[consume_many(10) for _ in range(5)])
        # 50 tokens consumed, should be empty
        assert not await limiter.consume("shared")
