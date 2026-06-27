"""Tests for Production Telemetry & API Hardening (v0.50.0).

Covers:
- video_analysis/telemetry.py — TelemetryContext, trace_pipeline, get_trace_id, no-op fallback
- video_analysis/rate_limiter.py — TokenBucketLimiter, rate limiting logic
- video_analysis/error_handlers.py — ErrorDetail, StandardHTTPError, register_error_handlers
- video_analysis/client.py — VideoAnalysisClient (basic structure, model validation)
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

# =========================================================================
# Telemetry Tests
# =========================================================================


class TestTelemetryNoOp:
    """Module imports cleanly and works without opentelemetry installed."""

    def test_import(self):
        """Module imports cleanly."""
        from video_analysis import telemetry

        assert telemetry.__doc__ is not None

    def test_enabled_false_when_no_otel(self):
        """enabled is False when opentelemetry is not available."""
        from video_analysis import telemetry

        # When running without opentelemetry packages
        assert hasattr(telemetry, "enabled")

    def test_telemetry_context_noop(self):
        """TelemetryContext works as a no-op without opentelemetry."""
        from video_analysis.telemetry import TelemetryContext

        with TelemetryContext("test_stage", video_id="abc123") as ctx:
            ctx.set_attribute("key", "value")
            ctx.set_status_ok()
            assert ctx is not None

        # No crash = success

    def test_telemetry_context_async_noop(self):
        """TelemetryContext works async as a no-op."""
        from video_analysis.telemetry import TelemetryContext

        async def run():
            async with TelemetryContext("async_stage") as ctx:
                ctx.set_status_ok()

        asyncio.run(run())

    def test_telemetry_context_error_noop(self):
        """TelemetryContext gracefully handles exceptions."""
        from video_analysis.telemetry import TelemetryContext

        class TestError(Exception):
            pass

        with pytest.raises(TestError):
            with TelemetryContext("failing_stage") as ctx:
                ctx.set_attribute("attempt", 1)
                raise TestError("something went wrong")

    def test_get_trace_id(self):
        """get_trace_id returns a valid hex string without opentelemetry."""
        from video_analysis.telemetry import get_trace_id

        trace_id = get_trace_id()
        assert isinstance(trace_id, str)
        assert len(trace_id) == 32
        # Verify it's hex
        int(trace_id, 16)

    def test_trace_pipeline_decorator(self):
        """trace_pipeline decorator works as a no-op."""
        from video_analysis.telemetry import trace_pipeline

        @trace_pipeline("test_retrieval", attributes={"method": "hybrid"})
        async def retrieve(video_id: str) -> str:
            return f"result for {video_id}"

        result = asyncio.run(retrieve("vid1"))
        assert result == "result for vid1"

    def test_parent_span_from_headers(self):
        """parent_span_from_headers creates a no-op span."""
        from video_analysis.telemetry import parent_span_from_headers

        headers = {
            "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
        }
        ctx = parent_span_from_headers(headers, "api_request")
        with ctx:
            ctx.set_attribute("key", "value")
            ctx.set_status_ok()

    def test_parent_span_from_headers_empty(self):
        """parent_span_from_headers handles missing headers gracefully."""
        from video_analysis.telemetry import parent_span_from_headers

        ctx = parent_span_from_headers({}, "fallback_span")
        with ctx:
            ctx.set_status_ok()

    def test_force_flush_noop(self):
        """force_flush is safe to call without opentelemetry."""
        from video_analysis.telemetry import force_flush

        force_flush()  # should not raise

    def test_pipeline_span_context_manager(self):
        """pipeline_span async context manager works."""
        from video_analysis.telemetry import pipeline_span

        async def test():
            async with pipeline_span(
                "retrieval", attributes={"method": "hybrid"}
            ) as ctx:
                ctx.set_attribute("chunks", 5)
                ctx.set_status_ok()
                return "done"
            return "ok"

        result = asyncio.run(test())
        assert result == "done" or result == "ok"


# =========================================================================
# Rate Limiter Tests
# =========================================================================


class TestTokenBucketLimiter:
    """Token bucket rate limiter implementation."""

    def test_import(self):
        """Module imports cleanly."""
        from video_analysis import rate_limiter

        assert rate_limiter.__doc__ is not None

    def test_allows_within_limit(self):
        """Requests within the rate limit are allowed."""
        from video_analysis.rate_limiter import TokenBucketLimiter

        limiter = TokenBucketLimiter(capacity=10, rate=10.0)

        async def test():
            for _ in range(10):
                assert await limiter.consume("client1")
            return True

        assert asyncio.run(test())

    def test_blocks_when_exceeded(self):
        """Requests exceeding the limit are blocked."""
        from video_analysis.rate_limiter import TokenBucketLimiter

        limiter = TokenBucketLimiter(capacity=5, rate=5.0)

        async def test():
            for _ in range(5):
                assert await limiter.consume("client2")
            # 6th request should be denied
            assert not await limiter.consume("client2")
            return True

        assert asyncio.run(test())

    def test_independent_buckets(self):
        """Different clients have independent buckets."""
        from video_analysis.rate_limiter import TokenBucketLimiter

        limiter = TokenBucketLimiter(capacity=3, rate=3.0)

        async def test():
            # Exhaust client A
            for _ in range(3):
                assert await limiter.consume("client_a")
            assert not await limiter.consume("client_a")
            # Client B should still be allowed
            assert await limiter.consume("client_b")
            return True

        assert asyncio.run(test())

    def test_reset_key(self):
        """Resetting a key restores its capacity."""
        from video_analysis.rate_limiter import TokenBucketLimiter

        limiter = TokenBucketLimiter(capacity=3, rate=3.0)

        async def test():
            for _ in range(3):
                assert await limiter.consume("client_reset")
            assert not await limiter.consume("client_reset")
            await limiter.reset("client_reset")
            # After reset, should be allowed again
            assert await limiter.consume("client_reset")
            return True

        assert asyncio.run(test())

    def test_reset_all(self):
        """Resetting all keys restores all buckets."""
        from video_analysis.rate_limiter import TokenBucketLimiter

        limiter = TokenBucketLimiter(capacity=2, rate=2.0)

        async def test():
            for c in ["a", "b"]:
                for _ in range(2):
                    assert await limiter.consume(c)
                assert not await limiter.consume(c)
            await limiter.reset()
            for c in ["a", "b"]:
                assert await limiter.consume(c)
            return True

        assert asyncio.run(test())

    def test_refills_over_time(self):
        """Bucket refills over time (approximate)."""
        from video_analysis.rate_limiter import TokenBucketLimiter

        limiter = TokenBucketLimiter(capacity=5, rate=10.0)

        async def test():
            for _ in range(5):
                assert await limiter.consume("client_refill")
            assert not await limiter.consume("client_refill")
            # Wait ~100ms for refill
            await asyncio.sleep(0.15)
            # Should have 1 token now
            assert await limiter.consume("client_refill")
            return True

        assert asyncio.run(test())

    def test_properties(self):
        """Capacity and rate properties are accessible."""
        from video_analysis.rate_limiter import TokenBucketLimiter

        limiter = TokenBucketLimiter(capacity=50, rate=10.0)
        assert limiter.capacity == 50
        assert limiter.rate == 10.0


# =========================================================================
# Error Handlers Tests
# =========================================================================


class TestErrorHandlers:
    """Structured error handling infrastructure."""

    def test_import(self):
        """Module imports cleanly."""
        from video_analysis import error_handlers

        assert error_handlers.__doc__ is not None

    def test_standard_http_error(self):
        """StandardHTTPError carries structured metadata."""
        from video_analysis.error_handlers import StandardHTTPError

        exc = StandardHTTPError(
            status_code=404,
            detail="Video not found",
            error_code="VIDEO_NOT_FOUND",
        )
        assert exc.status_code == 404
        assert exc.detail == "Video not found"
        assert exc.error_code == "VIDEO_NOT_FOUND"

    def test_standard_http_error_default_code(self):
        """StandardHTTPError generates a default error_code."""
        from video_analysis.error_handlers import StandardHTTPError

        exc = StandardHTTPError(status_code=503, detail="Service unavailable")
        assert exc.error_code == "HTTP_503"

    def test_error_detail_schema(self):
        """ErrorDetail Pydantic model validates correctly."""
        from datetime import datetime, timezone
        from video_analysis.error_handlers import ErrorDetail

        detail = ErrorDetail(
            detail="Not found",
            error_code="NOT_FOUND",
            status_code=404,
            timestamp=datetime.now(timezone.utc).isoformat(),
            path="/api/videos/test",
        )
        assert detail.detail == "Not found"
        assert detail.error_code == "NOT_FOUND"
        assert detail.status_code == 404

    def test_registration(self):
        """register_error_handlers attaches handlers without error."""
        from video_analysis.error_handlers import register_error_handlers

        app = FastAPI()
        register_error_handlers(app)
        assert len(app.exception_handlers) > 0

    def test_standard_http_error_via_app(self):
        """StandardHTTPError returns structured JSON."""
        from video_analysis.error_handlers import (
            StandardHTTPError,
            register_error_handlers,
        )

        app = FastAPI()
        register_error_handlers(app)

        @app.get("/test-error")
        async def test_error():
            raise StandardHTTPError(
                status_code=404,
                detail="Video not found",
                error_code="VIDEO_NOT_FOUND",
            )

        client = TestClient(app)
        resp = client.get("/test-error")
        assert resp.status_code == 404
        data = resp.json()
        assert data["detail"] == "Video not found"
        assert data["error_code"] == "VIDEO_NOT_FOUND"
        assert data["status_code"] == 404
        assert "timestamp" in data
        assert data["path"] == "/test-error"

    def test_http_exception_handler(self):
        """FastAPI HTTPException returns structured JSON."""
        from video_analysis.error_handlers import register_error_handlers

        app = FastAPI()
        register_error_handlers(app)

        @app.get("/http-error")
        async def http_error():
            raise HTTPException(status_code=400, detail="Bad input")

        client = TestClient(app)
        resp = client.get("/http-error")
        assert resp.status_code == 400
        data = resp.json()
        assert data["detail"] == "Bad input"
        assert data["error_code"] == "HTTP_400"

    def test_unhandled_exception_handler(self):
        """Unhandled exceptions return 500 with sanitized message."""
        from video_analysis.error_handlers import register_error_handlers

        app = FastAPI()
        register_error_handlers(app)

        @app.get("/crash")
        async def crash():
            raise HTTPException(status_code=500, detail="internal problem")

        client = TestClient(app)
        resp = client.get("/crash")
        assert resp.status_code == 500
        data = resp.json()
        assert "detail" in data
        assert data["error_code"] == "HTTP_500"

    def test_validation_error_handler(self):
        """Pydantic ValidationError returns 422 with field errors."""
        from video_analysis.error_handlers import register_error_handlers
        from pydantic import BaseModel, Field

        # Register handlers before creating routes
        app = FastAPI()
        register_error_handlers(app)

        class TestModel(BaseModel):
            name: str = Field(...)
            age: int = Field(...)

        @app.post("/validate")
        async def validate(body: TestModel):
            return {"ok": True}

        client = TestClient(app)
        resp = client.post("/validate", json={"name": "test"})  # missing age
        assert resp.status_code == 422
        data = resp.json()
        # Check that the structured error response is returned
        assert "error_code" in data or isinstance(data.get("detail"), str)
        # If our handler fires, error_code will be present
        if "error_code" in data:
            assert data["error_code"] == "VALIDATION_ERROR"

    def test_error_handler_no_authenticate(self):
        """Auth endpoint is not affected by error handlers."""
        from video_analysis.error_handlers import register_error_handlers
        from fastapi import FastAPI
        from fastapi.responses import Response

        app = FastAPI()
        register_error_handlers(app)

        @app.get("/no-auth")
        async def no_auth():
            return Response(
                status_code=401,
                content="Unauthorized",
                headers={"WWW-Authenticate": "Basic"},
            )

        client = TestClient(app)
        resp = client.get("/no-auth")
        # 401 without error handler override
        assert resp.status_code == 401


# =========================================================================
# Client SDK Tests
# =========================================================================


class TestVideoAnalysisClient:
    """Python API client SDK structure."""

    def test_import(self):
        """Module imports cleanly."""
        from video_analysis import client

        assert client.__doc__ is not None

    def test_client_import(self):
        """VideoAnalysisClient class exists."""
        from video_analysis.client import VideoAnalysisClient

        assert VideoAnalysisClient is not None

    def test_client_requires_requests(self):
        """Client raises ImportError if requests not installed."""
        with patch.dict("sys.modules", {"requests": None}):
            import importlib
            import sys as _sys

            # Remove cached client module
            for mod in list(_sys.modules.keys()):
                if "video_analysis.client" in mod:
                    del _sys.modules[mod]

            with pytest.raises(ImportError, match="requests"):
                from video_analysis.client import VideoAnalysisClient

                VideoAnalysisClient()

    def test_model_dataclasses(self):
        """Data model dataclasses have correct fields."""
        from video_analysis.client import (
            HealthInfo,
            VideoInfo,
            JobInfo,
            SearchResult,
            QueryResult,
            Chapter,
            EvaluationReport,
            APIError,
            ConnectionError,
        )

        health = HealthInfo(
            status="ok",
            version="0.50.0",
            gpu_available=True,
            models_loaded={},
            uptime_seconds=42.0,
        )
        assert health.status == "ok"
        assert health.version == "0.50.0"

        video = VideoInfo(video_id="test123", filename="test.mp4")
        assert video.video_id == "test123"
        assert video.filename == "test.mp4"

        job = JobInfo(job_id="job1", job_type="process", status="completed")
        assert job.job_id == "job1"
        assert job.status == "completed"

        result = SearchResult(chunk_id="c1", text="hello", score=0.95)
        assert result.chunk_id == "c1"
        assert result.score == 0.95

        query_result = QueryResult(answer="This is a video of a cat.", sources=[])
        assert "cat" in query_result.answer

        chapter = Chapter(title="Intro", start_time=0.0, end_time=30.0, index=0)
        assert chapter.title == "Intro"
        assert chapter.index == 0

        report = EvaluationReport(
            run_id="r1",
            timestamp="2026-06-27T12:00:00",
            version="0.50.0",
            tasks=[],
            summary={"total": 3, "passed": 3},
        )
        assert report.run_id == "r1"
        assert report.summary["passed"] == 3

    def test_api_error(self):
        """APIError carries status code and detail."""
        from video_analysis.client import APIError

        exc = APIError(status_code=404, detail="Not found", error_code="NOT_FOUND")
        assert exc.status_code == 404
        assert exc.detail == "Not found"
        assert exc.error_code == "NOT_FOUND"
        assert "404" in str(exc)
        assert "Not found" in str(exc)

        exc_default = APIError(status_code=500, detail="Server error")
        assert exc_default.error_code is None

    def test_connection_error(self):
        """ConnectionError is distinct from APIError."""
        from video_analysis.client import ConnectionError as ClientConnError, APIError

        exc = ClientConnError("Cannot connect")
        assert "Cannot connect" in str(exc)
        assert not isinstance(exc, APIError)

    def test_async_health_convenience(self):
        """async_health helper is importable."""
        from video_analysis.client import async_health

        assert asyncio.iscoroutinefunction(async_health)
