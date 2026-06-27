"""
Tests for the structured error handling module (v0.49.0).

Tests ErrorDetail model, StandardHTTPError, register_error_handlers,
and the _CatchAllMiddleware for unhandled exceptions.
"""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from video_analysis.error_handlers import (
    ErrorDetail,
    StandardHTTPError,
    register_error_handlers,
)


class TestErrorDetailModel:
    """Tests for the ErrorDetail Pydantic model."""

    def test_error_detail_fields(self):
        err = ErrorDetail(
            detail="Video not found",
            error_code="VIDEO_NOT_FOUND",
            status_code=404,
            timestamp="2026-06-27T12:00:00Z",
            path="/api/videos/abc",
        )
        assert err.detail == "Video not found"
        assert err.error_code == "VIDEO_NOT_FOUND"
        assert err.status_code == 404
        assert err.timestamp == "2026-06-27T12:00:00Z"
        assert err.path == "/api/videos/abc"

    def test_error_detail_serializes_to_json(self):
        err = ErrorDetail(
            detail="Not found",
            error_code="NOT_FOUND",
            status_code=404,
            timestamp="2026-06-27T12:00:00Z",
            path="/api/resource",
        )
        data = json.loads(err.model_dump_json())
        assert data["detail"] == "Not found"
        assert data["status_code"] == 404


class TestStandardHTTPError:
    """Tests for the StandardHTTPError exception class."""

    def test_standard_http_error(self):
        err = StandardHTTPError(
            status_code=404,
            detail="Video not found",
            error_code="VIDEO_NOT_FOUND",
        )
        assert err.status_code == 404
        assert err.detail == "Video not found"
        assert err.error_code == "VIDEO_NOT_FOUND"

    def test_standard_http_error_default_code(self):
        """When error_code is None, it defaults to HTTP_{status_code}."""
        err = StandardHTTPError(status_code=400, detail="Bad request")
        assert err.error_code == "HTTP_400"

    def test_standard_http_error_custom_error_code(self):
        err = StandardHTTPError(
            status_code=500, detail="Internal", error_code="DB_ERROR"
        )
        assert err.error_code == "DB_ERROR"


class TestRegisterErrorHandlers:
    """Tests for register_error_handlers with a real FastAPI app."""

    def test_register_handlers_does_not_crash(self):
        app = FastAPI()
        register_error_handlers(app)
        assert len(app.exception_handlers) > 0

    def test_standard_http_error_via_app(self):
        app = FastAPI()

        @app.get("/test")
        def test_route():
            raise StandardHTTPError(
                status_code=404,
                detail="Item not found",
                error_code="ITEM_NOT_FOUND",
            )

        register_error_handlers(app)
        client = TestClient(app)

        resp = client.get("/test")
        assert resp.status_code == 404
        data = resp.json()
        assert data["detail"] == "Item not found"
        assert data["error_code"] == "ITEM_NOT_FOUND"
        assert data["status_code"] == 404
        assert "timestamp" in data
        assert "path" in data
        # Should NOT have WWW-Authenticate header
        assert "www-authenticate" not in {k.lower(): v for k, v in resp.headers.items()}

    def test_http_exception(self):
        app = FastAPI()

        @app.get("/test")
        def test_route():
            raise HTTPException(status_code=403, detail="Forbidden")

        register_error_handlers(app)
        client = TestClient(app)

        resp = client.get("/test")
        assert resp.status_code == 403
        data = resp.json()
        assert data["detail"] == "Forbidden"
        # error_code defaults to HTTP_{status_code}
        assert data["error_code"] == "HTTP_403"

    def test_http_exception_other_status(self):
        app = FastAPI()

        @app.get("/test")
        def test_route():
            raise HTTPException(status_code=409, detail="Conflict")

        register_error_handlers(app)
        client = TestClient(app)

        resp = client.get("/test")
        assert resp.status_code == 409
        data = resp.json()
        assert data["error_code"] == "HTTP_409"

    def test_validation_error(self):
        app = FastAPI()

        @app.get("/items/{item_id}")
        def get_item(item_id: int):
            return {"item_id": item_id}

        register_error_handlers(app)
        client = TestClient(app)

        resp = client.get("/items/not-an-int")
        assert resp.status_code == 422
        data = resp.json()
        assert "detail" in data
        assert data["error_code"] == "VALIDATION_ERROR"
        assert "errors" in data  # field-level validation errors

    def test_unhandled_exception_caught_by_middleware(self):
        """Unhandled exceptions (any that aren't HTTPException or StandardHTTPError)
        are caught by _CatchAllMiddleware and returned as 500."""
        app = FastAPI()

        @app.get("/test")
        def test_route():
            raise RuntimeError("Unexpected failure")

        register_error_handlers(app)
        client = TestClient(app)

        resp = client.get("/test")
        assert resp.status_code == 500
        data = resp.json()
        assert data["error_code"] == "INTERNAL_ERROR"
        assert "Internal server error" in data["detail"]

    def test_value_error_caught_by_middleware(self):
        app = FastAPI()

        @app.get("/test")
        def test_route():
            raise ValueError("Invalid value")

        register_error_handlers(app)
        client = TestClient(app)

        resp = client.get("/test")
        # ValueError is not a special-case exception, so middleware returns 500
        assert resp.status_code == 500
        data = resp.json()
        assert data["error_code"] == "INTERNAL_ERROR"

    def test_successful_request_passes_through(self):
        app = FastAPI()

        @app.get("/ok")
        def ok():
            return {"status": "ok"}

        register_error_handlers(app)
        client = TestClient(app)

        resp = client.get("/ok")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_no_authenticate_header(self):
        """Error responses should not include WWW-Authenticate."""
        app = FastAPI()

        @app.get("/test")
        def test_route():
            raise HTTPException(status_code=401, detail="Unauthorized")

        register_error_handlers(app)
        client = TestClient(app)

        resp = client.get("/test")
        headers_lower = {k.lower(): v for k, v in resp.headers.items()}
        assert "www-authenticate" not in headers_lower

    def test_multiple_error_types(self):
        """Multiple endpoint routes with different error types work."""
        app = FastAPI()

        @app.get("/not-found")
        def not_found():
            raise StandardHTTPError(
                status_code=404, detail="Missing", error_code="MISSING"
            )

        @app.get("/forbidden")
        def forbidden():
            raise HTTPException(status_code=403, detail="Nope")

        @app.get("/ok")
        def ok():
            return {"status": "ok"}

        register_error_handlers(app)
        client = TestClient(app)

        assert client.get("/not-found").status_code == 404
        assert client.get("/forbidden").status_code == 403
        assert client.get("/ok").status_code == 200
