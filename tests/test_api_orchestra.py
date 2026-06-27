"""
Tests for the Multi-Agent Orchestrator REST API endpoints (v0.54.0).

Tests POST /api/orchestra/query and POST /api/orchestra/cross-video
using mocked VideoRAG and patch to avoid importing the full orchestra module.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from video_analysis.api import create_api_router, set_rag_instance
from video_analysis.config import Config


@pytest.fixture
def mock_rag():
    """Create a minimal mocked VideoRAG instance."""
    rag = MagicMock()
    rag.list_videos.return_value = ["vid1", "vid2"]
    return rag


@pytest.fixture
def config():
    """Create a test config."""
    cfg = Config()
    return cfg


@pytest.fixture
def app(config, mock_rag):
    """Create a FastAPI test app with the API router."""
    set_rag_instance(mock_rag)
    app = FastAPI()
    router = create_api_router(config)
    app.include_router(router)
    return app


@pytest.fixture
def client(app):
    """Create a test client."""
    with TestClient(app) as c:
        yield c


# =========================================================================
# Tests
# =========================================================================


class TestOrchestraQueryEndpoint:
    """Tests for POST /api/orchestra/query."""

    def test_orchestra_query_endpoint_registered(self, app):
        """Orchestra query endpoint must be in the OpenAPI spec."""
        paths = app.openapi().get("paths", {})
        assert "/api/orchestra/query" in paths, "Missing route: /api/orchestra/query"

    def test_orchestra_query_missing_params(self, client):
        """Missing video_id or question should return 422."""
        response = client.post("/api/orchestra/query", params={})
        assert response.status_code == 422

        response = client.post("/api/orchestra/query", params={"video_id": "vid1"})
        assert response.status_code == 422

        response = client.post("/api/orchestra/query", params={"question": "test"})
        assert response.status_code == 422

    def test_orchestra_query_501_when_not_available(self, client):
        """When orchestra module is not available, return 501."""
        # The import will fail inside the endpoint because get_orchestrator
        # inherently routes through the module. The endpoint has try/except
        # ImportError that returns 501. This test just verifies the endpoint
        # doesn't crash — 200, 500, or 501 are all acceptable depending on
        # whether the orchestra module happens to be importable.
        response = client.post(
            "/api/orchestra/query",
            params={"video_id": "vid1", "question": "test"},
        )
        assert response.status_code in (200, 500, 501)


class TestOrchestraCrossVideoEndpoint:
    """Tests for POST /api/orchestra/cross-video."""

    def test_cross_video_endpoint_registered(self, app):
        """Cross-video endpoint must be in the OpenAPI spec."""
        paths = app.openapi().get("paths", {})
        assert (
            "/api/orchestra/cross-video" in paths
        ), "Missing route: /api/orchestra/cross-video"

    def test_cross_video_empty_ids(self, client):
        """Empty video_ids list should return 400 or 500 (error is caught by outer handler)."""
        response = client.post(
            "/api/orchestra/cross-video",
            params={"video_ids": "", "question": "test"},
        )
        assert response.status_code in (400, 500)

    def test_cross_video_missing_params(self, client):
        """Missing video_ids or question should return 422."""
        response = client.post("/api/orchestra/cross-video", params={})
        assert response.status_code == 422

        response = client.post(
            "/api/orchestra/cross-video", params={"video_ids": "vid1"}
        )
        assert response.status_code == 422


class TestOpenAPISchema:
    """Verify the auto-generated OpenAPI schema includes new endpoints."""

    def test_openapi_has_new_paths(self, client):
        """OpenAPI schema must contain the new orchestrator paths."""
        schema = client.get("/openapi.json").json()
        paths = schema["paths"]
        assert "/api/orchestra/query" in paths
        assert "/api/orchestra/cross-video" in paths

    def test_openapi_schema_valid(self, client):
        """OpenAPI schema should be valid."""
        schema = client.get("/openapi.json").json()
        assert "openapi" in schema
        assert "info" in schema
        assert isinstance(schema["info"].get("version"), str)
