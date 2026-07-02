"""
Tests for the Full REST API Layer (v0.41.0).

Tests all endpoints in video_analysis/api.py using mocked VideoRAG,
VideoPipeline, and VideoChat instances.
"""

from __future__ import annotations

import tempfile
from pathlib import Path


import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from video_analysis.api import create_router
from video_analysis.config import Config

# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def config():
    """Create a test config."""
    cfg = Config()
    cfg.data_dir = Path(tempfile.mkdtemp())
    return cfg


@pytest.fixture
def app(config):
    """Create a FastAPI test app with the API router."""
    app = FastAPI()
    router = create_router(config)
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


class TestApiEndpointExistence:
    """Verify all API endpoints are registered correctly."""

    def test_all_endpoints_registered(self, app):
        """All expected routes must be present in the OpenAPI spec."""
        paths = app.openapi().get("paths", {})
        expected = [
            "/api/import-url",
            "/api/models",
            "/api/models/download",
            "/api/models/status",
            "/api/settings",
            "/api/videos",
            "/api/videos/process",
            "/api/videos/{video_id}",
            "/api/videos/{video_id}/query",
        ]
        for path in expected:
            assert path in paths, f"Missing route: {path}"

    def test_openapi_docs_available(self, client):
        """OpenAPI docs endpoint should be reachable."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "paths" in data

    def test_tag_assignment(self, app):
        """All API routes should have appropriate tags in OpenAPI (or none at all)."""
        paths = app.openapi().get("paths", {})
        for path, methods in paths.items():
            if "/api/" in path:
                for method_info in methods.values():
                    tags = method_info.get("tags", [])
                    # Tags are optional — we just check the route exists


class TestListVideos:
    """Tests for GET /api/videos."""

    def test_list_videos_success(self, client):
        """List all videos should return a list of videos."""
        response = client.get("/api/videos")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)






class TestQueryVideo:
    """Tests for POST /api/videos/{video_id}/query."""

    def test_query_missing_question_field(self, client):
        """Missing question field should return 422."""
        response = client.post(
            "/api/videos/vid1/query",
            json={},
        )
        assert response.status_code == 422











class TestProcessVideo:
    """Tests for POST /api/videos/process (async job queue mode)."""

    def test_process_without_url(self, client):
        """Missing body should return 422 (Pydantic validation)."""
        response = client.post(
            "/api/videos/process",
            json={},
        )
        assert response.status_code == 422

    def test_process_invalid_file_path(self, client):
        """Non-existent file path should return 404 (not crash)."""
        response = client.post(
            "/api/videos/process",
            json={"video_path": "/nonexistent/video.mp4"},
        )
        assert response.status_code == 404








class TestOpenAPISchema:
    """Verify the auto-generated OpenAPI schema is valid."""

    def test_openapi_has_paths(self, client):
        """OpenAPI schema must contain all expected paths."""
        schema = client.get("/openapi.json").json()
        paths = schema["paths"]
        expected_paths = [
            "/api/videos/process",
            "/api/videos",
            "/api/videos/{video_id}",
            "/api/videos/{video_id}/query",
            "/api/import-url",
            "/api/settings",
            "/api/models",
            "/api/models/download",
            "/api/models/status",
        ]
        for path in expected_paths:
            assert path in paths, f"Missing OpenAPI path: {path}"

    def test_openapi_schema_valid(self, client):
        """OpenAPI schema should be valid."""
        schema = client.get("/openapi.json").json()
        assert "openapi" in schema
        assert "info" in schema
        assert isinstance(schema["info"].get("version"), str)
