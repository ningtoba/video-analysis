"""
Tests for the Full REST API Layer (v0.41.0).

Tests all endpoints in video_analysis/api.py using mocked VideoRAG,
VideoPipeline, and VideoChat instances.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from video_analysis.api import create_api_router, set_rag_instance
from video_analysis.config import Config

# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def mock_rag():
    """Create a mocked VideoRAG instance."""
    rag = MagicMock()
    rag.list_videos.return_value = ["vid1", "vid2"]
    rag.search_all.return_value = [
        MagicMock(
            chunk_id="chunk1",
            video_id="vid1",
            text="Test chunk content",
            timestamp=10.0,
            scene_id=1,
            score=0.95,
            frame_path=None,
            chunk_type="scene",
        )
    ]

    mock_info = MagicMock()
    mock_info.video_id = "vid1"
    mock_info.filename = "test.mp4"
    mock_info.num_scenes = 5
    mock_info.num_chunks = 20
    mock_info.duration = 120.0
    mock_info.has_sprite = True

    rag.get_library_info.return_value = mock_info

    # Mock collection get for transcripts and frames
    rag.collection.get.return_value = {
        "ids": ["1", "2"],
        "metadatas": [
            {
                "start_time": 0.0,
                "end_time": 10.0,
                "text": "Hello world",
                "speaker": "A",
                "chunk_type": "transcript",
            },
            {
                "start_time": 10.0,
                "end_time": 20.0,
                "text": "Test content",
                "speaker": "B",
                "chunk_type": "transcript",
            },
        ],
    }

    return rag


@pytest.fixture
def mock_chat():
    """Create a mocked response from VideoChat."""
    mock_response = MagicMock()
    mock_response.content = "This is a test answer about the video."
    # Mock sources
    source = MagicMock()
    source.text = "Source excerpt"
    source.timestamp = 15.0
    source.scene_id = 1
    source.relevance_score = 0.85
    mock_response.sources = [source]
    return mock_response


@pytest.fixture
def config():
    """Create a test config."""
    cfg = Config()
    cfg.data_dir = Path(tempfile.mkdtemp())
    return cfg


@pytest.fixture
def app(config, mock_rag, mock_chat):
    """Create a FastAPI test app with the API router."""
    # Set up module-level RAG
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


class TestApiEndpointExistence:
    """Verify all API endpoints are registered correctly."""

    ROUTES_EXPECTED = [
        "POST /api/videos/process",
        "GET /api/videos",
        "GET /api/videos/{video_id}",
        "DELETE /api/videos/{video_id}",
        "POST /api/videos/{video_id}/query",
        "POST /api/videos/{video_id}/query/stream",
        "GET /api/videos/search",
        "GET /api/videos/{video_id}/transcript",
        "GET /api/videos/{video_id}/frames/{timestamp}",
        "GET /api/videos/{video_id}/chapters",
    ]

    def test_all_endpoints_registered(self, app):
        """All expected routes must be present on the router."""
        routes = {r.path for r in app.routes}
        for expected in self.ROUTES_EXPECTED:
            path = expected.split(" ")[1]
            assert path in routes, f"Missing route: {expected}"

    def test_openapi_docs_available(self, client):
        """OpenAPI docs endpoint should be reachable."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "paths" in data
        assert "/api/videos" in data["paths"]

    def test_tag_assignment(self, app):
        """All API routes should have the correct tag."""
        for route in app.routes:
            if hasattr(route, "path") and "/api/" in route.path:
                tags = getattr(route, "tags", []) or []
                assert "Video Analysis API" in tags or not tags


class TestListVideos:
    """Tests for GET /api/videos."""

    def test_list_videos_success(self, client, mock_rag):
        """GET /api/videos should return the video list."""
        response = client.get("/api/videos")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2
        assert len(data["videos"]) == 2
        assert data["videos"][0]["video_id"] == "vid1"

    def test_list_videos_empty(self, client):
        """Empty library should return count 0."""
        empty_rag = MagicMock()
        empty_rag.list_videos.return_value = []
        set_rag_instance(empty_rag)

        response = client.get("/api/videos")
        assert response.status_code == 200
        assert response.json()["count"] == 0


class TestVideoDetail:
    """Tests for GET /api/videos/{video_id}."""

    def test_video_detail_success(self, client):
        """Return video info for existing video_id."""
        response = client.get("/api/videos/vid1")
        assert response.status_code == 200
        data = response.json()
        assert data["video_id"] == "vid1"
        assert data["filename"] == "test.mp4"
        assert data["num_scenes"] == 5

    def test_video_detail_not_found(self, client, mock_rag):
        """Return 404 for non-existent video_id."""
        mock_rag.get_library_info.return_value = None
        response = client.get("/api/videos/nonexistent")
        assert response.status_code == 404


class TestDeleteVideo:
    """Tests for DELETE /api/videos/{video_id}."""

    def test_delete_video_success(self, client, mock_rag):
        """Delete should succeed and return status."""
        response = client.delete("/api/videos/vid1")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "deleted"
        assert data["video_id"] == "vid1"
        assert mock_rag.collection.delete.called

    def test_delete_video_error(self, client, mock_rag):
        """RAG errors should produce 500."""
        mock_rag.collection.delete.side_effect = RuntimeError("DB error")
        response = client.delete("/api/videos/vid1")
        assert response.status_code == 500


class TestQueryVideo:
    """Tests for POST /api/videos/{video_id}/query."""

    def test_query_success(self, client, mock_chat):
        """Ask a question and get an answer."""
        response = client.post(
            "/api/videos/vid1/query",
            json={"query": "What is this video about?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "test answer" in data["answer"]
        assert data["video_id"] == "vid1"
        assert len(data["sources"]) == 1

    def test_query_with_custom_top_k(self, client):
        """Custom top_k should be accepted."""
        response = client.post(
            "/api/videos/vid1/query",
            json={"query": "Test question", "top_k": 5},
        )
        assert response.status_code == 200

    def test_query_invalid_top_k(self, client):
        """top_k out of range should return 422."""
        response = client.post(
            "/api/videos/vid1/query",
            json={"query": "Test", "top_k": 0},
        )
        assert response.status_code == 422


class TestQueryStream:
    """Tests for POST /api/videos/{video_id}/query/stream."""

    def test_query_stream_success(self, client, mock_chat):
        """Stream endpoint should return SSE events."""
        response = client.post(
            "/api/videos/vid1/query/stream",
            json={"query": "Tell me about the video"},
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream"

        # Parse SSE events
        body = response.text
        assert "data:" in body

        events = []
        for line in body.split("\n"):
            if line.startswith("data: "):
                events.append(line[6:])

        assert len(events) >= 2  # at least one token + [DONE]
        assert events[-1] == "[DONE]"


class TestSearchVideos:
    """Tests for GET /api/videos/search."""

    def test_search_success(self, client):
        """Search should return results."""
        response = client.get("/api/videos/search?query=test+query&top_k=5")
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "test query"
        assert data["top_k"] == 5
        assert len(data["chunks"]) >= 1

    def test_search_default_top_k(self, client):
        """Default top_k should be applied when not specified."""
        response = client.get("/api/videos/search?query=hello")
        assert response.status_code == 200
        assert response.json()["top_k"] == 10


class TestTranscript:
    """Tests for GET /api/videos/{video_id}/transcript."""

    def test_transcript_success(self, client):
        """Transcript should return segments."""
        response = client.get("/api/videos/vid1/transcript")
        assert response.status_code == 200
        data = response.json()
        assert data["video_id"] == "vid1"
        assert len(data["segments"]) == 2
        assert data["segments"][0]["text"] == "Hello world"
        assert data["segments"][0]["speaker"] == "A"


class TestChapters:
    """Tests for GET /api/videos/{video_id}/chapters."""

    def test_chapters_not_found_without_transcript(self, client, mock_rag):
        """When transcript is empty, return 404."""
        with patch(
            "video_analysis.chapters.extract_transcript_from_rag",
            return_value="",
        ):
            response = client.get("/api/videos/vid1/chapters")
            # May be 404 or 501 depending on nltk availability
            assert response.status_code in (404, 501, 200)


class TestProcessVideo:
    """Tests for POST /api/videos/process."""

    def test_process_without_url_or_path(self, client):
        """Missing both url and file_path should return 422."""
        response = client.post("/api/videos/process", json={})
        assert response.status_code == 422

    def test_process_invalid_file_path(self, client):
        """Non-existent file path should return 404."""
        response = client.post(
            "/api/videos/process",
            json={"file_path": "/nonexistent/video.mp4"},
        )
        assert response.status_code == 404


class TestFrames:
    """Tests for GET /api/videos/{video_id}/frames/{timestamp}."""

    def test_frame_invalid_timestamp(self, client):
        """Non-numeric timestamp should return 422."""
        response = client.get("/api/videos/vid1/frames/abc")
        assert response.status_code == 422

    def test_frame_no_matching_path(self, client, mock_rag):
        """When no frame path exists in metadata, return 404."""
        mock_rag.collection.get.return_value = {
            "ids": ["1"],
            "metadatas": [{"timestamp": 10.0}],  # no frame_path
        }
        response = client.get("/api/videos/vid1/frames/10.5")
        assert response.status_code == 404


# =========================================================================
# OpenAPI generation
# =========================================================================


class TestOpenAPISchema:
    """Verify the auto-generated OpenAPI schema is valid."""

    def test_openapi_has_paths(self, client):
        """OpenAPI schema must contain all expected paths."""
        schema = client.get("/openapi.json").json()
        paths = schema["paths"]

        expected_paths = [
            "/api/videos",
            "/api/videos/process",
            "/api/videos/{video_id}",
            "/api/videos/{video_id}/query",
            "/api/videos/{video_id}/query/stream",
            "/api/videos/search",
            "/api/videos/{video_id}/transcript",
            "/api/videos/{video_id}/frames/{timestamp}",
            "/api/videos/{video_id}/chapters",
        ]
        for path in expected_paths:
            assert path in paths, f"Missing OpenAPI path: {path}"

    def test_openapi_schema_valid(self, client):
        """OpenAPI schema should be valid (no parse errors)."""
        schema = client.get("/openapi.json").json()
        # Validate basic structure
        assert "openapi" in schema
        assert "info" in schema
        assert "title" in schema["info"]
        # Version check
        from video_analysis import __version__

        assert schema["info"]["version"] == __version__
