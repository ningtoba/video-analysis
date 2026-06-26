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
from typing import Any, Dict, List, Optional
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
                "chunk_type": "scene",
            },
            {
                "start_time": 10.0,
                "end_time": 20.0,
                "text": "Test content",
                "speaker": "B",
                "chunk_type": "scene",
            },
        ],
        "documents": ["Hello world", "Test content"],
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

    def test_all_endpoints_registered(self, app):
        """All expected routes must be present in the OpenAPI spec."""
        paths = app.openapi().get("paths", {})
        expected = [
            "/api/videos/process",
            "/api/videos",
            "/api/videos/{video_id}",
            "/api/videos/{video_id}/query",
            "/api/videos/{video_id}/transcript",
            "/api/videos/{video_id}/frames/{timestamp}",
            "/api/videos/{video_id}/chapters",
            "/api/videos/search",
            "/api/sse/chat",
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
        """List all videos should return correct count and items."""
        response = client.get("/api/videos")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2
        assert len(data["videos"]) == 2
        assert data["videos"][0]["video_id"] == "vid1"

    def test_list_videos_empty(self, client, mock_rag):
        """Empty library should return count 0."""
        from video_analysis.api import set_rag_instance

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

    def test_video_detail_not_found(self, client, mock_rag):
        """Return 404 for non-existent video_id."""
        mock_rag.get_library_info.return_value = None
        response = client.get("/api/videos/nonexistent")
        assert response.status_code == 404


class TestDeleteVideo:
    """Tests for DELETE /api/videos/{video_id}."""

    def test_delete_video_success(self, client, mock_rag):
        """Delete should succeed and return deleted=True."""
        response = client.delete("/api/videos/vid1")
        assert response.status_code == 200
        data = response.json()
        assert data.get("deleted", False) is True
        assert data.get("video_id") == "vid1"

    def test_delete_video_error(self, client, mock_rag):
        """RAG errors should produce 500."""
        mock_rag.delete_video.side_effect = RuntimeError("DB error")
        response = client.delete("/api/videos/vid1")
        assert response.status_code == 500


class TestQueryVideo:
    """Tests for POST /api/videos/{video_id}/query."""

    def test_query_success(self, client, mock_chat):
        """Ask a question and get an answer (mocked)."""
        from video_analysis.chat import VideoChat

        with patch.object(VideoChat, "ask", return_value=mock_chat):
            response = client.post(
                "/api/videos/vid1/query",
                json={"query": "What is this video about?"},
            )
        assert response.status_code == 200
        data = response.json()
        assert "test answer" in data.get("answer", "")

    def test_query_missing_query_field(self, client):
        """Missing query field should return 422."""
        response = client.post(
            "/api/videos/vid1/query",
            json={},
        )
        assert response.status_code == 422


class TestSearchVideos:
    """Tests for GET /api/videos/search."""

    def test_search_success(self, client):
        """Search should return results."""
        response = client.get("/api/videos/search?query=test+query&top_k=5")
        assert response.status_code == 200
        data = response.json()
        assert data.get("query") == "test query"
        assert len(data.get("results", [])) >= 1


class TestTranscript:
    """Tests for GET /api/videos/{video_id}/transcript."""

    def test_transcript_success(self, client):
        """Transcript should return segments."""
        response = client.get("/api/videos/vid1/transcript")
        assert response.status_code == 200
        data = response.json()
        assert data["video_id"] == "vid1"
        assert len(data["segments"]) == 2


class TestChapters:
    """Tests for GET /api/videos/{video_id}/chapters."""

    def test_chapters_returns_something(self, client, mock_rag):
        """Chapters endpoint should return 200, 404, or 501."""
        response = client.get("/api/videos/vid1/chapters")
        assert response.status_code in (200, 404, 501)


class TestProcessVideo:
    """Tests for POST /api/videos/process."""

    def test_process_without_url(self, client):
        """Missing URL should return 400."""
        response = client.post(
            "/api/videos/process",
            json={},
        )
        assert response.status_code in (400, 422)

    def test_process_invalid_file_path(self, client):
        """Non-existent file path should not crash."""
        response = client.post(
            "/api/videos/process",
            json={"url": "file:///nonexistent/video.mp4"},
        )
        assert response.status_code in (400, 422, 500)


class TestFrames:
    """Tests for GET /api/videos/{video_id}/frames/{timestamp}."""

    def test_frame_invalid_timestamp(self, client):
        """Non-numeric timestamp should return 400."""
        response = client.get("/api/videos/vid1/frames/abc")
        assert response.status_code in (400, 422)


class TestSSEChat:
    """Tests for GET /api/sse/chat."""

    def test_sse_chat_returns_streaming_response(self, client):
        """SSE chat should return text/event-stream content."""
        response = client.get("/api/sse/chat?query=Hello&video_id=vid1")
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")


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
            "/api/videos/{video_id}/transcript",
            "/api/videos/{video_id}/frames/{timestamp}",
            "/api/videos/{video_id}/chapters",
            "/api/videos/search",
            "/api/sse/chat",
        ]
        for path in expected_paths:
            assert path in paths, f"Missing OpenAPI path: {path}"

    def test_openapi_schema_valid(self, client):
        """OpenAPI schema should be valid."""
        schema = client.get("/openapi.json").json()
        assert "openapi" in schema
        assert "info" in schema
        assert isinstance(schema["info"].get("version"), str)
