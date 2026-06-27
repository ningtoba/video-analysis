"""Tests for Knowledge Graph & Pipeline Health Monitor REST API endpoints (v0.54.0).

Tests the new endpoints added to video_analysis/api.py:
- GET /api/kg/stats
- GET /api/kg/entities
- GET /api/kg/timeline
- GET /api/kg/entities/{entity_id}/relationships
- GET /api/kg/videos/{video_id}/entities
- GET /api/kg/context
- GET /api/health/runs
- GET /api/health/summary
- GET /api/health/alerts
- POST /api/health/alerts/{alert_id}/acknowledge
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from video_analysis.api import create_api_router
from video_analysis.config import Config

# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def app() -> FastAPI:
    """Create a fresh FastAPI app with the API router."""
    app = FastAPI()
    config = Config()
    router = create_api_router(config)
    app.include_router(router)
    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def temp_kg_path(tmp_path: Path) -> Path:
    """Create a config pointing at a temp data dir for SQLite KG."""
    return tmp_path


@pytest.fixture
def kg_config(temp_kg_path: Path) -> Config:
    """Config with temp data directory for isolated tests."""
    cfg = Config()
    cfg.data_dir = temp_kg_path
    return cfg


@pytest.fixture
def kg_app(kg_config: Config) -> FastAPI:
    """FastAPI app using a temp-data-dir config for isolated KG/Health."""
    app = FastAPI()
    router = create_api_router(kg_config)
    app.include_router(router)
    return app


@pytest.fixture
def kg_client(kg_app: FastAPI) -> TestClient:
    return TestClient(kg_app)


# =========================================================================
# Knowledge Graph API tests (v0.54.0)
# =========================================================================


class TestKGAPI:
    """Tests for knowledge graph REST API endpoints."""

    def test_kg_stats_endpoint(self, client: TestClient):
        """GET /api/kg/stats returns stats dict with proper shape."""
        # The test client doesn't have a real KG, so the module-level
        # _get_kg() creates one from config. We mock the KnowledgeGraph.
        import video_analysis.api as api_mod

        mock_kg = MagicMock()
        mock_kg.stats.return_value = {
            "entity_count": 42,
            "relationship_count": 17,
            "video_count": 3,
            "type_breakdown": {"person": 20, "object": 15, "action": 7},
            "database_size_bytes": 65536,
            "last_indexed_video": {"video_id": "vid1", "indexed_at": 1000.0},
        }
        api_mod._kg_instance = mock_kg

        response = client.get("/api/kg/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["entity_count"] == 42
        assert data["relationship_count"] == 17
        assert data["video_count"] == 3
        assert data["type_breakdown"]["person"] == 20
        assert data["database_size_bytes"] == 65536
        assert data["last_indexed_video"]["video_id"] == "vid1"

    def test_kg_entities_endpoint(self, client: TestClient):
        """GET /api/kg/entities returns list of entities."""
        import video_analysis.api as api_mod

        from video_analysis.knowledge_graph import EntityRecord

        mock_kg = MagicMock()
        mock_kg.get_top_entities.return_value = [
            EntityRecord(
                id=1,
                name="Alice",
                entity_type="person",
                frequency=5,
                video_ids={"vid1", "vid2"},
            ),
            EntityRecord(
                id=2, name="Car", entity_type="object", frequency=3, video_ids={"vid1"}
            ),
        ]
        api_mod._kg_instance = mock_kg

        response = client.get("/api/kg/entities")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["name"] == "Alice"
        assert data[0]["entity_type"] == "person"
        assert data[0]["frequency"] == 5
        assert "vid1" in data[0]["video_ids"]

    def test_kg_entities_with_query(self, client: TestClient):
        """GET /api/kg/entities?query=... calls cross_video_search."""
        import video_analysis.api as api_mod

        from video_analysis.knowledge_graph import EntityRecord

        mock_kg = MagicMock()
        mock_kg.cross_video_search.return_value = [
            EntityRecord(
                id=3,
                name="Meeting Room",
                entity_type="location",
                frequency=2,
                video_ids={"vid3"},
            ),
        ]
        api_mod._kg_instance = mock_kg

        response = client.get("/api/kg/entities", params={"query": "meeting"})
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["name"] == "Meeting Room"
        mock_kg.cross_video_search.assert_called_once_with("meeting", limit=50)

    def test_kg_timeline_endpoint(self, client: TestClient):
        """GET /api/kg/timeline returns timeline items."""
        import video_analysis.api as api_mod

        mock_kg = MagicMock()
        mock_kg.get_timeline.return_value = [
            {
                "video_id": "vid1",
                "filename": "test.mp4",
                "duration_seconds": 120.0,
                "entity_count": 5,
                "indexed_at": 1000.0,
                "top_entities": ["Alice", "Bob"],
            }
        ]
        api_mod._kg_instance = mock_kg

        response = client.get("/api/kg/timeline")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["video_id"] == "vid1"
        assert data[0]["top_entities"] == ["Alice", "Bob"]

    def test_kg_entity_relationships(self, client: TestClient):
        """GET /api/kg/entities/{id}/relationships returns relations."""
        import video_analysis.api as api_mod

        from video_analysis.knowledge_graph import EntityRecord, RelationshipRecord

        mock_kg = MagicMock()
        mock_kg.get_relationships.return_value = [
            RelationshipRecord(
                id=10, source_id=1, target_id=2, relation_type="co_occurs", strength=3
            ),
        ]
        # Use mock instance directly
        api_mod._kg_instance = mock_kg

        # get_entity returns entities based on id lookup
        def _get_entity(entity_id):
            entities = {
                1: EntityRecord(
                    id=1,
                    name="Alice",
                    entity_type="person",
                    frequency=5,
                    video_ids={"vid1"},
                ),
                2: EntityRecord(
                    id=2,
                    name="Bob",
                    entity_type="person",
                    frequency=3,
                    video_ids={"vid1"},
                ),
            }
            return entities.get(entity_id)

        mock_kg.get_entity.side_effect = _get_entity

        response = client.get("/api/kg/entities/1/relationships")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["source_name"] == "Alice"
        assert data[0]["target_name"] == "Bob"
        assert data[0]["relation_type"] == "co_occurs"
        assert data[0]["strength"] == 3

    def test_kg_video_entities(self, client: TestClient):
        """GET /api/kg/videos/{id}/entities returns entities for a video."""
        import video_analysis.api as api_mod

        from video_analysis.knowledge_graph import EntityRecord

        mock_kg = MagicMock()
        mock_kg.get_entities_for_video.return_value = [
            EntityRecord(
                id=1,
                name="Alice",
                entity_type="person",
                frequency=5,
                video_ids={"vid1"},
            ),
        ]
        api_mod._kg_instance = mock_kg

        response = client.get("/api/kg/videos/vid1/entities")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["name"] == "Alice"

    def test_kg_context_endpoint(self, client: TestClient):
        """GET /api/kg/context returns LLM-friendly markdown context."""
        import video_analysis.api as api_mod

        mock_kg = MagicMock()
        mock_kg.get_knowledge_context.return_value = (
            "## Video Knowledge Graph Summary\n"
            "- **Videos indexed**: 3\n"
            "- **Unique entities**: 42\n"
        )
        api_mod._kg_instance = mock_kg

        response = client.get("/api/kg/context")
        assert response.status_code == 200
        data = response.json()
        assert "context" in data
        assert "Videos indexed" in data["context"]


# =========================================================================
# Pipeline Health Monitor API tests (v0.54.0)
# =========================================================================


class TestHealthAPI:
    """Tests for pipeline health monitor REST API endpoints."""

    def test_health_runs_endpoint(self, client: TestClient):
        """GET /api/health/runs returns health report."""
        import video_analysis.api as api_mod

        mock_health = MagicMock()
        mock_health.get_health_report.return_value = {
            "runs": [
                {
                    "run_id": 1,
                    "video_id": "vid1",
                    "timestamp": 1000.0,
                    "duration_s": 45.2,
                    "success": True,
                    "stage_timings": {"ocr": 12.0},
                    "ocr_confidence": 0.95,
                    "detection_confidence": 0.88,
                    "transcript_confidence": 0.92,
                }
            ],
            "health_score": 0.92,
            "active_alerts": [],
            "degraded_metrics": [],
        }
        api_mod._health_instance = mock_health

        response = client.get("/api/health/runs")
        assert response.status_code == 200
        data = response.json()
        assert data["run_count"] == 1
        assert len(data["runs"]) == 1
        assert data["runs"][0]["video_id"] == "vid1"
        assert data["runs"][0]["duration_s"] == 45.2
        assert data["health_score"] == 0.92

    def test_health_summary_endpoint(self, client: TestClient):
        """GET /api/health/summary returns concise summary."""
        import video_analysis.api as api_mod

        mock_health = MagicMock()
        mock_health.get_health_summary.return_value = {
            "health_score": 0.85,
            "total_runs": 10,
            "success_rate": 0.9,
            "active_alerts_count": 2,
        }
        api_mod._health_instance = mock_health

        response = client.get("/api/health/summary")
        assert response.status_code == 200
        data = response.json()
        assert data["health_score"] == 0.85
        assert data["total_runs"] == 10

    def test_health_alerts_endpoint(self, client: TestClient):
        """GET /api/health/alerts returns active alerts."""
        import video_analysis.api as api_mod

        mock_health = MagicMock()
        mock_health.get_active_alerts.return_value = [
            {"id": "alert1", "severity": "warning", "message": "OCR confidence dropped"}
        ]
        api_mod._health_instance = mock_health

        response = client.get("/api/health/alerts")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        assert data["alerts"][0]["severity"] == "warning"

    def test_health_alerts_filtered(self, client: TestClient):
        """GET /api/health/alerts?min_severity=... filters correctly."""
        import video_analysis.api as api_mod

        mock_health = MagicMock()
        mock_health.get_active_alerts.return_value = []
        api_mod._health_instance = mock_health

        response = client.get("/api/health/alerts", params={"min_severity": "error"})
        assert response.status_code == 200
        mock_health.get_active_alerts.assert_called_once_with(min_severity="error")

    def test_health_acknowledge_alert(self, client: TestClient):
        """POST /api/health/alerts/{id}/acknowledge works."""
        import video_analysis.api as api_mod

        mock_health = MagicMock()
        api_mod._health_instance = mock_health

        response = client.post("/api/health/alerts/alert1/acknowledge")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "acknowledged"
        assert data["alert_id"] == "alert1"
        mock_health.acknowledge_alert.assert_called_once_with("alert1")
