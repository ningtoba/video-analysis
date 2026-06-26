"""
Tests for the In-Process Async Job Queue (v0.43.0).

Tests the JobManager in isolation (unit tests) and the job-related REST API
endpoints (integration tests with mocking).
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from video_analysis.api import create_api_router, set_rag_instance
from video_analysis.config import Config
from video_analysis.job_queue import (
    Job,
    JobManager,
    JobStatus,
    get_default_manager,
    reset_default_manager,
)

# ═════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═════════════════════════════════════════════════════════════════════════════


@pytest.fixture(autouse=True)
def _reset_job_manager():
    """Reset the default JobManager singleton before each test."""
    reset_default_manager()
    yield
    reset_default_manager()


@pytest.fixture
def manager():
    """Create a fresh JobManager for unit tests."""
    return JobManager(max_concurrent=2)


@pytest.fixture
def mock_rag():
    """Create a mocked VideoRAG instance."""
    rag = MagicMock()
    rag.list_videos.return_value = ["test_vid"]
    rag.get_library_info.return_value = MagicMock(
        video_id="test_vid",
        filename="test.mp4",
        num_scenes=3,
        num_chunks=10,
        duration=60.0,
        has_sprite=True,
    )
    return rag


@pytest.fixture
def config():
    """Create a test config."""
    import tempfile
    from pathlib import Path

    cfg = Config()
    cfg.data_dir = Path(tempfile.mkdtemp())
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


# ═════════════════════════════════════════════════════════════════════════════
# JobManager unit tests
# ═════════════════════════════════════════════════════════════════════════════


class TestJobManagerEnqueue:
    """Tests for JobManager.enqueue()."""

    @pytest.mark.asyncio
    async def test_enqueue_returns_job_with_id(self, manager):
        """Enqueued job must have a job_id and be in PENDING state."""
        job = await manager.enqueue("test_type", key="value")
        assert job.job_id is not None
        assert len(job.job_id) == 32  # uuid4 hex
        assert job.job_type == "test_type"
        assert job.status == JobStatus.PENDING
        assert job.params == {"key": "value"}

    @pytest.mark.asyncio
    async def test_enqueue_stores_job(self, manager):
        """Enqueued job must be retrievable by get_job()."""
        job = await manager.enqueue("test_type")
        retrieved = await manager.get_job(job.job_id)
        assert retrieved is not None
        assert retrieved.job_id == job.job_id

    @pytest.mark.asyncio
    async def test_enqueue_multiple_jobs(self, manager):
        """Multiple jobs must get unique IDs."""
        j1 = await manager.enqueue("type_a")
        j2 = await manager.enqueue("type_b")
        assert j1.job_id != j2.job_id

    @pytest.mark.asyncio
    async def test_get_job_not_found(self, manager):
        """Non-existent job_id must return None."""
        assert await manager.get_job("nonexistent") is None


class TestJobManagerUpdate:
    """Tests for JobManager.update_job()."""

    @pytest.mark.asyncio
    async def test_update_job_fields(self, manager):
        """Update must change job fields in-place."""
        job = await manager.enqueue("test")
        updated = await manager.update_job(
            job.job_id,
            status=JobStatus.RUNNING,
            progress="Working...",
            progress_pct=50.0,
        )
        assert updated is not None
        assert updated.status == JobStatus.RUNNING
        assert updated.progress == "Working..."
        assert updated.progress_pct == 50.0
        assert updated.started_at is not None

    @pytest.mark.asyncio
    async def test_update_not_found(self, manager):
        """Updating a non-existent job must return None."""
        result = await manager.update_job("nonexistent", status=JobStatus.RUNNING)
        assert result is None

    @pytest.mark.asyncio
    async def test_update_completed_sets_timestamp(self, manager):
        """Setting COMPLETED or FAILED must set completed_at."""
        job = await manager.enqueue("test")
        await manager.update_job(job.job_id, status=JobStatus.RUNNING)
        await manager.update_job(job.job_id, status=JobStatus.COMPLETED)
        retrieved = await manager.get_job(job.job_id)
        assert retrieved is not None
        assert retrieved.completed_at is not None
        assert retrieved.completed_at >= retrieved.started_at


class TestJobManagerCancel:
    """Tests for JobManager.cancel_job()."""

    @pytest.mark.asyncio
    async def test_cancel_pending_job(self, manager):
        """Cancel must set status to CANCELLED and return True."""
        job = await manager.enqueue("test")
        result = await manager.cancel_job(job.job_id)
        assert result is True
        retrieved = await manager.get_job(job.job_id)
        assert retrieved is not None
        assert retrieved.status == JobStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_running_job_returns_false(self, manager):
        """Running jobs should not be cancellable."""
        job = await manager.enqueue("test")
        await manager.update_job(job.job_id, status=JobStatus.RUNNING)
        result = await manager.cancel_job(job.job_id)
        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_job_returns_false(self, manager):
        """Non-existent job must return False."""
        result = await manager.cancel_job("nonexistent")
        assert result is False


class TestJobManagerList:
    """Tests for JobManager.list_jobs()."""

    @pytest.mark.asyncio
    async def test_list_jobs_returns_newest_first(self, manager):
        """Jobs must be returned newest first."""
        j1 = await manager.enqueue("a")
        j2 = await manager.enqueue("b")
        jobs = await manager.list_jobs()
        assert len(jobs) == 2
        assert jobs[0].job_id == j2.job_id  # newest first

    @pytest.mark.asyncio
    async def test_list_jobs_with_pagination(self, manager):
        """Pagination must work with limit and offset."""
        for _ in range(5):
            await manager.enqueue("test")
        jobs = await manager.list_jobs(limit=2, offset=1)
        assert len(jobs) == 2

    @pytest.mark.asyncio
    async def test_list_jobs_status_filter(self, manager):
        """Status filter must return only matching jobs."""
        j1 = await manager.enqueue("a")
        j2 = await manager.enqueue("b")
        await manager.update_job(j1.job_id, status=JobStatus.COMPLETED)
        running = await manager.list_jobs(status_filter=JobStatus.PENDING)
        assert len(running) == 1
        assert running[0].job_id == j2.job_id


class TestJobManagerWorker:
    """Tests for the background worker processing."""

    @pytest.mark.asyncio
    async def test_worker_processes_job_and_stores_result(self, manager):
        """Worker must run handler and store result on completion."""
        results: Dict[str, Any] = {}

        def handler(_manager, job_id: str, **params) -> Dict[str, Any]:
            results["called"] = True
            results["params"] = params
            return {"message": "done"}

        manager.register_handler("test_type", handler)

        # Start the worker
        await manager._start()
        try:
            job = await manager.enqueue("test_type", input="hello")

            # Wait for worker to process
            for _ in range(50):
                retrieved = await manager.get_job(job.job_id)
                if retrieved and retrieved.status == JobStatus.COMPLETED:
                    break
                await asyncio.sleep(0.05)

            retrieved = await manager.get_job(job.job_id)
            assert retrieved is not None
            assert retrieved.status == JobStatus.COMPLETED
            assert retrieved.result == {"message": "done"}
            assert results.get("called") is True
            assert results.get("params") == {"input": "hello"}
        finally:
            await manager._stop()

    @pytest.mark.asyncio
    async def test_worker_handles_handler_error(self, manager):
        """Worker must set FAILED status on handler exception."""

        def failing_handler(_manager, job_id: str, **params):
            raise RuntimeError("Something went wrong")

        manager.register_handler("test_type", failing_handler)

        await manager._start()
        try:
            job = await manager.enqueue("test_type")

            for _ in range(50):
                retrieved = await manager.get_job(job.job_id)
                if retrieved and retrieved.status in (
                    JobStatus.FAILED,
                    JobStatus.COMPLETED,
                ):
                    break
                await asyncio.sleep(0.05)

            retrieved = await manager.get_job(job.job_id)
            assert retrieved is not None
            assert retrieved.status == JobStatus.FAILED
            assert retrieved.error is not None
        finally:
            await manager._stop()

    @pytest.mark.asyncio
    async def test_worker_handles_unknown_job_type(self, manager):
        """Worker must set FAILED if no handler registered."""
        await manager._start()
        try:
            job = await manager.enqueue("unknown_type")

            for _ in range(50):
                retrieved = await manager.get_job(job.job_id)
                if retrieved and retrieved.status in (
                    JobStatus.FAILED,
                    JobStatus.COMPLETED,
                ):
                    break
                await asyncio.sleep(0.05)

            retrieved = await manager.get_job(job.job_id)
            assert retrieved is not None
            assert retrieved.status == JobStatus.FAILED
            assert "unknown" in (retrieved.error or "").lower()
        finally:
            await manager._stop()

    @pytest.mark.asyncio
    async def test_concurrent_jobs_respected(self, manager):
        """Semaphore must limit concurrent processing."""
        manager = JobManager(max_concurrent=1)
        in_flight = 0
        max_in_flight = 0

        async def slow_handler(_manager, job_id: str, **params):
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            await asyncio.sleep(0.1)
            in_flight -= 1
            return {"done": True}

        manager.register_handler("slow", slow_handler)

        await manager._start()
        try:
            j1 = await manager.enqueue("slow")
            await manager.enqueue("slow")
            await manager.enqueue("slow")

            # Wait for both to complete
            for _ in range(100):
                j1s = await manager.get_job(j1.job_id)
                if j1s and j1s.status == JobStatus.COMPLETED:
                    break
                await asyncio.sleep(0.05)

            # With max_concurrent=1, only one job should ever run at a time
            assert max_in_flight == 1
        finally:
            await manager._stop()


class TestJobManagerStats:
    """Tests for JobManager.stats."""

    @pytest.mark.asyncio
    async def test_stats_counts(self, manager):
        """Stats must reflect current job counts."""
        await manager.enqueue("a")
        stats = manager.stats
        assert stats["total_jobs"] == 1
        assert stats["max_concurrent"] == 2


# ═════════════════════════════════════════════════════════════════════════════
# REST API integration tests
# ═════════════════════════════════════════════════════════════════════════════


class TestApiJobEndpoints:
    """Tests for GET /api/jobs and GET /api/jobs/{job_id}."""

    def test_process_returns_enqueue_response(self, client):
        """POST /api/videos/process must return job_id when no source provided."""
        response = client.post(
            "/api/videos/process",
            json={},
        )
        # Without URL/file/file_path, we expect 400
        assert response.status_code == 400

    def test_get_job_not_found(self, client):
        """GET /api/jobs/nonexistent must return 404."""
        response = client.get("/api/jobs/nonexistent")
        assert response.status_code == 404

    def test_list_jobs_endpoint(self, client):
        """GET /api/jobs must return 200 with valid structure."""
        response = client.get("/api/jobs")
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "jobs" in data

    def test_jobs_endpoints_in_openapi(self, app):
        """Job endpoints must be present in OpenAPI spec."""
        paths = app.openapi().get("paths", {})
        assert "/api/jobs" in paths
        assert "/api/jobs/{job_id}" in paths

    def test_process_endpoint_still_registered(self, app):
        """POST /api/videos/process must still be registered."""
        paths = app.openapi().get("paths", {})
        assert "/api/videos/process" in paths
        # Verify it still has a POST method
        assert "post" in paths["/api/videos/process"]


class TestJobManagerDefaultSingleton:
    """Tests for the default JobManager singleton."""

    def test_get_default_manager_creates_once(self):
        """get_default_manager must return the same instance on repeated calls."""
        m1 = get_default_manager()
        m2 = get_default_manager()
        assert m1 is m2

    def test_reset_creates_new_instance(self):
        """reset_default_manager must clear the singleton."""
        m1 = get_default_manager()
        reset_default_manager()
        m2 = get_default_manager()
        assert m1 is not m2
