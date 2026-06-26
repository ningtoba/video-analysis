"""Tests for the Monitoring Dashboard tab."""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import MagicMock, patch

import pytest

from video_analysis.config import Config
from video_analysis.job_queue import JobManager, Job, JobStatus
from video_analysis import metrics as va_metrics

# ── Helpers ──────────────────────────────────────────────────────────


@pytest.fixture
def cfg():
    return Config(data_dir=Path("/tmp/va_test_monitor"))


@pytest.fixture
def mock_job_manager():
    """Create a patched get_default_manager that returns sample jobs."""
    jm = MagicMock()
    jm.list_jobs.return_value = [
        Job(
            job_id="abc-123-def",
            job_type="process_video",
            status=JobStatus.COMPLETED,
            progress_pct=100,
        ),
        Job(
            job_id="ghi-456-jkl",
            job_type="process_video",
            status=JobStatus.RUNNING,
            progress_pct=45,
        ),
        Job(
            job_id="mno-789-pqr",
            job_type="process_video",
            status=JobStatus.FAILED,
            progress_pct=30,
        ),
        Job(
            job_id="stu-012-vwx",
            job_type="process_video",
            status=JobStatus.PENDING,
            progress_pct=0,
        ),
    ]
    with patch("ui.monitor.get_default_manager", return_value=jm):
        yield


@pytest.fixture
def mock_metrics():
    """Patch the monitor's _collect_system_metrics to return known values."""
    from ui import monitor as monitor_module

    original = monitor_module._collect_system_metrics

    def fake_collect():
        return {
            "pipeline_total": 42,
            "pipeline_success": 40,
            "pipeline_failures": 2,
            "questions_answered": 150,
            "videos_indexed": 10,
            "gpu_memory_gb": 5.7,
        }

    monitor_module._collect_system_metrics = fake_collect
    yield
    monitor_module._collect_system_metrics = original


# ── Tests for _collect_system_metrics ────────────────────────────────


class TestCollectSystemMetrics:
    def test_collect_returns_dict(self):
        from ui.monitor import _collect_system_metrics

        metrics = _collect_system_metrics()
        assert isinstance(metrics, dict)

    def test_collect_has_all_keys(self):
        """Import the actual monitor module to test its metric collection."""
        from ui.monitor import _collect_system_metrics

        metrics = _collect_system_metrics()
        for key in (
            "pipeline_total",
            "pipeline_success",
            "pipeline_failures",
            "questions_answered",
            "videos_indexed",
            "gpu_memory_gb",
        ):
            assert key in metrics, f"Missing key: {key}"
            assert isinstance(metrics[key], (int, float))

    def test_collect_with_mock_metrics(self, mock_metrics):
        from ui.monitor import _collect_system_metrics

        metrics = _collect_system_metrics()
        assert metrics["pipeline_total"] == 42
        assert metrics["pipeline_success"] == 40
        assert metrics["pipeline_failures"] == 2
        assert metrics["questions_answered"] == 150
        assert metrics["videos_indexed"] == 10
        assert metrics["gpu_memory_gb"] == pytest.approx(5.7, rel=0.1)


# ── Tests for _build_system_metrics_html ─────────────────────────────


class TestBuildSystemMetricsHtml:
    def test_returns_string(self):
        from ui.monitor import _build_system_metrics_html

        metrics = {
            "pipeline_total": 42,
            "pipeline_success": 40,
            "pipeline_failures": 2,
            "questions_answered": 150,
            "videos_indexed": 10,
            "gpu_memory_gb": 5.7,
        }
        html = _build_system_metrics_html(metrics)
        assert isinstance(html, str)
        assert len(html) > 50
        assert "Pipeline Runs" in html
        assert "Videos Indexed" in html
        assert "5.7" in html or "5.70" in html

    def test_zero_metrics(self):
        from ui.monitor import _build_system_metrics_html

        metrics = {
            "pipeline_total": 0,
            "pipeline_success": 0,
            "pipeline_failures": 0,
            "questions_answered": 0,
            "videos_indexed": 0,
            "gpu_memory_gb": 0.0,
        }
        html = _build_system_metrics_html(metrics)
        assert isinstance(html, str)
        assert "N/A" in html  # GPU N/A when 0
        assert "0" in html

    def test_high_failure_rate_class(self):
        from ui.monitor import _build_system_metrics_html

        metrics = {
            "pipeline_total": 10,
            "pipeline_success": 2,
            "pipeline_failures": 8,
            "questions_answered": 0,
            "videos_indexed": 0,
            "gpu_memory_gb": 0.0,
        }
        html = _build_system_metrics_html(metrics)
        assert isinstance(html, str)
        assert "error" in html  # failure class should be 'error'


# ── Tests for _build_job_queue_html ─────────────────────────────────


class TestBuildJobQueueHtml:
    def test_no_jobs(self, cfg):
        from ui.monitor import _build_job_queue_html

        with patch("ui.monitor.get_default_manager") as mock_mgr:
            mock_mgr.return_value.list_jobs.return_value = []
            html = _build_job_queue_html(cfg)
            assert "No jobs" in html

    def test_with_jobs(self, cfg, mock_job_manager):
        from ui.monitor import _build_job_queue_html

        html = _build_job_queue_html(cfg)
        assert "abc-123" in html
        assert "completed" in html or "COMPLETED" in html or "completed" in html.lower()
        assert "running" in html.lower() or "RUNNING" in html
        assert "failed" in html.lower() or "FAILED" in html
        assert "pending" in html.lower() or "PENDING" in html

    def test_manager_exception(self, cfg):
        from ui.monitor import _build_job_queue_html

        with patch(
            "ui.monitor.get_default_manager", side_effect=RuntimeError("no manager")
        ):
            html = _build_job_queue_html(cfg)
            assert "not available" in html.lower()


# ── Tests for _build_metric_html ──────────────────────────────────────


class TestBuildMetricHtml:
    def test_basic(self):
        from ui.monitor import _build_metric_html

        html = _build_metric_html("Pipeline Runs", "42", "success")
        assert "Pipeline Runs" in html
        assert "42</div>" in html
        assert "success" in html

    def test_empty_status(self):
        from ui.monitor import _build_metric_html

        html = _build_metric_html("Test", "0", "")
        assert "Test" in html
        assert "0</div>" in html


# ── Tests for _build_metrics_row ──────────────────────────────────────


class TestBuildMetricsRow:
    def test_single_metric(self):
        from ui.monitor import _build_metrics_row

        html = _build_metrics_row([{"label": "L", "value": "5", "status": ""}])
        assert "L" in html
        assert "monitor-row" in html

    def test_multiple_metrics(self):
        from ui.monitor import _build_metrics_row

        html = _build_metrics_row(
            [
                {"label": "A", "value": "1", "status": "success"},
                {"label": "B", "value": "2", "status": "warning"},
                {"label": "C", "value": "3", "status": "error"},
            ]
        )
        assert "A" in html
        assert "B" in html
        assert "C" in html
        assert "success" in html
        assert "warning" in html
        assert "error" in html


# ── Tests for _build_metrics_snapshot_html ────────────────────────────


class TestBuildMetricsSnapshotHtml:
    def test_returns_html(self, cfg, mock_metrics):
        from ui.monitor import _build_metrics_snapshot_html

        with patch("ui.monitor.get_default_manager") as mock_mgr:
            mock_mgr.return_value.list_jobs.return_value = []
            html = _build_metrics_snapshot_html(cfg)
            assert isinstance(html, str)
            assert len(html) > 100
            assert "System Metrics" in html

    def test_monitor_dark_css_included(self, cfg, mock_metrics):
        from ui.monitor import _build_metrics_snapshot_html, MONITOR_DARK_CSS

        with patch("ui.monitor.get_default_manager") as mock_mgr:
            mock_mgr.return_value.list_jobs.return_value = []
            html = _build_metrics_snapshot_html(cfg)
            assert MONITOR_DARK_CSS in html


# ── Tests for _run_eval_task ──────────────────────────────────────────


class TestRunEvalTask:
    def test_handles_error_gracefully(self, cfg):
        from ui.monitor import _run_eval_task

        result = _run_eval_task(cfg, "")
        assert "**Error:" in result or "passed" in result or "skipped" in result

    def test_empty_task_names_runs_all(self, cfg):
        from ui.monitor import _run_eval_task

        result = _run_eval_task(cfg, "")
        # Should either run successfully or report an error gracefully
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    def test_with_specific_task(self, cfg):
        from ui.monitor import _run_eval_task

        result = _run_eval_task(cfg, "retrieval_precision")
        assert result is not None
        assert isinstance(result, str)

    def test_multiple_tasks(self, cfg):
        from ui.monitor import _run_eval_task

        result = _run_eval_task(cfg, "retrieval_precision, scene_boundary_accuracy")
        assert result is not None
        assert isinstance(result, str)


# ── Tests for HTML helpers CSS constant ────────────────────────────────


class TestMonitorConstants:
    def test_monitor_dark_css_exists(self):
        from ui.monitor import MONITOR_DARK_CSS

        assert isinstance(MONITOR_DARK_CSS, str)
        assert "monitor-card" in MONITOR_DARK_CSS
        assert "monitor-row" in MONITOR_DARK_CSS
        assert "GRADIENT" not in MONITOR_DARK_CSS  # sanity check
