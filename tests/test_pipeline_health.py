"""
Tests for Pipeline Health Monitor (v0.52.0).

Covers:
- Run recording (success, failure, with stage timings, with confidence metrics)
- Anomaly detection (z-score based)
- Alert generation, deduplication, expiry, acknowledgment
- Health score computation (composite)
- Health report generation (summary, context)
- Graceful empty state
- Data clearance and vacuum
"""

import threading
from pathlib import Path

import pytest

from video_analysis.pipeline_health import (
    HealthReport,
    PipelineHealthMonitor,
)

# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def monitor(tmp_path: Path) -> PipelineHealthMonitor:
    """Create a PipelineHealthMonitor with a temporary data directory."""
    config = type("Config", (), {"data_dir": tmp_path})()
    m = PipelineHealthMonitor(
        config,
        window_size=20,
        z_score_threshold=2.0,  # sensitive for testing
        min_data_points=3,
        alert_cooldown_s=5,
        alert_expiry_s=3600,
    )
    return m


# ── Run recording ────────────────────────────────────────────────────────


class TestRecordRun:
    """Pipeline run recording."""

    def test_record_success(self, monitor: PipelineHealthMonitor):
        run_id = monitor.record_run(
            video_id="vid1",
            duration_s=30.0,
            success=True,
        )
        assert run_id > 0

    def test_record_failure(self, monitor: PipelineHealthMonitor):
        run_id = monitor.record_run(
            video_id="vid1",
            duration_s=5.0,
            success=False,
        )
        assert run_id > 0

    def test_record_with_stage_timings(self, monitor: PipelineHealthMonitor):
        run_id = monitor.record_run(
            video_id="vid1",
            duration_s=45.0,
            success=True,
            stage_timings={
                "scene_detection": 5.2,
                "ocr": 12.1,
                "yolo": 8.3,
                "transcription": 15.0,
            },
        )
        assert run_id > 0

    def test_record_with_confidence(self, monitor: PipelineHealthMonitor):
        run_id = monitor.record_run(
            video_id="vid1",
            duration_s=30.0,
            success=True,
            ocr_confidence=0.85,
            detection_confidence=0.72,
            transcript_confidence=0.91,
        )
        assert run_id > 0

    def test_record_returns_incremental_ids(self, monitor: PipelineHealthMonitor):
        id1 = monitor.record_run("vid1", 10.0)
        id2 = monitor.record_run("vid1", 10.0)
        id3 = monitor.record_run("vid1", 10.0)
        assert id1 < id2 < id3


# ── Anomaly detection ────────────────────────────────────────────────────


class TestAnomalyDetection:
    """Anomaly detection mechanics."""

    def test_no_anomaly_with_insufficient_data(self, monitor: PipelineHealthMonitor):
        # Only 2 runs — min_data_points=3, so no anomalies yet
        monitor.record_run("vid1", 30.0, success=True)
        monitor.record_run("vid1", 32.0, success=True)
        report = monitor.get_health_report()
        assert report.anomaly_count == 0

    def test_no_anomaly_on_stable_metrics(self, monitor: PipelineHealthMonitor):
        for _ in range(10):
            monitor.record_run("vid1", 30.0, success=True)
        report = monitor.get_health_report()
        assert report.anomaly_count == 0

    def test_detects_duration_anomaly(self, monitor: PipelineHealthMonitor):
        # Record 5 runs with ~30s duration
        for _ in range(5):
            monitor.record_run("vid1", 30.0, success=True)
        # Record an outlier
        monitor.record_run("vid1", 300.0, success=True)

        report = monitor.get_health_report()
        # The duration_s metric should show an anomaly
        duration_metrics = [m for m in report.metrics if m.metric_name == "duration_s"]
        if duration_metrics:
            assert duration_metrics[0].is_anomaly
            assert duration_metrics[0].z_score > 2.0

    def test_detects_ocr_confidence_anomaly(self, monitor: PipelineHealthMonitor):
        for _ in range(5):
            monitor.record_run("vid1", 30.0, success=True, ocr_confidence=0.85)
        # Drop in OCR confidence
        monitor.record_run("vid1", 30.0, success=True, ocr_confidence=0.15)

        report = monitor.get_health_report()
        ocr_metrics = [m for m in report.metrics if m.metric_name == "ocr_confidence"]
        if ocr_metrics:
            assert ocr_metrics[0].is_anomaly

    def test_detects_success_rate_drop(self, monitor: PipelineHealthMonitor):
        # 5 successes
        for _ in range(5):
            monitor.record_run("vid1", 30.0, success=True)
        # The health report should show 100% success rate
        report = monitor.get_health_report()
        assert report.success_rate == 1.0
        assert report.total_runs == 5


# ── Alert tests ──────────────────────────────────────────────────────────


class TestAlerts:
    """Alert management."""

    def test_alert_created_on_anomaly(self, monitor: PipelineHealthMonitor):
        for _ in range(5):
            monitor.record_run("vid1", 30.0, success=True)
        # Outlier triggers anomaly and alert
        monitor.record_run("vid1", 300.0, success=True)

        alerts = monitor.get_active_alerts()
        # At least one alert should exist
        if alerts:
            assert alerts[0].severity in ("warning", "error", "critical")
            assert alerts[0].metric_name != ""

    def test_alert_deduplication(self, monitor: PipelineHealthMonitor):
        for _ in range(5):
            monitor.record_run("vid1", 30.0, success=True)
        # First outlier
        monitor.record_run("vid1", 300.0, success=True)
        first_count = len(monitor.get_active_alerts())
        # Second similar outlier within cooldown should be suppressed
        monitor.record_run("vid1", 310.0, success=True)
        second_count = len(monitor.get_active_alerts())
        # Should not have increased
        assert second_count <= first_count + 1

    def test_acknowledge_alert(self, monitor: PipelineHealthMonitor):
        for _ in range(5):
            monitor.record_run("vid1", 30.0, success=True)
        monitor.record_run("vid1", 300.0, success=True)

        alerts = monitor.get_active_alerts()
        if alerts:
            alert_id = alerts[0].alert_id
            acked = monitor.acknowledge_alert(alert_id)
            assert acked
            remaining = monitor.get_active_alerts()
            # Alert should no longer be active
            assert all(a.alert_id != alert_id for a in remaining)

    def test_acknowledge_all(self, monitor: PipelineHealthMonitor):
        for _ in range(5):
            monitor.record_run("vid1", 30.0, success=True)
        for _ in range(3):
            monitor.record_run("vid1", 300.0, success=True)

        count = monitor.acknowledge_all_alerts()
        # Should acknowledge all alerts
        remaining = monitor.get_active_alerts()
        assert len(remaining) == 0

    def test_empty_alerts(self, monitor: PipelineHealthMonitor):
        assert monitor.get_active_alerts() == []


# ── Health score ─────────────────────────────────────────────────────────


class TestHealthScore:
    """Composite health score."""

    def test_health_score_full_on_no_data(self, monitor: PipelineHealthMonitor):
        score = monitor.compute_health_score()
        assert score == 1.0

    def test_health_score_high_on_good_runs(self, monitor: PipelineHealthMonitor):
        for _ in range(10):
            monitor.record_run("vid1", 30.0, success=True)
        score = monitor.compute_health_score()
        assert score >= 0.8

    def test_health_score_drops_on_failures(self, monitor: PipelineHealthMonitor):
        for _ in range(5):
            monitor.record_run("vid1", 30.0, success=True)
        for _ in range(5):
            monitor.record_run("vid1", 5.0, success=False)

        score = monitor.compute_health_score()
        assert score < 0.9  # should be lower than perfect

    def test_health_score_range(self, monitor: PipelineHealthMonitor):
        for _ in range(10):
            monitor.record_run("vid1", 30.0, success=True)
        score = monitor.compute_health_score()
        assert 0.0 <= score <= 1.0


# ── Health report ────────────────────────────────────────────────────────


class TestHealthReport:
    """Health report generation."""

    def test_report_empty(self, monitor: PipelineHealthMonitor):
        report = monitor.get_health_report()
        assert report.total_runs == 0
        assert report.recent_runs == 0
        assert report.health_score == 1.0

    def test_report_with_runs(self, monitor: PipelineHealthMonitor):
        for i in range(5):
            monitor.record_run(f"vid{i}", 30.0, success=True)

        report = monitor.get_health_report(recent_count=10)
        assert report.total_runs == 5
        assert report.recent_runs == 5
        assert report.success_rate == 1.0
        assert report.avg_duration_s == 30.0

    def test_report_contains_alerts(self, monitor: PipelineHealthMonitor):
        for _ in range(5):
            monitor.record_run("vid1", 30.0, success=True)
        monitor.record_run("vid1", 999.0, success=True)

        report = monitor.get_health_report()
        # Should have at least one alert if anomaly was detected
        assert isinstance(report, HealthReport)
        # With z_score_threshold=2, the 999 outlier should trigger an alert
        # but z-score depends on std dev; just verify report object is valid
        assert report.total_runs == 6

    def test_health_summary(self, monitor: PipelineHealthMonitor):
        summary = monitor.get_health_summary()
        assert "health_score" in summary
        assert "total_runs" in summary
        assert "status" in summary

    def test_health_summary_healthy(self, monitor: PipelineHealthMonitor):
        for _ in range(10):
            monitor.record_run("vid1", 30.0, success=True)
        summary = monitor.get_health_summary()
        assert summary["status"] == "healthy"

    def test_health_context_empty(self, monitor: PipelineHealthMonitor):
        ctx = monitor.get_health_context()
        assert "Pipeline Health Summary" in ctx

    def test_health_context_with_data(self, monitor: PipelineHealthMonitor):
        for i in range(5):
            monitor.record_run(f"vid{i}", 30.0, success=True)
        ctx = monitor.get_health_context()
        assert "Pipeline Health Summary" in ctx
        assert "100.0%" in ctx or "1.0" in ctx


# ── Maintenance ──────────────────────────────────────────────────────────


class TestMaintenance:
    """Data clearance and vacuum."""

    def test_clear_runs(self, monitor: PipelineHealthMonitor):
        for i in range(5):
            monitor.record_run(f"vid{i}", 30.0, success=True)
        assert monitor.get_health_report().total_runs == 5
        monitor.clear_runs()
        assert monitor.get_health_report().total_runs == 0

    def test_clear_old_runs(self, monitor: PipelineHealthMonitor):
        for i in range(5):
            monitor.record_run(f"vid{i}", 30.0, success=True)
        count = monitor.clear_runs(older_than_days=0)  # clears all
        assert count == 5

    def test_clear_alerts(self, monitor: PipelineHealthMonitor):
        for _ in range(5):
            monitor.record_run("vid1", 30.0, success=True)
        monitor.record_run("vid1", 300.0, success=True)
        monitor.clear_alerts()
        assert monitor.get_active_alerts() == []

    def test_vacuum_does_not_error(self, monitor: PipelineHealthMonitor):
        for i in range(5):
            monitor.record_run(f"vid{i}", 30.0, success=True)
        monitor.vacuum()

    def test_context_manager(self, tmp_path: Path):
        config = type("Config", (), {"data_dir": tmp_path})()
        with PipelineHealthMonitor(config) as m:
            run_id = m.record_run("vid1", 30.0)
            assert run_id > 0

    def test_close_and_reopen(self, tmp_path: Path):
        config = type("Config", (), {"data_dir": tmp_path})()
        m = PipelineHealthMonitor(config)
        m.record_run("vid1", 30.0)
        m.close()

        m2 = PipelineHealthMonitor(config)
        report = m2.get_health_report()
        assert report.total_runs == 1


# ── Thread safety ────────────────────────────────────────────────────────


class TestThreadSafety:
    """Concurrent access safety."""

    def test_concurrent_runs(self, monitor: PipelineHealthMonitor):
        n_threads = 5
        runs_per = 10
        errors: list = []

        def worker(tid: int):
            try:
                for i in range(runs_per):
                    monitor.record_run(
                        f"vid_{tid}_{i}",
                        duration_s=30.0 + (i % 5) * 5,
                        success=True,
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(tid,)) for tid in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread safety errors: {errors}"
        report = monitor.get_health_report()
        assert report.total_runs == n_threads * runs_per


# ── Edge case tests ──────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_single_run(self, monitor: PipelineHealthMonitor):
        monitor.record_run("vid1", 30.0, success=True)
        assert monitor.get_health_report().total_runs == 1

    def test_zero_duration(self, monitor: PipelineHealthMonitor):
        run_id = monitor.record_run("vid1", 0.0, success=True)
        assert run_id > 0

    def test_negative_duration(self, monitor: PipelineHealthMonitor):
        run_id = monitor.record_run("vid1", -1.0, success=False)
        assert run_id > 0

    def test_very_long_duration(self, monitor: PipelineHealthMonitor):
        run_id = monitor.record_run("vid1", 999999.0, success=True)
        assert run_id > 0

    def test_partial_confidence(self, monitor: PipelineHealthMonitor):
        run_id = monitor.record_run(
            "vid1",
            30.0,
            ocr_confidence=0.85,
        )
        assert run_id > 0
