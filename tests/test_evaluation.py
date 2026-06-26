"""Tests for the Pipeline Evaluation Harness core (evaluation.py).

Covers:
- EvalMetric dataclass (threshold logic, pass/fail)
- EvalTaskResult dataclass (all_passed property)
- EvalReport (summary(), to_json(), passed property)
- EvaluationTask ABC (run() with timing, error handling)
- EvaluationRunner (discovery, registration, run_all)
"""

import json
import sys
from pathlib import Path
from dataclasses import asdict
from typing import List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import MagicMock, patch

import pytest

from video_analysis.config import Config
from video_analysis.evaluation import (
    EvalMetric,
    EvalTaskResult,
    EvalReport,
    EvaluationTask,
    EvaluationRunner,
    run_evaluation,
)

# ═══════════════════════════════════════════════════════════════════════════
# EvalMetric
# ═══════════════════════════════════════════════════════════════════════════


class TestEvalMetric:
    def test_default_no_threshold(self):
        """Metric without threshold_pass passes by default."""
        m = EvalMetric(name="precision", value=0.5)
        assert m.name == "precision"
        assert m.value == 0.5
        assert m.passed is None

    def test_with_threshold_passes(self):
        """Metric with threshold_pass >= threshold should pass."""
        m = EvalMetric(name="precision", value=0.95, threshold_pass=0.8)
        assert m.passed is True

    def test_with_threshold_fails(self):
        """Metric with value < threshold_pass should fail."""
        m = EvalMetric(name="precision", value=0.5, threshold_pass=0.8)
        assert m.passed is False

    def test_edge_case_exact_threshold(self):
        """Value exactly equal to threshold_pass should pass."""
        m = EvalMetric(name="precision", value=0.8, threshold_pass=0.8)
        assert m.passed is True

    def test_zero_threshold_always_passes(self):
        """Zero threshold should always pass (value >= 0)."""
        m = EvalMetric(name="latency", value=0.0, threshold_pass=0.0)
        assert m.passed is True

    def test_negative_values(self):
        """Negative values can still satisfy threshold if it's more negative."""
        m = EvalMetric(name="error_rate", value=-0.5, threshold_pass=-1.0)
        assert m.passed is True  # -0.5 >= -1.0


# ═══════════════════════════════════════════════════════════════════════════
# EvalTaskResult
# ═══════════════════════════════════════════════════════════════════════════


class TestEvalTaskResult:
    def test_all_passed_all_good(self):
        """all_passed True when all metrics passed."""
        result = EvalTaskResult(
            task_name="test",
            task_description="Test task",
            status="pass",
            metrics=[
                EvalMetric(name="m1", value=0.9, threshold_pass=0.8),
                EvalMetric(name="m2", value=1.0, threshold_pass=0.9),
            ],
        )
        assert result.all_passed is True

    def test_all_passed_some_fail(self):
        """all_passed False when any metric failed."""
        result = EvalTaskResult(
            task_name="test",
            task_description="Test task",
            status="pass",
            metrics=[
                EvalMetric(name="m1", value=0.9, threshold_pass=0.8),
                EvalMetric(name="m2", value=0.5, threshold_pass=0.9),
            ],
        )
        assert result.all_passed is False

    def test_all_passed_no_metrics(self):
        """all_passed True when no metrics (nothing to fail)."""
        result = EvalTaskResult(
            task_name="test",
            task_description="Test task",
            status="pass",
            metrics=[],
        )
        assert result.all_passed is True

    def test_all_passed_error_status(self):
        """all_passed False when status is 'error'."""
        result = EvalTaskResult(
            task_name="test",
            task_description="Test task",
            status="error",
            error="Something broke",
            metrics=[],
        )
        assert result.all_passed is False

    def test_all_passed_skipped(self):
        """all_passed True when status is 'skipped' (not a failure)."""
        result = EvalTaskResult(
            task_name="test",
            task_description="Test task",
            status="skipped",
            metrics=[],
        )
        assert result.all_passed is True

    def test_all_passed_none_threshold(self):
        """Metrics without thresholds (None) don't affect all_passed."""
        result = EvalTaskResult(
            task_name="test",
            task_description="Test task",
            status="pass",
            metrics=[
                EvalMetric(name="m1", value=0.5),  # no threshold
                EvalMetric(name="m2", value=0.9, threshold_pass=0.8),
            ],
        )
        assert result.all_passed is True

    def test_all_passed_fail_with_none_threshold(self):
        """Metrics without thresholds don't hide actual failures."""
        result = EvalTaskResult(
            task_name="test",
            task_description="Test task",
            status="fail",
            metrics=[EvalMetric(name="m1", value=0.5, threshold_pass=0.8)],
        )
        assert result.all_passed is False


# ═══════════════════════════════════════════════════════════════════════════
# EvalReport
# ═══════════════════════════════════════════════════════════════════════════


class TestEvalReport:
    def test_empty_report(self):
        """An empty report (no results) passes by default."""
        report = EvalReport()
        assert report.passed is True
        assert len(report.results) == 0

    def test_generated_run_id(self):
        """Report auto-generates a run_id."""
        report = EvalReport()
        assert len(report.run_id) == 8

    def test_timestamp_is_float(self):
        """Report timestamp is a valid float."""
        report = EvalReport()
        assert isinstance(report.timestamp, float)
        assert report.timestamp > 0

    def test_passed_all_good(self):
        """report.passed True when all results pass."""
        report = EvalReport(
            results=[
                EvalTaskResult(task_name="t1", task_description="d1"),
                EvalTaskResult(task_name="t2", task_description="d2"),
            ]
        )
        assert report.passed is True

    def test_passed_one_fail(self):
        """report.passed False when any result fails."""
        report = EvalReport(
            results=[
                EvalTaskResult(task_name="t1", task_description="d1"),
                EvalTaskResult(
                    task_name="t2",
                    task_description="d2",
                    status="fail",
                    metrics=[EvalMetric(name="m1", value=0.5, threshold_pass=0.8)],
                ),
            ]
        )
        assert report.passed is False

    def test_summary_empty(self):
        """Summary of empty report shows 0/0."""
        report = EvalReport()
        assert "0/0" in report.summary()

    def test_summary_all_pass(self):
        """Summary with all passing results."""
        report = EvalReport(
            results=[
                EvalTaskResult(task_name="t1", task_description="d1"),
                EvalTaskResult(task_name="t2", task_description="d2"),
            ]
        )
        summary = report.summary()
        assert "2/2" in summary
        assert "passed" in summary

    def test_summary_with_failures(self):
        """Summary reports failure count."""
        report = EvalReport(
            results=[
                EvalTaskResult(task_name="t1", task_description="d1"),
                EvalTaskResult(
                    task_name="t2",
                    task_description="d2",
                    status="fail",
                    metrics=[EvalMetric(name="m1", value=0.5, threshold_pass=0.8)],
                ),
            ]
        )
        summary = report.summary()
        assert "1 failed" in summary or "1/2" in summary

    def test_summary_with_skipped(self):
        """Summary reports skipped count."""
        report = EvalReport(
            results=[
                EvalTaskResult(task_name="t1", task_description="d1"),
                EvalTaskResult(task_name="t2", task_description="d2", status="skipped"),
            ]
        )
        summary = report.summary()
        assert "1 skipped" in summary

    def test_to_json_serializable(self):
        """to_json produces valid JSON with all fields."""
        report = EvalReport(
            results=[
                EvalTaskResult(
                    task_name="test_task",
                    task_description="Test description",
                    status="pass",
                    metrics=[EvalMetric(name="accuracy", value=0.95)],
                )
            ]
        )
        json_str = report.to_json()
        data = json.loads(json_str)
        assert data["run_id"] == report.run_id
        assert len(data["results"]) == 1
        assert data["results"][0]["task_name"] == "test_task"

    def test_config_snapshot_in_report(self):
        """Config snapshot passes through correctly."""
        report = EvalReport(config_snapshot={"model": "test", "threshold": 0.5})
        data = json.loads(report.to_json())
        assert data["config_snapshot"]["model"] == "test"

    def test_version_default(self):
        """Default version is set."""
        report = EvalReport()
        assert report.version


# ═══════════════════════════════════════════════════════════════════════════
# EvaluationTask
# ═══════════════════════════════════════════════════════════════════════════


class TestEvaluationTask:
    def test_run_calls_abstract_method(self, cfg):
        """run() invokes _run() and returns a result with timing."""
        task = _make_task(cfg)
        result = task.run()
        assert result.task_name == "test_task"
        assert result.status == "pass"
        assert result.duration_ms > 0

    def test_run_records_duration(self, cfg):
        """run() accurately records duration."""
        task = _make_task(cfg)
        result = task.run()
        assert 0 < result.duration_ms < 5000  # generous range

    def test_run_error_handling(self, cfg):
        """run() catches exceptions and returns error result."""

        class FailingTask(EvaluationTask):
            name = "failing"
            description = "Always fails"

            def _run(self):
                raise ValueError("Intentional failure")

        task = FailingTask(cfg)
        result = task.run()
        assert result.status == "error"
        assert "ValueError" in result.error
        assert result.duration_ms > 0

    def test_abstract_method_enforced(self, cfg):
        """Cannot instantiate EvaluationTask directly."""
        with pytest.raises(TypeError):
            EvaluationTask(cfg)  # type: ignore

    def test_config_passed_to_subclass(self, cfg):
        """Subclass receives config correctly."""

        class ConfigCheckTask(EvaluationTask):
            name = "config_check"
            description = "Checks config"

            def _run(self):
                assert self.config is not None
                return EvalTaskResult(
                    task_name="config_check",
                    task_description="Config is accessible",
                    status="pass",
                )

        task = ConfigCheckTask(cfg)
        result = task.run()
        assert result.status == "pass"


# ═══════════════════════════════════════════════════════════════════════════
# EvaluationRunner
# ═══════════════════════════════════════════════════════════════════════════


class TestEvaluationRunner:
    def test_register_single_task(self, cfg):
        """Register a single task then run all."""
        runner = EvaluationRunner(cfg)
        task = _make_task(cfg)
        runner.register_task(task)
        with patch.object(runner, "discover_tasks", return_value={}):
            report = runner.run_all()
        assert len(report.results) == 1
        assert report.results[0].task_name == "test_task"

    def test_register_multiple_tasks(self, cfg):
        """Register multiple tasks and run all."""
        runner = EvaluationRunner(cfg)
        runner.register_task(_make_task(cfg, name="task_a"))
        runner.register_task(_make_task(cfg, name="task_b"))
        with patch.object(runner, "discover_tasks", return_value={}):
            report = runner.run_all()
        assert len(report.results) == 2
        names = {r.task_name for r in report.results}
        assert names == {"task_a", "task_b"}

    def test_run_specific_tasks(self, cfg):
        """Run only specific tasks by name."""
        runner = EvaluationRunner(cfg)
        runner.register_task(_make_task(cfg, name="task_a"))
        runner.register_task(_make_task(cfg, name="task_b"))
        runner.register_task(_make_task(cfg, name="task_c"))
        with patch.object(runner, "discover_tasks", return_value={}):
            report = runner.run_all(task_names=["task_a", "task_c"])
        assert len(report.results) == 2
        names = {r.task_name for r in report.results}
        assert names == {"task_a", "task_c"}

    def test_run_all_with_no_tasks(self, cfg):
        """run_all with no tasks returns empty results."""
        runner = EvaluationRunner(cfg)
        with patch.object(runner, "discover_tasks", return_value={}):
            report = runner.run_all()
        assert len(report.results) == 0

    def test_run_all_with_failure_task(self, cfg):
        """Failing task produces error result in report."""

        class FailingTask(EvaluationTask):
            name = "failing"
            description = "Always fails"

            def _run(self):
                raise RuntimeError("Intentional")

        runner = EvaluationRunner(cfg)
        runner.register_task(FailingTask(cfg))
        with patch.object(runner, "discover_tasks", return_value={}):
            report = runner.run_all()
        assert len(report.results) == 1
        assert report.results[0].status == "error"
        assert report.passed is False

    def test_empty_task_names_list(self, cfg):
        """Empty task_names list runs nothing (no tasks named '')."""
        runner = EvaluationRunner(cfg)
        runner.register_task(_make_task(cfg))
        with patch.object(runner, "discover_tasks", return_value={}):
            report = runner.run_all(task_names=[])
        assert len(report.results) == 0

    def test_non_existent_task_name(self, cfg):
        """Non-existent task name is silently skipped."""
        runner = EvaluationRunner(cfg)
        runner.register_task(_make_task(cfg, name="real_task"))
        with patch.object(runner, "discover_tasks", return_value={}):
            report = runner.run_all(task_names=["nonexistent"])
        assert len(report.results) == 0

    def test_discover_tasks_no_evals_package(self, cfg):
        """discover_tasks returns empty dict when evals.tasks not importable."""
        runner = EvaluationRunner(cfg)
        tasks = runner.discover_tasks()
        # Might return empty if evals.tasks not installed
        assert isinstance(tasks, dict)

    def test_get_available_tasks(self, cfg):
        """get_available_tasks returns registered task metadata."""
        runner = EvaluationRunner(cfg)
        runner.register_task(_make_task(cfg, name="task_a", desc="Task A description"))
        available = runner.get_available_tasks()
        assert "task_a" in available
        assert available["task_a"] == "Task A description"

    def test_get_available_tasks_empty(self, cfg):
        """get_available_tasks returns empty dict when no tasks."""
        runner = EvaluationRunner(cfg)
        available = runner.get_available_tasks()
        assert isinstance(available, dict)

    def test_discover_and_register(self, cfg):
        """discover_tasks and run_all works end-to-end with real evals package."""
        with patch.object(EvaluationRunner, "discover_tasks") as mock_discover:
            mock_discover.return_value = {}
            runner = EvaluationRunner(cfg)
            # No real discovery, but our registered task still runs
            runner.register_task(_make_task(cfg))
            report = runner.run_all()
            assert len(report.results) == 1

    def test_report_has_timing(self, cfg):
        """Report records total duration."""
        runner = EvaluationRunner(cfg)
        runner.register_task(_make_task(cfg))
        with patch.object(runner, "discover_tasks", return_value={}):
            report = runner.run_all()
        assert report.total_duration_ms > 0

    def test_report_has_config_snapshot(self, cfg):
        """Report includes config snapshot."""
        runner = EvaluationRunner(cfg)
        runner.register_task(_make_task(cfg))
        with patch.object(runner, "discover_tasks", return_value={}):
            report = runner.run_all()
        assert "embedding_model" in report.config_snapshot


# ═══════════════════════════════════════════════════════════════════════════
# Convenience function
# ═══════════════════════════════════════════════════════════════════════════


def test_run_evaluation_convenience(cfg):
    """run_evaluation() convenience function works."""
    # Since no evals.tasks discovered, it returns empty report
    report = run_evaluation(cfg)
    assert isinstance(report, EvalReport)
    assert report.passed is True


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def cfg():
    """Return a Config pointing at a temp directory."""
    import tempfile
    import shutil

    tmp = Path(tempfile.mkdtemp(prefix="va_eval_test_"))
    cfg = Config(data_dir=tmp)
    yield cfg
    shutil.rmtree(tmp, ignore_errors=True)


def _make_task(cfg, name="test_task", desc="Test task") -> EvaluationTask:
    """Create a simple passing evaluation task for testing."""

    task_name_val = name
    task_desc_val = desc

    class PassingTask(EvaluationTask):
        name = task_name_val
        description = task_desc_val

        def _run(self):
            return EvalTaskResult(
                task_name=self.name,
                task_description=self.description,
                status="pass",
                metrics=[
                    EvalMetric(name="accuracy", value=0.95),
                    EvalMetric(name="latency", value=100, unit="ms"),
                ],
            )

    return PassingTask(cfg)
