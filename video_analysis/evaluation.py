"""
Pipeline Evaluation Harness — benchmark-driven quality regression detection.

Provides a framework for running standardized video QA tasks through the
full pipeline (or individual stages) and measuring per-stage accuracy,
latency, and quality metrics.

Usage:
    from video_analysis.evaluation import EvaluationRunner

    runner = EvaluationRunner(config)
    report = runner.run_all()  # runs all registered tasks
    print(report.json())

CLI:
    python -m video_analysis --eval          # run all tasks
    python -m video_analysis --eval-list     # list available tasks
    python -m video_analysis --eval-tasks retrieval,scene  # specific tasks

Design:
    - Self-contained: generates synthetic test fixtures (no real videos needed)
    - Stateless: each run produces a timestamped JSON report
    - Extensible: add new tasks by subclassing EvaluationTask
    - Integrated: results exposed via API and Prometheus gauges
"""

from __future__ import annotations

import abc
import json
import time
import uuid
import importlib
import pkgutil
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from video_analysis.config import Config

# ── Data Models ──────────────────────────────────────────────────────────────


@dataclass
class EvalMetric:
    """A single measurement from an evaluation task."""

    name: str  # e.g. "precision@5", "cer", "f1_score"
    value: float
    unit: str = ""  # e.g. "%", "seconds", "score"
    threshold_pass: Optional[float] = None  # value must be >= this to pass
    passed: Optional[bool] = None

    def __post_init__(self) -> None:
        if self.threshold_pass is not None and self.passed is None:
            self.passed = self.value >= self.threshold_pass


@dataclass
class EvalTaskResult:
    """Result of running a single evaluation task."""

    task_name: str
    task_description: str
    status: str  # "pass", "fail", "error", "skipped"
    metrics: List[EvalMetric] = field(default_factory=list)
    error: Optional[str] = None
    duration_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def all_passed(self) -> bool:
        if self.status == "error":
            return False
        if self.status == "skipped":
            return True
        return all(m.passed is not False for m in self.metrics)


@dataclass
class EvalReport:
    """Complete report from an evaluation run."""

    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)
    version: str = "0.44.0"
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    results: List[EvalTaskResult] = field(default_factory=list)
    total_duration_ms: float = 0.0

    @property
    def passed(self) -> bool:
        return all(r.all_passed for r in self.results)

    def to_json(self, indent: int = 2) -> str:
        """Serialize report to JSON."""
        return json.dumps(asdict(self), indent=indent, default=str)

    def summary(self) -> str:
        """Human-readable summary line."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.all_passed and r.status != "skipped")
        failed = sum(1 for r in self.results if not r.all_passed)
        skipped = sum(1 for r in self.results if r.status == "skipped")
        return (
            f"Evaluation {self.run_id}: {passed}/{total} passed"
            + (f", {failed} failed" if failed else "")
            + (f", {skipped} skipped" if skipped else "")
            + f" in {self.total_duration_ms:.0f}ms"
        )


# ── Base Task ────────────────────────────────────────────────────────────────


class EvaluationTask(abc.ABC):
    """Abstract base class for an evaluation task.

    Subclasses must implement:
        - name: str (class attribute)
        - description: str (class attribute)
        - _run() -> EvalTaskResult
    """

    name: str = ""
    description: str = ""

    def __init__(self, config: Config, data_dir: Optional[Path] = None):
        self.config = config
        self._data_dir = data_dir or (config.data_dir / "eval_fixtures")
        self._data_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> EvalTaskResult:
        """Run this evaluation task with timing."""
        start = time.perf_counter()
        try:
            result = self._run()
        except Exception as e:
            result = EvalTaskResult(
                task_name=self.name,
                task_description=self.description,
                status="error",
                error=f"{type(e).__name__}: {e}",
            )
        result.duration_ms = (time.perf_counter() - start) * 1000
        return result

    @abc.abstractmethod
    def _run(self) -> EvalTaskResult:
        """Implement the actual evaluation logic."""


# ── Runner ────────────────────────────────────────────────────────────────────


class EvaluationRunner:
    """Orchestrates evaluation task discovery and execution."""

    def __init__(self, config: Config):
        self.config = config
        self._tasks: Dict[str, EvaluationTask] = {}

    def discover_tasks(self) -> Dict[str, type]:
        """Auto-discover EvaluationTask subclasses in evals.tasks package."""
        tasks: Dict[str, type] = {}
        try:
            import evals.tasks  # type: ignore

            for importer, modname, ispkg in pkgutil.iter_modules(
                evals.tasks.__path__  # type: ignore
            ):
                if ispkg:
                    continue
                module = importlib.import_module(f"evals.tasks.{modname}")
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, EvaluationTask)
                        and attr is not EvaluationTask
                        and hasattr(attr, "name")
                        and attr.name
                    ):
                        tasks[attr.name] = attr
        except (ImportError, ModuleNotFoundError):
            pass
        return tasks

    def register_task(self, task: EvaluationTask) -> None:
        """Manually register a task instance."""
        self._tasks[task.name] = task

    def get_available_tasks(self) -> Dict[str, str]:
        """Return {name: description} for all discovered + registered tasks."""
        result: Dict[str, str] = {}
        for name, cls in self.discover_tasks().items():
            result[name] = cls.description
        for name, instance in self._tasks.items():
            result[name] = instance.description
        return result

    def run_all(self, task_names: Optional[List[str]] = None) -> EvalReport:
        """Run all (or specific) evaluation tasks.

        Args:
            task_names: If given, only run these tasks (by name).

        Returns:
            EvalReport with all results.
        """
        report = EvalReport(
            config_snapshot={
                "embedding_model": self.config.embedding_model,
                "scene_detector": self.config.scene_detector,
                "top_k_retrieval": self.config.top_k_retrieval,
                "top_k_rerank": self.config.top_k_rerank,
            }
        )
        start_total = time.perf_counter()

        # Discover and instantiate tasks
        discovered = self.discover_tasks()
        all_task_instances: Dict[str, EvaluationTask] = dict(self._tasks)
        for name, cls in discovered.items():
            if task_names is None or name in task_names:
                all_task_instances[name] = cls(self.config)

        # Run
        for name, instance in all_task_instances.items():
            if task_names is not None and name not in task_names:
                continue
            result = instance.run()
            report.results.append(result)

        report.total_duration_ms = (time.perf_counter() - start_total) * 1000
        return report


# ── Convenience ──────────────────────────────────────────────────────────────


def run_evaluation(
    config: Config, task_names: Optional[List[str]] = None
) -> EvalReport:
    """Convenience function: create runner and run all tasks."""
    runner = EvaluationRunner(config)
    return runner.run_all(task_names=task_names)
