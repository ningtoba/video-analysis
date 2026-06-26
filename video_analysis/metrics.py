"""
Prometheus Metrics — production monitoring for the video analysis platform.

Provides counters, histograms, and gauges exposed via the /metrics HTTP
endpoint on the FastAPI health app.  Metrics are lazy-initialised so
importing this module has zero side effects — the registry populates on
first metric access.

Usage (auto-initialising)::

    from video_analysis.metrics import (
        pipeline_runs_total,
        pipeline_duration_seconds,
        videos_indexed_total,
        questions_answered_total,
        retrieval_chunks_total,
        gpu_memory_bytes,
        chroma_collection_size,
        increment_pipeline_run,
        observe_pipeline_duration,
        increment_question,
        observe_retrieval,
        update_gpu_memory,
    )

    # At the end of pipeline.process():
    increment_pipeline_run(mode="video_full", success=True)
    observe_pipeline_duration(duration_s, mode="video_full")

    # After RAG indexing:
    videos_indexed_total.inc()

    # After Q&A:
    increment_question()
    observe_retrieval(chunks=5, method="hybrid")

    # Periodic GPU memory gauge update:
    update_gpu_memory()
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Lazy initialisation helpers ────────────────────────────────────────

_initialised = False


def _ensure_metrics() -> None:
    """Lazy-init all metric objects on first use."""
    global _initialised
    if _initialised:
        return

    global pipeline_runs_total, pipeline_runs_success_total, pipeline_runs_failure_total
    global pipeline_duration_seconds, pipeline_duration_histogram
    global videos_indexed_total, questions_answered_total
    global retrieval_chunks_total, retrieval_duration_seconds
    global gpu_memory_bytes, chroma_collection_size
    global active_sessions_gauge

    try:
        from prometheus_client import Counter, Gauge, Histogram
    except ImportError:
        logger.warning(
            "prometheus_client not installed — metrics are no-ops. "
            "Install with: pip install prometheus-client"
        )

        # Create dummy metrics that behave like prometheus_client but are no-ops
        class _NoopCollector:
            """A no-op metric collector that works when prometheus_client is absent."""

            def inc(self, amount: float = 1) -> None:
                pass

            def dec(self, amount: float = 1) -> None:
                pass

            def set(self, value: float) -> None:
                pass

            def observe(self, amount: float) -> None:
                pass

            def labels(self, **labels: str) -> _NoopCollector:
                return self

        _Noop = _NoopCollector

        pipeline_runs_total = _Noop()
        pipeline_runs_success_total = _Noop()
        pipeline_runs_failure_total = _Noop()
        pipeline_duration_seconds = _Noop()
        pipeline_duration_histogram = _Noop()
        videos_indexed_total = _Noop()
        questions_answered_total = _Noop()
        retrieval_chunks_total = _Noop()
        retrieval_duration_seconds = _Noop()
        gpu_memory_bytes = _Noop()
        chroma_collection_size = _Noop()
        active_sessions_gauge = _Noop()
        _initialised = True
        return

    # ── Counters ───────────────────────────────────────────────────────
    pipeline_runs_total = Counter(
        "va_pipeline_runs_total",
        "Total number of video analysis pipeline runs",
        ["mode"],  # video_full, audio_only
    )
    pipeline_runs_success_total = Counter(
        "va_pipeline_runs_success_total",
        "Pipeline runs that completed successfully",
        ["mode"],
    )
    pipeline_runs_failure_total = Counter(
        "va_pipeline_runs_failure_total",
        "Pipeline runs that failed with an error",
        ["mode"],
    )
    videos_indexed_total = Counter(
        "va_videos_indexed_total",
        "Total number of videos successfully indexed in ChromaDB",
    )
    questions_answered_total = Counter(
        "va_questions_answered_total",
        "Total number of Q&A questions answered",
        ["method"],  # simple, agentic
    )

    # ── Histograms ─────────────────────────────────────────────────────
    pipeline_duration_seconds = Histogram(
        "va_pipeline_duration_seconds",
        "Pipeline total duration in seconds",
        ["mode"],
        buckets=(10, 30, 60, 120, 180, 240, 300, 420, 600, 900, 1800, float("inf")),
    )
    pipeline_duration_histogram = pipeline_duration_seconds  # alias
    retrieval_duration_seconds = Histogram(
        "va_retrieval_duration_seconds",
        "Time to perform a single retrieval+rerank cycle",
        ["method"],  # simple, agentic, multi_hop, scene_graph, self_check
        buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, float("inf")),
    )

    # ── Gauges ─────────────────────────────────────────────────────────
    gpu_memory_bytes = Gauge(
        "va_gpu_memory_bytes",
        "Current GPU memory allocated in bytes (0 if no GPU or no torch)",
    )
    chroma_collection_size = Gauge(
        "va_chroma_collection_size",
        "Number of documents in the ChromaDB collection",
    )
    active_sessions_gauge = Gauge(
        "va_active_sessions",
        "Number of currently active UI sessions",
    )

    _initialised = True


# ── Convenience helpers ────────────────────────────────────────────────


def increment_pipeline_run(
    *,
    mode: str = "video_full",
    success: bool = True,
    duration_s: Optional[float] = None,
) -> None:
    """Record a pipeline run outcome.

    Args:
        mode: Processing mode (``video_full``, ``audio_only``).
        success: Whether the run completed without error.
        duration_s: If provided, also records pipeline duration.
    """
    _ensure_metrics()
    pipeline_runs_total.labels(mode=mode).inc()
    if success:
        pipeline_runs_success_total.labels(mode=mode).inc()
    else:
        pipeline_runs_failure_total.labels(mode=mode).inc()
    if duration_s is not None and duration_s >= 0:
        pipeline_duration_seconds.labels(mode=mode).observe(duration_s)


def observe_pipeline_duration(duration_s: float, *, mode: str = "video_full") -> None:
    """Record pipeline duration without changing run counters."""
    _ensure_metrics()
    if duration_s >= 0:
        pipeline_duration_seconds.labels(mode=mode).observe(duration_s)


def increment_question(*, method: str = "simple") -> None:
    """Record a Q&A question answered."""
    _ensure_metrics()
    questions_answered_total.labels(method=method).inc()


def observe_retrieval(
    *,
    chunks: int = 0,
    method: str = "simple",
    duration_s: Optional[float] = None,
) -> None:
    """Record a retrieval operation (+optional duration).

    Args:
        chunks: Number of chunks returned from retrieval.
        method: Retrieval method (simple, agentic, multi_hop, scene_graph, self_check).
        duration_s: If provided, also records retrieval duration.
    """
    _ensure_metrics()
    if chunks > 0:
        retrieval_chunks_total.inc(chunks)
    if duration_s is not None and duration_s >= 0:
        retrieval_duration_seconds.labels(method=method).observe(duration_s)


def update_gpu_memory() -> None:
    """Sample current GPU memory and update the gauge.

    Safe to call even without a GPU — sets to 0 and returns.
    """
    _ensure_metrics()
    try:
        import torch

        if torch.cuda.is_available():
            gpu_memory_bytes.set(torch.cuda.memory_allocated())
        else:
            gpu_memory_bytes.set(0)
    except (ImportError, RuntimeError):
        gpu_memory_bytes.set(0)


def update_chroma_collection_size(count: int) -> None:
    """Set the ChromaDB collection document count."""
    _ensure_metrics()
    chroma_collection_size.set(max(0, count))


# ── Declaration stubs (initialised on first access) ───────────────────
pipeline_runs_total = None  # type: ignore[assignment]
pipeline_runs_success_total = None  # type: ignore[assignment]
pipeline_runs_failure_total = None  # type: ignore[assignment]
pipeline_duration_seconds = None  # type: ignore[assignment]
pipeline_duration_histogram = None  # type: ignore[assignment]
videos_indexed_total = None  # type: ignore[assignment]
questions_answered_total = None  # type: ignore[assignment]
retrieval_chunks_total = None  # type: ignore[assignment]
retrieval_duration_seconds = None  # type: ignore[assignment]
gpu_memory_bytes = None  # type: ignore[assignment]
chroma_collection_size = None  # type: ignore[assignment]
active_sessions_gauge = None  # type: ignore[assignment]

# ── Expose metrics endpoint handler for FastAPI integration ───────────


def metrics_endpoint() -> str:
    """Generate the Prometheus ``/metrics`` response body.

    Returns the plain-text exposition format that Prometheus scrapes.
    Returns an empty string with a comment if prometheus_client is not
    installed.
    """
    _ensure_metrics()
    try:
        from prometheus_client import generate_latest, REGISTRY

        return generate_latest(REGISTRY).decode("utf-8")
    except ImportError:
        return "# prometheus_client not installed — no metrics available\n"
    except Exception as exc:
        logger.error("Failed to generate Prometheus metrics: %s", exc)
        return "# Error generating Prometheus metrics\n"
