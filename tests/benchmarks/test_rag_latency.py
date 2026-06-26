"""Benchmarks for RAG retrieval and re-ranking latency.

These tests measure query-to-result latency for the VideoRAG pipeline,
including embedding, vector search, and optional ColBERT re-ranking.

Skipped by default (``-m "not benchmark"``).  Run with::

    pytest tests/benchmarks/ -v --benchmark-only
    pytest tests/benchmarks/ -v --benchmark-skip
"""

import logging
import shutil
import tempfile
from pathlib import Path

import pytest

from tests.benchmarks.conftest import GPUProfiler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rag_config() -> "Config":  # noqa: F821
    """Return a Config pointing at a temporary directory."""
    from video_analysis.config import Config

    tmp = Path(tempfile.mkdtemp(prefix="va_bench_rag_"))
    cfg = Config(data_dir=tmp)
    yield cfg
    shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_rag_harness_imports() -> None:
    """Verify RAG module can be imported (smoke)."""
    from video_analysis.config import Config  # noqa: F401
    from video_analysis.rag import VideoRAG  # noqa: F401


def test_rag_harness_init(rag_config: object) -> None:
    """Verify RAG initialisation (smoke)."""
    from video_analysis.rag import VideoRAG

    rag = VideoRAG(rag_config)  # type: ignore[arg-type]
    assert rag.config is not None


# ---------------------------------------------------------------------------
# Latency benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(min_rounds=3, warmup=True)
@pytest.mark.slow
def test_rag_import_time(benchmark: object, rag_config: object) -> None:
    """Benchmark how long VideoRAG takes to import and initialise."""

    def _init() -> None:
        from video_analysis.rag import VideoRAG

        VideoRAG(rag_config)  # type: ignore[arg-type]

    benchmark(_init)


@pytest.mark.benchmark(min_rounds=3, warmup=True)
@pytest.mark.slow
def test_rag_search_videos_empty(benchmark: object, rag_config: object) -> None:
    """Benchmark ``search_videos`` on an empty library (cold start)."""
    from video_analysis.rag import VideoRAG

    rag = VideoRAG(rag_config)  # type: ignore[arg-type]

    def _search() -> None:
        rag.search_videos("test query")

    benchmark(_search)


@pytest.mark.benchmark(min_rounds=3, warmup=True)
@pytest.mark.slow
def test_rag_get_library_info(benchmark: object, rag_config: object) -> None:
    """Benchmark ``get_library_info`` for a non-existent video ID."""
    from video_analysis.rag import VideoRAG

    rag = VideoRAG(rag_config)  # type: ignore[arg-type]

    def _info() -> None:
        rag.get_library_info("nonexistent_benchmark_video")

    benchmark(_info)


@pytest.mark.gpu
@pytest.mark.slow
def test_rag_gpu_memory(rag_config: object) -> None:
    """Measure GPU memory consumed by VideoRAG initialisation.

    Requires CUDA.  Skipped by default (``-m "gpu"``).
    """
    from video_analysis.rag import VideoRAG

    with GPUProfiler("VideoRAG init") as prof:
        VideoRAG(rag_config)  # type: ignore[arg-type]

    logger.info(
        "RAG init GPU — peak=%.0f MiB, elapsed=%.2fs", prof.peak_mib, prof.elapsed
    )
    assert prof.elapsed >= 0


# ---------------------------------------------------------------------------
# Query routing benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(min_rounds=3, warmup=True)
@pytest.mark.slow
def test_query_router_import(benchmark: object, rag_config: object) -> None:
    """Benchmark query_router import and initialisation."""
    from video_analysis.query_router import QueryRouter

    def _init() -> None:
        QueryRouter(rag_config)  # type: ignore[arg-type]

    benchmark(_init)


@pytest.mark.benchmark(min_rounds=3, warmup=True)
@pytest.mark.slow
def test_query_route_basic(benchmark: object, rag_config: object) -> None:
    """Benchmark routing a simple query through QueryRouter."""
    from video_analysis.query_router import QueryRouter

    router = QueryRouter(rag_config)  # type: ignore[arg-type]

    def _route() -> None:
        router.route("what objects are in the video?")

    benchmark(_route)


@pytest.mark.gpu
@pytest.mark.slow
def test_query_router_gpu_memory(rag_config: object) -> None:
    """Measure GPU memory consumed by QueryRouter init.

    Requires CUDA.  Skipped by default (``-m "gpu"``).
    """
    from video_analysis.query_router import QueryRouter

    with GPUProfiler("QueryRouter init") as prof:
        QueryRouter(rag_config)  # type: ignore[arg-type]

    logger.info(
        "QueryRouter GPU — peak=%.0f MiB, elapsed=%.2fs", prof.peak_mib, prof.elapsed
    )
    assert prof.elapsed >= 0
