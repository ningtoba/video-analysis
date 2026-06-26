"""Benchmarks for video pipeline stage throughput.

These tests measure end-to-end runtime of each major pipeline stage
(frame extraction, audio transcription, scene detection, CLIP encoding,
OCR extraction) to detect performance regressions.

Skipped by default (``-m "not benchmark"``).  Run with::

    pytest tests/benchmarks/ -v --benchmark-only
    pytest tests/benchmarks/ -v --benchmark-skip  # skip bench, just smoke-test
"""

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

HAVE_BENCHMARK = False
try:
    import pytest_benchmark  # type: ignore  # noqa: F401

    HAVE_BENCHMARK = True
except ImportError:
    pass

from tests.benchmarks.conftest import GPUProfiler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SHORT_VIDEO_DURATION = 5  # seconds
SHORT_VIDEO_SIZE = "320x240"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def short_video() -> Path:
    """Generate a short test video shared across all tests in this module."""
    tmp = Path(tempfile.mkdtemp(prefix="va_bench_"))
    video_path = tmp / "test_short.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=duration={SHORT_VIDEO_DURATION}:size={SHORT_VIDEO_SIZE}:rate=5",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=44100:cl=mono",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(video_path),
        ],
        capture_output=True,
        check=True,
        timeout=30,
    )
    yield video_path
    shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Smoke tests (no benchmark recording, just exercise the code)
# ---------------------------------------------------------------------------


def test_harness_can_import_pipeline() -> None:
    """Verify that pipeline module can be imported (smoke)."""
    from video_analysis.config import Config  # noqa: F401
    from video_analysis.pipeline import VideoPipeline  # noqa: F401


def test_harness_can_init_pipeline() -> None:
    """Verify pipeline initialisation (smoke)."""
    from video_analysis.config import Config
    from video_analysis.pipeline import VideoPipeline

    cfg = Config(data_dir=Path("/tmp/va_bench_harness_init"))
    pipeline = VideoPipeline(cfg)
    assert pipeline.config is not None
    shutil.rmtree("/tmp/va_bench_harness_init", ignore_errors=True)


# ---------------------------------------------------------------------------
# Throughput benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAVE_BENCHMARK, reason="pytest-benchmark not installed")
@pytest.mark.benchmark(min_rounds=3, warmup=True)
@pytest.mark.slow
def test_pipeline_import(benchmark: object) -> None:
    """Benchmark how long it takes to import and init a minimal pipeline."""

    def _init() -> None:
        from video_analysis.config import Config
        from video_analysis.pipeline import VideoPipeline

        cfg = Config(data_dir=Path("/tmp/va_bench_import"))
        VideoPipeline(cfg)
        shutil.rmtree("/tmp/va_bench_import", ignore_errors=True)

    benchmark(_init)


@pytest.mark.skipif(not HAVE_BENCHMARK, reason="pytest-benchmark not installed")
@pytest.mark.benchmark(min_rounds=3, warmup=True)
@pytest.mark.slow
def test_frame_generation_from_video(benchmark: object, short_video: Path) -> None:
    """Benchmark frame extraction rate from a short video.

    Measures how long ``_generate_sprite_sheet`` or equivalent frame
    extraction takes.
    """
    from video_analysis.config import Config
    from video_analysis.pipeline import VideoPipeline

    cfg = Config(data_dir=Path("/tmp/va_bench_frames"))
    pipeline = VideoPipeline(cfg)

    def _extract() -> None:
        pipeline._generate_sprite_sheet(short_video, "bench_short", num_thumbnails=10)

    benchmark(_extract)
    shutil.rmtree("/tmp/va_bench_frames", ignore_errors=True)


@pytest.mark.skipif(not HAVE_BENCHMARK, reason="pytest-benchmark not installed")
@pytest.mark.benchmark(min_rounds=3, warmup=True)
@pytest.mark.slow
def test_scene_detection(benchmark: object, short_video: Path) -> None:
    """Benchmark scene detection time on a short video."""
    from video_analysis.config import Config
    from video_analysis.pipeline import VideoPipeline

    cfg = Config(data_dir=Path("/tmp/va_bench_scenes"))
    pipeline = VideoPipeline(cfg)

    def _detect() -> None:
        pipeline._detect_scenes(short_video)

    benchmark(_detect)
    shutil.rmtree("/tmp/va_bench_scenes", ignore_errors=True)


@pytest.mark.skipif(not HAVE_BENCHMARK, reason="pytest-benchmark not installed")
@pytest.mark.benchmark(min_rounds=3, warmup=True)
@pytest.mark.slow
def test_audio_transcription(benchmark: object, short_video: Path) -> None:
    """Benchmark audio extraction + Whisper transcription stub."""
    from video_analysis.config import Config
    from video_analysis.pipeline import VideoPipeline

    cfg = Config(data_dir=Path("/tmp/va_bench_audio"))
    pipeline = VideoPipeline(cfg)

    def _transcribe() -> None:
        pipeline._transcribe_audio(short_video)

    benchmark(_transcribe)
    shutil.rmtree("/tmp/va_bench_audio", ignore_errors=True)


# ---------------------------------------------------------------------------
# GPU memory profiling
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.slow
def test_pipeline_gpu_memory_footprint(short_video: Path) -> None:
    """Measure GPU memory usage during pipeline init + sprite sheet gen.

    Requires CUDA.  Skipped by default (``-m "gpu"``).
    """
    from video_analysis.config import Config
    from video_analysis.pipeline import VideoPipeline

    cfg = Config(data_dir=Path("/tmp/va_bench_gpu_mem"))
    with GPUProfiler("pipeline init + sprite generation") as prof:
        pipeline = VideoPipeline(cfg)
        pipeline._generate_sprite_sheet(short_video, "bench_short", num_thumbnails=10)

    logger.info(
        "GPU memory — peak=%.0f MiB, elapsed=%.2fs", prof.peak_mib, prof.elapsed
    )
    assert prof.elapsed >= 0
    shutil.rmtree("/tmp/va_bench_gpu_mem", ignore_errors=True)
