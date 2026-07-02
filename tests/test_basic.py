"""Tests for video analysis platform."""

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from video_analysis.config import Config
from video_analysis.models import TranscriptSegment

logger = logging.getLogger(__name__)


# ====================================================================
# Core data model tests
# ====================================================================


def test_transcript_segment():
    seg = TranscriptSegment(start=0.0, end=2.5, text="Hello world")
    assert seg.start == 0.0
    assert seg.end == 2.5
    assert seg.text == "Hello world"


def test_format_timestamp():
    from video_analysis.models import format_timestamp

    assert format_timestamp(0) == "00:00:00.000"
    assert format_timestamp(3661.5) == "01:01:01.500"


def test_yt_dlp_import():
    """Test that yt-dlp can be imported (optional dep)."""
    try:
        import yt_dlp

        assert yt_dlp is not None
    except ImportError:
        pass  # optional dependency, not required


# ====================================================================
# Pipeline tests
# ====================================================================


def test_pipeline_imports():
    """Test that pipeline can be imported cleanly."""
    from video_analysis.pipeline import VideoPipeline

    p = VideoPipeline(Config(data_dir=Path("/tmp/va_test_pipeline")))
    assert p.config is not None
    import shutil

    shutil.rmtree("/tmp/va_test_pipeline", ignore_errors=True)


def test_pipeline_cleanup():
    """Test that pipeline.cleanup() exists and doesn't crash."""
    from video_analysis.pipeline import VideoPipeline

    config = Config(data_dir=Path("/tmp/va_test_cleanup"))
    pipeline = VideoPipeline(config)
    # Should not crash even with no models loaded
    pipeline.cleanup()
    import shutil

    shutil.rmtree("/tmp/va_test_cleanup", ignore_errors=True)


# ====================================================================
# Config tests
# ====================================================================


def test_config_scene_detector_options():
    """Test scene_detector supports all available options."""
    cfg = Config(data_dir=Path("/tmp/va_test_scene_opt"))
    assert cfg.scene_detector in ("adaptive", "content", "ffmpeg", "histogram", "hash")
    # Verify the options are handled in pipeline
    from video_analysis.pipeline import VideoPipeline

    pipeline = VideoPipeline(cfg)
    assert pipeline.config.scene_detector == "adaptive"
    import shutil

    shutil.rmtree("/tmp/va_test_scene_opt", ignore_errors=True)


def test_config_processing_mode_default():
    """Test that processing_mode defaults to 'video_full'."""
    cfg = Config(data_dir=Path("/tmp/va_test_proc_mode"))
    assert cfg.processing_mode == "video_full"
    import shutil

    shutil.rmtree("/tmp/va_test_proc_mode", ignore_errors=True)

# ====================================================================
# Health module tests
# ====================================================================

def test_health_check_module():
    """Test health module can be imported and has expected structure."""
    from ui.health import HealthStatus, add_health_endpoints

    # Verify the module is importable and classes exist
    assert HealthStatus.__name__ == "HealthStatus"
    assert callable(add_health_endpoints)


# ====================================================================
# Version test
# ====================================================================


def test_version_current():
    """Test that the package version is the current release."""
    from video_analysis import __version__

    assert __version__ == "0.0.0"
