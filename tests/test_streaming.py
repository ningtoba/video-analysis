"""Tests for the streaming pipeline module (v0.32.0)."""

import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from video_analysis.streaming import (
    StreamingPipeline,
    StreamingChunkResult,
    _ffprobe_duration,
)
from video_analysis.config import Config
from video_analysis.models import SceneInfo, TranscriptSegment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_config():
    """Provide a Config with a temp data dir, cleaned up after."""
    tmp = Path(tempfile.mkdtemp(prefix="va_stream_test_"))
    cfg = Config(data_dir=str(tmp))
    yield cfg
    import shutil

    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def streaming_pipeline(clean_config):
    """Create a StreamingPipeline with a clean temp config."""
    return StreamingPipeline(clean_config)


# ---------------------------------------------------------------------------
# StreamingChunkResult dataclass tests
# ---------------------------------------------------------------------------


def test_streaming_chunk_result_defaults():
    """Verify StreamingChunkResult dataclass fields and defaults."""
    result = StreamingChunkResult(
        chunk_index=0,
        start_time=0.0,
        end_time=30.0,
        duration=30.0,
    )
    assert result.chunk_index == 0
    assert result.start_time == 0.0
    assert result.end_time == 30.0
    assert result.duration == 30.0
    assert result.scenes == []
    assert result.transcript_segments == []
    assert result.full_transcript == ""
    assert result.objects_found == []
    assert result.has_video is True
    assert result.metadata == {}


def test_streaming_chunk_result_with_data():
    """Verify StreamingChunkResult with all fields populated."""
    scenes = [
        SceneInfo(scene_id=0, start_time=0.0, end_time=15.0),
        SceneInfo(scene_id=1, start_time=15.0, end_time=30.0),
    ]
    transcript = [
        TranscriptSegment(start=0.0, end=2.5, text="Hello"),
        TranscriptSegment(start=2.5, end=5.0, text="world"),
    ]
    result = StreamingChunkResult(
        chunk_index=1,
        start_time=30.0,
        end_time=60.0,
        duration=30.0,
        scenes=scenes,
        transcript_segments=transcript,
        full_transcript="Hello world",
        objects_found=["person", "car"],
        has_video=True,
        metadata={"video_id": "test"},
    )
    assert result.chunk_index == 1
    assert len(result.scenes) == 2
    assert len(result.transcript_segments) == 2
    assert result.full_transcript == "Hello world"
    assert result.objects_found == ["person", "car"]
    assert result.metadata["video_id"] == "test"


# ---------------------------------------------------------------------------
# _ffprobe_duration tests
# ---------------------------------------------------------------------------


def test_ffprobe_duration_missing_file():
    """Verify _ffprobe_duration returns None for non-existent files."""
    result = _ffprobe_duration(Path("/nonexistent/video.mp4"))
    assert result is None


def test_ffprobe_duration_empty_path():
    """Verify _ffprobe_duration handles Path to non-existent file."""
    result = _ffprobe_duration(Path(""))
    # Empty string path doesn't exist
    assert result is None


# ---------------------------------------------------------------------------
# StreamingPipeline tests
# ---------------------------------------------------------------------------


def test_streaming_pipeline_init(clean_config):
    """Verify StreamingPipeline initialisation."""
    pipeline = StreamingPipeline(clean_config)
    assert pipeline.config is not None
    assert pipeline._pipeline is None  # lazy init
    assert pipeline._temp_dir is not None
    assert pipeline._temp_dir.exists()
    assert pipeline.stats["chunks_processed"] == 0
    assert pipeline.stats["total_scenes"] == 0
    assert pipeline.stats["total_transcript_segments"] == 0


def test_streaming_pipeline_init_no_config():
    """Verify StreamingPipeline works with default config."""
    pipeline = StreamingPipeline()
    assert pipeline.config is not None
    assert pipeline._temp_dir is not None


def test_streaming_pipeline_properties(streaming_pipeline):
    """Verify initial property values."""
    pipeline = streaming_pipeline
    assert pipeline._processed_chunks == 0
    assert pipeline._all_scenes == []
    assert pipeline._all_objects == set()
    assert pipeline._video_id is None
    assert pipeline.final_index() is None


def test_streaming_pipeline_process_invalid_path(streaming_pipeline):
    """Verify process_streaming raises FileNotFoundError for invalid paths."""
    with pytest.raises(FileNotFoundError):
        for _ in streaming_pipeline.process_streaming("/nonexistent/video.mp4"):
            pass


def test_streaming_pipeline_process_empty_path(streaming_pipeline):
    """Verify process_streaming raises FileNotFoundError for empty paths."""
    with pytest.raises(FileNotFoundError):
        for _ in streaming_pipeline.process_streaming(""):
            pass


def test_streaming_pipeline_live_invalid_source(streaming_pipeline):
    """Verify process_live raises FileNotFoundError for invalid sources."""
    with pytest.raises(FileNotFoundError):
        for _ in streaming_pipeline.process_live("/nonexistent/source.mp4"):
            pass


# ---------------------------------------------------------------------------
# _segment_video tests
# ---------------------------------------------------------------------------


@patch("video_analysis.streaming._ffprobe_duration")
def test_segment_video_no_duration(mock_duration, streaming_pipeline):
    """Verify _segment_video handles None duration gracefully."""
    mock_duration.return_value = None
    segments = streaming_pipeline._segment_video(
        Path("test.mp4"), chunk_duration=30.0, overlap=2.0
    )
    assert segments == []


@patch("video_analysis.streaming._ffprobe_duration")
def test_segment_video_zero_duration(mock_duration, streaming_pipeline):
    """Verify _segment_video handles zero duration gracefully."""
    mock_duration.return_value = 0.0
    segments = streaming_pipeline._segment_video(
        Path("test.mp4"), chunk_duration=30.0, overlap=2.0
    )
    assert segments == []


@patch("video_analysis.streaming._ffprobe_duration")
def test_segment_video_basic(mock_duration, streaming_pipeline):
    """Verify _segment_video produces correct segment boundaries."""
    mock_duration.return_value = 90.0
    segments = streaming_pipeline._segment_video(
        Path("test.mp4"), chunk_duration=30.0, overlap=2.0
    )
    # 90s at 30s grid with 2s overlap = 3 segments
    assert len(segments) == 3
    assert segments[0] == (0.0, 30.0)
    assert segments[1] == (28.0, 60.0)
    assert segments[2] == (58.0, 90.0)


@patch("video_analysis.streaming._ffprobe_duration")
def test_segment_video_exact_fit(mock_duration, streaming_pipeline):
    """Verify _segment_video handles exact-fit durations."""
    mock_duration.return_value = 30.0
    segments = streaming_pipeline._segment_video(
        Path("test.mp4"), chunk_duration=30.0, overlap=2.0
    )
    assert len(segments) == 1
    assert segments[0] == (0.0, 30.0)


@patch("video_analysis.streaming._ffprobe_duration")
def test_segment_video_short(mock_duration, streaming_pipeline):
    """Verify _segment_video handles very short videos (single chunk)."""
    mock_duration.return_value = 5.0
    segments = streaming_pipeline._segment_video(
        Path("test.mp4"), chunk_duration=30.0, overlap=2.0
    )
    assert len(segments) == 1
    assert segments[0] == (0.0, 5.0)


@patch("video_analysis.streaming._ffprobe_duration")
def test_segment_video_no_overlap(mock_duration, streaming_pipeline):
    """Verify _segment_video works with zero overlap."""
    mock_duration.return_value = 90.0
    segments = streaming_pipeline._segment_video(
        Path("test.mp4"), chunk_duration=30.0, overlap=0.0
    )
    assert len(segments) == 3
    assert segments[0] == (0.0, 30.0)
    assert segments[1] == (30.0, 60.0)
    assert segments[2] == (60.0, 90.0)


# ---------------------------------------------------------------------------
# _process_segment tests
# ---------------------------------------------------------------------------


def test_process_segment_invalid_duration(streaming_pipeline):
    """Verify _process_segment handles zero/negative durations."""
    result = streaming_pipeline._process_segment(
        Path("test.mp4"), start_time=10.0, end_time=10.0, chunk_index=0, video_id="test"
    )
    assert result is None

    result = streaming_pipeline._process_segment(
        Path("test.mp4"), start_time=10.0, end_time=5.0, chunk_index=0, video_id="test"
    )
    assert result is None


def test_process_segment_missing_file(streaming_pipeline):
    """Verify _process_segment handles missing video file."""
    result = streaming_pipeline._process_segment(
        Path("/nonexistent/video.mp4"),
        start_time=0.0,
        end_time=30.0,
        chunk_index=0,
        video_id="test",
    )
    assert result is None


# ---------------------------------------------------------------------------
# stats tests
# ---------------------------------------------------------------------------


def test_stats_empty(streaming_pipeline):
    """Verify stats returns correct values for unprocessed pipeline."""
    stats = streaming_pipeline.stats
    assert stats["chunks_processed"] == 0
    assert stats["total_scenes"] == 0
    assert stats["total_transcript_segments"] == 0
    assert stats["unique_objects"] == 0


def test_stats_after_chunks(streaming_pipeline):
    """Verify stats reflect accumulated chunks."""
    # Simulate adding results
    result1 = StreamingChunkResult(
        chunk_index=0, start_time=0.0, end_time=30.0, duration=30.0
    )
    result2 = StreamingChunkResult(
        chunk_index=1,
        start_time=30.0,
        end_time=60.0,
        duration=30.0,
        objects_found=["person"],
    )

    streaming_pipeline._chunk_results = [result1, result2]
    streaming_pipeline._processed_chunks = 2
    streaming_pipeline._all_objects = {"person", "car"}

    stats = streaming_pipeline.stats
    assert stats["chunks_processed"] == 2
    assert stats["unique_objects"] == 2


# ---------------------------------------------------------------------------
# final_index tests
# ---------------------------------------------------------------------------


def test_final_index_empty(streaming_pipeline):
    """Verify final_index returns None when no chunks processed."""
    assert streaming_pipeline.final_index() is None


def test_final_index_with_data(streaming_pipeline):
    """Verify final_index merges accumulated data."""
    scenes = [SceneInfo(scene_id=0, start_time=0.0, end_time=10.0)]
    transcript = [TranscriptSegment(start=0.0, end=5.0, text="hello")]

    streaming_pipeline._video_id = "test_video"
    streaming_pipeline._all_scenes = scenes
    streaming_pipeline._all_transcript_segments = transcript
    streaming_pipeline._all_transcript_text = ["hello"]
    streaming_pipeline._chunk_results = [
        StreamingChunkResult(
            chunk_index=0, start_time=0.0, end_time=30.0, duration=30.0
        )
    ]

    index = streaming_pipeline.final_index()
    assert index is not None
    assert index.video_id == "test_video"
    assert len(index.scenes) == 1
    assert len(index.transcript) == 1
    assert index.full_transcript == "hello"


# ---------------------------------------------------------------------------
# _build_final_index tests
# ---------------------------------------------------------------------------


def test_build_final_index(streaming_pipeline):
    """Verify _build_final_index creates a proper VideoIndex."""
    scenes = [SceneInfo(scene_id=0, start_time=0.0, end_time=10.0)]
    streaming_pipeline._all_scenes = scenes
    streaming_pipeline._all_transcript_segments = [
        TranscriptSegment(start=0.0, end=5.0, text="hello")
    ]
    streaming_pipeline._all_transcript_text = ["hello", "world"]

    index = streaming_pipeline._build_final_index(
        "test_video", Path("/tmp/test.mp4"), duration=30.0
    )
    assert index.video_id == "test_video"
    assert index.filename == "test.mp4"
    assert index.duration == 30.0
    assert index.filepath == "/tmp/test.mp4"
    assert len(index.scenes) == 1
    assert index.full_transcript == "hello world"


# ---------------------------------------------------------------------------
# _index_chunk / _index_final tests
# ---------------------------------------------------------------------------


@patch("video_analysis.rag.VideoRAG")
def test_index_chunk_called(mock_rag, streaming_pipeline):
    """Verify _index_chunk creates a VideoRAG and calls index_video."""
    result = StreamingChunkResult(
        chunk_index=0,
        start_time=0.0,
        end_time=30.0,
        duration=30.0,
        metadata={"video_id": "test_video"},
    )
    streaming_pipeline._index_chunk(result)
    mock_rag_instance = mock_rag.return_value
    mock_rag_instance.index_video.assert_called_once()


@patch("video_analysis.rag.VideoRAG")
def test_index_chunk_no_video_id(mock_rag, streaming_pipeline):
    """Verify _index_chunk works without metadata video_id."""
    result = StreamingChunkResult(
        chunk_index=0, start_time=0.0, end_time=30.0, duration=30.0, metadata={}
    )
    streaming_pipeline._index_chunk(result)
    # Should not raise — generates its own video_id
    mock_rag_instance = mock_rag.return_value
    assert mock_rag_instance.index_video.call_count >= 1


@patch("video_analysis.rag.VideoRAG")
def test_index_final_called(mock_rag, streaming_pipeline):
    """Verify _index_final creates a VideoRAG and calls index_video."""
    scenes = [SceneInfo(scene_id=0, start_time=0.0, end_time=10.0)]
    streaming_pipeline._all_scenes = scenes
    streaming_pipeline._all_transcript_text = ["hello"]

    index = streaming_pipeline._build_final_index(
        "test_video", Path("/tmp/test.mp4"), duration=30.0
    )
    streaming_pipeline._index_final(index)
    mock_rag_instance = mock_rag.return_value
    mock_rag_instance.index_video.assert_called_once_with(index)


# ---------------------------------------------------------------------------
# Configuration integration tests
# ---------------------------------------------------------------------------


def test_streaming_config_env_vars():
    """Verify streaming config can be overridden by env vars."""
    import os

    os.environ["STREAMING_ENABLED"] = "true"
    os.environ["STREAMING_CHUNK_DURATION"] = "15.0"
    os.environ["STREAMING_OVERLAP"] = "3.0"
    os.environ["STREAMING_INCREMENTAL_INDEX"] = "false"
    os.environ["STREAMING_MAX_CHUNKS"] = "10"

    try:
        cfg = Config(data_dir="/tmp/va_stream_env_test")
        assert cfg.streaming_enabled is True
        assert cfg.streaming_chunk_duration == 15.0
        assert cfg.streaming_overlap == 3.0
        assert cfg.streaming_incremental_index is False
        assert cfg.streaming_max_chunks == 10
    finally:
        for key in [
            "STREAMING_ENABLED",
            "STREAMING_CHUNK_DURATION",
            "STREAMING_OVERLAP",
            "STREAMING_INCREMENTAL_INDEX",
            "STREAMING_MAX_CHUNKS",
        ]:
            os.environ.pop(key, None)
        import shutil

        shutil.rmtree("/tmp/va_stream_env_test", ignore_errors=True)


def test_streaming_config_defaults():
    """Verify streaming config defaults are sensible."""
    cfg = Config(data_dir="/tmp/va_stream_defaults_test")
    assert cfg.streaming_enabled is False
    assert cfg.streaming_chunk_duration == 30.0
    assert cfg.streaming_overlap == 2.0
    assert cfg.streaming_incremental_index is True
    assert cfg.streaming_max_chunks == 0  # unlimited
    import shutil

    shutil.rmtree("/tmp/va_stream_defaults_test", ignore_errors=True)


# ---------------------------------------------------------------------------
# Integration sanity: streaming module imports as expected
# ---------------------------------------------------------------------------


def test_streaming_module_importable():
    """Verify streaming module is importable."""
    from video_analysis import streaming  # noqa: F811

    assert hasattr(streaming, "StreamingPipeline")
    assert hasattr(streaming, "StreamingChunkResult")
    assert hasattr(streaming, "_ffprobe_duration")


def test_streaming_pipeline_exported():
    """Verify StreamingPipeline is exported from video_analysis."""
    from video_analysis import StreamingPipeline  # noqa: F401
    from video_analysis import streaming  # noqa: F811

    assert StreamingPipeline is streaming.StreamingPipeline


def test_version_bumped():
    """Verify version is bumped to 0.33.0."""
    from video_analysis import __version__

    assert __version__ == "0.39.0"
