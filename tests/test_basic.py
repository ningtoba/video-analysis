"""Tests for video analysis platform."""

import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from video_analysis.config import Config
from video_analysis.models import SceneInfo, FrameInfo, TranscriptSegment, VideoIndex


def test_config_defaults():
    """Test that config creates proper paths."""
    cfg = Config(data_dir="/tmp/test_va_data")
    assert cfg.data_dir.exists()
    assert cfg.video_dir.exists()
    assert cfg.frames_dir.exists()
    assert cfg.audio_dir.exists()
    assert cfg.whisper_model == "large-v3"
    assert cfg.whisper_device == "cuda"

    # Cleanup
    import shutil

    shutil.rmtree("/tmp/test_va_data", ignore_errors=True)


def test_scene_info():
    scene = SceneInfo(scene_id=0, start_time=10.0, end_time=30.5)
    assert scene.scene_id == 0
    assert scene.start_time == 10.0
    assert scene.end_time == 30.5
    assert len(scene.key_frames) == 0


def test_frame_info():
    frame = FrameInfo(timestamp=15.0, filepath="/tmp/frame.jpg", scene_id=0)
    assert frame.timestamp == 15.0
    assert frame.filepath == "/tmp/frame.jpg"
    assert frame.objects == []


def test_transcript_segment():
    seg = TranscriptSegment(start=0.0, end=2.5, text="Hello world")
    assert seg.start == 0.0
    assert seg.end == 2.5
    assert seg.text == "Hello world"


def test_video_index():
    index = VideoIndex(
        video_id="test",
        filename="test.mp4",
        duration=120.0,
        filepath="/tmp/test.mp4",
    )
    d = index.to_dict()
    assert d["video_id"] == "test"
    assert d["duration"] == 120.0
    assert len(d["scenes"]) == 0


def test_format_timestamp():
    from video_analysis.models import format_timestamp

    assert format_timestamp(0) == "00:00:00.000"
    assert format_timestamp(3661.5) == "01:01:01.500"


def test_config_custom_dir():
    cfg = Config(data_dir="/tmp/va_test_custom")
    assert cfg.data_dir.name == "va_test_custom"
    assert cfg.video_dir.parent == cfg.data_dir
    import shutil

    shutil.rmtree("/tmp/va_test_custom", ignore_errors=True)


if __name__ == "__main__":
    test_config_defaults()
    test_scene_info()
    test_frame_info()
    test_transcript_segment()
    test_video_index()
    test_format_timestamp()
    test_config_custom_dir()
    print("All tests passed! ✅")
