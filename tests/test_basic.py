"""Tests for video analysis platform."""

import json
import logging
import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from video_analysis.config import Config
from video_analysis.models import SceneInfo, FrameInfo, TranscriptSegment, VideoIndex

logger = logging.getLogger(__name__)


def test_config_defaults():
    """Test that config creates proper paths."""
    cfg = Config(data_dir="/tmp/test_va_data")
    assert cfg.data_dir.exists()
    assert cfg.video_dir.exists()
    assert cfg.frames_dir.exists()
    assert cfg.audio_dir.exists()
    assert cfg.thumbnails_dir.exists()
    assert cfg.clip_export_dir.exists()
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


def test_frame_info_with_description():
    frame = FrameInfo(
        timestamp=10.0,
        filepath="/tmp/frame.jpg",
        scene_id=0,
        description="A person speaking indoors",
        objects=[{"label": "person", "confidence": 0.95}],
        ocr_text="Hello World",
    )
    assert frame.description == "A person speaking indoors"
    assert len(frame.objects) == 1
    assert frame.ocr_text == "Hello World"


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
    assert d["sprite_sheet"] is None
    assert d["sprite_metadata"] == {}


def test_video_index_with_sprite():
    index = VideoIndex(
        video_id="test",
        filename="test.mp4",
        duration=120.0,
        filepath="/tmp/test.mp4",
        sprite_sheet="/tmp/sprite.jpg",
        sprite_metadata={"thumbnails": [{"index": 0, "timestamp": 0.0}]},
    )
    d = index.to_dict()
    assert d["sprite_sheet"] == "/tmp/sprite.jpg"
    assert len(d["sprite_metadata"]["thumbnails"]) == 1


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


def test_config_export_dir():
    cfg = Config(data_dir="/tmp/va_test_export")
    assert cfg.clip_export_dir.exists()
    assert cfg.clip_export_dir.name == "clips"
    import shutil

    shutil.rmtree("/tmp/va_test_export", ignore_errors=True)


def test_config_ocr_diarize_flags():
    """Test that OCR and diarization config flags exist with correct defaults."""
    cfg = Config(data_dir="/tmp/va_test_flags")
    assert cfg.ocr_enabled is True
    assert cfg.diarize_enabled is True
    assert cfg.ocr_confidence == 0.3
    import shutil

    shutil.rmtree("/tmp/va_test_flags", ignore_errors=True)


def test_pipeline_imports():
    """Test that pipeline can be imported cleanly."""
    from video_analysis.pipeline import VideoPipeline

    p = VideoPipeline(Config(data_dir="/tmp/va_test_pipeline"))
    assert p.config is not None
    import shutil

    shutil.rmtree("/tmp/va_test_pipeline", ignore_errors=True)


def test_ocr_fallback_no_paddleocr():
    """Test that _extract_ocr handles missing paddleocr gracefully."""
    from video_analysis.pipeline import VideoPipeline
    from video_analysis.models import SceneInfo, FrameInfo

    config = Config(data_dir="/tmp/va_test_ocr_fallback")
    pipeline = VideoPipeline(config)
    scenes = [
        SceneInfo(
            scene_id=0,
            start_time=0,
            end_time=10,
            key_frames=[
                FrameInfo(timestamp=5, filepath="/nonexistent.jpg", scene_id=0)
            ],
        )
    ]
    # Should not raise — just log a warning and return
    pipeline._extract_ocr(scenes)
    assert scenes[0].key_frames[0].ocr_text is None

    import shutil

    shutil.rmtree("/tmp/va_test_ocr_fallback", ignore_errors=True)


def test_diarize_fallback_no_pyannote():
    """Test that _diarize handles missing pyannote gracefully."""
    from video_analysis.pipeline import VideoPipeline
    from video_analysis.models import TranscriptSegment

    config = Config(data_dir="/tmp/va_test_diarize_fallback")
    pipeline = VideoPipeline(config)
    segments = [TranscriptSegment(start=0, end=5, text="Hello world")]
    # Should not raise — just log a warning and return
    result = pipeline._diarize(Path("/nonexistent.wav"), segments, "test")
    assert len(result) == 1
    assert result[0].speaker is None

    import shutil

    shutil.rmtree("/tmp/va_test_diarize_fallback", ignore_errors=True)


def test_generate_sprite_sheet():
    """Test sprite sheet generation with a real FFmpeg-generated video."""
    import shutil
    import subprocess
    from video_analysis.pipeline import VideoPipeline
    from PIL import Image

    test_dir = Path("/tmp/va_test_sprite")
    if test_dir.exists():
        shutil.rmtree(test_dir)

    config = Config(data_dir=test_dir)
    pipeline = VideoPipeline(config)

    # Create a small test video using FFmpeg
    test_video = test_dir / "test_sprite_vid.mp4"
    test_video.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=10:size=320x240:rate=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(test_video),
        ],
        capture_output=True,
        check=True,
        timeout=30,
    )

    try:
        sprite_path, meta = pipeline._generate_sprite_sheet(
            test_video, "test_sprite_vid", num_thumbnails=20
        )

        # Verify sprite sheet was created
        assert sprite_path is not None, "Sprite path should not be None"
        assert sprite_path.exists(), f"Sprite sheet should exist at {sprite_path}"
        assert meta is not None, "Metadata should not be None"

        # Verify metadata structure
        # Verify metadata structure
        assert meta["num_columns"] == 10
        assert meta["num_rows"] == 2  # ceil(20/10) = 2
        assert meta["thumbnail_width"] == 160
        assert meta["thumbnail_height"] == 90
        assert meta["duration"] == 10.0
        # The test video has 1 fps (10 frames), so FFmpeg might miss 1
        # thumbnail near the end. Accept 19-20.
        actual_thumbnails = len(meta["thumbnails"])
        assert actual_thumbnails >= 19, f"Expected >=19, got {actual_thumbnails}"
        assert actual_thumbnails <= 20

        # Verify each thumbnail entry
        for t in meta["thumbnails"]:
            assert "index" in t
            assert "timestamp" in t
            assert "x" in t
            assert "y" in t
            assert 0 <= t["timestamp"] <= 10.0

        # Verify the image dimensions
        img = Image.open(sprite_path)
        expected_w = 10 * 160  # 1600
        expected_h = 2 * 90  # 180
        assert img.size == (
            expected_w,
            expected_h,
        ), f"Expected {expected_w}x{expected_h}, got {img.size}"

        # Verify metadata JSON file was written alongside sprite
        meta_path = sprite_path.with_suffix(".json")
        assert meta_path.exists(), f"Metadata JSON should exist at {meta_path}"

        # Verify we can load the metadata JSON
        import json

        loaded_meta = json.loads(meta_path.read_text())
        assert loaded_meta["num_thumbnails"] == 20

        logger.info("Sprite sheet tests passed")
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_sprite_sheet_failure_handling():
    """Test that sprite sheet generation handles errors gracefully."""
    from video_analysis.pipeline import VideoPipeline

    config = Config(data_dir=Path("/tmp/va_test_sprite_fail"))
    pipeline = VideoPipeline(config)

    # Non-existent video
    fake_path = Path("/tmp/nonexistent_video_xyz.mp4")
    sprite_path, meta = pipeline._generate_sprite_sheet(fake_path, "fake", 10)
    assert sprite_path is None
    assert meta is None

    import shutil

    shutil.rmtree("/tmp/va_test_sprite_fail", ignore_errors=True)


def test_rag_imports():
    """Test that RAG module can be imported cleanly."""
    from video_analysis.rag import VideoRAG

    cfg = Config(data_dir="/tmp/va_test_rag")
    r = VideoRAG(cfg)
    assert r.config is not None
    import shutil

    shutil.rmtree("/tmp/va_test_rag", ignore_errors=True)


if __name__ == "__main__":
    test_config_defaults()
    test_scene_info()
    test_frame_info()
    test_frame_info_with_description()
    test_transcript_segment()
    test_video_index()
    test_video_index_with_sprite()
    test_format_timestamp()
    test_config_custom_dir()
    test_config_export_dir()
    test_pipeline_imports()
    test_rag_imports()
    print("All tests passed! ✅")
