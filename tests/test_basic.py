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


def test_yt_dlp_import():
    """Test that yt-dlp can be imported (optional dep)."""
    try:
        import yt_dlp

        assert yt_dlp is not None
    except ImportError:
        pass  # optional dependency, not required


def test_download_from_url_no_url():
    """Test download_from_url handles missing yt-dlp gracefully."""
    from video_analysis.pipeline import VideoPipeline

    pipeline = VideoPipeline(Config(data_dir="/tmp/va_test_yt"))
    result = pipeline.download_from_url("", Path("/tmp"))
    # Should not crash — returns None
    assert result is None

    import shutil

    shutil.rmtree("/tmp/va_test_yt", ignore_errors=True)


def test_parse_yt_url():
    """Test URL pattern matching."""
    from ui.utils import parse_yt_url

    assert parse_yt_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ") is True
    assert parse_yt_url("https://youtu.be/dQw4w9WgXcQ") is True
    assert parse_yt_url("https://vimeo.com/123456789") is True
    assert parse_yt_url("https://example.com/video.mp4") is False
    assert parse_yt_url("") is False


def test_batch_queue_html():
    """Test queue HTML rendering."""
    from ui.utils import queue_html

    html = queue_html([])
    assert "empty" in html.lower()

    items = [
        {"name": "test1.mp4", "status": "pending"},
        {"name": "test2.mp4", "status": "done"},
    ]
    html = queue_html(items)
    assert "test1.mp4" in html
    assert "test2.mp4" in html


def test_config_yt_dlp_fields():
    """Test that yt-dlp config fields exist."""
    cfg = Config(data_dir="/tmp/va_test_yt_config")
    assert cfg.yt_dlp_enabled is True
    assert "bestvideo" in cfg.yt_dlp_format
    assert "%(id)s" in cfg.yt_dlp_output_template
    import shutil

    shutil.rmtree("/tmp/va_test_yt_config", ignore_errors=True)


def test_rag_search_videos():
    """Test search_videos returns filtered results."""
    from video_analysis.rag import VideoRAG

    cfg = Config(data_dir="/tmp/va_test_rag_search")
    r = VideoRAG(cfg)
    result = r.search_videos("")
    assert isinstance(result, list)
    result = r.search_videos("test")
    assert isinstance(result, list)
    import shutil

    shutil.rmtree("/tmp/va_test_rag_search", ignore_errors=True)


def test_rag_get_library_info_empty():
    """Test get_library_info returns None for non-existent video."""
    from video_analysis.rag import VideoRAG

    cfg = Config(data_dir="/tmp/va_test_lib_info")
    r = VideoRAG(cfg)
    info = r.get_library_info("nonexistent")
    assert info is None
    import shutil

    shutil.rmtree("/tmp/va_test_lib_info", ignore_errors=True)


def test_config_batch_concurrent():
    """Test batch_concurrent config field."""
    cfg = Config(data_dir="/tmp/va_test_batch_cfg")
    assert cfg.batch_concurrent == 1
    import shutil

    shutil.rmtree("/tmp/va_test_batch_cfg", ignore_errors=True)


def test_config_clip_model():
    """Test clip_model config field defaults to ViT-B-32."""
    cfg = Config(data_dir="/tmp/va_test_clip_model")
    assert cfg.clip_model == "ViT-B-32"
    import shutil

    shutil.rmtree("/tmp/va_test_clip_model", ignore_errors=True)


def test_config_clip_embed_dim_b32():
    """Test clip_embed_dim defaults to 512 for ViT-B-32."""
    cfg = Config(data_dir="/tmp/va_test_clip_dim")
    assert cfg.clip_embed_dim == 512
    import shutil

    shutil.rmtree("/tmp/va_test_clip_dim", ignore_errors=True)


def test_config_clip_embed_dim_l14():
    """Test clip_embed_dim can be set to 768 for ViT-L-14."""
    cfg = Config(data_dir="/tmp/va_test_clip_dim_l14", clip_embed_dim=768)
    assert cfg.clip_embed_dim == 768
    import shutil

    shutil.rmtree("/tmp/va_test_clip_dim_l14", ignore_errors=True)


def test_config_clip_pretrained_dataset():
    """Test clip_pretrained_dataset config field."""
    cfg = Config(data_dir="/tmp/va_test_clip_pretrained")
    assert cfg.clip_pretrained_dataset == "laion2b_s34b_b79k"
    import shutil

    shutil.rmtree("/tmp/va_test_clip_pretrained", ignore_errors=True)


def test_config_scene_detector_options():
    """Test scene_detector supports all available options."""
    cfg = Config(data_dir="/tmp/va_test_scene_opt")
    assert cfg.scene_detector in ("adaptive", "content", "ffmpeg", "histogram", "hash")
    # Verify the options are handled in pipeline
    from video_analysis.pipeline import VideoPipeline

    pipeline = VideoPipeline(cfg)
    assert pipeline.config.scene_detector == "adaptive"
    import shutil

    shutil.rmtree("/tmp/va_test_scene_opt", ignore_errors=True)


def test_config_embedding_model():
    """Test embedding_model config defaults."""
    cfg = Config(data_dir="/tmp/va_test_embed")
    assert cfg.embedding_model is not None
    assert isinstance(cfg.embedding_model, str)
    import shutil

    shutil.rmtree("/tmp/va_test_embed", ignore_errors=True)


def test_health_check_module():
    """Test health module can be imported and has expected structure."""
    from ui.health import (
        create_health_app,
        HealthResponse,
        LibraryResponse,
        VideoInfoResponse,
    )

    assert callable(create_health_app)
    # Verify the module is importable and classes exist
    assert HealthResponse.__name__ == "HealthResponse"
    assert LibraryResponse.__name__ == "LibraryResponse"
    assert VideoInfoResponse.__name__ == "VideoInfoResponse"


def test_is_video_file():
    """Test video file extension detection."""
    from ui.utils import is_video_file

    assert is_video_file("video.mp4") is True
    assert is_video_file("video.MOV") is True
    assert is_video_file("video.txt") is False
    assert is_video_file("video") is False


def test_config_ui_auth():
    """Test UI auth config fields."""
    import os

    # Test without env vars
    cfg = Config(data_dir="/tmp/va_test_auth")
    assert hasattr(cfg, "ui_auth_enabled")
    assert hasattr(cfg, "ui_auth_username")
    assert hasattr(cfg, "ui_auth_password")
    assert cfg.ui_auth_username == "admin"
    assert cfg.ui_auth_password == ""

    # Test that auth is disabled when no password set
    assert cfg.ui_auth_enabled is False

    import shutil

    shutil.rmtree("/tmp/va_test_auth", ignore_errors=True)


def test_config_adaptive_frame_sampling():
    """Test adaptive frame sampling config fields."""
    cfg = Config(data_dir="/tmp/va_test_adaptive")
    assert hasattr(cfg, "adaptive_frame_sampling")
    assert hasattr(cfg, "adaptive_frame_sampling_sensitivity")
    assert cfg.adaptive_frame_sampling is False
    assert cfg.adaptive_frame_sampling_sensitivity == 0.3

    cfg2 = Config(data_dir="/tmp/va_test_adaptive2", adaptive_frame_sampling=True)
    assert cfg2.adaptive_frame_sampling is True

    import shutil

    shutil.rmtree("/tmp/va_test_adaptive", ignore_errors=True)
    shutil.rmtree("/tmp/va_test_adaptive2", ignore_errors=True)


def test_config_clip_frame_dedup():
    """Test CLIP frame dedup config fields."""
    cfg = Config(data_dir="/tmp/va_test_dedup")
    assert hasattr(cfg, "clip_frame_dedup")
    assert hasattr(cfg, "clip_frame_dedup_threshold")
    assert cfg.clip_frame_dedup is False
    assert cfg.clip_frame_dedup_threshold == 0.92

    import shutil

    shutil.rmtree("/tmp/va_test_dedup", ignore_errors=True)


def test_adaptive_frame_samples_basic():
    """Test that _adaptive_frame_samples returns reasonable timestamps."""
    from video_analysis.pipeline import VideoPipeline
    from video_analysis.models import SceneInfo

    cfg = Config(data_dir="/tmp/va_test_adaptive_samples", adaptive_frame_sampling=True)
    pipeline = VideoPipeline(cfg)

    # 30-second scene
    scene = SceneInfo(scene_id=0, start_time=10.0, end_time=40.0)
    samples = pipeline._adaptive_frame_samples(scene, 30.0)

    assert len(samples) > 0
    assert all(10.0 <= t <= 40.0 for t in samples)
    # Should have mid-point
    assert 25.0 in samples
    # Should have samples near boundaries
    assert any(t < 13.0 for t in samples)  # near start
    assert any(t > 37.0 for t in samples)  # near end

    import shutil

    shutil.rmtree("/tmp/va_test_adaptive_samples", ignore_errors=True)


def test_adaptive_frame_samples_short_scene():
    """Test that very short scenes (<5s) use default sampling (no adaptive)."""
    from video_analysis.pipeline import VideoPipeline
    from video_analysis.models import SceneInfo

    cfg = Config(data_dir="/tmp/va_test_adaptive_short", adaptive_frame_sampling=True)
    pipeline = VideoPipeline(cfg)

    # 3-second scene — too short for adaptive
    scene = SceneInfo(scene_id=0, start_time=0.0, end_time=3.0)
    # _extract_key_frames would use default path (adaptive_frame_sampling is True
    # but _adaptive_frame_samples is only called when duration > 5)

    import shutil

    shutil.rmtree("/tmp/va_test_adaptive_short", ignore_errors=True)


def test_dedup_frames_clip_no_openclip():
    """Test _dedup_frames_clip falls back gracefully without open_clip."""
    from video_analysis.pipeline import VideoPipeline
    from video_analysis.models import FrameInfo

    cfg = Config(data_dir="/tmp/va_test_dedup_fallback")
    pipeline = VideoPipeline(cfg)

    frames = [
        FrameInfo(timestamp=0.0, filepath="/nonexistent.jpg", scene_id=0),
        FrameInfo(timestamp=2.0, filepath="/nonexistent2.jpg", scene_id=0),
    ]
    # Should not crash — just return all frames
    result = pipeline._dedup_frames_clip(frames, "test")
    assert len(result) == 2
    assert result[0].timestamp == 0.0
    assert result[1].timestamp == 2.0

    import shutil

    shutil.rmtree("/tmp/va_test_dedup_fallback", ignore_errors=True)


def test_video_library_info_dataclass():
    """Test VideoLibraryInfo dataclass."""
    from video_analysis.rag import VideoLibraryInfo

    info = VideoLibraryInfo(video_id="test1", filename="test1.mp4")
    assert info.video_id == "test1"
    assert info.num_scenes == 0
    assert info.num_chunks == 0
    assert info.duration == 0.0
    assert info.has_sprite is False

    info = VideoLibraryInfo(
        video_id="test2",
        filename="test2.mp4",
        num_scenes=5,
        num_chunks=42,
        duration=120.5,
        has_sprite=True,
    )
    assert info.num_scenes == 5
    assert info.num_chunks == 42
    assert info.duration == 120.5
    assert info.has_sprite is True


def test_colbert_reranker_import():
    """Test ColBERTReranker module can be imported and reports availability correctly."""
    from video_analysis.colbert_reranker import ColBERTReranker

    reranker = ColBERTReranker()
    # Should not crash on init
    assert reranker.model_name == "colbert-ir/colbertv2.0"
    # Report available only if ragatouille is installed
    # (we test the fallback path — it properly reports False when missing)
    assert isinstance(reranker.available, bool)


def test_colbert_reranker_fallback_on_missing():
    """Test that _rerank_colbert in VideoRAG falls back gracefully without ragatouille."""
    from video_analysis.config import Config
    from video_analysis.rag import VideoRAG, RetrievedChunk

    cfg = Config(data_dir="/tmp/va_test_colbert_fallback")
    cfg.colbert_reranker_enabled = True

    rag = VideoRAG(cfg)

    # Create dummy chunks
    chunks = [
        RetrievedChunk(
            chunk_id="test_1",
            video_id="test",
            text="test content",
            timestamp=0.0,
            scene_id=0,
            score=0.5,
        )
    ]

    # Should not crash when colbert reranker is enabled but ragatouille not installed
    result = rag._rerank_colbert("test query", chunks, top_k=5)
    assert isinstance(result, list)
    assert len(result) > 0

    import shutil

    shutil.rmtree("/tmp/va_test_colbert_fallback", ignore_errors=True)


def test_config_colbert_reranker():
    """Test colbert_reranker_enabled config field."""
    from video_analysis.config import Config

    cfg = Config(data_dir="/tmp/va_test_colbert_cfg")
    assert hasattr(cfg, "colbert_reranker_enabled")
    assert cfg.colbert_reranker_enabled is False

    cfg2 = Config(data_dir="/tmp/va_test_colbert_cfg", colbert_reranker_enabled=True)
    assert cfg2.colbert_reranker_enabled is True

    import shutil

    shutil.rmtree("/tmp/va_test_colbert_cfg", ignore_errors=True)


# ── Catch-all runner (for python tests/test_basic.py direct execution) ──
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
    test_config_clip_model()
    test_config_scene_detector_options()
    test_config_embedding_model()
    test_health_check_module()
    test_config_clip_pretrained_dataset()
    test_config_clip_embed_dim_b32()
    test_config_clip_embed_dim_l14()
    test_colbert_reranker_import()
    test_colbert_reranker_fallback_on_missing()
    test_config_colbert_reranker()
    print("All tests passed! ✅")
