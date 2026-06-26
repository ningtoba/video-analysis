"""Tests for the PipelineCache module."""

import json
import time
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from video_analysis.cache import (
    PipelineCache,
    CacheEntry,
    ALL_STAGES,
    STAGE_CONFIG_KEYS,
    STAGE_TRANSCRIPTION,
    STAGE_SCENE_DETECTION,
    STAGE_CLIP_CLASSIFICATION,
    DEFAULT_TTL_SECONDS,
)


def test_cache_init_defaults():
    """Test PipelineCache default initialization."""
    cache = PipelineCache()
    assert cache.ttl_seconds == DEFAULT_TTL_SECONDS
    assert cache.cache_dir.name == "cache"
    assert cache._index == {}
    assert cache._index_loaded is False


def test_cache_init_custom():
    """Test PipelineCache custom initialization."""
    cache = PipelineCache(cache_dir="/tmp/va_test_cache_custom", ttl_seconds=3600)
    assert str(cache.cache_dir) == "/tmp/va_test_cache_custom"
    assert cache.ttl_seconds == 3600


def test_make_key_empty_video():
    """Test make_key returns empty string for non-existent video."""
    cache = PipelineCache()
    key = cache.make_key(STAGE_TRANSCRIPTION, "/tmp/nonexistent.mp4")
    assert key == ""


def test_make_key_consistent():
    """Test make_key returns the same key for the same input."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test_vid.mp4"
        test_file.write_text("dummy video content for hash test")

        cache = PipelineCache()
        key1 = cache.make_key(STAGE_TRANSCRIPTION, test_file)
        key2 = cache.make_key(STAGE_TRANSCRIPTION, test_file)

        assert key1 == key2
        assert len(key1) == 28


def test_make_key_different_video():
    """Test make_key returns different keys for different videos."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file1 = Path(tmpdir) / "test_vid1.mp4"
        test_file1.write_text("video content A")

        test_file2 = Path(tmpdir) / "test_vid2.mp4"
        test_file2.write_text("video content B")

        cache = PipelineCache()
        key1 = cache.make_key(STAGE_TRANSCRIPTION, test_file1)
        key2 = cache.make_key(STAGE_TRANSCRIPTION, test_file2)

        assert key1 != key2


def test_make_key_different_stage():
    """Test make_key returns different keys for different stages."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test_vid.mp4"
        test_file.write_text("same video")

        cache = PipelineCache()
        key1 = cache.make_key(STAGE_TRANSCRIPTION, test_file)
        key2 = cache.make_key(STAGE_SCENE_DETECTION, test_file)

        assert key1 != key2


def test_store_and_load():
    """Test store and load cycle."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cache"
        cache = PipelineCache(cache_dir=cache_dir)

        test_file = Path(tmpdir) / "test_vid.mp4"
        test_file.write_text("test video")

        key = cache.make_key(STAGE_TRANSCRIPTION, test_file)
        assert key != ""

        # Store
        cache.store(
            key=key,
            stage=STAGE_TRANSCRIPTION,
            video_id="test_video",
            output_files=["/tmp/cached_result.json"],
            output_metadata={"segment_count": 10},
        )

        # Contains
        assert key in cache

        # Load
        entry = cache.load(key)
        assert entry is not None
        assert entry.stage == STAGE_TRANSCRIPTION
        assert entry.video_id == "test_video"
        assert entry.output_metadata.get("segment_count") == 10


def test_contains_expired():
    """Test that __contains__ returns False for expired entries."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cache"
        cache = PipelineCache(cache_dir=cache_dir, ttl_seconds=0)  # expired immediately

        test_file = Path(tmpdir) / "test_vid.mp4"
        test_file.write_text("expired test")
        key = cache.make_key(STAGE_TRANSCRIPTION, test_file)

        cache.store(
            key=key, stage=STAGE_TRANSCRIPTION, video_id="test", output_files=[]
        )

        # Should be expired (ttl_seconds=0)
        assert key not in cache


def test_load_nonexistent_key():
    """Test load returns None for non-existent key."""
    cache = PipelineCache()
    result = cache.load("nonexistent_key")
    assert result is None


def test_load_expired():
    """Test load returns None for expired entries."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cache"
        cache = PipelineCache(cache_dir=cache_dir, ttl_seconds=0)

        test_file = Path(tmpdir) / "test_vid.mp4"
        test_file.write_text("load expired test")
        key = cache.make_key(STAGE_TRANSCRIPTION, test_file)

        cache.store(
            key=key, stage=STAGE_TRANSCRIPTION, video_id="test", output_files=[]
        )
        result = cache.load(key)
        assert result is None


def test_get_output_paths():
    """Test get_output_paths returns correct paths."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cache"
        cache = PipelineCache(cache_dir=cache_dir)

        test_file = Path(tmpdir) / "test_vid.mp4"
        test_file.write_text("test output paths")

        # Create a cached file
        cached_file = cache_dir / "transcription" / "result.json"
        cached_file.parent.mkdir(parents=True, exist_ok=True)
        cached_file.write_text('{"result": "ok"}')

        key = cache.make_key(STAGE_TRANSCRIPTION, test_file)
        cache.store(
            key=key,
            stage=STAGE_TRANSCRIPTION,
            video_id="test",
            output_files=[str(cached_file)],
        )

        paths = cache.get_output_paths(key)
        assert len(paths) == 1
        assert paths[0] == cached_file


def test_get_output_paths_nonexistent():
    """Test get_output_paths returns empty list for unknown key."""
    cache = PipelineCache()
    paths = cache.get_output_paths("nonexistent")
    assert paths == []


def test_invalidate_by_stage():
    """Test invalidation by stage name."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cache"
        cache = PipelineCache(cache_dir=cache_dir, ttl_seconds=3600)

        test_file = Path(tmpdir) / "test_vid.mp4"
        test_file.write_text("invalidate test")

        key_trans = cache.make_key(STAGE_TRANSCRIPTION, test_file)
        key_scene = cache.make_key(STAGE_SCENE_DETECTION, test_file)

        cache.store(
            key=key_trans, stage=STAGE_TRANSCRIPTION, video_id="test1", output_files=[]
        )
        cache.store(
            key=key_scene,
            stage=STAGE_SCENE_DETECTION,
            video_id="test1",
            output_files=[],
        )

        assert key_trans in cache
        assert key_scene in cache

        cache.invalidate(stage=STAGE_TRANSCRIPTION)

        assert key_trans not in cache
        assert key_scene in cache


def test_invalidate_by_video():
    """Test invalidation by video_id."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cache"
        cache = PipelineCache(cache_dir=cache_dir, ttl_seconds=3600)

        test_file = Path(tmpdir) / "test_vid.mp4"
        test_file.write_text("invalidate video")

        key = cache.make_key(STAGE_TRANSCRIPTION, test_file)

        cache.store(
            key=key, stage=STAGE_TRANSCRIPTION, video_id="test1", output_files=[]
        )
        assert key in cache

        cache.invalidate(video_id="test1")
        assert key not in cache


def test_clear():
    """Test clear removes all cache entries."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cache"
        cache = PipelineCache(cache_dir=cache_dir, ttl_seconds=3600)

        test_file = Path(tmpdir) / "test_vid.mp4"
        test_file.write_text("clear test")

        key = cache.make_key(STAGE_TRANSCRIPTION, test_file)
        cache.store(
            key=key, stage=STAGE_TRANSCRIPTION, video_id="test", output_files=[]
        )
        assert key in cache

        cache.clear()
        assert key not in cache


def test_stats_empty():
    """Test stats returns zeros for empty cache."""
    cache = PipelineCache()
    stats = cache.stats
    assert stats["entry_count"] == 0
    assert stats["size_bytes"] == 0


def test_stats_with_entries():
    """Test stats returns correct counts with entries."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cache"
        cache = PipelineCache(cache_dir=cache_dir, ttl_seconds=3600)

        test_file = Path(tmpdir) / "test_vid.mp4"
        test_file.write_text("stats test")

        key = cache.make_key(STAGE_TRANSCRIPTION, test_file)
        cache.store(
            key=key, stage=STAGE_TRANSCRIPTION, video_id="test", output_files=[]
        )
        assert key in cache

        stats = cache.stats
        assert stats["entry_count"] >= 1
        assert "transcription" in stats["stages"]


def test_index_persistence():
    """Test that cache index persists across PipelineCache instances."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cache"
        test_file = Path(tmpdir) / "test_vid.mp4"
        test_file.write_text("persistence test")

        # First instance
        cache1 = PipelineCache(cache_dir=cache_dir, ttl_seconds=3600)
        key = cache1.make_key(STAGE_TRANSCRIPTION, test_file)
        cache1.store(
            key=key, stage=STAGE_TRANSCRIPTION, video_id="test", output_files=[]
        )

        # Second instance (same cache_dir)
        cache2 = PipelineCache(cache_dir=cache_dir, ttl_seconds=3600)
        assert key in cache2


def test_config_key_relevance():
    """Test that STAGE_CONFIG_KEYS covers all expected stages."""
    for stage in ALL_STAGES:
        assert stage in STAGE_CONFIG_KEYS, f"Missing config keys for stage: {stage}"


def test_cache_entry_dataclass():
    """Test CacheEntry dataclass fields."""
    entry = CacheEntry(
        stage=STAGE_TRANSCRIPTION,
        video_id="test",
        hash_key="abc123",
        cache_dir=Path("/tmp/cache"),
        created_at=1000.0,
        expires_at=2000.0,
        config_snapshot={"whisper_model": "large-v3"},
        output_files=["result.json"],
        output_metadata={"segments": 5},
    )
    assert entry.stage == STAGE_TRANSCRIPTION
    assert entry.hash_key == "abc123"
    assert entry.output_metadata["segments"] == 5


def test_all_stages_set():
    """Test ALL_STAGES contains expected stages."""
    assert "transcription" in ALL_STAGES
    assert "scene_detection" in ALL_STAGES
    assert "frame_extraction" in ALL_STAGES
    assert "object_detection" in ALL_STAGES
    assert "ocr" in ALL_STAGES
    assert "clip_classification" in ALL_STAGES
    assert "rag_indexing" in ALL_STAGES
    assert "sprite_sheet" in ALL_STAGES
    assert len(ALL_STAGES) >= 12
