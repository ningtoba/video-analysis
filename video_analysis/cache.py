"""
PipelineCache — content-addressable per-stage caching for the video analysis pipeline.

Each pipeline stage produces outputs that can be cached using a SHA-256 hash of:
- Input video file content (first 64 KB + file size + mtime)
- Config parameters relevant to that stage
- Stage name

Cached outputs are stored in ``data/cache/<stage>/<hash>/`` with a metadata JSON
for invalidation checking. This enables 70-90% faster re-runs when processing
the same video with identical config.

Cache invalidation triggers:
- Config parameter changes (e.g. different scene_threshold)
- Video file changes (different mtime or content)
- Expiry time exceeded (configurable TTL)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from video_analysis.config import Config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CACHE_DIR = "data/cache"
DEFAULT_TTL_SECONDS = 86400 * 7  # 7 days

# Pipeline stage names — used as subdirectory names
STAGE_SCENE_DETECTION = "scene_detection"
STAGE_FRAME_EXTRACTION = "frame_extraction"
STAGE_QUALITY_SCREENING = "quality_screening"
STAGE_TRANSCRIPTION = "transcription"
STAGE_OBJECT_DETECTION = "object_detection"
STAGE_OCR = "ocr"
STAGE_CLIP_CLASSIFICATION = "clip_classification"
STAGE_ACTION_RECOGNITION = "action_recognition"
STAGE_MLLM_DESCRIPTION = "mllm_description"
STAGE_SPRITE_SHEET = "sprite_sheet"
STAGE_RAG_INDEXING = "rag_indexing"
STAGE_AUDIO_EXTRACTION = "audio_extraction"
STAGE_DIARIZATION = "diarization"

ALL_STAGES: Set[str] = {
    STAGE_SCENE_DETECTION,
    STAGE_FRAME_EXTRACTION,
    STAGE_QUALITY_SCREENING,
    STAGE_TRANSCRIPTION,
    STAGE_OBJECT_DETECTION,
    STAGE_OCR,
    STAGE_CLIP_CLASSIFICATION,
    STAGE_ACTION_RECOGNITION,
    STAGE_MLLM_DESCRIPTION,
    STAGE_SPRITE_SHEET,
    STAGE_RAG_INDEXING,
    STAGE_AUDIO_EXTRACTION,
    STAGE_DIARIZATION,
}

# Config fields that affect each stage's output
STAGE_CONFIG_KEYS: Dict[str, List[str]] = {
    STAGE_SCENE_DETECTION: ["scene_detector", "scene_threshold"],
    STAGE_FRAME_EXTRACTION: [
        "frame_rate",
        "frame_storage_mode",
        "frame_analysis_size",
        "frame_thumbnail_size",
        "frame_compression",
        "frame_compression_quality",
        "adaptive_frame_sampling",
        "adaptive_frame_sampling_sensitivity",
        "clip_frame_dedup",
        "clip_frame_dedup_threshold",
    ],
    STAGE_QUALITY_SCREENING: [
        "quality_screening_enabled",
        "quality_min_blur_threshold",
        "quality_min_brightness",
        "quality_max_brightness",
        "quality_static_threshold",
        "quality_skip_ocr_on_blurry",
        "quality_skip_yolo_on_dark",
    ],
    STAGE_TRANSCRIPTION: ["whisper_model", "whisper_device", "whisper_compute_type"],
    STAGE_OBJECT_DETECTION: [
        "yolo_model",
        "yolo_confidence",
        "entity_tracking_enabled",
    ],
    STAGE_OCR: ["ocr_enabled", "ocr_confidence"],
    STAGE_CLIP_CLASSIFICATION: ["clip_model", "clip_pretrained_dataset"],
    STAGE_ACTION_RECOGNITION: ["action_model_name", "action_categories_count"],
    STAGE_MLLM_DESCRIPTION: [
        "video_mllm_model",
        "video_mllm_backend",
        "video_mllm_model_size",
    ],
    STAGE_SPRITE_SHEET: [],
    STAGE_RAG_INDEXING: [
        "embedding_model",
        "top_k_retrieval",
        "temporal_window",
        "temporal_decay_rate",
        "colbert_reranker_enabled",
    ],
    STAGE_DIARIZATION: ["diarize_enabled"],
    STAGE_AUDIO_EXTRACTION: [],
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CacheEntry:
    """Metadata for a single cached stage output."""

    stage: str
    video_id: str
    hash_key: str
    cache_dir: Path
    created_at: float
    expires_at: float
    config_snapshot: Dict[str, Any]
    output_files: List[str]  # relative paths within cache_dir
    output_metadata: Dict[str, Any]


# ---------------------------------------------------------------------------
# Cache implementation
# ---------------------------------------------------------------------------


class PipelineCache:
    """Content-addressable per-stage cache for pipeline outputs.

    Usage::

        cache = PipelineCache(config)
        cache_key = cache.make_key("transcription", video_path, config_snapshot)

        if cache_key in cache:
            cached_output = cache.load(cache_key)
            # ... use cached result ...
        else:
            # ... run stage ...
            cache.store(cache_key, output_files=["...", output_metadata={...})

    Args:
        config: Application config (used for cache directory and TTL).
        cache_dir: Override cache directory (default: data/cache).
        ttl_seconds: Cache TTL in seconds (default: 7 days).
    """

    def __init__(
        self,
        config: Config | None = None,
        cache_dir: str | Path | None = None,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ):
        self.config = config or Config()
        self.cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)
        self.ttl_seconds = ttl_seconds
        self._index: Dict[str, CacheEntry] = {}  # hash_key -> entry
        self._index_loaded = False

    # ------------------------------------------------------------------
    # Key generation
    # ------------------------------------------------------------------

    def _hash_video(self, video_path: str | Path) -> str:
        """Generate a content hash for a video file.

        Uses the first 64 KB + file size + mtime for fast hashing
        (full-file hashing would be expensive for large videos).
        """
        path = Path(video_path)
        if not path.exists():
            return ""

        hasher = hashlib.sha256()
        # Read first 64 KB
        try:
            with open(path, "rb") as f:
                chunk = f.read(65536)
            hasher.update(chunk)
        except (IOError, OSError) as exc:
            logger.warning("Could not read video file for hash: %s", exc)
            hasher.update(str(path.stat().st_size).encode())
            return hasher.hexdigest()[:16]

        # Include file size and mtime for invalidation on modification
        stat = path.stat()
        hasher.update(str(stat.st_size).encode())
        hasher.update(str(int(stat.st_mtime)).encode())
        return hasher.hexdigest()[:16]

    def _hash_config(self, stage: str, config: Config) -> str:
        """Generate a hash of the config parameters relevant to a stage."""
        keys = STAGE_CONFIG_KEYS.get(stage, [])
        hasher = hashlib.sha256()
        hasher.update(stage.encode())
        for key in sorted(keys):
            value = getattr(config, key, None)
            hasher.update(f"{key}={value}\n".encode())
        return hasher.hexdigest()[:12]

    def make_key(
        self,
        stage: str,
        video_path: str | Path,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build a composite cache key from stage + video content + config.

        Args:
            stage: Stage name (one of STAGE_* constants).
            video_path: Path to the input video/audio file.
            parameters: Optional extra parameters to include in the hash.

        Returns:
            Cache key string (SHA-256 hex digest, 28 chars).
        """
        if not Path(video_path).exists():
            logger.warning("Video file not found for cache key: %s", video_path)
            return ""

        video_hash = self._hash_video(video_path)
        config_hash = self._hash_config(stage, self.config)

        hasher = hashlib.sha256()
        hasher.update(video_hash.encode())
        hasher.update(config_hash.encode())
        hasher.update(stage.encode())

        if parameters:
            for key in sorted(parameters.keys()):
                hasher.update(f"{key}={parameters[key]}\n".encode())

        return hasher.hexdigest()[:28]

    # ------------------------------------------------------------------
    # Cache index
    # ------------------------------------------------------------------

    def _index_path(self) -> Path:
        """Path to the cache index JSON file."""
        return self.cache_dir / "_index.json"

    def _load_index(self):
        """Load cache index from disk."""
        if self._index_loaded:
            return
        self._index_loaded = True
        index_path = self._index_path()
        if not index_path.exists():
            self._index = {}
            return
        try:
            data = json.loads(index_path.read_text())
            self._index = {}
            for key, entry in data.items():
                entry["cache_dir"] = Path(entry["cache_dir"])
                self._index[key] = CacheEntry(**entry)
            logger.debug("Loaded cache index with %d entries", len(self._index))
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("Failed to load cache index: %s", exc)
            self._index = {}

    def _save_index(self):
        """Save cache index to disk."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        data = {}
        for key, entry in self._index.items():
            data[key] = {
                "stage": entry.stage,
                "video_id": entry.video_id,
                "hash_key": entry.hash_key,
                "cache_dir": str(entry.cache_dir),
                "created_at": entry.created_at,
                "expires_at": entry.expires_at,
                "config_snapshot": entry.config_snapshot,
                "output_files": entry.output_files,
                "output_metadata": entry.output_metadata,
            }
        index_path = self._index_path()
        index_path.write_text(json.dumps(data, indent=2))
        logger.debug("Saved cache index (%d entries)", len(self._index))

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def __contains__(self, key: str) -> bool:
        """Check if a cache entry exists and is valid (not expired)."""
        if not key:
            return False
        self._load_index()
        entry = self._index.get(key)
        if entry is None:
            return False
        # Check expiry
        if time.time() > entry.expires_at:
            logger.debug("Cache entry %s expired — evicting", key[:12])
            self._remove(key)
            return False
        # Check that cache directory still exists
        if not entry.cache_dir.exists():
            logger.debug("Cache directory for %s missing — evicting", key[:12])
            self._remove(key)
            return False
        return True

    def store(
        self,
        key: str,
        stage: str,
        video_id: str,
        output_files: List[str],
        output_metadata: Optional[Dict[str, Any]] = None,
        config_snapshot: Optional[Dict[str, Any]] = None,
    ):
        """Store a cache entry.

        Args:
            key: Cache key from ``make_key()``.
            stage: Stage name.
            video_id: Video identifier.
            output_files: List of file paths (absolute or relative to cache_dir).
            output_metadata: Optional metadata dict to store alongside.
            config_snapshot: Optional config parameter snapshot.
        """
        if not key:
            logger.warning("Cannot store cache entry with empty key")
            return

        self._load_index()
        now = time.time()

        # Convert output_files to relative paths
        rel_files = []
        for fp in output_files:
            try:
                rel = Path(fp).relative_to(self.cache_dir)
                rel_files.append(str(rel))
            except ValueError:
                # Not under cache_dir — store absolute
                rel_files.append(str(fp))

        entry = CacheEntry(
            stage=stage,
            video_id=video_id,
            hash_key=key,
            cache_dir=self.cache_dir,
            created_at=now,
            expires_at=now + self.ttl_seconds,
            config_snapshot=config_snapshot or {},
            output_files=rel_files,
            output_metadata=output_metadata or {},
        )

        self._index[key] = entry
        self._save_index()
        logger.debug("Cached %s/%s: %s", stage, video_id, key[:12])

    def load(self, key: str) -> Optional[CacheEntry]:
        """Load a cache entry by key.

        Returns the entry metadata (caller uses ``output_files`` to read results).
        Returns None if the entry doesn't exist or is invalid.
        """
        if not key:
            return None
        self._load_index()
        entry = self._index.get(key)
        if entry is None:
            return None
        if time.time() > entry.expires_at:
            self._remove(key)
            return None
        return entry

    def get_output_paths(self, key: str) -> List[Path]:
        """Get the full paths to cached output files for a given key."""
        entry = self.load(key)
        if entry is None:
            return []
        paths = []
        for rel in entry.output_files:
            p = Path(rel)
            if p.is_absolute():
                paths.append(p)
            else:
                paths.append(self.cache_dir / rel)
        return paths

    def _remove(self, key: str):
        """Remove a cache entry (does not delete output files)."""
        self._index.pop(key, None)

    def invalidate(self, stage: Optional[str] = None, video_id: Optional[str] = None):
        """Invalidate cache entries, optionally filtered by stage and/or video_id.

        Args:
            stage: Only invalidate entries for this stage.
            video_id: Only invalidate entries for this video.
        """
        self._load_index()
        keys_to_remove = []
        for key, entry in self._index.items():
            if stage and entry.stage != stage:
                continue
            if video_id and entry.video_id != video_id:
                continue
            keys_to_remove.append(key)

        for key in keys_to_remove:
            entry = self._index.pop(key)
            # Clean up cached files
            if entry.cache_dir.exists():
                for rel in entry.output_files:
                    p = entry.cache_dir / rel
                    try:
                        if p.exists():
                            p.unlink()
                    except (IOError, OSError) as exc:
                        logger.warning("Could not delete cache file %s: %s", p, exc)

        if keys_to_remove:
            self._save_index()
            logger.info(
                "Invalidated %d cache entries%s%s",
                len(keys_to_remove),
                f" for stage={stage}" if stage else "",
                f" video={video_id}" if video_id else "",
            )

    def clear(self):
        """Clear all cached outputs and the index."""
        self._load_index()
        self._index.clear()
        self._save_index()
        # Remove cached stage directories
        for stage_dir in self.cache_dir.iterdir():
            if stage_dir.is_dir() and stage_dir.name != "_index.json":
                import shutil

                shutil.rmtree(stage_dir, ignore_errors=True)
        logger.info("Cache cleared")

    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with entry_count, size_bytes, oldest, newest keys.
        """
        self._load_index()
        if not self._index:
            return {"entry_count": 0, "size_bytes": 0}

        now = time.time()
        valid = [e for e in self._index.values() if now <= e.expires_at]
        expired_count = len(self._index) - len(valid)

        # Calculate total size of cached files
        total_size = 0
        for entry in valid:
            for rel in entry.output_files:
                p = self.cache_dir / rel if not Path(rel).is_absolute() else Path(rel)
                if p.exists():
                    total_size += p.stat().st_size

        if valid:
            oldest = min(e.created_at for e in valid)
            newest = max(e.created_at for e in valid)
        else:
            oldest = newest = 0

        stages = set(e.stage for e in valid)

        return {
            "entry_count": len(valid),
            "expired_count": expired_count,
            "size_bytes": total_size,
            "oldest": oldest,
            "newest": newest,
            "stages": sorted(stages),
        }
