"""
PipelineOrchestrator — automatic video type detection and stage selection.

Determines the pipeline stages to run based on file content sniffing:
- File extension (mp4/mkv/mov vs mp3/wav/flac)
- FFprobe content analysis (codec, resolution, has_video, has_audio, duration)
- Heuristic classification without GPU/ML dependencies

Returns a ``PipelineProfile`` with the set of active stages, processing mode,
and recommendations for config overrides.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Set

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class VideoType(str, Enum):
    """Classification of video/audio files based on content sniffing."""

    FULL_VIDEO = "full_video"  # Has video + audio tracks, normal content
    SCREEN_RECORDING = "screen_recording"  # Computer screen capture (static, UI-heavy)
    PODCAST = "podcast"  # Video with talking heads, minimal visual change
    LECTURE = "lecture"  # Slides/presentation video, text-heavy
    MOVIE = "movie"  # Long-form cinematic content
    AUDIO_ONLY = "audio_only"  # No video track (mp3, wav, flac, m4a)
    UNKNOWN = "unknown"  # Could not determine


@dataclass
class PipelineProfile:
    """Recommended pipeline configuration based on video type analysis.

    Attributes:
        video_type: Detected video type.
        processing_mode: 'video_full' or 'audio_only'.
        skipped_stages: Set of stage names to skip.
        confidence: How confident the classifier is (0.0 - 1.0).
        recommended_overrides: Dict of config field -> value for optimal processing.
        analysis_details: Raw ffprobe/ffmpeg analysis output for debugging.
    """

    video_type: VideoType
    processing_mode: str  # "video_full" or "audio_only"
    skipped_stages: Set[str] = field(default_factory=set)
    confidence: float = 0.5
    recommended_overrides: Dict[str, object] = field(default_factory=dict)
    analysis_details: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self):
        """Normalise processing_mode."""
        if self.processing_mode not in ("video_full", "audio_only"):
            logger.warning(
                "Invalid processing_mode %r — defaulting to video_full",
                self.processing_mode,
            )
            self.processing_mode = "video_full"


# ---------------------------------------------------------------------------
# Content probing
# ---------------------------------------------------------------------------

AUDIO_EXTENSIONS: Set[str] = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac", ".wma"}
VIDEO_EXTENSIONS: Set[str] = {
    ".mp4",
    ".mkv",
    ".mov",
    ".avi",
    ".webm",
    ".m4v",
    ".mpg",
    ".mpeg",
    ".wmv",
    ".flv",
    ".ts",
    ".3gp",
}


def probe_file(path: str | Path) -> Dict[str, object]:
    """Run ffprobe on a file and return structured stream metadata.

    Returns an empty dict on failure (so callers can fall through).
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            logger.debug("ffprobe returned %d for %s", result.returncode, path)
            return {}
        data = json.loads(result.stdout)
    except (json.JSONDecodeError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
        logger.debug("ffprobe failed for %s: %s", path, exc)
        return {}

    streams: list[dict] = data.get("streams", [])
    fmt: dict = data.get("format", {})

    has_video = any(s.get("codec_type") == "video" for s in streams)
    has_audio = any(s.get("codec_type") == "audio" for s in streams)

    # Get first video stream info
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
    width = int(video_stream.get("width", 0))
    height = int(video_stream.get("height", 0))
    codec = video_stream.get("codec_name", "")
    fps_parts = video_stream.get("r_frame_rate", "0/1").split("/")
    fps = (
        float(fps_parts[0]) / float(fps_parts[1])
        if len(fps_parts) == 2 and float(fps_parts[1]) > 0
        else 0.0
    )
    is_screen = video_stream.get("codec_tag_string", "").upper() in ("GIF", "GIF2")

    # Duration
    duration = float(fmt.get("duration", 0))
    file_size = int(fmt.get("size", 0))
    bitrate = fmt.get("bit_rate", "")

    return {
        "has_video": has_video,
        "has_audio": has_audio,
        "width": width,
        "height": height,
        "codec": codec,
        "fps": fps,
        "duration": duration,
        "file_size": file_size,
        "bitrate": bitrate,
        "format_name": fmt.get("format_name", ""),
        "is_screen_codec": is_screen,
        "streams": len(streams),
        "video_stream_count": sum(1 for s in streams if s.get("codec_type") == "video"),
    }


# ---------------------------------------------------------------------------
# Heuristic classifiers
# ---------------------------------------------------------------------------


def classify_by_extension(path: str | Path) -> Optional[VideoType]:
    """Classify by file extension alone (fast, no ffprobe needed)."""
    ext = Path(path).suffix.lower()
    if ext in AUDIO_EXTENSIONS:
        return VideoType.AUDIO_ONLY
    if ext in VIDEO_EXTENSIONS:
        return None  # need deeper analysis
    # Unknown extension — might be a video without a standard ext
    return VideoType.UNKNOWN


def classify_video_type(path: str | Path) -> PipelineProfile:
    """Classify a video/audio file and return the recommended pipeline profile.

    Uses a multi-stage heuristic:
    1. File extension check (instant)
    2. FFprobe content analysis (~100ms)
    3. Heuristic classification based on resolution, duration, codec, aspect ratio
    """
    path = Path(path)
    if not path.exists():
        logger.warning("File not found: %s", path)
        return PipelineProfile(
            video_type=VideoType.UNKNOWN,
            processing_mode="video_full",
            skipped_stages=set(),
            confidence=0.0,
            analysis_details={"error": "file_not_found"},
        )

    # --- Phase 1: Extension check ---
    ext_result = classify_by_extension(path)
    if ext_result == VideoType.AUDIO_ONLY:
        return PipelineProfile(
            video_type=VideoType.AUDIO_ONLY,
            processing_mode="audio_only",
            skipped_stages={
                "scene_detection",
                "frame_extraction",
                "quality_screening",
                "object_detection",
                "face_recognition",
                "ocr",
                "clip_classification",
                "video_mllm",
                "action_recognition",
                "sprite_sheet",
                "rag_indexing",
            },
            confidence=1.0,
            analysis_details={"classification_method": "extension"},
        )
    if ext_result == VideoType.UNKNOWN:
        return PipelineProfile(
            video_type=VideoType.UNKNOWN,
            processing_mode="video_full",
            skipped_stages=set(),
            confidence=0.1,
            analysis_details={"classification_method": "extension_unknown"},
        )

    # --- Phase 2: FFprobe analysis ---
    info = probe_file(path)
    if not info:
        # FFprobe failed — fall back to full pipeline
        return PipelineProfile(
            video_type=VideoType.UNKNOWN,
            processing_mode="video_full",
            skipped_stages=set(),
            confidence=0.2,
            analysis_details={"error": "ffprobe_failed"},
        )

    has_video = bool(info.get("has_video", False))
    has_audio = bool(info.get("has_audio", False))
    duration = float(info.get("duration", 0.0))
    width = int(info.get("width", 0))
    height = int(info.get("height", 0))
    fps = float(info.get("fps", 0.0))
    file_size = int(info.get("file_size", 0))
    is_screen_codec = bool(info.get("is_screen_codec", False))

    # --- Phase 3: Heuristic classification ---

    # Audio-only check (no video stream, or zero-resolution video)
    if not has_video or (width == 0 and height == 0):
        return PipelineProfile(
            video_type=VideoType.AUDIO_ONLY,
            processing_mode="audio_only",
            skipped_stages={
                "scene_detection",
                "frame_extraction",
                "quality_screening",
                "object_detection",
                "face_recognition",
                "ocr",
                "clip_classification",
                "video_mllm",
                "action_recognition",
                "sprite_sheet",
                "rag_indexing",
            },
            confidence=0.95,
            analysis_details={**info, "classification_method": "no_video_stream"},
        )

    # Screen recording heuristic: low FPS, low bitrate per pixel, high static area
    # Common for screen recordings: ~10-15 FPS, 1920x1080, small file size relative to duration
    aspect_ratio = width / height if height > 0 else 16 / 9
    is_low_fps = 0 < fps <= 15

    if is_screen_codec:
        video_type = VideoType.SCREEN_RECORDING
        confidence = 0.85
    elif is_low_fps and duration > 60 and file_size > 0:
        # Low FPS + long duration + no high-resolution video = likely screen recording
        mb_per_min = file_size / (duration / 60) / (1024 * 1024) if duration > 0 else 0
        if mb_per_min < 15:  # < 15 MB/min
            video_type = VideoType.SCREEN_RECORDING
            confidence = 0.7
        elif height > 400 and fps < 12:
            video_type = VideoType.LECTURE
            confidence = 0.65
        else:
            video_type = VideoType.UNKNOWN
            confidence = 0.3
    elif duration > 900 and height >= 480:  # >15min cinematic
        video_type = VideoType.MOVIE
        confidence = 0.6
    elif duration > 300 and height >= 480 and fps >= 20:
        # Long, normal FPS — lecture or podcast
        if height > 600:
            video_type = VideoType.LECTURE
            confidence = 0.55
        else:
            video_type = VideoType.PODCAST
            confidence = 0.5
    else:
        video_type = VideoType.FULL_VIDEO
        confidence = 0.5

    # Build profile
    skipped: Set[str] = set()
    overrides: Dict[str, object] = {}
    mode = "video_full"

    if video_type in (VideoType.AUDIO_ONLY,):
        mode = "audio_only"
    elif video_type == VideoType.SCREEN_RECORDING:
        # Screen recordings benefit from OCR (lots of text), but action recognition is useless
        overrides["action_recognition_enabled"] = False
    elif video_type == VideoType.PODCAST:
        # Podcasts have minimal visual changes — reduce frame rate
        overrides["action_recognition_enabled"] = False
    elif video_type == VideoType.LECTURE:
        # Lectures benefit from OCR (slides), no action recognition needed
        overrides["action_recognition_enabled"] = False

    return PipelineProfile(
        video_type=video_type,
        processing_mode=mode,
        skipped_stages=skipped,
        confidence=confidence,
        recommended_overrides=overrides,
        analysis_details={**info, "classification_method": "heuristic"},
    )


def suggest_pipeline(
    path: str | Path, existing_config_overrides: Optional[Dict[str, object]] = None
) -> PipelineProfile:
    """One-call convenience: classify a file and merge any existing config overrides.

    Args:
        path: Path to the video/audio file.
        existing_config_overrides: Existing config overrides (e.g. from user settings).

    Returns:
        PipelineProfile with the final recommended pipeline configuration.
    """
    profile = classify_video_type(path)

    # Merge existing config overrides — user settings take precedence
    if existing_config_overrides:
        for key, value in existing_config_overrides.items():
            profile.recommended_overrides[key] = value

    return profile
