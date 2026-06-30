"""
Adaptive Pipeline Stage Scaler — intelligent pipeline quality/resource management.

Adjusts per-stage quality and resource usage based on video properties
(duration, resolution, FPS, bitrate) and system state (GPU free memory,
CPU cores) to optimise the throughput-quality trade-off.

Scaling policies:
  - ``conservative``: maximise VRAM headroom, reduce frame rate, lower res
  - ``balanced`` (default): sensible defaults tuned for 12 GB VRAM
  - ``performance``: maximise throughput, higher resolution, more frames
  - ``auto``: video-aware: long videos use conservative, short use performance
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Video-aware policy thresholds
_VRAM_CRITICAL_RATIO: float = 0.95
_VRAM_WARNING_RATIO: float = 0.85
_VRAM_STAGE_PRUNE_RATIO: float = 0.9
_LONG_VIDEO_DURATION_S: float = 1800.0
_SHORT_VIDEO_DURATION_S: float = 60.0
_SHORT_VIDEO_HIGH_RES_MP: float = 2.0

# Defaults
_DEFAULT_SCENE_THRESHOLD: float = 0.3
_LONG_VIDEO_DINO_THRESHOLD: float = 0.95
_LONG_VIDEO_JPEG_QUALITY: int = 75
_SHORT_VIDEO_MAX_FRAME_SIZE: int = 1280

# Probe
_FFPROBE_TIMEOUT_S: int = 30

# ── Scaling profiles ──────────────────────────────────────────────────────────

# Type alias for policy names
ScalingPolicy = str  # "conservative" | "balanced" | "performance" | "auto"

# Frame rate (fps) tiers: how many frames per second to extract
FRAME_RATE_TIERS: Dict[ScalingPolicy, float] = {
    "conservative": 0.1,  # 1 frame per 10 seconds
    "balanced": 0.5,  # 1 frame per 2 seconds (current default)
    "performance": 2.0,  # 2 frames per second
}

# Analysis frame size (longest edge in pixels)
FRAME_SIZE_TIERS: Dict[ScalingPolicy, int] = {
    "conservative": 640,
    "balanced": 960,  # current default
    "performance": 1280,
}

# Thumbnail size
THUMBNAIL_SIZE_TIERS: Dict[ScalingPolicy, int] = {
    "conservative": 160,
    "balanced": 320,
    "performance": 480,
}

# YOLO confidence threshold (lower = more detections, more VRAM)
YOLO_CONFIDENCE_TIERS: Dict[ScalingPolicy, float] = {
    "conservative": 0.4,
    "balanced": 0.25,  # current default
    "performance": 0.15,
}

# OCR model tier
OCR_TIER_TIERS: Dict[ScalingPolicy, str] = {
    "conservative": "tiny",
    "balanced": "medium",  # current default
    "performance": "medium",
}

# DINOv2 compression threshold (lower = more aggressive dedup)
DINO_THRESHOLD_TIERS: Dict[ScalingPolicy, float] = {
    "conservative": 0.92,  # Keep fewer frames
    "balanced": 0.88,  # current default
    "performance": 0.80,  # Keep more frames
}

# Quality screening skip: when conservative, be more lenient about skipping
QUALITY_BLUR_THRESHOLD_TIERS: Dict[ScalingPolicy, float] = {
    "conservative": 50.0,  # Much more tolerant
    "balanced": 100.0,  # current default
    "performance": 150.0,  # Stricter — only skip truly blurry
}

# Estimated VRAM per stage in GB (approximate, RTX 4070 12GB profile)
STAGE_VRAM_ESTIMATES: Dict[str, float] = {
    "whisper": 3.5,
    "yolo": 1.0,
    "clip": 1.5,
    "xclip": 3.5,
    "video_mllm": 5.4,
    "face_recognition": 1.1,
    "dino": 0.085,
    "ocr": 0.0,  # CPU-only
    "diarization": 0.0,  # CPU-only
}


@dataclass
class ScalingResult:
    """Result of adaptive scaling — overrides to apply to the pipeline config."""

    # Frame extraction
    frame_rate: Optional[float] = None
    scene_threshold: Optional[float] = None

    # Frame storage
    frame_analysis_size: Optional[int] = None
    frame_thumbnail_size: Optional[int] = None
    frame_storage_mode: Optional[str] = None
    frame_compression_quality: Optional[int] = None

    # YOLO
    yolo_confidence: Optional[float] = None

    # OCR
    ocr_model_tier: Optional[str] = None

    # DINOv2 compression
    dino_frame_compression_threshold: Optional[float] = None

    # Quality screening
    quality_min_blur_threshold: Optional[float] = None

    # Action recognition
    action_recognition_enabled: Optional[bool] = None

    # Face recognition
    face_recognition_enabled: Optional[bool] = None

    # MLLM
    video_mllm_as_describer: Optional[bool] = None

    # Metadata
    policy_used: str = "balanced"
    vram_gb_available: float = 0.0
    estimated_vram_gb: float = 0.0
    video_duration: float = 0.0
    video_width: int = 0
    video_height: int = 0
    video_fps: float = 0.0
    total_frames_estimate: int = 0

    # Human-readable explanation
    reasoning: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        """Convert to config-override dict (omits None values)."""
        d: Dict[str, object] = {}
        for k in (
            "frame_rate",
            "scene_threshold",
            "frame_analysis_size",
            "frame_thumbnail_size",
            "frame_storage_mode",
            "frame_compression_quality",
            "yolo_confidence",
            "ocr_model_tier",
            "dino_frame_compression_threshold",
            "quality_min_blur_threshold",
            "action_recognition_enabled",
            "face_recognition_enabled",
            "video_mllm_as_describer",
        ):
            v = getattr(self, k, None)
            if v is not None:
                d[k] = v
        return d


def get_video_properties(video_path: Path) -> Dict[str, float]:
    """Extract video properties using ffprobe.

    Returns dict with keys: duration, width, height, fps, bitrate_kbps.
    All values are 0.0 on failure.
    """
    props: Dict[str, float] = {
        "duration": 0.0,
        "width": 0.0,
        "height": 0.0,
        "fps": 0.0,
        "bitrate_kbps": 0.0,
    }

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,duration,avg_frame_rate,bit_rate",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=_FFPROBE_TIMEOUT_S,
        )
        if result.returncode != 0:
            logger.warning("ffprobe failed: %s", result.stderr.strip())
            return props

        lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
        if len(lines) >= 5:
            props["width"] = float(lines[0]) if lines[0] != "N/A" else 0.0
            props["height"] = float(lines[1]) if lines[1] != "N/A" else 0.0
            # Parse frame rate as rational (lines[2] = avg_frame_rate)
            if lines[2] != "N/A" and "/" in lines[2]:
                num, den = lines[2].split("/")
                try:
                    n, d = float(num), float(den)
                    props["fps"] = n / d if d > 0 else 0.0
                except (ValueError, ZeroDivisionError):
                    props["fps"] = 0.0
            elif lines[2] != "N/A":
                props["fps"] = float(lines[2])
            # Duration (lines[3])
            if lines[3] != "N/A":
                props["duration"] = float(lines[3])
            # Bitrate (lines[4])
            if lines[4] != "N/A":
                bitrate_bps = float(lines[4])
                props["bitrate_kbps"] = bitrate_bps / 1000.0
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError) as e:
        logger.warning("Failed to probe video %s: %s", video_path.name, e)
    except Exception as e:
        logger.warning("Unexpected ffprobe error for %s: %s", video_path.name, e)

    return props


def get_free_vram_gb() -> float:
    """Return approximate free GPU VRAM in GB, or 0.0 if unavailable."""
    try:
        import pynvml

        try:
            pynvml.nvmlInit()
        except pynvml.NVMLError:
            return 0.0
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.free / (1024**3)
    except ImportError:
        logger.debug("pynvml not available — cannot detect free VRAM")
        return 0.0
    except Exception as e:
        logger.debug("VRAM detection failed: %s", e)
        return 0.0


def estimate_vram_usage(
    enabled_stages: Dict[str, bool],
    policy: str,
) -> float:
    """Estimate total GPU VRAM needed for the given stages at a given policy.

    Conservative uses smaller models / lower precision.
    Balanced uses standard models.
    Performance uses larger models or higher precision.
    """
    total = 0.0
    # Base whisper varies by model size
    whisper_factor = {"conservative": 0.7, "balanced": 1.0, "performance": 1.3}
    wf = whisper_factor.get(policy, 1.0)
    if enabled_stages.get("whisper", True):
        total += STAGE_VRAM_ESTIMATES["whisper"] * wf

    # YOLO model size scales with resolution
    yolo_factor = {"conservative": 0.6, "balanced": 1.0, "performance": 1.5}
    if enabled_stages.get("yolo", True):
        total += STAGE_VRAM_ESTIMATES["yolo"] * yolo_factor.get(policy, 1.0)

    # CLIP
    clip_factor = {"conservative": 0.5, "balanced": 1.0, "performance": 2.0}
    if enabled_stages.get("clip", True):
        total += STAGE_VRAM_ESTIMATES["clip"] * clip_factor.get(policy, 1.0)

    if enabled_stages.get("xclip", False):
        total += STAGE_VRAM_ESTIMATES["xclip"]
    if enabled_stages.get("video_mllm", False):
        total += STAGE_VRAM_ESTIMATES["video_mllm"]
    if enabled_stages.get("face_recognition", False):
        total += STAGE_VRAM_ESTIMATES["face_recognition"]
    if enabled_stages.get("dino", False):
        total += STAGE_VRAM_ESTIMATES["dino"]

    return total


def select_video_aware_policy(
    duration: float,
    resolution_megapixels: float,
    duration_threshold_long: float = 600.0,  # 10 minutes
    duration_threshold_short: float = 60.0,  # 1 minute
    resolution_threshold_high: float = 2.0,  # 1920×1080 ≈ 2.07 MP
) -> str:
    """Select scaling policy based on video properties.

    Long videos (>=10 min) → conservative (keep VRAM buffer for long runs).
    Short videos (<1 min) + high res → performance (short burst of analysis).
    Medium videos → balanced.
    """
    if duration >= duration_threshold_long:
        return "conservative"
    if duration <= duration_threshold_short and resolution_megapixels >= resolution_threshold_high:
        return "performance"
    return "balanced"


class AdaptivePipelineScaler:
    """Intelligent pipeline scaler that computes optimal per-stage quality settings.

    Usage::

        scaler = AdaptivePipelineScaler()
        result = scaler.analyze(video_path, config, enabled_stages)
        # Apply result.to_dict() as overrides to the config
    """

    def __init__(self, default_policy: str = "auto"):
        self.default_policy = default_policy

    def analyze(
        self,
        video_path: Path,
        enabled_stages: Optional[Dict[str, bool]] = None,
        policy_override: Optional[str] = None,
    ) -> ScalingResult:
        """Analyse a video and produce optimal scaling settings.

        Args:
            video_path: Path to the video file to analyse.
            enabled_stages: Dict of which stages are enabled (``True``) or
                disabled (``False``). If ``None``, all stages are assumed enabled.
            policy_override: Force a specific policy instead of auto-detecting.
                One of ``"conservative"``, ``"balanced"``, ``"performance"``,
                or ``"auto"`` (use default).

        Returns:
            A ``ScalingResult`` with computed overrides and metadata.
        """
        video_path = Path(video_path)
        props = get_video_properties(video_path)

        duration = props.get("duration", 0.0)
        width = int(props.get("width", 0))
        height = int(props.get("height", 0))
        fps = props.get("fps", 0.0)
        resolution_mp = (width * height) / 1_000_000.0 if width > 0 and height > 0 else 0.0

        if enabled_stages is None:
            enabled_stages = {
                "whisper": True,
                "yolo": True,
                "clip": True,
                "xclip": False,
                "video_mllm": False,
                "face_recognition": False,
                "dino": False,
                "ocr": True,
                "diarization": True,
            }

        free_vram = get_free_vram_gb()
        reasoning: List[str] = []

        # Determine policy
        policy = policy_override or self.default_policy
        if policy == "auto":
            policy = select_video_aware_policy(duration, resolution_mp)
            reasoning.append(
                f"Auto-selected '{policy}' policy "
                f"(duration={duration:.0f}s, resolution={width}×{height})"
            )
        else:
            reasoning.append(f"Using '{policy}' policy (configured)")

        # Estimate VRAM at this policy
        estimated_vram = estimate_vram_usage(enabled_stages, policy)
        if free_vram > 0:
            vram_ratio = estimated_vram / free_vram if free_vram > 0 else float("inf")
            reasoning.append(
                f"Estimated VRAM: {estimated_vram:.1f} GB / {free_vram:.1f} GB free "
                f"({vram_ratio:.0%})"
            )
            # Auto-downgrade if VRAM pressure is high
            if vram_ratio > _VRAM_CRITICAL_RATIO:
                old_policy = policy
                policy = "conservative"
                reasoning.append(
                    f"Critical VRAM pressure ({vram_ratio:.0%}) — forced policy "
                    f"from '{old_policy}' to '{policy}'"
                )
                estimated_vram = estimate_vram_usage(enabled_stages, policy)
            elif vram_ratio > _VRAM_WARNING_RATIO and policy == "performance":
                old_policy = policy
                policy = "balanced"
                reasoning.append(
                    f"VRAM pressure ({vram_ratio:.0%}) — downgraded policy "
                    f"from '{old_policy}' to '{policy}'"
                )
                estimated_vram = estimate_vram_usage(enabled_stages, policy)
        else:
            reasoning.append("VRAM detection unavailable — using policy defaults")

        # Number of total frames to extract (estimate)
        effective_fps = FRAME_RATE_TIERS.get(policy, 0.5)
        estimated_frames = int(duration * effective_fps)

        # Build the result
        result = ScalingResult(
            frame_rate=effective_fps,
            scene_threshold=_DEFAULT_SCENE_THRESHOLD,
            frame_analysis_size=FRAME_SIZE_TIERS.get(policy, 960),
            frame_thumbnail_size=THUMBNAIL_SIZE_TIERS.get(policy, 320),
            frame_storage_mode="tiered",
            yolo_confidence=YOLO_CONFIDENCE_TIERS.get(policy, 0.25),
            ocr_model_tier=OCR_TIER_TIERS.get(policy, "medium"),
            dino_frame_compression_threshold=DINO_THRESHOLD_TIERS.get(policy, 0.88),
            quality_min_blur_threshold=QUALITY_BLUR_THRESHOLD_TIERS.get(policy, 100.0),
            policy_used=policy,
            vram_gb_available=free_vram,
            estimated_vram_gb=estimated_vram,
            video_duration=duration,
            video_width=width,
            video_height=height,
            video_fps=fps,
            total_frames_estimate=estimated_frames,
            reasoning=reasoning,
        )

        # Decision: disable expensive stages when VRAM is critically tight
        if free_vram > 0 and estimated_vram > free_vram * _VRAM_STAGE_PRUNE_RATIO:
            # Stage-level pruning — disable the most expensive optional stages
            if enabled_stages.get("xclip", False):
                result.action_recognition_enabled = False
                reasoning.append("Disabling action recognition (X-CLIP) due to VRAM pressure")
            if enabled_stages.get("video_mllm", False):
                result.video_mllm_as_describer = False
                reasoning.append("Disabling MLLM describer due to VRAM pressure")
            if enabled_stages.get("face_recognition", False):
                result.face_recognition_enabled = False
                reasoning.append("Disabling face recognition due to VRAM pressure")

        # For very long videos, never enable optional heavy stages
        if duration > _LONG_VIDEO_DURATION_S:
            if enabled_stages.get("xclip", False):
                result.action_recognition_enabled = False
                reasoning.append(
                    "Disabling action recognition for long video "
                    f"({duration:.0f}s > {_LONG_VIDEO_DURATION_S:.0f}s)"
                )
            if enabled_stages.get("video_mllm", False):
                result.video_mllm_as_describer = False
                reasoning.append(
                    "Disabling MLLM describer for long video "
                    f"({duration:.0f}s > {_LONG_VIDEO_DURATION_S:.0f}s)"
                )
            result.dino_frame_compression_threshold = _LONG_VIDEO_DINO_THRESHOLD
            result.frame_compression_quality = _LONG_VIDEO_JPEG_QUALITY
            reasoning.append(
                "Aggressive compression for long video: DINO threshold=0.95, JPEG quality=75"
            )

        # For short high-resolution videos, boost quality
        if duration <= _SHORT_VIDEO_DURATION_S and resolution_mp >= _SHORT_VIDEO_HIGH_RES_MP:
            result.frame_analysis_size = max(
                result.frame_analysis_size or FRAME_SIZE_TIERS["balanced"],
                _SHORT_VIDEO_MAX_FRAME_SIZE,
            )
            reasoning.append("Short high-resolution video: boosting frame analysis size to 1280")

        logger.info(
            "Adaptive scaling for %s: policy=%s, framerate=%.2f, "
            "analysis_size=%d, yolo_conf=%.2f, ocr_tier=%s, "
            "~%d estimated frames, VRAM=%.1f/%.1f GB",
            video_path.name,
            policy,
            effective_fps,
            result.frame_analysis_size or 960,
            result.yolo_confidence or 0.25,
            result.ocr_model_tier or "medium",
            estimated_frames,
            estimated_vram,
            free_vram if free_vram > 0 else 0,
        )

        return result
