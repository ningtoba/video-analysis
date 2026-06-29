"""
Video quality pre-screening for the analysis pipeline.

Fast, CPU-only checks that run before expensive GPU stages:

  - Blur detection: Laplacian variance (lower = more blurry)
  - Brightness check: Mean pixel brightness (over/under-exposure)
  - Static frame detection: SSIM between consecutive frames
  - Corruption check: FFmpeg decode error detection

All operations are <1ms per frame, zero VRAM.
"""

import logging
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from video_analysis.config import Config

logger = logging.getLogger(__name__)


def _load_frame(filepath: str) -> Optional[np.ndarray]:
    """Load a frame as grayscale numpy array for analysis."""
    try:
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        return img
    except Exception:
        return None


def detect_blur(
    filepath: str,
    threshold: float = 100.0,
) -> Tuple[float, bool]:
    """Detect blur using Laplacian variance.

    Args:
        filepath: Path to the image file.
        threshold: Lower bound for acceptable sharpness. Common ranges:
            - <100: very blurry
            - 100-200: moderately sharp
            - >200: very sharp

    Returns:
        (variance, is_blurry) — where is_blurry is True when variance < threshold.
    """
    img = _load_frame(filepath)
    if img is None:
        return 0.0, True  # Can't read → assume blurry
    variance = cv2.Laplacian(img, cv2.CV_64F).var()
    return float(variance), bool(variance < threshold)


def check_brightness(
    filepath: str,
    min_brightness: float = 30.0,
    max_brightness: float = 225.0,
) -> Tuple[float, Optional[str]]:
    """Check mean pixel brightness of a frame.

    Args:
        filepath: Path to the image file.
        min_brightness: Lower bound for acceptable brightness (0-255).
        max_brightness: Upper bound for acceptable brightness (0-255).

    Returns:
        (mean_brightness, issue) where issue is None for normal,
        "too_dark" if below min, "too_bright" if above max.
    """
    img = _load_frame(filepath)
    if img is None:
        return 0.0, "too_dark"

    mean_val = float(np.mean(img))
    if mean_val < min_brightness:
        return mean_val, "too_dark"
    if mean_val > max_brightness:
        return mean_val, "too_bright"
    return mean_val, None


def detect_static_frame(
    filepath_a: str,
    filepath_b: str,
    ssim_threshold: float = 0.98,
) -> Tuple[float, bool]:
    """Detect if two consecutive frames are nearly identical (static).

    Uses pixel-level MSE as a fast proxy — frames with very low change
    are likely static (frozen, slides, paused video).

    Args:
        filepath_a: First frame.
        filepath_b: Second frame (consecutive).
        ssim_threshold: SSIM-like similarity above which frames are "static".

    Returns:
        (similarity_score, is_static).
    """
    img_a = _load_frame(filepath_a)
    img_b = _load_frame(filepath_b)
    if img_a is None or img_b is None:
        return 0.0, False

    # Normalize and compare
    a_norm = img_a.astype(np.float32) / 255.0
    b_norm = img_b.astype(np.float32) / 255.0
    mse = float(np.mean((a_norm - b_norm) ** 2))
    # Convert MSE to a similarity score (1.0 = identical, 0.0 = completely different)
    similarity = 1.0 - min(mse * 10, 1.0)
    return similarity, similarity >= ssim_threshold


def check_video_corruption(video_path: Path) -> Tuple[bool, Optional[str]]:
    """Check if a video file is corrupted using FFmpeg error detection.

    Args:
        video_path: Path to the video file.

    Returns:
        (is_corrupted, error_message).
        If is_corrupted is False, the video is usable.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return True, f"ffprobe error: {result.stderr.strip()}"
        duration = float(result.stdout.strip())
        if duration <= 0:
            return True, f"Zero or negative duration: {duration}s"
        return False, None
    except subprocess.TimeoutExpired:
        return True, "ffprobe timed out after 30s"
    except ValueError as e:
        return True, f"Invalid duration output: {e}"
    except FileNotFoundError:
        return True, "ffprobe not found"
    except Exception as e:
        return True, str(e)


def screen_frame_quality(
    filepath: str,
    config: Config,
    previous_filepath: Optional[str] = None,
) -> Dict[str, object]:
    """Run all quality checks on a single frame.

    Args:
        filepath: Path to the frame image.
        config: Platform config with quality screening settings.
        previous_filepath: Path to the previous frame (for static detection).

    Returns:
        Dict with keys:
            - blur_variance (float)
            - is_blurry (bool)
            - brightness (float)
            - brightness_issue (str or None)
            - is_static (bool, only if previous_filepath provided)
            - static_similarity (float, only if previous_filepath provided)
            - should_skip_ocr (bool — blurry or static)
            - should_skip_yolo (bool — too dark or too bright)
    """
    blur_var, is_blurry = detect_blur(filepath, config.quality_min_blur_threshold)
    brightness, brightness_issue = check_brightness(
        filepath,
        config.quality_min_brightness,
        config.quality_max_brightness,
    )

    result: Dict[str, object] = {
        "blur_variance": blur_var,
        "is_blurry": is_blurry,
        "brightness": brightness,
        "brightness_issue": brightness_issue,
    }

    if previous_filepath:
        sim, is_static = detect_static_frame(
            previous_filepath, filepath, config.quality_static_threshold
        )
        result["static_similarity"] = sim
        result["is_static"] = is_static
    else:
        result["static_similarity"] = 0.0
        result["is_static"] = False

    # Decision logic: skip heavy stages when quality is poor
    skip_ocr = is_blurry or bool(result.get("is_static", False))
    skip_yolo = brightness_issue is not None

    if config.quality_skip_ocr_on_blurry:
        result["should_skip_ocr"] = skip_ocr
    else:
        result["should_skip_ocr"] = False

    if config.quality_skip_yolo_on_dark:
        result["should_skip_yolo"] = skip_yolo
    else:
        result["should_skip_yolo"] = False

    return result
