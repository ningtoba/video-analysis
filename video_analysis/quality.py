"""
Frame quality screening — blur, brightness, and static frame detection.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def _image_to_array(img: Image.Image) -> np.ndarray:
    """Convert PIL Image to numpy array."""
    return np.array(img.convert("L"))  # Grayscale


def detect_blur(img: Image.Image, threshold: float = 100.0) -> tuple[bool, float]:
    """Detect blur using Laplacian variance.

    Returns (is_blurry, variance).
    """
    arr = _image_to_array(img)
    import cv2

    laplacian = cv2.Laplacian(arr, cv2.CV_64F)
    variance = laplacian.var()
    is_blurry = variance < threshold
    return is_blurry, variance


def detect_brightness(img: Image.Image, min_val: float = 30.0, max_val: float = 225.0) -> tuple[str, float]:
    """Check if image is too dark or too bright.

    Returns (status, mean_brightness) where status is "ok", "dark", or "bright".
    """
    arr = _image_to_array(img)
    mean_brightness = float(arr.mean())
    if mean_brightness < min_val:
        return "dark", mean_brightness
    if mean_brightness > max_val:
        return "bright", mean_brightness
    return "ok", mean_brightness


def screen_frame_quality(
    img: Image.Image,
    blur_threshold: float = 100.0,
    min_brightness: float = 30.0,
    max_brightness: float = 225.0,
    static_threshold: float = 0.98,
    previous_frame: Optional[np.ndarray] = None,
) -> dict:
    """Screen a frame for quality issues.

    Returns dict with keys: is_blurry, blur_variance, brightness_status,
    brightness_value, is_static (if previous_frame provided), is_usable.
    """
    result = {
        "is_blurry": False,
        "blur_variance": 0.0,
        "brightness_status": "ok",
        "brightness_value": 0.0,
        "is_static": False,
        "is_usable": True,
    }

    # Blur check
    try:
        is_blurry, variance = detect_blur(img, blur_threshold)
        result["is_blurry"] = is_blurry
        result["blur_variance"] = variance
    except Exception as e:
        logger.warning("Blur detection failed: %s", e)

    # Brightness check
    try:
        status, mean = detect_brightness(img, min_brightness, max_brightness)
        result["brightness_status"] = status
        result["brightness_value"] = mean
    except Exception as e:
        logger.warning("Brightness check failed: %s", e)

    # Static frame check
    if previous_frame is not None:
        try:
            arr = _image_to_array(img)
            diff = np.mean(np.abs(arr.astype(float) - previous_frame.astype(float)))
            is_static = diff < (1.0 - static_threshold) * 255
            result["is_static"] = bool(is_static)
        except Exception as e:
            logger.warning("Static frame check failed: %s", e)

    # Overall usability
    result["is_usable"] = not (
        result["is_blurry"] or result["brightness_status"] in ("dark", "bright")
    )

    return result
