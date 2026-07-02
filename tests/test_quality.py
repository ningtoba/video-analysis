"""
Tests for the video quality pre-screening module.
"""

import numpy as np
from PIL import Image

from video_analysis.quality import (
    detect_blur,
    detect_brightness,
    screen_frame_quality,
)


def _create_test_image(width=640, height=480, mean_brightness=128) -> Image.Image:
    """Create a test image with controlled brightness."""
    arr = np.full((height, width, 3), mean_brightness, dtype=np.uint8)
    return Image.fromarray(arr)


def _create_sharp_image() -> Image.Image:
    """Create a sharp test image with high-frequency content."""
    arr = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def test_detect_blur_sharp():
    """A sharp (random noise) image should not be flagged as blurry."""
    img = _create_sharp_image()
    is_blurry, variance = detect_blur(img, threshold=100.0)
    assert not is_blurry, f"Sharp image flagged blurry (variance={variance})"
    assert variance > 100, f"Unexpected low variance: {variance}"


def test_detect_blur_blurry():
    """A uniform (perfectly smooth) image should be flagged as blurry."""
    img = _create_test_image(mean_brightness=128)
    is_blurry, variance = detect_blur(img, threshold=100.0)
    assert is_blurry, f"Uniform image not flagged blurry (variance={variance})"
    assert variance < 10, f"Unexpected high variance for uniform image: {variance}"


def test_check_brightness_normal():
    """A mid-gray image should have normal brightness."""
    img = _create_test_image(mean_brightness=128)
    status, mean_val = detect_brightness(img)
    assert status == "ok", f"Normal brightness flagged as {status}"
    assert 120 < mean_val < 140


def test_check_brightness_too_dark():
    """A near-black image should be flagged too dark."""
    img = _create_test_image(mean_brightness=5)
    status, mean_val = detect_brightness(img, min_val=30.0)
    assert status == "dark", f"Dark image not flagged, got {status}"
    assert mean_val < 10


def test_check_brightness_too_bright():
    """A near-white image should be flagged too bright."""
    img = _create_test_image(mean_brightness=250)
    status, mean_val = detect_brightness(img, max_val=225.0)
    assert status == "bright", f"Bright image not flagged, got {status}"
    assert mean_val > 240


def test_screen_frame_quality_defaults():
    """screen_frame_quality returns all expected keys."""
    img = _create_sharp_image()
    result = screen_frame_quality(img, blur_threshold=100.0, min_brightness=30.0, max_brightness=225.0)
    assert "is_blurry" in result
    assert "blur_variance" in result
    assert "brightness_status" in result
    assert "brightness_value" in result
    assert "is_static" in result
    assert "is_usable" in result

    # Sharp + normal brightness = usable
    assert result["is_blurry"] == False
    assert result["brightness_status"] == "ok"
    assert result["is_usable"] is True


def test_screen_frame_quality_blurry_skips_ocr():
    """A blurry frame should be flagged as not usable."""
    img = _create_test_image(mean_brightness=128)  # uniform = blurry
    result = screen_frame_quality(img, blur_threshold=100.0)
    assert result["is_blurry"] == True
    assert result["is_usable"] is False


def test_screen_frame_quality_with_previous():
    """screen_frame_quality with a previous frame detects static."""
    img = _create_test_image(mean_brightness=128)
    prev_arr = np.array(img.convert("L"))
    result = screen_frame_quality(img, previous_frame=prev_arr)
    assert "is_static" in result
    assert result["is_static"] is True  # same image = identical
