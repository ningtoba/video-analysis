"""
Tests for the video quality pre-screening module.
"""

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from video_analysis.config import Config
from video_analysis.quality import (
    check_brightness,
    check_video_corruption,
    detect_blur,
    detect_static_frame,
    screen_frame_quality,
)


def _create_test_image(width=640, height=480, mean_brightness=128) -> str:
    """Create a test image with controlled brightness."""
    arr = np.full((height, width, 3), mean_brightness, dtype=np.uint8)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img = Image.fromarray(arr)
        img.save(f.name, "JPEG", quality=95)
        return f.name


def _create_sharp_image() -> str:
    """Create a sharp test image with high-frequency content."""
    arr = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img = Image.fromarray(arr)
        img.save(f.name, "JPEG", quality=95)
        return f.name


def test_detect_blur_sharp():
    """A sharp (random noise) image should not be flagged as blurry."""
    path = _create_sharp_image()
    try:
        variance, is_blurry = detect_blur(path, threshold=100.0)
        assert not is_blurry, f"Sharp image flagged blurry (variance={variance})"
        assert variance > 100, f"Unexpected low variance: {variance}"
    finally:
        Path(path).unlink(missing_ok=True)


def test_detect_blur_blurry():
    """A uniform (perfectly smooth) image should be flagged as blurry."""
    path = _create_test_image(mean_brightness=128)
    try:
        variance, is_blurry = detect_blur(path, threshold=100.0)
        assert is_blurry, f"Uniform image not flagged blurry (variance={variance})"
        assert variance < 10, f"Unexpected high variance for uniform image: {variance}"
    finally:
        Path(path).unlink(missing_ok=True)


def test_check_brightness_normal():
    """A mid-gray image should have normal brightness."""
    path = _create_test_image(mean_brightness=128)
    try:
        mean_val, issue = check_brightness(path)
        assert issue is None, f"Normal brightness flagged as {issue}"
        assert 120 < mean_val < 140
    finally:
        Path(path).unlink(missing_ok=True)


def test_check_brightness_too_dark():
    """A near-black image should be flagged too dark."""
    path = _create_test_image(mean_brightness=5)
    try:
        mean_val, issue = check_brightness(path, min_brightness=30.0)
        assert issue == "too_dark", f"Dark image not flagged, got {issue}"
        assert mean_val < 10
    finally:
        Path(path).unlink(missing_ok=True)


def test_check_brightness_too_bright():
    """A near-white image should be flagged too bright."""
    path = _create_test_image(mean_brightness=250)
    try:
        mean_val, issue = check_brightness(path, max_brightness=225.0)
        assert issue == "too_bright", f"Bright image not flagged, got {issue}"
        assert mean_val > 240
    finally:
        Path(path).unlink(missing_ok=True)


def test_detect_static_frame_identical():
    """Two identical frames should be detected as static."""
    path_a = _create_test_image(mean_brightness=128)
    try:
        # Same file = identical frames
        similarity, is_static = detect_static_frame(path_a, path_a, ssim_threshold=0.98)
        assert is_static, f"Identical frames not static (sim={similarity})"
        assert similarity >= 0.99
    finally:
        Path(path_a).unlink(missing_ok=True)


def test_detect_static_frame_different():
    """Two very different frames should NOT be detected as static."""
    path_a = _create_test_image(mean_brightness=128)
    path_b = _create_test_image(mean_brightness=5)
    try:
        similarity, is_static = detect_static_frame(path_a, path_b, ssim_threshold=0.98)
        assert not is_static, f"Different frames flagged static (sim={similarity})"
    finally:
        Path(path_a).unlink(missing_ok=True)
        Path(path_b).unlink(missing_ok=True)


def test_check_video_corruption_nonexistent():
    """A nonexistent video should be flagged as corrupted."""
    is_corrupted, error = check_video_corruption(Path("/nonexistent/video.mp4"))
    assert is_corrupted
    assert error is not None


def test_screen_frame_quality_defaults():
    """screen_frame_quality returns all expected keys."""
    config = Config()
    config.quality_min_blur_threshold = 100.0
    config.quality_min_brightness = 30.0
    config.quality_max_brightness = 225.0
    config.quality_static_threshold = 0.98
    config.quality_skip_ocr_on_blurry = True
    config.quality_skip_yolo_on_dark = True

    path = _create_sharp_image()
    try:
        result = screen_frame_quality(path, config)
        assert "blur_variance" in result
        assert "is_blurry" in result
        assert "brightness" in result
        assert "brightness_issue" in result
        assert "is_static" in result
        assert "should_skip_ocr" in result
        assert "should_skip_yolo" in result

        # Sharp + normal brightness = no skip
        assert result["should_skip_ocr"] is False
        assert result["should_skip_yolo"] is False
    finally:
        Path(path).unlink(missing_ok=True)


def test_screen_frame_quality_blurry_skips_ocr():
    """A blurry frame should have should_skip_ocr=True."""
    config = Config()
    config.quality_skip_ocr_on_blurry = True
    config.quality_min_blur_threshold = 100.0

    path = _create_test_image(mean_brightness=128)  # uniform = blurry
    try:
        result = screen_frame_quality(path, config)
        assert result["is_blurry"] is True
        assert result["should_skip_ocr"] is True
    finally:
        Path(path).unlink(missing_ok=True)


def test_screen_frame_quality_with_previous():
    """screen_frame_quality with a previous frame detects static."""
    config = Config()
    path = _create_test_image(mean_brightness=128)
    try:
        result = screen_frame_quality(path, config, previous_filepath=path)
        assert "static_similarity" in result
        assert result["is_static"] is True  # same file = identical
    finally:
        Path(path).unlink(missing_ok=True)
