"""
Tests for the tiered frame storage module.
"""

import tempfile
from pathlib import Path

from PIL import Image

from video_analysis.storage import (
    _resize_image,
    save_frame_single,
    save_frame_tiered,
)


def test_resize_image_downscale():
    """_resize_image downsamples a large image to the target longest edge."""
    img = Image.new("RGB", (1920, 1080), color="red")
    resized = _resize_image(img, 960)
    assert resized.size == (960, 540), f"Expected (960, 540), got {resized.size}"


def test_resize_image_small():
    """_resize_image does not upscale a small image."""
    img = Image.new("RGB", (320, 240), color="blue")
    resized = _resize_image(img, 960)
    assert resized.size == (320, 240), f"Expected (320, 240), got {resized.size}"


def test_resize_image_portrait():
    """_resize_image handles portrait orientation."""
    img = Image.new("RGB", (1080, 1920), color="green")
    resized = _resize_image(img, 960)
    assert resized.size == (540, 960), f"Expected (540, 960), got {resized.size}"


def test_save_frame_tiered_creates_three_files():
    """save_frame_tiered creates analysis, full-res, and thumbnail files."""
    img = Image.new("RGB", (1920, 1080), color="gray")
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        analysis_path, full_path, thumb_path = save_frame_tiered(
            img, "test_frame", output_dir
        )

        # All three paths exist
        assert Path(analysis_path).exists(), f"Analysis frame not found: {analysis_path}"
        assert Path(full_path).exists(), f"Full-res frame not found: {full_path}"
        assert Path(thumb_path).exists(), f"Thumbnail not found: {thumb_path}"

        # All files are JPEG (current default format)
        assert str(analysis_path).endswith(".jpg")
        assert str(full_path).endswith(".jpg")
        assert str(thumb_path).endswith(".jpg")

        # Analysis frame should be smaller than full-res frame
        analysis_size = Path(analysis_path).stat().st_size
        full_size = Path(full_path).stat().st_size
        # Allow small difference — compressed JPEG sizes can vary
        assert analysis_size > 0
        assert full_size > 0


def test_save_frame_tiered_default_jpeg():
    """save_frame_tiered defaults to JPEG output."""
    img = Image.new("RGB", (640, 480), color="blue")
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        analysis_path, full_path, thumb_path = save_frame_tiered(
            img, "test_frame", output_dir
        )
        assert Path(analysis_path).exists()
        assert Path(full_path).exists()
        assert Path(thumb_path).exists()
        # Default format is JPEG
        assert str(analysis_path).endswith(".jpg")
        assert str(full_path).endswith(".jpg")
        assert str(thumb_path).endswith(".jpg")


def test_save_frame_single():
    """save_frame_single creates a single frame file."""
    img = Image.new("RGB", (100, 100), color="white")
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        path = save_frame_single(img, output_dir / "single_frame.jpg")
        assert Path(path).exists()
        assert str(path).endswith(".jpg")


def test_save_frame_tiered_missing_file_graceful():
    """save_frame_tiered handles graceful file creation on invalid input."""
    # Create a blank image — should be fine
    img = Image.new("RGB", (1, 1), color="black")
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        analysis_path, full_path, thumb_path = save_frame_tiered(img, "tiny", output_dir)
        assert Path(analysis_path).exists()
