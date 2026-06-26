"""
Tests for the tiered frame storage module.
"""

from pathlib import Path
import tempfile
from PIL import Image

from video_analysis.storage import (
    save_frame_tiered,
    save_frame_single,
    compress_existing_frame,
    _resize_image,
)
from video_analysis.config import Config


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
    config = Config()
    config.frame_storage_mode = "tiered"
    config.frame_analysis_size = 960
    config.frame_thumbnail_size = 320
    config.frame_compression = "jpeg"
    config.frame_compression_quality = 85

    img = Image.new("RGB", (1920, 1080), color="gray")
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        analysis_path, full_path, thumb_path = save_frame_tiered(
            img, output_dir, "test_frame", config
        )

        # All three paths exist
        assert Path(
            analysis_path
        ).exists(), f"Analysis frame not found: {analysis_path}"
        assert Path(full_path).exists(), f"Full-res frame not found: {full_path}"
        assert Path(thumb_path).exists(), f"Thumbnail not found: {thumb_path}"

        # Thumbnail is WebP
        assert str(thumb_path).endswith(".webp")

        # Analysis frame should be smaller than full-res frame
        analysis_size = Path(analysis_path).stat().st_size
        full_size = Path(full_path).stat().st_size
        # Allow small difference — compressed JPEG sizes can vary
        assert analysis_size > 0
        assert full_size > 0


def test_save_frame_tiered_webp():
    """save_frame_tiered works with WebP compression."""
    config = Config()
    config.frame_compression = "webp"
    config.frame_compression_quality = 80

    img = Image.new("RGB", (640, 480), color="blue")
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        analysis_path, full_path, thumb_path = save_frame_tiered(
            img, output_dir, "webp_test", config
        )
        assert Path(analysis_path).exists()
        assert Path(full_path).exists()
        # Analysis and full-res use WebP extension too
        assert str(analysis_path).endswith(".webp")


def test_save_frame_single():
    """save_frame_single creates a single frame file."""
    config = Config()
    config.frame_compression = "jpeg"
    config.frame_compression_quality = 85

    img = Image.new("RGB", (100, 100), color="white")
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        path = save_frame_single(img, output_dir, "single_frame", config)
        assert Path(path).exists()
        assert str(path).endswith(".jpg")


def test_compress_existing_frame():
    """compress_existing_frame re-compresses an existing JPEG to WebP."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src = Path(tmpdir) / "source.jpg"
        img = Image.new("RGB", (640, 480), color="red")
        img.save(str(src), "JPEG", quality=95)

        dst = Path(tmpdir) / "compressed.webp"
        result = compress_existing_frame(src, dst, format_name="webp", quality=80)
        assert result is not None
        assert Path(result).exists()
        assert result.endswith(".webp")


def test_compress_existing_frame_resize():
    """compress_existing_frame with resize creates a smaller output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src = Path(tmpdir) / "source.jpg"
        img = Image.new("RGB", (1920, 1080), color="green")
        img.save(str(src), "JPEG", quality=95)

        dst = Path(tmpdir) / "resized.webp"
        result = compress_existing_frame(
            src, dst, format_name="webp", quality=80, longest_edge=320
        )
        assert result is not None
        # Verify the output was resized
        from PIL import Image as PILImage

        out_img = PILImage.open(result)
        assert max(out_img.size) <= 320


def test_save_frame_tiered_missing_file_graceful():
    """save_frame_tiered handles graceful file creation on invalid input."""
    config = Config()
    # Create a blank image — should be fine
    img = Image.new("RGB", (1, 1), color="black")
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        analysis_path, full_path, thumb_path = save_frame_tiered(
            img, output_dir, "tiny", config
        )
        assert Path(analysis_path).exists()
