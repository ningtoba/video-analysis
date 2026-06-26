"""
Tiered frame storage and compression optimization.

Implements three-tier frame storage to reduce disk usage by 60-75%:
  - Analysis tier: 960×540 JPEG 85% for CLIP/action recognition (~50-80 KB)
  - Full-res tier: Original resolution JPEG 90% for OCR/YOLO (~200-400 KB)
  - Thumbnail tier: 320×180 WebP 80% for timeline preview (~15-25 KB)

All operations are CPU-only — zero VRAM impact.
"""

from pathlib import Path
from typing import Optional, Tuple

from PIL import Image

from video_analysis.config import Config

# Supported compression formats and their PIL identifiers
COMPRESSION_FORMATS = {
    "jpeg": "JPEG",
    "webp": "WebP",
    "png": "PNG",
}


def _resize_image(img: Image.Image, longest_edge: int) -> Image.Image:
    """Resize an image so its longest edge matches `longest_edge`, preserving aspect ratio."""
    w, h = img.size
    if w >= h and w > longest_edge:
        new_w = longest_edge
        new_h = int(h * (longest_edge / w))
        return img.resize((new_w, new_h), Image.LANCZOS)
    elif h > w and h > longest_edge:
        new_h = longest_edge
        new_w = int(w * (longest_edge / h))
        return img.resize((new_w, new_h), Image.LANCZOS)
    return img


def save_frame_tiered(
    frame_data: Image.Image,
    output_dir: Path,
    frame_name: str,
    config: Config,
) -> Tuple[str, str, str]:
    """Save a frame in three tiers and return (analysis_path, full_path, thumbnail_path).

    Args:
        frame_data: The PIL Image to save.
        output_dir: Directory to save into (e.g. data/frames/<video_id>/scene_0001/).
        frame_name: Base filename stem (e.g. "frame_001.00").
        config: Platform config with storage settings.

    Returns:
        Tuple of (analysis_filepath, full_res_filepath, thumbnail_filepath).
        Each path is relative to the output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fmt = COMPRESSION_FORMATS.get(config.frame_compression, "JPEG")
    quality = config.frame_compression_quality
    ext = config.frame_compression  # e.g. "jpeg", "webp"

    # --- Full-resolution frame (for OCR/YOLO) ---
    full_ext = "jpg" if fmt == "JPEG" else ext
    full_path = output_dir / f"{frame_name}.{full_ext}"
    save_kwargs = {"quality": quality} if fmt in ("JPEG", "WebP") else {}
    frame_data.save(str(full_path), fmt, **save_kwargs)

    # --- Analysis-res frame (for CLIP, action recognition) ---
    analysis = _resize_image(frame_data.copy(), config.frame_analysis_size)
    analysis_path = output_dir / f"{frame_name}_analysis.{full_ext}"
    analysis.save(str(analysis_path), fmt, **save_kwargs)

    # --- Thumbnail frame (for timeline preview) ---
    thumb = _resize_image(frame_data.copy(), config.frame_thumbnail_size)
    thumb_path = output_dir / f"{frame_name}_thumb.webp"
    thumb.save(str(thumb_path), "WebP", quality=80)

    return str(analysis_path), str(full_path), str(thumb_path)


def save_frame_single(
    frame_data: Image.Image,
    output_dir: Path,
    frame_name: str,
    config: Config,
) -> str:
    """Save a single frame (non-tiered) using the configured compression settings.

    Returns the filepath of the saved frame.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fmt = COMPRESSION_FORMATS.get(config.frame_compression, "JPEG")
    quality = config.frame_compression_quality
    ext = "jpg" if fmt == "JPEG" else config.frame_compression
    path = output_dir / f"{frame_name}.{ext}"
    save_kwargs = {"quality": quality} if fmt in ("JPEG", "WebP") else {}
    frame_data.save(str(path), fmt, **save_kwargs)
    return str(path)


def compress_existing_frame(
    source_path: Path,
    output_path: Path,
    format_name: str = "webp",
    quality: int = 85,
    longest_edge: Optional[int] = None,
) -> Optional[str]:
    """Re-compress an existing frame, optionally resizing.

    Useful for batch post-processing (archive tier).

    Returns output path string, or None on failure.
    """
    try:
        img = Image.open(source_path).convert("RGB")
        if longest_edge:
            img = _resize_image(img, longest_edge)
        save_kwargs = {"quality": quality} if format_name in ("jpeg", "webp") else {}
        pil_fmt = COMPRESSION_FORMATS.get(format_name, "JPEG")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ext = "jpg" if pil_fmt == "JPEG" else format_name
        final_path = output_path.with_suffix(f".{ext}")
        img.save(str(final_path), pil_fmt, **save_kwargs)
        return str(final_path)
    except Exception as e:
        import logging

        logging.getLogger(__name__).warning(f"compress_existing_frame failed: {e}")
        return None
