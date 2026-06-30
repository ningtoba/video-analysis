"""
Tiered frame storage utilities.

Stores frames in three sizes: analysis, full-res, and thumbnail.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image


def _resize_image(img: Image.Image, longest_edge: int) -> Image.Image:
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
    img: Image.Image,
    output_stem: str,
    output_dir: Path,
    analysis_size: int = 960,
    thumbnail_size: int = 320,
) -> tuple[str, str, str]:
    """Save a frame in three sizes (analysis, full, thumbnail).

    Returns (analysis_path, full_path, thumbnail_path).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analysis tier (resized for LLM analysis)
    analysis = _resize_image(img, analysis_size)
    analysis_path = str(output_dir / f"{output_stem}_analysis.jpg")
    analysis.save(analysis_path, "JPEG", quality=85)

    # Full tier (original size)
    full_path = str(output_dir / f"{output_stem}_full.jpg")
    img.save(full_path, "JPEG", quality=90)

    # Thumbnail tier
    thumb = _resize_image(img, thumbnail_size)
    thumb_path = str(output_dir / f"{output_stem}_thumb.jpg")
    thumb.save(thumb_path, "JPEG", quality=80)

    return analysis_path, full_path, thumb_path


def save_frame_single(img: Image.Image, output_path: Path, quality: int = 85) -> str:
    """Save a single frame."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(output_path), "JPEG", quality=quality)
    return str(output_path)
