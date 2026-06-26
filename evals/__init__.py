"""
Synthetic fixture generation for evaluation tasks.

Generates test videos, images, and audio with known ground truth so
evaluation metrics can measure accuracy without requiring real video files.
"""

from __future__ import annotations

import struct
import wave
import math
from typing import Dict
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore


# ── Synthetic Image Fixtures ────────────────────────────────────────────────


def render_text_image(
    text: str,
    width: int = 640,
    height: int = 480,
    bg_color: tuple = (32, 32, 32),
    text_color: tuple = (255, 255, 255),
) -> Image.Image:
    """Create a PIL image with text overlaid on a solid background."""
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 32)
    except (OSError, IOError):
        font = ImageFont.load_default()
    # Center text
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((width - tw) // 2, (height - th) // 2), text, fill=text_color, font=font)
    return img


def render_ocr_test_image() -> Image.Image:
    """Create an image with known text for OCR evaluation."""
    return render_text_image("Hello World 42 ABC abc!", width=800, height=200)


def render_scene_transition_image(
    frame_number: int,
    scene_count: int = 3,
    width: int = 320,
    height: int = 240,
    frames_per_scene: int = 30,
) -> Image.Image:
    """Generate a frame from a synthetic multi-scene video.

    Each scene has a distinct color and shape, making scene boundaries
    easily detectable.

    Args:
        frame_number: Global frame index (0-based).
        scene_count: Total number of distinct scenes.
        width, height: Output dimensions.
        frames_per_scene: How many frames per scene.

    Returns:
        PIL Image for this frame.
    """
    scene_idx = min(frame_number // frames_per_scene, scene_count - 1)
    local_frame = frame_number % frames_per_scene

    scene_colors = [
        (64, 64, 180),  # blue
        (64, 180, 64),  # green
        (180, 64, 64),  # red
        (180, 180, 64),  # yellow
        (180, 64, 180),  # magenta
    ]
    color = scene_colors[scene_idx % len(scene_colors)]

    img = Image.new("RGB", (width, height), color)
    draw = ImageDraw.Draw(img)

    # Draw a moving circle within the frame for motion
    cx = width // 2 + int(40 * math.sin(2 * math.pi * local_frame / frames_per_scene))
    cy = height // 2
    draw.ellipse(
        [cx - 30, cy - 30, cx + 30, cy + 30],
        fill=(255 - color[0], 255 - color[1], 255 - color[2]),
    )

    return img


# ── Synthetic Video Fixture ─────────────────────────────────────────────────


def generate_scene_test_video(
    output_path: Path,
    scene_count: int = 3,
    frames_per_scene: int = 30,
    fps: int = 30,
    width: int = 320,
    height: int = 240,
) -> Path:
    """Generate a synthetic test video with known scene boundaries.

    The video has `scene_count` distinct scenes, each `frames_per_scene`
    frames long. Scene cuts occur at frames: 0, frames_per_scene, 2*...
    giving ground-truth boundaries for evaluation.

    Returns:
        Path to the generated video file.
    """
    if cv2 is None:
        raise ImportError("opencv-python required for video generation")

    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    except AttributeError:
        fourcc = 0x7634706D  # "mp4v" fallback

    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    total_frames = scene_count * frames_per_scene
    for i in range(total_frames):
        pil_img = render_scene_transition_image(
            i, scene_count, width, height, frames_per_scene
        )
        cv_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        writer.write(cv_frame)

    writer.release()
    return output_path


# ── Synthetic Audio Fixture ──────────────────────────────────────────────────


def generate_sine_wave_wav(
    output_path: Path,
    frequency_hz: float = 440.0,
    duration_sec: float = 2.0,
    sample_rate: int = 16000,
    amplitude: float = 0.5,
) -> Path:
    """Generate a simple sine wave WAV file.

    Produces a clean sine tone — useful for testing that audio extraction
    produces a non-empty file with the correct sample rate.
    """
    num_samples = int(sample_rate * duration_sec)
    samples = []
    for i in range(num_samples):
        t = i / sample_rate
        value = int(amplitude * 32767 * math.sin(2 * math.pi * frequency_hz * t))
        samples.append(value)

    with wave.open(str(output_path), "w") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(sample_rate)
        wav.writeframes(struct.pack(f"<{len(samples)}h", *samples))

    return output_path


def generate_test_fixtures(data_dir: Path) -> Dict[str, Path]:
    """Generate all synthetic test fixtures for evaluation tasks.

    Returns:
        Dict mapping fixture name to file path.
    """
    fixtures_dir = data_dir / "eval_fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    fixtures: Dict[str, Path] = {}

    # OCR test image
    ocr_img_path = fixtures_dir / "ocr_test.png"
    if not ocr_img_path.exists():
        render_ocr_test_image().save(ocr_img_path)
    fixtures["ocr_image"] = ocr_img_path

    # Scene test video
    scene_video_path = fixtures_dir / "scene_test.mp4"
    if not scene_video_path.exists():
        generate_scene_test_video(scene_video_path)
    fixtures["scene_video"] = scene_video_path

    # Audio test file
    audio_path = fixtures_dir / "sine_tone.wav"
    if not audio_path.exists():
        generate_sine_wave_wav(audio_path)
    fixtures["audio_file"] = audio_path

    return fixtures
