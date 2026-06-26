"""
Video analysis pipeline — the core engine that processes videos.

Extracts frames, detects scenes, transcribes audio, detects objects,
and produces a structured VideoIndex with all metadata.
"""

import json
import logging
import math
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from PIL import Image

from video_analysis.config import Config
from video_analysis.models import (
    VideoIndex,
    SceneInfo,
    FrameInfo,
    TranscriptSegment,
)

logger = logging.getLogger(__name__)

# Default candidate labels for OpenCLIP zero-shot scene classification.
# These cover a broad range of common video scenes.
DEFAULT_CLIP_LABELS = [
    "indoor scene",
    "outdoor scene",
    "cityscape",
    "nature",
    "people talking",
    "person speaking",
    "crowd",
    "empty room",
    "sports",
    "action scene",
    "calm scene",
    "technology/computer screen",
    "food",
    "vehicle",
    "animal",
    "text/document",
    "presentation",
    "interview",
    "lecture",
    "music performance",
    "cooking",
    "building",
    "landscape",
    "night scene",
    "underwater",
]


class VideoPipeline:
    """Main video processing pipeline."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self._whisper_model = None
        self._yolo_model = None
        self._clip_model = None
        self._clip_preprocess = None
        self._clip_tokenizer = None

    def process(self, video_path: str) -> VideoIndex:
        """
        Process a video file end-to-end:
        1. Extract metadata + audio
        2. Detect scenes and extract key frames
        3. Transcribe audio
        4. Describe frames (objects via YOLO, OCR, OpenCLIP zero-shot)
        5. Assemble VideoIndex
        """
        video_path = Path(video_path)
        video_id = video_path.stem
        logger.info(f"Processing video: {video_path.name}")

        # Step 1: Get duration
        duration = self._get_duration(video_path)
        logger.info(f"Duration: {duration:.1f}s")

        # Step 2: Extract audio
        audio_path = self._extract_audio(video_path, video_id)
        logger.info(f"Audio extracted: {audio_path}")

        # Step 3: Detect scenes
        scenes = self._detect_scenes(video_path, video_id)
        logger.info(f"Detected {len(scenes)} scenes")

        # Step 4: Extract key frames per scene
        for scene in scenes:
            frames = self._extract_key_frames(
                video_path,
                video_id,
                scene,
            )
            scene.key_frames = frames
        logger.info("Key frames extracted")

        # Step 5: Transcribe
        transcript_segments, full_transcript = self._transcribe(
            audio_path,
            video_id,
        )
        logger.info(f"Transcription: {len(transcript_segments)} segments")

        # Step 6: Run object detection on frames
        self._detect_objects_on_frames(scenes)
        logger.info("Object detection complete")

        # Step 7: Run OpenCLIP zero-shot scene classification on frames
        self._describe_scenes_clip(scenes)
        logger.info("OpenCLIP scene classification complete")

        # Step 8: Assign transcript to scenes
        self._assign_transcript_to_scenes(scenes, transcript_segments)

        # Step 9: Generate sprite sheet for timeline preview
        sprite_path, sprite_meta = self._generate_sprite_sheet(
            video_path, video_id, num_thumbnails=100
        )
        logger.info(f"Sprite sheet generated: {sprite_path}")

        # Step 10: Build index
        index = VideoIndex(
            video_id=video_id,
            filename=video_path.name,
            duration=duration,
            filepath=str(video_path),
            scenes=scenes,
            transcript=transcript_segments,
            full_transcript=full_transcript,
            sprite_sheet=str(sprite_path) if sprite_path else None,
            sprite_metadata=sprite_meta or {},
        )
        return index

    def _get_duration(self, video_path: Path) -> float:
        """Get video duration using ffprobe. Returns 0.0 on failure."""
        if not video_path.exists():
            logger.warning(f"Video file not found: {video_path}")
            return 0.0
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "json",
                    str(video_path),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            data = json.loads(result.stdout)
            return float(data["format"]["duration"])
        except (json.JSONDecodeError, KeyError, subprocess.TimeoutExpired) as e:
            logger.warning(f"Failed to get duration for {video_path}: {e}")
            return 0.0

    def _extract_audio(self, video_path: Path, video_id: str) -> Path:
        """Extract audio as 16kHz mono WAV."""
        audio_path = self.config.audio_dir / f"{video_id}.wav"
        if audio_path.exists():
            return audio_path
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                str(video_path),
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                "-y",
                str(audio_path),
            ],
            capture_output=True,
            timeout=300,
            check=True,
        )
        return audio_path

    def _detect_scenes(self, video_path: Path, video_id: str) -> List[SceneInfo]:
        """
        Detect scene boundaries using FFmpeg scene detection.
        Falls back to FFmpeg scene filter if PySceneDetect is unavailable.
        """
        scenes = []
        try:
            # Try FFmpeg scene detection (always available)
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    str(video_path),
                    "-vf",
                    f"select='gt(scene,{self.config.scene_threshold})',showinfo",
                    "-vsync",
                    "vfr",
                    "-f",
                    "null",
                    "-",
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )
            # Parse showinfo output for scene timestamps
            timestamps = [0.0]
            for line in result.stderr.split("\n"):
                if "pts_time:" in line:
                    for part in line.split():
                        if part.startswith("pts_time:"):
                            ts = float(part.split(":")[1])
                            if ts > timestamps[-1] + 0.5:  # min 0.5s gap
                                timestamps.append(ts)
        except Exception as e:
            logger.warning(f"Scene detection error: {e}")
            # Fallback: uniform 30-second chunks
            duration = self._get_duration(video_path)
            timestamps = list(range(0, int(duration), 30))

        # Build scenes from timestamps
        for i in range(len(timestamps)):
            start = timestamps[i]
            end = (
                timestamps[i + 1]
                if i + 1 < len(timestamps)
                else self._get_duration(video_path)
            )
            scenes.append(
                SceneInfo(
                    scene_id=i,
                    start_time=start,
                    end_time=end,
                )
            )
        return scenes

    def _extract_key_frames(
        self, video_path: Path, video_id: str, scene: SceneInfo
    ) -> List[FrameInfo]:
        """Extract representative frames from a scene."""
        frames = []
        scene_duration = scene.end_time - scene.start_time

        # Extract frames: 1 every 2 seconds within the scene, plus mid-point
        sample_times = {scene.start_time + scene_duration / 2}  # mid point
        for t in range(int(scene.start_time), int(scene.end_time), 2):
            sample_times.add(float(t))

        scene_dir = self.config.frames_dir / video_id / f"scene_{scene.scene_id:04d}"
        scene_dir.mkdir(parents=True, exist_ok=True)

        for ts in sorted(sample_times):
            if ts < scene.start_time or ts > scene.end_time:
                continue
            frame_path = scene_dir / f"frame_{ts:07.2f}.jpg"
            if not frame_path.exists():
                subprocess.run(
                    [
                        "ffmpeg",
                        "-ss",
                        str(ts),
                        "-i",
                        str(video_path),
                        "-vframes",
                        "1",
                        "-qscale:v",
                        "2",
                        "-y",
                        str(frame_path),
                    ],
                    capture_output=True,
                    timeout=60,
                    check=True,
                )
            frames.append(
                FrameInfo(
                    timestamp=ts,
                    filepath=str(frame_path),
                    scene_id=scene.scene_id,
                )
            )
        return frames

    def _transcribe(
        self, audio_path: Path, video_id: str
    ) -> tuple[List[TranscriptSegment], str]:
        """Transcribe audio using faster-whisper."""
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            logger.error("faster-whisper not installed")
            return [], ""

        if self._whisper_model is None:
            logger.info(f"Loading whisper model: {self.config.whisper_model}")
            self._whisper_model = WhisperModel(
                self.config.whisper_model,
                device=self.config.whisper_device,
                compute_type=self.config.whisper_compute_type,
            )

        segments, info = self._whisper_model.transcribe(
            str(audio_path),
            vad_filter=True,
            beam_size=5,
            language="en",
        )

        transcript_segments = []
        full_text_parts = []
        for seg in segments:
            transcript_segments.append(
                TranscriptSegment(
                    start=seg.start,
                    end=seg.end,
                    text=seg.text.strip(),
                    words=(
                        [
                            {"word": w.word, "start": w.start, "end": w.end}
                            for w in seg.words
                        ]
                        if hasattr(seg, "words") and seg.words
                        else []
                    ),
                )
            )
            full_text_parts.append(seg.text.strip())

        return transcript_segments, " ".join(full_text_parts)

    def _detect_objects_on_frames(self, scenes: List[SceneInfo]):
        """Run YOLO object detection on all key frames."""
        all_frames = []
        for scene in scenes:
            all_frames.extend(scene.key_frames)

        if not all_frames:
            return

        try:
            from ultralytics import YOLO
        except ImportError:
            logger.warning("ultralytics not installed, skipping object detection")
            return

        if self._yolo_model is None:
            try:
                logger.info(f"Loading YOLO model: {self.config.yolo_model}")
                self._yolo_model = YOLO(self.config.yolo_model)
                logger.info("YOLO model loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load YOLO model: {e}")
                # Try smaller model
                try:
                    self._yolo_model = YOLO("yolo26n.pt")
                except Exception as e2:
                    logger.warning(f"Could not load any YOLO model: {e2}")
                    return

        # Process frames in batches
        for frame in all_frames:
            try:
                results = self._yolo_model(
                    frame.filepath,
                    conf=self.config.yolo_confidence,
                    verbose=False,
                )
                detections = []
                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        label = r.names[cls_id] if r.names else str(cls_id)
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        detections.append(
                            {
                                "label": label,
                                "confidence": round(conf, 3),
                                "bbox": [round(x, 1) for x in [x1, y1, x2, y2]],
                            }
                        )
                frame.objects = detections
            except Exception as e:
                logger.warning(f"Detection error on {frame.filepath}: {e}")

    def _describe_scenes_clip(
        self, scenes: List[SceneInfo], labels: Optional[List[str]] = None
    ):
        """
        Run OpenCLIP zero-shot classification on key frames and set frame descriptions.

        For each key frame, the model scores the frame against a set of candidate
        labels and assigns a human-readable description string with the top-3 labels
        and their confidence percentages.

        Graceful fallback: if open_clip is not installed, logs a warning and skips.

        Args:
            scenes: List of SceneInfo with key_frames to classify.
            labels: Optional list of candidate labels. Uses DEFAULT_CLIP_LABELS if None.
        """
        # Collect all frames
        all_frames = []
        for scene in scenes:
            all_frames.extend(scene.key_frames)

        if not all_frames:
            return

        # Attempt to import open_clip — graceful fallback if not installed
        try:
            import open_clip
            import torch
        except ImportError:
            logger.warning(
                "open-clip-torch not installed. Skipping zero-shot scene classification. "
                "Install with: pip install open-clip-torch>=2.24.0"
            )
            return

        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        candidate_labels = labels or DEFAULT_CLIP_LABELS

        # Lazy-load the OpenCLIP model
        if self._clip_model is None:
            try:
                logger.info(
                    f"Loading OpenCLIP ViT-B-32 (laion2b_s34b_b79k) on {device}..."
                )
                model, _, preprocess = open_clip.create_model_and_transforms(
                    "ViT-B-32",
                    pretrained="laion2b_s34b_b79k",
                    device=device,
                )
                tokenizer = open_clip.get_tokenizer("ViT-B-32")
                self._clip_model = model
                self._clip_preprocess = preprocess
                self._clip_tokenizer = tokenizer
                logger.info("OpenCLIP model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load OpenCLIP model: {e}")
                return

        model = self._clip_model
        preprocess = self._clip_preprocess
        tokenizer = self._clip_tokenizer

        # Tokenize labels once
        import torch.nn.functional as F

        text_tokens = tokenizer(candidate_labels).to(device)

        with torch.no_grad():
            # Encode all text labels once
            text_features = model.encode_text(text_tokens)
            text_features = F.normalize(text_features, dim=-1)

            # Process frames in batches for GPU efficiency
            batch_size = getattr(self.config, "clip_batch_size", 16)

            for i in range(0, len(all_frames), batch_size):
                batch = all_frames[i : i + batch_size]

                # Load and preprocess images
                try:
                    from PIL import Image

                    images = []
                    valid_indices = []
                    for idx, frame in enumerate(batch):
                        try:
                            img = Image.open(frame.filepath).convert("RGB")
                            images.append(preprocess(img).unsqueeze(0))
                            valid_indices.append(idx)
                        except Exception as e:
                            logger.warning(
                                f"Could not load frame {frame.filepath}: {e}"
                            )

                    if not images:
                        continue

                    image_input = torch.cat(images).to(device)

                    # Encode images
                    image_features = model.encode_image(image_input)
                    image_features = F.normalize(image_features, dim=-1)

                    # Compute similarity scores (zero-shot classification)
                    similarity = (100.0 * image_features @ text_features.T).softmax(
                        dim=-1
                    )

                    # Get top-3 labels per frame
                    top_values, top_indices = similarity.topk(3, dim=-1)

                    for j, local_idx in enumerate(valid_indices):
                        frame = batch[local_idx]
                        labels_str = ", ".join(
                            f"{candidate_labels[top_indices[j, k].item()]} "
                            f"({top_values[j, k].item():.0f}%)"
                            for k in range(3)
                        )
                        frame.description = f"Scene: {labels_str}"

                except Exception as e:
                    logger.warning(f"CLIP classification batch error: {e}")
                    continue

        # Unload CLIP model from GPU if not needed for further processing
        # (keep it cached for potential reuse within the same pipeline run)
        # Explicit GPU memory management: if VRAM is tight, release after use
        import gc

        if device == "cuda":
            # Let caller manage via cleanup() — but we can free the model
            # if the user requested tight memory mode
            if getattr(self.config, "clip_unload_after_inference", False):
                del self._clip_model
                del self._clip_preprocess
                del self._clip_tokenizer
                self._clip_model = None
                self._clip_preprocess = None
                self._clip_tokenizer = None
                torch.cuda.empty_cache()
                gc.collect()
                logger.info("OpenCLIP model unloaded from GPU to free VRAM")

    def _assign_transcript_to_scenes(
        self, scenes: List[SceneInfo], transcript: List[TranscriptSegment]
    ):
        """Assign transcript segments to scenes based on timing."""
        for segment in transcript:
            for scene in scenes:
                if scene.start_time <= segment.start <= scene.end_time:
                    if scene.transcript:
                        scene.transcript += " " + segment.text
                    else:
                        scene.transcript = segment.text
                    break

    def _generate_sprite_sheet(
        self,
        video_path: Path,
        video_id: str,
        num_thumbnails: int = 100,
    ) -> Tuple[Optional[Path], Optional[dict]]:
        """
        Generate a thumbnail sprite sheet for timeline hover preview.

        Extracts evenly-spaced frames from the video, montages them into a
        single sprite sheet image (default: 10 columns x 10 rows = 100
        thumbnails), saves it to ``thumbnails_dir/{video_id}_sprite.jpg``,
        and writes a metadata JSON with the timestamp for each thumbnail.

        Sprite sheet format:
            - Each thumbnail: 160x90 pixels
            - Total width:  num_columns * 160
            - Total height: num_rows * 90
            - Arranged left-to-right, top-to-bottom

        Returns
        -------
        (sprite_path, metadata)
            sprite_path : Path or None if an error occurred.
            metadata    : dict with structure
                { "num_thumbnails": int,
                  "num_columns": int,
                  "num_rows": int,
                  "thumbnail_width": int,
                  "thumbnail_height": int,
                  "duration": float,
                  "thumbnails": [
                      {"index": 0, "timestamp": 0.0, "x": 0, "y": 0},
                      ...
                  ]
                } or None if an error occurred.
        """
        duration = self._get_duration(video_path)
        if duration <= 0 or num_thumbnails < 1:
            return None, None

        cols = 10
        rows = math.ceil(num_thumbnails / cols)
        thumb_w = 160
        thumb_h = 90
        total = rows * cols  # actual number of thumbnails in the grid

        sprite_path = self.config.thumbnails_dir / f"{video_id}_sprite.jpg"
        sprite_path.parent.mkdir(parents=True, exist_ok=True)

        # Work in a temp directory for individual frame extraction
        tmp_dir = self.config.thumbnails_dir / f"{video_id}_sprite_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        metadata: dict = {
            "num_thumbnails": total,
            "num_columns": cols,
            "num_rows": rows,
            "thumbnail_width": thumb_w,
            "thumbnail_height": thumb_h,
            "duration": duration,
            "thumbnails": [],
        }

        try:
            # Extract evenly-spaced frames using FFmpeg
            step = duration / total if total > 1 else duration

            frame_files: List[Path] = []
            for i in range(total):
                ts = i * step
                ts = min(ts, duration - 0.001)  # stay within bounds
                frame_file = tmp_dir / f"thumb_{i:04d}.jpg"
                result = subprocess.run(
                    [
                        "ffmpeg",
                        "-ss",
                        str(ts),
                        "-i",
                        str(video_path),
                        "-vframes",
                        "1",
                        "-qscale:v",
                        "3",  # high quality
                        "-vf",
                        f"scale={thumb_w}:{thumb_h}",
                        "-y",
                        str(frame_file),
                    ],
                    capture_output=True,
                    timeout=60,
                )
                # FFmpeg may return non-zero (e.g., 234) for end-of-stream
                # or when seeking past the last keyframe; that's okay —
                # we just check if the file was actually created.
                if result.returncode != 0:
                    logger.debug(
                        f"FFmpeg warn for thumb {i} @ {ts:.2f}s: "
                        f"exit={result.returncode}"
                    )
                if frame_file.exists():
                    frame_files.append(frame_file)
                    col = i % cols
                    row = i // cols
                    metadata["thumbnails"].append(
                        {
                            "index": i,
                            "timestamp": round(ts, 3),
                            "x": col * thumb_w,
                            "y": row * thumb_h,
                        }
                    )

            if not frame_files:
                logger.warning("No thumbnails extracted for sprite sheet")
                return None, None

            # Montage into a single sprite sheet
            sprite_img = Image.new("RGB", (cols * thumb_w, rows * thumb_h))
            for i, fpath in enumerate(frame_files):
                col = i % cols
                row = i // cols
                try:
                    img = Image.open(fpath).resize((thumb_w, thumb_h), Image.LANCZOS)
                    sprite_img.paste(img, (col * thumb_w, row * thumb_h))
                except Exception:
                    pass

            sprite_img.save(sprite_path, "JPEG", quality=85)
            logger.info(
                f"Sprite sheet saved: {sprite_path} "
                f"({sprite_img.size[0]}x{sprite_img.size[1]}, "
                f"{len(frame_files)} thumbnails)"
            )

            # Write metadata JSON alongside
            meta_path = sprite_path.with_suffix(".json")
            meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

            return sprite_path, metadata

        except Exception as e:
            logger.error(f"Sprite sheet generation failed: {e}")
            return None, None

        finally:
            # Clean up temp files
            import shutil

            shutil.rmtree(tmp_dir, ignore_errors=True)

    def cleanup(self):
        """Release GPU memory from all loaded models."""
        if self._whisper_model is not None:
            del self._whisper_model
            self._whisper_model = None
        if self._yolo_model is not None:
            del self._yolo_model
            self._yolo_model = None
        if self._clip_model is not None:
            del self._clip_model
            self._clip_model = None
            self._clip_preprocess = None
            self._clip_tokenizer = None
        import torch

        torch.cuda.empty_cache()
        import gc

        gc.collect()

    def export_clip(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        output_name: Optional[str] = None,
    ) -> str:
        """
        Export a clip from the video.

        Args:
            video_path: Path to the source video
            start_time: Start time in seconds
            end_time: End time in seconds
            output_name: Optional output filename (without extension)

        Returns:
            Path to the exported clip
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        video_id = video_path.stem
        output_name = output_name or f"{video_id}_clip_{start_time:.0f}_{end_time:.0f}"
        output_path = self.config.clip_export_dir / f"{output_name}.mp4"

        if output_path.exists():
            return str(output_path)

        duration = end_time - start_time
        logger.info(
            f"Exporting clip: {start_time:.1f}s → {end_time:.1f}s "
            f"(duration: {duration:.1f}s)"
        )

        subprocess.run(
            [
                "ffmpeg",
                "-ss",
                str(start_time),
                "-i",
                str(video_path),
                "-t",
                str(duration),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "22",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-y",
                str(output_path),
            ],
            capture_output=True,
            timeout=300,
            check=True,
        )

        logger.info(f"Clip exported: {output_path}")
        return str(output_path)
