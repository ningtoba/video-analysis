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
from video_analysis.storage import save_frame_tiered
from video_analysis.quality import screen_frame_quality, check_video_corruption

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
        self._action_recognizer = None
        self._ocr_model = None
        self._video_mllm = None  # Optional VideoChat-Flash MLLM

        # Graceful shutdown support
        self._shutdown_requested = False
        try:
            import signal

            signal.signal(signal.SIGTERM, self._handle_signal)
            signal.signal(signal.SIGINT, self._handle_signal)
        except (ValueError, RuntimeError):
            # Not in main thread — signal registration fails silently
            pass

    def _handle_signal(self, signum, frame):
        """Handle SIGTERM/SIGINT for graceful shutdown."""
        import signal as _signal

        sig_name = _signal.Signals(signum).name
        logger.warning(f"Signal {sig_name} received — finishing current operation...")
        self._shutdown_requested = True

    def _unload_model(self, model_attr: str):
        """Safely unload a model from GPU memory.

        Args:
            model_attr: Name of the instance attribute holding the model
                (e.g. ``\"_whisper_model\"``, ``\"_yolo_model\"``).
        """
        import gc
        import torch as _torch

        model = getattr(self, model_attr, None)
        if model is not None:
            setattr(self, model_attr, None)
            del model
        gc.collect()
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()
            _torch.cuda.synchronize()
            logger.debug(f"Model {model_attr} unloaded from GPU memory")

    def cleanup(self):
        """Unload all GPU models and free memory.

        Call this after processing completes to ensure no residual GPU
        memory usage from the pipeline.
        """
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
        if self._action_recognizer is not None:
            try:
                self._action_recognizer.unload()
            except Exception:
                pass
            self._action_recognizer = None
        if self._video_mllm is not None:
            try:
                self._video_mllm.unload()
            except Exception:
                pass
            self._video_mllm = None
        import torch

        torch.cuda.empty_cache()
        import gc

        gc.collect()
        logger.info("Pipeline GPU models cleaned up")

    def _get_active_stages(self) -> set:
        """Return the set of stage names that should be SKIPPED based on processing_mode.

        Modes:
        - ``"video_full"`` (default): no stages skipped, full visual + audio pipeline.
        - ``"audio_only"``: skip all visual stages (scene_detection, frame_extraction, etc.).
        - ``"auto"``: use file-type heuristic classifier (classifier.py) to determine
          the appropriate stages based on the video file being processed.

        ``auto`` mode requires ``self._current_video_path`` to be set before calling
        ``process()``.
        """
        if self.config.processing_mode == "video_full":
            return set()
        if self.config.processing_mode == "audio_only":
            logger.info("Audio-only mode: skipping all visual stages")
            return {
                "scene_detection",
                "frame_extraction",
                "quality_screening",
                "object_detection",
                "face_recognition",
                "ocr",
                "clip_classification",
                "video_mllm",
                "action_recognition",
                "sprite_sheet",
                "rag_indexing",
            }

        # Auto mode: delegate to the file-type classifier
        video_path = getattr(self, "_current_video_path", None)
        if video_path is None:
            logger.warning(
                "Auto mode requested but _current_video_path not set — "
                "falling back to video_full (no stages skipped)"
            )
            return set()

        try:
            from video_analysis.classifier import pipeline_skipped_stages

            skipped = pipeline_skipped_stages(
                video_path,
                processing_mode="auto",
                use_ml=False,  # ML classifier is opt-in; extension + ffprobe are sufficient
            )
            logger.info(
                f"Auto-classified {video_path.name}: "
                f"skipping stages: {skipped if skipped else 'none'}"
            )
            return skipped
        except Exception as e:
            logger.warning(
                f"Auto-classification failed ({e}) — falling back to video_full"
            )
            return set()

    def process(self, video_path: str) -> VideoIndex:
        """
        Process a video file end-to-end:
        1. Extract metadata + audio
        2. Detect scenes and extract key frames
        3. Transcribe audio
        4. Speaker diarization (PyAnnote)
        5. Run object detection on frames (YOLO)
        6. Run OCR text extraction (PaddleOCR)
        7. Run OpenCLIP zero-shot scene classification
        8. Run Video MLLM scene description (optional, VideoChat-Flash)
        9. Run action recognition (X-CLIP, optional)
        10. Assign transcript to scenes
        11. Generate sprite sheet timeline preview
        12. Build VideoIndex
        """
        import time as _time

        _process_start = _time.perf_counter()
        video_path = Path(video_path)
        video_id = video_path.stem
        logger.info(f"Processing video: {video_path.name}")

        # Set the current video path for auto-classification in _get_active_stages()
        self._current_video_path = video_path

        # Determine which stages to skip based on processing_mode
        skipped_stages = self._get_active_stages()
        logger.info(
            f"Processing mode: {self.config.processing_mode} "
            f"(skipped stages: {skipped_stages if skipped_stages else 'none'})"
        )

        # Step 1: Get duration
        duration = self._get_duration(video_path)
        logger.info(f"Duration: {duration:.1f}s")

        # Step 2: Extract audio
        audio_path = self._extract_audio(video_path, video_id)
        logger.info(f"Audio extracted: {audio_path}")

        # Step 3: Detect scenes
        if "scene_detection" not in skipped_stages:
            scenes = self._detect_scenes(video_path, video_id)
            logger.info(f"Detected {len(scenes)} scenes")
        else:
            scenes = []
            logger.info("Scene detection skipped (audio-only mode)")

        # Step 4: Extract key frames per scene
        if "frame_extraction" not in skipped_stages:
            for scene in scenes:
                frames = self._extract_key_frames(
                    video_path,
                    video_id,
                    scene,
                )
                scene.key_frames = frames
            logger.info("Key frames extracted")

            # Step 4.5: Video quality pre-screening (zero VRAM, CPU-only)
            if (
                self.config.quality_screening_enabled
                and "quality_screening" not in skipped_stages
            ):
                quality_results = self._screen_frame_quality(scenes)
                poor_count = sum(
                    1
                    for r in quality_results
                    if r.get("is_blurry") or r.get("is_static")
                )
                if poor_count:
                    logger.info(
                        f"Quality screening flagged {poor_count}/{len(quality_results)} "
                        f"frames as low-quality"
                    )
            logger.info("Quality screening complete")
        else:
            logger.info("Frame extraction skipped (audio-only mode)")

        # Step 5: Transcribe (GPU — faster-whisper large-v3, ~4 GB VRAM)
        transcript_segments, full_transcript = self._transcribe(
            audio_path,
            video_id,
        )
        logger.info(f"Transcription: {len(transcript_segments)} segments")
        # Unload whisper model to free ~4 GB VRAM
        self._unload_model("_whisper_model")

        # Step 6: Speaker diarization (PyAnnote — CPU)
        if self.config.diarize_enabled:
            transcript_segments = self._diarize(
                audio_path, transcript_segments, video_id
            )
            diarized_count = sum(
                1 for s in transcript_segments if s.speaker is not None
            )
            logger.info(
                f"Speaker diarization: {diarized_count}/{len(transcript_segments)} segments labeled"
            )
        else:
            logger.info("Speaker diarization disabled by config")

        # Step 7: Run object detection on frames (GPU — YOLO, ~1 GB VRAM)
        if "object_detection" not in skipped_stages:
            self._detect_objects_on_frames(scenes)
            logger.info("Object detection complete")
        else:
            logger.info("Object detection skipped (audio-only mode)")
        # Unload YOLO to free ~1 GB VRAM
        self._unload_model("_yolo_model")

        # Step 7b: Face recognition (InsightFace, GPU, ~1.1 GB VRAM, optional)
        if "face_recognition" not in skipped_stages:
            if self.config.face_recognition_enabled:
                self._detect_faces(scenes)
                logger.info("Face recognition complete")
                self._unload_face_model()
            else:
                logger.info("Face recognition disabled by config")
        else:
            logger.info("Face recognition skipped (audio-only mode)")

        # Step 8: Run OCR text extraction on frames (PaddleOCR — CPU)
        if "ocr" not in skipped_stages:
            if self.config.ocr_enabled:
                self._extract_ocr(scenes)
                logger.info("OCR text extraction complete")
            else:
                logger.info("OCR text extraction disabled by config")
        else:
            logger.info("OCR text extraction skipped (audio-only mode)")

        # Step 9: Run OpenCLIP zero-shot scene classification on frames (GPU, ~2 GB VRAM)
        if "clip_classification" not in skipped_stages:
            self._describe_scenes_clip(scenes)
            logger.info("OpenCLIP scene classification complete")
        else:
            logger.info("OpenCLIP scene classification skipped (audio-only mode)")
        # Unload CLIP model to free ~2 GB VRAM
        self._unload_model("_clip_model")
        self._unload_model("_clip_preprocess")
        self._unload_model("_clip_tokenizer")

        # Step 10: Optional Video MLLM scene description (VideoChat-Flash 2B, GPU, ~5.4 GB VRAM)
        if "video_mllm" not in skipped_stages:
            if self.config.video_mllm_enabled and self.config.video_mllm_as_describer:
                self._describe_scenes_mllm(scenes)
                logger.info("Video MLLM scene description complete")
                self._unload_model("_video_mllm")
            else:
                logger.info("Video MLLM scene description disabled by config")
        else:
            logger.info("Video MLLM scene description skipped (audio-only mode)")

        # Step 11: Action recognition (optional X-CLIP, GPU, ~4 GB VRAM)
        if "action_recognition" not in skipped_stages:
            if self.config.action_recognition_enabled:
                self._classify_actions(scenes)
                logger.info("Action recognition complete")
                # Action recognizer already unloads itself; explicit safety call
                self._unload_model("_action_recognizer")
            else:
                logger.info("Action recognition disabled by config")
        else:
            logger.info("Action recognition skipped (audio-only mode)")

        # Step 11: Assign transcript to scenes
        self._assign_transcript_to_scenes(scenes, transcript_segments)

        # Step 12: Generate sprite sheet for timeline preview
        if "sprite_sheet" not in skipped_stages:
            sprite_path, sprite_meta = self._generate_sprite_sheet(
                video_path, video_id, num_thumbnails=100
            )
            logger.info(f"Sprite sheet generated: {sprite_path}")
        else:
            sprite_path, sprite_meta = None, None
            logger.info("Sprite sheet generation skipped (audio-only mode)")

        # Step 13: Build index
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

        # Record metrics
        try:
            from video_analysis.metrics import increment_pipeline_run

            _process_dur = _time.perf_counter() - _process_start
            increment_pipeline_run(
                mode=self.config.processing_mode,
                success=True,
                duration_s=_process_dur,
            )
        except Exception:
            pass

        # Fire webhook: pipeline.complete (v0.59.0)
        try:
            from video_analysis.webhook import get_webhook_dispatcher

            wh = get_webhook_dispatcher(self.config)
            if wh.enabled:
                wh.fire(
                    "pipeline.complete",
                    {
                        "video_id": video_id,
                        "filename": video_path.name,
                        "duration": duration,
                        "processing_mode": self.config.processing_mode,
                        "scene_count": len(scenes),
                        "transcript_segments": len(transcript_segments),
                    },
                )
        except Exception:
            pass

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
        Detect scene boundaries using PySceneDetect or FFmpeg.

        Uses the detector configured via ``config.scene_detector``:

        - ``"adaptive"`` (default): :class:`scenedetect.AdaptiveDetector` — rolling
          average of HSV differences; good for camera motion.
        - ``"content"``: :class:`scenedetect.ContentDetector` — fixed-threshold
          HSV-weighted pixel changes; classic fast-cut detection.
        - ``"histogram"``: :class:`scenedetect.HistogramDetector` — Y-channel
          histogram differences for fast cuts.
        - ``"hash"``: :class:`scenedetect.HashDetector` — perceptual hashing
          similarity comparison.
        - ``"ffmpeg"``: legacy FFmpeg ``select='gt(scene,threshold)'`` approach.

        Falls through gracefully:
          1. PySceneDetect installed  →  use ``detect(video, detector)``.
          2. PySceneDetect unavailable →  FFmpeg scene filter.
          3. FFmpeg also fails         →  uniform 30 s chunks.
        """
        scenes: List[SceneInfo] = []
        duration = self._get_duration(video_path)

        # ------------------------------------------------------------------
        # 1. Try PySceneDetect (adaptive, content, histogram, or hash detector)
        # ------------------------------------------------------------------
        if self.config.scene_detector in ("adaptive", "content", "histogram", "hash"):
            try:
                from scenedetect import detect

                if self.config.scene_detector == "adaptive":
                    from scenedetect import AdaptiveDetector

                    detector = AdaptiveDetector(
                        adaptive_threshold=3.0,
                        min_scene_len=15,
                        min_content_val=15.0,
                    )
                elif self.config.scene_detector == "content":
                    from scenedetect import ContentDetector

                    detector = ContentDetector(
                        threshold=self.config.scene_threshold,
                        min_scene_len=15,
                    )
                elif self.config.scene_detector == "histogram":
                    from scenedetect import HistogramDetector

                    detector = HistogramDetector(
                        threshold=self.config.scene_threshold,
                        min_scene_len=15,
                    )
                else:  # hash
                    from scenedetect import HashDetector

                    detector = HashDetector(
                        threshold=self.config.scene_threshold,
                        min_scene_len=15,
                    )

                scene_list = detect(str(video_path), detector)
                timestamps = [0.0]
                for start_tc, end_tc in scene_list:
                    timestamps.append(end_tc.get_seconds())

            except ImportError:
                logger.info(
                    "scenedetect not installed; falling back to FFmpeg scene detection. "
                    "Install with: pip install scenedetect>=0.7.0"
                )
                timestamps = self._detect_scenes_ffmpeg(video_path, duration)
            except Exception as e:
                logger.warning(f"PySceneDetect failed ({e}); falling back to FFmpeg.")
                timestamps = self._detect_scenes_ffmpeg(video_path, duration)

        # ------------------------------------------------------------------
        # 2. FFmpeg mode (explicit or fallback)
        # ------------------------------------------------------------------
        else:
            timestamps = self._detect_scenes_ffmpeg(video_path, duration)

        # ------------------------------------------------------------------
        # 3. Build SceneInfo list from timestamps
        # ------------------------------------------------------------------
        for i in range(len(timestamps)):
            start = timestamps[i]
            end = timestamps[i + 1] if i + 1 < len(timestamps) else duration
            scenes.append(
                SceneInfo(
                    scene_id=i,
                    start_time=start,
                    end_time=end,
                )
            )
        return scenes

    def _detect_scenes_ffmpeg(self, video_path: Path, duration: float) -> List[float]:
        """
        Detect scene boundaries using FFmpeg's ``select='gt(scene,...)'`` filter.

        Returns a list of boundary timestamps (starting with 0.0).  On failure,
        falls back to uniform 30-second chunks.
        """
        try:
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
            timestamps = [0.0]
            for line in result.stderr.split("\n"):
                if "pts_time:" in line:
                    for part in line.split():
                        if part.startswith("pts_time:"):
                            ts = float(part.split(":")[1])
                            if ts > timestamps[-1] + 0.5:  # min 0.5s gap
                                timestamps.append(ts)
            return timestamps
        except Exception as e:
            logger.warning(f"FFmpeg scene detection error: {e}")
            # Fallback: uniform 30-second chunks
            return list(range(0, int(duration), 30))

    def _extract_key_frames(
        self, video_path: Path, video_id: str, scene: SceneInfo
    ) -> List[FrameInfo]:
        """Extract representative frames from a scene.

        Uses configurable strategies:
        - Default: 1 frame every 2 seconds + mid-point of scene
        - Adaptive: more frames near scene boundaries (motion-based), fewer in static regions
        - CLIP dedup: removes near-duplicate frames after extraction
        - Tiered storage: saves analysis-res, full-res, and thumbnail frames when enabled
        """
        frames = []
        scene_duration = scene.end_time - scene.start_time

        if self.config.adaptive_frame_sampling and scene_duration > 5:
            sample_times = self._adaptive_frame_samples(scene, scene_duration)
        else:
            # Default: mid-point + 1 every 2 seconds
            sample_times = {scene.start_time + scene_duration / 2}  # mid point
            for t in range(int(scene.start_time), int(scene.end_time), 2):
                sample_times.add(float(t))

        scene_dir = self.config.frames_dir / video_id / f"scene_{scene.scene_id:04d}"
        scene_dir.mkdir(parents=True, exist_ok=True)

        for ts in sorted(sample_times):
            if ts < scene.start_time or ts > scene.end_time:
                continue
            frame_name = f"frame_{ts:07.2f}"
            frame_path = scene_dir / f"{frame_name}.jpg"
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

            # Apply tiered storage: re-save as analysis/thumbnail variants
            if self.config.frame_storage_mode == "tiered":
                try:
                    from PIL import Image

                    img = Image.open(frame_path).convert("RGB")
                    analysis_path, full_path, thumb_path = save_frame_tiered(
                        img, scene_dir, frame_name, self.config
                    )
                    # Don't delete the original — it's the full-res frame
                    # Record the additional paths for later use
                    frame_info = FrameInfo(
                        timestamp=ts,
                        filepath=str(full_path),
                        scene_id=scene.scene_id,
                        metadata={
                            "analysis_path": analysis_path,
                            "thumbnail_path": thumb_path,
                        },
                    )
                except Exception as e:
                    logger.warning(f"Tiered storage failed for {frame_name}: {e}")
                    frame_info = FrameInfo(
                        timestamp=ts,
                        filepath=str(frame_path),
                        scene_id=scene.scene_id,
                    )
            else:
                frame_info = FrameInfo(
                    timestamp=ts,
                    filepath=str(frame_path),
                    scene_id=scene.scene_id,
                )
            frames.append(frame_info)

        # Optional: CLIP-similarity frame deduplication
        if self.config.clip_frame_dedup and len(frames) > 1:
            frames = self._dedup_frames_clip(frames, video_id)

        # Optional: DINOv2 perceptual frame compression (LongVU-style)
        if self.config.dino_frame_compression and len(frames) > 3:
            frames = self._apply_dino_compression(frames)

        return frames

    def _adaptive_frame_samples(self, scene: SceneInfo, duration: float) -> set:
        """Generate sample times using motion-based adaptive frame sampling.

        Samples more densely near scene boundaries (where change is likely)
        and less frequently in the middle of static scenes. Uses a simple
        cosine-based density function:
        - 3x density in the first 10% and last 10% of the scene
        - Base rate: 1 frame per 2 seconds
        - Dense rate: 1 frame per 0.67 seconds
        """
        sensitivity = self.config.adaptive_frame_sampling_sensitivity
        base_interval = max(0.5, 2.0 * sensitivity)
        dense_interval = base_interval / 3.0

        start, end = scene.start_time, scene.end_time
        region_len = duration * 0.1

        sample_times = set()
        sample_times.add(start + duration / 2)  # mid-point

        # Dense sampling near boundaries
        t = start
        while t <= start + region_len:
            sample_times.add(round(t, 2))
            t += dense_interval

        t = end - region_len
        while t <= end:
            sample_times.add(round(t, 2))
            t += dense_interval

        # Sparse sampling in the middle
        t = start + region_len
        while t <= end - region_len:
            sample_times.add(round(t, 2))
            t += base_interval

        return sample_times

    def _dedup_frames_clip(
        self, frames: List[FrameInfo], video_id: str
    ) -> List[FrameInfo]:
        """Remove near-duplicate frames using CLIP embedding similarity.

        Compares each consecutive pair of frames' CLIP embeddings. If the cosine
        similarity exceeds clip_frame_dedup_threshold, the later frame is considered
        a near-duplicate and removed.

        Requires open-clip-torch to be installed. Falls back to returning all frames.
        """
        if len(frames) < 2:
            return frames

        try:
            import open_clip
            import torch
            import torch.nn.functional as F
        except ImportError:
            logger.debug("open_clip not available for frame dedup — skipping")
            return frames

        device = "cuda" if torch.cuda.is_available() else "cpu"
        threshold = self.config.clip_frame_dedup_threshold

        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device
            )
            model.eval()

            from PIL import Image

            embeddings = []
            valid_indices = []
            for i, frame in enumerate(frames):
                try:
                    img = Image.open(frame.filepath).convert("RGB")
                    img_tensor = preprocess(img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        emb = model.encode_image(img_tensor)
                        emb = F.normalize(emb, dim=-1)
                    embeddings.append(emb.cpu())
                    valid_indices.append(i)
                except Exception:
                    valid_indices.append(i)
                    embeddings.append(None)

            # Dedup: keep frame if similarity to previous kept frame is below threshold
            kept = [frames[0]]
            last_emb = embeddings[0]
            for i in range(1, len(frames)):
                if last_emb is None or embeddings[i] is None:
                    kept.append(frames[i])
                    last_emb = embeddings[i]
                    continue
                sim = (last_emb @ embeddings[i].T).item()
                if sim < threshold:
                    kept.append(frames[i])
                    last_emb = embeddings[i]
                else:
                    logger.debug(
                        f"Dedup frame {frames[i].timestamp:.1f}s "
                        f"(sim={sim:.3f} >= {threshold})"
                    )

            del model
            if device == "cuda":
                torch.cuda.empty_cache()

            dedup_count = len(frames) - len(kept)
            if dedup_count > 0:
                logger.info(
                    f"CLIP dedup removed {dedup_count}/{len(frames)} frames "
                    f"(threshold={threshold})"
                )
            return kept

        except Exception as e:
            logger.warning(f"CLIP frame dedup failed: {e}")
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

    def _screen_frame_quality(self, scenes: List[SceneInfo]) -> List[dict]:
        """Run quality pre-screening on all key frames across all scenes.

        Each frame's quality info is stored in FrameInfo metadata dict.
        Returns a flat list of quality result dicts.
        """
        if not self.config.quality_screening_enabled:
            return []

        all_results = []
        for scene in scenes:
            prev_path = None
            for frame in scene.key_frames:
                result = screen_frame_quality(frame.filepath, self.config, prev_path)
                # Store quality info on the frame
                if not hasattr(frame, "metadata") or frame.metadata is None:
                    frame.metadata = {}
                frame.metadata["quality"] = {
                    "blur_variance": result.get("blur_variance"),
                    "is_blurry": result.get("is_blurry"),
                    "brightness": result.get("brightness"),
                    "is_static": result.get("is_static", False),
                    "skip_ocr": result.get("should_skip_ocr", False),
                    "skip_yolo": result.get("should_skip_yolo", False),
                }
                all_results.append(result)
                prev_path = frame.filepath
        return all_results

    def _detect_objects_on_frames(self, scenes: List[SceneInfo]):
        """Run YOLO object detection on all key frames.

        When ``config.entity_tracking_enabled`` is True, uses YOLO's built-in
        ByteTrack/BoT-SORT tracker to assign persistent track IDs to detected
        objects across consecutive frames. This enables entity-level indexing
        and cross-scene / cross-video entity matching.

        Track IDs are stored as ``track_id`` in each detection dict.
        When tracking is disabled, each frame is processed independently
        (original behavior).
        """
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

        tracking_enabled = self.config.entity_tracking_enabled

        if tracking_enabled:
            # Use YOLO's built-in tracking (ByteTrack by default)
            # Track across all frames in temporal order for persistent IDs
            logger.info(
                f"Entity tracking enabled (tracker: {self.config.entity_tracker_type})"
            )
            # Build a temporal frame list: sort all frames across all scenes by timestamp
            temporal_frames = sorted(all_frames, key=lambda f: f.timestamp)
            # We need to run track() on each frame sequentially so ByteTrack
            # maintains its kalman filter state across frames.
            # Use persist=True so track IDs carry across frame-to-frame calls.
            # We group by scene and process scenes in order so track() state
            # persists within-but-not-across scenes (objects don't survive cuts).
            # Actually, objects CAN survive cuts (e.g. person on both sides),
            # so process all frames in single temporal order.
            for frame in temporal_frames:
                try:
                    results = self._yolo_model.track(
                        frame.filepath,
                        conf=self.config.yolo_confidence,
                        tracker=self.config.entity_tracker_type,
                        persist=True,
                        verbose=False,
                    )
                    detections = []
                    for r in results:
                        if r.boxes is None or r.boxes.id is None:
                            continue
                        for i in range(len(r.boxes)):
                            cls_id = int(r.boxes.cls[i])
                            conf = float(r.boxes.conf[i])
                            track_id = int(r.boxes.id[i])
                            label = r.names[cls_id] if r.names else str(cls_id)
                            x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
                            detections.append(
                                {
                                    "label": label,
                                    "confidence": round(conf, 3),
                                    "bbox": [round(x, 1) for x in [x1, y1, x2, y2]],
                                    "track_id": track_id,
                                }
                            )
                    frame.objects = detections
                except Exception as e:
                    logger.warning(
                        f"Tracking error on {frame.filepath} (t={frame.timestamp:.1f}s): {e}"
                    )
        else:
            # Original per-frame detection without tracking
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

    def _detect_faces(self, scenes: List[SceneInfo]):
        """Run InsightFace face detection on all key frames (optional, GPU).

        Detects faces, extracts 512-d ArcFace embeddings, and stores results
        in each frame's ``faces`` field.  When ``face_recognition_enabled`` is
        True in config, this step runs after object detection.

        Embeddings enable cross-video person identity matching via
        ``FaceRecognizer.match_faces()`` and ``cluster_faces()``.

        Graceful fallback: if insightface is not installed, logs a warning and skips.
        """
        all_frames = []
        for scene in scenes:
            all_frames.extend(scene.key_frames)

        if not all_frames:
            return

        try:
            from video_analysis.face import FaceRecognizer
        except ImportError:
            logger.warning(
                "video_analysis.face module not available — face recognition skipped"
            )
            return

        try:
            logger.info("Loading InsightFace face recognizer...")
            self._face_recognizer = FaceRecognizer(
                match_threshold=self.config.face_match_threshold,
                det_model=self.config.face_detection_model,
            )
            if not self._face_recognizer.available:
                logger.warning(
                    "InsightFace not installed — face recognition skipped. "
                    "Install with: pip install insightface onnxruntime-gpu"
                )
                self._face_recognizer = None
                return
        except Exception as exc:
            logger.warning("Failed to initialise InsightFace: %s", exc)
            self._face_recognizer = None
            return

        recognizer = self._face_recognizer
        max_faces = self.config.face_max_faces
        total_detected = 0

        for idx, frame in enumerate(all_frames):
            if self._shutdown_requested:
                logger.warning("Shutdown requested — stopping face detection")
                break

            try:
                if not Path(frame.filepath).exists():
                    continue

                faces = recognizer.detect_faces(
                    frame.filepath,
                    extract_embedding=True,
                )

                if max_faces > 0:
                    faces = faces[:max_faces]

                if faces:
                    face_dicts = []
                    for f in faces:
                        entry = {
                            "bbox": [round(x, 1) for x in f.bbox],
                            "confidence": round(f.confidence, 3),
                        }
                        if f.embedding:
                            entry["embedding"] = f.embedding
                        if f.face_id:
                            entry["face_id"] = f.face_id
                        if f.gender is not None:
                            entry["gender"] = f.gender
                        if f.age is not None:
                            entry["age"] = f.age
                        face_dicts.append(entry)

                    frame.faces = face_dicts
                    total_detected += len(face_dicts)

            except Exception as e:
                logger.debug(
                    "Face detection error on %s (t=%.1fs): %s",
                    frame.filepath,
                    frame.timestamp,
                    e,
                )

            if (idx + 1) % 50 == 0:
                logger.debug(
                    "Face detection progress: %d/%d frames processed (%d faces detected)",
                    idx + 1,
                    len(all_frames),
                    total_detected,
                )

        logger.info(
            "Face detection complete: %d faces detected across %d frames",
            total_detected,
            len(all_frames),
        )

    def _unload_face_model(self):
        """Release InsightFace model from GPU memory."""
        if hasattr(self, "_face_recognizer") and self._face_recognizer is not None:
            try:
                self._face_recognizer.unload()
            except Exception:
                pass
            self._face_recognizer = None

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
                    f"Loading OpenCLIP {self.config.clip_model} "
                    f"({self.config.clip_pretrained_dataset}) on {device}..."
                )
                model, _, preprocess = open_clip.create_model_and_transforms(
                    self.config.clip_model,
                    pretrained=self.config.clip_pretrained_dataset,
                    device=device,
                )
                tokenizer = open_clip.get_tokenizer(self.config.clip_model)
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

    def _describe_scenes_mllm(self, scenes: List[SceneInfo]):
        """Run VideoChat-Flash MLLM on key frames for rich scene descriptions.

        Optional replacement for OpenCLIP's zero-shot labels.  Uses the
        VideoChat-Flash 2B model (~5.4 GB VRAM) to generate natural language
        descriptions of each scene, including visible objects, people, actions,
        setting, and mood.

        Only runs when ``config.video_mllm_enabled`` *and*
        ``config.video_mllm_as_describer`` are both True.

        Graceful fallback: if the model is unavailable, logs a warning and
        falls back to the existing OpenCLIP descriptions (if any).
        """
        from video_analysis.video_mllm import VideoMLLM

        if self._video_mllm is None:
            self._video_mllm = VideoMLLM(
                model_name=self.config.video_mllm_model,
            )

        if not self._video_mllm.load():
            logger.warning(
                "Video MLLM not available for scene description — "
                "falling back to OpenCLIP descriptions"
            )
            return

        success_count = 0
        for scene in scenes:
            if not scene.key_frames:
                continue

            frame_paths = [f.filepath for f in scene.key_frames if f.filepath]
            if not frame_paths:
                continue

            description = self._video_mllm.describe_scene(frame_paths)
            if description:
                # Append the MLLM description to each frame
                for frame in scene.key_frames:
                    base = frame.description or ""
                    frame.description = (
                        f"{base} | MLLM: {description[:300]}"
                        if base
                        else f"Scene detail: {description[:300]}"
                    )
                success_count += 1

        if success_count > 0:
            logger.info(f"Video MLLM described {success_count}/{len(scenes)} scenes")
        else:
            logger.info("Video MLLM scene description completed (no scenes described)")

    def _classify_actions(self, scenes: List[SceneInfo]):
        """Run X-CLIP zero-shot action recognition on key frames.

        Loads the ActionRecognizer lazily on first call.  After classifying
        all frames, unloads the model to free VRAM.  Graceful fallback if
        ``transformers`` is unavailable or the model download fails.

        Sets ``FrameInfo.action`` and ``FrameInfo.action_confidence`` on
        each frame.
        """
        all_frames = []
        for scene in scenes:
            all_frames.extend(scene.key_frames)

        if not all_frames:
            return

        try:
            from video_analysis.action import (
                ActionRecognizer,
                DEFAULT_ACTION_CATEGORIES,
            )

            if self._action_recognizer is None:
                categories = DEFAULT_ACTION_CATEGORIES
                self._action_recognizer = ActionRecognizer(
                    model_name=self.config.action_model_name,
                    categories=categories,
                )

            recognizer = self._action_recognizer
            results = recognizer.classify(all_frames)

            action_count = 0
            for frame, action, conf in results:
                if action is not None:
                    frame.action = action
                    frame.action_confidence = conf
                    action_count += 1

            # Unload model to free VRAM
            recognizer.unload()
            self._action_recognizer = None

            logger.info(
                f"Action recognition: {action_count}/{len(all_frames)} frames classified"
            )
        except ImportError:
            logger.debug(
                "transformers not available for action recognition -- skipping"
            )
        except Exception as e:
            logger.warning(f"Action recognition failed: {e}")

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

    def _extract_ocr(self, scenes: List[SceneInfo]):
        """
        Run OCR text extraction on key frames using PaddleOCR.

        Graceful fallback: if paddleocr is not installed, logs a warning and skips.
        Extracted text is stored in FrameInfo.ocr_text for each frame.
        GPU used if available, otherwise CPU.
        """
        all_frames = []
        for scene in scenes:
            all_frames.extend(scene.key_frames)

        if not all_frames:
            return

        # Attempt to import PaddleOCR — graceful fallback
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            logger.warning(
                "paddleocr not installed. Skipping OCR text extraction. "
                "Install with: pip install paddleocr"
            )
            return

        if not hasattr(self, "_ocr_model") or self._ocr_model is None:
            try:
                logger.info("Loading PaddleOCR model...")
                self._ocr_model = PaddleOCR(
                    use_angle_cls=True,
                    lang="en",
                    show_log=False,
                    use_gpu=False,  # PaddlePaddle GPU lags CUDA 13.x; use CPU
                )
                logger.info("PaddleOCR model loaded successfully (CPU mode)")
            except Exception as e:
                logger.warning(f"Failed to load PaddleOCR model: {e}")
                return

        ocr = self._ocr_model
        for frame in all_frames:
            try:
                result = ocr.ocr(frame.filepath, cls=True)
                if result and result[0]:
                    texts = []
                    for line in result[0]:
                        bbox, (text, confidence) = line[0], line[1]
                        if confidence >= 0.3:
                            texts.append(f"{text} ({confidence:.0%})")
                    if texts:
                        frame.ocr_text = "; ".join(texts)
                        logger.debug(
                            f"OCR on {Path(frame.filepath).name}: {frame.ocr_text[:80]}"
                        )
            except Exception as e:
                logger.debug(f"OCR error on {frame.filepath}: {e}")

    def _diarize(
        self,
        audio_path: Path,
        transcript_segments: List[TranscriptSegment],
        video_id: str,
    ) -> List[TranscriptSegment]:
        """
        Run speaker diarization using PyAnnote Audio.

        Assigns speaker labels ('SPEAKER_00', 'SPEAKER_01', etc.) to each
        transcript segment based on overlapping diarization turns.

        Graceful fallback: if pyannote.audio is not installed, logs a warning
        and returns the transcript unmodified.

        Args:
            audio_path: Path to the extracted audio WAV file.
            transcript_segments: List of TranscriptSegment from transcription.
            video_id: Unused, kept for future caching.

        Returns:
            List of TranscriptSegment with speaker labels populated.
        """
        try:
            from pyannote.audio import Pipeline
            import torch
        except ImportError:
            logger.warning(
                "pyannote.audio not installed. Skipping speaker diarization. "
                "Install with: pip install pyannote.audio"
            )
            return transcript_segments

        if not transcript_segments:
            return transcript_segments

        if not audio_path.exists():
            logger.warning(f"Audio file not found for diarization: {audio_path}")
            return transcript_segments

        if (
            not hasattr(self, "_diarization_pipeline")
            or self._diarization_pipeline is None
        ):
            try:
                logger.info("Loading PyAnnote speaker diarization pipeline...")
                # Use a local pretrained pipeline from huggingface hub
                # No token needed for pyannote/speaker-diarization-3.1 (MIT license)
                self._diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=None,
                )
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self._diarization_pipeline.to(device)
                logger.info(f"PyAnnote pipeline loaded on {device}")
            except Exception as e:
                logger.warning(
                    f"Failed to load PyAnnote diarization pipeline: {e}. "
                    "This may require huggingface hub access. "
                    "Falling back to no diarization."
                )
                return transcript_segments

        pipeline = self._diarization_pipeline
        try:
            diarization = pipeline({"audio": str(audio_path)})

            # Build a speaker timeline: list of (start, end, speaker) tuples
            speaker_turns = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_turns.append((turn.start, turn.end, speaker))

            if not speaker_turns:
                logger.info("No speakers detected by diarization")
                return transcript_segments

            # Assign speaker labels to transcript segments by overlap
            for seg in transcript_segments:
                seg_start = seg.start
                seg_end = seg.end
                # Find which speaker turn has the most overlap
                best_speaker = None
                best_overlap = 0.0
                for sp_start, sp_end, speaker in speaker_turns:
                    overlap_start = max(seg_start, sp_start)
                    overlap_end = min(seg_end, sp_end)
                    overlap = max(0, overlap_end - overlap_start)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_speaker = speaker
                if best_speaker and best_overlap > 0.1:
                    seg.speaker = best_speaker

            logger.info(
                f"Diarization complete: {len(speaker_turns)} turns, "
                f"{len(set(s for _, _, s in speaker_turns))} speakers"
            )
        except Exception as e:
            logger.warning(f"Diarization failed: {e}")

        return transcript_segments

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
        if self._action_recognizer is not None:
            try:
                self._action_recognizer.unload()
            except Exception:
                pass
            self._action_recognizer = None
        import torch

        torch.cuda.empty_cache()
        import gc

        gc.collect()

    @staticmethod
    def download_from_url(
        url: str, output_dir: Optional[Path] = None, use_config: bool = True
    ) -> Optional[Path]:
        """
        Download a video from YouTube or other supported platforms using yt-dlp.

        Args:
            url: Video URL (YouTube, Vimeo, etc.)
            output_dir: Directory to save the downloaded video. Defaults to config video_dir.
            use_config: If True (default), uses a Config instance for output_dir.

        Returns:
            Path to the downloaded video file, or None on failure.
        """
        try:
            import yt_dlp
        except ImportError:
            logger.error("yt-dlp not installed. Install with: pip install yt-dlp")
            return None

        output_dir = output_dir or Path("data/videos")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / "%(id)s.%(ext)s"
        ydl_opts = {
            "format": "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
            "outtmpl": str(output_path),
            "quiet": True,
            "no_warnings": True,
            "merge_output_format": "mp4",
            "postprocessors": [
                {
                    "key": "FFmpegVideoConvertor",
                    "preferedformat": "mp4",
                }
            ],
        }

        try:
            logger.info(f"Downloading video from URL: {url}")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                video_id = info.get("id", "unknown")
                # The actual file path after download
                downloaded = output_dir / f"{video_id}.mp4"
                if downloaded.exists():
                    logger.info(f"Downloaded: {downloaded}")
                    return downloaded
                # Fallback: try to find any video file in output dir
                import glob

                candidates = list(output_dir.glob(f"{video_id}.*"))
                video_exts = {".mp4", ".mkv", ".webm", ".mov", ".avi"}
                for c in candidates:
                    if c.suffix.lower() in video_exts:
                        logger.info(f"Downloaded: {c}")
                        return c
                return None
        except Exception as e:
            logger.error(f"Failed to download video: {e}")
            return None

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

    def _apply_dino_compression(self, frames: List[FrameInfo]) -> List[FrameInfo]:
        """Apply DINOv2 perceptual frame compression — drop near-duplicate
        frames using cosine similarity of DINOv2 [CLS] features.

        Only called when ``config.dino_frame_compression`` is enabled
        and there are more than 3 frames.  Graceful fallback on failure.

        Args:
            frames: List of FrameInfo to compress.

        Returns:
            Reduced list with redundant frames removed.
        """
        frame_paths = []
        for f in frames:
            p = f.metadata.get("analysis_path") if f.metadata else None
            if not p or not Path(p).exists():
                p = f.filepath
            frame_paths.append(p)

        try:
            from video_analysis.frame_compression import DINOv2FrameCompressor

            compressor = DINOv2FrameCompressor(
                model_name=self.config.dino_frame_compression_model,
                device="cuda",
                threshold=self.config.dino_frame_compression_threshold,
            )

            if not compressor.available:
                logger.info(
                    "DINOv2 frame compression unavailable (transformers not found)"
                )
                return frames

            kept_indices = compressor.compress(frame_paths)
            compressor.unload()

            kept_frames = [frames[i] for i in kept_indices]
            logger.info(
                "DINOv2 compression: %d → %d frames (%.0f%% reduction)",
                len(frames),
                len(kept_frames),
                (1 - len(kept_frames) / len(frames)) * 100,
            )
            return kept_frames
        except Exception as e:
            logger.warning("DINOv2 frame compression failed: %s — skipping", e)
            return frames
