"""
Video analysis pipeline — the core engine that processes videos.

Extracts frames, detects scenes, transcribes audio, detects objects,
and produces a structured VideoIndex with all metadata.
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np

from video_analysis.config import Config
from video_analysis.models import (
    VideoIndex,
    SceneInfo,
    FrameInfo,
    TranscriptSegment,
)

logger = logging.getLogger(__name__)


class VideoPipeline:
    """Main video processing pipeline."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self._whisper_model = None
        self._yolo_model = None

    def process(self, video_path: str) -> VideoIndex:
        """
        Process a video file end-to-end:
        1. Extract metadata + audio
        2. Detect scenes and extract key frames
        3. Transcribe audio
        4. Describe frames (objects via YOLO, OCR)
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

        # Step 7: Assign transcript to scenes
        self._assign_transcript_to_scenes(scenes, transcript_segments)

        # Step 8: Build index
        index = VideoIndex(
            video_id=video_id,
            filename=video_path.name,
            duration=duration,
            filepath=str(video_path),
            scenes=scenes,
            transcript=transcript_segments,
            full_transcript=full_transcript,
        )
        return index

    def _get_duration(self, video_path: Path) -> float:
        """Get video duration using ffprobe."""
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

    def cleanup(self):
        """Release GPU memory."""
        if self._whisper_model is not None:
            del self._whisper_model
            self._whisper_model = None
        if self._yolo_model is not None:
            del self._yolo_model
            self._yolo_model = None
        import torch

        torch.cuda.empty_cache()
