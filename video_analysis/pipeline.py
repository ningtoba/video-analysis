"""
Video analysis pipeline — extracts audio, transcribes with Whisper, extracts frames,
and analyzes them via LLM Vision API.

The pipeline has no local vision models. All scene understanding, object detection,
OCR, and analysis is done by the configured LLM Vision API (GPT-4o, Claude, Gemini, etc.).
"""
from __future__ import annotations

import base64
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from video_analysis.config import Config
from video_analysis.model_manager import ensure_whisper_model
from video_analysis.models import FrameInfo, SceneInfo, TranscriptSegment, VideoAnalysis

logger = logging.getLogger(__name__)

FFPROBE_TIMEOUT = 30
FFMPEG_AUDIO_EXTRACT_TIMEOUT = 300
FFMPEG_FRAME_EXTRACT_TIMEOUT = 120
FFMPEG_CLIP_EXPORT_TIMEOUT = 300

# ── LLM Vision prompt for frame analysis ────────────────────────────

FRAME_ANALYSIS_PROMPT = """You are analyzing a video frame. Describe what you see in JSON format:
{
  "scene_type": "indoor/outdoor/nature/city/office/etc",
  "description": "Brief 1-sentence description of the scene",
  "objects": ["list", "of", "visible", "objects"],
  "people_count": 0,
  "text_visible": "any visible text in the frame, or empty string",
  "actions": ["any", "observed", "actions"]
}
Return ONLY valid JSON, no other text."""


def _run_ffprobe(video_path: Path) -> Dict[str, Any]:
    """Get video metadata via ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=FFPROBE_TIMEOUT)
        return json.loads(result.stdout)
    except Exception as e:
        logger.warning("ffprobe failed: %s", e)
        return {}


def _extract_audio(video_path: Path, output_path: Path) -> bool:
    """Extract audio to 16kHz mono WAV."""
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                str(output_path),
            ],
            capture_output=True,
            timeout=FFMPEG_AUDIO_EXTRACT_TIMEOUT,
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        logger.error("Audio extraction failed: %s", e.stderr.decode() if e.stderr else str(e))
        return False
    except subprocess.TimeoutExpired:
        logger.error("Audio extraction timed out")
        return False


def _transcribe_audio(
    audio_path: Path,
    model_name: str,
    device: str,
    compute_type: str,
) -> List[TranscriptSegment]:
    """Transcribe audio using faster-whisper."""
    try:
        from faster_whisper import WhisperModel

        model = WhisperModel(model_name, device=device, compute_type=compute_type)

        segments, info = model.transcribe(str(audio_path), beam_size=5)

        result: List[TranscriptSegment] = []
        for seg in segments:
            result.append(TranscriptSegment(
                start=seg.start,
                end=seg.end,
                text=seg.text.strip(),
            ))

        logger.info("Transcribed %d segments (duration: %.1fs)", len(result), info.duration)
        return result

    except Exception as e:
        logger.error("Transcription failed: %s", e)
        return []


def _detect_scenes(video_path: Path, threshold: float = 0.3) -> List[SceneInfo]:
    """Detect scene changes using PySceneDetect."""
    try:
        from scenedetect import ContentDetector, SceneManager, open_video

        video = open_video(str(video_path))
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=threshold))
        scene_manager.detect_scenes(video)

        scenes: List[SceneInfo] = []
        for i, (start, end) in enumerate(scene_manager.get_scene_list()):
            scenes.append(SceneInfo(
                scene_id=i,
                start_time=start.get_seconds(),
                end_time=end.get_seconds(),
            ))

        logger.info("Detected %d scenes", len(scenes))
        return scenes

    except ImportError:
        logger.warning("scenedetect not installed, skipping scene detection")
        return []
    except Exception as e:
        logger.warning("Scene detection failed: %s", e)
        return []


def _extract_frames(
    video_path: Path,
    output_dir: Path,
    frame_rate: float = 0.5,
    scenes: Optional[List[SceneInfo]] = None,
    max_frames: int = 100,
) -> List[FrameInfo]:
    """Extract keyframes from video.

    If scenes are available, extracts from each scene midpoint.
    Otherwise extracts at regular intervals.
    """
    frames: List[FrameInfo] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    if scenes:
        # Extract from scene midpoints
        for scene in scenes:
            if len(frames) >= max_frames:
                break
            mid_time = (scene.start_time + scene.end_time) / 2
            frame_path = output_dir / f"frame_{mid_time:.3f}.jpg"
            success = _extract_frame_at(video_path, mid_time, frame_path)
            if success:
                frames.append(FrameInfo(
                    timestamp=mid_time,
                    filepath=str(frame_path),
                    scene_id=scene.scene_id,
                ))
    else:
        # Extract at regular intervals
        duration = _get_duration(video_path)
        if duration <= 0:
            return frames

        interval = 1.0 / frame_rate if frame_rate > 0 else 5.0
        t = 0.0
        while t < duration and len(frames) < max_frames:
            frame_path = output_dir / f"frame_{t:.3f}.jpg"
            success = _extract_frame_at(video_path, t, frame_path)
            if success:
                frames.append(FrameInfo(timestamp=t, filepath=str(frame_path)))
            t += interval

    logger.info("Extracted %d frames", len(frames))
    return frames


def _extract_frame_at(video_path: Path, timestamp: float, output_path: Path) -> bool:
    """Extract a single frame at the given timestamp."""
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", str(timestamp),
                "-i", str(video_path),
                "-vframes", "1",
                "-q:v", "2",
                str(output_path),
            ],
            capture_output=True,
            timeout=FFMPEG_FRAME_EXTRACT_TIMEOUT,
            check=True,
        )
        return output_path.exists()
    except Exception:
        return False


def _get_duration(video_path: Path) -> float:
    """Get video duration in seconds using ffprobe."""
    info = _run_ffprobe(video_path)
    try:
        return float(info.get("format", {}).get("duration", 0))
    except (ValueError, TypeError):
        return 0.0


def _encode_frame(frame_path: str, max_size: int = 1024) -> Optional[str]:
    """Read and resize a frame, return base64 JPEG."""
    try:
        import io

        from PIL import Image

        img = Image.open(frame_path)
        # Resize if needed
        w, h = img.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        logger.warning("Failed to encode frame %s: %s", frame_path, e)
        return None


def _analyze_frames_with_llm(
    frames: List[FrameInfo],
    llm_provider,
    prompt: str = FRAME_ANALYSIS_PROMPT,
    max_frames: int = 30,
) -> List[FrameInfo]:
    """Send frames to LLM Vision API for analysis.

    Processes frames in batches to avoid exceeding API limits.
    """
    if not frames or not llm_provider:
        return frames

    # Limit frames
    frames_to_analyze = frames[:max_frames]
    logger.info("Analyzing %d frames with LLM Vision...", len(frames_to_analyze))

    for frame in frames_to_analyze:
        encoded = _encode_frame(frame.filepath)
        if not encoded:
            continue

        try:
            response = llm_provider.chat_with_images(
                messages=[{"role": "user", "content": prompt}],
                images=[encoded],
            )

            if response:
                # Parse JSON response
                try:
                    # Find JSON in response
                    json_str = response.strip()
                    if "```json" in json_str:
                        json_str = json_str.split("```json")[1].split("```")[0].strip()
                    elif "```" in json_str:
                        json_str = json_str.split("```")[1].split("```")[0].strip()

                    data = json.loads(json_str)
                    frame.llm_description = data.get("description", "")
                    frame.llm_objects = data.get("objects", [])
                    frame.llm_ocr = data.get("text_visible", "")
                except json.JSONDecodeError:
                    # Use raw response as description
                    frame.llm_description = response[:500]

        except Exception as e:
            logger.warning("LLM Vision analysis failed for frame at %.1fs: %s", frame.timestamp, e)

        # Small delay to avoid rate limiting

        time.sleep(0.1)

    return frames_to_analyze


def _generate_video_summary(
    video_analysis: VideoAnalysis,
    llm_provider,
) -> Optional[str]:
    """Generate a concise video summary using LLM."""
    if not llm_provider:
        return None

    # Build context from transcript and frame descriptions
    transcript_text = "\n".join(
        f"[{seg.start:.1f}s-{seg.end:.1f}s] {seg.text}"
        for seg in video_analysis.transcript[:50]
    )

    frame_descriptions = "\n".join(
        f"[{f.timestamp:.1f}s] {f.llm_description or '(no description)'}"
        for f in video_analysis.frames[:20]
        if f.llm_description
    )

    summary_prompt = f"""Summarize this video based on its transcript and visual analysis.

Title: {video_analysis.title or video_analysis.filename}
Duration: {video_analysis.duration:.1f}s

Transcript excerpts:
{transcript_text or "(no transcript)"}

Visual descriptions:
{frame_descriptions or "(no visual data)"}

Provide a concise 2-3 paragraph summary of what happens in the video."""

    try:
        response = llm_provider.chat(
            messages=[{"role": "user", "content": summary_prompt}],
            system="You are a helpful video analyst.",
        )
        return response
    except Exception as e:
        logger.warning("Summary generation failed: %s", e)
        return None


# ── Main pipeline class ─────────────────────────────────────────────

class VideoPipeline:
    """Main video processing pipeline.

    Simplified: only Whisper ASR + FFmpeg + LLM Vision API.
    No local vision models needed.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self._whisper_model = None
        self._llm_provider = None
        self._shutdown_requested = False

    def _get_llm_provider(self):
        """Lazy-load LLM provider for vision analysis."""
        if self._llm_provider is None:
            from video_analysis.llm_provider import LLMProviderConfig, get_llm_provider

            cfg = LLMProviderConfig(
                provider=self.config.llm_provider,
                api_key=self.config.llm_api_key,
                api_base=self.config.llm_api_base,
                model=self.config.llm_model,
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens,
            )
            self._llm_provider = get_llm_provider(cfg)
        return self._llm_provider

    def process_video(
        self,
        video_path: str,
        video_id: Optional[str] = None,
        skip_llm_vision: bool = False,
    ) -> VideoAnalysis:
        """Process a video file through the full pipeline.

        Args:
            video_path: Path to video file.
            video_id: Optional custom video ID (defaults to filename stem).
            skip_llm_vision: Skip LLM Vision analysis (useful for audio-only).

        Returns:
            VideoAnalysis with transcript, scenes, frames, and LLM analysis.
        """
        path = Path(video_path)
        if not path.exists():
            return VideoAnalysis(
                video_id=video_id or path.stem,
                filename=path.name,
                duration=0,
                error=f"File not found: {video_path}",
            )

        video_id = video_id or path.stem
        duration = _get_duration(path)
        logger.info("Processing video: %s (%.1fs)", path.name, duration)

        analysis = VideoAnalysis(
            video_id=video_id,
            filename=path.name,
            duration=duration,
        )

        # ── Step 1: Extract audio ──
        audio_path = self.config.audio_dir / f"{video_id}.wav"
        logger.info("Extracting audio...")
        if _extract_audio(path, audio_path):
            analysis.transcript = self._transcribe(audio_path)
        else:
            logger.warning("Audio extraction failed, continuing without transcript")

        # ── Step 2: Detect scenes ──
        logger.info("Detecting scenes...")
        analysis.scenes = _detect_scenes(path, self.config.scene_threshold)

        # ── Step 3: Extract frames ──
        if self.config.processing_mode != "audio_only":
            logger.info("Extracting frames...")
            frames_dir = self.config.frames_dir / video_id
            analysis.frames = _extract_frames(
                path, frames_dir,
                frame_rate=self.config.frame_rate,
                scenes=analysis.scenes,
                max_frames=self.config.max_frames_for_llm,
            )

            # ── Step 4: LLM Vision analysis (if not skipped) ──
            if not skip_llm_vision and analysis.frames:
                llm = self._get_llm_provider()
                if llm and self.config.llm_api_key:
                    logger.info("Analyzing frames with LLM Vision...")
                    analyzed = _analyze_frames_with_llm(
                        analysis.frames, llm,
                        max_frames=self.config.max_frames_for_llm,
                    )
                    # Update frames with analysis results
                    analyzed_dict = {f.timestamp: f for f in analyzed}
                    for i, frame in enumerate(analysis.frames):
                        if frame.timestamp in analyzed_dict:
                            analysis.frames[i] = analyzed_dict[frame.timestamp]

                # ── Step 5: Generate summary ──
                if llm and self.config.llm_api_key:
                    logger.info("Generating video summary...")
                    analysis.llm_summary = _generate_video_summary(analysis, llm)

        logger.info("Pipeline complete for %s", video_id)
        return analysis

    def _transcribe(self, audio_path: Path) -> List[TranscriptSegment]:
        """Transcribe audio with auto-selected Whisper model."""
        model_name, device, compute_type = ensure_whisper_model(self.config.whisper_model)
        return _transcribe_audio(audio_path, model_name, device, compute_type)

    def cleanup(self):
        """Clean up resources."""
        self._whisper_model = None
        self._llm_provider = None
        logger.info("Pipeline resources cleaned up")
