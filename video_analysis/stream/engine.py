"""
Stream engine — orchestrates frame capture, sampling, motion detection,
LLM analysis, and event logging into a single processing loop.

Can run in two modes:
  - realtime: Process live RTSP/webcam streams (blocks, runs forever)
  - file: Process uploaded video files (runs at full speed, no sleep)

Key differentiators:
  1. Frame clustering: sends only representative frames per similarity cluster
  2. Context-aware analysis: feeds event chain so LLM describes *what changed*
  3. Parallel Whisper: for file mode, transcribes audio as a second event feed
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

from video_analysis.event_memory import EventMemory, StoredEvent
from video_analysis.stream.analyzer import LLMAnalyzer
from video_analysis.stream.motion import MotionDetector
from video_analysis.stream.sampler import FrameSampler, SampledFrame
from video_analysis.stream.source import FrameSource, open_source
from video_analysis.yolo_detector import YOLODetector

logger = logging.getLogger(__name__)


class StreamEngine:
    """Orchestrates real-time video analysis from any source.

    Usage:
        engine = StreamEngine(
            source="rtsp://camera:554/stream",
            llm_chat_fn=my_llm,
            target_fps=1.0,
        )
        engine.start()
        # ... later ...
        events = engine.get_recent_events()
        engine.stop()
    """

    def __init__(
        self,
        source: str,
        llm_chat_fn: Callable,
        stream_id: Optional[str] = None,
        target_fps: float = 1.0,
        buffer_seconds: float = 300.0,
        motion_strategy: str = "diff",
        motion_threshold: float = 0.02,
        periodic_interval: float = 300.0,
        cooldown_seconds: float = 15.0,
        db_path: str = "",
        retention_days: int = 30,
        store_frames: bool = True,
        frame_dir: str = "",
        on_event: Optional[Callable[[StoredEvent], None]] = None,
    ):
        self._source_str = source
        self._source: Optional[FrameSource] = None
        self._stream_id = stream_id or f"stream_{int(time.time())}"
        self._target_fps = target_fps
        self._buffer_seconds = buffer_seconds
        self._store_frames = store_frames
        self._frame_dir = Path(frame_dir or "data/stream_frames")
        self._on_event = on_event
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._transcript_thread: Optional[threading.Thread] = None

        # Components (lazy-init)
        self._sampler: Optional[FrameSampler] = None
        self._motion: Optional[MotionDetector] = None
        self._llm: Optional[LLMAnalyzer] = None
        self._yolo: Optional[YOLODetector] = None
        self._event_memory: Optional[EventMemory] = None
        self._llm_chat_fn = llm_chat_fn

        # Config
        self._motion_strategy = motion_strategy
        self._motion_threshold = motion_threshold
        self._periodic_interval = periodic_interval
        self._cooldown_seconds = cooldown_seconds
        self._db_path = db_path
        self._retention_days = retention_days

        # Stats
        self._frame_count = 0
        self._event_count = 0
        self._start_time = 0.0

    def _init_components(self):
        """Initialize all components (lazy, called from processing thread)."""
        self._source = open_source(self._source_str)
        self._frame_dir.mkdir(parents=True, exist_ok=True)

        self._sampler = FrameSampler(
            read_fn=self._source.read,
            target_fps=self._target_fps,
            buffer_seconds=self._buffer_seconds,
            native_fps=self._source.fps(),
        )

        self._motion = MotionDetector(
            strategy=self._motion_strategy,
            threshold=self._motion_threshold,
        )

        self._llm = LLMAnalyzer(
            chat_fn=self._llm_chat_fn,
            cooldown_seconds=self._cooldown_seconds,
            periodic_interval=self._periodic_interval,
        )

        self._yolo = YOLODetector(
            model_path="yolo11n.pt",
            confidence_threshold=0.25,
            device="cuda" if self._llm_chat_fn else "cpu",
        )

        self._event_memory = EventMemory(
            db_path=self._db_path or str(Path(self._frame_dir).parent / "event_memory.db"),
            retention_days=self._retention_days,
        )

        # For file sources, start parallel Whisper transcription
        if self._source and not self._source.is_realtime:
            self._start_parallel_transcription()

    def _start_parallel_transcription(self):
        """Transcribe audio in a background thread (file mode only).

        Runs Whisper extraction + transcription in parallel with frame analysis.
        Transcript segments are added to the event timeline as [TRANSCRIPT] events.
        """
        def _job():
            try:
                from video_analysis.model_manager import ensure_whisper_model
                from video_analysis.pipeline import _extract_audio, _transcribe_audio


                audio_path = Path(self._db_path).parent / "audio" / f"{self._stream_id}.wav"
                audio_path.parent.mkdir(parents=True, exist_ok=True)

                video_path = Path(self._source_str)
                if not video_path.exists():
                    logger.warning("Video file not found for transcription: %s", self._source_str)
                    return

                logger.info("Starting parallel transcription for %s", video_path.name)
                if _extract_audio(video_path, audio_path):
                    model_name, device, compute_type = ensure_whisper_model("auto")
                    segments = _transcribe_audio(audio_path, model_name, device, compute_type)

                    if self._event_memory:
                        for seg in segments:
                            self._event_memory.store(
                                stream_id=self._stream_id,
                                timestamp=seg.start,
                                objects=[],
                                motion_score=0.0,
                                triggered_by="transcript",
                                description=f"[TRANSCRIPT] {seg.text.strip()}",
                            )
                        logger.info(
                            "Parallel transcription complete: %d segments", len(segments)
                        )
                else:
                    logger.warning("Audio extraction failed for %s", video_path.name)

            except ImportError as e:
                logger.warning("Whisper transcription unavailable: %s", e)
            except Exception as e:
                logger.warning("Parallel transcription failed: %s", e)

        self._transcript_thread = threading.Thread(target=_job, daemon=True)
        self._transcript_thread.start()
        logger.info("Parallel transcription thread started")

    def _save_frame(self, sf: SampledFrame) -> Optional[str]:
        """Save a frame to disk for later reference. Returns path."""
        try:
            import cv2
            filename = f"{self._stream_id}_{sf.frame_index:06d}_{sf.timestamp:.1f}.jpg"
            path = str(self._frame_dir / filename)
            cv2.imwrite(path, sf.frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return path
        except Exception as e:
            logger.warning("Failed to save frame: %s", e)
            return None

    def _process_loop(self):
        """Main processing loop — runs in a background thread."""
        self._init_components()
        self._start_time = time.time()
        self._running = True
        logger.info(
            "Stream engine started: %s (fps=%.1f, buffer=%.0fs, motion=%s)",
            self._source_str, self._target_fps, self._buffer_seconds,
            self._motion_strategy,
        )

        while self._running:
            result = self._sampler.step()
            now = time.time()

            if result is None:
                logger.info("Source ended, stream engine stopping")
                break

            if result:
                # A frame was sampled
                sf = self._sampler.buffer.latest
                if sf is None:
                    continue

                self._frame_count += 1

                # Save frame if configured
                frame_path = None
                if self._store_frames:
                    frame_path = self._save_frame(sf)

                # Detect motion
                motion_score, motion_triggered = self._motion.detect(sf.frame_bgr)

                # Run YOLO detection on every sampled frame
                if self._yolo and sf is not None:
                    try:
                        detections = self._yolo.detect(sf.frame_bgr)
                        if detections:
                            # Convert detections to dict for storage
                            objects = []
                            labels_seen = {}
                            for d in detections:
                                label_key = d.label
                                if label_key not in labels_seen:
                                    labels_seen[label_key] = {'label': label_key, 'count': 0, 'max_conf': 0.0}
                                labels_seen[label_key]['count'] += 1
                                labels_seen[label_key]['max_conf'] = max(labels_seen[label_key]['max_conf'], d.confidence)
                                if d.track_id is not None:
                                    labels_seen[label_key].setdefault('track_ids', []).append(d.track_id)
                            objects = list(labels_seen.values())

                            # Store in event memory
                            self._event_memory.store(
                                stream_id=self._stream_id,
                                timestamp=sf.timestamp,
                                objects=objects,
                                motion_score=motion_score,
                                triggered_by="detection",
                                frame_path=frame_path or "",
                            )
                            self._event_count += 1
                    except Exception as e:
                        logger.warning("YOLO detection failed: %s", e)

                # Check if we should analyze
                should_analyze = False
                triggered_by = ""

                if self._llm.should_analyze_periodic(now):
                    should_analyze = True
                    triggered_by = "periodic"
                    logger.debug("Periodic analysis due")

                if motion_triggered and self._llm.should_analyze_motion(now, motion_score):
                    should_analyze = True
                    triggered_by = "motion"
                    logger.info("Motion triggered: score=%.3f", motion_score)

                if should_analyze:
                    # Use clustered frames: representative per similarity cluster
                    context_frames = self._sampler.buffer.get_clustered(
                        n_clusters=3, max_frames=5
                    )
                    context_frames.reverse()  # Most recent first

                    desc = self._llm.analyze(
                        frames=context_frames,
                        triggered_by=triggered_by,
                        motion_score=motion_score,
                    )

                    if desc and self._event_memory:
                        # Store LLM scene description in event memory
                        self._event_memory.store(
                            stream_id=self._stream_id,
                            timestamp=sf.timestamp,
                            objects=[],
                            motion_score=motion_score,
                            triggered_by=triggered_by,
                            frame_path=frame_path or "",
                            description=desc,
                        )
                        self._event_count += 1
                        logger.info(
                            "Event #%d: %s (motion=%.3f, trigger=%s)",
                            self._event_count, desc[:60], motion_score, triggered_by,
                        )

                        # Notify callback
                        if self._on_event:
                            recent = self._event_memory.query_time_range(
                                self._stream_id, sf.timestamp - 1, sf.timestamp + 1, limit=1
                            )
                            if recent:
                                self._on_event(recent[0])
            # file mode processes at warp speed (no sleep at all).
            if self._source and self._source.is_realtime:
                elapsed = time.time() - now
                sleep = max(0, 1.0 / self._target_fps - elapsed)
                if sleep > 0:
                    time.sleep(sleep)

        self._cleanup()

    def _cleanup(self):
        # Wait for parallel transcription to finish (file mode)
        if self._transcript_thread and self._transcript_thread.is_alive():
            logger.info("Waiting for parallel transcription to finish...")
            self._transcript_thread.join(timeout=60)

        if self._sampler:
            self._sampler.close()
        if self._source:
            self._source.close()
        if self._event_memory:
            self._event_memory.close()
        self._running = False
        elapsed = time.time() - self._start_time
        logger.info(
            "Stream engine stopped: %.1fs, %d frames, %d events",
            elapsed, self._frame_count, self._event_count,
        )

    def start(self, block: bool = False):
        """Start processing.

        Args:
            block: If True, blocks forever (for CLI usage).
                   If False, runs in background thread (for web UI).
        """
        if block:
            self._process_loop()
        else:
            self._thread = threading.Thread(target=self._process_loop, daemon=True)
            self._thread.start()
            logger.info("Stream engine started in background thread")

    def stop(self):
        """Stop processing."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._cleanup()

    def get_recent_events(self, limit: int = 50) -> List[StoredEvent]:
        """Get recent events from event memory."""
        if self._event_memory:
            return self._event_memory.query_time_range(self._stream_id, 0, time.time(), limit=limit)
        return []

    def get_events_in_range(self, start_ts: float, end_ts: float) -> List[StoredEvent]:
        if self._event_memory:
            return self._event_memory.query_time_range(self._stream_id, start_ts, end_ts)
        return []

    @property
    def stats(self) -> Dict:
        return {
            "stream_id": self._stream_id,
            "running": self._running,
            "uptime": time.time() - self._start_time if self._start_time else 0,
            "frames_sampled": self._frame_count,
            "events_created": self._event_count,
            "buffer_count": self._sampler.buffer.count if self._sampler else 0,
            "buffer_duration": self._sampler.buffer.duration_seconds if self._sampler else 0,
            "source": self._source_str,
            "target_fps": self._target_fps,
        }
