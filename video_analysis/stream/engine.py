"""
Stream engine — orchestrates frame capture, sampling, motion detection,
LLM analysis, and event logging into a single processing loop.

Can run in two modes:
  - realtime: Process live RTSP/webcam streams (blocks, runs forever)
  - file: Process uploaded video files (runs at full speed)
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

from video_analysis.stream.analyzer import LLMAnalyzer
from video_analysis.stream.motion import MotionDetector
from video_analysis.stream.sampler import FrameSampler, SampledFrame
from video_analysis.stream.source import FrameSource, open_source
from video_analysis.stream.store import EventStore, TimelineEvent

logger = logging.getLogger(__name__)


class StreamEngine:
    """Orchestrates real-time video analysis from any source.

    Usage:
        engine = StreamEngine(
            source="rtsp://camera:554/stream",
            llm_provider=my_llm,
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
        periodic_interval: float = 30.0,
        cooldown_seconds: float = 15.0,
        db_path: str = "",
        retention_days: int = 30,
        store_frames: bool = True,
        frame_dir: str = "",
        on_event: Optional[Callable[[TimelineEvent], None]] = None,
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

        # Components (lazy-init)
        self._sampler: Optional[FrameSampler] = None
        self._motion: Optional[MotionDetector] = None
        self._llm: Optional[LLMAnalyzer] = None
        self._store: Optional[EventStore] = None
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

        self._store = EventStore(
            db_path=self._db_path,
            retention_days=self._retention_days,
        )

    def _save_frame(self, sf: SampledFrame) -> Optional[str]:
        """Save a frame to disk for later reference. Returns path."""
        import cv2

        filename = f"{self._stream_id}_{sf.frame_index:06d}_{sf.timestamp:.1f}.jpg"
        path = str(self._frame_dir / filename)
        cv2.imwrite(path, sf.frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return path

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
                    # Collect frames for analysis (latest + context frames)
                    context_frames = self._sampler.buffer.get_recent(5)
                    context_frames.reverse()  # Most recent first

                    desc = self._llm.analyze(
                        frames=context_frames,
                        triggered_by=triggered_by,
                        motion_score=motion_score,
                    )

                    if desc and self._store:
                        event = self._store.add_event(
                            stream_id=self._stream_id,
                            timestamp=sf.timestamp,
                            description=desc,
                            frame_path=frame_path,
                            motion_score=motion_score,
                            triggered_by=triggered_by,
                            metadata={
                                "frame_index": sf.frame_index,
                                "motion_score": motion_score,
                                "buffer_size": self._sampler.buffer.count,
                                "buffer_duration": self._sampler.buffer.duration_seconds,
                            },
                        )
                        self._event_count += 1
                        logger.info(
                            "Event #%d: %s (motion=%.3f, trigger=%s)",
                            self._event_count, desc[:60], motion_score, triggered_by,
                        )

                        # Notify callback
                        if self._on_event:
                            # Re-fetch the event for full details
                            ev = self._store.get_latest_event(self._stream_id)
                            if ev:
                                self._on_event(ev)

            # Sleep to match real-time (for live sources)
            if self._source and self._source.is_realtime:
                time.sleep(1.0 / self._target_fps * 0.5)
            else:
                # File source: process as fast as possible (small sleep to yield)
                time.sleep(0.001)

        self._cleanup()

    def _cleanup(self):
        if self._sampler:
            self._sampler.close()
        if self._source:
            self._source.close()
        if self._store:
            self._store.close()
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

    def get_recent_events(self, limit: int = 50) -> List[TimelineEvent]:
        """Get recent events from the event store."""
        if self._store:
            return self._store.get_recent(self._stream_id, limit)
        return []

    def get_events_in_range(self, start_ts: float, end_ts: float) -> List[TimelineEvent]:
        if self._store:
            return self._store.get_range(self._stream_id, start_ts, end_ts)
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
