"""
Stream manager — manages active StreamEngine instances across the web UI.

Holds a dict of running stream engines, each in its own background thread.
Provides start/stop/list/events API for the web UI.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Dict, List, Optional

from video_analysis.stream.engine import StreamEngine
from video_analysis.stream.store import TimelineEvent

logger = logging.getLogger(__name__)


class StreamManager:
    """Manages active stream engines. Thread-safe."""

    def __init__(self):
        self._engines: Dict[str, StreamEngine] = {}
        self._lock = threading.Lock()
        self._llm_chat_fn: Optional[Callable] = None
        self._next_id = 0

    def set_llm_chat_fn(self, fn: Callable):
        """Set the LLM chat function used by all stream engines."""
        self._llm_chat_fn = fn

    def start(
        self,
        source: str,
        fps: float = 1.0,
        interval: float = 30.0,
        motion_threshold: float = 0.02,
        buffer_seconds: float = 300.0,
        db_path: str = "",
    ) -> str:
        """Start a new stream engine. Returns stream_id."""
        if not self._llm_chat_fn:
            raise RuntimeError("LLM chat function not set — configure LLM provider first")

        with self._lock:
            stream_id = f"stream_{int(time.time())}_{self._next_id}"
            self._next_id += 1

            engine = StreamEngine(
                source=source,
                llm_chat_fn=self._llm_chat_fn,
                stream_id=stream_id,
                target_fps=fps,
                periodic_interval=interval,
                motion_threshold=motion_threshold,
                buffer_seconds=buffer_seconds,
                db_path=db_path,
                store_frames=True,
            )

            engine.start(block=False)
            self._engines[stream_id] = engine
            logger.info("Stream started: %s (source=%s, fps=%.1f)", stream_id, source, fps)
            return stream_id

    def stop(self, stream_id: str) -> bool:
        """Stop a stream engine. Returns True if found."""
        with self._lock:
            engine = self._engines.pop(stream_id, None)
            if engine:
                engine.stop()
                logger.info("Stream stopped: %s", stream_id)
                return True
            return False

    def stop_all(self):
        """Stop all running stream engines."""
        with self._lock:
            for stream_id, engine in list(self._engines.items()):
                engine.stop()
                logger.info("Stream stopped: %s", stream_id)
            self._engines.clear()

    def list(self) -> List[dict]:
        """Return list of active stream states."""
        with self._lock:
            return [
                {
                    "stream_id": sid,
                    **engine.stats,
                }
                for sid, engine in self._engines.items()
            ]

    def get_events(self, stream_id: str, limit: int = 50) -> List[TimelineEvent]:
        """Get recent events for a stream."""
        with self._lock:
            engine = self._engines.get(stream_id)
            if engine:
                return engine.get_recent_events(limit)
            return []

    def get(self, stream_id: str) -> Optional[StreamEngine]:
        """Get engine by ID."""
        with self._lock:
            return self._engines.get(stream_id)

    @property
    def running_count(self) -> int:
        with self._lock:
            return len(self._engines)

    @property
    def is_any_running(self) -> bool:
        with self._lock:
            return len(self._engines) > 0
