"""
Frame sampler — captures frames at configurable FPS, maintains circular buffer.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SampledFrame:
    """A single sampled frame with metadata."""
    timestamp: float
    frame_bgr: np.ndarray
    frame_index: int
    source_fps: float


class CircularFrameBuffer:
    """Fixed-size circular buffer of sampled frames."""

    def __init__(self, max_frames: int = 300):
        self._buffer: deque = deque(maxlen=max_frames)

    def push(self, frame: SampledFrame):
        self._buffer.append(frame)

    def get_all(self) -> List[SampledFrame]:
        return list(self._buffer)

    def get_recent(self, n: int) -> List[SampledFrame]:
        return list(self._buffer)[-n:]

    def get_since(self, since_ts: float) -> List[SampledFrame]:
        return [f for f in self._buffer if f.timestamp >= since_ts]

    def get_window(self, start_ts: float, end_ts: float) -> List[SampledFrame]:
        return [f for f in self._buffer if start_ts <= f.timestamp <= end_ts]

    @property
    def latest(self) -> Optional[SampledFrame]:
        return self._buffer[-1] if self._buffer else None

    @property
    def count(self) -> int:
        return len(self._buffer)

    @property
    def duration_seconds(self) -> float:
        if len(self._buffer) < 2:
            return 0.0
        return self._buffer[-1].timestamp - self._buffer[0].timestamp


class FrameSampler:
    """Samples frames from a source at a target FPS into a circular buffer.

    Returns True from step() when a frame is sampled,
    False when it's not time yet, None when source exhausted.
    """

    def __init__(
        self,
        read_fn: Callable[[], Optional[Tuple[float, np.ndarray]]],
        target_fps: float = 1.0,
        buffer_seconds: float = 300.0,
        native_fps: float = 30.0,
    ):
        self._read_fn = read_fn
        self._target_fps = max(0.1, target_fps)
        self._native_fps = max(1.0, native_fps)
        self._index = 0
        self._alive = True

        max_frames = int(buffer_seconds * target_fps) + 10
        self._buffer = CircularFrameBuffer(max_frames=max(max_frames, 10))
        self._last_sample_time = 0.0
        self._frame_count = 0
        self._last_print = 0.0

    @property
    def buffer(self) -> CircularFrameBuffer:
        return self._buffer

    def step(self) -> Optional[bool]:
        """Read one frame from source, sample if due.

        Returns:
            True  — frame was sampled and added to buffer
            False — frame read but not sampled yet (waiting for interval)
            None  — source exhausted / dead
        """
        if not self._alive:
            return None

        result = self._read_fn()
        if result is None:
            self._alive = False
            return None

        ts, frame = result
        self._frame_count += 1
        now = time.time()

        if now - self._last_print > 10:
            logger.debug(
                "Sampler: %d frames read, %d sampled, buffer=%d (%.1fs)",
                self._frame_count, self._index,
                self._buffer.count, self._buffer.duration_seconds,
            )
            self._last_print = now

        sample_interval = 1.0 / self._target_fps
        if ts - self._last_sample_time >= sample_interval:
            self._last_sample_time = ts
            sf = SampledFrame(
                timestamp=ts,
                frame_bgr=frame,
                frame_index=self._index,
                source_fps=self._native_fps,
            )
            self._index += 1
            self._buffer.push(sf)
            return True
        return False

    def process_all(self, max_frames: int = 0) -> int:
        """Process all frames from a file source (non-realtime). Returns count."""
        sampled = 0
        while True:
            result = self.step()
            if result is None:
                break
            if result:
                sampled += 1
                if max_frames and sampled >= max_frames:
                    break
        logger.info("Processed %d/%d frames (sampled)", sampled, self._frame_count)
        return sampled

    def run_loop(
        self,
        on_sample: Optional[Callable[[SampledFrame], None]] = None,
        realtime: bool = True,
        max_frames: int = 0,
    ):
        """Run the sampling loop. Blocks for realtime sources."""
        sampled = 0
        while True:
            result = self.step()
            if result is None:
                logger.info("Source exhausted, stopping sampler")
                break
            if result:
                sampled += 1
                latest = self._buffer.latest
                if on_sample and latest:
                    on_sample(latest)
                if max_frames and sampled >= max_frames:
                    break

            if realtime:
                time.sleep(1.0 / self._native_fps * 0.5)

    def close(self):
        self._alive = False
        logger.info("Sampler closed: %d frames sampled", self._index)
