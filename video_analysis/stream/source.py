"""
Frame source abstraction — RTSP, webcam, file, or URL.

Handles capture, auto-reconnect, and metadata extraction.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FrameSource(ABC):
    """Abstract frame source. Yields frames on demand."""

    @abstractmethod
    def read(self) -> Optional[Tuple[float, np.ndarray]]:
        """Read next frame. Returns (timestamp_sec, frame_bgr) or None on end."""

    @abstractmethod
    def fps(self) -> float:
        """Native frame rate of the source."""

    @abstractmethod
    def close(self):
        """Release resources."""

    @property
    @abstractmethod
    def is_realtime(self) -> bool:
        """True if this is a live source (not a file)."""


class CV2CaptureSource(FrameSource):
    """Generic OpenCV VideoCapture wrapper for files, webcams, and RTSP."""

    def __init__(self, source: str):
        self._source = source
        self._cap: Optional[cv2.VideoCapture] = None
        self._fps: float = 30.0
        self._is_file: bool = False
        self._open()

    def _open(self):
        if self._cap is not None:
            self._cap.release()

        # Detect source type
        src = self._source.strip()
        if src.isdigit():
            # Webcam index
            self._cap = cv2.VideoCapture(int(src))
            self._is_file = False
        elif Path(src).exists():
            # File path
            self._cap = cv2.VideoCapture(src)
            self._is_file = True
        elif src.startswith(("rtsp://", "rtmp://", "http://", "https://")):
            # RTSP/RTMP/HTTP stream
            self._cap = cv2.VideoCapture(src)
            self._is_file = False
            # Use TCP for RTSP reliability
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            raise ValueError(f"Unknown source type: {src}")

        if not self._cap or not self._cap.isOpened():
            raise RuntimeError(f"Failed to open source: {self._source}")

        # Get native FPS
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._fps = fps if fps > 0 else 30.0

        logger.info(
            "Opened source: %s (fps=%.1f, w=%.0f, h=%.0f, realtime=%s)",
            self._source[:80], self._fps,
            self._cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            not self._is_file,
        )

    def read(self) -> Optional[Tuple[float, np.ndarray]]:
        if self._cap is None:
            return None

        ret, frame = self._cap.read()
        if not ret:
            if self._is_file:
                return None  # End of file
            # Reconnect for live streams
            logger.warning("Frame read failed, reconnecting...")
            time.sleep(1)
            self._open()
            return self.read()

        return time.time(), frame

    def fps(self) -> float:
        return self._fps

    def close(self):
        if self._cap:
            self._cap.release()
            self._cap = None

    @property
    def is_realtime(self) -> bool:
        return not self._is_file


class FileSource(CV2CaptureSource):
    """Explicit file source — same as CV2CaptureSource but always treated as file."""

    def __init__(self, path: str):
        super().__init__(path)

    @property
    def is_realtime(self) -> bool:
        return False


def open_source(source: str) -> FrameSource:
    """Convenience: open any supported source type."""
    return CV2CaptureSource(source)
