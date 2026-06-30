"""
Lightweight change/motion detection between video frames.

Runs on CPU at <5ms per frame. Multiple detection strategies available.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class MotionDetector:
    """Detects motion/change between consecutive frames.

    Strategies:
        - "diff": Mean Squared Error between grayscale frames (fastest, ~1ms)
        - "hist": Histogram correlation (lighting-robust, ~2ms)
        - "background": MOG2 background subtraction (most accurate, ~5ms)
    """

    def __init__(
        self,
        strategy: str = "diff",
        threshold: float = 0.02,
        min_area: float = 0.001,
    ):
        """
        Args:
            strategy: "diff", "hist", or "background"
            threshold: Motion threshold (0-1, lower = more sensitive)
            min_area: Minimum motion area as fraction of frame (0-1)
        """
        self._strategy = strategy
        self._threshold = threshold
        self._min_area = min_area
        self._prev_gray: Optional[np.ndarray] = None
        self._bg_subtractor = None
        self._last_motion_score = 0.0

        if strategy == "background":
            import cv2
            self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=500, varThreshold=36, detectShadows=False,
            )

    def detect(self, frame_bgr: np.ndarray) -> Tuple[float, bool]:
        """Detect motion in a frame. Returns (motion_score, triggered).

        motion_score: 0.0 (no motion) to ~1.0 (full frame motion)
        triggered: True if motion_score > threshold
        """
        import cv2

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        score = 0.0

        if self._strategy == "background" and self._bg_subtractor:
            fgmask = self._bg_subtractor.apply(frame_bgr)
            motion_pixels = np.count_nonzero(fgmask)
            total_pixels = fgmask.shape[0] * fgmask.shape[1]
            score = motion_pixels / total_pixels

        elif self._strategy == "hist":
            if self._prev_gray is not None:
                hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
                hist_prev = cv2.calcHist([self._prev_gray], [0], None, [64], [0, 256])
                cv2.normalize(hist, hist)
                cv2.normalize(hist_prev, hist_prev)
                score = 1.0 - cv2.compareHist(hist, hist_prev, cv2.HISTCMP_CORREL)
            else:
                score = 0.0

        else:  # "diff" — default, fastest
            if self._prev_gray is not None:
                diff = cv2.absdiff(gray, self._prev_gray)
                score = float(np.mean(diff) / 255.0)
            else:
                score = 0.0

        self._prev_gray = gray
        self._last_motion_score = score
        triggered = score > self._threshold and score >= self._last_motion_score * 0.5

        return score, triggered

    def reset(self):
        """Clear motion history (e.g. when switching sources)."""
        self._prev_gray = None
        if self._bg_subtractor:
            import cv2
            self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=500, varThreshold=36, detectShadows=False,
            )
        self._last_motion_score = 0.0

    @property
    def last_score(self) -> float:
        return self._last_motion_score
