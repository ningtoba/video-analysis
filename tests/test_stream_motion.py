"""
Tests for the stream motion detection module.

MotionDetector wraps cv2 internally (``import cv2`` inside detect / reset),
so ``cv2`` is never a module-level attribute on ``video_analysis.stream.motion``.
We inject the mock via ``sys.modules`` so that the local import picks it up.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from video_analysis.stream.motion import MotionDetector


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════


def _gray_frame(height: int = 100, width: int = 100, value: int = 0) -> np.ndarray:
    """Return a single-channel uint8 array (as cv2.cvtColor returns after BGR->GRAY)."""
    return np.full((height, width), value, dtype=np.uint8)


def _bgr_frame(height: int = 100, width: int = 100, value: int = 0) -> np.ndarray:
    """Return a 3-channel BGR uint8 array simulating a video frame."""
    return np.full((height, width, 3), value, dtype=np.uint8)


def _make_cv2_mock(**config) -> MagicMock:
    """Build a ``cv2`` mock pre-configured with common constants and return values.

    Keyword args are set as attributes on the mock *after* the defaults, so
    callers can override any of them::

        mock = _make_cv2_mock(absdiff=MagicMock(return_value=…))
    """
    mock = MagicMock()
    mock.COLOR_BGR2GRAY = 6  # standard OpenCV constant

    # Default return values (tests override as needed)
    mock.cvtColor.return_value = _gray_frame()
    mock.absdiff.return_value = _gray_frame(value=0)

    for key, val in config.items():
        setattr(mock, key, val)

    return mock


from contextlib import contextmanager


@contextmanager
def cv2_patch(**cv2_config):
    """Context manager that injects a mock ``cv2`` into ``sys.modules`` and yields it.

    Use instead of ``patch("video_analysis.stream.motion.cv2")`` because
    ``cv2`` is imported *inside* methods, never at module scope.

    Usage::

        with cv2_patch(absdiff=MagicMock(return_value=…)) as mock_cv2:
            detector.detect(frame)
    """
    mock = _make_cv2_mock(**cv2_config)
    with patch.dict(sys.modules, {"cv2": mock}):
        yield mock


# ═════════════════════════════════════════════════════════════════════════════
# MotionDetector unit tests
# ═════════════════════════════════════════════════════════════════════════════


class TestMotionDetectorInit:
    """Construction and parameter defaults."""

    def test_default_params(self):
        """MotionDetector uses ``diff`` strategy, 0.02 threshold, 0.001 min_area."""
        detector = MotionDetector()

        assert detector._strategy == "diff"
        assert detector._threshold == 0.02
        assert detector._min_area == 0.001
        assert detector._prev_gray is None
        assert detector._last_motion_score == 0.0
        assert detector._bg_subtractor is None

    def test_custom_params(self):
        """MotionDetector accepts non-default strategy, threshold, min_area."""
        detector = MotionDetector(strategy="hist", threshold=0.1, min_area=0.01)
        assert detector._strategy == "hist"
        assert detector._threshold == 0.1
        assert detector._min_area == 0.01



class TestMotionDetectorDetect:
    """detect() contract: (motion_score: float, triggered: bool)."""

    # ── Return type contract ──────────────────────────────────────────────

    def test_bg_subtractor_created_for_background_strategy(self):
        """Background strategy lazily imports cv2 and creates a MOG2 subtractor."""
        mock_sub = MagicMock()
        mock_cv2 = MagicMock()
        mock_cv2.createBackgroundSubtractorMOG2.return_value = mock_sub

        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            detector = MotionDetector(strategy="background")

        assert detector._bg_subtractor is mock_sub
        mock_cv2.createBackgroundSubtractorMOG2.assert_called_once_with(
            history=500, varThreshold=36, detectShadows=False,
        )

    # ── First call (no previous frame to compare) ────────────────────────

    def test_first_call_returns_zero_score_not_triggered(self):
        """The first call to detect has no prior frame, so score == 0.0, triggered == False."""
        detector = MotionDetector()

        with cv2_patch() as mock_cv2:
            score, triggered = detector.detect(_bgr_frame())

        assert score == 0.0
        assert triggered == False
        # Ensure absdiff was never consulted
        mock_cv2.absdiff.assert_not_called()

    # ── Identical frames (no motion) ─────────────────────────────────────

    def test_identical_frames_no_motion(self):
        """Two consecutive identical frames yield score ~0.0, triggered=False."""
        detector = MotionDetector()
        frame = _bgr_frame()

        with cv2_patch() as mock_cv2:
            # first call — seeds prev_gray
            detector.detect(frame)

            # second call — absdiff of identical frames = all zeros
            mock_cv2.absdiff.return_value = _gray_frame(value=0)
            score, triggered = detector.detect(frame)

        assert score == 0.0
        assert triggered == False

    # ── Different frames (motion) ────────────────────────────────────────

    def test_different_frames_detects_motion(self):
        """Different consecutive frames yield score > threshold, triggered=True."""
        detector = MotionDetector(threshold=0.02)
        frame = _bgr_frame()

        with cv2_patch() as mock_cv2:
            # seed prev_gray
            detector.detect(frame)

            # second frame: absdiff returns values averaging ~0.196 => score > 0.02
            mock_cv2.absdiff.return_value = _gray_frame(value=50)
            score, triggered = detector.detect(frame)

        # np.mean(50) / 255 ~= 0.196078
        assert score == pytest.approx(50.0 / 255.0)
        assert score > 0.02
        assert triggered == True

    # ── Threshold boundary ───────────────────────────────────────────────

    def test_motion_below_threshold_not_triggered(self):
        """Score below threshold does NOT trigger."""
        detector = MotionDetector(threshold=0.1)
        frame = _bgr_frame()

        with cv2_patch() as mock_cv2:
            detector.detect(frame)

            # score = 25 / 255 ~= 0.098 < 0.1
            mock_cv2.absdiff.return_value = _gray_frame(value=25)
            score, triggered = detector.detect(frame)

        assert score < 0.1
        assert triggered == False

    def test_motion_above_threshold_triggered(self):
        """Score immediately above threshold triggers."""
        detector = MotionDetector(threshold=0.1)
        frame = _bgr_frame()

        with cv2_patch() as mock_cv2:
            detector.detect(frame)

            # score = 26 / 255 ~= 0.102 > 0.1
            mock_cv2.absdiff.return_value = _gray_frame(value=26)
            score, triggered = detector.detect(frame)

        assert score > 0.1
        assert triggered == True

    # ── detect is called multiple times ──────────────────────────────────

    def test_repeated_calls_accumulate_state(self):
        """Multiple detect calls each compute motion against the _previous_ frame."""
        detector = MotionDetector()

        with cv2_patch() as mock_cv2:
            detector.detect(_bgr_frame())  # seed

            mock_cv2.absdiff.return_value = _gray_frame(value=10)
            s1, _ = detector.detect(_bgr_frame())

            mock_cv2.absdiff.return_value = _gray_frame(value=80)
            s2, _ = detector.detect(_bgr_frame())

        assert s2 > s1  # more difference => higher score

    # ── Edge: full-frame motion ──────────────────────────────────────────

    def test_full_frame_motion_saturates(self):
        """Extreme pixel differences produce scores approaching 1.0."""
        detector = MotionDetector()
        frame = _bgr_frame()

        with cv2_patch() as mock_cv2:
            detector.detect(frame)

            mock_cv2.absdiff.return_value = _gray_frame(value=255)
            score, triggered = detector.detect(frame)

        # 255 / 255 = 1.0 — every pixel at maximum difference
        assert score == pytest.approx(1.0)
        assert triggered == True

    # ── None / invalid frame ─────────────────────────────────────────────

    def test_none_frame_raises_type_error(self):
        """Calling detect(None) propagates the cv2-level TypeError."""
        detector = MotionDetector()

        with cv2_patch(cvtColor=MagicMock(side_effect=TypeError("Expected numpy array"))):
            with pytest.raises(TypeError):
                detector.detect(None)  # type: ignore[arg-type]

    def test_non_array_frame_raises_type_error(self):
        """Calling detect with a non-numpy value raises TypeError."""
        detector = MotionDetector()

        with cv2_patch(cvtColor=MagicMock(side_effect=TypeError("Expected numpy array"))):
            with pytest.raises(TypeError):
                detector.detect("not-a-frame")  # type: ignore[arg-type]


class TestMotionDetectorReset:
    """reset() behaviour."""

    def test_reset_clears_prev_gray_and_score(self):
        """reset() nullifies the previous frame and zeros last_score."""
        detector = MotionDetector()

        with cv2_patch() as mock_cv2:
            mock_cv2.absdiff.return_value = _gray_frame(value=50)
            detector.detect(_bgr_frame())
            detector.detect(_bgr_frame())

        assert detector.last_score > 0  # internal state built up

        detector.reset()

        assert detector._prev_gray is None
        assert detector.last_score == 0.0

    def test_reset_restarts_motion_history(self):
        """After reset(), the next detect call behaves like the first call (score=0)."""
        detector = MotionDetector()

        with cv2_patch() as mock_cv2:
            mock_cv2.absdiff.return_value = _gray_frame(value=100)
            detector.detect(_bgr_frame())
            detector.detect(_bgr_frame())

        detector.reset()

        with cv2_patch() as mock_cv2:
            score, triggered = detector.detect(_bgr_frame())

        assert score == 0.0
        assert triggered == False

    def test_reset_with_bg_subtractor_recreates_it(self):
        """Reset on a background-strategy detector creates a fresh subtractor."""
        mock_sub = MagicMock()
        mock_sub.apply.return_value = np.zeros((100, 100), dtype=np.uint8)
        fresh_sub = MagicMock()
        fresh_sub.apply.return_value = np.zeros((100, 100), dtype=np.uint8)
        mock_cv2 = MagicMock()
        mock_cv2.createBackgroundSubtractorMOG2.return_value = mock_sub
        mock_cv2.cvtColor.return_value = _gray_frame()

        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            detector = MotionDetector(strategy="background")
            detector.detect(_bgr_frame())

            # reset will call createBackgroundSubtractorMOG2 again
            mock_cv2.createBackgroundSubtractorMOG2.return_value = fresh_sub
            detector.reset()

        assert detector._bg_subtractor is fresh_sub


class TestMotionDetectorLastScore:
    """last_score property."""

    def test_initial_last_score_is_zero(self):
        """Before any detect call, last_score is 0.0."""
        detector = MotionDetector()
        assert detector.last_score == 0.0

    def test_last_score_after_first_detect(self):
        """After the first detect (no prior frame), last_score is 0.0."""
        detector = MotionDetector()

        with cv2_patch():
            detector.detect(_bgr_frame())

        assert detector.last_score == 0.0

    def test_last_score_reflects_latest_motion(self):
        """last_score returns the score from the most recent detect call."""
        detector = MotionDetector()

        with cv2_patch() as mock_cv2:
            detector.detect(_bgr_frame())  # score = 0

            mock_cv2.absdiff.return_value = _gray_frame(value=77)
            detector.detect(_bgr_frame())  # score = 77/255

        expected = 77.0 / 255.0
        assert detector.last_score == pytest.approx(expected, abs=1e-9)


class TestMotionDetectorStrategyDiff:
    """Diff strategy (default) behaviour."""

    def test_diff_strategy_explicit(self):
        """``diff`` strategy can be set explicitly via constructor."""
        detector = MotionDetector(strategy="diff")
        frame = _bgr_frame()

        with cv2_patch() as mock_cv2:
            detector.detect(frame)
            mock_cv2.absdiff.return_value = _gray_frame(value=30)
            score, triggered = detector.detect(frame)

        assert score == pytest.approx(30.0 / 255.0)
        assert triggered


class TestMotionDetectorStrategyHist:
    """Hist strategy behaviour."""

    def test_hist_no_prev_frame(self):
        """Hist strategy returns 0 on the first detect call."""
        detector = MotionDetector(strategy="hist")
        frame = _bgr_frame()

        with cv2_patch():
            score, triggered = detector.detect(frame)

        assert score == 0.0
        assert triggered == False

    def test_hist_different_frames(self):
        """Hist strategy computes histogram correlation when prev_gray exists."""
        detector = MotionDetector(strategy="hist")
        frame = _bgr_frame()

        with cv2_patch() as mock_cv2:
            detector.detect(frame)

            mock_cv2.calcHist.return_value = np.array([[1.0]], dtype=np.float32)
            mock_cv2.compareHist.return_value = 0.3  # correlation
            score, triggered = detector.detect(frame)

        # score = 1.0 - compareHist = 1.0 - 0.3 = 0.7
        assert score == pytest.approx(1.0 - 0.3)
        assert triggered == True

    def test_hist_normalize_called(self):
        """Hist strategy normalises both current and previous histograms."""
        detector = MotionDetector(strategy="hist")
        frame = _bgr_frame()

        with cv2_patch() as mock_cv2:
            detector.detect(frame)

            mock_cv2.calcHist.return_value = np.array([[1.0]], dtype=np.float32)
            mock_cv2.compareHist.return_value = 0.0
            detector.detect(frame)

        # normalize called twice (current + previous histograms)
        assert mock_cv2.normalize.call_count == 2


class TestMotionDetectorStrategyBackground:
    """Background subtraction strategy."""

    def test_background_first_call(self):
        """Background strategy processes the first frame, returns score based on fgmask."""
        mock_sub = MagicMock()
        mock_sub.apply.return_value = np.zeros((100, 100), dtype=np.uint8)
        mock_cv2 = MagicMock()
        mock_cv2.createBackgroundSubtractorMOG2.return_value = mock_sub
        mock_cv2.cvtColor.return_value = _gray_frame()

        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            detector = MotionDetector(strategy="background")
            score, triggered = detector.detect(_bgr_frame())

        assert score == 0.0
        assert triggered == False
        mock_sub.apply.assert_called_once()

    def test_background_with_motion(self):
        """Background strategy detects motion when fgmask has non-zero pixels."""
        mock_sub = MagicMock()
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[0:10, 0:10] = 255  # 100 motion pixels out of 10000 = 0.01
        mock_sub.apply.return_value = mask
        mock_cv2 = MagicMock()
        mock_cv2.createBackgroundSubtractorMOG2.return_value = mock_sub
        mock_cv2.cvtColor.return_value = _gray_frame()

        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            detector = MotionDetector(strategy="background", threshold=0.005)
            score, triggered = detector.detect(_bgr_frame())

        assert score == pytest.approx(100 / 10_000)
        assert triggered == True
