"""
Tests for DINOv2-based perceptual frame compression (v0.30.0).

Tests the DINOv2FrameCompressor lifecycle and compression logic
using mocked model outputs. The real `available` property checks
for transformers at import time — tests mock this at the right level.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest

from video_analysis.frame_compression import DINOv2FrameCompressor


class TestDINOv2FrameCompressorInit:
    """Tests for compressor initialization."""

    def test_init_defaults(self):
        compressor = DINOv2FrameCompressor()
        assert compressor.model_name == "facebook/dinov2-small"
        assert compressor.threshold == 0.88
        assert compressor.batch_size == 8

    def test_init_custom_model(self):
        compressor = DINOv2FrameCompressor(model_name="facebook/dinov2-base")
        assert compressor.model_name == "facebook/dinov2-base"

    def test_init_custom_device(self):
        compressor = DINOv2FrameCompressor(device="cpu")
        assert compressor.device == "cpu"

    def test_device_default(self):
        compressor = DINOv2FrameCompressor()
        assert isinstance(compressor.device, str)


class TestDINOv2FrameCompressorAvailable:
    """Tests for the `available` property."""

    def test_available_false_when_transformers_missing(self):
        """When transformers can't be imported, available=False."""
        compressor = DINOv2FrameCompressor()
        # Mock at the import level inside the available property
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "transformers":
                raise ImportError("No module named 'transformers'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            assert not compressor.available

    def test_available_true_when_transformers_present(self):
        """When transformers is importable, available=True."""
        compressor = DINOv2FrameCompressor()
        assert compressor._available is None
        # In a test environment with transformers installed, this should be True
        try:
            import transformers  # noqa: F401

            assert compressor.available is True
        except ImportError:
            pass  # transformers not installed — skip

    def test_available_cached(self):
        """available should cache after first check."""
        compressor = DINOv2FrameCompressor()
        assert compressor._available is None
        # First call
        _ = compressor.available
        cached = compressor._available
        # Second call should use cache
        assert compressor.available == cached


class TestDINOv2FrameCompressorCompress:
    """Tests for compress() logic."""

    def test_compress_empty_frames(self):
        """Empty list returns []."""
        compressor = DINOv2FrameCompressor()
        with patch.object(
            DINOv2FrameCompressor,
            "available",
            new_callable=PropertyMock,
            return_value=True,
        ):
            result = compressor.compress([])
        assert result == []

    def test_compress_single_frame(self):
        """Single frame returns [0]."""
        compressor = DINOv2FrameCompressor()
        with patch.object(
            DINOv2FrameCompressor,
            "available",
            new_callable=PropertyMock,
            return_value=True,
        ):
            result = compressor.compress([Path("/tmp/fake_frame.jpg")])
        assert result == [0]

    def test_compress_raises_when_not_available(self):
        compressor = DINOv2FrameCompressor()
        with patch.object(
            DINOv2FrameCompressor,
            "available",
            new_callable=PropertyMock,
            return_value=False,
        ):
            with pytest.raises(RuntimeError, match="DINOv2 is not available"):
                compressor.compress([Path("/tmp/fake.jpg")])

    def test_compress_two_identical_keep_none(self):
        """Two identical frames with threshold=0.99 should keep only first
        since cosine sim (1.0) < 0.99 is False -> second dropped."""
        frames = [Path(f"/tmp/frame_{i}.jpg") for i in range(2)]

        compressor = DINOv2FrameCompressor()
        with (
            patch.object(
                DINOv2FrameCompressor,
                "available",
                new_callable=PropertyMock,
                return_value=True,
            ),
            patch.object(DINOv2FrameCompressor, "_compute_features") as mock_feat,
        ):
            mock_feat.return_value = [
                np.array([1.0, 0.0, 0.0], dtype=np.float32),
                np.array([1.0, 0.0, 0.0], dtype=np.float32),
            ]
            result = compressor.compress(frames, threshold=0.99)
        # Identical features (cosine = 1.0). 1.0 < 0.99 is False -> drop second
        assert result == [0]

    def test_compress_two_different_keep_both(self):
        """Two different frames with mid-threshold: cos ~0 < 0.5 -> keep both."""
        frames = [Path(f"/tmp/frame_{i}.jpg") for i in range(2)]

        compressor = DINOv2FrameCompressor()
        with (
            patch.object(
                DINOv2FrameCompressor,
                "available",
                new_callable=PropertyMock,
                return_value=True,
            ),
            patch.object(DINOv2FrameCompressor, "_compute_features") as mock_feat,
        ):
            mock_feat.return_value = [
                np.array([1.0, 0.0, 0.0], dtype=np.float32),
                np.array([0.0, 1.0, 0.0], dtype=np.float32),
            ]
            result = compressor.compress(frames, threshold=0.5)
        # Very different features (cosine ~0), 0 < 0.5 -> keep both
        assert result == [0, 1]

    def test_compress_three_middle_dropped(self):
        """Three frames: second similar to first (dropped), third different (kept)."""
        frames = [Path(f"/tmp/frame_{i}.jpg") for i in range(3)]

        compressor = DINOv2FrameCompressor()
        with (
            patch.object(
                DINOv2FrameCompressor,
                "available",
                new_callable=PropertyMock,
                return_value=True,
            ),
            patch.object(DINOv2FrameCompressor, "_compute_features") as mock_feat,
        ):
            mock_feat.return_value = [
                np.array([1.0, 0.0, 0.0], dtype=np.float32),
                np.array([1.0, 0.0, 0.0], dtype=np.float32),
                np.array([0.0, 1.0, 0.0], dtype=np.float32),
            ]
            result = compressor.compress(frames, threshold=0.85)
        assert result == [0, 2]

    def test_compress_high_threshold_keeps_more(self):
        """threshold=1.1 means even 100% similar frames are kept (no frame dropped)."""
        frames = [Path(f"/tmp/frame_{i}.jpg") for i in range(3)]

        compressor = DINOv2FrameCompressor()
        with (
            patch.object(
                DINOv2FrameCompressor,
                "available",
                new_callable=PropertyMock,
                return_value=True,
            ),
            patch.object(DINOv2FrameCompressor, "_compute_features") as mock_feat,
        ):
            mock_feat.return_value = [
                np.array([1.0, 0.0, 0.0], dtype=np.float32),
                np.array([1.0, 0.0, 0.0], dtype=np.float32),
                np.array([1.0, 0.0, 0.0], dtype=np.float32),
            ]
            result = compressor.compress(frames, threshold=1.1)
        # 1.0 < 1.1 is True for all -> all kept
        assert result == [0, 1, 2]

    def test_compress_threshold_passed_to_call(self):
        """Per-call threshold should override instance threshold."""
        frames = [Path(f"/tmp/frame_{i}.jpg") for i in range(2)]

        compressor = DINOv2FrameCompressor(threshold=0.99)  # tight by default
        with (
            patch.object(
                DINOv2FrameCompressor,
                "available",
                new_callable=PropertyMock,
                return_value=True,
            ),
            patch.object(DINOv2FrameCompressor, "_compute_features") as mock_feat,
        ):
            mock_feat.return_value = [
                np.array([1.0, 0.0, 0.0], dtype=np.float32),
                np.array([0.0, 1.0, 0.0], dtype=np.float32),
            ]
            # Override with low threshold -> keep all (0 < 0.5)
            result = compressor.compress(frames, threshold=0.5)
        assert result == [0, 1]


class TestDINOv2FrameCompressorUnload:
    """Tests for unload functionality."""

    def test_unload_clears_model(self):
        compressor = DINOv2FrameCompressor()
        compressor._model = MagicMock()
        compressor._processor = MagicMock()
        compressor._available = True

        compressor.unload()

        assert compressor._model is None
        assert compressor._processor is None
        # available should still be True (cached — unload doesn't affect it)
        assert compressor._available is True

    def test_unload_safe_when_no_model(self):
        """unload() should not crash when called before load."""
        compressor = DINOv2FrameCompressor()
        compressor.unload()  # should not raise
        assert compressor._model is None

    def test_unload_twice_safe(self):
        compressor = DINOv2FrameCompressor()
        compressor._model = MagicMock()
        compressor._processor = MagicMock()
        compressor.unload()
        compressor.unload()  # should not raise
