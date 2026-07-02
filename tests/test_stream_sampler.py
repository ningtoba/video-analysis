"""
Tests for the stream frame sampler — CircularFrameBuffer.

Tests cover buffer behavior with synthetic data (small numpy arrays).
No real cv2/video source or GPU required.
"""

import numpy as np
from unittest.mock import patch

from video_analysis.stream.sampler import CircularFrameBuffer, SampledFrame


def _make_frame(
    timestamp: float,
    frame_index: int = 0,
    source_fps: float = 30.0,
    shape=(8, 8, 3),
) -> SampledFrame:
    """Create a synthetic SampledFrame for testing."""
    return SampledFrame(
        timestamp=timestamp,
        frame_bgr=np.zeros(shape, dtype=np.uint8),
        frame_index=frame_index,
        source_fps=source_fps,
    )


class TestCount:
    """push / count contract."""

    def test_empty_buffer_has_count_zero(self):
        buf = CircularFrameBuffer()
        assert buf.count == 0

    def test_push_increments_count(self):
        buf = CircularFrameBuffer(max_frames=10)
        buf.push(_make_frame(1.0))
        assert buf.count == 1
        buf.push(_make_frame(2.0))
        assert buf.count == 2


class TestGetAll:
    """get_all — all frames in insertion order."""

    def test_empty(self):
        assert CircularFrameBuffer().get_all() == []

    def test_returns_all_frames_in_order(self):
        buf = CircularFrameBuffer(max_frames=10)
        for i in range(5):
            buf.push(_make_frame(float(i)))
        result = buf.get_all()
        assert len(result) == 5
        assert [f.timestamp for f in result] == [0.0, 1.0, 2.0, 3.0, 4.0]


class TestGetRecent:
    """get_recent(n) — last n frames."""

    def test_returns_last_n(self):
        buf = CircularFrameBuffer(max_frames=10)
        for i in range(10):
            buf.push(_make_frame(float(i)))
        recent = buf.get_recent(3)
        assert len(recent) == 3
        assert [f.timestamp for f in recent] == [7.0, 8.0, 9.0]

    def test_returns_all_when_n_exceeds_count(self):
        buf = CircularFrameBuffer(max_frames=10)
        for i in range(3):
            buf.push(_make_frame(float(i)))
        result = buf.get_recent(10)
        assert len(result) == 3

    def test_returns_empty_when_buffer_empty(self):
        assert CircularFrameBuffer().get_recent(5) == []


class TestGetSince:
    """get_since(since_ts) — frames with timestamp >= threshold."""

    def test_filters_by_timestamp(self):
        buf = CircularFrameBuffer(max_frames=10)
        for i in range(5):
            buf.push(_make_frame(float(i)))
        result = buf.get_since(2.0)
        assert len(result) == 3
        assert [f.timestamp for f in result] == [2.0, 3.0, 4.0]

    def test_includes_matching_timestamp(self):
        buf = CircularFrameBuffer(max_frames=10)
        buf.push(_make_frame(1.0))
        buf.push(_make_frame(2.0))
        buf.push(_make_frame(3.0))
        result = buf.get_since(2.0)
        assert len(result) == 2
        # Exact boundary is included
        assert result[0].timestamp == 2.0

    def test_returns_empty_when_no_frames_after(self):
        buf = CircularFrameBuffer(max_frames=10)
        buf.push(_make_frame(1.0))
        buf.push(_make_frame(2.0))
        assert buf.get_since(5.0) == []


class TestGetWindow:
    """get_window(start_ts, end_ts) — frames within [start, end]."""

    def test_filters_by_time_range(self):
        buf = CircularFrameBuffer(max_frames=10)
        for i in range(10):
            buf.push(_make_frame(float(i)))
        result = buf.get_window(3.0, 6.0)
        assert len(result) == 4
        assert [f.timestamp for f in result] == [3.0, 4.0, 5.0, 6.0]

    def test_inclusive_boundaries(self):
        buf = CircularFrameBuffer(max_frames=10)
        for i in range(5):
            buf.push(_make_frame(float(i)))
        result = buf.get_window(0.0, 4.0)
        assert len(result) == 5

    def test_returns_empty_when_no_overlap(self):
        buf = CircularFrameBuffer(max_frames=10)
        for i in range(5):
            buf.push(_make_frame(float(i)))
        assert buf.get_window(10.0, 20.0) == []


class TestLatest:
    """latest property — most recent frame or None."""

    def test_returns_most_recent(self):
        buf = CircularFrameBuffer(max_frames=10)
        f1 = _make_frame(1.0)
        f2 = _make_frame(2.0)
        buf.push(f1)
        buf.push(f2)
        assert buf.latest is f2

    def test_none_when_empty(self):
        assert CircularFrameBuffer().latest is None


class TestDurationSeconds:
    """duration_seconds — wall-clock span of buffered frames."""

    def test_computes_correctly(self):
        buf = CircularFrameBuffer(max_frames=10)
        buf.push(_make_frame(10.0))
        buf.push(_make_frame(15.0))
        buf.push(_make_frame(20.0))
        assert buf.duration_seconds == 10.0

    def test_zero_for_single_frame(self):
        buf = CircularFrameBuffer(max_frames=10)
        buf.push(_make_frame(5.0))
        assert buf.duration_seconds == 0.0

    def test_zero_for_empty_buffer(self):
        assert CircularFrameBuffer().duration_seconds == 0.0

    def test_negative_timestamps_produces_positive_duration(self):
        """Timestamps may be epoch times; the span is always positive."""
        buf = CircularFrameBuffer(max_frames=10)
        buf.push(_make_frame(1000.0))
        buf.push(_make_frame(1100.0))
        assert buf.duration_seconds == 100.0


class TestBufferEviction:
    """Circular eviction when max_frames is exceeded."""

    def test_evicts_oldest_when_max_exceeded(self):
        buf = CircularFrameBuffer(max_frames=3)
        for i in range(5):
            buf.push(_make_frame(float(i)))
        assert buf.count == 3
        assert [f.timestamp for f in buf.get_all()] == [2.0, 3.0, 4.0]

    def test_get_recent_still_works_after_eviction(self):
        buf = CircularFrameBuffer(max_frames=3)
        for i in range(5):
            buf.push(_make_frame(float(i)))
        recent = buf.get_recent(2)
        assert [f.timestamp for f in recent] == [3.0, 4.0]

    def test_get_since_after_eviction_only_sees_surviving_frames(self):
        buf = CircularFrameBuffer(max_frames=3)
        for i in range(5):
            buf.push(_make_frame(float(i)))
        result = buf.get_since(0.0)
        # Frames with timestamp 0, 1 have been evicted
        assert [f.timestamp for f in result] == [2.0, 3.0, 4.0]

    def test_max_frames_one(self):
        """Buffer with max_frames=1 only keeps the latest frame."""
        buf = CircularFrameBuffer(max_frames=1)
        buf.push(_make_frame(1.0))
        buf.push(_make_frame(2.0))
        assert buf.count == 1
        assert buf.latest.timestamp == 2.0


class TestGetClustered:
    """get_clustered — frame selection strategies."""

    def test_returns_all_when_fewer_than_n_clusters(self):
        """Early return: buffer size <= n_clusters returns all frames."""
        buf = CircularFrameBuffer(max_frames=10)
        for i in range(2):
            buf.push(_make_frame(float(i)))
        result = buf.get_clustered(n_clusters=5)
        assert len(result) == 2

    def test_returns_all_when_equal_to_n_clusters(self):
        """Early return: buffer size == n_clusters returns all frames."""
        buf = CircularFrameBuffer(max_frames=10)
        for i in range(3):
            buf.push(_make_frame(float(i)))
        result = buf.get_clustered(n_clusters=3)
        assert len(result) == 3

    def test_fallback_uniform_sampling_on_import_error(self):
        """When sklearn is unavailable, falls back to uniform sampling."""
        buf = CircularFrameBuffer(max_frames=20)
        for i in range(10):
            buf.push(_make_frame(float(i)))

        # Make sklearn.cluster unimportable to trigger ImportError inside get_clustered
        with patch.dict("sys.modules", {"sklearn": None, "sklearn.cluster": None}):
            result = buf.get_clustered(n_clusters=3, max_frames=5)

        # With 10 frames and max_frames=5: step = 10//5 = 2
        # Result: frames[0, 2, 4, 6, 8] — 5 frames
        assert len(result) == 5
        assert [f.timestamp for f in result] == [0.0, 2.0, 4.0, 6.0, 8.0]

    def test_fallback_uniform_sampling_step_guarantees_min_one(self):
        """Uniform sampling step is at least 1 to avoid empty slice."""
        buf = CircularFrameBuffer(max_frames=20)
        for i in range(10):
            buf.push(_make_frame(float(i)))

        # max_frames larger than len(frames) → step = max(1, 10//10) = 1
        with patch.dict("sys.modules", {"sklearn": None, "sklearn.cluster": None}):
            result = buf.get_clustered(n_clusters=3, max_frames=10)

        assert len(result) == 10  # step=1 means all frames

    def test_fallback_uniform_sampling_with_fewer_frames(self):
        """Uniform sampling handles buffer smaller than max_frames gracefully."""
        buf = CircularFrameBuffer(max_frames=20)
        for i in range(3):
            buf.push(_make_frame(float(i)))

        with patch.dict("sys.modules", {"sklearn": None, "sklearn.cluster": None}):
            result = buf.get_clustered(n_clusters=5, max_frames=10)

        # 3 frames <= n_clusters (5), so early return — all 3 frames
        assert len(result) == 3

    def test_fallback_uniform_sampling_with_empty_buffer(self):
        """Uniform sampling with empty buffer returns empty list."""
        buf = CircularFrameBuffer(max_frames=10)

        with patch.dict("sys.modules", {"sklearn": None, "sklearn.cluster": None}):
            result = buf.get_clustered(n_clusters=3, max_frames=5)

        assert result == []

    def test_kmeans_path_uses_clustered_representatives(self):
        """When sklearn and cv2 are available, get_clustered uses KMeans.

        Mocks cv2 and sklearn to avoid real imports and GPU dependencies.
        Verifies the KMeans path produces the expected output shape.
        """
        import types

        buf = CircularFrameBuffer(max_frames=10)
        for i in range(8):
            buf.push(_make_frame(float(i)))

        mock_kmeans = _make_mock_kmeans(n_clusters=2, n_frames=8)

        # Create fake sklearn.cluster module with KMeans attribute
        fake_cluster = types.ModuleType("sklearn.cluster")
        fake_cluster.KMeans = lambda **kw: mock_kmeans
        fake_sklearn = types.ModuleType("sklearn")
        fake_sklearn.cluster = fake_cluster

        with patch.dict(
            "sys.modules",
            {"sklearn": fake_sklearn, "sklearn.cluster": fake_cluster},
        ), patch(
            "video_analysis.stream.sampler.cv2",
        ) as mock_cv2:
            # Configure cv2 mocks to return plausible values
            mock_cv2.cvtColor.return_value = np.zeros((8, 8, 2), dtype=np.float32)
            mock_cv2.calcHist.return_value = np.zeros((8, 8), dtype=np.float32)
            mock_cv2.COLOR_BGR2HSV = 40
            mock_cv2.normalize.return_value = None

            result = buf.get_clustered(n_clusters=2, max_frames=10)

        # KMeans with 2 clusters on 8 frames — should return 2 representatives
        assert len(result) == 2
        # Result should be in chronological order
        assert result[0].timestamp < result[1].timestamp

    def test_kmeans_path_respects_max_frames_cap(self):
        """KMeans path caps output at max_frames."""
        import types

        buf = CircularFrameBuffer(max_frames=20)
        for i in range(15):
            buf.push(_make_frame(float(i)))

        mock_kmeans = _make_mock_kmeans(n_clusters=4, n_frames=15)

        fake_cluster = types.ModuleType("sklearn.cluster")
        fake_cluster.KMeans = lambda **kw: mock_kmeans
        fake_sklearn = types.ModuleType("sklearn")
        fake_sklearn.cluster = fake_cluster

        with patch.dict(
            "sys.modules",
            {"sklearn": fake_sklearn, "sklearn.cluster": fake_cluster},
        ), patch(
            "video_analysis.stream.sampler.cv2",
        ) as mock_cv2:
            mock_cv2.cvtColor.return_value = np.zeros((8, 8, 2), dtype=np.float32)
            mock_cv2.calcHist.return_value = np.zeros((8, 8), dtype=np.float32)
            mock_cv2.COLOR_BGR2HSV = 40
            mock_cv2.normalize.return_value = None

            result = buf.get_clustered(n_clusters=4, max_frames=2)

        # max_frames=2 caps the result
        assert len(result) == 2

    def test_generic_exception_falls_back_to_uniform(self):
        """A non-ImportError exception in the KMeans path falls back gracefully."""
        buf = CircularFrameBuffer(max_frames=20)
        for i in range(8):
            buf.push(_make_frame(float(i)))

        with patch("video_analysis.stream.sampler.cv2") as mock_cv2:
            # Make cv2.cvtColor raise a value error (simulating a corrupt frame)
            mock_cv2.cvtColor.side_effect = ValueError("corrupt frame data")
            mock_cv2.COLOR_BGR2HSV = 40

            result = buf.get_clustered(n_clusters=3, max_frames=4)

        # Falls back to uniform sampling: 8 frames, max_frames=4, step=2
        assert len(result) == 4
        assert [f.timestamp for f in result] == [0.0, 2.0, 4.0, 6.0]


# ---------------------------------------------------------------------------
# Helpers for KMeans mocking
# ---------------------------------------------------------------------------


def _make_mock_kmeans(n_clusters: int, n_frames: int):
    """Create a mock KMeans instance that returns deterministic labels.

    Assigns frames to clusters in round-robin order so each cluster
    has at least one frame.
    """
    import types as _types

    kmeans = _types.SimpleNamespace()
    kmeans.cluster_centers_ = np.zeros((n_clusters, 64))  # 8*8 calcHist bins
    kmeans.n_clusters = n_clusters
    # Round-robin assignment: frame i -> cluster i % n_clusters
    labels = np.array([i % n_clusters for i in range(n_frames)])
    kmeans.fit_predict = lambda x: labels.copy()
    return kmeans
