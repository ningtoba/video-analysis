"""Unit tests for yolo_detector.py — data classes and detector behavior.

Tests the public API surface of the YOLO detection module without loading
a real model or importing ultralytics. Every test in this file is:
  - deterministic (no real time, randomness, or model weights)
  - fast (<1s)
  - isolated (no GPU, no webcam, no network)
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from video_analysis.yolo_detector import Detection, TrackedObject, YOLODetector


# ====================================================================
# Detection dataclass
# ====================================================================


class TestDetection:
    """Detection dataclass: fields, defaults, mutability."""

    def test_fields(self):
        """Detection stores label, confidence, bbox, and track_id."""
        d = Detection(
            label="person",
            confidence=0.95,
            bbox=(0.1, 0.2, 0.8, 0.9),
            track_id=42,
        )
        assert d.label == "person"
        assert d.confidence == 0.95
        assert d.bbox == (0.1, 0.2, 0.8, 0.9)
        assert d.track_id == 42

    def test_track_id_default(self):
        """track_id defaults to None."""
        d = Detection(label="car", confidence=0.80, bbox=(0.0, 0.0, 1.0, 1.0))
        assert d.track_id is None

    def test_bbox_type(self):
        """bbox is a 4-tuple of floats."""
        d = Detection(label="dog", confidence=0.50, bbox=(0.0, 0.0, 0.5, 0.5))
        assert isinstance(d.bbox, tuple)
        assert len(d.bbox) == 4
        assert all(isinstance(v, float) for v in d.bbox)

    def test_mutability(self):
        """Detection is a mutable dataclass — track_id can be set after init."""
        d = Detection(label="bicycle", confidence=0.70, bbox=(0, 0, 1, 1))
        d.track_id = 7
        assert d.track_id == 7

        d.label = "motorbike"
        assert d.label == "motorbike"


# ====================================================================
# TrackedObject dataclass
# ====================================================================


class TestTrackedObject:
    """TrackedObject dataclass: fields and derived timing."""

    def test_fields(self):
        """TrackedObject stores label, track_id, timestamps, and confidence."""
        t = TrackedObject(
            label="person",
            track_id=1,
            first_seen=100.0,
            last_seen=200.0,
            confidence=0.90,
        )
        assert t.label == "person"
        assert t.track_id == 1
        assert t.first_seen == 100.0
        assert t.last_seen == 200.0
        assert t.confidence == 0.90

    def test_duration_derivation(self):
        """Duration = last_seen - first_seen (computed externally)."""
        t = TrackedObject("car", 2, first_seen=500.0, last_seen=523.5, confidence=0.85)
        assert t.last_seen - t.first_seen == 23.5

    def test_zero_duration(self):
        """Single-frame detection yields zero duration."""
        t = TrackedObject("dog", 3, first_seen=1000.0, last_seen=1000.0, confidence=0.60)
        assert t.last_seen - t.first_seen == 0.0


# ====================================================================
# YOLODetector constructor
# ====================================================================


@pytest.fixture
def mock_load():
    """Prevent YOLODetector._load_model from importing ultralights."""
    with patch.object(YOLODetector, "_load_model") as m:
        yield m


class TestYOLODetectorConstructor:
    """Constructor stores params and calls _load_model, without real imports."""

    def test_constructor_defaults(self, mock_load):
        """Defaults are stored and _load_model is called once."""
        detector = YOLODetector()
        assert detector._model_path == "yolo11n.pt"
        assert detector._conf == 0.25
        assert detector._iou == 0.45
        assert detector._device == "cpu"
        assert detector._enable_tracking is True
        assert detector._frame_count == 0
        assert detector._tracked_objects == {}
        mock_load.assert_called_once_with()

    def test_constructor_custom_params(self, mock_load):
        """Custom constructor params are stored correctly."""
        detector = YOLODetector(
            model_path="custom.pt",
            confidence_threshold=0.5,
            iou_threshold=0.6,
            device="cuda",
            enable_tracking=False,
        )
        assert detector._model_path == "custom.pt"
        assert detector._conf == 0.5
        assert detector._iou == 0.6
        assert detector._device == "cuda"
        assert detector._enable_tracking is False

    def test_load_model_is_called_on_init(self, mock_load):
        """_load_model is invoked during __init__."""
        YOLODetector()
        mock_load.assert_called_once()

    def test_constructor_model_stays_none(self, mock_load):
        """Without _load_model, _model stays None."""
        detector = YOLODetector()
        assert detector._model is None


# ====================================================================
# YOLODetector.detect
# ====================================================================


class TestYOLODetectorDetect:
    """detect() returns normalized Detection lists from mocked model output."""

    @pytest.fixture
    def detector(self, mock_load):
        """YOLODetector with _load_model suppressed and a mock model installed."""
        det = YOLODetector()
        det._model = MagicMock()
        return det

    # -- mocks -----------------------------------------------------------------

    @staticmethod
    def _make_mock_result(boxes_data, names, tracking_ids=None):
        """Build a mock ultralytics Results object.

        Parameters
        ----------
        boxes_data : list of dict
            Each entry: {"xyxy": [[x1,y1,x2,y2]], "cls": [c], "conf": [p]}
        names : dict[int, str]
            Class index → label mapping.
        tracking_ids : list[int] | None
            ByteTrack IDs matched 1:1 with boxes_data.
        """
        mock_result = MagicMock()
        mock_result.names = names
        mock_result.boxes = MagicMock()
        mock_result.boxes.id = tracking_ids  # None or list

        boxes = []
        for bd in boxes_data:
            b = MagicMock()
            b.xyxy = np.array(bd["xyxy"], dtype=np.float32)
            b.cls = np.array(bd["cls"], dtype=np.float32)
            b.conf = np.array(bd["conf"], dtype=np.float32)
            boxes.append(b)

        mock_result.boxes.__iter__.return_value = iter(boxes)
        return mock_result

    # -- tests -----------------------------------------------------------------

    def test_detect_returns_list(self, detector):
        """detect() returns List[Detection] when model produces boxes."""
        mock_result = self._make_mock_result(
            boxes_data=[
                {"xyxy": [[10.0, 20.0, 100.0, 200.0]], "cls": [0], "conf": [0.95]},
            ],
            names={0: "person"},
            tracking_ids=None,
        )
        detector._model.return_value = [mock_result]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame)

        assert len(detections) == 1
        d = detections[0]
        assert d.label == "person"
        assert d.confidence == pytest.approx(0.95)
        # bbox is normalized: x/w, y/h
        assert d.bbox == (10.0 / 640, 20.0 / 480, 100.0 / 640, 200.0 / 480)
        assert d.track_id is None

    def test_detect_multiple_objects(self, detector):
        """Multiple objects are returned in order."""
        mock_result = self._make_mock_result(
            boxes_data=[
                {"xyxy": [[0.0, 0.0, 50.0, 100.0]], "cls": [0], "conf": [0.90]},
                {"xyxy": [[60.0, 10.0, 200.0, 150.0]], "cls": [1], "conf": [0.75]},
                {"xyxy": [[30.0, 40.0, 80.0, 90.0]], "cls": [0], "conf": [0.60]},
            ],
            names={0: "person", 1: "car"},
            tracking_ids=None,
        )
        detector._model.return_value = [mock_result]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame)

        assert len(detections) == 3
        assert detections[0].label == "person"
        assert detections[1].label == "car"
        assert detections[2].label == "person"

    def test_detect_empty_when_no_boxes(self, detector):
        """Empty list when model returns no boxes (boxes is None)."""
        mock_result = MagicMock()
        mock_result.boxes = None
        detector._model.return_value = [mock_result]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame)

        assert detections == []

    def test_detect_empty_when_model_none(self, mock_load):
        """Empty list when _model is None (load never happened)."""
        detector = YOLODetector()
        # _model is already None from mock_load
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame)
        assert detections == []

    def test_detect_increments_frame_count(self, detector):
        """Each detect call increments _frame_count."""
        mock_result = self._make_mock_result(
            boxes_data=[
                {"xyxy": [[0.0, 0.0, 1.0, 1.0]], "cls": [0], "conf": [0.50]},
            ],
            names={0: "person"},
            tracking_ids=None,
        )
        detector._model.return_value = [mock_result]
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        assert detector._frame_count == 0
        detector.detect(frame)
        assert detector._frame_count == 1
        detector.detect(frame)
        assert detector._frame_count == 2

    def test_detect_normalizes_bbox_coordinates(self, detector):
        """Bbox coordinates are normalized to 0-1 by dividing by w,h."""
        # Frame is 200x300 — non-square to catch x vs y mix-ups
        mock_result = self._make_mock_result(
            boxes_data=[
                {"xyxy": [[30.0, 40.0, 180.0, 160.0]], "cls": [0], "conf": [0.90]},
            ],
            names={0: "person"},
            tracking_ids=None,
        )
        detector._model.return_value = [mock_result]

        frame = np.zeros((200, 300, 3), dtype=np.uint8)
        detections = detector.detect(frame)

        # h=200, w=300
        x1n, y1n, x2n, y2n = detections[0].bbox
        assert x1n == 30.0 / 300
        assert y1n == 40.0 / 200
        assert x2n == 180.0 / 300
        assert y2n == 160.0 / 200


# ====================================================================
# YOLODetector tracking integration
# ====================================================================


class TestYOLODetectorTracking:
    """Tracking: _update_track and detect integrating with track ID."""

    @pytest.fixture
    def detector(self, mock_load):
        det = YOLODetector()
        det._model = MagicMock()
        return det

    @pytest.fixture
    def mock_time(self):
        with patch("video_analysis.yolo_detector.time.time") as m:
            m.return_value = 1000.0
            yield m

    def test_detect_stores_tracked_objects(self, detector, mock_time):
        """With tracking enabled and IDs present, _tracked_objects is populated."""
        mock_result = MagicMock()
        mock_result.names = {0: "person"}
        mock_result.boxes = MagicMock()
        mock_result.boxes.id = np.array([1, 2], dtype=np.int32)

        box1 = MagicMock()
        box1.xyxy = np.array([[10.0, 20.0, 100.0, 200.0]], dtype=np.float32)
        box1.cls = np.array([0], dtype=np.float32)
        box1.conf = np.array([0.95], dtype=np.float32)

        box2 = MagicMock()
        box2.xyxy = np.array([[50.0, 60.0, 150.0, 180.0]], dtype=np.float32)
        box2.cls = np.array([0], dtype=np.float32)
        box2.conf = np.array([0.80], dtype=np.float32)

        mock_result.boxes.__iter__.return_value = iter([box1, box2])
        detector._model.return_value = [mock_result]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame)

        assert len(detections) == 2
        assert detections[0].track_id == 1
        assert detections[1].track_id == 2

        # Tracked objects stored internally
        assert 1 in detector._tracked_objects
        assert 2 in detector._tracked_objects
        assert detector._tracked_objects[1].track_id == 1
        assert detector._tracked_objects[1].confidence == pytest.approx(0.95)
        assert detector._tracked_objects[2].confidence == pytest.approx(0.80)

    def test_detect_updates_existing_track(self, detector, mock_time):
        """Repeated detection of same track_id updates last_seen and max confidence."""
        mock_result = MagicMock()
        mock_result.names = {0: "person"}
        mock_result.boxes = MagicMock()
        mock_result.boxes.id = np.array([1], dtype=np.int32)

        box = MagicMock()
        box.xyxy = np.array([[10.0, 20.0, 100.0, 200.0]], dtype=np.float32)
        box.cls = np.array([0], dtype=np.float32)
        box.conf = np.array([0.95], dtype=np.float32)

        mock_result.boxes.__iter__.side_effect = lambda: iter([box])
        detector._model.return_value = [mock_result]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # First detection
        mock_time.return_value = 1000.0
        detector.detect(frame)
        obj = detector._tracked_objects[1]
        assert obj.first_seen == 1000.0
        assert obj.last_seen == 1000.0
        assert obj.confidence == pytest.approx(0.95)

        # Second detection — lower confidence
        box.conf = np.array([0.70], dtype=np.float32)
        mock_time.return_value = 1010.0
        detector.detect(frame)
        obj = detector._tracked_objects[1]
        assert obj.first_seen == 1000.0  # unchanged
        assert obj.last_seen == 1010.0  # updated
        assert obj.confidence == pytest.approx(0.95)

        # Third detection — higher confidence
        box.conf = np.array([0.99], dtype=np.float32)
        mock_time.return_value = 1020.0
        detector.detect(frame)
        obj = detector._tracked_objects[1]
        assert obj.confidence == pytest.approx(0.99)  # max updated

    def test_detect_skips_tracking_when_disabled(self, mock_load, mock_time):
        """When enable_tracking=False, _tracked_objects stays empty."""
        detector = YOLODetector(enable_tracking=False)
        detector._model = MagicMock()

        mock_result = MagicMock()
        mock_result.names = {0: "person"}
        mock_result.boxes = MagicMock()
        mock_result.boxes.id = np.array([1], dtype=np.int32)

        box = MagicMock()
        box.xyxy = np.array([[10.0, 20.0, 100.0, 200.0]], dtype=np.float32)
        box.cls = np.array([0], dtype=np.float32)
        box.conf = np.array([0.95], dtype=np.float32)

        mock_result.boxes.__iter__.return_value = iter([box])
        detector._model.return_value = [mock_result]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame)

        # Detection still works
        assert len(detections) == 1
        assert detections[0].track_id is None
        # But no tracking state
        assert detector._tracked_objects == {}

    def test_detect_no_tracking_when_id_none(self, detector, mock_time):
        """When boxes.id is None, tracking is skipped even if enabled."""
        mock_result = MagicMock()
        mock_result.names = {0: "person"}
        mock_result.boxes = MagicMock()
        mock_result.boxes.id = None  # No ByteTrack output

        box = MagicMock()
        box.xyxy = np.array([[10.0, 20.0, 100.0, 200.0]], dtype=np.float32)
        box.cls = np.array([0], dtype=np.float32)
        box.conf = np.array([0.95], dtype=np.float32)

        mock_result.boxes.__iter__.return_value = iter([box])
        detector._model.return_value = [mock_result]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame)

        assert len(detections) == 1
        assert detections[0].track_id is None
        assert detector._tracked_objects == {}


# ====================================================================
# get_active_objects
# ====================================================================


class TestGetActiveObjects:
    """get_active_objects filters by recency."""

    @pytest.fixture
    def detector(self, mock_load):
        return YOLODetector()

    @pytest.fixture
    def mock_time(self):
        with patch("video_analysis.yolo_detector.time.time") as m:
            yield m

    def test_filters_by_timeout_boundary(self, detector, mock_time):
        """Objects seen within timeout are active; older ones are not."""
        mock_time.return_value = 1000.0
        detector._tracked_objects = {
            1: TrackedObject("person", 1, first_seen=900.0, last_seen=995.0, confidence=0.9),
            # last_seen=995, now=1000 => 5s ago => active (timeout=30)
            2: TrackedObject("car", 2, first_seen=800.0, last_seen=960.0, confidence=0.8),
            # last_seen=960, now=1000 => 40s ago => inactive (timeout=30)
            3: TrackedObject("dog", 3, first_seen=950.0, last_seen=999.0, confidence=0.7),
            # last_seen=999, now=1000 => 1s ago => active (timeout=30)
        }

        active = detector.get_active_objects(timeout=30.0)

        assert len(active) == 2
        track_ids = {o.track_id for o in active}
        assert track_ids == {1, 3}

    def test_exactly_at_timeout_boundary(self, detector, mock_time):
        """Object seen exactly timeout seconds ago is included (<, not <=)."""
        mock_time.return_value = 1000.0
        # last_seen = 970, now=1000 => 30s ago, and 30 < 30 is False
        detector._tracked_objects = {
            1: TrackedObject("person", 1, first_seen=900.0, last_seen=970.0, confidence=0.9),
        }

        active = detector.get_active_objects(timeout=30.0)
        # now - last_seen (30) is NOT < timeout (30), so excluded
        assert len(active) == 0

    def test_just_inside_timeout(self, detector, mock_time):
        """Object seen just under timeout seconds ago is included."""
        mock_time.return_value = 1000.0
        detector._tracked_objects = {
            1: TrackedObject("person", 1, first_seen=900.0, last_seen=970.001, confidence=0.9),
        }

        active = detector.get_active_objects(timeout=30.0)
        assert len(active) == 1

    def test_returns_all_when_timeout_is_large(self, detector, mock_time):
        """All objects returned when timeout covers everything."""
        mock_time.return_value = 1000.0
        detector._tracked_objects = {
            1: TrackedObject("a", 1, 0.0, 999.0, 0.5),
            2: TrackedObject("b", 2, 0.0, 500.0, 0.5),
        }
        active = detector.get_active_objects(timeout=10000.0)
        assert len(active) == 2

    def test_returns_empty_when_none_active(self, detector, mock_time):
        """Empty list when no objects within timeout."""
        mock_time.return_value = 1000.0
        detector._tracked_objects = {
            1: TrackedObject("old", 1, 0.0, 500.0, 0.5),
        }
        active = detector.get_active_objects(timeout=30.0)
        assert active == []

    def test_returns_empty_when_no_tracked_objects(self, detector, mock_time):
        """Empty list when _tracked_objects is empty."""
        mock_time.return_value = 1000.0
        active = detector.get_active_objects()
        assert active == []


# ====================================================================
# get_object_summary
# ====================================================================


class TestGetObjectSummary:
    """get_object_summary produces human-readable text from tracked objects."""

    @pytest.fixture
    def detector(self, mock_load):
        return YOLODetector()

    @pytest.fixture
    def mock_time(self):
        with patch("video_analysis.yolo_detector.time.time") as m:
            yield m

    def test_summary_multiple_labels(self, detector, mock_time):
        """Returns counts per label, sorted by label name."""
        mock_time.return_value = 1000.0
        detector._tracked_objects = {
            1: TrackedObject("car", 1, 0.0, 990.0, 0.9),
            2: TrackedObject("car", 2, 0.0, 995.0, 0.8),
            3: TrackedObject("person", 3, 0.0, 998.0, 0.7),
        }
        summary = detector.get_object_summary()
        assert summary == "2x car, 1x person"

    def test_summary_single_object(self, detector, mock_time):
        """Single label with count 1."""
        mock_time.return_value = 1000.0
        detector._tracked_objects = {
            1: TrackedObject("person", 1, 0.0, 999.0, 0.9),
        }
        assert detector.get_object_summary() == "1x person"

    def test_summary_no_active_objects(self, detector, mock_time):
        """Fallback text when no active objects."""
        mock_time.return_value = 1000.0
        detector._tracked_objects = {
            1: TrackedObject("old", 1, 0.0, 500.0, 0.9),
        }
        assert detector.get_object_summary() == "No objects detected"

    def test_summary_empty_tracked_objects(self, detector, mock_time):
        """Fallback text when no tracked objects at all."""
        mock_time.return_value = 1000.0
        assert detector.get_object_summary() == "No objects detected"

    def test_summary_uses_active_objects_only(self, detector, mock_time):
        """Stale objects are excluded from summary."""
        mock_time.return_value = 1000.0
        detector._tracked_objects = {
            1: TrackedObject("person", 1, 0.0, 999.0, 0.9),   # active
            2: TrackedObject("old_car", 2, 0.0, 500.0, 0.8),  # stale
            3: TrackedObject("person", 3, 0.0, 998.0, 0.7),   # active
        }
        summary = detector.get_object_summary()
        assert summary == "2x person"
