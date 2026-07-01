"""
YOLO object detector wrapper for real-time CCTV analysis.
Uses YOLOv11n (nano) for lightweight detection with ByteTrack tracking.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """A single detected object."""

    label: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2 (normalized 0-1)
    track_id: Optional[int] = None


@dataclass
class TrackedObject:
    """An object tracked across frames."""

    label: str
    track_id: int
    first_seen: float  # timestamp
    last_seen: float
    confidence: float


class YOLODetector:
    """YOLOv11n detector with ByteTrack tracking."""

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cpu",
        enable_tracking: bool = True,
    ):
        self._model = None
        self._model_path = model_path
        self._conf = confidence_threshold
        self._iou = iou_threshold
        self._device = device
        self._enable_tracking = enable_tracking
        self._tracked_objects: Dict[int, TrackedObject] = {}
        self._frame_count = 0
        self._load_model()

    def _load_model(self):
        from ultralytics import YOLO

        self._model = YOLO(self._model_path)
        logger.info("YOLO model loaded: %s (device=%s)", self._model_path, self._device)

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        """Run detection on a frame. Returns list of Detection objects."""
        if self._model is None:
            return []

        self._frame_count += 1
        results = self._model(
            frame_bgr,
            conf=self._conf,
            iou=self._iou,
            device=self._device,
            verbose=False,
        )[0]

        detections: List[Detection] = []
        h, w = frame_bgr.shape[:2]

        if results.boxes is not None:
            for i, box in enumerate(results.boxes):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                det = Detection(
                    label=results.names[int(box.cls[0])],
                    confidence=float(box.conf[0]),
                    bbox=(x1 / w, y1 / h, x2 / w, y2 / h),  # normalize
                )

                if self._enable_tracking and results.boxes.id is not None:
                    track_id = int(results.boxes.id[i])
                    det.track_id = track_id
                    self._update_track(track_id, det)

                detections.append(det)

        return detections

    def _update_track(self, track_id: int, det: Detection):
        """Update tracking state for a detected object."""
        now = time.time()
        if track_id in self._tracked_objects:
            obj = self._tracked_objects[track_id]
            obj.last_seen = now
            obj.confidence = max(obj.confidence, det.confidence)
        else:
            self._tracked_objects[track_id] = TrackedObject(
                label=det.label,
                track_id=track_id,
                first_seen=now,
                last_seen=now,
                confidence=det.confidence,
            )

    def get_active_objects(self, timeout: float = 30.0) -> List[TrackedObject]:
        """Get objects seen within the last *timeout* seconds (default 30)."""
        now = time.time()
        return [
            obj
            for obj in self._tracked_objects.values()
            if now - obj.last_seen < timeout
        ]

    def get_object_summary(self) -> str:
        """Get a text summary of all currently tracked objects."""
        active = self.get_active_objects()
        if not active:
            return "No objects detected"
        counts: Dict[str, int] = {}
        for obj in active:
            counts[obj.label] = counts.get(obj.label, 0) + 1
        return ", ".join(f"{count}x {label}" for label, count in counts.items())
