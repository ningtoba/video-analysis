"""Tests for face detection and recognition module (InsightFace)."""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from video_analysis.face import (
    FaceRecognizer,
    DetectedFace,
    FaceRecognitionResult,
    DEFAULT_MATCH_THRESHOLD,
)

# ---------------------------------------------------------------------------
# Data class tests
# ---------------------------------------------------------------------------


class TestDetectedFace:
    def test_defaults(self):
        face = DetectedFace(bbox=[10, 20, 100, 200], confidence=0.95)
        assert face.bbox == [10, 20, 100, 200]
        assert face.confidence == 0.95
        assert face.embedding is None
        assert face.face_id is None
        assert face.gender is None
        assert face.age is None
        assert face.landmark is None

    def test_full_init(self):
        face = DetectedFace(
            bbox=[0, 0, 50, 50],
            confidence=0.99,
            landmark=[0.1] * 10,
            embedding=[0.5] * 512,
            face_id="PERSON_0",
            age=30,
            gender="Male",
        )
        assert face.face_id == "PERSON_0"
        assert face.age == 30
        assert face.gender == "Male"
        assert len(face.embedding) == 512


class TestFaceRecognitionResult:
    def test_defaults(self):
        result = FaceRecognitionResult(frame_timestamp=10.0, frame_path="/tmp/f.jpg")
        assert result.frame_timestamp == 10.0
        assert result.frame_path == "/tmp/f.jpg"
        assert result.faces == []
        assert result.error is None

    def test_with_faces(self):
        face = DetectedFace(bbox=[0, 0, 10, 10], confidence=0.9)
        result = FaceRecognitionResult(
            frame_timestamp=15.5,
            frame_path="/tmp/f2.jpg",
            faces=[face],
        )
        assert len(result.faces) == 1
        assert result.error is None


# ---------------------------------------------------------------------------
# FaceRecognizer — unit tests (no InsightFace dependency)
# ---------------------------------------------------------------------------


class TestFaceRecognizer:
    def test_default_threshold(self):
        recognizer = FaceRecognizer()
        assert recognizer.match_threshold == DEFAULT_MATCH_THRESHOLD
        assert recognizer._available is None  # lazy check

    def test_available_returns_bool(self):
        recognizer = FaceRecognizer()
        # Should always return bool (even on first check)
        result = recognizer.available
        assert isinstance(result, bool)

    def test_custom_config(self):
        recognizer = FaceRecognizer(
            match_threshold=0.6,
            det_model="buffalo_sc",
            providers=["CPUExecutionProvider"],
        )
        assert recognizer.match_threshold == 0.6
        assert recognizer.det_model == "buffalo_sc"
        assert recognizer._providers == ["CPUExecutionProvider"]

    def test_cosine_similarity_identical(self):
        emb = [1.0, 0.0, 0.0]
        sim = FaceRecognizer.cosine_similarity(emb, emb)
        assert abs(sim - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal(self):
        sim = FaceRecognizer.cosine_similarity([1.0, 0.0], [0.0, 1.0])
        assert abs(sim) < 1e-6

    def test_cosine_similarity_opposite(self):
        sim = FaceRecognizer.cosine_similarity([1.0, 0.0], [-1.0, 0.0])
        assert abs(sim - (-1.0)) < 1e-6

    def test_cosine_similarity_zero_vector(self):
        sim = FaceRecognizer.cosine_similarity([0.0, 0.0], [1.0, 0.0])
        assert abs(sim) < 1e-6  # division by zero handled

    def test_match_faces_no_matches_below_threshold(self):
        recognizer = FaceRecognizer(match_threshold=0.9)
        # Two nearly orthogonal vectors
        query = [1.0, 0.0]
        gallery = [("person_a", [0.0, 1.0])]
        matches = recognizer.match_faces(query, gallery)
        assert matches == []

    def test_match_faces_above_threshold(self):
        recognizer = FaceRecognizer(match_threshold=0.8)
        query = [1.0, 0.0, 0.0]
        gallery = [
            ("person_a", [0.99, 0.02, 0.01]),
            ("person_b", [0.0, 1.0, 0.0]),  # orthogonal, below threshold
        ]
        matches = recognizer.match_faces(query, gallery)
        # person_a should match (cos ~0.999), person_b should not (cos ~0.0 below 0.8)
        assert len(matches) == 1
        assert matches[0][0] == "person_a"
        assert matches[0][1] > 0.8

    def test_match_faces_sorted_by_similarity(self):
        recognizer = FaceRecognizer(match_threshold=0.0)  # all pass
        query = [1.0, 0.0]
        gallery = [
            ("far", [-1.0, 0.0]),
            ("close", [0.5, 0.5]),
        ]
        matches = recognizer.match_faces(query, gallery)
        assert matches[0][0] == "close"  # higher similarity first

    def test_cluster_faces_empty(self):
        recognizer = FaceRecognizer()
        clusters = recognizer.cluster_faces([])
        assert clusters == {}

    def test_cluster_faces_single(self):
        recognizer = FaceRecognizer(match_threshold=0.5)
        clusters = recognizer.cluster_faces([("id1", [1.0, 0.0])])
        assert len(clusters) == 1
        assert list(clusters.values())[0] == ["id1"]

    def test_cluster_faces_two_clusters(self):
        recognizer = FaceRecognizer(match_threshold=0.7)
        # Two groups: (1,0) and (0,1)
        embeddings = [
            ("a1", [1.0, 0.0]),
            ("a2", [0.99, 0.02]),
            ("b1", [0.0, 1.0]),
            ("b2", [0.01, 0.99]),
        ]
        clusters = recognizer.cluster_faces(embeddings)
        assert len(clusters) >= 2
        # The two largest clusters should each have 2 members
        cluster_sizes = sorted((len(v) for v in clusters.values()), reverse=True)
        assert cluster_sizes[0] >= 2
        assert cluster_sizes[1] >= 2

    def test_extract_timestamp_from_filename(self):
        # Pattern: frame_123.45.jpg
        ts = FaceRecognizer._extract_timestamp("frames/frame_123.45.jpg")
        assert abs(ts - 123.45) < 0.01

    def test_extract_timestamp_invalid(self):
        ts = FaceRecognizer._extract_timestamp("no_timestamp_here.png")
        assert ts == 0.0

    def test_unload_noop_when_not_loaded(self):
        recognizer = FaceRecognizer()
        # Should not raise
        recognizer.unload()

    def test_detect_faces_no_image_returns_empty(self):
        """detect_faces on a non-existent file should return empty, not crash."""
        recognizer = FaceRecognizer()
        # If insightface is not installed, detect_faces should still not crash
        if not recognizer.available:
            # When insightface is unavailable, the method returns empty gracefully
            # because _load_model() logs a warning and raises RuntimeError,
            # but detect_faces calls _load_model and catches exceptions
            pass  # skip — can't test without insightface installed
        else:
            result = recognizer.detect_faces("/tmp/nonexistent_file_xyz.jpg")
            assert result == []

    def test_detect_faces_batch_empty_input(self):
        recognizer = FaceRecognizer()
        results = recognizer.detect_faces_batch([], extract_embedding=True)
        assert results == []

    def test_detect_faces_batch_with_errors(self):
        recognizer = FaceRecognizer()
        results = recognizer.detect_faces_batch(
            ["/tmp/missing1.jpg", "/tmp/missing2.jpg"]
        )
        assert len(results) == 2
        for r in results:
            assert r.error is not None or r.faces == []


# ---------------------------------------------------------------------------
# Config-derived tests
# ---------------------------------------------------------------------------


class TestFaceRecognitionConfig:
    def test_config_defaults_disabled(self):
        """Face recognition should be disabled by default (no extra dep requirement)."""
        # We verify the config default is False by importing it
        from video_analysis.config import Config

        cfg = Config(data_dir="/tmp/test_va_face_cfg")
        assert cfg.face_recognition_enabled is False
        assert cfg.face_detection_model == "buffalo_l"
        assert cfg.face_match_threshold == 0.45
        assert cfg.face_max_faces == 0

        import shutil

        shutil.rmtree("/tmp/test_va_face_cfg", ignore_errors=True)
