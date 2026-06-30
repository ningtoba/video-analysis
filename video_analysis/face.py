"""
Face detection and recognition module using InsightFace.

Provides face detection (RetinaFace/SCRFD), face recognition (ArcFace),
and cross-video person identity matching via face embeddings.

Optional dependency — import guard ensures graceful fallback when
insightface is not installed.

Usage:
    from video_analysis.face import FaceRecognizer

    recognizer = FaceRecognizer()
    faces = recognizer.detect_faces("frame.jpg")  # detection
    matches = recognizer.match_across_videos(embeddings_a, embeddings_b)  # cross-video
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DetectedFace:
    """A single detected face with landmarks, embedding, and metadata."""

    bbox: List[float]  # [x1, y1, x2, y2] in absolute pixel coordinates
    confidence: float  # detection confidence (0-1)
    landmark: Optional[List[float]] = None  # [x1,y1, x2,y2, x3,y3, x4,y4, x5,y5]
    embedding: Optional[List[float]] = None  # 512-d ArcFace embedding
    face_id: Optional[str] = None  # cluster / identity label, e.g. "PERSON_0"
    age: Optional[int] = None  # estimated age (if gender/age model loaded)
    gender: Optional[str] = None  # "Male" / "Female" (if gender/age model loaded)


@dataclass
class FaceRecognitionResult:
    """Result of face processing for a single frame."""

    frame_timestamp: float
    frame_path: str
    faces: List[DetectedFace] = field(default_factory=list)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Backend constants
# ---------------------------------------------------------------------------

# Default detection model — SCRFD 10G (RetinaFace successor, faster, more accurate)
DEFAULT_DET_MODEL = "buffalo_l"  # includes detection, recognition, and landmark models
DEFAULT_REC_MODEL = "buffalo_l"  # ArcFace W50 (512-d embeddings)

# Minimal face embedding similarity to consider a match (cosine similarity)
DEFAULT_MATCH_THRESHOLD = 0.45  # ~0.45 is standard for ArcFace with buffalo_l

# ---------------------------------------------------------------------------
# FaceRecognizer
# ---------------------------------------------------------------------------


class FaceRecognizer:
    """Face detection and recognition using InsightFace.

    Uses InsightFace's Python API (``insightface`` pip package) with
    the ``buffalo_l`` model pack (SCRFD-10G detection + ArcFace W50
    recognition).

    Lazy-loads on first call — no import-time GPU allocation.

    Attributes:
        enabled: Whether InsightFace is available.
        match_threshold: Cosine similarity threshold for identity matching.
    """

    def __init__(
        self,
        match_threshold: float = DEFAULT_MATCH_THRESHOLD,
        det_model: str = DEFAULT_DET_MODEL,
        rec_model: str = DEFAULT_REC_MODEL,
        providers: Optional[List[str]] = None,
    ):
        self.match_threshold = match_threshold
        self.det_model = det_model
        self.rec_model = rec_model
        self._providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._model = None
        self._available = None  # None = not checked yet
        self._import_error: Optional[str] = None

    # ------------------------------------------------------------------
    # Availability check
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """Check if InsightFace is installed and importable (lazy)."""
        if self._available is None:
            self._check_import()
        return self._available

    def _check_import(self) -> None:
        """Try importing insightface and onnxruntime."""
        try:
            import insightface  # noqa: F401
            import onnxruntime  # noqa: F401

            self._available = True
        except ImportError as exc:
            self._available = False
            self._import_error = str(exc)
            logger.warning(
                "InsightFace not available: %s — face recognition disabled. "
                "Install with: pip install insightface onnxruntime-gpu",
                exc,
            )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self):
        """Lazy-load the InsightFace model."""
        if self._model is not None:
            return
        if not self.available:
            raise RuntimeError(f"InsightFace not available: {self._import_error}")

        try:
            from insightface.app import FaceAnalysis

            self._model = FaceAnalysis(
                name=self.det_model,
                root=str(Path.home() / ".insightface"),
                providers=self._providers,
            )
            self._model.prepare(ctx_id=0, det_size=(640, 640))
            logger.info(
                "InsightFace FaceAnalysis loaded (model=%s, providers=%s)",
                self.det_model,
                self._providers,
            )
        except Exception as exc:
            logger.error("Failed to load InsightFace model: %s", exc)
            self._available = False
            self._import_error = str(exc)
            raise

    # ------------------------------------------------------------------
    # Core detection API
    # ------------------------------------------------------------------

    def detect_faces(
        self,
        frame_path: str,
        extract_embedding: bool = True,
    ) -> List[DetectedFace]:
        """Detect faces in a single frame image.

        Args:
            frame_path: Path to the frame image (JPEG/PNG).
            extract_embedding: Whether to compute face embeddings (needed for
                cross-video matching). Disable for pure detection (faster).

        Returns:
            List of ``DetectedFace`` objects, empty if no faces found.
        """
        self._load_model()

        try:
            img = self._read_image(frame_path)
            if img is None:
                logger.warning("Could not read image: %s", frame_path)
                return []

            raw_faces: list = self._model.get(img)
            results: List[DetectedFace] = []

            for raw in raw_faces:
                bbox_list = raw.bbox.astype(float).tolist() if hasattr(raw, "bbox") else []
                confidence = float(raw.det_score) if hasattr(raw, "det_score") else 0.0
                embedding = (
                    raw.embedding.astype(float).tolist()
                    if (
                        extract_embedding
                        and hasattr(raw, "embedding")
                        and raw.embedding is not None
                    )
                    else None
                )

                landmark = None
                if hasattr(raw, "landmark") and raw.landmark is not None:
                    landmark = raw.landmark.astype(float).flatten().tolist()

                results.append(
                    DetectedFace(
                        bbox=bbox_list,
                        confidence=confidence,
                        landmark=landmark,
                        embedding=embedding,
                        gender=getattr(raw, "gender", None),
                        age=getattr(raw, "age", None),
                    )
                )

            logger.debug("Detected %d face(s) in %s", len(results), Path(frame_path).name)
            return results

        except Exception as exc:
            logger.error("Face detection error on %s: %s", frame_path, exc)
            return []

    def detect_faces_batch(
        self,
        frame_paths: List[str],
        extract_embedding: bool = True,
    ) -> List[FaceRecognitionResult]:
        """Detect faces in multiple frames.

        Args:
            frame_paths: List of paths to frame images.
            extract_embedding: Whether to compute face embeddings.

        Returns:
            List of ``FaceRecognitionResult`` objects, one per input frame.
        """
        results: List[FaceRecognitionResult] = []
        for fp in frame_paths:
            timestamp = self._extract_timestamp(fp)
            try:
                faces = self.detect_faces(fp, extract_embedding=extract_embedding)
                results.append(
                    FaceRecognitionResult(
                        frame_timestamp=timestamp,
                        frame_path=fp,
                        faces=faces,
                    )
                )
            except Exception as exc:
                results.append(
                    FaceRecognitionResult(
                        frame_timestamp=timestamp,
                        frame_path=fp,
                        error=str(exc),
                    )
                )
        return results

    # ------------------------------------------------------------------
    # Cross-video face matching
    # ------------------------------------------------------------------

    def match_faces(
        self,
        query_embedding: List[float],
        gallery_embeddings: List[Tuple[str, List[float]]],
    ) -> List[Tuple[str, float]]:
        """Match a query face embedding against a gallery.

        Args:
            query_embedding: 512-d face embedding vector.
            gallery_embeddings: List of (label, embedding) tuples to match against.

        Returns:
            List of (label, cosine_similarity) sorted by similarity descending.
            Empty if no matches above ``match_threshold``.
        """
        query_np = np.array(query_embedding, dtype=np.float32)
        query_norm = query_np / (np.linalg.norm(query_np) + 1e-12)

        matches: List[Tuple[str, float]] = []
        for label, emb in gallery_embeddings:
            emb_np = np.array(emb, dtype=np.float32)
            emb_norm = emb_np / (np.linalg.norm(emb_np) + 1e-12)
            sim = float(np.dot(query_norm, emb_norm))
            if sim >= self.match_threshold:
                matches.append((label, sim))

        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def cluster_faces(
        self,
        all_embeddings: List[Tuple[str, List[float]]],
        threshold: Optional[float] = None,
    ) -> Dict[str, List[str]]:
        """Cluster face embeddings into identity groups.

        Uses agglomerative clustering with cosine distance and a
        similarity threshold.  Each face is assigned an identity label
        (``PERSON_0``, ``PERSON_1``, …).

        Args:
            all_embeddings: List of (source_id, embedding) where source_id is
                a unique identifier for each face instance (e.g. ``frame_path:face_idx``).
            threshold: Cosine similarity threshold for grouping (default: match_threshold).

        Returns:
            Dict mapping identity labels to lists of source_ids.
        """
        threshold = threshold or self.match_threshold
        if not all_embeddings:
            return {}

        source_ids = [sid for sid, _ in all_embeddings]
        embeddings_np = np.array([emb for _, emb in all_embeddings], dtype=np.float32)
        norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True) + 1e-12
        embeddings_normed = embeddings_np / norms

        # Simple greedy clustering — O(N²) but fine for typical video face counts (<500)
        n = len(embeddings_normed)
        assigned = [False] * n
        clusters: Dict[str, List[int]] = {}
        cluster_idx = 0

        for i in range(n):
            if assigned[i]:
                continue
            label = f"PERSON_{cluster_idx}"
            clusters[label] = [i]
            assigned[i] = True

            for j in range(i + 1, n):
                if assigned[j]:
                    continue
                sim = float(np.dot(embeddings_normed[i], embeddings_normed[j]))
                if sim >= threshold:
                    clusters[label].append(j)
                    assigned[j] = True

            cluster_idx += 1

        # Map indices back to source_ids
        result: Dict[str, List[str]] = {}
        for label, indices in clusters.items():
            result[label] = [source_ids[idx] for idx in indices]

        # Sort by cluster size descending
        sorted_result = dict(sorted(result.items(), key=lambda x: len(x[1]), reverse=True))
        return sorted_result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def cosine_similarity(emb_a: List[float], emb_b: List[float]) -> float:
        """Compute cosine similarity between two embedding vectors."""
        a = np.array(emb_a, dtype=np.float32)
        b = np.array(emb_b, dtype=np.float32)
        a_norm = a / (np.linalg.norm(a) + 1e-12)
        b_norm = b / (np.linalg.norm(b) + 1e-12)
        return float(np.dot(a_norm, b_norm))

    @staticmethod
    def _read_image(path: str) -> Optional[np.ndarray]:
        """Read an image file as a numpy array (BGR for InsightFace)."""
        try:
            from PIL import Image

            pil_img = Image.open(path).convert("RGB")
            return np.array(pil_img)[:, :, ::-1]  # RGB → BGR
        except Exception as exc:
            logger.debug("Failed to read image %s: %s", path, exc)
            return None

    @staticmethod
    def _extract_timestamp(frame_path: str) -> float:
        """Try to extract a float timestamp from the frame filename."""
        name = Path(frame_path).stem
        # Common patterns: "frame_123.45.jpg", "keyframe_000042.jpg" (frame number / fps)
        try:
            parts = name.split("_")
            for part in reversed(parts):
                return float(part)
        except (ValueError, IndexError):
            pass
        return 0.0

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def unload(self):
        """Release GPU memory by dropping the model reference."""
        if self._model is not None:
            logger.info("Unloading InsightFace model from GPU")
            self._model = None
