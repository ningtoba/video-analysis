"""Per-evidence confidence scoring framework for the video understanding agent.

Inspired by Robust-TO (arXiv:2606.26904), this module adds frame-level
trustworthiness assessment and evidence weighting to the existing video agent.

Components:
    - FrameQualityScorer: Assesses individual frame quality/trustworthiness
    - EvidenceTrustScorer: Scores evidence from agent tools
    - EvidenceWeighter: Three-tier evidence weighting (high/medium/low)
    - RobustAgentFrame: Wrapper integrating with the existing agent

Usage:
    from video_analysis.agent_confidence import RobustAgentFrame

    agent = VideoUnderstandingAgent(...)
    robust = RobustAgentFrame(agent, config)
    result = robust.query_with_trust("What objects are visible at 2:30?")
    print(robust.format_confidence_report(result))
"""

import logging
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from video_analysis.agent import (
    AgentQueryResult,
    AgentToolResult,
    VideoUnderstandingAgent,
)
from video_analysis.config import Config
from video_analysis.rag import RetrievedChunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Frame Quality / Trustworthiness Scorer
# ---------------------------------------------------------------------------


class FrameQualityScorer:
    """Assess individual frame quality and trustworthiness.

    Produces a normalized trustworthiness score (0.0–1.0) by combining
    signal-quality metrics from a single numpy frame array (BGR or RGB).
    """

    @staticmethod
    def score_frame(frame: np.ndarray) -> Dict[str, float]:
        """Score a single frame on multiple quality dimensions.

        Args:
            frame: Numpy array of shape (H, W, C) in BGR or RGB order.

        Returns:
            Dict with keys:
                - blur_score: Laplacian variance (higher = sharper)
                - brightness_score: Mean pixel value (0–255)
                - motion_score: Placeholder (0.0) — use score_frames_batch
                - occlusion_score: Edge-density ratio (lower = potential occlusion)
                - trustworthiness: Combined 0.0–1.0 score
        """
        if frame is None:
            return {
                "blur_score": 0.0,
                "brightness_score": 0.0,
                "motion_score": 0.0,
                "occlusion_score": 0.0,
                "trustworthiness": 0.0,
            }

        gray = FrameQualityScorer._to_gray(frame)

        # --- Blur: Laplacian variance -------------------------------------------
        laplacian_var = FrameQualityScorer._laplacian_variance(gray)
        blur_score = laplacian_var

        # --- Brightness: mean pixel value (0–255) --------------------------------
        mean_brightness = float(np.mean(gray))
        brightness_score = mean_brightness

        # --- Motion: place at 0.0 (batch method populates this) -------------------
        motion_score = 0.0

        # --- Occlusion: edge-density ratio ---------------------------------------
        occlusion_score = FrameQualityScorer._edge_density_ratio(gray)

        # --- Overall trustworthiness (0.0–1.0) -----------------------------------
        trustworthiness = FrameQualityScorer._combine_trust(
            laplacian_var, mean_brightness, motion_score, occlusion_score
        )

        return {
            "blur_score": float(blur_score),
            "brightness_score": float(brightness_score),
            "motion_score": float(motion_score),
            "occlusion_score": float(occlusion_score),
            "trustworthiness": float(trustworthiness),
        }

    @staticmethod
    def score_frames_batch(frames: List[np.ndarray]) -> List[Dict[str, float]]:
        """Score a sequence of frames, including motion between consecutive frames.

        The first frame in the sequence receives a motion_score of 0.0; subsequent
        frames are scored against their immediate predecessor.

        Args:
            frames: List of numpy frame arrays in temporal order.

        Returns:
            List of score dicts, one per frame, with motion_score populated.
        """
        if not frames:
            return []

        results: List[Dict[str, float]] = []
        prev_gray: Optional[np.ndarray] = None

        for frame in frames:
            gray = FrameQualityScorer._to_gray(frame)
            laplacian_var = FrameQualityScorer._laplacian_variance(gray)
            mean_brightness = float(np.mean(gray))
            occlusion_score = FrameQualityScorer._edge_density_ratio(gray)

            # Motion: normalized frame-difference magnitude
            motion_score = 0.0
            if prev_gray is not None and gray.shape == prev_gray.shape:
                import cv2

                diff = cv2.absdiff(gray, prev_gray)
                motion_score = float(np.mean(diff)) / 255.0  # 0.0–1.0

            trustworthiness = FrameQualityScorer._combine_trust(
                laplacian_var, mean_brightness, motion_score, occlusion_score
            )

            results.append(
                {
                    "blur_score": float(laplacian_var),
                    "brightness_score": float(mean_brightness),
                    "motion_score": float(motion_score),
                    "occlusion_score": float(occlusion_score),
                    "trustworthiness": float(trustworthiness),
                }
            )

            prev_gray = gray

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        """Convert a BGR or RGB frame to grayscale."""
        import cv2

        if frame.ndim == 3 and frame.shape[2] >= 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if frame.ndim == 2:
            return frame
        return frame  # fallback, unlikely

    @staticmethod
    def _laplacian_variance(gray: np.ndarray) -> float:
        """Compute Laplacian variance as a blur/sharpness indicator."""
        import cv2

        lap = cv2.Laplacian(gray, cv2.CV_64F)
        return float(lap.var())

    @staticmethod
    def _edge_density_ratio(gray: np.ndarray) -> float:
        """Ratio of edge pixels to total pixels via Canny edge detection.

        Lower values can indicate occlusion or a featureless scene.
        Returns a value in (0.0, 1.0), clamped.
        """
        import cv2

        edges = cv2.Canny(gray, 50, 150)
        total_pixels = max(gray.size, 1)
        edge_pixels = int(np.count_nonzero(edges))
        ratio = edge_pixels / total_pixels
        return float(min(ratio, 1.0))

    @staticmethod
    def _combine_trust(
        laplacian_var: float,
        mean_brightness: float,
        motion_score: float,
        occlusion_score: float,
    ) -> float:
        """Combine quality metrics into a single trustworthiness score (0.0–1.0).

        Heuristics calibrated for typical video frames:
          - Blur: Laplacian variance > 100 is sharp; < 20 is very blurry.
          - Brightness: 30–225 is reasonable; extremes reduce trust.
          - Motion: 0.0 (static) is fine; very high motion (>0.6) reduces frame clarity.
          - Occlusion: edge ratio < 0.02 may indicate occlusion/featureless frame.
        """
        # Blur component: map Laplacian variance to [0, 1]
        # ~100+ => 1.0, ~0 => 0.0
        blur_norm = min(laplacian_var / 100.0, 1.0)

        # Brightness component: Gaussian-style penalty at extremes
        # Optimal ~120, penalty outside [30, 225]
        if mean_brightness < 30.0:
            brightness_norm = max(mean_brightness / 30.0, 0.0)
        elif mean_brightness > 225.0:
            brightness_norm = max((255.0 - mean_brightness) / 30.0, 0.0)
        else:
            brightness_norm = 1.0

        # Motion component: high motion (>0.6) reduces frame clarity
        motion_penalty = 1.0 - max(0.0, (motion_score - 0.6) / 0.4)
        motion_norm = max(min(motion_penalty, 1.0), 0.0)

        # Occlusion component: very low edge ratio (<0.02) may indicate occlusion
        if occlusion_score < 0.02:
            occlusion_norm = occlusion_score / 0.02
        else:
            occlusion_norm = min(occlusion_score / 0.15, 1.0)  # cap at 0.15

        # Weighted combination (weights sum to 1.0)
        w_blur = 0.40
        w_brightness = 0.20
        w_motion = 0.25
        w_occlusion = 0.15

        combined = (
            w_blur * blur_norm
            + w_brightness * brightness_norm
            + w_motion * motion_norm
            + w_occlusion * occlusion_norm
        )

        return max(0.0, min(combined, 1.0))


# ---------------------------------------------------------------------------
# Evidence Trust Scorer
# ---------------------------------------------------------------------------


class EvidenceTrustScorer:
    """Score evidence produced by agent tools based on source quality."""

    @staticmethod
    def score_rag_chunk(chunk: RetrievedChunk) -> Dict[str, Any]:
        """Score a single RAG-retrieved chunk for trustworthiness.

        Factors:
          - chunk.score: retrieval relevance score (0–1 range expected)
          - chunk_type: semantic precision of the source
          - temporal proximity: chunks with timestamps are more verifiable

        Args:
            chunk: A RetrievedChunk from the RAG index.

        Returns:
            Dict with keys: source_confidence, adjusted_score, factors.
        """
        base_score = max(0.0, min(chunk.score, 1.0)) if chunk.score else 0.5

        # Chunk-type multiplier: structured/grounded sources are more reliable
        type_multipliers = {
            "transcript": 1.0,
            "scene": 0.95,
            "frame": 0.90,
            "fixed_60s": 0.85,
            "sliding_30s": 0.85,
        }
        type_mult = type_multipliers.get(chunk.chunk_type or "scene", 0.80)

        # Temporal bonus: has a timestamp → more verifiable
        temporal_bonus = 1.1 if chunk.timestamp is not None else 1.0

        adjusted = base_score * type_mult * temporal_bonus
        source_confidence = max(0.0, min(adjusted, 1.0))

        return {
            "source_confidence": source_confidence,
            "adjusted_score": adjusted,
            "factors": {
                "base_score": base_score,
                "type_multiplier": type_mult,
                "temporal_bonus": temporal_bonus,
                "chunk_type": chunk.chunk_type,
            },
        }

    @staticmethod
    def score_detection(
        detections: List[Dict[str, Any]],
        frame_quality: Dict[str, float],
    ) -> Dict[str, Any]:
        """Adjust detection confidences by frame trustworthiness.

        Each detection dict should have a 'confidence' key (0–1).
        frame_quality is the output of FrameQualityScorer.score_frame.

        Returns:
            Dict with: adjusted_detections, mean_confidence, frame_quality_factor.
        """
        frame_trust = frame_quality.get("trustworthiness", 0.5)
        adjusted: List[Dict[str, Any]] = []

        for det in detections:
            raw_conf = det.get("confidence", 0.5)
            adjusted_conf = raw_conf * frame_trust
            adjusted.append(
                {
                    "label": det.get("label", "unknown"),
                    "raw_confidence": raw_conf,
                    "adjusted_confidence": round(adjusted_conf, 4),
                    "frame_trust_factor": frame_trust,
                }
            )

        mean_raw = statistics.mean(d["raw_confidence"] for d in adjusted) if adjusted else 0.0
        mean_adj = statistics.mean(d["adjusted_confidence"] for d in adjusted) if adjusted else 0.0

        return {
            "adjusted_detections": adjusted,
            "mean_raw_confidence": round(mean_raw, 4),
            "mean_adjusted_confidence": round(mean_adj, 4),
            "frame_quality_factor": frame_trust,
            "num_detections": len(adjusted),
        }

    @staticmethod
    def score_transcript_segment(
        segment: Dict[str, Any],
        confidence: float = 1.0,
    ) -> Dict[str, Any]:
        """Score a transcript segment with speaker overlap adjustment.

        Args:
            segment: Dict with keys like 'text', 'start', 'end', 'speaker',
                     'confidence' (optional).
            confidence:   Override or base confidence (default 1.0).

        Returns:
            Dict with: text_confidence, adjusted_confidence, speaker_overlap_penalty.
        """
        base_conf = segment.get("confidence", confidence)
        base_conf = max(0.0, min(base_conf, 1.0))

        # Speaker overlap penalty: if multiple speakers near the same time,
        # transcription may be less reliable.
        speaker = segment.get("speaker", "")
        overlap_penalty = 1.0

        # If metadata has an explicit overlap flag, apply penalty
        metadata_overlap = segment.get("metadata", {}).get("speaker_overlap", False) or segment.get(
            "speaker_overlap", False
        )
        if metadata_overlap:
            overlap_penalty = 0.85

        # If no speaker identified, slight penalty (diarization uncertainty)
        if not speaker or speaker in ("UNKNOWN", ""):
            overlap_penalty *= 0.95

        adjusted = base_conf * overlap_penalty

        return {
            "text_confidence": round(base_conf, 4),
            "adjusted_confidence": round(max(0.0, min(adjusted, 1.0)), 4),
            "speaker_overlap_penalty": round(overlap_penalty, 4),
            "speaker": speaker or "unknown",
        }

    @staticmethod
    def score_ocr_result(
        ocr_data: List,
        frame_quality: Dict[str, float],
    ) -> Dict[str, Any]:
        """Score OCR confidence adjusted by frame blur.

        Args:
            ocr_data:   List of OCR text results, each being a tuple
                        (bbox, (text, confidence)) as returned by PaddleOCR.
            frame_quality: Output of FrameQualityScorer.score_frame.

        Returns:
            Dict with: adjusted_texts, mean_ocr_confidence, blur_penalty.
        """
        blur_score = frame_quality.get("blur_score", 0.0)
        frame_trust = frame_quality.get("trustworthiness", 0.5)

        # Blur penalty: Laplacian variance < 50 degrades OCR quality
        if blur_score < 20.0:
            blur_penalty = max(blur_score / 20.0, 0.05)
        elif blur_score < 50.0:
            blur_penalty = 0.5 + 0.5 * ((blur_score - 20.0) / 30.0)
        else:
            blur_penalty = 1.0

        adjusted_texts: List[Dict[str, Any]] = []
        confidences: List[float] = []

        for entry in ocr_data:
            if entry is None:
                continue
            # PaddleOCR returns list of [bbox, (text, confidence)]
            for line in entry:
                if line is None or len(line) < 2:
                    continue
                _, (text, raw_conf) = line[0], line[1]
                adjusted_conf = raw_conf * blur_penalty * frame_trust
                confidences.append(raw_conf)
                adjusted_texts.append(
                    {
                        "text": text,
                        "raw_confidence": round(raw_conf, 4),
                        "adjusted_confidence": round(adjusted_conf, 4),
                        "blur_penalty": round(blur_penalty, 4),
                        "frame_trust_factor": frame_trust,
                    }
                )

        mean_ocr_conf = statistics.mean(confidences) if confidences else 0.0

        return {
            "adjusted_texts": adjusted_texts,
            "mean_ocr_confidence": round(mean_ocr_conf, 4),
            "mean_adjusted_confidence": round(
                (
                    statistics.mean([t["adjusted_confidence"] for t in adjusted_texts])
                    if adjusted_texts
                    else 0.0
                ),
                4,
            ),
            "blur_penalty": round(blur_penalty, 4),
            "frame_trust_factor": frame_trust,
        }

    @staticmethod
    def score_mllm_response(
        response: str,
        frame_quality: float,
        num_frames: int,
    ) -> Dict[str, Any]:
        """Score MLLM response confidence based on visual quality and frame count.

        Args:
            response:      Raw text response from the MLLM.
            frame_quality: Combined trustworthiness (0.0–1.0) of the frames used.
            num_frames:    Number of frames provided to the MLLM.

        Returns:
            Dict with: mllm_confidence, quality_factor, response_length_factor.
        """
        # Quality factor = frame trustworthiness
        quality_factor = max(0.0, min(frame_quality, 1.0))

        # Length factor: very short/long responses may indicate issues
        resp_len = len(response.strip())
        if resp_len < 10:
            length_factor = 0.3
        elif resp_len < 50:
            length_factor = 0.6
        elif resp_len > 5000:
            length_factor = 0.85  # very long may be repetitive
        else:
            length_factor = 0.95

        # Frame-count factor: more frames = more context
        frame_factor = min(num_frames / 8.0, 1.0)

        mllm_confidence = quality_factor * length_factor * (0.6 + 0.4 * frame_factor)

        return {
            "mllm_confidence": round(max(0.0, min(mllm_confidence, 1.0)), 4),
            "quality_factor": round(quality_factor, 4),
            "response_length_factor": round(length_factor, 4),
            "frame_factor": round(frame_factor, 4),
            "num_frames": num_frames,
        }


# ---------------------------------------------------------------------------
# Evidence Weighter
# ---------------------------------------------------------------------------


class EvidenceWeighter:
    """Three-tier evidence weighting (high/medium/low) with consensus detection."""

    HIGH_THRESHOLD: float = 0.8
    MEDIUM_THRESHOLD: float = 0.5

    @staticmethod
    def tier(score: float) -> str:
        """Classify a confidence score into a discrete tier.

        Args:
            score: Confidence value in [0.0, 1.0].

        Returns:
            "high" if score >= 0.8,
            "medium" if score >= 0.5 and < 0.8,
            "low" otherwise.
        """
        if score >= EvidenceWeighter.HIGH_THRESHOLD:
            return "high"
        if score >= EvidenceWeighter.MEDIUM_THRESHOLD:
            return "medium"
        return "low"

    @staticmethod
    def weighted_combine(evidences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Weighted average of evidence with confidence propagation.

        Each evidence dict must contain a 'confidence' key (float in 0–1).
        An optional 'weight' key (float) can override the default equal weight.
        If no 'weight' is provided, equal weight is assumed.

        Also propagates per-source metadata into the combined result.

        Args:
            evidences: List of evidence dicts, each with 'confidence' and
                       optionally 'weight', 'source', 'label'.

        Returns:
            Dict with:
                combined_confidence: Weighted average confidence.
                tiers:               List of tier strings.
                unweighted_mean:     Simple mean of confidences.
                num_sources:         Count of evidence sources.
                source_tiers:        Per-source tier classification.
        """
        if not evidences:
            return {
                "combined_confidence": 0.0,
                "tiers": [],
                "unweighted_mean": 0.0,
                "num_sources": 0,
                "source_tiers": {},
            }

        total_weight = 0.0
        weighted_sum = 0.0
        confidences: List[float] = []
        source_tiers: Dict[str, str] = {}
        tiers: List[str] = []

        for i, ev in enumerate(evidences):
            conf = max(0.0, min(float(ev.get("confidence", 0.0)), 1.0))
            weight = float(ev.get("weight", 1.0))
            source = ev.get("source", f"source_{i}")
            label = ev.get("label", "")

            confidences.append(conf)
            weighted_sum += conf * weight
            total_weight += weight

            tier_label = EvidenceWeighter.tier(conf)
            tiers.append(tier_label)
            key = f"{source}:{label}" if label else source
            source_tiers[key] = tier_label

        combined = weighted_sum / max(total_weight, 1e-9)
        unweighted_mean = statistics.mean(confidences) if confidences else 0.0

        return {
            "combined_confidence": round(max(0.0, min(combined, 1.0)), 4),
            "tiers": tiers,
            "unweighted_mean": round(unweighted_mean, 4),
            "num_sources": len(evidences),
            "source_tiers": source_tiers,
        }

    @staticmethod
    def max_confidence(evidences: List[Dict[str, Any]]) -> float:
        """Return the highest confidence among evidence sources.

        Args:
            evidences: List of dicts with 'confidence' key.

        Returns:
            Maximum confidence value (0.0 if empty).
        """
        if not evidences:
            return 0.0
        return max(max(0.0, min(float(ev.get("confidence", 0.0)), 1.0)) for ev in evidences)

    @staticmethod
    def consensus_score(evidences: List[Dict[str, Any]]) -> float:
        """Agreement-based confidence — fraction of sources in the same tier.

        A high consensus score means most sources agree on the same tier
        (high/medium/low), which indicates corroboration.

        Args:
            evidences: List of dicts with 'confidence' key.

        Returns:
            Fraction in [0, 1] where 1.0 = perfect agreement.
        """
        if not evidences:
            return 0.0

        tiers_list = [
            EvidenceWeighter.tier(max(0.0, min(float(ev.get("confidence", 0.0)), 1.0)))
            for ev in evidences
        ]

        # Count votes per tier
        vote_counts: Dict[str, int] = {}
        for t in tiers_list:
            vote_counts[t] = vote_counts.get(t, 0) + 1

        # Agreement = fraction of sources in the most common tier
        max_votes = max(vote_counts.values())
        return max_votes / max(len(evidences), 1)


# ---------------------------------------------------------------------------
# Robust Agent Frame — Wrapper around VideoUnderstandingAgent
# ---------------------------------------------------------------------------


@dataclass
class TrustEvidence:
    """A single piece of trust-scored evidence from the agent."""

    tool_name: str
    success: bool
    data: str
    confidence: float = 0.0
    tier: str = "low"
    frame_trust: float = 0.0
    confidence_details: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentQueryTrustResult:
    """Agent query result with per-evidence confidence tracking."""

    query: str
    answer: str
    overall_confidence: float = 0.0
    consensus: float = 0.0
    max_confidence: float = 0.0
    weighted_confidence: float = 0.0
    trust_evidences: List[TrustEvidence] = field(default_factory=list)
    reasoning_steps: List[str] = field(default_factory=list)
    tools_used: int = 0
    duration_seconds: float = 0.0


class RobustAgentFrame:
    """Confidence-aware wrapper around VideoUnderstandingAgent.

    Adds frame-level trustworthiness assessment and evidence weighting to
    every tool invocation.  Frames below the configured trust threshold are
    skipped, and detection/OCR/MLLM confidences are adjusted based on the
    quality of the source frame.
    """

    def __init__(
        self,
        agent: VideoUnderstandingAgent,
        config: Optional[Config] = None,
    ):
        self._agent = agent
        self._tools = agent._tools
        self._config = config or agent.config

        # Pull confidence config
        self._enabled: bool = getattr(self._config, "agent_confidence_enabled", False)
        self._min_trust: float = getattr(self._config, "agent_confidence_min_trust", 0.3)
        self._weight_mode: str = getattr(self._config, "agent_confidence_weight_mode", "tiered")

        self._quality_scorer = FrameQualityScorer()
        self._trust_scorer = EvidenceTrustScorer()
        self._weighter = EvidenceWeighter()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def analyze_frames_trustworthy(
        self,
        timestamps: List[float],
        prompt: str = "Describe what you see in detail.",
    ) -> AgentToolResult:
        """Score frame quality and only send trustworthy frames to the MLLM.

        Frames below the configured trust threshold are excluded.  The
        MLLM response confidence is adjusted based on the average quality
        of frames actually used.
        """
        if not self._tools.video_path or not self._tools.video_path.exists():
            return AgentToolResult(
                tool_name="analyze_frames_trustworthy",
                success=False,
                data="Video file not available.",
                metadata={"timestamps": timestamps},
            )

        import cv2

        cap = cv2.VideoCapture(str(self._tools.video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames: List[np.ndarray] = []
        kept_timestamps: List[float] = []
        frame_scores: List[Dict[str, float]] = []

        for ts in timestamps:
            frame_idx = int(ts * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
            ret, frame = cap.read()
            if not ret:
                continue

            score = self._quality_scorer.score_frame(frame)
            frame_scores.append(score)

            if self._enabled and score["trustworthiness"] < self._min_trust:
                logger.debug(
                    "Skipping frame @ %.1fs (trust=%.3f < min=%.3f)",
                    ts,
                    score["trustworthiness"],
                    self._min_trust,
                )
                continue

            frames.append(frame)
            kept_timestamps.append(ts)

        cap.release()

        if not frames:
            return AgentToolResult(
                tool_name="analyze_frames_trustworthy",
                success=False,
                data="No trustworthy frames available at the specified timestamps.",
                metadata={
                    "timestamps": timestamps,
                    "frames_skipped": len(timestamps),
                    "min_trust_threshold": self._min_trust,
                },
            )

        # Compute average frame trustworthiness
        avg_trust = (
            statistics.mean(s["trustworthiness"] for s in frame_scores) if frame_scores else 0.5
        )

        # Convert frames for MLLM
        from PIL import Image

        pil_frames = [
            Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB) if f.shape[2] == 3 else f)
            for f in frames
        ]

        mllm = self._tools._get_mllm()
        if mllm is None:
            return AgentToolResult(
                tool_name="analyze_frames_trustworthy",
                success=False,
                data="Video MLLM not available.",
                metadata={"timestamps": kept_timestamps},
            )

        try:
            description = mllm.describe_scene(pil_frames, prompt=prompt)
            # Score the MLLM response
            mllm_score = self._trust_scorer.score_mllm_response(
                response=description or "",
                frame_quality=avg_trust,
                num_frames=len(pil_frames),
            )

            return AgentToolResult(
                tool_name="analyze_frames_trustworthy",
                success=True,
                data=description or "No description generated.",
                metadata={
                    "timestamps": kept_timestamps,
                    "num_frames": len(pil_frames),
                    "frames_skipped": len(timestamps) - len(pil_frames),
                    "avg_frame_trust": round(avg_trust, 4),
                    "mllm_confidence": mllm_score["mllm_confidence"],
                    "confidence_details": mllm_score,
                },
            )
        except Exception as exc:
            logger.exception("analyze_frames_trustworthy failed")
            return AgentToolResult(
                tool_name="analyze_frames_trustworthy",
                success=False,
                data=f"Error: {exc}",
                metadata={"timestamps": kept_timestamps},
            )

    def detect_objects_trustworthy(self, timestamp: float) -> AgentToolResult:
        """Run YOLO detection and adjust confidence by frame trustworthiness."""
        if not self._tools.video_path or not self._tools.video_path.exists():
            return AgentToolResult(
                tool_name="detect_objects_trustworthy",
                success=False,
                data="Video file not available.",
                metadata={"timestamp": timestamp},
            )

        yolo = self._tools._get_yolo()
        if yolo is None:
            return AgentToolResult(
                tool_name="detect_objects_trustworthy",
                success=False,
                data="YOLO model not available.",
                metadata={"timestamp": timestamp},
            )

        import cv2

        cap = cv2.VideoCapture(str(self._tools.video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_idx = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return AgentToolResult(
                tool_name="detect_objects_trustworthy",
                success=False,
                data="Could not extract frame at the given timestamp.",
                metadata={"timestamp": timestamp},
            )

        # Score frame quality
        frame_quality = self._quality_scorer.score_frame(frame)

        if self._enabled and frame_quality["trustworthiness"] < self._min_trust:
            return AgentToolResult(
                tool_name="detect_objects_trustworthy",
                success=False,
                data=(
                    f"Frame at {timestamp:.1f}s below trust threshold "
                    f"(trust={frame_quality['trustworthiness']:.3f} < "
                    f"min={self._min_trust})."
                ),
                metadata={
                    "timestamp": timestamp,
                    "frame_quality": frame_quality,
                    "min_trust_threshold": self._min_trust,
                },
            )

        results = yolo(frame, verbose=False)
        raw_detections: List[Dict[str, Any]] = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = r.names[cls_id]
                raw_detections.append({"label": label, "confidence": conf})

        # Adjust detection confidence by frame trustworthiness
        scored = self._trust_scorer.score_detection(raw_detections, frame_quality)

        if not raw_detections:
            return AgentToolResult(
                tool_name="detect_objects_trustworthy",
                success=True,
                data="No objects detected in this frame.",
                metadata={
                    "timestamp": timestamp,
                    "num_objects": 0,
                    "frame_quality": frame_quality,
                    "confidence_adjustment": scored,
                },
            )

        # Format with adjusted confidences
        adj_lines = []
        for d in scored["adjusted_detections"]:
            adj_lines.append(
                f"{d['label']} ({d['adjusted_confidence']:.2f}) [raw: {d['raw_confidence']:.2f}]"
            )

        return AgentToolResult(
            tool_name="detect_objects_trustworthy",
            success=True,
            data="Detected objects: " + ", ".join(adj_lines),
            metadata={
                "timestamp": timestamp,
                "num_objects": len(raw_detections),
                "raw_detections": raw_detections,
                "frame_quality": frame_quality,
                "confidence_adjustment": scored,
            },
        )

    def extract_text_trustworthy(self, timestamp: float) -> AgentToolResult:
        """Run OCR and adjust confidence by frame blur/quality."""
        if not self._tools.video_path or not self._tools.video_path.exists():
            return AgentToolResult(
                tool_name="extract_text_trustworthy",
                success=False,
                data="Video file not available.",
                metadata={"timestamp": timestamp},
            )

        ocr = self._tools._get_ocr()
        if ocr is None:
            return AgentToolResult(
                tool_name="extract_text_trustworthy",
                success=False,
                data="PaddleOCR not available.",
                metadata={"timestamp": timestamp},
            )

        import cv2

        cap = cv2.VideoCapture(str(self._tools.video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_idx = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return AgentToolResult(
                tool_name="extract_text_trustworthy",
                success=False,
                data="Could not extract frame at the given timestamp.",
                metadata={"timestamp": timestamp},
            )

        # Score frame quality
        frame_quality = self._quality_scorer.score_frame(frame)

        if self._enabled and frame_quality["trustworthiness"] < self._min_trust:
            return AgentToolResult(
                tool_name="extract_text_trustworthy",
                success=False,
                data=(
                    f"Frame at {timestamp:.1f}s below trust threshold "
                    f"(trust={frame_quality['trustworthiness']:.3f})."
                ),
                metadata={
                    "timestamp": timestamp,
                    "frame_quality": frame_quality,
                },
            )

        result = ocr.ocr(frame, cls=True)
        ocr_data = result if result else []

        # Score OCR with blur adjustment
        scored = self._trust_scorer.score_ocr_result(ocr_data, frame_quality)

        texts = [t["text"] for t in scored["adjusted_texts"]]

        if not texts:
            return AgentToolResult(
                tool_name="extract_text_trustworthy",
                success=True,
                data="No text detected in this frame.",
                metadata={
                    "timestamp": timestamp,
                    "num_texts": 0,
                    "frame_quality": frame_quality,
                    "confidence_adjustment": scored,
                },
            )

        return AgentToolResult(
            tool_name="extract_text_trustworthy",
            success=True,
            data="Extracted text: " + " | ".join(texts),
            metadata={
                "timestamp": timestamp,
                "num_texts": len(texts),
                "texts": texts,
                "frame_quality": frame_quality,
                "confidence_adjustment": scored,
            },
        )

    def query_with_trust(
        self,
        query: str,
        context: Optional[List[RetrievedChunk]] = None,
        max_tools: int = 5,
    ) -> AgentQueryTrustResult:
        """Full agent query with confidence-aware evidence weighting.

        Runs the standard agent query but wraps each tool result with
        trust scoring, then computes overall confidence metrics including
        weighted combination, max confidence, and consensus score.
        """
        import time

        start = time.time()

        # Run the standard agent query
        agent_result: AgentQueryResult = self._agent.query(
            question=query,
            context=context,
            max_tools=max_tools,
        )

        # Score each piece of evidence with trust
        trust_evidences: List[TrustEvidence] = []
        evidence_confs: List[Dict[str, Any]] = []

        for ev in agent_result.evidence:
            trust_ev = self._score_agent_tool_result(ev)
            trust_evidences.append(trust_ev)

            evidence_confs.append(
                {
                    "confidence": trust_ev.confidence,
                    "source": ev.tool_name,
                    "label": ev.tool_name,
                    "weight": 1.0,
                }
            )

        # Compute overall confidence metrics
        weighted = self._weighter.weighted_combine(evidence_confs)
        max_conf = self._weighter.max_confidence(evidence_confs)
        consensus = self._weighter.consensus_score(evidence_confs)

        # Decide overall_confidence based on weight_mode
        if self._weight_mode == "continuous":
            overall = weighted["combined_confidence"]
        else:
            # tiered: use the highest tier's average as anchor
            overall = max(
                weighted["combined_confidence"],
                consensus * 0.8 + max_conf * 0.2,
            )
        overall = max(0.0, min(overall, 1.0))

        elapsed = time.time() - start

        return AgentQueryTrustResult(
            query=query,
            answer=agent_result.answer,
            overall_confidence=round(overall, 4),
            consensus=round(consensus, 4),
            max_confidence=round(max_conf, 4),
            weighted_confidence=round(weighted["combined_confidence"], 4),
            trust_evidences=trust_evidences,
            reasoning_steps=agent_result.reasoning_steps,
            tools_used=agent_result.tools_used,
            duration_seconds=elapsed,
        )

    @staticmethod
    def format_confidence_report(result: AgentQueryTrustResult) -> str:
        """Generate a human-readable confidence report from a query result.

        Args:
            result: AgentQueryTrustResult from query_with_trust().

        Returns:
            Formatted markdown string with confidence breakdown per evidence
            source and overall metrics.
        """
        lines: List[str] = [
            "## Confidence Report",
            "",
            f"**Query**: {result.query}",
            f"**Answer**: {result.answer[:300]}{'...' if len(result.answer) > 300 else ''}",
            "",
            "### Overall Confidence",
            f"- Overall: **{result.overall_confidence:.2f}** / 1.00",
            f"- Weighted: {result.weighted_confidence:.2f}",
            f"- Max source: {result.max_confidence:.2f}",
            f"- Consensus: {result.consensus:.2f}",
            f"- Tools used: {result.tools_used}",
            f"- Duration: {result.duration_seconds:.1f}s",
            "",
            "### Per-Evidence Confidence",
            "",
        ]

        if not result.trust_evidences:
            lines.append("_(No evidence collected)_")
        else:
            for i, te in enumerate(result.trust_evidences, 1):
                lines.append(f"**{i}. {te.tool_name}** (tier: **{te.tier}**)")
                lines.append(f"   - Confidence: {te.confidence:.4f}")
                if te.frame_trust > 0:
                    lines.append(f"   - Frame trust: {te.frame_trust:.4f}")
                if te.confidence_details:
                    for k, v in te.confidence_details.items():
                        if isinstance(v, float):
                            lines.append(f"   - {k}: {v:.4f}")
                        else:
                            lines.append(f"   - {k}: {v}")
                lines.append(f"   - Success: {te.success}")
                if te.metadata:
                    meta_preview = dict(
                        (k, v) for k, v in te.metadata.items() if k not in ("confidence_details",)
                    )
                    if meta_preview:
                        lines.append(f"   - Metadata: {meta_preview}")
                lines.append("")

        lines.append("### Confidence Tiers")
        tier_counts: Dict[str, int] = {}
        for te in result.trust_evidences:
            tier_counts[te.tier] = tier_counts.get(te.tier, 0) + 1
        for tier_name in ("high", "medium", "low"):
            count = tier_counts.get(tier_name, 0)
            lines.append(f"- **{tier_name}**: {count} evidence(s)")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_agent_tool_result(
        self,
        result: AgentToolResult,
    ) -> TrustEvidence:
        """Score an AgentToolResult based on its tool type and metadata.

        The scoring strategy depends on the tool_name:
          - analyze_frames → uses num_frames metadata + any frame_quality
          - detect_objects → uses confidence_adjustment if present
          - extract_text → uses confidence_adjustment if present
          - search_rag / search_transcript → uses num_results + score metadata
          - Others → heuristic based on success + data length
        """
        tool_name = result.tool_name
        metadata = result.metadata or {}
        confidence_details: Dict[str, Any] = {}
        frame_trust: float = 0.0

        if tool_name == "analyze_frames_trustworthy":
            # Check for MLLM confidence from metadata
            mllm_conf = metadata.get("mllm_confidence")
            if mllm_conf is not None:
                confidence = float(mllm_conf)
            else:
                # Heuristic: success + data length
                confidence = 0.6 if result.success and len(result.data) > 20 else 0.3

            frame_trust = metadata.get("avg_frame_trust", 0.0)
            confidence_details = metadata.get("confidence_details", {})

        elif tool_name == "detect_objects_trustworthy":
            adj = metadata.get("confidence_adjustment", {})
            if adj:
                confidence = adj.get("mean_adjusted_confidence", 0.5)
            else:
                confidence = 0.5 if result.success else 0.0
            frame_trust = metadata.get("frame_quality", {}).get("trustworthiness", 0.0)
            confidence_details = adj

        elif tool_name == "extract_text_trustworthy":
            adj = metadata.get("confidence_adjustment", {})
            if adj:
                confidence = adj.get("mean_adjusted_confidence", 0.5)
            else:
                confidence = 0.5 if result.success else 0.0
            frame_trust = metadata.get("frame_quality", {}).get("trustworthiness", 0.0)
            confidence_details = adj

        elif tool_name in ("search_rag", "search_transcript", "temporal_grounding"):
            num_results = metadata.get("num_results", 0) or metadata.get("num_matches", 0)
            if result.success and num_results > 0:
                confidence = min(0.5 + 0.1 * num_results, 0.95)
            elif result.success:
                confidence = 0.4
            else:
                confidence = 0.0
            confidence_details = {"num_results": num_results}

        elif tool_name == "summarize_video":
            confidence = 0.7 if result.success and len(result.data) > 50 else 0.3
            confidence_details = {"has_mllm": metadata.get("mllm_available", False)}

        elif tool_name == "context_bootstrap":
            num_chunks = metadata.get("num_chunks", 0)
            confidence = min(0.5 + 0.05 * num_chunks, 0.9)
            confidence_details = {"num_chunks": num_chunks}

        else:
            # Generic fallback
            confidence = 0.5 if result.success else 0.0
            confidence_details = {"success": result.success}

        # Clamp to [0, 1]
        confidence = max(0.0, min(float(confidence), 1.0))
        tier = self._weighter.tier(confidence)

        return TrustEvidence(
            tool_name=tool_name,
            success=result.success,
            data=result.data,
            confidence=confidence,
            tier=tier,
            frame_trust=frame_trust,
            confidence_details=confidence_details,
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

__all__ = [
    "FrameQualityScorer",
    "EvidenceTrustScorer",
    "EvidenceWeighter",
    "RobustAgentFrame",
    "AgentQueryTrustResult",
    "TrustEvidence",
]
