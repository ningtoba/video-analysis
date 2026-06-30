"""
Tests for the Confidence-Aware Robust Agent Framework (v0.51.0).

Matches the subagent-produced API of video_analysis/agent_confidence.py.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from video_analysis.agent_confidence import (
    EvidenceTrustScorer,
    EvidenceWeighter,
    FrameQualityScorer,
)

# ===========================================================================
# FrameQualityScorer Tests
# ===========================================================================


class TestFrameQualityScorer:
    def test_score_valid_frame(self):
        frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        result = FrameQualityScorer.score_frame(frame)
        assert isinstance(result, dict)
        for key in (
            "blur_score",
            "brightness_score",
            "motion_score",
            "occlusion_score",
            "trustworthiness",
        ):
            assert key in result
        assert 0.0 <= result["trustworthiness"] <= 1.0

    def test_score_blank_frame(self):
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        result = FrameQualityScorer.score_frame(blank)
        assert result["trustworthiness"] < 0.5

    def test_score_frames_batch(self):
        frames = [np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8) for _ in range(3)]
        results = FrameQualityScorer.score_frames_batch(frames)
        assert len(results) == 3
        assert all(isinstance(r, dict) for r in results)

    def test_batch_motion_detection(self):
        static = np.full((100, 100, 3), 128, dtype=np.uint8)
        results = FrameQualityScorer.score_frames_batch([static, static])
        assert results[1]["motion_score"] < 0.1

    def test_batch_different_frames(self):
        f1 = np.full((100, 100, 3), 0, dtype=np.uint8)
        f2 = np.full((100, 100, 3), 255, dtype=np.uint8)
        results = FrameQualityScorer.score_frames_batch([f1, f2])
        assert results[1]["motion_score"] > 0.5


# ===========================================================================
# EvidenceTrustScorer Tests
# ===========================================================================


class TestEvidenceTrustScorer:
    def test_score_rag_chunk_high(self):
        chunk = SimpleNamespace(
            score=0.95,
            chunk_type="transcript",
            timestamp=10.0,
            text="a",
            scene_id=0,
            metadata={},
        )
        result = EvidenceTrustScorer.score_rag_chunk(chunk)
        assert result["source_confidence"] > 0.7

    def test_score_rag_chunk_low(self):
        chunk = SimpleNamespace(
            score=0.3,
            chunk_type="sliding_30s",
            timestamp=None,
            text="b",
            scene_id=0,
            metadata={},
        )
        result = EvidenceTrustScorer.score_rag_chunk(chunk)
        assert result["source_confidence"] < 0.6

    def test_score_detection_adjusted(self):
        dets = [{"confidence": 0.9}, {"confidence": 0.8}]
        result = EvidenceTrustScorer.score_detection(dets, {"trustworthiness": 0.5})
        assert "frame_quality_factor" in result
        assert result["frame_quality_factor"] == 0.5

    def test_score_detection_empty(self):
        result = EvidenceTrustScorer.score_detection([], {"trustworthiness": 0.5})
        assert "mean_raw_confidence" in result

    def test_score_transcript(self):
        result = EvidenceTrustScorer.score_transcript_segment({"text": "hello", "confidence": 0.95})
        assert "text_confidence" in result
        assert result["text_confidence"] > 0.5

    def test_score_ocr(self):
        # PaddleOCR format: list of entries, each entry = [[bbox, (text, conf)], ...]
        ocr_data = [[[[0, 0, 10, 10], ("hello", 0.9)]]]
        result = EvidenceTrustScorer.score_ocr_result(
            ocr_data, frame_quality={"trustworthiness": 0.9, "blur_score": 200.0}
        )
        assert "mean_ocr_confidence" in result
        assert result["mean_ocr_confidence"] > 0.5

    def test_score_mllm_short(self):
        result = EvidenceTrustScorer.score_mllm_response("ok", frame_quality=0.2, num_frames=1)
        assert "mllm_confidence" in result

    def test_score_mllm_good(self):
        result = EvidenceTrustScorer.score_mllm_response("A" * 500, frame_quality=0.9, num_frames=4)
        assert result["mllm_confidence"] > 0.0


# ===========================================================================
# EvidenceWeighter Tests
# ===========================================================================


class TestEvidenceWeighter:
    def test_tier_high(self):
        assert EvidenceWeighter.tier(0.85) == "high"
        assert EvidenceWeighter.tier(0.80) == "high"

    def test_tier_medium(self):
        assert EvidenceWeighter.tier(0.65) == "medium"
        assert EvidenceWeighter.tier(0.50) == "medium"

    def test_tier_low(self):
        assert EvidenceWeighter.tier(0.45) == "low"
        assert EvidenceWeighter.tier(0.0) == "low"

    def test_weight_empty(self):
        r = EvidenceWeighter.weighted_combine([])
        assert isinstance(r, dict)
        assert r["num_sources"] == 0

    def test_weight_single(self):
        r = EvidenceWeighter.weighted_combine([{"confidence": 0.9, "source": "rag"}])
        assert r["combined_confidence"] >= 0.8
        assert r["num_sources"] == 1

    def test_weight_mixed(self):
        sources = [
            {"confidence": 0.9, "source": "rag"},
            {"confidence": 0.5, "source": "yolo"},
            {"confidence": 0.2, "source": "ocr"},
        ]
        r = EvidenceWeighter.weighted_combine(sources)
        assert r["num_sources"] == 3
        assert 0.0 < r["combined_confidence"] < 1.0

    def test_consensus_agreement(self):
        sources = [
            {"confidence": 0.8, "source": "a"},
            {"confidence": 0.85, "source": "b"},
        ]
        assert EvidenceWeighter.consensus_score(sources) > 0.5

    def test_consensus_disagreement(self):
        sources = [
            {"confidence": 0.9, "source": "a"},
            {"confidence": 0.1, "source": "b"},
        ]
        assert EvidenceWeighter.consensus_score(sources) >= 0.0

    def test_max_confidence(self):
        sources = [
            {"confidence": 0.5, "source": "a"},
            {"confidence": 0.95, "source": "b"},
        ]
        assert EvidenceWeighter.max_confidence(sources) == 0.95
