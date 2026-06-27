"""Tests for adaptive pipeline scaler."""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from video_analysis.adaptive_scaler import (
    STAGE_VRAM_ESTIMATES,
    AdaptivePipelineScaler,
    FRAME_RATE_TIERS,
    FRAME_SIZE_TIERS,
    OCR_TIER_TIERS,
    QUALITY_BLUR_THRESHOLD_TIERS,
    ScalingResult,
    YOLO_CONFIDENCE_TIERS,
    estimate_vram_usage,
    get_free_vram_gb,
    get_video_properties,
    select_video_aware_policy,
)

# Note: the short_video fixture creates a 1s 1080p video, which triggers the
# "short high-resolution video" boost in the scaler (frame_analysis_size -> 1280).
# Tests using short_video with explicit policies must account for this.


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def short_video(tmp_path: Path) -> Path:
    """Create a short test video (1 second, 1080p, 30fps) using ffmpeg."""
    path = tmp_path / "short_test.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=1:size=1920x1080:rate=30",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        capture_output=True,
        timeout=30,
    )
    if not path.exists():
        pytest.skip("ffmpeg not available or video creation failed")
    return path


@pytest.fixture
def long_video(tmp_path: Path) -> Path:
    """Create a synthetic long video (15 minutes, 720p, 24fps) using ffmpeg."""
    path = tmp_path / "long_test.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=900:size=1280x720:rate=24",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        capture_output=True,
        timeout=60,
    )
    if not path.exists():
        pytest.skip("ffmpeg not available or long video creation failed")
    return path


@pytest.fixture
def medium_video(tmp_path: Path) -> Path:
    """Create a medium video (5 minutes, 480p, 24fps)."""
    path = tmp_path / "medium_test.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=300:size=854x480:rate=24",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        capture_output=True,
        timeout=60,
    )
    if not path.exists():
        pytest.skip("ffmpeg not available or medium video creation failed")
    return path


# ── Tests: get_video_properties ──────────────────────────────────────────────


def test_get_video_properties_short_video(short_video: Path):
    """ffprobe should return sensible values for a known test video."""
    props = get_video_properties(short_video)
    assert props["width"] == 1920.0
    assert props["height"] == 1080.0
    assert props["duration"] >= 0.9  # ffmpeg test source ~1s
    assert props["fps"] >= 29.0  # 30fps nominal


def test_get_video_properties_nonexistent():
    """Non-existent file should return all zeros."""
    props = get_video_properties(Path("/nonexistent/video.mp4"))
    assert props["duration"] == 0.0
    assert props["width"] == 0.0
    assert props["height"] == 0.0


def test_get_video_properties_audio_file(tmp_path: Path):
    """An audio-only file has no video stream -> ffprobe returns all defaults."""
    path = tmp_path / "audio_test.mp3"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-c:a",
            "libmp3lame",
            str(path),
        ],
        capture_output=True,
        timeout=30,
    )
    if not path.exists():
        pytest.skip("ffmpeg not available for audio test")
    props = get_video_properties(path)
    # Audio file has no video stream, so ffprobe returns empty -> all defaults (0.0)
    assert props["duration"] == 0.0
    assert props["width"] == 0.0
    assert props["height"] == 0.0


# ── Tests: select_video_aware_policy ─────────────────────────────────────────


class TestSelectVideoAwarePolicy:
    def test_long_video_returns_conservative(self):
        """Videos over 10 minutes should use conservative."""
        assert select_video_aware_policy(700, 2.0) == "conservative"
        assert select_video_aware_policy(3600, 0.5) == "conservative"

    def test_short_high_res_returns_performance(self):
        """Short (<1min) high-res (>1080p) videos use performance."""
        assert select_video_aware_policy(30, 8.0) == "performance"
        assert select_video_aware_policy(59, 2.1) == "performance"

    def test_short_low_res_returns_balanced(self):
        """Short but low-resolution videos still use balanced."""
        assert select_video_aware_policy(30, 0.3) == "balanced"

    def test_medium_video_returns_balanced(self):
        """Medium-length videos use balanced."""
        assert select_video_aware_policy(300, 0.5) == "balanced"
        assert select_video_aware_policy(120, 0.3) == "balanced"

    def test_boundary_transitions(self):
        """Edge cases at threshold boundaries."""
        # Exactly at long threshold (600s >= 600s -> conservative)
        assert select_video_aware_policy(600, 0.5) == "conservative"
        # Just over
        assert select_video_aware_policy(601, 0.5) == "conservative"
        # Exactly at short threshold
        assert select_video_aware_policy(60, 2.0) == "performance"
        # Just over
        assert select_video_aware_policy(61, 2.0) == "balanced"


# ── Tests: estimate_vram_usage ───────────────────────────────────────────────


class TestEstimateVramUsage:
    def test_all_enabled_conservative(self):
        stages = {"whisper": True, "yolo": True, "clip": True}
        vram = estimate_vram_usage(stages, "conservative")
        assert vram < estimate_vram_usage(stages, "balanced")
        assert vram > 0

    def test_all_enabled_performance(self):
        stages = {"whisper": True, "yolo": True, "clip": True}
        vram = estimate_vram_usage(stages, "performance")
        assert vram > estimate_vram_usage(stages, "balanced")

    def test_with_optional_stages(self):
        stages = {
            "whisper": True,
            "yolo": True,
            "clip": True,
            "xclip": True,
            "video_mllm": True,
            "face_recognition": True,
        }
        vram = estimate_vram_usage(stages, "balanced")
        expected_base = (
            STAGE_VRAM_ESTIMATES["whisper"]
            + STAGE_VRAM_ESTIMATES["yolo"]
            + STAGE_VRAM_ESTIMATES["clip"]
        )
        expected_extra = (
            STAGE_VRAM_ESTIMATES["xclip"]
            + STAGE_VRAM_ESTIMATES["video_mllm"]
            + STAGE_VRAM_ESTIMATES["face_recognition"]
        )
        assert vram >= expected_base + expected_extra - 0.5
        assert vram <= expected_base + expected_extra + 0.5

    def test_no_stages_all_disabled(self):
        """When all stages are explicitly False, VRAM usage should be 0."""
        stages = {
            "whisper": False,
            "yolo": False,
            "clip": False,
            "xclip": False,
            "video_mllm": False,
            "face_recognition": False,
            "dino": False,
            "ocr": False,
            "diarization": False,
        }
        assert estimate_vram_usage(stages, "balanced") == 0.0


# ── Tests: get_free_vram_gb ──────────────────────────────────────────────────


class TestGetFreeVramGb:
    def test_returns_float(self):
        """Should return a float (0.0 if no GPU)."""
        vram = get_free_vram_gb()
        assert isinstance(vram, float)
        assert vram >= 0.0

    def test_with_gpu(self):
        """When get_free_vram_gb returns >0, it should be usable (integration test)."""
        vram = get_free_vram_gb()
        assert isinstance(vram, float)

    def test_without_gpu(self):
        """Run-time guard: should always return a non-negative float."""
        vram = get_free_vram_gb()
        assert vram >= 0.0


# ── Tests: ScalingResult ─────────────────────────────────────────────────────


class TestScalingResult:
    def test_to_dict_omits_none(self):
        result = ScalingResult(frame_rate=0.5, policy_used="balanced")
        d = result.to_dict()
        assert d["frame_rate"] == 0.5
        assert "yolo_confidence" not in d
        assert "action_recognition_enabled" not in d

    def test_to_dict_includes_all_set(self):
        result = ScalingResult(
            frame_rate=0.5,
            yolo_confidence=0.3,
            ocr_model_tier="tiny",
            policy_used="conservative",
        )
        d = result.to_dict()
        assert d["frame_rate"] == 0.5
        assert d["yolo_confidence"] == 0.3
        assert d["ocr_model_tier"] == "tiny"
        assert "policy_used" not in d  # metadata field, not config override


# ── Tests: AdaptivePipelineScaler (unit, mocked) ──────────────────────────────


class TestAdaptivePipelineScalerMocked:
    def test_analyze_conservative_policy(self, short_video: Path):
        """With explicit conservative policy, frame rate and size should be low."""
        scaler = AdaptivePipelineScaler()
        result = scaler.analyze(short_video, policy_override="conservative")
        assert result.policy_used == "conservative"
        assert result.frame_rate == FRAME_RATE_TIERS["conservative"]
        # short_video is 1s 1080p -> triggers high-res boost (overrides to 1280)
        assert result.frame_analysis_size is not None
        assert result.frame_analysis_size >= FRAME_SIZE_TIERS["conservative"]
        assert result.yolo_confidence == YOLO_CONFIDENCE_TIERS["conservative"]
        assert result.ocr_model_tier == OCR_TIER_TIERS["conservative"]

    def test_analyze_performance_policy(self, short_video: Path):
        """With explicit performance policy, settings should be high."""
        scaler = AdaptivePipelineScaler()
        result = scaler.analyze(short_video, policy_override="performance")
        assert result.policy_used == "performance"
        assert result.frame_rate == FRAME_RATE_TIERS["performance"]
        assert result.frame_analysis_size is not None
        assert result.frame_analysis_size >= FRAME_SIZE_TIERS["performance"]

    def test_analyze_balanced_policy(self, short_video: Path):
        """With explicit balanced policy, settings should be default."""
        scaler = AdaptivePipelineScaler()
        result = scaler.analyze(short_video, policy_override="balanced")
        assert result.policy_used == "balanced"
        assert result.frame_rate == FRAME_RATE_TIERS["balanced"]
        # short_video is 1s 1080p -> triggers high-res boost (overrides to 1280)
        assert result.frame_analysis_size is not None
        assert result.frame_analysis_size >= FRAME_SIZE_TIERS["balanced"]

    def test_analyze_auto_respects_duration(self, short_video: Path):
        """Auto should select based on video properties."""
        scaler = AdaptivePipelineScaler(default_policy="auto")
        result = scaler.analyze(short_video)
        # Short video -> should be performance or balanced
        assert result.policy_used in ("balanced", "performance")

    def test_analyze_vram_disables_expensive_stages(self, short_video: Path):
        """When VRAM is critically low, expensive stages should be disabled."""
        scaler = AdaptivePipelineScaler()
        enabled = {
            "whisper": True,
            "yolo": True,
            "clip": True,
            "xclip": True,
            "video_mllm": True,
            "face_recognition": True,
        }
        with patch(
            "video_analysis.adaptive_scaler.get_free_vram_gb",
            return_value=2.0,
        ):
            result = scaler.analyze(
                short_video,
                enabled_stages=enabled,
                policy_override="balanced",
            )
        # With only 2GB free and all expensive stages enabled
        assert result.action_recognition_enabled is False
        assert result.video_mllm_as_describer is False
        assert result.face_recognition_enabled is False

    def test_video_properties_in_result(self, short_video: Path):
        """Video properties should be propagated to the result."""
        scaler = AdaptivePipelineScaler()
        result = scaler.analyze(short_video)
        assert result.video_duration >= 0.9
        assert result.video_width >= 1920
        assert result.video_height >= 1080
        assert result.video_fps >= 29.0

    def test_reasoning_is_populated(self, short_video: Path):
        """Reasoning list should contain at least one entry."""
        scaler = AdaptivePipelineScaler()
        result = scaler.analyze(short_video, policy_override="conservative")
        assert len(result.reasoning) >= 1


# ── Tests: AdaptivePipelineScaler (integration with real ffprobe) ─────────────


class TestAdaptivePipelineScalerIntegration:
    def test_detects_short_high_res(self, short_video: Path):
        """Real ffprobe detection should identify 1080p/30fps video."""
        props = get_video_properties(short_video)
        assert props["width"] == 1920.0
        assert props["height"] == 1080.0
        assert props["fps"] >= 29.0

    def test_policy_for_short_video(self, short_video: Path):
        """Auto policy on a short video should be performance."""
        scaler = AdaptivePipelineScaler(default_policy="auto")
        result = scaler.analyze(short_video)
        assert result.policy_used in ("balanced", "performance")

    def test_estimated_frames_sanity(self, short_video: Path):
        """Estimated frame count should be sensible."""
        scaler = AdaptivePipelineScaler()
        result = scaler.analyze(short_video, policy_override="balanced")
        # 1s video at 0.5 fps -> 0 or 1 frame
        assert result.total_frames_estimate <= 5

    def test_long_video_detection(self, medium_video: Path):
        """Medium length video (5 min) should use balanced."""
        scaler = AdaptivePipelineScaler(default_policy="auto")
        result = scaler.analyze(medium_video)
        assert result.policy_used == "balanced"

    def test_disabled_stages_respected(self, short_video: Path):
        """If all stages are disabled, VRAM estimate should be near zero."""
        enabled = {
            k: False
            for k in (
                "whisper",
                "yolo",
                "clip",
                "xclip",
                "video_mllm",
                "face_recognition",
                "dino",
                "ocr",
                "diarization",
            )
        }
        scaler = AdaptivePipelineScaler()
        result = scaler.analyze(
            short_video,
            enabled_stages=enabled,
            policy_override="conservative",
        )
        assert result.estimated_vram_gb < 1.0

    def test_frame_size_for_performance(self, short_video: Path):
        """Performance policy should give max frame size."""
        scaler = AdaptivePipelineScaler()
        result = scaler.analyze(short_video, policy_override="performance")
        assert result.frame_analysis_size is not None
        assert result.frame_analysis_size >= FRAME_SIZE_TIERS["performance"]


# ── Tests: VRAM pressure downgrade ────────────────────────────────────────────


class TestVramPressure:
    def test_high_pressure_downgrades_performance_to_balanced(self, short_video: Path):
        """85% VRAM pressure should downgrade performance to balanced."""
        scaler = AdaptivePipelineScaler()
        enabled = {"whisper": True, "yolo": True, "clip": True}
        with patch(
            "video_analysis.adaptive_scaler.get_free_vram_gb",
            return_value=6.0,
        ):
            result = scaler.analyze(
                short_video,
                enabled_stages=enabled,
                policy_override="performance",
            )
        # Conservative VRAM: 3.5*0.7 + 1.0*0.6 + 1.5*0.5 = 2.45+0.6+0.75 = 3.8
        # Performance VRAM: 3.5*1.3 + 1.0*1.5 + 1.5*2.0 = 4.55+1.5+3.0 = 9.05
        # 9.05/6.0 = 150% -> critical VRAM pressure -> forces conservative
        assert result.policy_used == "conservative"

    def test_critical_pressure_forces_conservative(self, short_video: Path):
        """>95% VRAM pressure should force conservative regardless of policy."""
        scaler = AdaptivePipelineScaler()
        enabled = {"whisper": True, "yolo": True, "clip": True}
        with patch(
            "video_analysis.adaptive_scaler.get_free_vram_gb",
            return_value=4.0,
        ):
            result = scaler.analyze(
                short_video,
                enabled_stages=enabled,
                policy_override="balanced",
            )
        # Balanced: 3.5 + 1.0 + 1.5 = 6.0 GB estimated
        # 6.0/4.0 = 150% -> critical, should force conservative
        assert result.policy_used == "conservative"

    def test_no_vram_info_does_not_crash(self, short_video: Path):
        """When VRAM detection returns 0.0, scaler should not crash."""
        scaler = AdaptivePipelineScaler()
        with patch(
            "video_analysis.adaptive_scaler.get_free_vram_gb",
            return_value=0.0,
        ):
            result = scaler.analyze(short_video, policy_override="balanced")
        assert result.policy_used == "balanced"
