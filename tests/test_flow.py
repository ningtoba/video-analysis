"""Tests for the Sparse-frame Optical Flow module (video_analysis/flow.py)."""

import subprocess

import pytest

from video_analysis.flow import FFmpegMotionExtractor


class TestFFmpegMotionExtractor:
    def test_init_checks_ffmpeg(self):
        """Extractor initialises and checks for ffprobe."""
        ext = FFmpegMotionExtractor()
        # ffprobe should be available on the test machine
        assert ext._ffmpeg_available is True

    def test_extract_nonexistent_file(self):
        """Returns empty list for missing file."""
        ext = FFmpegMotionExtractor()
        result = ext.extract("/nonexistent/video.mp4")
        assert result == []

    def test_motion_score_zero_for_empty(self):
        """Default motion score is 0.0 for empty frame dict."""
        ext = FFmpegMotionExtractor()
        assert ext.motion_score({}) == 0.0

    def test_motion_score_from_frame(self):
        ext = FFmpegMotionExtractor()
        frame = {"motion_score": 0.75}
        assert ext.motion_score(frame) == 0.75

    def test_is_static(self):
        ext = FFmpegMotionExtractor()
        assert ext.is_static({"motion_score": 0.01}) is True
        assert ext.is_static({"motion_score": 0.5}) is False

    def test_scene_cut_candidates_empty(self):
        ext = FFmpegMotionExtractor()
        assert ext.scene_cut_candidates([]) == []
        assert ext.scene_cut_candidates([{"motion_score": 0.1}]) == []

    def test_scene_cut_candidates_detects_cuts(self):
        ext = FFmpegMotionExtractor()
        frames = [
            {"frame": 0, "motion_score": 0.1},
            {"frame": 1, "motion_score": 0.12},
            {"frame": 2, "motion_score": 0.5},  # sudden jump
            {"frame": 3, "motion_score": 0.52},
        ]
        candidates = ext.scene_cut_candidates(frames, sensitivity=0.3)
        assert 2 in candidates  # frame 2 is a cut

    def test_direction_entropy_uniform(self):
        """All MVs same direction = low entropy."""
        from video_analysis.flow import FFmpegMotionExtractor as FME

        mvs = [
            {"magnitude": 10, "angle": 1.0, "motion_x": 5, "motion_y": 5},
            {"magnitude": 12, "angle": 1.0, "motion_x": 6, "motion_y": 6},
            {"magnitude": 8, "angle": 1.0, "motion_x": 4, "motion_y": 4},
        ]
        entropy = FME._direction_entropy(mvs)
        assert entropy < 0.5  # low entropy

    def test_direction_entropy_few_mvs(self):
        """Fewer than 4 MVs = zero entropy."""
        from video_analysis.flow import FFmpegMotionExtractor as FME

        mvs = [{"magnitude": 10, "angle": 1.0, "motion_x": 5, "motion_y": 5}]
        assert FME._direction_entropy(mvs) == 0.0

    def test_is_video_frame(self):
        from video_analysis.flow import FFmpegMotionExtractor as FME

        assert FME._is_video_frame({"media_type": "video"}) is True
        assert FME._is_video_frame({"media_type": "audio"}) is False
        assert FME._is_video_frame({}) is False

    def test_extract_mv_from_side_data_empty(self):
        from video_analysis.flow import FFmpegMotionExtractor as FME

        assert FME._extract_mv_from_side_data({}) == []
        assert FME._extract_mv_from_side_data({"side_data_list": []}) == []

    def test_extract_mv_from_side_data(self):
        from video_analysis.flow import FFmpegMotionExtractor as FME

        frame = {
            "side_data_list": [
                {
                    "side_data_type": "Motion Vectors",
                    "mvs": [
                        {"motion_x": 3, "motion_y": 4, "src_x": 0, "src_y": 0},
                        {"motion_x": -2, "motion_y": 1, "src_x": 16, "src_y": 0},
                    ],
                }
            ]
        }
        mvs = FME._extract_mv_from_side_data(frame)
        assert len(mvs) == 2
        assert mvs[0]["magnitude"] == 5.0  # 3-4-5 triangle
        assert mvs[0]["motion_x"] == 3
        assert mvs[0]["motion_y"] == 4
        assert mvs[1]["magnitude"] == pytest.approx(2.236, 0.1)

    def test_parse_mvs_with_side_data(self):
        ext = FFmpegMotionExtractor()
        data = {
            "frames": [
                {
                    "media_type": "video",
                    "coded_picture_number": 0,
                    "pict_type": "B",
                    "side_data_list": [
                        {
                            "side_data_type": "Motion Vectors",
                            "mvs": [{"motion_x": 1, "motion_y": 1, "src_x": 0, "src_y": 0}],
                        }
                    ],
                },
                {
                    "media_type": "video",
                    "coded_picture_number": 1,
                    "pict_type": "P",
                    "side_data_list": [],
                },
            ]
        }
        frames = ext._parse_mvs(data, max_frames=100)
        assert len(frames) == 2
        assert frames[0]["mv_count"] == 1
        assert frames[0]["motion_score"] > 0
        assert frames[1]["mv_count"] == 0
        assert frames[1]["motion_score"] == 0.0

    def test_fallback_frame_diff(self, tmp_path):
        """Fallback works with a real MP4 file."""
        ext = FFmpegMotionExtractor()
        if not ext._ffmpeg_available:
            pytest.skip("ffprobe not available")

        # Create a minimal MP4 using ffmpeg
        mp4_path = tmp_path / "test.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=64x64:d=1:r=5",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(mp4_path),
            ],
            capture_output=True,
            timeout=15,
        )
        assert mp4_path.exists()

        frames = ext._fallback_frame_diff(mp4_path, max_frames=10)
        assert len(frames) <= 10
        assert len(frames) >= 1
        assert frames[0]["mv_count"] == 0  # fallback has no real MVs
        assert "motion_score" in frames[0]
        assert "pict_type" in frames[0]
