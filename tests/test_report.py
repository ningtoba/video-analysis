"""
Tests for the Structured Video Report Generator (v0.51.0).

Covers:
- VideoReport dataclass construction
- ReportGenerator: from_video_index, from_video_id
- Serialisation (to_json, save, load)
- Rendering (summary_text, to_chunk_context)
- Key dataclasses (VideoMetadata, TimelineSummary, etc.)
"""

import json
import tempfile
from pathlib import Path

import pytest

from video_analysis.models import SceneInfo, FrameInfo, VideoIndex, TranscriptSegment
from video_analysis.report import (
    ActionSummary,
    ChapterSummary,
    CurationSummary,
    FaceSummary,
    KeyMoment,
    OCRSummary,
    ObjectCatalog,
    QualityMetrics,
    RAGStats,
    ReportGenerator,
    SceneReport,
    TimelineSummary,
    TranscriptReport,
    VideoMetadata,
    VideoReport,
    _fmt_duration,
    _fmt_size,
)
from video_analysis.config import Config

# ===========================================================================
# Helper Tests
# ===========================================================================


class TestFormatHelpers:
    def test_fmt_duration_zero(self):
        assert _fmt_duration(0) == "0:00"

    def test_fmt_duration_seconds(self):
        assert _fmt_duration(90) == "1:30"

    def test_fmt_duration_hours(self):
        assert _fmt_duration(3661) == "1:01:01"

    def test_fmt_duration_negative(self):
        assert _fmt_duration(-5) == "0:00"

    def test_fmt_size_bytes(self):
        assert _fmt_size(500) == "500.0 B"

    def test_fmt_size_kb(self):
        assert _fmt_size(2048) == "2.0 KB"

    def test_fmt_size_mb(self):
        assert _fmt_size(1048576) == "1.0 MB"

    def test_fmt_size_zero(self):
        assert _fmt_size(0) == "N/A"


# ===========================================================================
# Dataclass Tests
# ===========================================================================


class TestVideoMetadata:
    def test_defaults(self):
        m = VideoMetadata()
        assert m.video_id == ""
        assert m.duration == 0.0
        assert m.fps == 0.0

    def test_with_values(self):
        m = VideoMetadata(
            video_id="test123",
            title="My Video",
            duration=120.5,
            fps=30.0,
            width=1920,
            height=1080,
            checksum="abc123",
        )
        assert m.video_id == "test123"
        assert m.duration == 120.5
        assert m.checksum == "abc123"


class TestTimelineSummary:
    def test_defaults(self):
        t = TimelineSummary()
        assert t.num_scenes == 0
        assert t.scene_boundaries == []


class TestSceneReport:
    def test_minimal(self):
        s = SceneReport(scene_id=0, start_time=0.0, end_time=10.0, duration=10.0)
        assert s.duration == 10.0
        assert s.objects == []
        assert s.key_moments == []

    def test_with_key_moments(self):
        km = KeyMoment(timestamp=5.0, description="explosion", source="transcript")
        s = SceneReport(
            scene_id=1,
            start_time=0.0,
            end_time=30.0,
            description="Action scene",
            objects=["car", "person"],
            key_moments=[km],
        )
        assert len(s.key_moments) == 1
        assert s.key_moments[0].description == "explosion"


class TestTranscriptReport:
    def test_with_speakers(self):
        t = TranscriptReport(
            total_segments=10,
            total_duration=100.0,
            speaker_count=2,
            speakers={
                "SPEAKER_00": {
                    "segment_count": 6,
                    "total_words": 200,
                    "word_count_pct": 0.6,
                },
                "SPEAKER_01": {
                    "segment_count": 4,
                    "total_words": 133,
                    "word_count_pct": 0.4,
                },
            },
            key_phrases=["hello", "world"],
            total_words=333,
        )
        assert t.speaker_count == 2
        assert t.total_words == 333


class TestObjectCatalog:
    def test_top_objects(self):
        o = ObjectCatalog(
            unique_objects=["person", "car"],
            total_detections=50,
            object_frequency={"person": 30, "car": 20},
            top_objects=[("person", 30, 5), ("car", 20, 3)],
        )
        assert len(o.top_objects) == 2
        assert o.top_objects[0][0] == "person"


class TestVideoReport:
    def test_minimal(self):
        r = VideoReport()
        assert r.schema_version == "1.0"
        assert r.generated_at == ""

    def test_full(self):
        r = VideoReport(
            schema_version="1.0",
            generated_at="2026-06-27T12:00:00+00:00",
            video=VideoMetadata(video_id="v1", duration=60.0),
            timeline=TimelineSummary(num_scenes=3),
            scenes=[
                SceneReport(scene_id=0, start_time=0.0, end_time=20.0),
                SceneReport(scene_id=1, start_time=20.0, end_time=40.0),
            ],
            transcript=TranscriptReport(total_segments=5),
            objects=ObjectCatalog(unique_objects=["a", "b"]),
        )
        assert r.video.video_id == "v1"
        assert len(r.scenes) == 2
        assert r.transcript.total_segments == 5
        assert len(r.objects.unique_objects) == 2


# ===========================================================================
# ReportGenerator Tests
# ===========================================================================


class TestReportGenerator:
    def setup_method(self):
        self.config = Config(data_dir=tempfile.mkdtemp())
        self.generator = ReportGenerator(config=self.config)

    def test_from_video_index_empty(self):
        """Should handle an empty/minimal VideoIndex."""
        index = VideoIndex(
            video_id="test_video",
            filename="test.mp4",
            duration=60.0,
            filepath="/path/to/test.mp4",
        )
        report = self.generator.from_video_index(index)
        assert report.video.video_id == "test_video"
        assert report.video.duration == 60.0
        assert report.timeline.num_scenes == 0
        assert report.generated_at != ""

    def test_from_video_index_with_scenes(self):
        """Should populate scenes correctly."""
        frames = [
            FrameInfo(
                timestamp=5.0,
                filepath="/f1.jpg",
                scene_id=0,
                objects=[{"label": "person", "confidence": 0.9}],
                description="A person speaking",
            ),
            FrameInfo(
                timestamp=15.0,
                filepath="/f2.jpg",
                scene_id=0,
                objects=[{"label": "chair", "confidence": 0.8}],
            ),
        ]
        scenes = [
            SceneInfo(
                scene_id=0,
                start_time=0.0,
                end_time=20.0,
                key_frames=frames,
                summary="Opening scene",
                transcript="Hello and welcome",
            ),
            SceneInfo(
                scene_id=1,
                start_time=20.0,
                end_time=40.0,
                key_frames=[FrameInfo(timestamp=30.0, filepath="/f3.jpg", scene_id=1)],
                summary="Middle scene",
            ),
        ]
        transcript = [
            TranscriptSegment(
                start=0.0, end=5.0, text="Hello world", speaker="SPEAKER_00"
            ),
            TranscriptSegment(
                start=6.0, end=10.0, text="How are you", speaker="SPEAKER_01"
            ),
        ]
        index = VideoIndex(
            video_id="test",
            filename="test.mp4",
            duration=40.0,
            filepath="/v/test.mp4",
            scenes=scenes,
            transcript=transcript,
        )
        report = self.generator.from_video_index(index)
        assert len(report.scenes) == 2
        assert report.timeline.num_scenes == 2
        assert len(report.timeline.scene_boundaries) == 2
        assert report.timeline.mean_scene_duration == 20.0

        # Check transcript fields
        assert report.transcript.total_segments == 2
        assert report.transcript.speaker_count == 2
        assert "SPEAKER_00" in report.transcript.speakers
        assert "SPEAKER_01" in report.transcript.speakers

        # Check object catalog
        assert len(report.objects.unique_objects) >= 2
        assert report.objects.total_detections >= 2

    def test_from_video_index_with_transcript_only(self):
        """Transcript-only index should still produce a valid report."""
        index = VideoIndex(
            video_id="audio_only",
            filename="podcast.mp3",
            duration=300.0,
            filepath="/v/podcast.mp3",
            transcript=[
                TranscriptSegment(
                    start=0.0,
                    end=10.0,
                    text="Introduction segment",
                    speaker="SPEAKER_00",
                ),
                TranscriptSegment(
                    start=30.0,
                    end=60.0,
                    text="Main topic discussion",
                    speaker="SPEAKER_00",
                ),
            ],
        )
        report = self.generator.from_video_index(index)
        assert report.transcript.total_segments == 2
        assert report.timeline.num_scenes == 0  # no scenes
        # Gap between 10.0 and 30.0 = 20 seconds > 2s threshold
        assert len(report.transcript.silent_periods) >= 1

    def test_to_json_and_load_roundtrip(self):
        """JSON serialisation round-trip should preserve all fields."""
        index = VideoIndex(
            video_id="roundtrip_test",
            filename="test.mp4",
            duration=60.0,
            filepath="/v/test.mp4",
        )
        report = self.generator.from_video_index(index)
        json_str = self.generator.to_json(report)
        assert isinstance(json_str, str)
        assert len(json_str) > 50

        # Save and reload
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            f.write(json_str)
            tmp_path = f.name

        loaded = self.generator.load(Path(tmp_path))
        assert loaded.video.video_id == "roundtrip_test"
        assert loaded.video.duration == 60.0
        assert loaded.schema_version == "1.0"
        Path(tmp_path).unlink()

    def test_save_and_load(self):
        """save() and load() should produce identical data."""
        index = VideoIndex(
            video_id="save_test",
            filename="test.mp4",
            duration=120.0,
            filepath="/v/test.mp4",
        )
        report = self.generator.from_video_index(index)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = Path(f.name)

        try:
            saved_path = self.generator.save(report, tmp_path)
            assert saved_path.exists()
            assert saved_path.suffix == ".json"

            loaded = self.generator.load(saved_path)
            assert loaded.video.video_id == "save_test"
            assert loaded.video.duration == 120.0
            assert loaded.schema_version == "1.0"
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_summary_text(self):
        """Summary text should be readable and contain key sections."""
        index = VideoIndex(
            video_id="summary_test",
            filename="summary_test.mp4",
            duration=100.0,
            filepath="/v/summary_test.mp4",
            scenes=[
                SceneInfo(
                    scene_id=0,
                    start_time=0.0,
                    end_time=50.0,
                    key_frames=[
                        FrameInfo(
                            timestamp=10.0,
                            filepath="/f.jpg",
                            objects=[{"label": "person"}],
                            description="A person speaks",
                        ),
                    ],
                    summary="Opening",
                ),
            ],
        )
        report = self.generator.from_video_index(index)
        summary = ReportGenerator.summary_text(report)
        assert "Video Analysis Report" in summary
        assert "summary_test" in summary
        assert "1:40" in summary  # 100 seconds = 1:40
        assert "Scenes" in summary

    def test_to_chunk_context(self):
        """Chunk context should be concise and LLM-friendly."""
        report = VideoReport(
            video=VideoMetadata(
                video_id="ctx_test", title="Context Test", duration=60.0
            ),
            timeline=TimelineSummary(num_scenes=5),
            objects=ObjectCatalog(
                unique_objects=["person", "dog"],
                total_detections=20,
                object_frequency={"person": 15, "dog": 5},
                top_objects=[("person", 15, 3), ("dog", 5, 2)],
            ),
            transcript=TranscriptReport(
                total_segments=8,
                total_words=500,
                speaker_count=2,
                speakers={
                    "SPEAKER_00": {
                        "segment_count": 4,
                        "total_words": 300,
                        "word_count_pct": 0.6,
                    },
                },
                key_phrases=["important", "keyword"],
            ),
        )
        ctx = ReportGenerator.to_chunk_context(report)
        assert "Context Test" in ctx
        assert "person" in ctx
        assert "# scenes: 5" in ctx or "Scenes:" in ctx or "5" in ctx

    def test_from_video_id_no_rag(self):
        """from_video_id without RAG should return a minimal report."""
        report = self.generator.from_video_id("no_rag_video")
        assert report.video.video_id == "no_rag_video"
        assert report.timeline.num_scenes == 0


class TestReportGeneratorPipelineVersion:
    def test_custom_pipeline_version(self):
        generator = ReportGenerator(
            config=Config(data_dir=tempfile.mkdtemp()), pipeline_version="0.51.0"
        )
        index = VideoIndex(
            video_id="ver_test",
            filename="test.mp4",
            duration=10.0,
            filepath="/v/test.mp4",
        )
        report = generator.from_video_index(index, processing_time=42.5)
        assert report.video.pipeline_version == "0.51.0"
        assert report.video.processing_time_seconds == 42.5


class TestChecksum:
    def test_compute_checksum(self):
        """Checksum should produce consistent hex output."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test video data" * 1000)
            tmp_path = Path(f.name)

        try:
            checksum = ReportGenerator._compute_checksum(tmp_path)
            assert len(checksum) == 16  # truncated SHA-256
            assert all(c in "0123456789abcdef" for c in checksum)
        finally:
            tmp_path.unlink()

    def test_checksum_nonexistent(self):
        checksum = ReportGenerator._compute_checksum(Path("/nonexistent/video.mp4"))
        assert checksum == ""
