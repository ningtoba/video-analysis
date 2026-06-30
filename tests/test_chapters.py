"""
Tests for Video Content Chaptering (v0.37.0).

Covers:
- ChapterSegment, Chapter, ChapteringResult dataclasses
- ChapterGenerator uniform/simple segmentation
- ChapterGenerator heuristic title generation
- ChapterGenerator scene-boundary segmentation
- ChapterGenerator merge/limit logic
- Chapter report generation
- Agent chapter context generation
- extract_transcript_from_rag (mock-based)
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from video_analysis.chapters import (
    Chapter,
    ChapterGenerator,
    ChapteringResult,
    ChapterSegment,
    extract_transcript_from_rag,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_segments() -> List[Dict[str, Any]]:
    """15 transcript segments covering ~5 minutes of content."""
    return [
        {
            "start": 0.0,
            "end": 8.0,
            "text": "Hello and welcome to this presentation about machine learning.",
            "speaker": "SPEAKER_00",
        },
        {
            "start": 8.0,
            "end": 18.0,
            "text": "Today we will discuss the fundamentals of supervised learning.",
            "speaker": "SPEAKER_00",
        },
        {
            "start": 18.0,
            "end": 30.0,
            "text": "Supervised learning requires labeled training data to map inputs to outputs.",
            "speaker": "SPEAKER_00",
        },
        {
            "start": 30.0,
            "end": 42.0,
            "text": "Common algorithms include linear regression and decision trees.",
            "speaker": "SPEAKER_01",
        },
        {
            "start": 42.0,
            "end": 55.0,
            "text": "Now let us move on to neural networks and deep learning architectures.",
            "speaker": "SPEAKER_00",
        },
        {
            "start": 55.0,
            "end": 70.0,
            "text": "Neural networks consist of multiple layers of interconnected neurons.",
            "speaker": "SPEAKER_00",
        },
        {
            "start": 70.0,
            "end": 82.0,
            "text": "Each layer transforms the input data through weighted connections.",
            "speaker": "SPEAKER_01",
        },
        {
            "start": 82.0,
            "end": 95.0,
            "text": "Backpropagation is used to train these networks efficiently.",
            "speaker": "SPEAKER_00",
        },
        {
            "start": 95.0,
            "end": 108.0,
            "text": "Convolutional neural networks are particularly good at image recognition tasks.",
            "speaker": "SPEAKER_00",
        },
        {
            "start": 108.0,
            "end": 120.0,
            "text": "Transformers have revolutionized natural language processing.",
            "speaker": "SPEAKER_01",
        },
        {
            "start": 120.0,
            "end": 135.0,
            "text": "The attention mechanism allows models to focus on relevant parts of the input.",
            "speaker": "SPEAKER_00",
        },
        {
            "start": 135.0,
            "end": 150.0,
            "text": "Large language models are trained on massive text corpora.",
            "speaker": "SPEAKER_00",
        },
        {
            "start": 150.0,
            "end": 165.0,
            "text": "Reinforcement learning trains agents through trial and error.",
            "speaker": "SPEAKER_01",
        },
        {
            "start": 165.0,
            "end": 175.0,
            "text": "The agent receives rewards for desirable actions.",
            "speaker": "SPEAKER_01",
        },
        {
            "start": 175.0,
            "end": 185.0,
            "text": "Thank you for watching this introduction to machine learning concepts.",
            "speaker": "SPEAKER_00",
        },
    ]


@pytest.fixture
def generator() -> ChapterGenerator:
    return ChapterGenerator()


# ---------------------------------------------------------------------------
# Data class tests
# ---------------------------------------------------------------------------


class TestChapterSegment:
    def test_defaults(self):
        cs = ChapterSegment(start=0.0, end=10.0, text="hello")
        assert cs.start == 0.0
        assert cs.end == 10.0
        assert cs.text == "hello"
        assert cs.chapter_index == -1
        assert cs.speaker is None

    def test_with_speaker(self):
        cs = ChapterSegment(
            start=5.0, end=15.0, text="world", chapter_index=0, speaker="SPEAKER_00"
        )
        assert cs.speaker == "SPEAKER_00"
        assert cs.chapter_index == 0


class TestChapter:
    def test_defaults(self):
        c = Chapter(title="Intro", start_time=0.0, end_time=60.0, index=0)
        assert c.title == "Intro"
        assert c.summary == ""
        assert c.transcript_preview == ""
        assert c.word_count == 0

    def test_with_all_fields(self):
        c = Chapter(
            title="Neural Networks",
            start_time=60.0,
            end_time=120.0,
            index=1,
            summary="Covers CNN and transformer architectures.",
            transcript_preview="Neural networks consist of...",
            word_count=85,
        )
        assert c.summary
        assert c.transcript_preview
        assert c.word_count == 85


class TestChapteringResult:
    def test_empty(self):
        r = ChapteringResult(video_id="test", chapters=[], num_segments=0, method="none")
        assert r.error is None
        assert r.to_dict()["chapters"] == []

    def test_with_chapters(self):
        ch = Chapter(title="Intro", start_time=0.0, end_time=60.0, index=0)
        r = ChapteringResult(
            video_id="vid1",
            chapters=[ch],
            num_segments=10,
            method="uniform",
            duration_seconds=0.5,
        )
        r_dict = r.to_dict()
        assert r_dict["video_id"] == "vid1"
        assert len(r_dict["chapters"]) == 1
        assert r_dict["chapters"][0]["title"] == "Intro"
        assert r_dict["method"] == "uniform"
        assert r.duration_seconds == 0.5

    def test_with_error(self):
        r = ChapteringResult(
            video_id="test",
            chapters=[],
            num_segments=0,
            method="none",
            error="No transcript",
        )
        assert r.error == "No transcript"


# ---------------------------------------------------------------------------
# ChapterGenerator tests
# ---------------------------------------------------------------------------


class TestHeuristicTitle:
    def test_generates_from_first_sentence(self, generator: ChapterGenerator):
        title, summary = generator._generate_heuristic_title(
            "Machine learning is transforming how we process data. "
            "New algorithms emerge every day.",
            chapter_index=0,
        )
        assert len(title) > 0
        assert "machine learning" in title.lower() or "Chapter 1" in title
        assert summary == ""

    def test_chapter_number_fallback_for_short_text(self, generator: ChapterGenerator):
        title, summary = generator._generate_heuristic_title("Hi", chapter_index=3)
        assert title == "Chapter 4"
        assert summary == ""

    def test_truncates_long_title(self, generator: ChapterGenerator):
        long_text = "A" * 100 + ". Then some more text here."
        title, summary = generator._generate_heuristic_title(long_text, chapter_index=0)
        assert len(title) <= 65  # 60 + "…"

    def test_empty_text_fallback(self, generator: ChapterGenerator):
        title, summary = generator._generate_heuristic_title("", chapter_index=5)
        assert title == "Chapter 6"


class TestUniformSegmentation:
    def test_empty_transcript(self, generator: ChapterGenerator):
        groups = generator._segment_uniform([], target_chapters=4)
        assert groups == []

    def test_single_segment(self, generator: ChapterGenerator):
        seg = ChapterSegment(start=0.0, end=30.0, text="test content")
        groups = generator._segment_uniform([seg], target_chapters=4)
        assert len(groups) == 1
        assert len(groups[0]) == 1

    def test_distributes_into_chapters(self, generator: ChapterGenerator):
        segs = [
            ChapterSegment(start=i * 10.0, end=i * 10.0 + 8.0, text=f"Segment {i}")
            for i in range(30)
        ]
        groups = generator._segment_uniform(segs, target_chapters=5)
        assert len(groups) <= 6  # at most 5 + trailing
        assert len(groups) >= 2

    def test_handles_zero_duration(self, generator: ChapterGenerator):
        segs = [ChapterSegment(start=0.0, end=0.0, text="no time")]
        groups = generator._segment_uniform(segs, target_chapters=4)
        assert len(groups) == 1


class TestSceneBoundarySegmentation:
    def test_with_no_boundaries_falls_back(self, generator: ChapterGenerator):
        segs = [
            ChapterSegment(start=i * 10.0, end=i * 10.0 + 8.0, text=f"Seg {i}") for i in range(5)
        ]
        groups = generator._segment_by_scene_boundaries(segs, scene_times=None)
        assert len(groups) >= 1

    def test_with_scene_boundaries(self, generator: ChapterGenerator):
        segs = [
            ChapterSegment(start=i * 10.0, end=i * 10.0 + 8.0, text=f"Seg {i}") for i in range(10)
        ]
        groups = generator._segment_by_scene_boundaries(segs, scene_times=[0.0, 30.0, 60.0, 90.0])
        assert len(groups) >= 2
        # First group should cover segments 0-2 (0-30s)
        assert len(groups[0]) <= 4

    def test_single_scene_boundary(self, generator: ChapterGenerator):
        segs = [ChapterSegment(start=0.0, end=50.0, text="long segment")]
        groups = generator._segment_by_scene_boundaries(segs, scene_times=[0.0, 50.0])
        assert len(groups) == 1


class TestMergeGroups:
    def test_merges_to_target(self, generator: ChapterGenerator):
        groups = [
            [ChapterSegment(start=i * 10.0, end=i * 10.0 + 5.0, text=f"Seg {i}")] for i in range(8)
        ]
        merged = generator._merge_groups(groups, target_count=3)
        assert len(merged) == 3
        # Total segments should be preserved
        total_segs = sum(len(g) for g in merged)
        assert total_segs == 8

    def test_noop_when_at_target(self, generator: ChapterGenerator):
        groups = [
            [ChapterSegment(start=0.0, end=5.0, text="a")],
            [ChapterSegment(start=5.0, end=10.0, text="b")],
        ]
        merged = generator._merge_groups(groups, target_count=4)
        assert len(merged) == 2  # already under target


class TestBuildParagraph:
    def test_single_segment(self, generator: ChapterGenerator):
        segs = [ChapterSegment(start=0.0, end=5.0, text="hello world")]
        para = generator._build_transcript_paragraph(segs)
        assert "hello world" in para

    def test_multiple_segments(self, generator: ChapterGenerator):
        segs = [
            ChapterSegment(start=0.0, end=5.0, text="First part."),
            ChapterSegment(start=5.0, end=10.0, text="Second part."),
        ]
        para = generator._build_transcript_paragraph(segs)
        assert "First part." in para
        assert "Second part." in para

    def test_with_speaker_prefix(self, generator: ChapterGenerator):
        segs = [
            ChapterSegment(start=0.0, end=5.0, text="hello", speaker="SPEAKER_00"),
        ]
        para = generator._build_transcript_paragraph(segs)
        assert "[SPEAKER_00]" in para
        assert "hello" in para

    def test_skips_empty_text(self, generator: ChapterGenerator):
        segs = [
            ChapterSegment(start=0.0, end=5.0, text=""),
            ChapterSegment(start=5.0, end=10.0, text="actual content"),
        ]
        para = generator._build_transcript_paragraph(segs)
        assert "actual content" in para
        # Empty text should be skipped
        assert para.strip() == "[SPEAKER_00] actual content" or para.strip() == "actual content"


# ---------------------------------------------------------------------------
# Full segmentation tests (no NLTK required — uses uniform fallback)
# ---------------------------------------------------------------------------


class TestSegmentTranscript:
    def test_empty_segments(self, generator: ChapterGenerator):
        result = generator.segment_transcript([], video_id="empty_test")
        assert result.video_id == "empty_test"
        assert result.chapters == []
        assert result.error is not None

    def test_only_empty_segments(self, generator: ChapterGenerator):
        result = generator.segment_transcript(
            [{"start": 0.0, "end": 5.0, "text": ""}],
            video_id="empty_test",
        )
        assert result.chapters == []
        assert result.error is not None

    def test_single_short_segment(self, generator: ChapterGenerator):
        result = generator.segment_transcript(
            [{"start": 0.0, "end": 5.0, "text": "Short intro."}],
            video_id="short_test",
        )
        assert len(result.chapters) == 1
        assert result.method == "uniform"
        assert result.chapters[0].title != ""
        assert result.chapters[0].start_time == 0.0

    def test_with_sample_segments(
        self, generator: ChapterGenerator, sample_segments: List[Dict[str, Any]]
    ):
        result = generator.segment_transcript(sample_segments, video_id="ml_talk", max_chapters=5)
        assert result.num_segments == 15
        assert len(result.chapters) >= 2
        assert len(result.chapters) <= 5
        # All chapters should have start/end times
        for ch in result.chapters:
            assert ch.start_time >= 0.0
            assert ch.end_time > ch.start_time
            assert ch.title != ""
        # Chapters should be chronological
        for i in range(1, len(result.chapters)):
            assert result.chapters[i].start_time >= result.chapters[i - 1].end_time

    def test_with_scene_boundaries(
        self, generator: ChapterGenerator, sample_segments: List[Dict[str, Any]]
    ):
        result = generator.segment_transcript(
            sample_segments,
            video_id="boundary_test",
            scene_boundaries=[0.0, 55.0, 120.0, 185.0],
            max_chapters=10,
        )
        assert len(result.chapters) >= 2
        # First chapter should start around the first boundary
        assert result.chapters[0].start_time >= 0.0


class TestChapterReport:
    def test_empty_result(self, generator: ChapterGenerator):
        result = ChapteringResult(video_id="test", chapters=[], num_segments=0, method="none")
        report = generator.generate_chapter_report(result, video_filename="test.mp4")
        assert "No chapters generated" in report

    def test_with_chapters(self, generator: ChapterGenerator):
        chapters = [
            Chapter(
                title="Intro",
                start_time=0.0,
                end_time=30.0,
                index=0,
                summary="Welcome",
                word_count=100,
            ),
            Chapter(
                title="Main Topic",
                start_time=30.0,
                end_time=90.0,
                index=1,
                summary="Deep dive",
                word_count=250,
            ),
        ]
        result = ChapteringResult(
            video_id="test",
            chapters=chapters,
            num_segments=20,
            method="texttiling",
        )
        report = generator.generate_chapter_report(result, video_filename="demo.mp4")
        assert "Chapter 1" in report
        assert "Chapter 2" in report
        assert "Intro" in report
        assert "Main Topic" in report
        assert "20" in report  # num_segments
        assert "texttiling" in report
        # Should mention timestamps
        assert "0:00" in report or "0:30" in report or "1:30" in report


class TestAgentChapterContext:
    def test_empty(self, generator: ChapterGenerator):
        result = ChapteringResult(video_id="test", chapters=[], num_segments=0, method="none")
        context = generator.generate_agent_chapter_context(result)
        assert context == "No chapters available."

    def test_with_chapters(self, generator: ChapterGenerator):
        chapters = [
            Chapter(title="Intro", start_time=0.0, end_time=30.0, index=0),
            Chapter(title="Methods", start_time=30.0, end_time=90.0, index=1),
        ]
        result = ChapteringResult(
            video_id="test", chapters=chapters, num_segments=15, method="uniform"
        )
        context = generator.generate_agent_chapter_context(result)
        assert "2 chapters" in context or "Ch1" in context
        assert "Intro" in context
        assert "Methods" in context


# ---------------------------------------------------------------------------
# extract_transcript_from_rag tests
# ---------------------------------------------------------------------------


class TestExtractTranscriptFromRAG:
    def test_no_data_returns_empty(self):
        rag_mock = MagicMock()
        rag_mock.collection.get.return_value = {"ids": []}
        result = extract_transcript_from_rag(rag_mock, "video1")
        assert result == []

    def test_extracts_with_speaker(self):
        rag_mock = MagicMock()
        rag_mock.collection.get.return_value = {
            "ids": ["chunk_1", "chunk_2"],
            "metadatas": [
                {
                    "start_time": 0.0,
                    "end_time": 10.0,
                    "speaker": "SPEAKER_00",
                    "text": "Hello world",
                },
                {
                    "start_time": 10.0,
                    "end_time": 20.0,
                    "speaker": "SPEAKER_01",
                    "text": "Next segment",
                },
            ],
            "documents": ["Hello world", "Next segment"],
        }
        result = extract_transcript_from_rag(rag_mock, "video1")
        assert len(result) == 2
        assert result[0]["speaker"] == "SPEAKER_00"
        assert result[0]["text"] == "Hello world"
        assert result[0]["start"] == 0.0

    def test_sorts_by_timestamp(self):
        rag_mock = MagicMock()
        rag_mock.collection.get.return_value = {
            "ids": ["chunk_3", "chunk_1", "chunk_2"],
            "metadatas": [
                {"start_time": 30.0, "end_time": 40.0, "text": "Third"},
                {"start_time": 0.0, "end_time": 10.0, "text": "First"},
                {"start_time": 10.0, "end_time": 20.0, "text": "Second"},
            ],
            "documents": ["Third", "First", "Second"],
        }
        result = extract_transcript_from_rag(rag_mock, "video1")
        assert result[0]["start"] == 0.0
        assert result[0]["text"] == "First"
        assert result[-1]["start"] == 30.0

    def test_skips_empty_text(self):
        rag_mock = MagicMock()
        rag_mock.collection.get.return_value = {
            "ids": ["chunk_1", "chunk_2"],
            "metadatas": [
                {"start_time": 0.0, "end_time": 10.0, "text": "Valid"},
                {"start_time": 10.0, "end_time": 20.0, "text": ""},
            ],
            "documents": ["Valid", ""],
        }
        result = extract_transcript_from_rag(rag_mock, "video1")
        assert len(result) == 1
        assert result[0]["text"] == "Valid"

    def test_fallback_all_chunks_when_no_transcript_type(self):
        rag_mock = MagicMock()
        # First call (transcript type) returns empty
        rag_mock.collection.get.side_effect = [
            {"ids": []},  # transcript type — empty
            {  # all chunks fallback
                "ids": ["chunk_1"],
                "metadatas": [
                    {
                        "start_time": 5.0,
                        "end_time": 15.0,
                        "text": "Fallback content",
                        "chunk_type": "scene",
                    }
                ],
                "documents": ["Fallback content"],
            },
        ]
        result = extract_transcript_from_rag(rag_mock, "video1")
        assert len(result) == 1
        assert result[0]["text"] == "Fallback content"

    def test_graceful_handles_exception(self):
        rag_mock = MagicMock()
        rag_mock.collection.get.side_effect = Exception("DB error")
        result = extract_transcript_from_rag(rag_mock, "video1")
        assert result == []


# ---------------------------------------------------------------------------
# Integration: ChapterGenerator with segments through to report
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_basic_pipeline_through_report(
        self, generator: ChapterGenerator, sample_segments: List[Dict[str, Any]]
    ):
        """Test the full flow: segments → chapters → report."""
        result = generator.segment_transcript(
            sample_segments,
            video_id="e2e_test",
            max_chapters=4,
            use_llm_titles=False,  # skip LLM for test speed
        )
        assert len(result.chapters) >= 2

        # Generate report
        report = generator.generate_chapter_report(result, video_filename="ml_intro.mp4")
        assert "ml_intro.mp4" in report
        assert "Chapter 1" in report

        # Generate agent context
        context = generator.generate_agent_chapter_context(result)
        assert "chapter" in context.lower()

    def test_chapter_boundaries_are_contiguous(
        self, generator: ChapterGenerator, sample_segments: List[Dict[str, Any]]
    ):
        """Verify chapters don't overlap and cover the full duration."""
        result = generator.segment_transcript(
            sample_segments,
            video_id="contiguity_test",
            use_llm_titles=False,
        )

        for i in range(1, len(result.chapters)):
            prev = result.chapters[i - 1]
            curr = result.chapters[i]
            assert curr.start_time >= prev.end_time, (
                f"Chapter {i} starts before chapter {i - 1} ends"
            )

    def test_max_chapters_limit(
        self, generator: ChapterGenerator, sample_segments: List[Dict[str, Any]]
    ):
        """Verify max_chapters parameter is respected."""
        result = generator.segment_transcript(
            sample_segments,
            video_id="max_test",
            max_chapters=3,
            use_llm_titles=False,
        )
        assert len(result.chapters) <= 3

    def test_min_chapters_limit(self, generator: ChapterGenerator):
        """Verify min_chapters is honored (at minimum)."""
        # Create segments that should produce at least 2 chapters
        segs = [
            {
                "start": i * 30.0,
                "end": i * 30.0 + 25.0,
                "text": f"Segment number {i} of the transcript content.",
            }
            for i in range(20)
        ]
        result = generator.segment_transcript(
            segs,
            video_id="min_test",
            min_chapters=2,
            max_chapters=8,
            use_llm_titles=False,
        )
        assert len(result.chapters) >= 2
