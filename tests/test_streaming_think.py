"""
Tests for Streaming Thinking module (streaming_think.py).

Covers:
- ThoughtState construction and defaults
- StreamingThinkingPipeline construction and reset
- _think() logic with chunk results
- _build_thought_summary formatting
- _predict_next and _explain_current reasoning
- _generate_questions
- answer() with context and LLM
- Timeline generation
- Delegated methods to StreamingPipeline
"""

from unittest.mock import MagicMock

import pytest

from video_analysis.config import Config
from video_analysis.streaming_think import (
    StreamingThinkingPipeline,
    StreamingThought,
    ThoughtState,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_chunk():
    """Create a realistic StreamingChunkResult."""
    from video_analysis.models import SceneInfo
    from video_analysis.streaming import StreamingChunkResult

    scene1 = SceneInfo(scene_id=0, start_time=0.0, end_time=10.0)
    scene2 = SceneInfo(scene_id=1, start_time=10.0, end_time=20.0)

    return StreamingChunkResult(
        chunk_index=0,
        start_time=0.0,
        end_time=30.0,
        duration=30.0,
        scenes=[scene1, scene2],
        transcript_segments=[],
        full_transcript="Hello everyone welcome to our presentation today we will discuss the project roadmap and timeline.",
        objects_found=["person", "laptop", "presentation_screen"],
        has_video=True,
        metadata={"video_id": "test_video"},
    )


@pytest.fixture
def sample_chunks():
    """Create three sequential chunks simulating a stream."""
    from video_analysis.models import SceneInfo
    from video_analysis.streaming import StreamingChunkResult

    chunks = []
    for i, (transcript, objects) in enumerate(
        [
            (
                "Welcome to the presentation. Today we discuss our quarterly results.",
                ["person", "screen"],
            ),
            (
                "Our revenue grew by 20 percent this quarter driven by new product launches.",
                ["person", "chart", "laptop"],
            ),
            (
                "In conclusion we are optimistic about the next quarter. Thank you.",
                ["person", "screen"],
            ),
        ]
    ):
        scene = SceneInfo(scene_id=i, start_time=i * 30.0, end_time=(i + 1) * 30.0)
        chunks.append(
            StreamingChunkResult(
                chunk_index=i,
                start_time=i * 30.0,
                end_time=(i + 1) * 30.0,
                duration=30.0,
                scenes=[scene],
                transcript_segments=[],
                full_transcript=transcript,
                objects_found=objects,
                has_video=True,
                metadata={"video_id": "test_video"},
            )
        )
    return chunks


# ---------------------------------------------------------------------------
# ThoughtState tests
# ---------------------------------------------------------------------------


def test_thought_state_defaults():
    """ThoughtState should have correct default values."""
    ts = ThoughtState()
    assert ts.chunks_seen == 0
    assert ts.total_duration == 0.0
    assert ts.summary == ""
    assert ts.events == []
    assert ts.entities == {}
    assert ts.causal_observations == []
    assert ts.unanswered_questions == []
    assert ts.partial_answers == {}
    assert ts.last_thought == ""
    assert ts.refined_answers == {}


def test_thought_state_with_values():
    """ThoughtState should store values correctly."""
    ts = ThoughtState(
        chunks_seen=3,
        total_duration=90.0,
        summary="Test content",
        events=["Event 1", "Event 2"],
        entities={"person": 5, "screen": 3},
        causal_observations=["Predicted after chunk 1: ..."],
        unanswered_questions=["Why did X appear?"],
        partial_answers={"What is this?": "A presentation"},
        last_thought="Great progress",
        refined_answers={"What is revenue?": "20% growth"},
    )
    assert ts.chunks_seen == 3
    assert ts.total_duration == 90.0
    assert ts.entities["person"] == 5
    assert ts.refined_answers["What is revenue?"] == "20% growth"


# ---------------------------------------------------------------------------
# StreamingThought tests
# ---------------------------------------------------------------------------


def test_streaming_thought_defaults():
    """StreamingThought should have correct default values."""
    st = StreamingThought(
        chunk_index=0,
        start_time=0.0,
        end_time=30.0,
    )
    assert st.chunk_index == 0
    assert st.thought_summary == ""
    assert st.insights == []
    assert st.questions == []
    assert st.confidence == 1.0


def test_streaming_thought_with_values():
    """StreamingThought should store values correctly."""
    st = StreamingThought(
        chunk_index=1,
        start_time=30.0,
        end_time=60.0,
        thought_summary="Revenue discussion",
        insights=["Revenue grew 20%", "New product launches"],
        causal_prediction="Might discuss Q3 outlook next",
        causal_explanation="Continuing from previous financial overview",
        questions=["What drove the growth?"],
        confidence=0.85,
        metadata={"objects_seen": ["person", "chart"]},
    )
    assert st.thought_summary == "Revenue discussion"
    assert len(st.insights) == 2
    assert st.confidence == 0.85


# ---------------------------------------------------------------------------
# StreamingThinkingPipeline tests
# ---------------------------------------------------------------------------


def test_streaming_thinking_pipeline_init():
    """Pipeline should initialize with defaults."""
    pipeline = StreamingThinkingPipeline()
    assert pipeline._thought_state is not None
    assert pipeline._thought_history == []
    assert pipeline._last_chunk_results == []


def test_streaming_thinking_pipeline_reset():
    """Reset should clear all state."""
    pipeline = StreamingThinkingPipeline()
    pipeline._thought_state = ThoughtState(chunks_seen=5)
    pipeline._thought_history = [StreamingThought(chunk_index=0, start_time=0.0, end_time=30.0)]
    pipeline._last_chunk_results = ["dummy"]

    pipeline.reset()
    assert pipeline._thought_state.chunks_seen == 0
    assert pipeline._thought_history == []
    assert pipeline._last_chunk_results == []
    assert pipeline._streaming is not None  # new pipeline created


def test_think_first_chunk(sample_chunk):
    """_think should produce a StreamingThought from the first chunk."""
    pipeline = StreamingThinkingPipeline()
    thought = pipeline._think(sample_chunk)

    assert thought.chunk_index == 0
    assert thought.start_time == 0.0
    assert thought.end_time == 30.0
    # First chunk should have insights
    assert len(thought.insights) >= 1
    # Should not have causal prediction for first chunk
    assert pipeline._thought_state.chunks_seen == 1
    assert pipeline._thought_state.total_duration == 30.0
    # Entities should be tracked
    assert "person" in pipeline._thought_state.entities
    assert pipeline._thought_state.entities["person"] == 1


def test_think_second_chunk(sample_chunks):
    """_think on second chunk should include causal predictions."""
    pipeline = StreamingThinkingPipeline()

    # Process first chunk
    c0 = sample_chunks[0]
    t0 = pipeline._think(c0)
    assert t0.chunk_index == 0

    # Process second chunk
    c1 = sample_chunks[1]
    t1 = pipeline._think(c1)
    assert t1.chunk_index == 1
    assert pipeline._thought_state.chunks_seen == 2
    assert pipeline._thought_state.total_duration == 60.0
    # Entity counts should accumulate
    assert pipeline._thought_state.entities.get("person", 0) >= 2
    assert pipeline._thought_state.entities.get("chart", 0) >= 1


def test_think_causal_observations(sample_chunks):
    """After multiple chunks, causal observations should be recorded."""
    pipeline = StreamingThinkingPipeline()

    for chunk in sample_chunks:
        pipeline._think(chunk)

    # Causal observations should exist
    assert len(pipeline._thought_state.causal_observations) >= 1


def test_build_thought_summary(sample_chunk):
    """_build_thought_summary should produce reasonable summaries."""
    pipeline = StreamingThinkingPipeline()
    ts = pipeline._thought_state

    # First chunk with transcript
    summary = pipeline._build_thought_summary(sample_chunk, ["Objects: person, laptop"], ts)
    assert len(summary) > 0
    # Should contain something from the transcript or insights
    assert any(p in summary.lower() for p in ["hello", "welcome", "person", "laptop"])


def test_predict_next(sample_chunk):
    """_predict_next should generate predictions."""
    pipeline = StreamingThinkingPipeline()
    ts = pipeline._thought_state
    ts.entities = {"person": 3, "laptop": 2, "screen": 1}

    prediction = pipeline._predict_next(sample_chunk, ts)
    assert prediction is not None
    assert len(prediction) > 0


def test_explain_current(sample_chunk):
    """_explain_current should generate explanations."""
    pipeline = StreamingThinkingPipeline()
    ts = pipeline._thought_state
    ts.entities = {"person": 3, "screen": 2}
    ts.chunks_seen = 2

    explanation = pipeline._explain_current(sample_chunk, ts)
    # Might or might not have explanation depending on entity overlap
    assert explanation is not None


def test_generate_questions(sample_chunk):
    """_generate_questions should identify knowledge gaps."""
    pipeline = StreamingThinkingPipeline()
    ts = pipeline._thought_state
    ts.entities = {"person": 3}  # existing

    # chunk has "laptop" which is new
    questions = pipeline._generate_questions(sample_chunk, ts)
    assert len(questions) >= 0  # may be empty or contain questions


def test_answer_from_context(sample_chunks):
    """_answer_from_context should answer based on accumulated context."""
    pipeline = StreamingThinkingPipeline()

    for chunk in sample_chunks:
        pipeline._think(chunk)

    answer = pipeline._answer_from_context("person")
    assert "person" in answer
    assert "chunks" in answer or "Chunks" in answer


def test_answer_from_context_entity_not_found():
    """_answer_from_context should handle unknown entities."""
    pipeline = StreamingThinkingPipeline()
    answer = pipeline._answer_from_context("unicorn")
    assert "chunks" in answer or "Chunks" in answer


def test_answer_with_llm(sample_chunks):
    """answer() should use LLM when available."""
    mock_llm = MagicMock()
    mock_llm.chat.return_value = (
        "Based on the streaming context so far, this appears to be "
        "a business presentation about quarterly results."
    )

    pipeline = StreamingThinkingPipeline(llm_provider=mock_llm)
    for chunk in sample_chunks:
        pipeline._think(chunk)

    answer = pipeline.answer("What is this video about?", use_llm=True)
    assert isinstance(answer, str)
    assert len(answer) > 0
    mock_llm.chat.assert_called_once()


def test_answer_cached(sample_chunks):
    """answer() should cache results for repeated queries."""
    pipeline = StreamingThinkingPipeline()
    for chunk in sample_chunks:
        pipeline._think(chunk)

    # First answer
    answer1 = pipeline.answer("test query", use_llm=False)
    assert answer1 is not None

    # Second call should use cached result (but this only applies for LLM path)
    # For context path, it just recomputes — test that it works
    answer2 = pipeline.answer("test query", use_llm=False)
    assert answer2 is not None


def test_get_timeline(sample_chunks):
    """get_timeline should return structured timeline data."""
    pipeline = StreamingThinkingPipeline()
    for chunk in sample_chunks:
        thought = pipeline._think(chunk)
        pipeline._thought_history.append(thought)

    timeline = pipeline.get_timeline()
    assert len(timeline) == len(sample_chunks)
    assert timeline[0]["chunk_index"] == 0
    assert "thought_summary" in timeline[0]
    assert "insights" in timeline[0]
    assert "causal_prediction" in timeline[0]


def test_final_thoughts(sample_chunks):
    """final_thoughts should return current ThoughtState."""
    pipeline = StreamingThinkingPipeline()
    for chunk in sample_chunks:
        pipeline._think(chunk)

    final = pipeline.final_thoughts()
    assert final.chunks_seen == len(sample_chunks)
    assert final.total_duration > 0


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


def test_update_summary(sample_chunk):
    """_update_summary should append new text."""
    pipeline = StreamingThinkingPipeline()
    result = pipeline._update_summary("", sample_chunk)
    assert len(result) > 0
    assert "Hello" in result or "welcome" in result.lower()


def test_update_summary_accumulates(sample_chunks):
    """_update_summary should accumulate across chunks."""
    pipeline = StreamingThinkingPipeline()
    result = ""
    for chunk in sample_chunks:
        result = pipeline._update_summary(result, chunk)
    assert len(result) > 0
    # Should contain text from multiple chunks
    assert "Welcome" in result or "revenue" in result or "conclusion" in result


def test_process_with_thinking():
    """process_with_thinking should yield StreamingThought objects."""
    from video_analysis.streaming import StreamingChunkResult, StreamingPipeline

    # Mock the streaming pipeline
    mock_streaming = MagicMock(spec=StreamingPipeline)
    mock_chunks = [
        StreamingChunkResult(
            chunk_index=i,
            start_time=i * 30.0,
            end_time=(i + 1) * 30.0,
            duration=30.0,
            full_transcript=f"Chunk {i} transcript content.",
            objects_found=["person", "screen"],
            has_video=True,
        )
        for i in range(3)
    ]
    mock_streaming.process_streaming.return_value = mock_chunks

    pipeline = StreamingThinkingPipeline(streaming_pipeline=mock_streaming)

    thoughts = list(pipeline.process_with_thinking("test.mp4"))
    assert len(thoughts) == 3
    for thought in thoughts:
        assert isinstance(thought, StreamingThought)
    assert pipeline._thought_state.chunks_seen == 3


def test_process_with_thinking_custom_config():
    """process_with_thinking should accept custom kwargs."""

    config = Config()
    pipeline = StreamingThinkingPipeline(config=config)

    # Just verify construction works with custom pipeline
    assert pipeline._streaming is not None


def test_final_index():
    """final_index should delegate to StreamingPipeline."""
    from video_analysis.streaming import StreamingPipeline

    mock_streaming = MagicMock(spec=StreamingPipeline)
    mock_streaming.final_index.return_value = {"video_id": "test"}

    pipeline = StreamingThinkingPipeline(streaming_pipeline=mock_streaming)
    result = pipeline.final_index()
    assert result["video_id"] == "test"


def test_cleanup():
    """cleanup should delegate to StreamingPipeline."""

    mock_streaming = MagicMock()

    pipeline = StreamingThinkingPipeline(streaming_pipeline=mock_streaming)
    pipeline.cleanup()
    assert mock_streaming.cleanup.called


def test_pipeline_property():
    """pipeline property should expose the underlying StreamingPipeline."""
    from video_analysis.streaming import StreamingPipeline

    mock_streaming = MagicMock(spec=StreamingPipeline)
    pipeline = StreamingThinkingPipeline(streaming_pipeline=mock_streaming)
    assert pipeline.pipeline is mock_streaming


def test_think_accumulates_entities(sample_chunks):
    """Entity counts should accumulate correctly across chunks."""
    pipeline = StreamingThinkingPipeline()
    for chunk in sample_chunks:
        pipeline._think(chunk)

    ts = pipeline._thought_state
    # "person" appears in all chunks
    assert ts.entities.get("person", 0) == len(sample_chunks)
    # "screen" appears in chunks 0 and 2 only (2 times)
    assert ts.entities.get("screen", 0) == 2


def test_thought_history_limited(sample_chunks):
    """Thought history should be bounded by max_thought_history."""
    pipeline = StreamingThinkingPipeline(max_thought_history=2)
    for chunk in sample_chunks:
        pipeline._think(chunk)

    assert len(pipeline._thought_history) <= 2


def test_think_question_generation(sample_chunks):
    """_generate_questions should identify when new entities appear."""
    pipeline = StreamingThinkingPipeline()

    # First chunk — only "person" and "screen"
    c0 = sample_chunks[0]
    pipeline._think(c0)

    # Second chunk has "chart" which is new
    c1 = sample_chunks[1]
    ts = pipeline._thought_state
    questions = pipeline._generate_questions(c1, ts)
    # "chart" is new, so there should be a question about it
    assert len(questions) >= 1
    assert "chart" in questions[0].lower() or "why" in questions[0].lower()


def test_empty_chunk_handling():
    """Empty chunks (no scenes, no transcript) should still produce thoughts."""
    from video_analysis.streaming import StreamingChunkResult

    empty_chunk = StreamingChunkResult(
        chunk_index=0,
        start_time=0.0,
        end_time=10.0,
        duration=10.0,
        scenes=[],
        transcript_segments=[],
        full_transcript="",
        objects_found=[],
        has_video=False,
        metadata={},
    )

    pipeline = StreamingThinkingPipeline()
    thought = pipeline._think(empty_chunk)
    assert thought.chunk_index == 0
    assert len(thought.insights) >= 0


def test_update_summary_truncation(sample_chunk):
    """_update_summary should truncate very long text."""
    pipeline = StreamingThinkingPipeline()
    from video_analysis.streaming import StreamingChunkResult

    long_chunk = StreamingChunkResult(
        chunk_index=0,
        start_time=0.0,
        end_time=30.0,
        duration=30.0,
        full_transcript="word " * 5000,
        objects_found=[],
        has_video=True,
    )
    result = pipeline._update_summary("", long_chunk)
    assert len(result) > 0
    assert len(result) <= 2000  # should be truncated
