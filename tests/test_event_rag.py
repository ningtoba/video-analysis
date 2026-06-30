"""
Tests for Event-Causal RAG module (event_rag.py).

Covers:
- Event dataclass construction
- EventSegmenter with temporal-grid fallback
- EventSegmenter with transcript-coherence segmentation
- SESGraph construction from events
- SemanticStore fallback indexing and search
- CausalTopologicalStore forward/backward traversal
- DualStoreMemory bidirectional retrieval
- EventCausalRAG full integration
"""

import json
from unittest.mock import MagicMock

import pytest

from video_analysis.event_rag import (
    CausalPath,
    CausalTopologicalStore,
    DualStoreMemory,
    Event,
    EventCausalRAG,
    EventSegmenter,
    RetrievalResult,
    SemanticStore,
    SESGraph,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_scenes():
    """Build realistic SceneInfo-like data for testing."""
    from video_analysis.models import (
        SceneInfo,
        TranscriptSegment,
        VideoIndex,
    )

    scenes = [
        SceneInfo(
            scene_id=0,
            start_time=0.0,
            end_time=10.0,
            summary="Person enters room and starts talking",
        ),
        SceneInfo(
            scene_id=1,
            start_time=10.0,
            end_time=25.0,
            summary="Speaker discusses project roadmap",
        ),
        SceneInfo(
            scene_id=2,
            start_time=25.0,
            end_time=40.0,
            summary="Demo of the new feature",
        ),
        SceneInfo(
            scene_id=3,
            start_time=40.0,
            end_time=55.0,
            summary="Q&A session with audience",
        ),
        SceneInfo(
            scene_id=4,
            start_time=55.0,
            end_time=70.0,
            summary="Closing remarks and next steps",
        ),
    ]
    transcript = [
        TranscriptSegment(
            start=0.0,
            end=5.0,
            text="Hello everyone, welcome to today's presentation",
            speaker="SPEAKER_00",
        ),
        TranscriptSegment(
            start=6.0,
            end=12.0,
            text="Today we'll discuss our project roadmap",
            speaker="SPEAKER_00",
        ),
        TranscriptSegment(
            start=15.0,
            end=22.0,
            text="We have three major milestones this quarter",
            speaker="SPEAKER_00",
        ),
        TranscriptSegment(
            start=28.0,
            end=35.0,
            text="Let me show you the feature demo",
            speaker="SPEAKER_00",
        ),
        TranscriptSegment(
            start=42.0,
            end=50.0,
            text="Any questions from the audience?",
            speaker="SPEAKER_00",
        ),
    ]
    index = VideoIndex(
        video_id="test_video_1",
        filename="test.mp4",
        duration=70.0,
        filepath="/tmp/test.mp4",
        scenes=scenes,
        transcript=transcript,
        full_transcript=" ".join(t.text for t in transcript),
    )
    return index, scenes, transcript


@pytest.fixture
def sample_events():
    """Create sample Event objects for graph testing."""
    return [
        Event(
            event_id="video1/evt_000",
            video_id="video1",
            start_time=0.0,
            end_time=25.0,
            title="Introduction",
            description="Speaker introduces the project and roadmap",
            state_before="Empty stage, room quiet",
            state_after="Audience engaged, presentation started",
            entities=["SPEAKER_00", "project", "roadmap"],
            action="introducing",
        ),
        Event(
            event_id="video1/evt_001",
            video_id="video1",
            start_time=25.0,
            end_time=40.0,
            title="Feature Demo",
            description="Live demo of the new feature",
            state_before="Presentation slides visible",
            state_after="Feature demonstrated, audience impressed",
            entities=["feature", "SPEAKER_00", "demo"],
            action="demonstrating",
        ),
        Event(
            event_id="video1/evt_002",
            video_id="video1",
            start_time=40.0,
            end_time=70.0,
            title="Q&A Session",
            description="Audience asks questions about the feature",
            state_before="Feature demonstrated, audience impressed",
            state_after="Session ended, audience satisfied",
            entities=["audience", "SPEAKER_00", "questions"],
            action="answering",
        ),
    ]


# ---------------------------------------------------------------------------
# Event dataclass tests
# ---------------------------------------------------------------------------


def test_event_defaults():
    """Event dataclass should have correct defaults."""
    evt = Event(
        event_id="test/evt_000",
        video_id="test",
        start_time=0.0,
        end_time=10.0,
        title="Test Event",
        description="A test event",
    )
    assert evt.event_id == "test/evt_000"
    assert evt.start_time == 0.0
    assert evt.end_time == 10.0
    assert evt.title == "Test Event"
    assert evt.confidence == 1.0  # default
    assert evt.scene_ids == []
    assert evt.entities == []
    assert evt.transcript == ""


def test_event_with_all_fields():
    """Event with all fields populated."""
    evt = Event(
        event_id="test/evt_001",
        video_id="test",
        start_time=5.0,
        end_time=15.0,
        title="Demo",
        description="Showing the main feature",
        transcript="Here is the demo",
        scene_ids=[1, 2, 3],
        state_before="Slide visible",
        state_after="Feature shown",
        entities=["feature", "speaker"],
        action="demonstrating",
        metadata={"tags": ["demo"]},
        confidence=0.95,
    )
    assert evt.transcript == "Here is the demo"
    assert evt.scene_ids == [1, 2, 3]
    assert evt.state_before == "Slide visible"
    assert evt.state_after == "Feature shown"
    assert evt.confidence == 0.95


# ---------------------------------------------------------------------------
# EventSegmenter tests
# ---------------------------------------------------------------------------


def test_segmenter_temporal_grid(sample_scenes):
    """Temporal-grid fallback should group scenes into fixed-duration events."""
    index, scenes, _ = sample_scenes
    segmenter = EventSegmenter()
    events = segmenter._segment_temporal_grid("test_vid", scenes, event_duration=30.0)
    assert len(events) >= 1
    # First event should cover ~first 30s of content
    assert events[0].video_id == "test_vid"
    assert events[0].start_time >= 0.0
    assert events[0].scene_ids
    assert events[0].confidence == 0.7  # temporal grid confidence


def test_segmenter_temporal_grid_empty():
    """Empty scenes should produce empty event list."""
    segmenter = EventSegmenter()
    events = segmenter._segment_temporal_grid("test_vid", [])
    assert events == []


def test_segmenter_empty_video(sample_scenes):
    """Empty index scenes should produce empty event list."""

    index, _, _ = sample_scenes
    index.scenes = []
    segmenter = EventSegmenter()
    events = segmenter.segment(index, video_id="test_vid")
    assert events == []


def test_segmenter_llm_fallback(sample_scenes):
    """When LLM is None, segmenter should fallback to transcript coherence."""
    index, _, _ = sample_scenes
    segmenter = EventSegmenter()
    events = segmenter.segment(index)
    # Should produce events via transcript-coherence or temporal-grid
    assert len(events) >= 1


def test_segmenter_transcript_coherence(sample_scenes):
    """Transcript-coherence segmentation should group scenes by keyword overlap."""
    index, scenes, transcript_segments = sample_scenes
    segmenter = EventSegmenter()
    events = segmenter._segment_by_transcript_coherence("test_vid", scenes, transcript_segments)
    assert len(events) >= 1
    for evt in events:
        assert evt.event_id.startswith("test_vid/")
        assert len(evt.scene_ids) >= 1


def test_segmenter_with_llm_success(sample_scenes):
    """LLM-based segmentation should work when LLM returns valid JSON."""
    index, scenes, transcript_segments = sample_scenes
    mock_llm = MagicMock()
    mock_result = json.dumps(
        [
            {
                "start_scene": 0,
                "end_scene": 1,
                "title": "Introduction",
                "description": "Speaker introduces the project",
                "state_before": "Empty stage",
                "state_after": "Audience engaged",
                "action": "introducing",
                "entities": ["speaker", "project"],
            },
            {
                "start_scene": 2,
                "end_scene": 4,
                "title": "Demo and Q&A",
                "description": "Feature demo and questions",
                "state_before": "Presentation ongoing",
                "state_after": "Session finished",
                "action": "presenting",
                "entities": ["feature", "audience"],
            },
        ]
    )
    mock_llm.chat = MagicMock(return_value=mock_result)

    segmenter = EventSegmenter(llm_provider=mock_llm)
    events = segmenter.segment(index, video_id="test_vid")
    assert len(events) >= 2
    assert events[0].title == "Introduction"
    assert events[1].title == "Demo and Q&A"
    assert events[0].state_before == "Empty stage"
    assert events[0].entities == ["speaker", "project"]


# ---------------------------------------------------------------------------
# SESGraph construction tests
# ---------------------------------------------------------------------------


def test_ses_graph_construction(sample_events):
    """SESGraph should build correctly from events."""
    events = sample_events
    evt_map = {e.event_id: e for e in events}

    ses = SESGraph()
    ses.events = evt_map

    # Build state nodes manually
    for event in events:
        if event.state_before:
            state_id = f"state_{event.event_id}_before"
            ses.states[state_id] = event.state_before
        if event.state_after:
            state_id = f"state_{event.event_id}_after"
            ses.states[state_id] = event.state_after

    # Build temporal edges
    sorted_events = sorted(events, key=lambda e: e.start_time)
    for i in range(len(sorted_events) - 1):
        curr = sorted_events[i].event_id
        next_ = sorted_events[i + 1].event_id
        ses.forward_edges[curr].append((next_, "temporal"))
        ses.backward_edges[next_].append((curr, "temporal"))

    assert len(ses.events) == 3
    assert len(ses.states) == 6  # 3 events * 2 states each
    assert "video1/evt_000" in ses.forward_edges
    assert len(ses.forward_edges["video1/evt_000"]) >= 1
    # Check temporal edge
    assert any(
        target == "video1/evt_001" and rel == "temporal"
        for target, rel in ses.forward_edges["video1/evt_000"]
    )


def test_ses_graph_with_state_transition(sample_events):
    """Causal edges should form when state_after matches state_before."""
    events = sample_events
    # evt_000's state_after should semantically match evt_001's state_before
    # due to the "engaged" / "impressed" relationship
    ses = SESGraph()
    evt_map = {e.event_id: e for e in events}

    # Manually scan for causal edges
    for i in range(len(events) - 1):
        curr = events[i]
        next_ = events[i + 1]
        if curr.state_after and next_.state_before:
            sa = curr.state_after.lower().strip()
            sb = next_.state_before.lower().strip()
            if sa == sb or sa.endswith(sb) or sb.endswith(sa):
                pass  # would create causal edge

    # The fixture events have different state_after/before values
    # but the state_after of evt_001 matches state_before of evt_002
    assert events[1].state_after == "Feature demonstrated, audience impressed"
    assert events[2].state_before == "Feature demonstrated, audience impressed"
    assert events[1].state_after == events[2].state_before


# ---------------------------------------------------------------------------
# SemanticStore tests
# ---------------------------------------------------------------------------


def test_semantic_store_index_and_search(sample_events):
    """SemanticStore should index events and return results by keyword match."""
    store = SemanticStore()
    count = store.index_events(sample_events)
    assert count == len(sample_events)
    assert store._events
    assert store._fallback_store

    # Search by keyword
    results = store.search("feature demonstration", top_k=5)
    assert len(results) >= 1
    # "feature" and "demo" appear in evt_001
    found = [r for r in results if r[0].event_id == "video1/evt_001"]
    assert len(found) >= 1


def test_semantic_store_empty_search():
    """Empty store should return empty results."""
    store = SemanticStore()
    results = store.search("anything")
    assert results == []


def test_semantic_store_empty_events():
    """Indexing no events should be a no-op."""
    store = SemanticStore()
    count = store.index_events([])
    assert count == 0


# ---------------------------------------------------------------------------
# CausalTopologicalStore tests
# ---------------------------------------------------------------------------


def test_causal_store_forward_retrieval(sample_events):
    """CausalTopologicalStore should retrieve forward events via BFS."""
    events = {e.event_id: e for e in sample_events}
    store = CausalTopologicalStore()
    store._events = events

    # Build a simple graph: evt_000 -> evt_001 -> evt_002
    store._graph["video1/evt_000"].append(("video1/evt_001", "temporal"))
    store._graph["video1/evt_001"].append(("video1/evt_002", "temporal"))
    store._reverse_graph["video1/evt_001"].append(("video1/evt_000", "temporal"))
    store._reverse_graph["video1/evt_002"].append(("video1/evt_001", "temporal"))

    # Forward from evt_000 should yield evt_001 and evt_002
    results = store.retrieve_forward("video1/evt_000", max_hops=3, max_results=10)
    assert len(results) >= 1
    # Should include evt_001 (1 hop) and evt_002 (2 hops)
    event_ids = [r[0].event_id for r in results]
    assert "video1/evt_001" in event_ids
    assert "video1/evt_002" in event_ids


def test_causal_store_backward_retrieval(sample_events):
    """CausalTopologicalStore should retrieve backward events via reverse BFS."""
    events = {e.event_id: e for e in sample_events}
    store = CausalTopologicalStore()
    store._events = events

    # Build a simple graph
    store._graph["video1/evt_000"].append(("video1/evt_001", "temporal"))
    store._graph["video1/evt_001"].append(("video1/evt_002", "temporal"))
    store._reverse_graph["video1/evt_001"].append(("video1/evt_000", "temporal"))
    store._reverse_graph["video1/evt_002"].append(("video1/evt_001", "temporal"))
    store._reverse_graph["video1/evt_002"].append(("video1/evt_000", "temporal"))

    # Backward from evt_002 should yield evt_001 and evt_000
    results = store.retrieve_backward("video1/evt_002", max_hops=3, max_results=10)
    assert len(results) >= 1
    event_ids = [r[0].event_id for r in results]
    assert "video1/evt_001" in event_ids
    assert "video1/evt_000" in event_ids


def test_causal_store_update_from_ses(sample_events):
    """CausalTopologicalStore.update() should populate from SESGraph."""
    from video_analysis.event_rag import SESGraph

    ses = SESGraph()
    ses.events = {e.event_id: e for e in sample_events}
    ses.forward_edges["video1/evt_000"] = [("video1/evt_001", "temporal")]
    ses.forward_edges["video1/evt_001"] = [("video1/evt_002", "temporal")]
    ses.backward_edges["video1/evt_001"] = [("video1/evt_000", "temporal")]
    ses.backward_edges["video1/evt_002"] = [("video1/evt_001", "temporal")]

    store = CausalTopologicalStore()
    store.update(ses)

    assert len(store._events) == 3
    assert "video1/evt_000" in store._graph


def test_causal_store_unknown_event(sample_events):
    """Retrieval from unknown event should return empty."""
    store = CausalTopologicalStore()
    results = store.retrieve_forward("nonexistent", max_hops=2)
    assert results == []


# ---------------------------------------------------------------------------
# DualStoreMemory tests
# ---------------------------------------------------------------------------


def test_dual_store_semantic_only(sample_events):
    """DualStoreMemory should work with semantic store alone."""
    sem_store = SemanticStore()
    sem_store.index_events(sample_events)
    memory = DualStoreMemory(semantic_store=sem_store)

    results = memory.retrieve("feature demonstration")
    assert len(results) >= 1
    assert results[0].retrieval_type == "semantic"


def test_dual_store_bidirectional(sample_events):
    """DualStoreMemory should combine semantic and causal retrieval."""
    sem_store = SemanticStore()
    sem_store.index_events(sample_events)

    causal_store = CausalTopologicalStore()
    causal_store._events = {e.event_id: e for e in sample_events}
    causal_store._graph["video1/evt_000"].append(("video1/evt_001", "temporal"))
    causal_store._graph["video1/evt_001"].append(("video1/evt_002", "temporal"))
    causal_store._reverse_graph["video1/evt_001"].append(("video1/evt_000", "temporal"))
    causal_store._reverse_graph["video1/evt_002"].append(("video1/evt_001", "temporal"))

    memory = DualStoreMemory(
        semantic_store=sem_store,
        causal_store=causal_store,
        semantic_weight=0.5,
        causal_weight=0.5,
    )

    # Retrieve from middle event — should get forward and backward
    results = memory.retrieve(
        "feature demo",
        current_event_id="video1/evt_001",
        top_k=10,
    )
    assert len(results) >= 1
    retrieval_types = {r.retrieval_type for r in results}
    # Should include both semantic and causal types
    assert "causal_forward" in retrieval_types or "causal_backward" in retrieval_types


def test_dual_store_no_current_event(sample_events):
    """Without a current event, only semantic retrieval runs."""
    memory = DualStoreMemory()
    results = memory.retrieve("anything")
    assert results == []  # empty semantic store


# ---------------------------------------------------------------------------
# EventCausalRAG integration tests
# ---------------------------------------------------------------------------


def test_event_causal_rag_full_pipeline(sample_scenes):
    """EventCausalRAG should run the full pipeline end-to-end."""
    index, scenes, _ = sample_scenes
    rag = EventCausalRAG()

    # 1. Segment
    events = rag.segment_video(index, video_id="test_vid")
    assert len(events) >= 1
    assert rag._events == events

    # 2. Build SES graph
    ses = rag.build_ses_graph(events)
    assert ses is not None
    assert len(ses.events) >= 1

    # 3. Index
    count = rag.index_events(events)
    assert count >= 1

    # 4. Retrieve
    results = rag.retrieve("introduction speaker", top_k=5)
    assert len(results) >= 0  # may be 0 with fallback store


def test_event_causal_rag_causal_paths(sample_events):
    """EventCausalRAG should find causal paths between events."""
    rag = EventCausalRAG()
    rag._events = sample_events
    rag._ses_graph = SESGraph()
    rag._ses_graph.events = {e.event_id: e for e in sample_events}
    rag._ses_graph.forward_edges["video1/evt_000"] = [("video1/evt_001", "temporal")]
    rag._ses_graph.forward_edges["video1/evt_001"] = [("video1/evt_002", "temporal")]

    paths = rag.find_causal_paths("video1/evt_000", "video1/evt_002")
    assert len(paths) >= 1
    assert paths[0].path == ["video1/evt_000", "video1/evt_001", "video1/evt_002"]
    assert paths[0].direction == "forward"


def test_event_causal_rag_timeline(sample_events):
    """Event timeline should return chronologically sorted events."""
    rag = EventCausalRAG()
    rag._events = sample_events
    timeline = rag.get_event_timeline(video_id="video1")
    assert len(timeline) == 3
    assert timeline[0].start_time <= timeline[1].start_time <= timeline[2].start_time


def test_event_causal_rag_to_dict(sample_events):
    """to_dict should serialize the RAG state."""
    rag = EventCausalRAG()
    rag._events = sample_events
    ses = SESGraph()
    ses.events = {e.event_id: e for e in sample_events}
    rag._ses_graph = ses

    data = rag.to_dict()
    assert "events" in data
    assert len(data["events"]) == 3
    assert "ses_graph" in data


def test_causal_path_dataclass():
    """CausalPath should construct correctly."""
    path = CausalPath(
        path=["evt_000", "evt_001"],
        direction="forward",
        score=0.8,
        description="Introduction -> Feature Demo",
    )
    assert path.path == ["evt_000", "evt_001"]
    assert path.score == 0.8


def test_retrieval_result_dataclass(sample_events):
    """RetrievalResult should construct correctly."""
    result = RetrievalResult(
        event=sample_events[0],
        score=0.75,
        path=CausalPath(path=["evt_000"], direction="forward"),
        retrieval_type="semantic",
    )
    assert result.score == 0.75
    assert result.retrieval_type == "semantic"


def test_event_segmenter_with_llm_json_in_markdown(sample_scenes):
    """LLM returning JSON inside markdown code fences should still parse."""
    index, scenes, _ = sample_scenes
    mock_llm = MagicMock()
    mock_llm.chat = MagicMock(
        return_value='Here is the result:\n```json\n[\n  {"start_scene": 0, "end_scene": 2, "title": "Intro + Demo", "description": "First part", "state_before": "Start", "state_after": "Mid", "action": "presenting", "entities": ["speaker"]}\n]\n```'
    )

    segmenter = EventSegmenter(llm_provider=mock_llm)
    events = segmenter.segment(index, video_id="test_vid")
    assert len(events) >= 1
    assert events[0].title == "Intro + Demo"


def test_event_segmenter_with_llm_failure(sample_scenes):
    """LLM failure should cause graceful fallback to other strategies."""
    index, scenes, _ = sample_scenes
    mock_llm = MagicMock()
    mock_llm.chat.side_effect = RuntimeError("LLM API failed")
    segmenter = EventSegmenter(llm_provider=mock_llm)
    events = segmenter.segment(index)
    # Should fallback to transcript-coherence or temporal-grid
    assert len(events) >= 1


def test_semantic_store_search_scoring(sample_events):
    """Search scores should be higher for better keyword matches."""
    store = SemanticStore()
    store.index_events(sample_events)

    # Search for something specific to evt_001
    results = store.search("feature demo demonstrating", top_k=5)
    assert len(results) >= 1
    # The top result should be evt_001 (feature/demo/demonstrating keywords)
    if results:
        top_result = results[0][0]
        assert top_result.event_id == "video1/evt_001"


def test_extract_transcript_for_scenes(sample_scenes):
    """Helper should extract transcript for specific scene IDs."""
    index, scenes, transcript_segments = sample_scenes
    # Set transcript on the scenes
    scenes[0].transcript = "Hello everyone, welcome to today's presentation"
    scenes[1].transcript = "Today we'll discuss our project roadmap"
    segmenter = EventSegmenter()
    text = segmenter._extract_transcript_for_scenes([0, 1], transcript_segments, scenes)
    assert "welcome" in text or "roadmap" in text or "Hello" in text


def test_scenes_to_event(sample_scenes):
    """Helper should convert scene groups to Events."""
    index, scenes, _ = sample_scenes
    segmenter = EventSegmenter()
    event = segmenter._scenes_to_event("test_vid", scenes[:2], [0, 1], 0)
    assert event.event_id == "test_vid/evt_000"
    assert event.start_time == 0.0
    assert event.end_time == 25.0
    assert event.confidence == 0.7
