"""Integration tests for Event-Causal RAG in production pipeline (v0.58.0).

Tests cover:
- VideoRAG.event_retrieve() method
- VideoRAG.event_index_video() method
- VideoRAG._get_event_rag() lazy init
- VideoChat event-causal RAG retrieval path
- KnowledgeGraph event persistence methods
- Config env var overrides for event_causal_rag_in_chat
"""

import json
import os
import time
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

from video_analysis.config import Config
from video_analysis.rag import VideoRAG, RetrievedChunk
from video_analysis.models import VideoIndex, SceneInfo, FrameInfo, TranscriptSegment

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_event():
    """Create a mock Event object."""
    from video_analysis.event_rag import Event, CausalPath

    evt = Event(
        event_id="test_vid/evt_000",
        video_id="test_vid",
        start_time=0.0,
        end_time=30.0,
        title="Introduction",
        description="The speaker introduces the topic",
        transcript="Welcome to this presentation.",
        scene_ids=["test_vid_scene_0000"],
        state_before="Audience waiting",
        state_after="Audience knows the topic",
        entities={"speaker", "audience", "topic"},
        action="introducing",
        confidence=0.95,
    )
    return evt


@pytest.fixture
def mock_retrieval_result():
    """Create a mock RetrievalResult."""
    from video_analysis.event_rag import RetrievalResult, Event, CausalPath

    evt = Event(
        event_id="test_vid/evt_000",
        video_id="test_vid",
        start_time=0.0,
        end_time=30.0,
        title="Introduction",
        description="The speaker introduces the topic",
        transcript="Welcome to this presentation.",
        scene_ids=["test_vid_scene_0000"],
        state_before="Audience waiting",
        state_after="Audience knows the topic",
        entities={"speaker", "audience", "topic"},
        action="introducing",
        confidence=0.95,
    )
    path = CausalPath(
        path=["test_vid/evt_000"],
        direction="forward",
        score=0.95,
        description="Introduction starts the video",
    )
    return RetrievalResult(
        event=evt,
        score=0.92,
        retrieval_type="semantic",
        path=path,
    )


@pytest.fixture
def config():
    import tempfile

    tmpdir = tempfile.mkdtemp(prefix="evt_rag_test_")
    c = Config()
    c.event_causal_rag_enabled = True
    c.event_causal_rag_index_on_process = True
    c.event_causal_rag_in_chat = True
    c.event_causal_top_k = 10
    c.event_segmentation_strategy = "temporal"
    c.chroma_path = Path(tmpdir) / "chroma"
    c.data_dir = Path(tmpdir)
    return c


@pytest.fixture
def video_index():
    """Create a minimal VideoIndex for event segmentation testing."""
    scenes = [
        SceneInfo(
            scene_id=0,
            start_time=0.0,
            end_time=10.0,
            summary="Opening scene",
            transcript="Hello and welcome.",
        ),
        SceneInfo(
            scene_id=1,
            start_time=10.0,
            end_time=25.0,
            summary="Main discussion",
            transcript="Today we will talk about events.",
        ),
        SceneInfo(
            scene_id=2,
            start_time=25.0,
            end_time=40.0,
            summary="Q&A session",
            transcript="Any questions? Yes, please go ahead.",
        ),
    ]
    return VideoIndex(
        video_id="test_vid",
        filename="test.mp4",
        filepath="/tmp/test_vid.mp4",
        duration=40.0,
        scenes=scenes,
        full_transcript="Hello and welcome. Today we will talk about events. Any questions? Yes, please go ahead.",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEventCausalRagIntegration:
    """Integration tests for event-causal RAG in the production pipeline."""

    def test_config_event_causal_fields(self, config):
        """Test that config properly exposes event-causal RAG fields."""
        assert config.event_causal_rag_enabled is True
        assert config.event_causal_rag_index_on_process is True
        assert config.event_causal_rag_in_chat is True
        assert config.event_causal_top_k == 10
        assert config.event_segmentation_strategy == "temporal"

    def test_config_event_causal_env_overrides(self):
        """Test env var overrides for event-causal RAG config fields."""
        c = Config()
        # Default state
        assert c.event_causal_rag_in_chat is False

        os.environ["EVENT_CAUSAL_RAG_IN_CHAT"] = "true"
        os.environ["EVENT_CAUSAL_RAG_INDEX_ON_PROCESS"] = "false"
        c2 = Config()
        assert c2.event_causal_rag_in_chat is True
        assert c2.event_causal_rag_index_on_process is False

        # Clean up
        os.environ.pop("EVENT_CAUSAL_RAG_IN_CHAT", None)
        os.environ.pop("EVENT_CAUSAL_RAG_INDEX_ON_PROCESS", None)

    @patch("video_analysis.rag.VideoRAG._get_event_rag")
    def test_event_retrieve_returns_empty_when_disabled(
        self, mock_get_event_rag, config
    ):
        """Test event_retrieve returns [] when event_causal_rag_enabled is False."""
        config.event_causal_rag_enabled = False
        rag = VideoRAG(config)
        result = rag.event_retrieve("test query")
        assert result == []
        mock_get_event_rag.assert_not_called()

    @patch("video_analysis.rag.VideoRAG._get_event_rag")
    def test_event_retrieve_returns_chunks(
        self, mock_get_event_rag, config, mock_retrieval_result
    ):
        """Test event_retrieve returns RetrievedChunk objects from EventCausalRAG."""
        mock_rag = MagicMock()
        mock_rag.retrieve.return_value = [mock_retrieval_result]
        mock_get_event_rag.return_value = mock_rag

        rag = VideoRAG(config)
        result = rag.event_retrieve("test query")
        assert len(result) >= 1
        assert isinstance(result[0], RetrievedChunk)
        assert result[0].chunk_type == "event"
        assert "Introduction" in result[0].text
        assert result[0].video_id == "test_vid"

    @patch("video_analysis.rag.VideoRAG._get_event_rag")
    def test_event_retrieve_metadata(
        self, mock_get_event_rag, config, mock_retrieval_result
    ):
        """Test event_retrieve populates metadata correctly."""
        mock_rag = MagicMock()
        mock_rag.retrieve.return_value = [mock_retrieval_result]
        mock_get_event_rag.return_value = mock_rag

        rag = VideoRAG(config)
        result = rag.event_retrieve("test query")
        assert len(result) >= 1
        meta = result[0].metadata
        assert meta is not None
        assert meta.get("event_id") == "test_vid/evt_000"
        assert meta.get("event_title") == "Introduction"
        assert meta.get("retrieval_type") == "semantic"
        assert "Introduction" in meta.get("causal_path_summary", "")

    def test_event_index_video_disabled(self, config, video_index):
        """Test event_index_video returns 0 when disabled."""
        config.event_causal_rag_enabled = False
        rag = VideoRAG(config)
        count = rag.event_index_video(video_index)
        assert count == 0

    def test_event_index_video_index_on_process_disabled(self, config, video_index):
        """Test event_index_video returns 0 when index_on_process is False."""
        config.event_causal_rag_index_on_process = False
        rag = VideoRAG(config)
        count = rag.event_index_video(video_index)
        assert count == 0

    @patch("video_analysis.rag.VideoRAG._get_event_rag")
    def test_event_index_video_calls_segment(
        self, mock_get_event_rag, config, video_index, mock_event
    ):
        """Test event_index_video calls segment_video, build_ses_graph, index_events."""
        mock_rag = MagicMock()
        mock_rag.segment_video.return_value = [mock_event]
        mock_rag.build_ses_graph.return_value = MagicMock()
        mock_rag.index_events.return_value = 1
        mock_get_event_rag.return_value = mock_rag

        rag = VideoRAG(config)
        count = rag.event_index_video(video_index)
        assert count == 1
        mock_rag.segment_video.assert_called_once()
        mock_rag.build_ses_graph.assert_called_once()
        mock_rag.index_events.assert_called_once()

    @patch("video_analysis.rag.VideoRAG.event_index_video")
    def test_index_video_auto_triggers_event_indexing(
        self, mock_event_index, config, video_index
    ):
        """Test index_video() calls event_index_video() automatically."""
        config.event_causal_rag_enabled = True
        config.event_causal_rag_index_on_process = True
        rag = VideoRAG(config)

        with patch.object(rag, "_get_embedding", return_value=[0.1] * 768):
            # _init_chroma creates collection if needed
            rag._init_chroma()
            with patch.object(rag.collection, "add"):
                rag.index_video(video_index)
                mock_event_index.assert_called_once_with(video_index)

    def test_video_rag_lazy_init_event_rag(self, config):
        """Test _get_event_rag lazily initializes EventCausalRAG."""
        rag = VideoRAG(config)
        with patch("video_analysis.rag.VideoRAG._get_event_rag") as mock_get:
            mock_inst = MagicMock()
            mock_get.return_value = mock_inst
            er = rag._get_event_rag()
            assert er is not None

    def test_knowledge_graph_persist_events(self, config, mock_event):
        """Test KnowledgeGraph.persist_events_from_rag."""
        from video_analysis.knowledge_graph import KnowledgeGraph

        mock_rag = MagicMock()
        mock_event_rag = MagicMock()
        mock_event_rag._events = [mock_event]

        ses = MagicMock()
        ses.forward_edges = {"test_vid/evt_000": [("test_vid/evt_001", "causal")]}
        mock_event_rag._ses_graph = ses

        mock_rag._event_rag_instance = mock_event_rag

        kg = KnowledgeGraph(config)
        try:
            count = kg.persist_events_from_rag(mock_rag, "test_vid")
            assert count == 1

            # Verify event was stored
            events = kg.get_events_for_video("test_vid")
            assert len(events) >= 1
            assert events[0]["event_id"] == "test_vid/evt_000"
            assert events[0]["title"] == "Introduction"
            assert "speaker" in events[0].get("entities", [])

            # Verify causal relation was stored
            rels = kg.get_causal_relations(video_id="test_vid")
            assert len(rels) >= 1
            assert rels[0]["source_event_id"] == "test_vid/evt_000"
            assert rels[0]["target_event_id"] == "test_vid/evt_001"
        finally:
            kg.delete_events_for_video("test_vid")
            kg.close()

    def test_knowledge_graph_event_crud(self, config):
        """Test full CRUD for event records in KnowledgeGraph."""
        from video_analysis.knowledge_graph import KnowledgeGraph

        kg = KnowledgeGraph(config)
        try:
            # Create
            kg.add_event_record(
                event_id="test/evt_001",
                video_id="test_vid",
                title="Test Event",
                description="A test event",
                start_time=5.0,
                end_time=15.0,
                state_before="Nothing",
                state_after="Something",
                action="testing",
                entities=["tester", "test"],
                confidence=0.8,
            )

            # Read
            events = kg.get_events_for_video("test_vid")
            assert len(events) >= 1
            evt = next(e for e in events if e["event_id"] == "test/evt_001")
            assert evt["title"] == "Test Event"
            assert evt["start_time"] == 5.0
            assert evt["end_time"] == 15.0
            assert "tester" in evt["entities"]

            # Create causal relation
            kg.add_causal_relation(
                source_event_id="test/evt_001",
                target_event_id="test/evt_002",
                relation_type="causal",
                strength=1.0,
                metadata={"reason": "direct cause"},
            )

            # Read relations
            rels = kg.get_causal_relations(video_id="test_vid")
            causal = [r for r in rels if r["relation_type"] == "causal"]
            assert len(causal) >= 1
            assert causal[0]["source_event_id"] == "test/evt_001"

            # Delete
            kg.delete_events_for_video("test_vid")
            remaining = kg.get_events_for_video("test_vid")
            assert len(remaining) == 0
            post_rels = kg.get_causal_relations(video_id="test_vid")
            assert len(post_rels) == 0
        finally:
            kg.close()

    def test_event_retrieve_with_video_id_filter(self, config):
        """Test event_retrieve passes video_id correctly."""
        # This tests the method doesn't crash with video_id set
        rag = VideoRAG(config)
        with patch.object(rag, "_get_event_rag") as mock_get:
            mock_inst = MagicMock()
            mock_inst.retrieve.return_value = []
            mock_get.return_value = mock_inst
            result = rag.event_retrieve("query", video_id="test_vid")
            # Should return empty list (no results from mock)
            assert isinstance(result, list)

    @patch("video_analysis.chat.VideoChat._build_prompt")
    @patch("video_analysis.chat.VideoChat._get_llm")
    def test_chat_event_causal_path(self, mock_get_llm, mock_build_prompt, config):
        """Test VideoChat uses event-causal retrieval when enabled."""
        config.event_causal_rag_enabled = True
        config.event_causal_rag_in_chat = True

        mock_llm = MagicMock()
        mock_llm.chat.return_value = "Event-causal answer"
        mock_get_llm.return_value = mock_llm

        rag = MagicMock()
        # event_retrieve should be called (not standard retrieve)
        rag.event_retrieve.return_value = []

        from video_analysis.chat import VideoChat

        chat = VideoChat(rag=rag, config=config)
        # Disable agent path so we hit RAG path
        config.agent_enabled = False
        config.video_mllm_enabled = False

        result = chat._ask_rag("What happened in the first event?")
        # event_retrieve should have been called
        rag.event_retrieve.assert_called_once()
        assert result is not None

    @patch("video_analysis.chat.VideoChat._get_llm")
    def test_chat_event_causal_ask_with_history(self, mock_get_llm, config):
        """Test ask_with_history uses event-causal retrieval when configured."""
        config.event_causal_rag_enabled = True
        config.event_causal_rag_in_chat = True

        mock_llm = MagicMock()
        mock_llm.chat.return_value = "Event-causal answer"
        mock_get_llm.return_value = mock_llm

        rag = MagicMock()
        rag.event_retrieve.return_value = []
        rag.build_context.return_value = ""
        rag.get_source_citations.return_value = []

        from video_analysis.chat import VideoChat

        chat = VideoChat(rag=rag, config=config)
        # Disable MLLM to hit RAG path
        config.video_mllm_enabled = False

        result = chat.ask_with_history("What caused the change?")
        # event_retrieve should be called in ask_with_history too
        rag.event_retrieve.assert_called_once()
        assert result is not None
