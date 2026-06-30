"""
Tests for the Agentic Video Understanding Agent (v0.36.0).

Covers:
- VideoUnderstandingAgent query routing and tool dispatch
- AgentTools individual tools (unit-testable components)
- AgentQueryResult data class
- Timestamp extraction from natural language
- Integration with chat.py
"""

from video_analysis.agent import (
    AgentQueryResult,
    AgentToolResult,
    AgentTools,
    VideoUnderstandingAgent,
)

# ---------------------------------------------------------------------------
# Data class tests
# ---------------------------------------------------------------------------


class TestAgentToolResult:
    def test_defaults(self):
        r = AgentToolResult(tool_name="test", success=True, data="ok")
        assert r.tool_name == "test"
        assert r.success is True
        assert r.data == "ok"
        assert r.metadata == {}

    def test_with_metadata(self):
        r = AgentToolResult(
            tool_name="search",
            success=True,
            data="found",
            metadata={"num_results": 5},
        )
        assert r.metadata["num_results"] == 5


class TestAgentQueryResult:
    def test_defaults(self):
        r = AgentQueryResult(query="test?", answer="answer", confidence=0.8)
        assert r.query == "test?"
        assert r.answer == "answer"
        assert r.confidence == 0.8
        assert r.evidence == []
        assert r.reasoning_steps == []
        assert r.tools_used == 0
        assert r.duration_seconds == 0.0

    def test_with_evidence(self):
        ev = AgentToolResult(tool_name="rag", success=True, data="ctx")
        r = AgentQueryResult(
            query="q",
            answer="a",
            confidence=0.9,
            evidence=[ev],
            reasoning_steps=["step1"],
            tools_used=1,
            duration_seconds=1.5,
        )
        assert len(r.evidence) == 1
        assert r.tools_used == 1
        assert r.duration_seconds == 1.5


# ---------------------------------------------------------------------------
# Timestamp extraction from natural language queries
# ---------------------------------------------------------------------------


class TestTimestampExtraction:
    def test_extract_mmss(self):
        ts = VideoUnderstandingAgent._extract_timestamps("at 2:30 mark")
        assert ts == [150.0]

    def test_extract_hmmss(self):
        ts = VideoUnderstandingAgent._extract_timestamps("at 1:02:30")
        assert ts == [3750.0]

    def test_extract_seconds(self):
        ts = VideoUnderstandingAgent._extract_timestamps("around 45 seconds in")
        assert ts == [45.0]

    def test_extract_multiple(self):
        ts = VideoUnderstandingAgent._extract_timestamps("at 1:00 and 2:30 and 45 seconds")
        assert 60.0 in ts
        assert 150.0 in ts
        assert 45.0 in ts
        assert len(ts) == 3

    def test_extract_minutes(self):
        ts = VideoUnderstandingAgent._extract_timestamps("at 2 minutes")
        assert ts == [120.0]

    def test_empty_text(self):
        ts = VideoUnderstandingAgent._extract_timestamps("what objects are visible")
        assert ts == []


# ---------------------------------------------------------------------------
# AgentTools — unit tests with minimal dependencies
# ---------------------------------------------------------------------------


class TestAgentTools:
    def test_init(self, tmp_path):
        """AgentTools initialises without errors."""
        from video_analysis.config import Config

        config = Config()
        config.video_dir = tmp_path
        tools = AgentTools(config=config)
        assert tools.config is not None
        assert tools.rag is None

    def test_tool_result_no_video(self, tmp_path):
        """Tools that need a video file return an error gracefully."""
        from video_analysis.config import Config

        config = Config()
        config.video_dir = tmp_path
        tools = AgentTools(config=config, video_path="/nonexistent/video.mp4")

        result = tools.detect_objects(30.0)
        assert result.success is False
        assert "not available" in result.data.lower()
        assert result.tool_name == "detect_objects"

        result = tools.extract_text(30.0)
        assert result.success is False
        assert "not available" in result.data.lower() or "available" in result.data.lower()
        assert result.tool_name == "extract_text"

    def test_analyze_frames_no_video(self, tmp_path):
        """analyze_frames gracefully handles missing video."""
        from video_analysis.config import Config

        config = Config()
        config.video_dir = tmp_path
        tools = AgentTools(config=config)

        result = tools.analyze_frames([30.0, 60.0])
        assert result.success is False
        assert result.tool_name == "analyze_frames"

    def test_search_rag_no_rag(self, tmp_path):
        """search_rag returns error when no RAG is configured."""
        from video_analysis.config import Config

        config = Config()
        tools = AgentTools(config=config)
        result = tools.search_rag("test query")
        assert result.success is False
        assert "rag index not available" in result.data.lower()
        assert result.tool_name == "search_rag"

    def test_search_transcript_no_rag(self, tmp_path):
        """search_transcript returns error when no RAG is configured."""
        from video_analysis.config import Config

        config = Config()
        tools = AgentTools(config=config)
        result = tools.search_transcript("hello")
        assert result.success is False
        assert result.tool_name == "search_transcript"

    def test_temporal_grounding_no_rag(self, tmp_path):
        """temporal_grounding returns error when no RAG is configured."""
        from video_analysis.config import Config

        config = Config()
        tools = AgentTools(config=config)
        result = tools.temporal_grounding("a person walking")
        assert result.success is False
        assert result.tool_name == "temporal_grounding"

    def test_summarize_video_no_video(self, tmp_path):
        """summarize_video gracefully handles missing video."""
        from video_analysis.config import Config

        config = Config()
        tools = AgentTools(config=config)
        result = tools.summarize_video()
        assert result.success is False
        assert result.tool_name == "summarize_video"


# ---------------------------------------------------------------------------
# VideoUnderstandingAgent — unit tests
# ---------------------------------------------------------------------------


class TestVideoUnderstandingAgent:
    def test_init(self, tmp_path):
        """Agent initialises without errors."""
        from video_analysis.config import Config

        config = Config()
        config.video_dir = tmp_path
        agent = VideoUnderstandingAgent(config=config, video_path=str(tmp_path))
        assert agent.config is not None
        assert agent.video_path == str(tmp_path)

    def test_query_without_video(self, tmp_path):
        """Query gracefully handles missing video path."""
        from video_analysis.config import Config

        config = Config()
        config.video_dir = tmp_path
        agent = VideoUnderstandingAgent(config=config)

        result = agent.query("what objects are visible")
        assert isinstance(result, AgentQueryResult)
        assert result.query == "what objects are visible"
        assert len(result.answer) > 0
        assert result.tools_used >= 0

    def test_answer_not_too_short(self, tmp_path):
        """Even with no data, the agent produces a meaningful response."""
        from video_analysis.config import Config

        config = Config()
        agent = VideoUnderstandingAgent(config=config)
        result = agent.query("summarize the video")
        assert len(result.answer) > 20

    def test_reasoning_steps_present(self, tmp_path):
        """Agent produces reasoning steps."""
        from video_analysis.config import Config

        config = Config()
        agent = VideoUnderstandingAgent(config=config)
        result = agent.query("what is happening at 1:30")
        assert len(result.reasoning_steps) > 0

    def test_evidence_contains_tool_results(self, tmp_path):
        """Agent evidence list contains tool results."""
        from video_analysis.config import Config

        config = Config()
        agent = VideoUnderstandingAgent(config=config)
        result = agent.query("describe the scene")
        tool_names = [e.tool_name for e in result.evidence]
        # At minimum, should have context_bootstrap or analyze_frames
        assert len(result.evidence) > 0

    def test_question_classification_summarization(self, tmp_path):
        """'summarize' questions dispatch to summarize_video tool."""
        from video_analysis.config import Config

        config = Config()
        agent = VideoUnderstandingAgent(config=config, video_path="/nonexistent.mp4")
        result = agent.query("summarize what happens in this video")
        # The summarize_video tool should be invoked (even if it fails due to no video)
        tool_names = [e.tool_name for e in result.evidence]
        assert "summarize_video" in tool_names or any("summarize" in t for t in tool_names)

    def test_question_classification_temporal(self, tmp_path):
        """'find/when' questions dispatch to temporal_grounding."""
        from video_analysis.config import Config

        config = Config()
        agent = VideoUnderstandingAgent(config=config)
        result = agent.query("when did the person enter the room")
        tool_names = [e.tool_name for e in result.evidence]
        assert "temporal_grounding" in tool_names

    def test_question_classification_objects(self, tmp_path):
        """'objects' questions dispatch to detect_objects."""
        from video_analysis.config import Config

        config = Config()
        agent = VideoUnderstandingAgent(config=config)
        result = agent.query("what objects are visible")
        tool_names = [e.tool_name for e in result.evidence]
        assert "detect_objects" in tool_names

    def test_question_classification_transcript(self, tmp_path):
        """'said/speak/transcript' questions dispatch to search_transcript."""
        from video_analysis.config import Config

        config = Config()
        agent = VideoUnderstandingAgent(config=config)
        result = agent.query("what did the narrator say")
        tool_names = [e.tool_name for e in result.evidence]
        assert "search_transcript" in tool_names

    def test_question_classification_ocr(self, tmp_path):
        """'text/read/OCR' questions dispatch to extract_text."""
        from video_analysis.config import Config

        config = Config()
        agent = VideoUnderstandingAgent(config=config)
        result = agent.query("read the text on screen")
        tool_names = [e.tool_name for e in result.evidence]
        assert "extract_text" in tool_names

    def test_generate_report_basic(self, tmp_path):
        """generate_report returns markdown without crashing."""
        from video_analysis.config import Config

        config = Config()
        agent = VideoUnderstandingAgent(config=config, video_path=str(tmp_path))
        report = agent.generate_report()
        assert isinstance(report, str)
        assert len(report) > 0


# ---------------------------------------------------------------------------
# Config integration tests
# ---------------------------------------------------------------------------


class TestAgentConfig:
    def test_agent_enabled_default(self):
        from video_analysis.config import Config

        c = Config()
        assert c.agent_enabled is False
        assert c.agent_max_tools == 5

    def test_agent_enabled_env_var(self, monkeypatch):
        monkeypatch.setenv("AGENT_ENABLED", "true")
        from video_analysis.config import Config

        c = Config()
        assert c.agent_enabled is True


# ---------------------------------------------------------------------------
# Chat integration test
# ---------------------------------------------------------------------------


class TestChatAgentIntegration:
    def test_get_agent_video_path_none(self):
        """_get_agent_video_path returns None when video_id is None."""
        from video_analysis.chat import VideoChat
        from video_analysis.config import Config

        config = Config()

        # We need a mock RAG or None — VideoChat init requires rag
        # But we can test the method directly if VideoChat is constructed
        # with a minimal rag. For simplicity, test that the method
        # handles None gracefully.
        class MockRAG:
            collection = None

        chat = VideoChat(rag=MockRAG(), config=config)
        result = chat._get_agent_video_path(None)
        assert result is None

    def test_get_agent_video_path_not_found(self, tmp_path):
        """_get_agent_video_path returns None when the video file doesn't exist."""
        from video_analysis.chat import VideoChat
        from video_analysis.config import Config

        config = Config()
        config.video_dir = tmp_path

        chat = VideoChat(rag=None, config=config)  # type: ignore[arg-type]
        result = chat._get_agent_video_path("nonexistent_video")
        assert result is None
