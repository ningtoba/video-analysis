"""Tests for video_analysis.stream.chat — StreamChat event timeline chat."""

from unittest.mock import MagicMock, patch

import pytest

from video_analysis.llm_provider import LLMProviderConfig
from video_analysis.stream.chat import StreamChat
from video_analysis.stream.store import TimelineEvent


def _make_event(
    event_id=1,
    stream_id="cam1",
    timestamp=1000.0,
    description="Motion detected",
    motion_score=0.5,
):
    """Helper to create a TimelineEvent with sensible defaults."""
    return TimelineEvent(
        id=event_id,
        stream_id=stream_id,
        timestamp=timestamp,
        description=description,
        frame_path=None,
        motion_score=motion_score,
        triggered_by="motion",
        metadata={},
        created_at=0.0,
    )


@pytest.fixture
def mock_llm():
    """Patch get_llm_provider to return a mock LLM with a canned response."""
    with patch("video_analysis.stream.chat.get_llm_provider") as mock_get:
        llm = MagicMock()
        llm.chat.return_value = "People were walking near the entrance."
        mock_get.return_value = llm
        yield llm


@pytest.fixture
def mock_store():
    """Return a mock EventStore pre-populated with sample events."""
    store = MagicMock()
    store.get_recent.return_value = [
        _make_event(1, "cam1", 1000.0, "Person detected", 0.42),
        _make_event(2, "cam1", 1005.0, "Vehicle detected", 0.78),
        _make_event(3, "cam1", 1010.0, "Person left frame", 0.15),
    ]
    store.get_range.return_value = [
        _make_event(2, "cam1", 1005.0, "Vehicle detected", 0.78),
    ]
    return store


@pytest.fixture
def chat(mock_store, mock_llm):
    """Return a StreamChat configured with a mock LLM and mock store."""
    config = LLMProviderConfig(provider="openai", api_key="test-key")
    return StreamChat(event_store=mock_store, stream_id="cam1", llm_provider_config=config)


class TestStreamChat:
    """StreamChat — event timeline chat interface."""

    def test_ask_returns_llm_response(self, chat):
        """ask() returns the response from the LLM."""
        response = chat.ask("What is happening?")
        assert response == "People were walking near the entrance."

    def test_ask_passes_event_context_to_llm(self, chat, mock_llm):
        """ask() builds context from timeline events and passes it to the LLM.

        The context should include the stream ID, event descriptions,
        motion scores, and the user's question.
        """
        chat.ask("Who was at the door?")

        mock_llm.chat.assert_called_once()
        call_kwargs = mock_llm.chat.call_args[1]
        messages = call_kwargs["messages"]
        system = call_kwargs.get("system", "")

        # System prompt is non-empty
        assert len(system) > 0

        # Messages contains one user message with context + question
        assert len(messages) == 1
        user_content = messages[0]["content"]
        assert "Stream: cam1" in user_content
        assert "Person detected" in user_content
        assert "Vehicle detected" in user_content
        assert "Person left frame" in user_content
        assert "motion=0.420" in user_content
        assert "motion=0.780" in user_content
        assert "motion=0.150" in user_content
        assert "Question: Who was at the door?" in user_content

    def test_ask_with_time_range_filters_events(self, chat, mock_llm, mock_store):
        """ask() with time_range calls get_range instead of get_recent."""
        chat.ask("What happened between 1000 and 1010?", time_range=(1000.0, 1010.0))

        mock_store.get_range.assert_called_once_with("cam1", 1000.0, 1010.0)
        mock_store.get_recent.assert_not_called()

        # Context should be built from the range-filtered events only
        call_kwargs = mock_llm.chat.call_args[1]
        user_content = call_kwargs["messages"][0]["content"]
        assert "Vehicle detected" in user_content

    def test_ask_empty_timeline_returns_fallback(self, mock_store, mock_llm):
        """ask() returns the fallback message when there are no events."""
        mock_store.get_recent.return_value = []
        config = LLMProviderConfig(provider="openai", api_key="test-key")
        chat = StreamChat(
            event_store=mock_store, stream_id="cam1", llm_provider_config=config
        )

        response = chat.ask("Anything happening?")
        assert response == "No events recorded yet for this stream."
        mock_llm.chat.assert_not_called()

    def test_clear_history_does_not_break_subsequent_asks(self, chat):
        """clear_history() does not error and subsequent ask() calls still work."""
        chat.ask("What is happening?")
        chat.clear_history()
        response = chat.ask("Anything new?")
        assert response is not None

    def test_no_llm_configured_returns_none(self, mock_store):
        """ask() returns None when no LLM provider was configured."""
        chat = StreamChat(event_store=mock_store, stream_id="cam1")
        response = chat.ask("What's happening?")
        assert response is None
