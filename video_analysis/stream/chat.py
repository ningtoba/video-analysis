"""
Chat engine for stream events — queries the event timeline and can
request re-analysis of specific frames.

Same interface as video_analysis.chat.VideoChat but operates over the
temporal event stream instead of offline analysis JSON.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from video_analysis.llm_provider import LLMProviderConfig, get_llm_provider
from video_analysis.stream.store import EventStore

logger = logging.getLogger(__name__)

_CHAT_SYSTEM_PROMPT = """You are a video surveillance analyst. You answer questions about
what's happening in a live or recorded video feed based on the event timeline.

The timeline contains timestamped descriptions of events detected by the system.

Rules:
1. Answer based ONLY on the provided event timeline
2. When referencing specific moments, include timestamps
3. If the timeline doesn't have enough information, say so
4. Be concise and factual

Context format:
[TIMESTAMP] Description (motion_score)
[TIMESTAMP] Description (motion_score)
..."""


class StreamChat:
    """Chat interface over a stream's event timeline."""

    def __init__(
        self,
        event_store: EventStore,
        stream_id: str,
        llm_provider_config: Optional[LLMProviderConfig] = None,
    ):
        self._store = event_store
        self._stream_id = stream_id
        self._llm = None
        if llm_provider_config:
            self._llm = get_llm_provider(llm_provider_config)
        self._history: List[dict] = []

    def ask(self, question: str, time_range: Optional[tuple] = None) -> Optional[str]:
        """Ask a question about the stream's events.

        Args:
            question: Natural language question.
            time_range: Optional (start_ts, end_ts) to scope the query.

        Returns:
            Answer text, or None if LLM not configured.
        """
        if not self._llm:
            logger.warning("No LLM configured for StreamChat")
            return None

        # Get events
        if time_range:
            events = self._store.get_range(
                self._stream_id, time_range[0], time_range[1]
            )
        else:
            events = self._store.get_recent(self._stream_id, limit=100)

        if not events:
            return "No events recorded yet for this stream."

        # Build context
        context_parts = [f"Stream: {self._stream_id}"]
        for e in events:
            ts_str = _format_ts(e.timestamp)
            context_parts.append(
                f"[{ts_str}] {e.description} (motion={e.motion_score:.3f})"
            )

        context = "\n".join(context_parts)

        try:
            response = self._llm.chat(
                messages=[{"role": "user", "content": f"{context}\n\nQuestion: {question}"}],
                system=_CHAT_SYSTEM_PROMPT,
            )
            if response:
                self._history.append({"role": "user", "content": question})
                self._history.append({"role": "assistant", "content": response})
            return response
        except Exception as e:
            logger.error("Stream chat failed: %s", e)
            return None

    def clear_history(self):
        self._history.clear()


def _format_ts(ts: float) -> str:
    import datetime
    dt = datetime.datetime.fromtimestamp(ts)
    return dt.strftime("%H:%M:%S")
