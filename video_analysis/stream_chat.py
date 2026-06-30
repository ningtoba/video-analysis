"""
MLLM Streaming Q&A — token-by-token streaming of LLM responses.

Provides the ``StreamChatManager`` coordinator class and a top-level
convenience generator ``stream_llm_response()``.  Both support
``HermesProvider`` and ``OpenAIProvider`` from the existing
:mod:`video_analysis.llm_provider` module.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import AsyncGenerator, Dict, List, Optional

from video_analysis.llm_provider import (
    LLMProvider,
    LLMProviderConfig,
    get_llm_provider,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Convenience generator
# ---------------------------------------------------------------------------


async def stream_llm_response(
    prompt: str,
    system: str = "",
    provider: Optional[LLMProvider] = None,
    config: Optional[LLMProviderConfig] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout: Optional[int] = None,
) -> AsyncGenerator[str, None]:
    """Yield response tokens from an LLM as they become available.

    Uses the provider's ``stream_chat()`` method.  If the provider does
    not support native streaming (i.e. it yields a single token), this
    generator will still produce exactly one token — the full response.

    Args:
        prompt: The user message / prompt text.
        system: Optional system prompt.
        provider: An already-initialised LLM provider, or *None* to
            auto-resolve via :func:`get_llm_provider`.
        config: Provider configuration (ignored when *provider* is passed).
        temperature: Temperature override.
        max_tokens: Max tokens override.
        timeout: Network / subprocess timeout in seconds.

    Yields:
        Individual tokens (``str``) as they arrive.
    """
    llm = provider or get_llm_provider(config)

    async for token in llm.stream_chat(
        prompt=prompt,
        system=system,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    ):
        yield token


# ---------------------------------------------------------------------------
# Stream Chat Manager
# ---------------------------------------------------------------------------


@dataclass
class StreamSession:
    """Represents a single streaming chat session."""

    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    history: List[Dict[str, str]] = field(default_factory=list)
    is_active: bool = True
    created_at: float = field(default_factory=time.time)


class StreamChatManager:
    """Coordinator for multiple streaming LLM chat sessions.

    Manages session lifecycle, history, and concurrent streaming
    generators per session.
    """

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        config: Optional[LLMProviderConfig] = None,
        max_sessions: int = 50,
    ):
        self._provider = provider or get_llm_provider(config)
        self._config = config
        self._sessions: Dict[str, StreamSession] = {}
        self._max_sessions = max_sessions

    @property
    def provider(self) -> LLMProvider:
        """The underlying LLM provider."""
        return self._provider

    def create_session(self) -> str:
        """Create a new streaming session and return its ID."""
        self._evict_stale()
        session = StreamSession()
        self._sessions[session.session_id] = session
        logger.debug("Created stream session %s", session.session_id)
        return session.session_id

    def get_session(self, session_id: str) -> Optional[StreamSession]:
        """Look up a session by ID."""
        return self._sessions.get(session_id)

    def end_session(self, session_id: str) -> None:
        """Mark a session as inactive and optionally remove it."""
        session = self._sessions.get(session_id)
        if session:
            session.is_active = False
            self._sessions.pop(session_id, None)
            logger.debug("Ended stream session %s", session_id)

    async def stream(
        self,
        session_id: str,
        prompt: str,
        system: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream a response in the context of a session.

        Appends the user message to session history, then yields tokens
        from the provider.  The full answer is also appended to history
        after streaming completes.
        """
        session = self._sessions.get(session_id)
        if session is None:
            logger.warning("Unknown session %s — creating ephemeral session", session_id)
            session = StreamSession(session_id=session_id)
            self._sessions[session_id] = session

        # Track user message
        session.history.append({"role": "user", "content": prompt})

        # Collect the full response while streaming
        full_response: List[str] = []
        try:
            async for token in self._provider.stream_chat(
                prompt=prompt,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            ):
                full_response.append(token)
                yield token
        except Exception as e:
            logger.error("Stream error in session %s: %s", session_id, e)
            error_msg = f"\n\n⚠️ Streaming error: {str(e)[:200]}"
            full_response.append(error_msg)
            yield error_msg
        finally:
            # Store full response in history
            answer = "".join(full_response)
            session.history.append({"role": "assistant", "content": answer})

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """Return the conversation history for a session."""
        session = self._sessions.get(session_id)
        return list(session.history) if session else []

    def clear_history(self, session_id: str) -> None:
        """Clear conversation history for a session."""
        session = self._sessions.get(session_id)
        if session:
            session.history.clear()

    def active_session_count(self) -> int:
        """Number of currently active sessions."""
        return sum(1 for s in self._sessions.values() if s.is_active)

    def _evict_stale(self) -> None:
        """Evict the oldest session if we've exceeded ``max_sessions``."""
        while len(self._sessions) >= self._max_sessions:
            oldest_id = min(
                self._sessions.keys(),
                key=lambda sid: self._sessions[sid].created_at,
            )
            self.end_session(oldest_id)
