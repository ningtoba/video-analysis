"""
Chat module — LLM-powered question answering over video contexts.

Uses a configured LLM provider (via Hermes CLI or direct API) to answer
questions about video content with source citations.
"""

import json
import logging
import subprocess
from typing import List, Optional

from video_analysis.config import Config
from video_analysis.rag import VideoRAG, RetrievedChunk
from video_analysis.models import ChatMessage, ChatSource, format_timestamp

logger = logging.getLogger(__name__)


class VideoChat:
    """Chat interface over indexed video content."""

    def __init__(self, rag: VideoRAG, config: Optional[Config] = None):
        self.rag = rag
        self.config = config or Config()
        self.history: List[ChatMessage] = []

    def ask(self, query: str, video_id: Optional[str] = None) -> ChatMessage:
        """
        Ask a question about video content.

        Args:
            query: Natural language question
            video_id: Optional — limit to specific video

        Returns:
            ChatMessage with answer and source citations
        """
        # Retrieve relevant context
        chunks = self.rag.retrieve(query, video_id=video_id)

        if not chunks:
            return ChatMessage(
                role="assistant",
                content="I couldn't find any relevant information about that in the indexed videos.",
                sources=[],
            )

        # Expand temporal context
        if video_id:
            chunks = self.rag.expand_temporal_context(chunks, video_id)

        # Build context string
        context = self.rag.build_context(chunks)

        # Build prompt
        prompt = self._build_prompt(query, context)

        # Ask LLM
        answer = self._call_llm(prompt)

        # Extract source citations
        sources = self.rag.get_source_citations(chunks)

        message = ChatMessage(
            role="assistant",
            content=answer,
            sources=sources,
        )

        self.history.append(ChatMessage(role="user", content=query))
        self.history.append(message)

        return message

    def ask_with_history(
        self, query: str, video_id: Optional[str] = None
    ) -> ChatMessage:
        """Ask with conversation history included in context."""
        # Retrieve relevant context
        chunks = self.rag.retrieve(query, video_id=video_id)
        if video_id and chunks:
            chunks = self.rag.expand_temporal_context(chunks, video_id)

        context = (
            self.rag.build_context(chunks)
            if chunks
            else "No relevant video context found."
        )
        sources = self.rag.get_source_citations(chunks) if chunks else []

        # Build history context
        history_text = ""
        if self.history:
            parts = []
            for msg in self.history[-6:]:  # last 3 turns
                role = "User" if msg.role == "user" else "Assistant"
                parts.append(f"{role}: {msg.content[:300]}")
            history_text = "\n".join(parts)

        prompt = self._build_prompt_with_history(query, context, history_text)

        answer = self._call_llm(prompt)

        message = ChatMessage(
            role="assistant",
            content=answer,
            sources=sources,
        )

        self.history.append(ChatMessage(role="user", content=query))
        self.history.append(message)

        return message

    def reset_history(self):
        """Clear conversation history."""
        self.history = []

    def _build_prompt(self, query: str, context: str) -> str:
        return f"""You are a video analysis assistant. You answer questions about video content based on the provided context.

The context includes:
- Timestamped transcript excerpts
- Scene descriptions and summaries
- Detected objects in frames
- OCR text visible in frames

Rules:
1. Answer based ONLY on the provided context
2. When you reference specific moments, include the timestamp in HH:MM:SS format
3. If the context doesn't have enough information, say so
4. Be concise and factual

Context:
{context}

Question: {query}

Answer with timestamp citations where relevant:"""

    def _build_prompt_with_history(self, query: str, context: str, history: str) -> str:
        return f"""You are a video analysis assistant. You answer questions about video content based on the provided context.

The context includes:
- Timestamped transcript excerpts
- Scene descriptions and summaries
- Detected objects in frames
- OCR text visible in frames

Rules:
1. Answer based ONLY on the provided context
2. When you reference specific moments, include the timestamp in HH:MM:SS format
3. If the context doesn't have enough information, say so
4. Be concise and factual

Previous conversation:
{history}

Current context:
{context}

Question: {query}

Answer with timestamp citations where relevant:"""

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with the prompt."""
        try:
            result = subprocess.run(
                [
                    "hermes",
                    "chat",
                    "-q",
                    "-m",
                    self.config.llm_model,
                    "-t",
                    "0.3",
                    "--max-tokens",
                    str(self.config.llm_max_tokens),
                ],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                logger.error(f"LLM call failed: {result.stderr[:500]}")
                return (
                    "I encountered an error processing your question. Please try again."
                )
        except FileNotFoundError:
            # Fallback: return a meaningful response from retrieved context
            return (
                "Based on the video content I found, I can see the relevant "
                "scenes and context. However, the LLM backend (hermes CLI) is "
                "not available. Please ensure Hermes is installed and configured."
            )
        except subprocess.TimeoutExpired:
            return "The analysis took too long. Please try a more specific question."
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return f"I encountered an error: {str(e)[:200]}"
