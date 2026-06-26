"""
Chat module — LLM-powered question answering over video contexts.

Uses a configured LLM provider (via Hermes CLI or direct API) to answer
questions about video content with source citations.
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import List, Optional

from video_analysis.config import Config
from video_analysis.rag import VideoRAG, RetrievedChunk
from video_analysis.models import ChatMessage, ChatSource, format_timestamp
from video_analysis.memory import ConversationMemory

logger = logging.getLogger(__name__)


class VideoChat:
    """Chat interface over indexed video content.

    Supports three backends:
    1. **Agentic Video Agent** (optional, v0.36.0) — multi-tool agent that
       dynamically selects tools (analyze_frames, search_rag, detect_objects,
       extract_text, search_transcript, temporal_grounding, summarize_video)
       based on the question type.
    2. **Video MLLM** (optional) — Video MLLM video-native Q&A that sees frame images directly
    3. **Hermes CLI** (default) — text-only RAG with retrieved chunks + Hermes/DeepSeek LLM

    The agent backend (when ``agent_enabled`` is set) takes precedence over
    the other backends because it can combine visual, textual, and object-level
    understanding in a single query. Falls back through Video MLLM → RAG when
    the agent is unavailable.
    """

    def __init__(self, rag: VideoRAG, config: Optional[Config] = None):
        self.rag = rag
        self.config = config or Config()
        self.history: List[ChatMessage] = []
        self._mllm = None  # lazy-loaded Video MLLM
        # Conversation memory (optional, ChromaDB-backed)
        self.memory: Optional[ConversationMemory] = None
        if self.config.conversation_memory_enabled:
            try:
                self.memory = ConversationMemory(self.config)
                logger.info("Conversation memory initialised")
            except Exception as exc:
                logger.warning(
                    "Failed to initialise conversation memory: %s — proceeding without it",
                    exc,
                )

    def _get_mllm(self):
        """Lazy-load Video MLLM for video-native Q&A."""
        if self._mllm is None:
            from video_analysis.video_mllm import VideoMLLM

            self._mllm = VideoMLLM(model_name=self.config.video_mllm_model)
        return self._mllm if self._mllm.available else None

    def ask(self, query: str, video_id: Optional[str] = None) -> ChatMessage:
        """
        Ask a question about video content.

        Three backends tried in priority order:

        1. **Agentic Video Agent** (``agent_enabled``) — multi-tool agent
           that dynamically selects tools based on question type. Requires
           the video file path for frame-level tools (detection, OCR,
           frame analysis).
        2. **Video MLLM** (``video_mllm_as_chat_backend``) — video-native
           visual Q&A using frame images directly.
        3. **RAG + Hermes CLI** (default) — text-only retrieval with
           Hermes/DeepSeek LLM.

        Args:
            query: Natural language question
            video_id: Optional — limit to specific video

        Returns:
            ChatMessage with answer and source citations
        """
        # Try Agentic Video Agent first (if enabled — v0.36.0)
        if self.config.agent_enabled:
            result = self._ask_agent(query, video_id)
            if result is not None:
                return result
            logger.info(
                "Agentic Video Agent returned None — falling through to next backend"
            )

        # Try Video MLLM backend next (video-native Q&A with visual context)
        if self.config.video_mllm_enabled and self.config.video_mllm_as_chat_backend:
            mllm = self._get_mllm()
            if mllm is not None:
                result = self._ask_mllm(query, video_id, mllm)
                if result is not None:
                    return result
                logger.info("Video MLLM QA returned None — falling back to RAG path")

        # RAG-based retrieval + Hermes CLI (default path)
        return self._ask_rag(query, video_id)

    def _get_agent_video_path(self, video_id: Optional[str]) -> Optional[str]:
        """Resolve the video file path for a given video_id.

        Searches the configured video directory and the RAG metadata.
        """
        if video_id is None:
            return None

        # Check video directory first
        vid_dir = self.config.video_dir
        for ext in (".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".m4v"):
            candidate = vid_dir / f"{video_id}{ext}"
            if candidate.exists():
                return str(candidate)

        # Check via RAG metadata for the filename
        if self.rag is not None:
            try:
                meta = self.rag.collection.get(
                    where={"video_id": video_id},
                    include=["metadatas"],
                    limit=1,
                )
                if meta["metadatas"]:
                    fname = meta["metadatas"][0].get("filename", "")
                    if fname:
                        for ext in (
                            ".mp4",
                            ".mkv",
                            ".avi",
                            ".mov",
                            ".webm",
                            ".flv",
                            ".m4v",
                        ):
                            candidate = vid_dir / f"{fname}{ext}"
                            if candidate.exists():
                                return str(candidate)
                            # Also try without extension if filename already has one
                            candidate = vid_dir / fname
                            if candidate.exists():
                                return str(candidate)
            except Exception:
                pass

        return None

    def _ask_agent(self, query: str, video_id: Optional[str]) -> Optional[ChatMessage]:
        """Ask using the Agentic Video Understanding Agent.

        Requires the video file path for frame-level tools (detection,
        OCR, frame analysis). Falls back gracefully when the video file
        or agent dependencies are unavailable.
        """
        import time as _time

        agent_start = _time.perf_counter()

        # Resolve video path
        video_path = self._get_agent_video_path(video_id)
        if video_path is None:
            logger.debug(
                "Agentic agent requires video file path — video_id=%s not found",
                video_id,
            )
            return None

        try:
            from video_analysis.agent import VideoUnderstandingAgent

            agent = VideoUnderstandingAgent(
                config=self.config,
                rag=self.rag,
                video_path=video_path,
                video_id=video_id,
            )

            result = agent.query(
                question=query,
                max_tools=self.config.agent_max_tools,
            )

            agent_dur = _time.perf_counter() - agent_start

            logger.info(
                "Agentic agent: %d tools used in %.1fs, confidence=%.2f",
                result.tools_used,
                agent_dur,
                result.confidence,
            )

            if not result.answer or len(result.answer.strip()) < 10:
                logger.debug("Agent produced empty/short answer — falling through")
                return None

            message = ChatMessage(
                role="assistant",
                content=result.answer,
                sources=[],  # Agent evidence is embedded in the answer text
            )
            self.history.append(ChatMessage(role="user", content=query))
            self.history.append(message)
            return message

        except ImportError as exc:
            logger.debug(
                "Agentic agent dependencies not available: %s — falling through", exc
            )
            return None
        except Exception as exc:
            logger.warning(
                "Agentic agent failed: %s — falling through to next backend", exc
            )
            return None

    def _ask_mllm(
        self, query: str, video_id: Optional[str], mllm
    ) -> Optional[ChatMessage]:
        """Ask using Video MLLM with frame images as visual context."""
        # Collect frame images from the video
        frame_paths = []
        if video_id:
            try:
                meta_result = self.rag.collection.get(
                    where={"video_id": video_id},
                    include=["metadatas"],
                )
                if meta_result["ids"]:
                    seen = set()
                    for meta in meta_result["metadatas"]:
                        fp = meta.get("frame_path")
                        if fp and fp not in seen:
                            seen.add(fp)
                            path = Path(fp)
                            if path.exists():
                                frame_paths.append(str(path))
            except Exception:
                pass

        if not frame_paths:
            logger.debug("No frames found for MLLM QA — returning None")
            return None

        # Build a prompt that includes the query
        prompt = (
            f"You are a video analysis assistant. Answer based on the video frames shown.\n\n"
            f"Question: {query}\n\n"
            f"Provide a concise answer with timestamp references where possible."
        )

        answer = mllm.answer(query=prompt, frames=frame_paths[:8])
        if not answer:
            return None

        message = ChatMessage(
            role="assistant",
            content=answer,
            sources=[],
        )
        self.history.append(ChatMessage(role="user", content=query))
        self.history.append(message)
        return message

    def _ask_rag(self, query: str, video_id: Optional[str] = None) -> ChatMessage:
        import time as _time

        _start = _time.perf_counter()
        # Retrieve relevant context — use agentic retrieval if enabled,
        # otherwise use routed retrieval (or standard retrieval)
        if self.config.agentic_retrieval_enabled:
            chunks = self.rag.agentic_retrieve(query, video_id=video_id)
            _method = "agentic"
        elif (
            self.config.query_routing_enabled
            or self.config.scene_graph_enabled
            or self.config.multi_hop_enabled
        ):
            chunks = self.rag.routed_retrieve(query, video_id=video_id)
            _method = "routed"
        else:
            chunks = self.rag.retrieve(query, video_id=video_id)
            _method = "simple"

        _retrieval_dur = _time.perf_counter() - _start

        # Record retrieval metrics
        try:
            from video_analysis.metrics import increment_question, observe_retrieval

            increment_question(method=_method)
            observe_retrieval(
                chunks=len(chunks) if chunks else 0,
                method=_method,
                duration_s=_retrieval_dur,
            )
        except Exception:
            pass

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

        # Retrieve relevant conversation memories and prepend to context
        memory_context = self._get_memory_context(query)

        # Build prompt (with memory context if available)
        if memory_context:
            prompt = self._build_prompt_with_memory(query, context, memory_context)
        else:
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

        # Store Q&A pair in conversation memory
        if self.memory is not None:
            try:
                self.memory.add_entry(question=query, answer=answer, video_id=video_id)
            except Exception as exc:
                logger.debug("Failed to store conversation memory entry: %s", exc)

        return message

    def ask_with_history(
        self, query: str, video_id: Optional[str] = None
    ) -> ChatMessage:
        """Ask with conversation history included in context.

        When ``video_mllm_as_chat_backend`` is enabled, uses Video MLLM
        for video-native Q&A. Falls back to the text-only RAG path
        when the MLLM is unavailable.
        """
        # Try Video MLLM backend first (video-native Q&A with visual context)
        if self.config.video_mllm_enabled and self.config.video_mllm_as_chat_backend:
            mllm = self._get_mllm()
            if mllm is not None:
                result = self._ask_mllm(query, video_id, mllm)
                if result is not None:
                    return result
                logger.info("Video MLLM QA returned None — falling back to RAG path")

        # RAG-based retrieval + Hermes CLI (default path)
        if self.config.agentic_retrieval_enabled:
            chunks = self.rag.agentic_retrieve(query, video_id=video_id)
        elif (
            self.config.query_routing_enabled
            or self.config.scene_graph_enabled
            or self.config.multi_hop_enabled
        ):
            chunks = self.rag.routed_retrieve(query, video_id=video_id)
        else:
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

        # Retrieve conversation memory context
        memory_context = self._get_memory_context(query)

        # Build prompt with relevant memories prepended to context
        prompt = self._build_prompt_with_history_and_memory(
            query, context, history_text, memory_context
        )

        answer = self._call_llm(prompt)

        message = ChatMessage(
            role="assistant",
            content=answer,
            sources=sources,
        )

        self.history.append(ChatMessage(role="user", content=query))
        self.history.append(message)

        # Store Q&A pair in conversation memory
        if self.memory is not None:
            try:
                self.memory.add_entry(question=query, answer=answer, video_id=video_id)
            except Exception as exc:
                logger.debug("Failed to store conversation memory entry: %s", exc)

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

    # ------------------------------------------------------------------
    # Conversation Memory helpers
    # ------------------------------------------------------------------

    def _get_memory_context(self, query: str) -> str:
        """Retrieve relevant conversation memories for context.

        Returns a formatted string of past Q&A pairs, or empty string if
        memory is not available or no relevant memories found.
        """
        if self.memory is None:
            return ""

        try:
            memories = self.memory.get_relevant(query, top_k=3)
        except Exception as exc:
            logger.debug("Failed to retrieve conversation memories: %s", exc)
            return ""

        if not memories:
            return ""

        parts = ["Previous relevant conversations:"]
        for i, mem in enumerate(memories, 1):
            vid = f" (video: {mem.video_id})" if mem.video_id else ""
            parts.append(f"{i}. Q: {mem.question}{vid}\n   A: {mem.answer[:300]}")
        return "\n".join(parts)

    def _build_prompt_with_memory(
        self, query: str, context: str, memory_context: str
    ) -> str:
        """Build a prompt that includes conversation memory context."""
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

{memory_context}

Current context:
{context}

Question: {query}

Answer with timestamp citations where relevant:"""

    def _build_prompt_with_history_and_memory(
        self, query: str, context: str, history: str, memory_context: str
    ) -> str:
        """Build a prompt that includes both conversation history and memory context."""
        parts = (
            [memory_context, f"Previous conversation:\n{history}"]
            if memory_context
            else [f"Previous conversation:\n{history}"]
        )
        combined_memory = "\n\n".join(parts)
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

{combined_memory}

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
