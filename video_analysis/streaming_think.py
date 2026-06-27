"""
Streaming Thinking — Amortized Video Reasoning During Real-Time Streaming.

Inspired by:

- **Video Streaming Thinking (VST)** (arXiv:2603.12262, mid-2026) — "watch
  and think simultaneously": activates reasoning over incoming video chunks
  during streaming, amortizing LLM reasoning latency over video playback.
  VST-SFT adapts offline VideoLLM to causal streaming reasoning.  VST-RL
  provides end-to-end improvement via multi-turn video interaction.
  VST-7B achieves 79.5% on StreamingBench, 15.7× faster than Video-R1,
  +5.4% on VideoHolmes.
- **OneClip-RAG** (arXiv:2512.08410) — query-guided video chunking that
  unifies clip chunking and cross-modal retrieval in one step.
- **DynFrame** (arXiv:2605.26680) — learnable frame sampling via SD-GRPO.

This module implements the *inference-time* streaming thinking pattern:
- Chunks arrive from StreamingPipeline incrementally.
- Each chunk triggers lightweight "thinking" (reasoning) that builds on
  the previous chunk's accumulated context.
- The accumulated "thought state" persists across chunks.
- Queries can be answered incrementally — partial answers refine as more
  chunks arrive.
- StreamingThinkingPipeline wraps StreamingPipeline with thinking support.

Usage::

    from video_analysis.streaming_think import StreamingThinkingPipeline

    pipeline = StreamingThinkingPipeline(config)
    for thought in pipeline.process_with_thinking("video.mp4"):
        print(f"Chunk {thought.chunk_index}: {thought.thought_summary}")
        print(f"Partial answers: {thought.partial_answers}")

    # Get final accumulated wisdom
    final = pipeline.final_thoughts()
"""

from __future__ import annotations

import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Tuple

from video_analysis.config import Config
from video_analysis.streaming import (
    StreamingPipeline,
    StreamingChunkResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ThoughtState:
    """Accumulated thinking state across streaming chunks.

    Attributes:
        chunks_seen: Number of chunks processed so far.
        total_duration: Total duration processed.
        summary: Accumulated high-level summary of what's been seen.
        events: Events detected across chunks.
        entities: Entities observed.
        causal_observations: Causal chains observed so far.
        unanswered_questions: Questions raised but not yet answered.
        partial_answers: Answer fragments accumulated so far.
        last_thought: Last thought/insight generated.
        refined_answers: Dictionary of query -> best answer so far.
    """

    chunks_seen: int = 0
    total_duration: float = 0.0
    summary: str = ""
    events: List[str] = field(default_factory=list)
    entities: Dict[str, int] = field(default_factory=dict)
    causal_observations: List[str] = field(default_factory=list)
    unanswered_questions: List[str] = field(default_factory=list)
    partial_answers: Dict[str, str] = field(default_factory=dict)
    last_thought: str = ""
    refined_answers: Dict[str, str] = field(default_factory=dict)


@dataclass
class StreamingThought:
    """A single thinking step produced for a streaming chunk.

    Attributes:
        chunk_index: Index of the chunk that triggered this thought.
        start_time: Start time of the chunk.
        end_time: End time of the chunk.
        thought_summary: Short sentence summarizing what was learned.
        insights: Specific insights extracted from this chunk.
        causal_prediction: Predicted next event (forward thinking).
        causal_explanation: Explanation of current event (backward thinking).
        questions: New questions raised by this chunk.
        confidence: Confidence in the thought [0, 1].
        metadata: Extra metadata.
    """

    chunk_index: int
    start_time: float
    end_time: float
    thought_summary: str = ""
    insights: List[str] = field(default_factory=list)
    causal_prediction: str = ""
    causal_explanation: str = ""
    questions: List[str] = field(default_factory=list)
    confidence: float = 1.0
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Streaming Thinking Pipeline
# ---------------------------------------------------------------------------


class StreamingThinkingPipeline:
    """Wraps StreamingPipeline with amortized reasoning.

    As chunks arrive from the streaming pipeline, the ThinkingPipeline:
    1. Runs the chunk through the existing pipeline (transcription, scene
       detection, object detection, etc.)
    2. Applies **thinking** — lightweight reasoning over the chunk's content,
       building on accumulated context from previous chunks.
    3. Maintains a **ThoughtState** that persists across the entire stream.
    4. Supports **incremental answers** — queries can be answered at any
       point, with accuracy improving as more chunks arrive.

    Unlike VST which requires model fine-tuning, this module implements
    the *inference-time* streaming thinking pattern:
    - Accumulates temporal context via the ThoughtState.
    - Uses LLM for lightweight chunk-level reasoning.
    - Maintains causal hypotheses that are refined over time.
    - Supports "pause and answer" — answer queries at any point with
      best-effort partial context.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        llm_provider=None,
        streaming_pipeline: Optional[StreamingPipeline] = None,
        thinking_interval: int = 1,  # think on every chunk
        max_thought_history: int = 100,
    ):
        self.config = config or Config()
        self._llm = llm_provider
        self._streaming = streaming_pipeline or StreamingPipeline(config)
        self.thinking_interval = thinking_interval
        self.max_thought_history = max_thought_history

        self._thought_state = ThoughtState()
        self._thought_history: List[StreamingThought] = []
        self._last_chunk_results: List[StreamingChunkResult] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_with_thinking(
        self,
        video_path: str,
        chunk_duration: Optional[float] = None,
        **streaming_kwargs,
    ) -> Generator[StreamingThought, None, None]:
        """Process a video with streaming thinking.

        Yields a StreamingThought for each chunk processed.

        Args:
            video_path: Path to the video file.
            chunk_duration: Duration of each chunk (default from config).
            **streaming_kwargs: Additional args passed to StreamingPipeline.

        Yields:
            StreamingThought for each processed chunk.
        """
        chunk_duration = chunk_duration or self.config.streaming_chunk_duration

        for chunk_result in self._streaming.process_streaming(
            video_path,
            chunk_duration=chunk_duration,
            **streaming_kwargs,
        ):
            thought = self._think(chunk_result)
            self._thought_history.append(thought)
            if len(self._thought_history) > self.max_thought_history:
                self._thought_history.pop(0)
            self._last_chunk_results.append(chunk_result)
            yield thought

    def process_live_with_thinking(
        self,
        stream_url: str,
        chunk_duration: Optional[float] = None,
        **streaming_kwargs,
    ) -> Generator[StreamingThought, None, None]:
        """Process a live stream with streaming thinking.

        Args:
            stream_url: RTMP/RTSP/HLS URL.
            chunk_duration: Duration of each chunk.
            **streaming_kwargs: Additional StreamingPipeline args.

        Yields:
            StreamingThought for each live chunk.
        """
        chunk_duration = chunk_duration or self.config.live_stream_chunk_duration

        for chunk_result in self._streaming.process_live_stream(
            stream_url,
            chunk_duration=chunk_duration,
            **streaming_kwargs,
        ):
            thought = self._think(chunk_result)
            self._thought_history.append(thought)
            if len(self._thought_history) > self.max_thought_history:
                self._thought_history.pop(0)
            self._last_chunk_results.append(chunk_result)
            yield thought

    def answer(
        self,
        query: str,
        use_llm: bool = True,
    ) -> str:
        """Answer a query using the accumulated streaming context.

        Args:
            query: Natural language question about the video.
            use_llm: If True and LLM is available, use LLM for synthesis.
                     Otherwise, returns best-effort context summary.

        Returns:
            Best-effort answer based on accumulated knowledge.
        """
        # Check if we already have a refined answer
        if query in self._thought_state.refined_answers:
            return self._thought_state.refined_answers[query]

        if use_llm and self._llm is not None:
            return self._answer_with_llm(query)
        else:
            return self._answer_from_context(query)

    def final_thoughts(self) -> ThoughtState:
        """Get the final accumulated thinking state."""
        return self._thought_state

    def get_timeline(self) -> List[Dict[str, Any]]:
        """Get a structured timeline of streaming thoughts.

        Returns:
            List of dicts with chunk_index, start_time, end_time,
            thought_summary, insights.
        """
        return [
            {
                "chunk_index": t.chunk_index,
                "start_time": t.start_time,
                "end_time": t.end_time,
                "thought_summary": t.thought_summary,
                "insights": t.insights,
                "causal_prediction": t.causal_prediction,
                "causal_explanation": t.causal_explanation,
            }
            for t in self._thought_history
        ]

    def reset(self) -> None:
        """Reset the thinking state for a new video."""
        self._thought_state = ThoughtState()
        self._thought_history = []
        self._last_chunk_results = []
        # Reset the underlying streaming pipeline
        self._streaming = StreamingPipeline(self.config)

    # ------------------------------------------------------------------
    # Thinking logic
    # ------------------------------------------------------------------

    def _think(self, chunk: StreamingChunkResult) -> StreamingThought:
        """Generate a thinking step from a streaming chunk result.

        Updates the thought state and returns a StreamingThought with
        insights, causal predictions, and questions.

        Args:
            chunk: The result from processing a streaming chunk.

        Returns:
            StreamingThought with reasoning insights.
        """
        ts = self._thought_state
        ts.chunks_seen += 1
        ts.total_duration += chunk.duration

        # Update entity counts
        for obj in chunk.objects_found:
            ts.entities[obj] = ts.entities.get(obj, 0) + 1

        # Build transcript summary
        transcript_excerpt = chunk.full_transcript
        if len(transcript_excerpt) > 500:
            transcript_excerpt = (
                transcript_excerpt[:250] + "...[...]..." + transcript_excerpt[-250:]
            )

        # Scene summary
        scene_texts = []
        for scene in chunk.scenes:
            scene_start = scene.start_time if scene.start_time is not None else 0.0
            scene_texts.append(f"[{scene_start:.1f}s] Scene {scene.scene_id}")
        scene_summary = "; ".join(scene_texts) if scene_texts else "(no scenes)"

        thought = StreamingThought(
            chunk_index=chunk.chunk_index,
            start_time=chunk.start_time,
            end_time=chunk.end_time,
        )

        # Generate insights
        insights = []

        # 1. Object/entity insight
        if chunk.objects_found:
            obj_str = ", ".join(chunk.objects_found[:5])
            insights.append(f"Objects detected: {obj_str}")

        # 2. Scene change insight
        if chunk.scenes:
            insights.append(f"Scene changes: {len(chunk.scenes)} scenes in this chunk")

        # 3. Transcript insight
        if chunk.full_transcript:
            words = chunk.full_transcript.split()
            insights.append(f"Speech: ~{len(words)} words spoken")

        # 4. Trend detection — entity appearing more
        if ts.entities:
            top_entities = sorted(
                ts.entities.items(), key=lambda x: x[1], reverse=True
            )[:3]
            if top_entities:
                insights.append(
                    f"Top entities: {', '.join(f'{e} ({c}x)' for e, c in top_entities)}"
                )

        thought.insights = insights

        # Generate thought summary
        thought.thought_summary = self._build_thought_summary(chunk, insights, ts)

        # Causal prediction (forward thinking)
        if ts.chunks_seen > 1:
            thought.causal_prediction = self._predict_next(chunk, ts)
            if thought.causal_prediction:
                ts.causal_observations.append(
                    f"Predicted after chunk {chunk.chunk_index}: "
                    f"{thought.causal_prediction}"
                )

        # Causal explanation (backward thinking)
        if ts.chunks_seen > 1:
            thought.causal_explanation = self._explain_current(chunk, ts)

        # Generate questions
        thought.questions = self._generate_questions(chunk, ts)
        ts.unanswered_questions.extend(thought.questions)

        # Update last thought
        ts.last_thought = thought.thought_summary
        if chunk.full_transcript:
            ts.summary = self._update_summary(ts.summary, chunk)

        return thought

    def _build_thought_summary(
        self,
        chunk: StreamingChunkResult,
        insights: List[str],
        ts: ThoughtState,
    ) -> str:
        """Build a short summary of what happened in this chunk."""
        parts = []
        if chunk.full_transcript:
            # Extract first meaningful sentence
            text = chunk.full_transcript.strip()
            if text:
                # Take first sentence or first 80 chars
                first_sentence = text.split(".")[0]
                if len(first_sentence) > 80:
                    first_sentence = first_sentence[:80] + "..."
                parts.append(f'"{first_sentence}"')
        if ts.causal_observations:
            # Use last causal observation
            parts.append(ts.causal_observations[-1][:80])
        if not parts:
            parts.append(f"Chunk {chunk.chunk_index}: {len(chunk.scenes)} scenes")
        return "; ".join(parts)

    def _predict_next(
        self,
        chunk: StreamingChunkResult,
        ts: ThoughtState,
    ) -> str:
        """Predict what might happen next (forward causal thinking).

        Uses entity trends and scene trajectory to make a lightweight
        prediction about the next chunk's likely content.
        """
        predictions = []

        # If entities have been consistently appearing, predict they continue
        if ts.entities:
            top = sorted(ts.entities.items(), key=lambda x: x[1], reverse=True)
            if top:
                entity, count = top[0]
                if count > 2:
                    predictions.append(f"Likely continuing {entity} presence")

        # If transcript was ongoing, predict it continues
        if chunk.full_transcript and len(chunk.full_transcript) > 50:
            last_words = chunk.full_transcript.strip().split()[-5:]
            if last_words:
                predictions.append(
                    f"Speech may continue about \"{' '.join(last_words)}\""
                )

        # Scene trajectory
        if chunk.scenes:
            predictions.append("Scene may transition or continue")

        return "; ".join(predictions[:2]) if predictions else ""

    def _explain_current(
        self,
        chunk: StreamingChunkResult,
        ts: ThoughtState,
    ) -> str:
        """Explain how the current chunk relates to previous ones (backward causal).

        Looks at entity continuity and topic transition.
        """
        explanations = []

        # Entity continuity
        if ts.entities and chunk.objects_found:
            new_entities = set(chunk.objects_found)
            existing = set(ts.entities.keys()) - new_entities
            shared = new_entities & set(ts.entities.keys())
            if shared:
                explanations.append(f"Continuation of: {', '.join(list(shared)[:3])}")
            if new_entities - set(ts.entities.keys()):
                explanations.append(
                    f"New elements introduced: {', '.join(list(new_entities - set(ts.entities.keys()))[:3])}"
                )

        return "; ".join(explanations[:2]) if explanations else ""

    def _generate_questions(
        self,
        chunk: StreamingChunkResult,
        ts: ThoughtState,
    ) -> List[str]:
        """Generate open questions based on this chunk's content."""
        questions = []
        if chunk.objects_found and ts.entities:
            # If new objects appeared without explanation
            new_objs = set(chunk.objects_found) - set(ts.entities.keys())
            if new_objs:
                items = ", ".join(list(new_objs)[:2])
                questions.append(f"Why did {items} appear?")
        return questions

    def _update_summary(self, current_summary: str, chunk: StreamingChunkResult) -> str:
        """Update the accumulated summary with new chunk info."""
        if not chunk.full_transcript:
            return current_summary
        # Keep last ~500 chars
        new_text = chunk.full_transcript
        if current_summary:
            combined = current_summary + "\n" + new_text
            if len(combined) > 2000:
                # Keep last 2000 chars
                combined = "..." + combined[-2000:]
            return combined
        return new_text[:2000]

    def _answer_with_llm(self, query: str) -> str:
        """Use LLM to answer a query from accumulated streaming context."""
        if self._llm is None:
            return self._answer_from_context(query)

        # Build streaming context
        ts = self._thought_state
        context_parts = [
            f"Video duration seen: {ts.total_duration:.0f}s across {ts.chunks_seen} chunks.",
        ]
        if ts.summary:
            context_parts.append(f"Transcript history: {ts.summary[-1500:]}")
        if ts.entities:
            context_parts.append(
                f"Entities observed: {', '.join(f'{k} ({v}x)' for k, v in sorted(ts.entities.items(), key=lambda x: x[1], reverse=True)[:10])}"
            )
        context = "\n".join(context_parts)

        prompt = (
            "You are answering a question about a video that is being streamed in "
            "real-time. You have partial context — you've seen some chunks but not "
            "the full video. Answer based on what you know so far, and note any "
            "uncertainty.\n\n"
            f"### Accumulated Streaming Context\n{context}\n\n"
            f"### Question\n{query}\n\n"
            "Provide the best answer you can from the available context. "
            "If the question cannot be fully answered yet, explain what's missing."
        )

        try:
            if hasattr(self._llm, "chat"):
                result = self._llm.chat(prompt)
            elif hasattr(self._llm, "generate"):
                result = self._llm.generate(prompt, max_tokens=1024)
            else:
                result = self._llm(prompt)
            answer = result
            if hasattr(result, "content"):
                answer = result.content
            # Cache the answer
            self._thought_state.refined_answers[query] = str(answer)
            return str(answer)
        except Exception as e:
            logger.warning(f"LLM answer failed: {e}")
            return self._answer_from_context(query)

    def _answer_from_context(self, query: str) -> str:
        """Direct context-based answer without LLM."""
        ts = self._thought_state
        query_lower = query.lower()

        # Simple keyword matching for entity questions
        for entity, count in sorted(
            ts.entities.items(), key=lambda x: x[1], reverse=True
        ):
            if entity.lower() in query_lower:
                return (
                    f"Based on {ts.chunks_seen} chunks ({ts.total_duration:.0f}s "
                    f"of video): '{entity}' was observed {count} time(s). "
                    f"Transcript context available."
                )

        # General summary
        return (
            f"Processed {ts.chunks_seen} chunks ({ts.total_duration:.0f}s). "
            f"Entities: {list(ts.entities.keys())[:5] if ts.entities else 'none detected so far'}. "
            f"{'Transcript available.' if ts.summary else 'No transcript yet.'}"
        )

    # ------------------------------------------------------------------
    # Delegated StreamingPipeline methods
    # ------------------------------------------------------------------

    @property
    def pipeline(self) -> StreamingPipeline:
        """Access the underlying StreamingPipeline."""
        return self._streaming

    def final_index(self) -> Any:
        """Get the final VideoIndex from the streaming pipeline."""
        return self._streaming.final_index()

    def cleanup(self) -> None:
        """Clean up temporary files."""
        self._streaming.cleanup()
