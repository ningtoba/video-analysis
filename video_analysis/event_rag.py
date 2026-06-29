"""
Event-Causal RAG — Semantic Event Segmentation, State-Event-State Graphs,
and Bidirectional Causal-Topological Retrieval for Long-Video Understanding.

Inspired by:

- **Event-Causal RAG** (arXiv:2605.06185, mid-2026) — segments streaming
  videos into semantically coherent events, represents each event as a
  State-Event-State (SES) graph, and merges into a global Event Knowledge
  Graph with dual-store memory (semantic matching + causal-topological
  retrieval).  Bidirectional retrieval strategy: forward (causal prediction)
  and backward (explanatory reasoning).
- **EgoGraph** (arXiv:2602.23709) — training-free dynamic KG for ultra-long
  video with cross-entity temporal dependencies.
- **VideoStir** (arXiv:2604.05418, ACL 2026) — spatio-temporal intent-aware
  retrieval with MLLM-backed intent-relevance scoring.

Architecture::

    EventSegmenter
        │
        ▼
    ┌─────────────────────────────────────┐
    │  Event Knowledge Graph (SESGraph)   │
    │  ┌─────────┐   ┌─────────┐          │
    │  │ State A │──►│ Event   │──► State C│
    │  └─────────┘   └─────────┘          │
    │  State-Event-State Triples           │
    └─────────────────────────────────────┘
        │
        ┌──────┴──────┐
        ▼              ▼
    Semantic        Causal-Topo
    Store           Store
    (BGE-VL/        (graph
     ChromaDB)       adjacency)
        │              │
        └──────┬───────┘
               ▼
    BidirectionalRetriever
        ┌─────────┐
        │ Forward │  "What might happen next?"
        ├─────────┤
        │ Backward│  "What caused this?"
        └─────────┘

Usage::

    from video_analysis.event_rag import EventCausalRAG

    event_rag = EventCausalRAG(config)
    events = event_rag.segment_video(video_index)
    event_rag.build_ses_graph(events)
    # fwd_results = event_rag.retrieve_forward(query, current_event)
    # bwd_results = event_rag.retrieve_backward(query, current_event)
    # results = event_rag.retrieve_bidirectional(query, current_event)
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from video_analysis.config import Config
from video_analysis.models import VideoIndex, SceneInfo

logger = logging.getLogger(__name__)

# ── LLM prompt truncation limits ───────────────────────────────────────
_LLM_TRANSCRIPT_MAX_LENGTH = 4000
_LLM_TRANSCRIPT_TRUNC_SIDE = 2000
_LLM_SCENE_SUMMARY_MAX_LENGTH = 6000
_LLM_SCENE_SUMMARY_TRUNC_SIDE = 3000
_LLM_SEGMENTATION_MAX_TOKENS = 4096

# ── Fallback segmentation ──────────────────────────────────────────────
_DEFAULT_TEMPORAL_GRID_DURATION = 60.0  # seconds per event

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Event:
    """A semantically coherent video event.

    Attributes:
        event_id: Unique identifier (e.g. ``video_id/evt_000``).
        video_id: Source video identifier.
        start_time: Start time in seconds.
        end_time: End time in seconds.
        title: Short human-readable title (e.g. "Introduction", "Q&A session").
        description: Detailed description of what happens in this event.
        transcript: Concatenated transcript text for this event.
        scene_ids: List of scene IDs belonging to this event.
        state_before: Description of the state *before* this event.
        state_after: Description of the state *after* this event.
        entities: Key entities involved (people, objects, locations, concepts).
        action: The core action/transition of this event.
        metadata: Arbitrary metadata dict for extensibility.
        confidence: Confidence score for the event segmentation [0, 1].
    """

    event_id: str
    video_id: str
    start_time: float
    end_time: float
    title: str
    description: str
    transcript: str = ""
    scene_ids: List[int] = field(default_factory=list)
    state_before: str = ""
    state_after: str = ""
    entities: List[str] = field(default_factory=list)
    action: str = ""
    metadata: dict = field(default_factory=dict)
    confidence: float = 1.0


@dataclass
class SESGraph:
    """A collection of State-Event-State triples forming the event graph.

    Nodes:
        - ``state`` nodes: named state descriptions (e.g. "empty room",
          "person standing at podium").
        - ``event`` nodes: action/transition between states.

    Edges:
        - ``state -> event -> state``:  A state transitions *through* an
          event to a new state.
        - ``event -> event (causal)``:  One event causes another.
        - ``event -> event (temporal)``:  One event follows another in time.
    """

    events: Dict[str, Event] = field(default_factory=dict)
    # Adjacency lists: event_id -> list of (target_event_id, relation_type)
    forward_edges: Dict[str, List[Tuple[str, str]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    backward_edges: Dict[str, List[Tuple[str, str]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    # State nodes: state_id -> description
    states: Dict[str, str] = field(default_factory=dict)
    # event_id -> (state_before_id, state_after_id)
    event_state_map: Dict[str, Tuple[str, str]] = field(default_factory=dict)


@dataclass
class CausalPath:
    """A causal reasoning path through the event graph.

    Attributes:
        path: List of event_ids forming the causal chain.
        direction: ``"forward"`` (prediction) or ``"backward"`` (explanation).
        score: Relevance/salience score [0, 1].
        description: Human-readable description of the causal chain.
    """

    path: List[str]
    direction: str
    score: float = 1.0
    description: str = ""


@dataclass
class RetrievalResult:
    """A single retrieval result from the Event-Causal RAG.

    Attributes:
        event: The matched Event.
        score: Relevance score [0, 1].
        path: Optional CausalPath if this result was found via graph traversal.
        retrieval_type: ``"semantic"``, ``"causal_forward"``, ``"causal_backward"``,
            or ``"temporal"``.
    """

    event: Event
    score: float
    path: Optional[CausalPath] = None
    retrieval_type: str = "semantic"


# ---------------------------------------------------------------------------
# Event Segmenter
# ---------------------------------------------------------------------------


class EventSegmenter:
    """Segments a processed video into semantically coherent events.

    Uses a combination of scene boundaries, transcript coherence, and
    optional LLM assistance to identify event boundaries.  Events are
    semantic units — they span multiple scenes if those scenes are part
    of the same coherent action (e.g. a presentation, a conversation, a
    tutorial step).

    Three segmentation strategies (tried in order):

    1. **LLM-based** — uses the configured LLM provider to analyse scene
       descriptions and transcript, producing event boundaries with titles,
       state descriptions, and entity lists.
    2. **Transcript-coherence** — groups consecutive scenes based on
       transcript topic continuity (keyword overlap, speaker consistency).
    3. **Temporal-grid fallback** — merges scenes into fixed-duration
       events (default 60s) when neither LLM nor transcript is available.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        llm_provider=None,
    ):
        self.config = config or Config()
        self._llm = llm_provider

    def segment(
        self,
        video_index: VideoIndex,
        video_id: Optional[str] = None,
    ) -> List[Event]:
        """Segment a processed video into coherent events.

        Args:
            video_index: The processed VideoIndex from the pipeline.
            video_id: Optional video ID override (defaults to VideoIndex.id).

        Returns:
            List of Event objects, ordered chronologically.
        """
        vid = video_id or video_index.video_id or "unknown"
        scenes = video_index.scenes
        transcript = video_index.full_transcript or ""
        transcript_segments = video_index.transcript or []

        if not scenes:
            logger.warning(f"No scenes found for video {vid} — empty event list")
            return []

        # Try LLM-based segmentation first
        if self._llm is not None:
            events = self._segment_with_llm(
                vid, scenes, transcript, transcript_segments
            )
            if events:
                logger.info(
                    f"LLM-based segmentation produced {len(events)} events "
                    f"for video {vid}"
                )
                return events

        # Fallback: transcript-coherence based segmentation
        events = self._segment_by_transcript_coherence(vid, scenes, transcript_segments)
        if events:
            logger.info(
                f"Transcript-coherence segmentation produced {len(events)} events "
                f"for video {vid}"
            )
            return events

        # Final fallback: temporal-grid
        events = self._segment_temporal_grid(vid, scenes)
        logger.info(
            f"Temporal-grid fallback produced {len(events)} events for video {vid}"
        )
        return events

    def _segment_with_llm(
        self,
        video_id: str,
        scenes: List[SceneInfo],
        transcript: str,
        transcript_segments: List,
    ) -> List[Event]:
        """Segment using the LLM provider.

        Builds a compact scene+transcript summary and asks the LLM to
        identify event boundaries with titles, state descriptions, and
        entities.  Falls back gracefully if LLM is unavailable or fails.
        """
        if self._llm is None:
            return []

        # Build a scene summary for the LLM
        scene_lines = []
        for i, scene in enumerate(scenes):
            start = scene.start_time if scene.start_time is not None else 0.0
            summary = scene.summary or ""
            scene_lines.append(f"Scene {i} [t={start:.1f}s]: {summary}")
        scene_summary = "\n".join(scene_lines)
        transcript_preview = transcript
        if len(transcript_preview) > _LLM_TRANSCRIPT_MAX_LENGTH:
            transcript_preview = (
                transcript_preview[:_LLM_TRANSCRIPT_TRUNC_SIDE]
                + "\n...[truncated]..."
                + transcript_preview[-_LLM_TRANSCRIPT_TRUNC_SIDE:]
            )

        # Truncate scene_summary if too long
        if len(scene_summary) > _LLM_SCENE_SUMMARY_MAX_LENGTH:
            scene_summary = (
                scene_summary[:_LLM_SCENE_SUMMARY_TRUNC_SIDE]
                + "\n...[truncated]..."
                + scene_summary[-_LLM_SCENE_SUMMARY_TRUNC_SIDE:]
            )

        prompt = (
            "You are an expert video event segmenter. Given the following scene "
            "descriptions and transcript from a video, identify the semantically "
            "coherent **events** (groups of consecutive scenes that form a single "
            "narrative unit — e.g. a tutorial step, a conversation topic, a "
            "presentation section).\n\n"
            f"Total scenes: {len(scenes)}\n"
            f"Transcript length: {len(transcript)} chars\n\n"
            "### Scene Descriptions\n"
            f"{scene_summary}\n\n"
            "### Transcript Preview\n"
            f"{transcript_preview}\n\n"
            "### Instructions\n"
            "Return a JSON array of event objects, each with:\n"
            "- `start_scene`: index of the first scene (integer)\n"
            "- `end_scene`: index of the last scene (integer, inclusive)\n"
            "- `title`: short human-readable title (2-6 words)\n"
            "- `description`: 1-2 sentence description\n"
            "- `state_before`: what was happening before this event\n"
            "- `state_after`: what changed after this event\n"
            '- `action`: the core action/transition (e.g. "explaining", "demonstrating")\n'
            "- `entities`: list of key entities (people, objects, concepts)\n"
            "\nOnly return valid JSON — no markdown, no explanations."
        )

        try:
            if hasattr(self._llm, "chat"):
                result = self._llm.chat(prompt)
            elif hasattr(self._llm, "generate"):
                result = self._llm.generate(prompt, max_tokens=_LLM_SEGMENTATION_MAX_TOKENS)
            else:
                # Try calling it as a callable
                result = self._llm(prompt)
            if not result:
                return []
            # Parse JSON from the response
            raw = result
            if hasattr(result, "content"):
                raw = result.content
            if isinstance(raw, str):
                # Strip markdown fences if present
                text = raw.strip()
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0].strip()
                raw = json.loads(text)
            raw_events = (
                raw
                if isinstance(raw, list)
                else raw.get("events", raw.get("segments", []))
            )

            events = []
            for i, re in enumerate(raw_events):
                start_idx = int(re.get("start_scene", 0))
                end_idx = int(re.get("end_scene", len(scenes) - 1))
                start_idx = max(0, min(start_idx, len(scenes) - 1))
                end_idx = max(start_idx, min(end_idx, len(scenes) - 1))

                event_scenes = scenes[start_idx : end_idx + 1]
                evt_start = (
                    event_scenes[0].start_time
                    if event_scenes[0].start_time is not None
                    else 0.0
                )
                evt_end = (
                    event_scenes[-1].end_time
                    if event_scenes[-1].end_time is not None
                    else (event_scenes[-1].start_time or 0.0)
                )
                # Gather transcript text for this event's scenes
                evt_scene_ids = list(range(start_idx, end_idx + 1))
                evt_transcript = self._extract_transcript_for_scenes(
                    evt_scene_ids, transcript_segments, scenes
                )

                event = Event(
                    event_id=f"{video_id}/evt_{i:03d}",
                    video_id=video_id,
                    start_time=evt_start,
                    end_time=evt_end,
                    title=re.get("title", f"Event {i}"),
                    description=re.get("description", ""),
                    transcript=evt_transcript,
                    scene_ids=evt_scene_ids,
                    state_before=re.get("state_before", ""),
                    state_after=re.get("state_after", ""),
                    entities=re.get("entities", []),
                    action=re.get("action", ""),
                    confidence=1.0,
                )
                events.append(event)

            return events

        except Exception as e:
            logger.warning(f"LLM-based event segmentation failed: {e}")
            return []

    def _segment_by_transcript_coherence(
        self,
        video_id: str,
        scenes: List[SceneInfo],
        transcript_segments: List,
    ) -> List[Event]:
        """Segment using transcript topic continuity.

        Groups consecutive scenes where transcript segments share keywords
        or where speaker labels are consistent.  Falls back when transcript
        is unavailable.
        """
        if not transcript_segments or not scenes:
            return []

        # Build scene-to-keyword mapping from transcript segments
        scene_keywords: Dict[int, Set[str]] = defaultdict(set)
        for seg in transcript_segments:
            text = ""
            if hasattr(seg, "text"):
                text = seg.text
            elif isinstance(seg, dict):
                text = seg.get("text", "")
            scene_id = -1
            if hasattr(seg, "scene_id"):
                scene_id = seg.scene_id or -1
            elif isinstance(seg, dict):
                scene_id = seg.get("scene_id", -1)

            if scene_id >= 0 and text:
                # Extract top keywords (simple: words longer than 4 chars)
                words = {
                    w.lower().strip(".,!?;:")
                    for w in text.split()
                    if len(w) > 4 and w.isalpha()
                }
                scene_keywords[scene_id].update(words)

        # Group scenes where consecutive scenes share keywords
        events: List[Event] = []
        current_start = 0

        for i in range(1, len(scenes)):
            kw_prev = scene_keywords.get(i - 1, set())
            kw_curr = scene_keywords.get(i, set())
            # Check overlap
            overlap = kw_prev & kw_curr
            # A gap in speaker or <20% keyword overlap = event boundary
            if len(overlap) < max(1, len(kw_prev) * 0.2):
                # Break event
                evt_scenes = scenes[current_start:i]
                if evt_scenes:
                    events.append(
                        self._scenes_to_event(
                            video_id,
                            evt_scenes,
                            list(range(current_start, i)),
                            len(events),
                        )
                    )
                current_start = i

        # Last event
        if current_start < len(scenes):
            evt_scenes = scenes[current_start:]
            events.append(
                self._scenes_to_event(
                    video_id,
                    evt_scenes,
                    list(range(current_start, len(scenes))),
                    len(events),
                )
            )

        return events

    def _segment_temporal_grid(
        self,
        video_id: str,
        scenes: List[SceneInfo],
        event_duration: float = _DEFAULT_TEMPORAL_GRID_DURATION,
    ) -> List[Event]:
        """Fallback: merge scenes into fixed-duration events.

        Args:
            video_id: Video ID.
            scenes: List of SceneInfo objects.
            event_duration: Target event duration in seconds (default 60s).

        Returns:
            List of Event objects.
        """
        if not scenes:
            return []

        events: List[Event] = []
        current_group: List[SceneInfo] = []
        current_indices: List[int] = []
        current_start_time = 0.0

        for i, scene in enumerate(scenes):
            scene_start = scene.start_time if scene.start_time is not None else 0.0
            scene_end = (
                scene.end_time if scene.end_time is not None else scene_start + 5.0
            )

            if not current_group:
                current_start_time = scene_start

            current_group.append(scene)
            current_indices.append(i)

            # Check if we've hit the target duration
            if scene_end - current_start_time >= event_duration:
                events.append(
                    self._scenes_to_event(
                        video_id, current_group, current_indices, len(events)
                    )
                )
                current_group = []
                current_indices = []

        # Remaining scenes
        if current_group:
            events.append(
                self._scenes_to_event(
                    video_id, current_group, current_indices, len(events)
                )
            )

        return events

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _scenes_to_event(
        self,
        video_id: str,
        scenes: List[SceneInfo],
        scene_ids: List[int],
        event_num: int,
    ) -> Event:
        """Convert a group of scenes into a single Event."""
        if not scenes:
            return Event(
                event_id=f"{video_id}/evt_{event_num:03d}",
                video_id=video_id,
                start_time=0.0,
                end_time=0.0,
                title=f"Event {event_num}",
                description="",
            )

        start = scenes[0].start_time if scenes[0].start_time is not None else 0.0
        end = (
            scenes[-1].end_time
            if scenes[-1].end_time is not None
            else (scenes[-1].start_time or start)
        )
        summaries = []
        for s in scenes:
            if s.summary:
                summaries.append(s.summary)

        title = f"Event {event_num}"
        if summaries:
            desc = "; ".join(summaries[:3])
        else:
            desc = f"Scenes {scene_ids[0]}-{scene_ids[-1]}" if scene_ids else ""

        return Event(
            event_id=f"{video_id}/evt_{event_num:03d}",
            video_id=video_id,
            start_time=start,
            end_time=end,
            title=title,
            description=desc,
            scene_ids=scene_ids,
            confidence=0.7,
        )

    def _extract_transcript_for_scenes(
        self,
        scene_ids: List[int],
        transcript_segments: List,
        scenes: List[SceneInfo],
    ) -> str:
        """Extract transcript text belonging to the given scene IDs.

        Uses ``SceneInfo.transcript`` when available (set during pipeline
        processing). Falls back to time-aligned matching against
        TranscriptSegment timestamps.
        """
        parts = []
        # First: use SceneInfo.transcript directly
        for sid in scene_ids:
            if sid < len(scenes):
                scene_transcript = getattr(scenes[sid], "transcript", None) or ""
                if scene_transcript:
                    parts.append(scene_transcript)

        # Second: time-aligned matching as fallback
        if not parts and transcript_segments and scene_ids and scenes:
            scene_start = scenes[scene_ids[0]].start_time
            scene_end = (
                scenes[scene_ids[-1]].end_time
                if scene_ids[-1] < len(scenes)
                else (scenes[scene_ids[-1]].start_time or scene_start + 10.0)
            )
            for seg in transcript_segments:
                seg_start = 0.0
                seg_end = 0.0
                seg_text = ""
                if hasattr(seg, "start"):
                    seg_start = seg.start or 0.0
                    seg_end = seg.end or seg_start
                    seg_text = seg.text or ""
                elif isinstance(seg, dict):
                    seg_start = seg.get("start", 0.0)
                    seg_end = seg.get("end", seg_start)
                    seg_text = seg.get("text", "")
                # Check if this transcript segment falls within the scene timeframe
                if seg_start >= scene_start and seg_end <= scene_end and seg_text:
                    parts.append(seg_text)

        return "\n".join(parts) if parts else ""


# ---------------------------------------------------------------------------
# Dual-Store Memory
# ---------------------------------------------------------------------------


class SemanticStore:
    """Semantic (dense embedding) store for event retrieval.

    Wraps ChromaDB (via VideoRAG) to index event descriptions and retrieve
    by semantic similarity.  When ChromaDB is not available, falls back to
    an in-memory cosine-similarity store using simple TF-IDF-like vectors.
    """

    def __init__(self, config: Optional[Config] = None, rag_instance=None):
        self.config = config or Config()
        self._rag = rag_instance
        self._events: Dict[str, Event] = {}
        self._fallback_store: Dict[str, Dict[str, int]] = {}

    def index_events(self, events: List[Event]) -> int:
        """Index a list of events for semantic retrieval.

        Stores events in ChromaDB (via rag) with event descriptions as
        documents.  Falls back to in-memory storage.

        Args:
            events: List of Event objects to index.

        Returns:
            Number of events successfully indexed.
        """
        self._events.update({e.event_id: e for e in events})

        if self._rag is not None:
            try:
                # Use the RAG's collection to index event descriptions
                chroma = self._rag.collection
                ids = []
                documents = []
                metadatas = []
                for event in events:
                    ids.append(f"evt_{event.event_id}")
                    doc = (
                        f"[Event: {event.title}] {event.description} "
                        f"State before: {event.state_before} "
                        f"State after: {event.state_after} "
                        f"Action: {event.action}"
                    )
                    documents.append(doc)
                    metadatas.append(
                        {
                            "event_id": event.event_id,
                            "video_id": event.video_id,
                            "start_time": event.start_time,
                            "end_time": event.end_time,
                            "type": "event",
                            "title": event.title,
                        }
                    )
                # Upsert into the collection
                if ids:
                    chroma.upsert(
                        ids=ids,
                        documents=documents,
                        metadatas=metadatas,
                    )
                    logger.info(f"Indexed {len(ids)} events in semantic store")
                    return len(ids)
            except Exception as e:
                logger.warning(f"ChromaDB event indexing failed: {e} — using fallback")

        # Fallback: simple keyword term-frequency vector
        for event in events:
            words = (event.description + " " + event.title).lower().split()
            vec = {}
            for w in words:
                vec[w] = vec.get(w, 0) + 1
            self._fallback_store[event.event_id] = vec
        logger.info(f"Indexed {len(events)} events in fallback semantic store")
        return len(events)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Event, float]]:
        """Search for events semantically similar to the query.

        Args:
            query: Natural language query string.
            top_k: Maximum results to return.

        Returns:
            List of (Event, score) tuples sorted by descending score.
        """
        # Primary path: query ChromaDB via the RAG embedding pipeline.
        if self._rag is not None:
            try:
                query_emb = self._rag._get_query_embedding(query)
                chroma = self._rag.collection
                results = chroma.query(
                    query_embeddings=[query_emb],
                    n_results=top_k,
                    where={"type": "event"},
                    include=["metadatas", "distances"],
                )
                if results.get("ids") and results["ids"][0]:
                    scored: List[Tuple[Event, float]] = []
                    for i, _doc_id in enumerate(results["ids"][0]):
                        meta = results["metadatas"][0][i]
                        evt_id = meta.get("event_id", "")
                        if evt_id and evt_id in self._events:
                            dist = results["distances"][0][i]
                            score = 1.0 - dist  # cosine distance → similarity
                            scored.append((self._events[evt_id], float(score)))
                    scored.sort(key=lambda x: x[1], reverse=True)
                    return scored[:top_k]
            except Exception as exc:
                logger.debug("ChromaDB event search failed: %s — using fallback", exc)

        # Fallback: TF-IDF cosine similarity on term frequencies
        if not self._fallback_store:
            return []

        query_words = query.lower().split()
        query_counts: Dict[str, int] = {}
        for w in query_words:
            query_counts[w] = query_counts.get(w, 0) + 1

        scores: List[Tuple[str, float]] = []
        for evt_id, vec in self._fallback_store.items():
            dot = sum(
                query_counts.get(w, 0) * vec.get(w, 0)
                for w in set(list(query_counts.keys()) + list(vec.keys()))
            )
            q_norm = sum(v * v for v in query_counts.values()) ** 0.5
            d_norm = sum(v * v for v in vec.values()) ** 0.5
            if q_norm > 0 and d_norm > 0:
                sim = dot / (q_norm * d_norm)
                scores.append((evt_id, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for evt_id, score in scores[:top_k]:
            if evt_id in self._events:
                results.append((self._events[evt_id], score))
        return results


class CausalTopologicalStore:
    """Causal-topological store for event graph retrieval.

    Maintains an adjacency-based graph of events and supports:
    - Forward traversal: given an event, what events are caused by it?
    - Backward traversal: given an event, what events caused it?
    - K-hop expansion: retrieve events up to K hops away.
    - Causal path enumeration: find all causal chains between events.
    """

    def __init__(self):
        self._graph: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self._reverse_graph: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self._events: Dict[str, Event] = {}

    def update(self, ses_graph: SESGraph) -> None:
        """Update the causal-topological store from an SES graph.

        Args:
            ses_graph: The SESGraph with events, forward_edges, backward_edges.
        """
        self._events.update(ses_graph.events)
        for evt_id, edges in ses_graph.forward_edges.items():
            for target, rel_type in edges:
                self._graph[evt_id].append((target, rel_type))
                self._reverse_graph[target].append((evt_id, rel_type))
        # Also add temporal edges from chronological ordering
        sorted_events = sorted(ses_graph.events.values(), key=lambda e: e.start_time)
        for i in range(len(sorted_events) - 1):
            curr = sorted_events[i].event_id
            next_ = sorted_events[i + 1].event_id
            if curr not in dict(self._graph[curr]):
                self._graph[curr].append((next_, "temporal"))
                self._reverse_graph[next_].append((curr, "temporal"))

    def retrieve_forward(
        self,
        from_event_id: str,
        max_hops: int = 3,
        max_results: int = 10,
    ) -> List[Tuple[Event, float, CausalPath]]:
        """Retrieve events causally *after* the given event (forward/ prediction).

        Performs BFS up to ``max_hops`` on the forward graph.

        Returns:
            List of (Event, score, CausalPath) tuples.
        """
        return self._traverse(
            from_event_id, self._graph, max_hops, max_results, "forward"
        )

    def retrieve_backward(
        self,
        from_event_id: str,
        max_hops: int = 3,
        max_results: int = 10,
    ) -> List[Tuple[Event, float, CausalPath]]:
        """Retrieve events causally *before* the given event (backward/ explanation).

        Performs BFS up to ``max_hops`` on the reverse graph.

        Returns:
            List of (Event, score, CausalPath) tuples.
        """
        return self._traverse(
            from_event_id, self._reverse_graph, max_hops, max_results, "backward"
        )

    def _traverse(
        self,
        start_id: str,
        graph: Dict[str, List[Tuple[str, str]]],
        max_hops: int,
        max_results: int,
        direction: str,
    ) -> List[Tuple[Event, float, CausalPath]]:
        """BFS traversal of the event graph.

        Args:
            start_id: Starting event ID.
            graph: Forward or reverse adjacency list.
            max_hops: Maximum BFS depth.
            max_results: Max results to return.
            direction: ``"forward"`` or ``"backward"``.

        Returns:
            List of (Event, score, CausalPath) tuples.
        """
        results: List[Tuple[Event, float, CausalPath]] = []
        visited: Set[str] = {start_id}
        # BFS queue: (event_id, current_path, hop_count)
        queue: List[Tuple[str, List[str], int]] = [(start_id, [start_id], 0)]

        while queue and len(results) < max_results:
            current, path, hops = queue.pop(0)
            if hops > 0:
                # Found a reachable event
                if current in self._events:
                    score = 1.0 / (hops + 1)  # closer = higher score
                    causal_path = CausalPath(
                        path=path,
                        direction=direction,
                        score=score,
                        description=self._path_description(path, direction),
                    )
                    results.append((self._events[current], score, causal_path))

            if hops < max_hops:
                for neighbor, rel_type in graph.get(current, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor], hops + 1))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:max_results]

    def _path_description(self, path: List[str], direction: str) -> str:
        """Generate a human-readable description of a causal path."""
        event_titles = []
        for evt_id in path:
            evt = self._events.get(evt_id)
            if evt:
                event_titles.append(evt.title)
            else:
                event_titles.append(evt_id)

        if direction == "forward":
            return " → ".join(event_titles)
        else:
            return " ← ".join(reversed(event_titles))


class DualStoreMemory:
    """Dual-store memory combining semantic and causal-topological stores.

    Implements the dual-store memory from Event-Causal RAG (arXiv:2605.06185):
    - **Semantic Store**: dense embedding retrieval from event descriptions.
    - **Causal-Topological Store**: graph adjacency retrieval for causal chaining.

    A query is dispatched to both stores in parallel, and results are fused
    via a weighted score combination.
    """

    def __init__(
        self,
        semantic_store: Optional[SemanticStore] = None,
        causal_store: Optional[CausalTopologicalStore] = None,
        semantic_weight: float = 0.5,
        causal_weight: float = 0.5,
    ):
        self.semantic = semantic_store or SemanticStore()
        self.causal = causal_store or CausalTopologicalStore()
        self.semantic_weight = semantic_weight
        self.causal_weight = causal_weight

    def retrieve(
        self,
        query: str,
        current_event_id: Optional[str] = None,
        top_k: int = 10,
    ) -> List[RetrievalResult]:
        """Bidirectional retrieval from both stores.

        Args:
            query: Natural language query.
            current_event_id: Optional current event for causal retrieval.
            top_k: Maximum results to return.

        Returns:
            Fused and ranked list of RetrievalResult objects.
        """
        results: Dict[str, RetrievalResult] = {}

        # 1. Semantic retrieval
        sem_results = self.semantic.search(query, top_k=top_k)
        for evt, score in sem_results:
            results[evt.event_id] = RetrievalResult(
                event=evt,
                score=score * self.semantic_weight,
                retrieval_type="semantic",
            )

        # 2. Causal retrieval (if we have a current event)
        if current_event_id:
            # Forward
            fwd_results = self.causal.retrieve_forward(
                current_event_id, max_hops=3, max_results=top_k
            )
            for evt, score, path in fwd_results:
                existing = results.get(evt.event_id)
                if existing:
                    existing.score += score * self.causal_weight
                    existing.retrieval_type = "causal_forward"
                    existing.path = path
                else:
                    results[evt.event_id] = RetrievalResult(
                        event=evt,
                        score=score * self.causal_weight,
                        path=path,
                        retrieval_type="causal_forward",
                    )

            # Backward
            bwd_results = self.causal.retrieve_backward(
                current_event_id, max_hops=3, max_results=top_k
            )
            for evt, score, path in bwd_results:
                existing = results.get(evt.event_id)
                if existing:
                    existing.score += score * self.causal_weight
                    existing.retrieval_type = "causal_backward"
                    existing.path = path
                else:
                    results[evt.event_id] = RetrievalResult(
                        event=evt,
                        score=score * self.causal_weight,
                        path=path,
                        retrieval_type="causal_backward",
                    )

        # Sort by score descending
        sorted_results = sorted(results.values(), key=lambda r: r.score, reverse=True)
        return sorted_results[:top_k]


# ---------------------------------------------------------------------------
# Event-Causal RAG — Main Orchestrator
# ---------------------------------------------------------------------------


class EventCausalRAG:
    """Main orchestrator for Event-Causal RAG.

    Wraps EventSegmenter, SESGraph construction, DualStoreMemory, and
    the bidirectional retrieval strategy into a single cohesive module.

    Usage::

        event_rag = EventCausalRAG(config)
        events = event_rag.segment_video(video_index)
        event_rag.build_ses_graph(events)
        event_rag.index_events(events)

        # Bidirectional retrieval
        results = event_rag.retrieve(
            query="What caused the protagonist to change their mind?",
            current_event_id="video1/evt_002",
        )
        for r in results:
            print(f"{r.retrieval_type}: {r.event.title} (score={r.score:.3f})")

        # Causal path analysis
        paths = event_rag.find_causal_paths("video1/evt_000", "video1/evt_005")
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        llm_provider=None,
        rag_instance=None,
    ):
        self.config = config or Config()
        self.segmenter = EventSegmenter(config=self.config, llm_provider=llm_provider)
        self.semantic_store = SemanticStore(
            config=self.config, rag_instance=rag_instance
        )
        self.causal_store = CausalTopologicalStore()
        self.memory = DualStoreMemory(
            semantic_store=self.semantic_store,
            causal_store=self.causal_store,
        )
        self._ses_graph: Optional[SESGraph] = None
        self._events: List[Event] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def segment_video(
        self,
        video_index: VideoIndex,
        video_id: Optional[str] = None,
    ) -> List[Event]:
        """Segment a processed video into events.

        Args:
            video_index: The processed VideoIndex.
            video_id: Optional video ID override.

        Returns:
            List of Event objects.
        """
        self._events = self.segmenter.segment(video_index, video_id=video_id)
        return self._events

    def build_ses_graph(self, events: Optional[List[Event]] = None) -> SESGraph:
        """Build a State-Event-State graph from event list.

        Args:
            events: Event list (uses stored events if None).

        Returns:
            Built SESGraph.
        """
        events = events or self._events
        if not events:
            return SESGraph()

        ses = SESGraph()
        evt_map: Dict[str, Event] = {}

        for event in events:
            evt_map[event.event_id] = event

        ses.events = evt_map

        # Create state nodes
        for event in events:
            if event.state_before:
                state_id = f"state_{event.event_id}_before"
                ses.states[state_id] = event.state_before
            if event.state_after:
                state_id = f"state_{event.event_id}_after"
                ses.states[state_id] = event.state_after
            ses.event_state_map[event.event_id] = (
                f"state_{event.event_id}_before",
                f"state_{event.event_id}_after",
            )

        # Build temporal edges (chronological ordering)
        sorted_events = sorted(events, key=lambda e: e.start_time)
        for i in range(len(sorted_events) - 1):
            curr = sorted_events[i].event_id
            next_ = sorted_events[i + 1].event_id
            # Temporal edge
            ses.forward_edges[curr].append((next_, "temporal"))
            ses.backward_edges[next_].append((curr, "temporal"))

            # If state_after of curr matches state_before of next, it's causal
            if sorted_events[i].state_after and sorted_events[i + 1].state_before:
                # Check if the state transitions align
                sa = sorted_events[i].state_after.lower().strip()
                sb = sorted_events[i + 1].state_before.lower().strip()
                if sa == sb or sa.endswith(sb) or sb.endswith(sa):
                    ses.forward_edges[curr].append((next_, "causal"))
                    ses.backward_edges[next_].append((curr, "causal"))

        self._ses_graph = ses
        return ses

    def index_events(self, events: Optional[List[Event]] = None) -> int:
        """Index events in the dual-store memory.

        Args:
            events: Event list (uses stored events if None).

        Returns:
            Number of events indexed.
        """
        events = events or self._events
        if not events:
            return 0

        # Index in semantic store
        sem_count = self.semantic_store.index_events(events)

        # Update causal store
        if self._ses_graph:
            self.causal_store.update(self._ses_graph)

        return sem_count

    def retrieve(
        self,
        query: str,
        current_event_id: Optional[str] = None,
        top_k: int = 10,
    ) -> List[RetrievalResult]:
        """Bidirectional retrieval from the dual-store memory.

        Args:
            query: Natural language query.
            current_event_id: Optional current event for causal traversal.
            top_k: Max results.

        Returns:
            Fused, ranked retrieval results.
        """
        return self.memory.retrieve(query, current_event_id, top_k=top_k)

    def retrieve_forward(
        self,
        from_event_id: str,
        max_hops: int = 3,
        max_results: int = 10,
    ) -> List[Tuple[Event, float, CausalPath]]:
        """Retrieve causally-forward events (prediction mode).

        Args:
            from_event_id: Starting event ID.
            max_hops: Max BFS depth.
            max_results: Max results.

        Returns:
            List of (Event, score, CausalPath).
        """
        return self.causal_store.retrieve_forward(from_event_id, max_hops, max_results)

    def retrieve_backward(
        self,
        from_event_id: str,
        max_hops: int = 3,
        max_results: int = 10,
    ) -> List[Tuple[Event, float, CausalPath]]:
        """Retrieve causally-backward events (explanation mode).

        Args:
            from_event_id: Starting event ID.
            max_hops: Max BFS depth.
            max_results: Max results.

        Returns:
            List of (Event, score, CausalPath).
        """
        return self.causal_store.retrieve_backward(from_event_id, max_hops, max_results)

    def find_causal_paths(
        self,
        from_event_id: str,
        to_event_id: str,
        max_hops: int = 5,
    ) -> List[CausalPath]:
        """Find all causal paths between two events.

        Uses BFS to enumerate paths from ``from_event_id`` to ``to_event_id``
        up to ``max_hops`` in length.

        Args:
            from_event_id: Start event.
            to_event_id: Target event.
            max_hops: Max path length.

        Returns:
            List of CausalPath objects.
        """
        paths: List[CausalPath] = []
        # BFS for all paths
        queue: List[Tuple[str, List[str]]] = [(from_event_id, [from_event_id])]

        while queue:
            current, path = queue.pop(0)
            if current == to_event_id and len(path) > 1:
                cp = CausalPath(
                    path=path,
                    direction="forward",
                    score=1.0 / len(path),
                    description=self.causal_store._path_description(path, "forward"),
                )
                paths.append(cp)
                continue
            if len(path) >= max_hops:
                continue
            for neighbor, _ in (
                self._ses_graph.forward_edges.get(current, [])
                if self._ses_graph
                else []
            ):
                if neighbor not in path:
                    queue.append((neighbor, path + [neighbor]))

        paths.sort(key=lambda p: p.score, reverse=True)
        return paths

    def get_event_timeline(
        self,
        video_id: Optional[str] = None,
    ) -> List[Event]:
        """Get all events, optionally filtered by video_id, sorted chronologically."""
        events = [e for e in self._events if video_id is None or e.video_id == video_id]
        events.sort(key=lambda e: e.start_time)
        return events

    def to_dict(self) -> dict:
        """Serialize the Event-Causal RAG state to a dictionary."""
        return {
            "events": [asdict(e) for e in self._events],
            "ses_graph": (
                {
                    "event_ids": list(self._ses_graph.events.keys()),
                    "states": self._ses_graph.states,
                    "forward_edges": {
                        k: list(v) for k, v in self._ses_graph.forward_edges.items()
                    },
                }
                if self._ses_graph
                else {}
            ),
        }
