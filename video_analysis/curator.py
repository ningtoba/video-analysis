"""
Autonomous Video Curator — closed-loop video exploration & knowledge curation.

Inspired by InternVideo3's Multimodal Contextual Reasoning (MCR, arXiv:2606.12195)
and VideoAgent (HKUDS, all-in-one agentic framework for understanding).

Whereas the existing VideoUnderstandingAgent is reactive (answers questions),
the VideoCurator is **proactive** — it initiates its own exploration of video
content, maintains a structured knowledge base of findings, decides what to
explore next, and produces comprehensive autonomous reports.

Architecture (closed-loop MCR):
────────────────────────────────
  Observation ──→ Analysis ──→ Memory ──→ Reasoning ──→ Action
     ↑                                                        │
     └─────────────────────── Loop ───────────────────────────┘

  - Observation: Sample frames, query RAG, extract transcript/OCR
  - Analysis: Use Video MLLM to interpret observations
  - Memory: Maintain structured discovery store with entity, scene, concept indexes
  - Reasoning: LLM decides what to explore next based on knowledge gaps
  - Action: Invoke tools (analyze_frames, search_rag, detect_objects, extract_text)

Usage:
    from video_analysis.curator import VideoCurator

    curator = VideoCurator(
        video_path="/path/to/video.mp4",
        rag=rag_instance,
        curiosity_threshold=0.6,  # how aggressively to explore
        max_iterations=20,        # max closed-loop iterations
    )
    report = curator.curate()  # returns VideoCuratorReport
"""

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from video_analysis.config import Config
from video_analysis.models import format_timestamp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class CuratorObservation:
    """A single observation made during video exploration.

    Stores the raw findings from one analysis action.
    """

    observation_id: str  # unique id
    timestamp_seconds: float  # video timestamp in seconds
    observation_type: (
        str  # "scene", "entity", "transcript_segment", "ocr_text", "object", "concept"
    )
    content: str  # text content of the observation
    confidence: float  # 0.0 - 1.0
    source_tool: str  # which tool produced this observation
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CuratorEntity:
    """A persisted entity extracted from the video.

    Entities represent people, objects, locations, or concepts that appear
    across multiple observations and persist in the curator's memory.
    """

    entity_id: str
    name: str
    entity_type: str  # "person", "object", "location", "concept", "event"
    first_seen: float  # timestamp
    last_seen: float  # timestamp
    appearances: int  # how many times observed
    description: str  # accumulated description
    related_timestamps: List[float] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CuratorKnowledge:
    """The curator's accumulated structured knowledge about a video.

    This is the 'shared evolving context' — the memory that persists across
    MCR closed-loop iterations, containing all observations, entities,
    unanswered questions, and exploration metadata.
    """

    video_id: str
    video_path: str
    duration_seconds: float = 0.0
    observations: List[CuratorObservation] = field(default_factory=list)
    entities: Dict[str, CuratorEntity] = field(default_factory=dict)
    exploration_questions: List[str] = field(default_factory=list)
    answered_questions: List[str] = field(default_factory=list)
    knowledge_gaps: List[str] = field(default_factory=list)  # what we don't know yet
    iteration_count: int = 0
    exploration_timeline: List[str] = field(default_factory=list)

    def add_observation(self, obs: CuratorObservation) -> None:
        """Record an observation and update entity memory."""
        self.observations.append(obs)
        self._update_entities(obs)

    def _update_entities(self, obs: CuratorObservation) -> None:
        """Extract or update entities from an observation."""
        # Simple entity extraction from observation content
        # Look for known patterns or people/object references
        if not obs.content:
            return

        content_lower = obs.content.lower()

        # Extract person names (capitalized words after "person", "speaker", etc.)
        person_hints = re.findall(
            r"(?:person|speaker|man|woman|individual) [A-Z][a-z]+ [A-Z][a-z]+",
            obs.content,
        )
        for hint in person_hints:
            name = hint.split(" ", 1)[1] if " " in hint else hint
            if name not in self.entities:
                self.entities[name] = CuratorEntity(
                    entity_id=f"person_{len(self.entities)}",
                    name=name,
                    entity_type="person",
                    first_seen=obs.timestamp_seconds,
                    last_seen=obs.timestamp_seconds,
                    appearances=1,
                    description=obs.content[:300],
                    related_timestamps=[obs.timestamp_seconds],
                )
            else:
                ent = self.entities[name]
                ent.last_seen = obs.timestamp_seconds
                ent.appearances += 1
                ent.related_timestamps.append(obs.timestamp_seconds)

        # Extract object mentions
        object_pattern = re.findall(r"object[s]?:?\s+([a-zA-Z]+(?:,?\s+[a-zA-Z]+)*)", obs.content)
        for obj_text in object_pattern:
            for obj_name in [o.strip() for o in obj_text.split(",") if o.strip()]:
                key = f"obj_{obj_name.lower()}"
                if key not in self.entities:
                    self.entities[key] = CuratorEntity(
                        entity_id=key,
                        name=obj_name,
                        entity_type="object",
                        first_seen=obs.timestamp_seconds,
                        last_seen=obs.timestamp_seconds,
                        appearances=1,
                        description=f"Object '{obj_name}' detected",
                        related_timestamps=[obs.timestamp_seconds],
                    )
                else:
                    ent = self.entities[key]
                    ent.last_seen = obs.timestamp_seconds
                    ent.appearances += 1
                    ent.related_timestamps.append(obs.timestamp_seconds)

    def add_gap(self, gap: str) -> None:
        """Record a knowledge gap (something the curator doesn't know yet)."""
        if gap not in self.knowledge_gaps:
            self.knowledge_gaps.append(gap)

    def add_exploration_question(self, question: str) -> None:
        """Record an exploration question the curator generated."""
        if question not in self.exploration_questions:
            self.exploration_questions.append(question)

    def mark_answered(self, question: str) -> None:
        """Move a question from exploration to answered."""
        if question in self.exploration_questions:
            self.exploration_questions.remove(question)
        if question not in self.answered_questions:
            self.answered_questions.append(question)

    def record_action(self, action_desc: str) -> None:
        """Record an exploration action on the timeline."""
        ts = datetime.now().strftime("%H:%M:%S")
        self.exploration_timeline.append(f"[{ts}] {action_desc}")

    def summary(self) -> Dict[str, Any]:
        """Produce a compact summary of knowledge state."""
        return {
            "video_id": self.video_id,
            "duration_seconds": self.duration_seconds,
            "total_observations": len(self.observations),
            "entities_found": len(self.entities),
            "entity_types": {
                t: sum(1 for e in self.entities.values() if e.entity_type == t)
                for t in set(e.entity_type for e in self.entities.values())
            },
            "exploration_questions": self.exploration_questions,
            "answered_questions": len(self.answered_questions),
            "knowledge_gaps": self.knowledge_gaps,
            "iteration_count": self.iteration_count,
        }


@dataclass
class CuratorReportChunk:
    """One section of the generated curator report."""

    title: str
    content: str
    level: int = 2  # markdown heading level


@dataclass
class VideoCuratorReport:
    """Complete report from an autonomous curation session."""

    video_id: str
    video_path: str
    title: str  # auto-generated title
    overview: str  # one-paragraph overview
    sections: List[CuratorReportChunk] = field(default_factory=list)
    key_entities: Dict[str, CuratorEntity] = field(default_factory=dict)
    key_timeline: List[Dict[str, Any]] = field(default_factory=list)
    exploration_summary: str = ""
    curation_duration_seconds: float = 0.0
    iterations_completed: int = 0
    observations_count: int = 0
    generated_at: str = ""

    def to_markdown(self) -> str:
        """Render the full report as Markdown."""
        lines = [
            f"# {self.title}",
            "",
            self.overview,
            "",
            f"_Curation completed in {self.curation_duration_seconds:.1f}s "
            f"across {self.iterations_completed} iterations "
            f"({self.observations_count} observations)_",
            "",
            "---",
            "",
        ]
        for section in self.sections:
            prefix = "#" * section.level
            lines.append(f"{prefix} {section.title}")
            lines.append("")
            lines.append(section.content)
            lines.append("")

        # Entities section
        if self.key_entities:
            lines.append("## Key Entities Discovered")
            lines.append("")
            for entity in self.key_entities.values():
                if entity.appearances > 1:
                    ts_first = format_timestamp(entity.first_seen)
                    ts_last = format_timestamp(entity.last_seen)
                    lines.append(
                        f"- **{entity.name}** ({entity.entity_type}) — "
                        f"seen {entity.appearances}× from {ts_first} to {ts_last}"
                    )
                    if entity.description:
                        lines.append(f"  _{entity.description[:200]}_")
            lines.append("")

        # Timeline
        if self.key_timeline:
            lines.append("## Key Timeline")
            lines.append("")
            for event in self.key_timeline:
                ts = event.get("timestamp", 0)
                desc = event.get("description", "")
                lines.append(f"- **{format_timestamp(ts)}** — {desc}")
            lines.append("")

        # Exploration trajectory
        if self.exploration_summary:
            lines.append("## Exploration Trajectory")
            lines.append("")
            lines.append(self.exploration_summary)
            lines.append("")

        lines.append("---")
        lines.append(f"_Generated: {self.generated_at}_")

        return "\n".join(lines)

    def to_json(self) -> str:
        """Serialize report to JSON."""
        return json.dumps(
            {
                "title": self.title,
                "overview": self.overview,
                "section_count": len(self.sections),
                "entity_count": len(self.key_entities),
                "timeline_events": len(self.key_timeline),
                "duration_seconds": self.curation_duration_seconds,
                "iterations": self.iterations_completed,
                "observations": self.observations_count,
            },
            indent=2,
        )


# ---------------------------------------------------------------------------
# Curiosity Engine — decides what to explore next
# ---------------------------------------------------------------------------


class CuriosityStrategy:
    """Strategy for deciding what to explore next.

    Implements the 'Reasoning → Action' part of the MCR closed-loop.
    Higher curiosity thresholds lead to more aggressive exploration.
    """

    def __init__(self, config: Config, curiosity_threshold: float = 0.5):
        self.config = config
        self.curiosity_threshold = curiosity_threshold  # 0.0 (conservative) - 1.0 (exploratory)

    def suggest_next_action(
        self,
        knowledge: CuratorKnowledge,
        available_tools: List[str],
    ) -> Tuple[str, Dict[str, Any]]:
        """Based on current knowledge state, suggest what to explore next.

        Args:
            knowledge: Current knowledge state.
            available_tools: List of available tool names.

        Returns:
            Tuple of (action_name, action_params).
        """
        # Strategy rules (simple heuristic-based, no LLM call needed for decisions)

        # 1. If no observations exist, start with broad exploration
        if not knowledge.observations:
            return ("sample_timeline", {"mode": "broad"})

        # 2. If there are unanswered exploration questions, investigate those
        if knowledge.exploration_questions:
            q = knowledge.exploration_questions[0]
            return ("search_topic", {"query": q})

        # 3. If we haven't sampled late portions of the video, explore those
        if knowledge.duration_seconds > 0:
            sampled_timestamps = [o.timestamp_seconds for o in knowledge.observations]
            if sampled_timestamps:
                coverage = max(sampled_timestamps) / knowledge.duration_seconds
                if coverage < 0.7 and self.curiosity_threshold >= 0.3:
                    next_ts = min(
                        knowledge.duration_seconds * 0.75,
                        max(sampled_timestamps) + knowledge.duration_seconds * 0.2,
                    )
                    return ("sample_timestamps", {"timestamps": [next_ts]})

        # 4. Check knowledge gaps and investigate them
        if knowledge.knowledge_gaps:
            gap = knowledge.knowledge_gaps[0]
            # Find timestamps near this gap
            gap_keywords = gap.lower().split()[:3]
            return ("search_topic", {"query": gap})

        # 5. If high curiosity and iterations < cap, do a deep focus on
        #    the most interesting observed region
        if self.curiosity_threshold >= 0.7 and len(knowledge.observations) > 2:
            # Find the entity with the most appearances and dive deeper
            if knowledge.entities:
                top_entity = max(
                    knowledge.entities.values(),
                    key=lambda e: e.appearances,
                )
                if top_entity.appearances > 1:
                    return (
                        "analyze_timestamps",
                        {
                            "timestamps": top_entity.related_timestamps[:3],
                            "focus": top_entity.name,
                        },
                    )

        # 6. Default: generate a curiosity question
        return ("generate_question", {"context": knowledge.summary()})

    def generate_curiosity_questions(self, knowledge: CuratorKnowledge) -> List[str]:
        """Generate questions about what we don't know yet.

        These questions drive the next exploration cycle.
        """
        questions = []
        summary = knowledge.summary()

        # If few entities found, ask about objects/people
        if summary["entities_found"] < 3:
            questions.append("What objects and people are visible in the video?")

        # If observations are sparse, ask about content
        if summary["total_observations"] < 5:
            questions.append("What is the overall content and setting of the video?")

        # Check for temporal gaps
        if knowledge.duration_seconds > 60:
            sampled = [o.timestamp_seconds for o in knowledge.observations]
            if sampled:
                max_ts = max(sampled)
                if max_ts < knowledge.duration_seconds * 0.5:
                    questions.append(
                        f"What happens in the second half of the video "
                        f"(after {format_timestamp(max_ts)})?"
                    )

        # If we've seen entities but don't know their interactions
        entity_count = summary["entities_found"]
        if entity_count >= 2:
            questions.append("How do the discovered entities/people interact?")

        return questions


# ---------------------------------------------------------------------------
# Autonomous Video Curator
# ---------------------------------------------------------------------------


class VideoCurator:
    """Autonomous video explorer with closed-loop MCR architecture.

    The curator runs a closed loop:
      1. OBSERVE — sample frames, query transcript, run analysis
      2. ANALYZE — use Video MLLM to interpret observations
      3. MEMORIZE — store findings in CuratorKnowledge (shared evolving context)
      4. REASON — decide what to explore next (knowledge gaps, curiosity)
      5. ACT — invoke the right tool to gather more information
      6. REPEAT — until max_iterations or saturation

    Attributes:
        video_path: Path to video file.
        rag: Optional VideoRAG instance for retrieval.
        config: Platform config.
        knowledge: The shared evolving context (memory across iterations).
        curiosity_strategy: Strategy for deciding what to explore.
    """

    def __init__(
        self,
        video_path: Optional[str] = None,
        rag: Optional[Any] = None,
        video_id: Optional[str] = None,
        config: Optional[Config] = None,
        curiosity_threshold: float = 0.5,
        max_iterations: int = 15,
        output_dir: Optional[str] = None,
    ):
        self.video_path = Path(video_path) if video_path else None
        self.rag = rag
        self.config = config or Config()
        self.max_iterations = max_iterations
        self.output_dir = (
            Path(output_dir) if output_dir else (Path(self.config.data_dir) / "curations")
        )

        # Resolve video_id
        self.video_id = video_id or (
            self.video_path.stem if self.video_path else f"video_{int(time.time())}"
        )

        # Determine video duration
        self._duration: float = 0.0
        if self.video_path and self.video_path.exists():
            self._duration = self._get_duration(self.video_path)

        # MCR shared evolving context
        self.knowledge = CuratorKnowledge(
            video_id=self.video_id,
            video_path=str(self.video_path) if self.video_path else "",
            duration_seconds=self._duration,
        )

        # Curiosity engine
        self.curiosity_strategy = CuriosityStrategy(
            self.config, curiosity_threshold=curiosity_threshold
        )

        # Lazy-loaded tools (reuse agent tools)
        self._agent_tools: Optional[Any] = None

        # Report metadata
        self._start_time: float = 0.0

    # ------------------------------------------------------------------
    # Duration helper
    # ------------------------------------------------------------------

    @staticmethod
    def _get_duration(video_path: Path) -> float:
        """Get video duration using ffprobe."""
        import subprocess

        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "csv=p=0",
                    str(video_path),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
        except Exception:
            pass
        return 0.0

    # ------------------------------------------------------------------
    # Lazy AgentTools loader
    # ------------------------------------------------------------------

    def _get_agent_tools(self):
        """Lazy-load AgentTools from video_analysis.agent."""
        if self._agent_tools is None:
            try:
                from video_analysis.agent import AgentTools

                self._agent_tools = AgentTools(
                    config=self.config,
                    rag=self.rag,
                    video_path=str(self.video_path) if self.video_path else None,
                    video_id=self.video_id,
                )
            except Exception as exc:
                logger.warning("Could not load AgentTools: %s", exc)
                return None
        return self._agent_tools

    # ------------------------------------------------------------------
    # Core MCR loop
    # ------------------------------------------------------------------

    def curate(self) -> VideoCuratorReport:
        """Run the autonomous curation closed-loop.

        Returns:
            VideoCuratorReport with all findings.
        """
        self._start_time = time.time()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Starting autonomous curation for video=%s duration=%.0fs max_iter=%d",
            self.video_id,
            self._duration,
            self.max_iterations,
        )

        tools = self._get_agent_tools()
        report_sections: List[CuratorReportChunk] = []

        # Phase 1: Broad observation sweep
        self.knowledge.record_action("Starting broad observation sweep")
        self._broad_observation_sweep(tools)
        report_sections.append(
            CuratorReportChunk(
                title="Initial Observation Sweep",
                content=f"Sampled {len([o for o in self.knowledge.observations if o.observation_type == 'scene'])} "
                f"scenes, generating {len(self.knowledge.observations)} total observations. "
                f"Discovered {len(self.knowledge.entities)} entities.",
            )
        )

        # Phase 2: MCR closed-loop — reason → act → observe → memorize
        self.knowledge.record_action("Entering MCR closed-loop exploration")
        for iteration in range(self.max_iterations):
            self.knowledge.iteration_count = iteration + 1

            # Check saturation: if no new entities in last N iterations, slow down
            if self._is_saturated(iteration):
                self.knowledge.record_action(
                    f"Iteration {iteration + 1}: knowledge saturated, stopping early"
                )
                break

            # REASON: decide what to explore next
            action_name, action_params = self.curiosity_strategy.suggest_next_action(
                self.knowledge,
                available_tools=[
                    "analyze_frames",
                    "search_rag",
                    "detect_objects",
                    "extract_text",
                    "search_transcript",
                ],
            )

            self.knowledge.record_action(f"Iteration {iteration + 1}: action={action_name}")

            # ACT + OBSERVE + MEMORIZE
            self._execute_action(action_name, action_params, tools)

        # Phase 3: Generate comprehensive report
        self.knowledge.record_action("Generating final curation report")
        report = self._generate_report(report_sections)

        # Save knowledge state to disk
        self._save_knowledge_state()

        logger.info(
            "Curation complete: %d iterations, %d observations, %d entities",
            self.knowledge.iteration_count,
            len(self.knowledge.observations),
            len(self.knowledge.entities),
        )

        return report

    def _is_saturated(self, iteration: int) -> bool:
        """Check if exploration is saturated (no new knowledge added)."""
        if iteration < 3:
            return False  # always run at least 3 iterations

        # Check if no new entities in last 5 iterations
        recent_entity_count = len(self.knowledge.entities)
        if hasattr(self, "_prev_entity_count"):
            if recent_entity_count == self._prev_entity_count and iteration >= 8:
                # Check if any new observations had meaningful content
                recent_obs = [
                    o
                    for o in self.knowledge.observations
                    if o.timestamp_seconds > getattr(self, "_checkpoint_ts", 0)
                ]
                meaningful = [o for o in recent_obs if len(o.content) > 50 and o.confidence > 0.3]
                if len(meaningful) < 2:
                    return True
        self._prev_entity_count = recent_entity_count
        self._checkpoint_ts = time.time()
        return False

    def _broad_observation_sweep(self, tools) -> None:
        """Phase 1: Sample the video broadly to establish baseline knowledge."""
        if not tools:
            logger.warning("No tools available — skipping observation sweep")
            self.knowledge.record_action("No tools available (agent tools not loaded)")
            return

        if not self.video_path or not self.video_path.exists():
            # RAG-based sweep
            self._rag_sweep(tools)
            return

        import cv2

        cap = cv2.VideoCapture(str(self.video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        if total_frames <= 0:
            self._rag_sweep(tools)
            return

        # Sample frames at strategic points: early, mid, late + interests
        duration = self._duration or (total_frames / max(fps, 1))
        sample_points = [
            0.05 * duration,  # near start
            0.25 * duration,  # early-mid
            0.50 * duration,  # midpoint
            0.75 * duration,  # mid-late
            0.95 * duration,  # near end
        ]

        # Also sample at scene boundaries if available
        if self.rag:
            try:
                scene_chunks = self.rag.retrieve(query="scene boundary transition", top_k=5)
                for c in scene_chunks:
                    if c.timestamp and 0 < c.timestamp < duration:
                        sample_points.append(c.timestamp)
            except Exception:
                pass

        # Deduplicate and sort
        sample_points = sorted(set(max(0, min(duration - 1, p)) for p in sample_points))[:10]

        self.knowledge.record_action(
            f"Broad sweep: sampling {len(sample_points)} points across {duration:.0f}s video"
        )

        # Analyze frames at sample points
        try:
            frame_result = tools.analyze_frames(
                sample_points,
                prompt=(
                    "Describe this video scene in detail. Include: "
                    "visible people, objects, setting/location, "
                    "text/overlays, lighting, colors, and any notable activity. "
                    "If people are visible, describe their appearance and actions."
                ),
            )
            if frame_result and frame_result.success:
                for i, ts in enumerate(sample_points):
                    # Take segments from the frame result
                    obs_content = frame_result.data
                    if obs_content:
                        obs = CuratorObservation(
                            observation_id=f"sweep_{i}",
                            timestamp_seconds=ts,
                            observation_type="scene",
                            content=obs_content[:500],
                            confidence=0.8,
                            source_tool="analyze_frames",
                            metadata={"sample_index": i, "point_type": "sweep"},
                        )
                        self.knowledge.add_observation(obs)
        except Exception as exc:
            logger.warning("Frame analysis sweep failed: %s", exc)
            self.knowledge.record_action(f"Frame analysis failed: {exc}")

        # Also get RAG context if available
        if self.rag:
            try:
                summary_result = tools.summarize_video(num_frames=8)
                if summary_result and summary_result.success:
                    summary_obs = CuratorObservation(
                        observation_id="rag_summary_0",
                        timestamp_seconds=0.0,
                        observation_type="scene",
                        content=summary_result.data[:1000],
                        confidence=0.7,
                        source_tool="summarize_video",
                        metadata={"source": "rag"},
                    )
                    self.knowledge.add_observation(summary_obs)
            except Exception:
                pass

    def _rag_sweep(self, tools) -> None:
        """Fallback: use RAG to establish baseline when video file unavailable."""
        if not self.rag:
            self.knowledge.record_action("No video file and no RAG — cannot explore")
            return

        try:
            rag_result = tools.search_rag("overview of video content", top_k=10)
            if rag_result and rag_result.success:
                obs = CuratorObservation(
                    observation_id="rag_base_0",
                    timestamp_seconds=0.0,
                    observation_type="scene",
                    content=rag_result.data[:1000],
                    confidence=0.6,
                    source_tool="search_rag",
                )
                self.knowledge.add_observation(obs)
        except Exception as exc:
            logger.warning("RAG sweep failed: %s", exc)

    def _execute_action(
        self,
        action_name: str,
        action_params: Dict[str, Any],
        tools,
    ) -> None:
        """Execute a single MCR action and memorize the result.

        Args:
            action_name: The action to execute.
            action_params: Parameters for the action.
            tools: The AgentTools instance.
        """
        if not tools:
            return

        try:
            if action_name == "sample_timestamps":
                timestamps = action_params.get("timestamps", [])
                if timestamps:
                    result = tools.analyze_frames(
                        timestamps,
                        prompt="Describe what's happening in this video frame in detail.",
                    )
                    if result and result.success:
                        for i, ts in enumerate(timestamps):
                            obs = CuratorObservation(
                                observation_id=f"ts_analysis_{len(self.knowledge.observations)}",
                                timestamp_seconds=ts,
                                observation_type="scene",
                                content=result.data[:500],
                                confidence=0.7,
                                source_tool="analyze_frames",
                                metadata={"timestamps": timestamps},
                            )
                            self.knowledge.add_observation(obs)

            elif action_name == "search_topic":
                query = action_params.get("query", "")
                if query:
                    result = tools.search_rag(query, top_k=5)
                    if result and result.success:
                        obs = CuratorObservation(
                            observation_id=f"search_{len(self.knowledge.observations)}",
                            timestamp_seconds=0.0,
                            observation_type="scene",
                            content=result.data[:800],
                            confidence=0.6,
                            source_tool="search_rag",
                            metadata={"query": query},
                        )
                        self.knowledge.add_observation(obs)
                        self.knowledge.mark_answered(query)

            elif action_name == "analyze_timestamps":
                timestamps = action_params.get("timestamps", [])
                focus = action_params.get("focus", "")
                if timestamps:
                    prompt = (
                        f"Focus on '{focus}' — describe this element in detail. "
                        f"What is it doing? What are its characteristics?"
                        if focus
                        else "Describe this scene in detail."
                    )
                    result = tools.analyze_frames(timestamps, prompt=prompt)
                    if result and result.success:
                        for i, ts in enumerate(timestamps):
                            obs = CuratorObservation(
                                observation_id=f"focus_{len(self.knowledge.observations)}",
                                timestamp_seconds=ts,
                                observation_type="entity",
                                content=result.data[:500],
                                confidence=0.8,
                                source_tool="analyze_frames",
                                metadata={"focus": focus, "timestamp": ts},
                            )
                            self.knowledge.add_observation(obs)

            elif action_name == "sample_timeline":
                mode = action_params.get("mode", "broad")
                if mode == "broad" and self._duration > 0:
                    # Sample evenly across all thirds
                    timestamps = [
                        self._duration * 0.1,
                        self._duration * 0.3,
                        self._duration * 0.5,
                        self._duration * 0.7,
                        self._duration * 0.9,
                    ]
                    result = tools.analyze_frames(
                        timestamps,
                        prompt="Describe this video scene in detail. "
                        "What is happening? Who or what is visible?",
                    )
                    if result and result.success:
                        for i, ts in enumerate(timestamps):
                            obs = CuratorObservation(
                                observation_id=f"timeline_{i}",
                                timestamp_seconds=ts,
                                observation_type="scene",
                                content=result.data[:500],
                                confidence=0.7,
                                source_tool="analyze_frames",
                                metadata={"mode": mode, "sample_index": i},
                            )
                            self.knowledge.add_observation(obs)

            elif action_name == "generate_question":
                # Generate curiosity questions and log them
                questions = self.curiosity_strategy.generate_curiosity_questions(self.knowledge)
                for q in questions:
                    self.knowledge.add_exploration_question(q)

            elif action_name == "detect_objects":
                # Sample object detection at strategic points
                if self._duration > 0:
                    timestamps = [
                        self._duration * 0.2,
                        self._duration * 0.5,
                        self._duration * 0.8,
                    ]
                    for ts in timestamps:
                        try:
                            result = tools.detect_objects(ts)
                            if result and result.success and "No objects" not in result.data:
                                obs = CuratorObservation(
                                    observation_id=f"objects_{len(self.knowledge.observations)}",
                                    timestamp_seconds=ts,
                                    observation_type="object",
                                    content=result.data[:300],
                                    confidence=0.9,
                                    source_tool="detect_objects",
                                    metadata={"timestamp": ts},
                                )
                                self.knowledge.add_observation(obs)
                        except Exception:
                            pass

        except Exception as exc:
            logger.debug("Action '%s' failed: %s", action_name, exc)
            self.knowledge.record_action(f"Action '{action_name}' failed: {exc}")

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def _generate_report(self, initial_sections: List[CuratorReportChunk]) -> VideoCuratorReport:
        """Generate the final curation report from all accumulated knowledge."""
        elapsed = time.time() - self._start_time

        # Generate overview
        total_scenes = len(
            [o for o in self.knowledge.observations if o.observation_type == "scene"]
        )
        total_objects = len(
            [o for o in self.knowledge.observations if o.observation_type == "object"]
        )
        entity_summary = (
            ", ".join(
                f"{e.name} ({e.entity_type})" for e in list(self.knowledge.entities.values())[:10]
            )
            if self.knowledge.entities
            else "none discovered"
        )

        overview = (
            f"Autonomous analysis of **{self.video_path.name if self.video_path else self.video_id}** "
            f"({format_timestamp(self._duration)} duration). "
            f"The curator completed {self.knowledge.iteration_count} closed-loop exploration iterations, "
            f"producing {len(self.knowledge.observations)} observations across {total_scenes} scenes, "
            f"{total_objects} object detections, and "
            f"{len(self.knowledge.entities)} discovered entities. "
            f"Key entities: {entity_summary}."
        )

        # Build content sections
        all_sections = list(initial_sections)
        self._add_entity_sections(all_sections)
        self._add_observation_sections(all_sections)
        self._add_exploration_section(all_sections)

        # Build timeline
        timeline = []
        observed_timestamps = sorted(
            set(o.timestamp_seconds for o in self.knowledge.observations if o.timestamp_seconds > 0)
        )
        for ts in observed_timestamps:
            obs_at_ts = [
                o for o in self.knowledge.observations if abs(o.timestamp_seconds - ts) < 1.0
            ]
            descriptions = [o.content[:100] for o in obs_at_ts if len(o.content) > 20]
            if descriptions:
                timeline.append(
                    {
                        "timestamp": ts,
                        "description": descriptions[0],
                        "observation_count": len(obs_at_ts),
                    }
                )

        # Exploration trajectory
        exploration_lines = (
            "\n".join(f"- {action}" for action in self.knowledge.exploration_timeline)
            if self.knowledge.exploration_timeline
            else "No exploration trajectory recorded."
        )

        report = VideoCuratorReport(
            video_id=self.video_id,
            video_path=str(self.video_path) if self.video_path else "",
            title=f"Autonomous Curation Report: {self.video_path.name if self.video_path else self.video_id}",
            overview=overview,
            sections=all_sections,
            key_entities=dict(self.knowledge.entities),
            key_timeline=timeline[:20],
            exploration_summary=exploration_lines,
            curation_duration_seconds=elapsed,
            iterations_completed=self.knowledge.iteration_count,
            observations_count=len(self.knowledge.observations),
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        return report

    def _add_entity_sections(self, sections: List[CuratorReportChunk]) -> None:
        """Add entity-based sections to the report."""
        entity_types = {}
        for entity in self.knowledge.entities.values():
            entity_types.setdefault(entity.entity_type, []).append(entity)

        for etype, entities in entity_types.items():
            if len(entities) > 15:
                entities = entities[:15]  # cap at 15 per type

            entity_lines = []
            for ent in sorted(entities, key=lambda e: e.appearances, reverse=True):
                ts_first = format_timestamp(ent.first_seen)
                ts_last = format_timestamp(ent.last_seen)
                desc = ent.description[:100].replace("\n", " ") if ent.description else ""
                entity_lines.append(
                    f"- **{ent.name}** — seen {ent.appearances}× ({ts_first} → {ts_last}) {desc}"
                )

            if entity_lines:
                sections.append(
                    CuratorReportChunk(
                        title=f"Discovered {etype.title()}s",
                        content="\n".join(entity_lines),
                    )
                )

    def _add_observation_sections(self, sections: List[CuratorReportChunk]) -> None:
        """Add observation-based content sections."""
        # Group observations by proximity (within 30s of each other)
        if not self.knowledge.observations:
            return

        # Sort by timestamp
        sorted_obs = sorted(
            self.knowledge.observations,
            key=lambda o: o.timestamp_seconds,
        )

        # Cluster into time regions
        clusters: List[List[CuratorObservation]] = []
        current_cluster: List[CuratorObservation] = []
        last_ts = -100.0

        for obs in sorted_obs:
            if abs(obs.timestamp_seconds - last_ts) > 30.0 and current_cluster:
                clusters.append(current_cluster)
                current_cluster = []
            current_cluster.append(obs)
            last_ts = obs.timestamp_seconds

        if current_cluster:
            clusters.append(current_cluster)

        # Create a section per cluster
        for i, cluster in enumerate(clusters):
            if not cluster:
                continue
            start_ts = cluster[0].timestamp_seconds
            end_ts = cluster[-1].timestamp_seconds
            if start_ts == end_ts:
                title = f"At {format_timestamp(start_ts)}"
            else:
                title = f"Region {i + 1}: {format_timestamp(start_ts)} — {format_timestamp(end_ts)}"

            content_lines = []
            for obs in cluster:
                snippet = obs.content[:200].replace("\n", " ")
                if snippet:
                    source = obs.source_tool
                    ctype = obs.observation_type
                    content_lines.append(
                        f"- [{obs.timestamp_seconds:.0f}s, {ctype} via {source}] {snippet}"
                    )

            if content_lines:
                sections.append(
                    CuratorReportChunk(
                        title=title,
                        content="\n".join(content_lines[:10]),  # cap
                    )
                )

    def _add_exploration_section(self, sections: List[CuratorReportChunk]) -> None:
        """Add the exploration trajectory section."""
        if not self.knowledge.exploration_timeline:
            return

        content = "\n".join(f"- {action}" for action in self.knowledge.exploration_timeline)
        sections.append(
            CuratorReportChunk(
                title="Exploration Trajectory",
                content=content,
            )
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_knowledge_state(self) -> Optional[Path]:
        """Save the knowledge state to disk for cross-session persistence."""
        try:
            out_path = self.output_dir / f"{self.video_id}_knowledge.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            state = {
                "video_id": self.knowledge.video_id,
                "duration_seconds": self.knowledge.duration_seconds,
                "observations": [o.to_dict() for o in self.knowledge.observations],
                "entities": {k: v.to_dict() for k, v in self.knowledge.entities.items()},
                "exploration_questions": self.knowledge.exploration_questions,
                "answered_questions": self.knowledge.answered_questions,
                "knowledge_gaps": self.knowledge.knowledge_gaps,
                "iteration_count": self.knowledge.iteration_count,
                "exploration_timeline": self.knowledge.exploration_timeline,
                "saved_at": datetime.now().isoformat(),
            }
            with open(out_path, "w") as f:
                json.dump(state, f, indent=2)
            return out_path
        except Exception as exc:
            logger.warning("Failed to save knowledge state: %s", exc)
            return None

    def load_knowledge_state(self, path: Optional[Path] = None) -> bool:
        """Load a previously saved knowledge state.

        Args:
            path: Path to saved state JSON. If None, auto-discover.

        Returns:
            True if loaded successfully.
        """
        if not path:
            auto_path = self.output_dir / f"{self.video_id}_knowledge.json"
            if auto_path.exists():
                path = auto_path

        if not path or not path.exists():
            logger.warning("No saved knowledge state found at %s", path)
            return False

        try:
            with open(path) as f:
                state = json.load(f)

            self.knowledge = CuratorKnowledge(
                video_id=state.get("video_id", self.video_id),
                video_path=state.get("video_path", ""),
                duration_seconds=state.get("duration_seconds", 0.0),
                observations=[CuratorObservation(**o) for o in state.get("observations", [])],
                entities={k: CuratorEntity(**v) for k, v in state.get("entities", {}).items()},
                exploration_questions=state.get("exploration_questions", []),
                answered_questions=state.get("answered_questions", []),
                knowledge_gaps=state.get("knowledge_gaps", []),
                iteration_count=state.get("iteration_count", 0),
                exploration_timeline=state.get("exploration_timeline", []),
            )
            logger.info(
                "Loaded knowledge state from %s (%d observations, %d entities)",
                path,
                len(self.knowledge.observations),
                len(self.knowledge.entities),
            )
            return True
        except Exception as exc:
            logger.warning("Failed to load knowledge state: %s", exc)
            return False


# ---------------------------------------------------------------------------
# Convenience: run curation from CLI
# ---------------------------------------------------------------------------


def run_curation(
    video_path: Optional[str] = None,
    video_id: Optional[str] = None,
    rag=None,
    curiosity: float = 0.5,
    max_iterations: int = 15,
    output_dir: Optional[str] = None,
) -> VideoCuratorReport:
    """Run autonomous curation and return the report.

    Args:
        video_path: Path to video file (or None for RAG-only).
        video_id: Video identifier.
        rag: Optional VideoRAG instance.
        curiosity: Curiosity threshold (0.0-1.0).
        max_iterations: Maximum MCR loop iterations.
        output_dir: Output directory for report and knowledge state.

    Returns:
        VideoCuratorReport.
    """
    config = Config()
    curator = VideoCurator(
        video_path=video_path,
        rag=rag,
        video_id=video_id,
        config=config,
        curiosity_threshold=curiosity,
        max_iterations=max_iterations,
        output_dir=output_dir,
    )
    report = curator.curate()

    # Save report
    out_dir = Path(output_dir) if output_dir else (Path(config.data_dir) / "curations")
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{report.video_id}_report.md"
    json_path = out_dir / f"{report.video_id}_report.json"

    try:
        md_path.write_text(report.to_markdown())
        json_path.write_text(report.to_json())
        logger.info("Report saved to %s", md_path)
    except Exception as exc:
        logger.warning("Failed to save report: %s", exc)

    return report
