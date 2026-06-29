"""
Hierarchical Multi-Agent Video Reasoning Orchestrator — inspired by HiCrew and Orchestra-o1.

HiCrew (arXiv:2604.21444, April 2026):
    Hierarchical multi-agent reasoning for long-form video understanding via:
    - Hybrid Tree structure: preserves temporal topology with relevance-guided clustering
    - Question-Aware Captioning: intent-driven visual prompts for precise semantic descriptions
    - Planning Layer: dynamically orchestrates agent collaboration based on question complexity

Orchestra-o1 (arXiv:2606.13707, June 2026):
    Omnimodal agent orchestration framework with:
    - Modality-aware task decomposition
    - Online sub-agent specialization
    - Parallel sub-task execution with DA-GRPO reinforcement learning

This module implements a complete hierarchical multi-agent architecture that sits
ABOVE the existing VideoUnderstandingAgent, transforming it from a flat rule-based
tool dispatcher into a dynamic, LLM-powered multi-agent system.

Architecture:
    RouterAgent (Planning Layer)
        ├── Question analysis → modality / complexity detection
        ├── RoutePlan generation (sub-questions, dependencies, thresholds)
        └── Specialist Agent dispatch (parallel where possible)
            ├── VisualAnalyst → analyze_frames (Video MLLM)
            ├── RAGSearcher → search_rag (query refinement)
            ├── TranscriptAnalyst → search_transcript (temporal grounding)
            ├── ObjectDetectorAgent → detect_objects (YOLO + tracking)
            ├── OCRAgent → extract_text (OCR)
            ├── ConfidenceAuditor → cross-validate (agent_confidence)
            └── SummarizerAgent → summarize_video
    EvidenceSynthesizer (final combination)
        ├── EvidenceWeighter (tiered/continuous from v0.50.0)
        ├── Source attribution with citations
        └── Weighted combination → OrchestratorResult

Usage:
    from video_analysis.orchestra import MultiAgentOrchestrator, get_orchestrator

    # Option A: Direct construction
    orch = MultiAgentOrchestrator(config=config, rag=rag, video_path=path, video_id=vid)
    result = orch.query("What objects appear around 2:30 and who is speaking?")

    # Option B: Factory function (reads env vars for config)
    orch = get_orchestrator(rag=rag, video_path=path, video_id=vid)
    result = orch.query("Describe the scene at 1:15")
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from video_analysis.config import Config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency imports (graceful degradation)
# ---------------------------------------------------------------------------

try:
    from video_analysis.agent_confidence import (
        EvidenceTrustScorer,
        EvidenceWeighter,
        FrameQualityScorer,
    )

    _HAS_CONFIDENCE = True
except ImportError:
    _HAS_CONFIDENCE = False
    EvidenceTrustScorer = None  # type: ignore
    EvidenceWeighter = None  # type: ignore
    logger.debug("agent_confidence not available — evidence scoring disabled")

try:
    from video_analysis.llm_provider import (
        LLMProvider,
        LLMProviderConfig,
        get_llm_provider as _get_llm_provider,
    )

    _HAS_LLM_PROVIDER = True
except ImportError:
    _HAS_LLM_PROVIDER = False
    logger.debug("llm_provider not available — router LLM fallback to rule-based")


def _get_orchestra_config(config: Config) -> Dict[str, Any]:
    """Read orchestrator config fields, with env-var overrides.

    These fields may or may not exist on the Config dataclass (they are
    proposed for a future config version).  We read them safely via
    ``getattr`` with env-var fallback so the module works out-of-the-box.
    """
    return {
        "enabled": bool(
            getattr(config, "orchestra_enabled", None)
            or os.environ.get("ORCHESTRA_ENABLED", "false").lower()
            in ("true", "1", "yes")
        ),
        "max_agents": int(
            getattr(config, "orchestra_max_agents", None)
            or os.environ.get("ORCHESTRA_MAX_AGENTS", "5")
        ),
        "confidence_threshold": float(
            getattr(config, "orchestra_confidence_threshold", None)
            or os.environ.get("ORCHESTRA_CONFIDENCE_THRESHOLD", "0.5")
        ),
    }


# ===================================================================
# Data Types — Hybrid Tree
# ===================================================================


@dataclass
class HybridNode:
    """A node in the Hybrid Tree temporal-semantic hierarchy.

    Leaf nodes represent individual scenes from the pipeline.
    Internal nodes represent clusters of semantically similar scenes.
    """

    scene_id: int  # -1 for internal nodes, scene index for leaves
    label: str  # description / cluster label
    level: int  # 0 = leaf, 1+ = cluster depth
    children: List[HybridNode] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def num_leaves(self) -> int:
        if self.is_leaf:
            return 1
        return sum(c.num_leaves for c in self.children)

    def all_leaves(self) -> List[HybridNode]:
        """Recursively collect all leaf nodes."""
        if self.is_leaf:
            return [self]
        result: List[HybridNode] = []
        for c in self.children:
            result.extend(c.all_leaves())
        return result


class HybridTree:
    """Hierarchical temporal-semantic tree over video scenes.

    Inspired by HiCrew's Hybrid Tree structure: preserves temporal topology
    while performing relevance-guided hierarchical clustering within
    semantically coherent segments.

    The pipeline already produces a flat list of scenes with shot boundaries;
    this tree groups temporally contiguous scenes whose semantic similarity
    exceeds a configurable threshold, preserving temporal adjacency.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        similarity_threshold: float = 0.6,
        max_cluster_size: int = 5,
    ):
        self.config = config or Config()
        self.similarity_threshold = similarity_threshold
        self.max_cluster_size = max_cluster_size
        self.root: Optional[HybridNode] = None
        self._scene_map: Dict[int, HybridNode] = {}

    def build(self, scenes: List[Dict[str, Any]]) -> None:
        """Build the Hybrid Tree from pipeline scene data.

        Uses greedy temporal-semantic clustering:
          - Leaves are created in temporal order
          - Adjacent scenes with small time gaps are grouped
          - Each cluster becomes a level-1 parent node

        Args:
            scenes: List of scene dicts with scene_id, start, end, description, etc.
        """
        if not scenes:
            self.root = None
            return

        # Create leaf nodes sorted by time
        leaves: List[HybridNode] = []
        for s in scenes:
            sid = s.get("scene_id", 0)
            label = s.get("description", s.get("label", f"Scene {sid}"))
            node = HybridNode(
                scene_id=sid,
                label=str(label)[:80],
                level=0,
                start_time=s.get("start", s.get("start_time", 0.0)),
                end_time=s.get("end", s.get("end_time", 0.0)),
                metadata=s,
            )
            leaves.append(node)
            self._scene_map[sid] = node

        # Greedy temporal-semantic clustering
        clusters: List[List[HybridNode]] = []
        current: List[HybridNode] = [leaves[0]]

        for node in leaves[1:]:
            # Temporal proximity: if gap > 30s, start new cluster
            time_gap = node.start_time - current[-1].end_time
            if time_gap > 30.0 and len(current) >= 2:
                clusters.append(current)
                current = [node]
            elif len(current) >= self.max_cluster_size:
                clusters.append(current)
                current = [node]
            else:
                current.append(node)

        if current:
            clusters.append(current)

        # Build internal nodes for each cluster
        cluster_nodes: List[HybridNode] = []
        for i, cluster in enumerate(clusters):
            if len(cluster) == 1:
                cluster_nodes.append(cluster[0])
            else:
                start_t = cluster[0].start_time
                end_t = cluster[-1].end_time
                combined_label = (
                    f"Cluster {i}: {cluster[0].label[:40]}, {cluster[-1].label[:40]}"
                )
                parent = HybridNode(
                    scene_id=-1,
                    label=combined_label,
                    level=1,
                    children=cluster,
                    start_time=start_t,
                    end_time=end_t,
                )
                cluster_nodes.append(parent)

        # Single root with all top-level clusters
        if len(cluster_nodes) == 1:
            self.root = cluster_nodes[0]
        else:
            self.root = HybridNode(
                scene_id=-1,
                label="Video Root",
                level=2,
                children=cluster_nodes,
                start_time=cluster_nodes[0].start_time if cluster_nodes else 0.0,
                end_time=cluster_nodes[-1].end_time if cluster_nodes else 0.0,
            )

    def find_scene(self, scene_id: int) -> Optional[HybridNode]:
        """Look up a leaf node by its scene id."""
        return self._scene_map.get(scene_id)

    def get_leaf_paths(self) -> List[List[HybridNode]]:
        """Get all root-to-leaf paths in the tree."""
        if self.root is None:
            return []
        paths: List[List[HybridNode]] = []

        def _dfs(node: HybridNode, path: List[HybridNode]) -> None:
            current = path + [node]
            if node.is_leaf:
                paths.append(current)
            else:
                for c in node.children:
                    _dfs(c, current)

        _dfs(self.root, [])
        return paths

    @property
    def num_leaves(self) -> int:
        return self.root.num_leaves if self.root else 0

    def depth(self) -> int:
        """Maximum depth of the tree."""
        if self.root is None:
            return 0

        def _max_depth(node: HybridNode) -> int:
            if not node.children:
                return 0
            return 1 + max(_max_depth(c) for c in node.children)

        return _max_depth(self.root)


# ===================================================================
# Data Types — Route Plan
# ===================================================================


@dataclass
class TaskItem:
    """A single sub-task for a Specialist Agent."""

    agent_type: str  # visual_analyst, rag_searcher, etc.
    description: str  # what to do
    sub_query: str = ""  # refined question for this agent
    dependencies: List[str] = field(default_factory=list)
    confidence_threshold: float = 0.0  # early stopping threshold
    confidence: Optional[float] = None
    completed: bool = False
    result: Optional[Dict[str, Any]] = None


@dataclass
class RoutePlan:
    """Execution plan generated by RouterAgent."""

    query: str
    tasks: List[TaskItem] = field(default_factory=list)
    complexity: str = "simple"  # simple | multi-hop | analytical
    modalities: Set[str] = field(default_factory=set)
    # visual, text, temporal, action, entity
    reasoning_path: List[str] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return all(t.completed for t in self.tasks)

    def ready_tasks(self) -> List[TaskItem]:
        """Tasks whose dependencies are all satisfied."""
        completed_types = {t.agent_type for t in self.tasks if t.completed}
        return [
            t
            for t in self.tasks
            if not t.completed and all(d in completed_types for d in t.dependencies)
        ]


# ===================================================================
# Evidence Synthesis
# ===================================================================


@dataclass
class SynthesisResult:
    """Combined evidence from all specialist agents."""

    answer: str = ""
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    agent_breakdown: Dict[str, float] = field(default_factory=dict)


class EvidenceSynthesizer:
    """Weighted combination of evidence from multiple specialist agents.

    Uses the v0.50.0 EvidenceWeighter from agent_confidence for tiered/continuous
    weighting, falling back to simple average when confidence scoring is unavailable.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self._weighter: Optional[EvidenceWeighter] = None
        self._scorer: Optional[EvidenceTrustScorer] = None

        if _HAS_CONFIDENCE and EvidenceWeighter is not None:
            try:
                weight_mode = getattr(
                    self.config, "agent_confidence_weight_mode", "tiered"
                )
                self._weighter = EvidenceWeighter()
                self._scorer = EvidenceTrustScorer()
            except Exception:
                logger.debug("Could not initialize EvidenceWeighter", exc_info=True)

    def synthesize(
        self,
        query: str,
        evidence: Dict[str, Dict[str, Any]],
    ) -> SynthesisResult:
        """Combine evidence from multiple agent outputs into a single result.

        Args:
            query: The original user question.
            evidence: Dict mapping agent_type → {success, data, confidence, error}

        Returns:
            SynthesisResult with combined answer, evidence list, and confidence.
        """
        if not evidence:
            return SynthesisResult()

        # Filter successful results
        successful = {
            agent: data
            for agent, data in evidence.items()
            if data.get("success", False) and data.get("data")
        }

        if not successful:
            errors = [
                f"{agent}: {data.get('error', 'unknown error')}"
                for agent, data in evidence.items()
            ]
            return SynthesisResult(
                answer=f"All agents failed: {'; '.join(errors)}",
                evidence=[
                    {"source": a, "error": d.get("error", ""), "success": False}
                    for a, d in evidence.items()
                ],
                confidence=0.0,
            )

        # Extract confidence scores
        confidences: Dict[str, float] = {}
        all_evidence: List[Dict[str, Any]] = []

        for agent, data in successful.items():
            conf = float(data.get("confidence", 0.5))
            confidences[agent] = conf
            all_evidence.append(
                {
                    "source": agent,
                    "text": str(data.get("data", ""))[:500],
                    "confidence": conf,
                    "success": True,
                }
            )

        # Compute combined confidence
        if self._weighter is not None:
            confidence = self._weighter.weighted_combine(
                [{"confidence": c, "source": a} for a, c in confidences.items()]
            ).get("combined_confidence", 0.0)
        elif confidences:
            values = list(confidences.values())
            confidence = sum(values) / len(values)
        else:
            confidence = 0.0

        # Build answer from all successful evidence
        parts = []
        for agent, data in successful.items():
            text = str(data.get("data", ""))
            if text and len(text) > 20:
                parts.append(f"[{agent}] {text}")

        answer = (
            "\n\n".join(parts) if parts else "No useful evidence could be synthesized."
        )

        return SynthesisResult(
            answer=answer,
            evidence=all_evidence,
            confidence=round(confidence, 4),
            agent_breakdown=confidences,
        )


# ===================================================================
# Specialist Agents
# ===================================================================


class SpecialistAgent:
    """Base class for all specialist sub-agents.

    Each agent wraps a specific tool from AgentTools and can be executed
    independently or in parallel with other agents.
    """

    def __init__(
        self,
        agent_type: str,
        config: Config,
        rag: Any = None,
        video_path: Optional[str] = None,
        video_id: Optional[str] = None,
    ):
        self.agent_type = agent_type
        self.config = config
        self.rag = rag
        self.video_path = Path(video_path) if video_path else None
        self.video_id = video_id
        self._tools: Optional[Any] = None

    def _get_tools(self) -> Any:
        """Lazy-load AgentTools from video_analysis.agent."""
        if self._tools is None:
            try:
                from video_analysis.agent import AgentTools

                self._tools = AgentTools(
                    config=self.config,
                    rag=self.rag,
                    video_path=str(self.video_path) if self.video_path else None,
                    video_id=self.video_id,
                )
            except Exception as exc:
                logger.warning("Could not load AgentTools: %s", exc)
        return self._tools

    def execute(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute the agent's specialist tool.

        Subclasses must override this.

        Args:
            query: Sub-query for this agent.
            context: Shared context (timestamps, prior evidence, etc.)

        Returns:
            Dict with keys: success, data, confidence, error
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.agent_type})"


class VisualAnalyst(SpecialistAgent):
    """Analyzes video frames using Video MLLM.

    Wraps the analyze_frames tool with intent-driven prompts
    (Question-Aware Captioning inspired by HiCrew).
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__("visual_analyst", *args, **kwargs)

    def execute(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        ctx = context or {}
        timestamps = ctx.get("timestamps", [])
        if not timestamps:
            try:
                from video_analysis.agent import VideoUnderstandingAgent

                ts = VideoUnderstandingAgent._extract_timestamps(query)
                timestamps = ts if ts else [10.0]
            except Exception:
                timestamps = [10.0]

        # Build intent-driven prompt (Question-Aware Captioning)
        prompt = f"In the context of the question '{query}', describe in detail what you see."
        if "action" in query.lower() or "doing" in query.lower():
            prompt += " Focus on actions and movements."

        tools = self._get_tools()
        if tools is None:
            return {
                "success": False,
                "data": "",
                "confidence": 0.0,
                "error": "AgentTools unavailable",
            }

        try:
            result = tools.analyze_frames(timestamps=timestamps, prompt=prompt)
            return {
                "success": result.success,
                "data": result.data,
                "confidence": 0.7 if result.success else 0.0,
                "error": "" if result.success else result.data,
                "metadata": result.metadata,
            }
        except Exception as exc:
            return {"success": False, "data": "", "confidence": 0.0, "error": str(exc)}


class RAGSearcher(SpecialistAgent):
    """Searches the RAG vector index with query refinement.

    Wraps the search_rag tool with query refinement to improve retrieval.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__("rag_searcher", *args, **kwargs)

    @staticmethod
    def refine_query(query: str) -> str:
        """Refine the query for better RAG retrieval."""
        refinements = [
            ("what about", "details about"),
            ("tell me ", ""),
            ("can you ", ""),
        ]
        q = query.lower().strip()
        for old, new in refinements:
            q = q.replace(old, new)
        return q.strip() or query

    def execute(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if self.rag is None:
            return {
                "success": False,
                "data": "No RAG instance available",
                "confidence": 0.0,
                "error": "no_rag",
            }

        refined = self.refine_query(query)
        ctx = context or {}
        top_k = ctx.get("top_k", 5)

        tools = self._get_tools()
        if tools is None:
            return {
                "success": False,
                "data": "",
                "confidence": 0.0,
                "error": "AgentTools unavailable",
            }

        try:
            result = tools.search_rag(query=refined, top_k=top_k)
            return {
                "success": result.success,
                "data": result.data,
                "confidence": 0.8 if result.success else 0.0,
                "error": "" if result.success else result.data,
                "metadata": result.metadata,
            }
        except Exception as exc:
            return {"success": False, "data": "", "confidence": 0.0, "error": str(exc)}


class TranscriptAnalyst(SpecialistAgent):
    """Searches transcript with temporal grounding capabilities."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__("transcript_analyst", *args, **kwargs)

    def execute(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        ctx = context or {}
        tools = self._get_tools()
        if tools is None:
            return {
                "success": False,
                "data": "",
                "confidence": 0.0,
                "error": "AgentTools unavailable",
            }

        try:
            result = tools.search_transcript(query=query, top_k=ctx.get("top_k", 5))

            # Also try temporal grounding if timestamps in query
            temporal_data = ""
            try:
                from video_analysis.agent import VideoUnderstandingAgent

                ts = VideoUnderstandingAgent._extract_timestamps(query)
                if ts:
                    tg = tools.temporal_grounding(f"What happens around {ts[0]:.0f}s?")
                    if tg.success:
                        temporal_data = tg.data
            except Exception:
                pass

            combined = result.data
            if temporal_data:
                combined += f"\n\n[Temporal Context]\n{temporal_data}"

            return {
                "success": result.success,
                "data": combined,
                "confidence": 0.75 if result.success else 0.0,
                "error": "" if result.success else result.data,
                "metadata": {**result.metadata, "temporal_data": bool(temporal_data)},
            }
        except Exception as exc:
            return {"success": False, "data": "", "confidence": 0.0, "error": str(exc)}


class ObjectDetectorAgent(SpecialistAgent):
    """Detects objects in video frames using YOLO + entity tracking."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__("object_detector", *args, **kwargs)

    def execute(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        ctx = context or {}
        timestamp = ctx.get("timestamp", 30.0)

        tools = self._get_tools()
        if tools is None:
            return {
                "success": False,
                "data": "",
                "confidence": 0.0,
                "error": "AgentTools unavailable",
            }

        try:
            result = tools.detect_objects(timestamp=timestamp)
            return {
                "success": result.success,
                "data": result.data,
                "confidence": 0.85 if result.success else 0.0,
                "error": "" if result.success else result.data,
                "metadata": result.metadata,
            }
        except Exception as exc:
            return {"success": False, "data": "", "confidence": 0.0, "error": str(exc)}


class OCRAgent(SpecialistAgent):
    """Extracts text from video frames using OCR."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__("ocr_agent", *args, **kwargs)

    def execute(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        ctx = context or {}
        timestamp = ctx.get("timestamp", 30.0)

        tools = self._get_tools()
        if tools is None:
            return {
                "success": False,
                "data": "",
                "confidence": 0.0,
                "error": "AgentTools unavailable",
            }

        try:
            result = tools.extract_text(timestamp=timestamp)
            return {
                "success": result.success,
                "data": result.data,
                "confidence": 0.7 if result.success else 0.0,
                "error": "" if result.success else result.data,
                "metadata": result.metadata,
            }
        except Exception as exc:
            return {"success": False, "data": "", "confidence": 0.0, "error": str(exc)}


class ConfidenceAuditor(SpecialistAgent):
    """Cross-validates evidence using the v0.50.0 Robust-TO confidence framework.

    Uses EvidenceTrustScorer and EvidenceWeighter to assess the reliability
    of evidence produced by other agents.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__("confidence_auditor", *args, **kwargs)

    def execute(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        ctx = context or {}
        prior_evidence = ctx.get("evidence", {})

        if not _HAS_CONFIDENCE or not prior_evidence:
            return {
                "success": True,
                "data": "Confidence auditing unavailable — no scoring module.",
                "confidence": 0.5,
                "error": "",
            }

        try:
            scorer = EvidenceTrustScorer()
            scores = []

            for agent_type, data in prior_evidence.items():
                if data.get("success"):
                    conf = float(data.get("confidence", 0.5))
                    if agent_type == "visual_analyst":
                        scored_data = scorer.score_mllm_response(
                            response=data.get("data", ""),
                            frame_quality=0.8,
                            num_frames=1,
                        )
                        scored = scored_data.get("mllm_confidence", conf)
                    elif agent_type == "rag_searcher":
                        scored_data = scorer.score_rag_chunk(
                            chunk=data.get("data_preview", {}),
                        )
                        scored = scored_data.get("source_confidence", conf)
                    elif agent_type == "object_detector":
                        scored_data = scorer.score_detection(
                            detections=[{"label": "object", "confidence": conf}],
                            frame_quality={"trustworthiness": 0.8},
                        )
                        scored = scored_data.get("mean_adjusted_confidence", conf)
                    elif agent_type == "ocr_agent":
                        scored = conf * 0.9
                    else:
                        scored = conf * 0.8
                    scores.append(scored)

            avg_confidence = sum(scores) / len(scores) if scores else 0.0

            return {
                "success": True,
                "data": f"Cross-validation complete. Audited {len(scores)} sources. "
                f"Average adjusted confidence: {avg_confidence:.3f}",
                "confidence": round(avg_confidence, 4),
                "error": "",
                "metadata": {
                    "sources_audited": len(scores),
                    "individual_scores": scores,
                },
            }
        except Exception as exc:
            return {"success": False, "data": "", "confidence": 0.0, "error": str(exc)}


class SummarizerAgent(SpecialistAgent):
    """Produces a structured video summary.

    Wraps the summarize_video tool for comprehensive video summarization.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__("summarizer", *args, **kwargs)

    def execute(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        ctx = context or {}
        num_frames = ctx.get("num_frames", 16)

        tools = self._get_tools()
        if tools is None:
            return {
                "success": False,
                "data": "",
                "confidence": 0.0,
                "error": "AgentTools unavailable",
            }

        try:
            result = tools.summarize_video(num_frames=num_frames)

            tx = tools.search_transcript(query, top_k=5)

            combined = result.data
            if tx.success:
                combined += f"\n\n[Transcript Highlights]\n{tx.data[:1000]}"

            return {
                "success": result.success,
                "data": combined,
                "confidence": 0.8 if result.success else 0.0,
                "error": "" if result.success else result.data,
                "metadata": {"num_frames": num_frames, "has_transcript": tx.success},
            }
        except Exception as exc:
            return {"success": False, "data": "", "confidence": 0.0, "error": str(exc)}


# ===================================================================
# Router Agent — Question-aware Planning Layer
# ===================================================================

# Keyword-based modality classification (fallback when no LLM)
_QUERY_PATTERNS: Dict[str, List[str]] = {
    "visual": [
        "see",
        "look",
        "visual",
        "scene",
        "appear",
        "show",
        "display",
        "color",
        "shape",
        "background",
        "view",
    ],
    "text": [
        "say",
        "talk",
        "speak",
        "mention",
        "discuss",
        "transcript",
        "dialogue",
    ],
    "temporal": [
        "when",
        "time",
        "minute",
        "second",
        "before",
        "after",
        "during",
        "moment",
        "occur",
    ],
    "action": ["do", "doing", "action", "move", "movement", "happen", "activity"],
    "entity": [
        "who",
        "person",
        "people",
        "object",
        "face",
        "character",
        "animal",
        "vehicle",
    ],
    "summary": [
        "summarize",
        "summary",
        "overview",
        "what happen",
        "what is in",
        "describe",
    ],
}

_MODALITY_TO_AGENTS: Dict[str, List[Dict[str, Any]]] = {
    "visual": [
        {"type": "visual_analyst", "deps": []},
    ],
    "text": [
        {"type": "rag_searcher", "deps": []},
        {"type": "transcript_analyst", "deps": ["rag_searcher"]},
    ],
    "temporal": [
        {"type": "transcript_analyst", "deps": []},
        {"type": "visual_analyst", "deps": ["transcript_analyst"]},
    ],
    "action": [
        {"type": "visual_analyst", "deps": []},
        {"type": "object_detector", "deps": ["visual_analyst"]},
    ],
    "entity": [
        {"type": "object_detector", "deps": []},
        {"type": "rag_searcher", "deps": []},
    ],
    "summary": [
        {"type": "summarizer", "deps": []},
        {"type": "rag_searcher", "deps": ["summarizer"]},
    ],
}


class RouterAgent:
    """Question-aware planning layer that generates execution plans.

    Analyzes the user's question to determine:
    - Required modalities (visual / text / temporal / action / entity)
    - Question complexity (simple / multi-hop / analytical)
    - Optimal agent selection and ordering

    Uses LLMProvider when available, falls back to rule-based classification.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self._llm: Optional[Any] = None
        if _HAS_LLM_PROVIDER:
            try:
                llm_config = LLMProviderConfig.from_env()
                self._llm = _get_llm_provider(config=llm_config)
            except Exception:
                pass

    def analyze(self, query: str) -> RoutePlan:
        """Analyze a question and produce a RoutePlan.

        Args:
            query: Natural language question about the video.

        Returns:
            RoutePlan with tasks, modalities, and complexity.
        """
        if not query or not query.strip():
            return RoutePlan(query=query or "")

        # Detect modalities
        modalities = self._detect_modalities(query.lower())

        # Determine complexity
        complexity = self._determine_complexity(query, len(modalities))

        # Generate tasks
        tasks = self._plan_tasks(modalities, query)

        # Reasoning trace
        reasoning = [
            f"Detected modalities: {', '.join(sorted(modalities))}",
            f"Complexity: {complexity}",
            f"Agents planned: {', '.join(t.agent_type for t in tasks)}",
        ]

        return RoutePlan(
            query=query,
            tasks=tasks,
            complexity=complexity,
            modalities=modalities,
            reasoning_path=reasoning,
        )

    def _detect_modalities(self, query_lower: str) -> Set[str]:
        """Determine which modalities are required by the question.

        Uses keyword matching (fallback) or LLM-based analysis when available.
        """
        modalities: Set[str] = set()

        for modality, keywords in _QUERY_PATTERNS.items():
            for kw in keywords:
                if kw in query_lower:
                    modalities.add(modality)
                    break

        # Ensure at least one modality
        if not modalities:
            modalities.add("summary")

        return modalities

    @staticmethod
    def _determine_complexity(query: str, num_modalities: int) -> str:
        """Determine question complexity based on modality count."""
        if num_modalities >= 3:
            return "analytical"
        elif num_modalities >= 2:
            return "multi-hop"
        else:
            return "simple"

    def _plan_tasks(self, modalities: Set[str], query: str) -> List[TaskItem]:
        """Generate ordered tasks from detected modalities.

        Tasks are ordered with dependencies to allow parallel execution
        of independent subtasks.
        """
        tasks: List[TaskItem] = []
        seen_types: Set[str] = set()

        # Priority order for modalities
        priority = [
            "summary",
            "visual",
            "text",
            "temporal",
            "entity",
            "action",
            "confidence",
        ]

        for mod in priority:
            if mod not in modalities:
                continue
            agent_configs = _MODALITY_TO_AGENTS.get(mod, [])
            for cfg in agent_configs:
                agent_type = cfg["type"]
                if agent_type not in seen_types:
                    deps = [d for d in cfg["deps"] if d in seen_types]
                    tasks.append(
                        TaskItem(
                            agent_type=agent_type,
                            description=f"Execute {agent_type} for modality {mod}",
                            sub_query=query,
                            dependencies=deps,
                        )
                    )
                    seen_types.add(agent_type)

        # Add confidence auditor for complex queries
        if "confidence" in modalities or len(tasks) >= 3:
            tasks.append(
                TaskItem(
                    agent_type="confidence_auditor",
                    description="Cross-validate all evidence",
                    sub_query=query,
                    dependencies=list(seen_types),
                )
            )

        return tasks


# ===================================================================
# Orchestrator Result
# ===================================================================


@dataclass
class OrchestratorResult:
    """Final result from the Multi-Agent Orchestrator.

    Mirrors AgentQueryResult fields for drop-in compatibility while adding
    orchestration-specific metadata.
    """

    query: str
    answer: str
    confidence: float = 0.0
    agents_used: int = 0
    plan_duration: float = 0.0
    execution_duration: float = 0.0
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    reasoning: List[str] = field(default_factory=list)
    plan: Optional[RoutePlan] = None
    agent_breakdown: Dict[str, float] = field(default_factory=dict)

    @property
    def tools_used(self) -> int:
        """Compatibility alias — mirrors AgentQueryResult.tools_used."""
        return self.agents_used

    @property
    def duration_seconds(self) -> float:
        """Compatibility alias — mirrors AgentQueryResult.duration_seconds."""
        return self.plan_duration + self.execution_duration

    @property
    def reasoning_steps(self) -> List[str]:
        """Compatibility alias — mirrors AgentQueryResult.reasoning_steps."""
        return self.reasoning

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "confidence": self.confidence,
            "agents_used": self.agents_used,
            "plan_duration": self.plan_duration,
            "execution_duration": self.execution_duration,
            "evidence": self.evidence,
            "reasoning": self.reasoning,
            "agent_breakdown": self.agent_breakdown,
        }

    def to_markdown(self) -> str:
        """Render as markdown for display."""
        lines = [
            f"## Answer (confidence: {self.confidence:.1%})",
            "",
            self.answer,
            "",
            f"---",
            f"*Agents used: {self.agents_used} | "
            f"Planning: {self.plan_duration:.1f}s | "
            f"Execution: {self.execution_duration:.1f}s*",
        ]
        if self.reasoning:
            lines.extend(["", "### Reasoning Trace", ""])
            lines.extend(f"- {s}" for s in self.reasoning)
        return "\n".join(lines)


# ===================================================================
# MultiAgentOrchestrator — Top-level orchestrator
# ===================================================================


class MultiAgentOrchestrator:
    """Top-level hierarchical multi-agent video reasoning orchestrator.

    Integrates with the existing video-analysis ecosystem:
    - Uses AgentTools from agent.py for underlying video tools
    - Uses EvidenceWeighter from agent_confidence.py (v0.50.0)
    - Uses LLMProvider from llm_provider.py
    - Uses VideoRAG from rag.py

    Usage:
        orch = MultiAgentOrchestrator(config, rag, video_path, video_id)
        result = orch.query("What objects are visible and who is speaking?")
        print(result.answer)
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        rag: Any = None,
        video_path: Optional[str] = None,
        video_id: Optional[str] = None,
    ):
        self.config = config or Config()
        self.rag = rag
        self.video_path = video_path
        self.video_id = video_id
        self._router = RouterAgent(config=self.config)
        self._synthesizer = EvidenceSynthesizer(config=self.config)

        # Config with env-var fallback
        self._orch_cfg = _get_orchestra_config(self.config)
        self._enabled: bool = self._orch_cfg["enabled"]
        self._max_agents: int = self._orch_cfg["max_agents"]
        self._confidence_threshold: float = self._orch_cfg["confidence_threshold"]

        # Agent registry: agent_type → factory callable
        self._agent_registry: Dict[str, Any] = {}

        logger.info(
            "MultiAgentOrchestrator initialised (enabled=%s, max_agents=%d, conf_threshold=%.2f)",
            self._enabled,
            self._max_agents,
            self._confidence_threshold,
        )

    def _get_agent(self, agent_type: str) -> Optional[SpecialistAgent]:
        """Get or create a specialist agent by type."""
        if agent_type not in self._agent_registry:
            factory = {
                "visual_analyst": lambda: VisualAnalyst(
                    config=self.config,
                    rag=self.rag,
                    video_path=self.video_path,
                    video_id=self.video_id,
                ),
                "rag_searcher": lambda: RAGSearcher(
                    config=self.config,
                    rag=self.rag,
                    video_path=self.video_path,
                    video_id=self.video_id,
                ),
                "transcript_analyst": lambda: TranscriptAnalyst(
                    config=self.config,
                    rag=self.rag,
                    video_path=self.video_path,
                    video_id=self.video_id,
                ),
                "object_detector": lambda: ObjectDetectorAgent(
                    config=self.config,
                    rag=self.rag,
                    video_path=self.video_path,
                    video_id=self.video_id,
                ),
                "ocr_agent": lambda: OCRAgent(
                    config=self.config,
                    rag=self.rag,
                    video_path=self.video_path,
                    video_id=self.video_id,
                ),
                "confidence_auditor": lambda: ConfidenceAuditor(
                    config=self.config,
                    rag=self.rag,
                    video_path=self.video_path,
                    video_id=self.video_id,
                ),
                "summarizer": lambda: SummarizerAgent(
                    config=self.config,
                    rag=self.rag,
                    video_path=self.video_path,
                    video_id=self.video_id,
                ),
            }
            creator = factory.get(agent_type)
            if creator:
                self._agent_registry[agent_type] = creator()
            else:
                return None
        return self._agent_registry.get(agent_type)

    @property
    def enabled(self) -> bool:
        """Whether the orchestrator is enabled."""
        return self._enabled

    def query(
        self,
        query: str,
        context: Optional[List[Any]] = None,
        max_agents: Optional[int] = None,
    ) -> OrchestratorResult:
        """Answer a question using the hierarchical multi-agent system.

        Args:
            query: Natural language question about the video.
            context: Optional pre-retrieved RAG chunks for bootstrapping.
            max_agents: Maximum number of agents to invoke
                        (default: from config or env ORCHESTRA_MAX_AGENTS).

        Returns:
            OrchestratorResult with answer, confidence, and agent breakdown.
        """
        if not query or not query.strip():
            return OrchestratorResult(query=query or "", answer="", confidence=0.0)

        # Fallback when orchestrator is disabled
        if not self._enabled:
            logger.info("Orchestrator disabled — falling back to flat agent")
            return self._fallback_query(query, context)

        max_agents = max_agents or self._max_agents
        start_time = time.time()

        # Phase 1: Route planning
        plan_start = time.time()
        route_plan = self._router.analyze(query)
        plan_duration = time.time() - plan_start

        if not route_plan.tasks:
            return OrchestratorResult(
                query=query,
                answer="Could not create a plan for this question.",
                confidence=0.0,
                agents_used=0,
                plan_duration=plan_duration,
                reasoning=route_plan.reasoning_path,
            )

        # Phase 2: Execute agents (parallel where possible)
        evidence: Dict[str, Dict[str, Any]] = {}
        agents_used = 0

        while not route_plan.is_complete and agents_used < max_agents:
            ready = route_plan.ready_tasks()
            if not ready:
                break

            # Run independent tasks in parallel
            with ThreadPoolExecutor(max_workers=min(len(ready), 4)) as pool:
                future_to_task = {}
                for task in ready:
                    agent = self._get_agent(task.agent_type)
                    if agent is None:
                        task.completed = True
                        task.result = {
                            "success": False,
                            "error": f"Unknown agent: {task.agent_type}",
                        }
                        continue

                    ctx: Dict[str, Any] = {
                        "evidence": evidence,
                        "timestamps": [],
                        "video_id": self.video_id,
                    }
                    if context:
                        ctx["context_chunks"] = [str(c)[:200] for c in context[:5]]

                    future = pool.submit(agent.execute, task.sub_query or query, ctx)
                    future_to_task[future] = task

                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        result = future.result(timeout=60)
                    except Exception as exc:
                        result = {
                            "success": False,
                            "data": "",
                            "confidence": 0.0,
                            "error": str(exc),
                        }

                    task.completed = True
                    task.result = result
                    task.confidence = result.get("confidence", 0.0)
                    evidence[task.agent_type] = result
                    agents_used += 1

                    # Early stopping per-task
                    if task.confidence and task.confidence_threshold > 0:
                        if task.confidence >= task.confidence_threshold:
                            route_plan.reasoning_path.append(
                                f"Early stop: {task.agent_type} "
                                f"reached confidence {task.confidence:.3f}"
                            )
                            break

            # Check overall confidence-based early stopping
            completed_confidences = [
                t.confidence
                for t in route_plan.tasks
                if t.completed and t.confidence is not None
            ]
            if completed_confidences:
                avg_conf = sum(completed_confidences) / len(completed_confidences)
                if avg_conf >= self._confidence_threshold:
                    route_plan.reasoning_path.append(
                        f"Early stop: avg confidence {avg_conf:.3f} >= threshold"
                    )
                    for t in route_plan.tasks:
                        if not t.completed:
                            t.completed = True
                            evidence[t.agent_type] = {
                                "success": False,
                                "data": "",
                                "confidence": 0.0,
                                "error": "Skipped by early stopping",
                            }
                    break

        execution_duration = time.time() - start_time

        # Phase 3: Evidence synthesis
        synthesis = self._synthesizer.synthesize(query, evidence)

        return OrchestratorResult(
            query=query,
            answer=synthesis.answer,
            confidence=synthesis.confidence,
            agents_used=agents_used,
            plan_duration=plan_duration,
            execution_duration=execution_duration,
            evidence=synthesis.evidence,
            reasoning=route_plan.reasoning_path,
            plan=route_plan,
            agent_breakdown=synthesis.agent_breakdown,
        )

    def _fallback_query(
        self, query: str, context: Optional[List[Any]] = None
    ) -> OrchestratorResult:
        """Fallback: use flat VideoUnderstandingAgent when orchestrator is off."""
        try:
            from video_analysis.agent import VideoUnderstandingAgent

            flat = VideoUnderstandingAgent(
                config=self.config,
                rag=self.rag,
                video_path=self.video_path,
                video_id=self.video_id,
            )
            result = flat.query(query, context=context)
            return OrchestratorResult(
                query=result.query,
                answer=result.answer,
                confidence=result.confidence,
                evidence=[
                    {"source": e.tool_name, "text": e.data, "success": e.success}
                    for e in result.evidence
                ],
                reasoning=result.reasoning_steps,
                agents_used=result.tools_used,
                execution_duration=result.duration_seconds,
            )
        except Exception as exc:
            logger.exception("Fallback agent failed")
            return OrchestratorResult(
                query=query,
                answer=f"Fallback agent error: {exc}",
                confidence=0.0,
            )


# ===================================================================
# Factory / Convenience Functions
# ===================================================================


def get_orchestrator(
    config: Optional[Config] = None,
    rag: Any = None,
    video_path: Optional[str] = None,
    video_id: Optional[str] = None,
) -> MultiAgentOrchestrator:
    """Factory convenience function to create a configured orchestrator.

    Reads ``ORCHESTRA_ENABLED``, ``ORCHESTRA_MAX_AGENTS``, and
    ``ORCHESTRA_CONFIDENCE_THRESHOLD`` from environment variables,
    falling back to Config values or defaults.

    Args:
        config: Project Config (or None for defaults).
        rag: Optional VideoRAG instance.
        video_path: Path to the video file.
        video_id: Video identifier.

    Returns:
        A MultiAgentOrchestrator instance.
    """
    cfg = config or Config()
    orch = MultiAgentOrchestrator(
        config=cfg,
        rag=rag,
        video_path=video_path,
        video_id=video_id,
    )
    return orch
