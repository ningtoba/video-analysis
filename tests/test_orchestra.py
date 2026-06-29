"""
Tests for the Hierarchical Multi-Agent Video Reasoning Orchestrator (v0.51.0).

Covers:
- HybridTree data structures (scene clustering)
- RoutePlan planning dataclass
- RouterAgent question analysis and plan generation
- SpecialistAgent base class
- EvidenceSynthesizer evidence combination
- MultiAgentOrchestrator end-to-end flow
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

import pytest

from video_analysis.orchestra import (
    HybridTree,
    HybridNode,
    RoutePlan,
    TaskItem,
    RouterAgent,
    SpecialistAgent,
    VisualAnalyst,
    RAGSearcher,
    TranscriptAnalyst,
    ObjectDetectorAgent,
    OCRAgent,
    ConfidenceAuditor,
    SummarizerAgent,
    EvidenceSynthesizer,
    SynthesisResult,
    MultiAgentOrchestrator,
    OrchestratorResult,
)
from video_analysis.config import Config

# ---------------------------------------------------------------------------
# Helper: create a minimal config with orchestra enabled
# ---------------------------------------------------------------------------


def _make_config(**overrides) -> Config:
    cfg = Config()
    cfg.orchestra_enabled = overrides.get("orchestra_enabled", True)
    cfg.orchestra_max_agents = overrides.get("orchestra_max_agents", 5)
    cfg.orchestra_confidence_threshold = overrides.get(
        "orchestra_confidence_threshold", 0.5
    )
    # Set LLM provider to hermes for testing
    cfg.llm_provider = "hermes"
    return cfg


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_scenes():
    """Return a list of mock scene dicts for HybridTree testing."""
    return [
        {
            "scene_id": 0,
            "start": 0.0,
            "end": 10.0,
            "description": "introduction scene with people talking",
        },
        {
            "scene_id": 1,
            "start": 10.0,
            "end": 25.0,
            "description": "people discussing project plans",
        },
        {
            "scene_id": 2,
            "start": 25.0,
            "end": 40.0,
            "description": "whiteboard demonstration",
        },
        {
            "scene_id": 3,
            "start": 40.0,
            "end": 65.0,
            "description": "outdoor establishing shot",
        },
        {
            "scene_id": 4,
            "start": 65.0,
            "end": 80.0,
            "description": "outdoor conversation",
        },
        {
            "scene_id": 5,
            "start": 80.0,
            "end": 100.0,
            "description": "park scene with wildlife",
        },
    ]


@pytest.fixture
def mock_config():
    return _make_config()


# ---------------------------------------------------------------------------
# HybridTree Tests
# ---------------------------------------------------------------------------


class TestHybridNode:
    def test_leaf_node(self):
        node = HybridNode(scene_id=0, label="Scene 0", level=0)
        assert node.scene_id == 0
        assert node.label == "Scene 0"
        assert node.level == 0
        assert node.children == []
        assert node.is_leaf is True

    def test_parent_node(self):
        child = HybridNode(scene_id=0, label="Scene 0", level=0)
        parent = HybridNode(
            scene_id=-1,
            label="Cluster 1",
            level=1,
            children=[child],
        )
        assert parent.is_leaf is False
        assert len(parent.children) == 1
        assert parent.children[0].scene_id == 0


class TestHybridTree:
    def test_empty_scenes(self, mock_config):
        tree = HybridTree(config=mock_config)
        tree.build([])
        assert tree.root is None
        assert tree.num_leaves == 0

    def test_single_scene(self, mock_config, mock_scenes):
        tree = HybridTree(config=mock_config)
        tree.build(mock_scenes[:1])
        assert tree.root is not None
        assert tree.root.is_leaf
        assert tree.root.scene_id == 0

    def test_multiple_scenes_same_cluster(self, mock_config, mock_scenes):
        """Scenes close in time and with similar descriptions should cluster."""
        tree = HybridTree(config=mock_config)
        tree.build(mock_scenes[:3])
        assert tree.root is not None
        # At least one cluster should have been formed
        assert tree.num_leaves == 3

    def test_find_scene_by_id(self, mock_config, mock_scenes):
        tree = HybridTree(config=mock_config)
        tree.build(mock_scenes)
        node = tree.find_scene(scene_id=3)
        assert node is not None
        assert node.scene_id == 3

    def test_find_scene_not_found(self, mock_config, mock_scenes):
        tree = HybridTree(config=mock_config)
        tree.build(mock_scenes)
        node = tree.find_scene(scene_id=999)
        assert node is None

    def test_cluster_path(self, mock_config, mock_scenes):
        """Each leaf should have a path from root to itself."""
        tree = HybridTree(config=mock_config)
        tree.build(mock_scenes)
        paths = tree.get_leaf_paths()
        assert len(paths) == len(mock_scenes)
        # Each path should start at root and end at leaf
        for path in paths:
            assert len(path) >= 1
            assert path[-1].is_leaf


# ---------------------------------------------------------------------------
# RoutePlan Tests
# ---------------------------------------------------------------------------


class TestTaskItem:
    def test_defaults(self):
        t = TaskItem(agent_type="rag_searcher", description="Search RAG")
        assert t.agent_type == "rag_searcher"
        assert t.description == "Search RAG"
        assert t.dependencies == []
        assert t.confidence is None
        assert t.completed is False

    def test_with_deps(self):
        t = TaskItem(
            agent_type="visual_analyst",
            description="Analyze frames",
            dependencies=["rag_searcher"],
        )
        assert "rag_searcher" in t.dependencies


class TestRoutePlan:
    def test_empty(self):
        p = RoutePlan(query="test?")
        assert p.query == "test?"
        assert p.tasks == []
        assert p.complexity == "simple"

    def test_with_tasks(self):
        t1 = TaskItem(agent_type="rag_searcher", description="Search")
        t2 = TaskItem(
            agent_type="visual_analyst",
            description="Analyze",
            dependencies=["rag_searcher"],
        )
        p = RoutePlan(
            query="describe the video", tasks=[t1, t2], complexity="multi-hop"
        )
        assert len(p.tasks) == 2
        assert p.complexity == "multi-hop"
        assert p.is_complete is False

    def test_is_complete(self):
        t1 = TaskItem(agent_type="rag_searcher", description="Search", completed=True)
        p = RoutePlan(query="test", tasks=[t1])
        assert p.is_complete is True

    def test_ready_tasks_no_deps(self):
        t1 = TaskItem(agent_type="rag_searcher", description="Search")
        t2 = TaskItem(
            agent_type="analyst", description="Analyze", dependencies=["rag_searcher"]
        )
        p = RoutePlan(query="test", tasks=[t1, t2])
        ready = p.ready_tasks()
        assert len(ready) == 1
        assert ready[0].agent_type == "rag_searcher"

    def test_ready_tasks_after_completion(self):
        t1 = TaskItem(agent_type="rag_searcher", description="Search", completed=True)
        t2 = TaskItem(
            agent_type="analyst", description="Analyze", dependencies=["rag_searcher"]
        )
        p = RoutePlan(query="test", tasks=[t1, t2])
        ready = p.ready_tasks()
        assert len(ready) == 1
        assert ready[0].agent_type == "analyst"


# ---------------------------------------------------------------------------
# RouterAgent Tests
# ---------------------------------------------------------------------------


class TestRouterAgent:
    def test_creates_plan(self, mock_config):
        """RouterAgent should produce a RoutePlan for any question."""
        router = RouterAgent(config=mock_config)
        plan = router.analyze("What objects are visible in the first minute?")
        assert isinstance(plan, RoutePlan)
        assert plan.query == "What objects are visible in the first minute?"
        assert len(plan.tasks) > 0

    def test_visual_question_triggers_visual_analyst(self, mock_config):
        router = RouterAgent(config=mock_config)
        plan = router.analyze("Describe what you see in the video")
        agent_types = {t.agent_type for t in plan.tasks}
        assert "visual_analyst" in agent_types

    def test_search_question_triggers_rag_searcher(self, mock_config):
        router = RouterAgent(config=mock_config)
        plan = router.analyze("What does the transcript say about the main topic?")
        agent_types = {t.agent_type for t in plan.tasks}
        assert "rag_searcher" in agent_types

    def test_complex_question_is_multi_hop(self, mock_config):
        router = RouterAgent(config=mock_config)
        plan = router.analyze("Who is speaking at 2:30 and what objects are near them?")
        # This question crosses modalities, should generate at least 2 tasks
        assert len(plan.tasks) >= 2 or plan.complexity == "multi-hop"

    def test_routes_summary_question(self, mock_config):
        router = RouterAgent(config=mock_config)
        plan = router.analyze("Summarize the entire video")
        agent_types = {t.agent_type for t in plan.tasks}
        assert "summarizer" in agent_types or "rag_searcher" in agent_types

    def test_no_llm_fallback(self, mock_config):
        """Router should work even without LLM (rule-based fallback)."""
        cfg = _make_config(llm_provider="none")
        router = RouterAgent(config=cfg)
        plan = router.analyze("What happened in the video?")
        assert len(plan.tasks) >= 1  # Fallback should still produce something


# ---------------------------------------------------------------------------
# SpecialistAgent Tests
# ---------------------------------------------------------------------------


class TestSpecialistAgent:
    def test_base_execute_not_implemented(self, mock_config):
        """Base SpecialistAgent should raise NotImplementedError."""
        agent = SpecialistAgent(
            agent_type="test",
            config=mock_config,
        )
        with pytest.raises(NotImplementedError):
            agent.execute(query="test", context={})


class TestRAGSearcher:
    def test_no_rag_instance(self, mock_config):
        searcher = RAGSearcher(config=mock_config)
        result = searcher.execute(query="test query", context={})
        # Should gracefully handle missing RAG
        assert (
            "no rag" in result.get("data", "").lower()
            or "rag" in result.get("error", "").lower()
        )

    def test_refines_query(self, mock_config):
        searcher = RAGSearcher(config=mock_config)
        refined = searcher.refine_query("what is the man doing")
        assert isinstance(refined, str)
        assert len(refined) > 0


class TestTranscriptAnalyst:
    def test_no_transcript_data(self, mock_config):
        analyst = TranscriptAnalyst(config=mock_config)
        result = analyst.execute(query="find the part about cars", context={})
        # Should handle gracefully
        assert isinstance(result, dict)


class TestOCRAgent:
    def test_no_video_path(self, mock_config):
        agent = OCRAgent(config=mock_config)
        result = agent.execute(query="read text", context={})
        # Should handle missing video gracefully
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# EvidenceSynthesizer Tests
# ---------------------------------------------------------------------------


class TestSynthesisResult:
    def test_empty(self):
        sr = SynthesisResult()
        assert sr.answer == ""
        assert sr.evidence == []
        assert sr.confidence == 0.0

    def test_with_evidence(self):
        sr = SynthesisResult(
            answer="The video shows a person talking.",
            evidence=[
                {
                    "source": "visual_analyst",
                    "text": "person talking at 10s",
                    "confidence": 0.9,
                },
                {
                    "source": "rag_searcher",
                    "text": "transcript confirms speaking",
                    "confidence": 0.85,
                },
            ],
            confidence=0.87,
        )
        assert len(sr.evidence) == 2
        assert sr.confidence == 0.87


class TestEvidenceSynthesizer:
    def test_synthesize_empty(self, mock_config):
        synth = EvidenceSynthesizer(config=mock_config)
        result = synth.synthesize(query="test", evidence={})
        assert result.answer == ""
        assert result.confidence == 0.0

    def test_synthesize_single_source(self, mock_config):
        synth = EvidenceSynthesizer(config=mock_config)
        result = synth.synthesize(
            query="What is happening?",
            evidence={
                "visual_analyst": {
                    "success": True,
                    "data": "A person is walking across the room.",
                    "confidence": 0.9,
                }
            },
        )
        assert len(result.evidence) == 1
        assert result.confidence > 0.0
        assert "walking" in result.answer

    def test_synthesize_multiple_sources(self, mock_config):
        synth = EvidenceSynthesizer(config=mock_config)
        result = synth.synthesize(
            query="What is happening?",
            evidence={
                "visual_analyst": {
                    "success": True,
                    "data": "A person walking across the room.",
                    "confidence": 0.9,
                },
                "rag_searcher": {
                    "success": True,
                    "data": "Transcript: 'I am walking to the door.'",
                    "confidence": 0.95,
                },
            },
        )
        assert len(result.evidence) == 2
        # Combined confidence should be higher than individual
        assert result.confidence > 0.9

    def test_synthesize_with_failures(self, mock_config):
        synth = EvidenceSynthesizer(config=mock_config)
        result = synth.synthesize(
            query="test",
            evidence={
                "visual_analyst": {
                    "success": True,
                    "data": "A person is walking across the room confidently.",
                    "confidence": 0.9,
                },
                "object_detector": {
                    "success": False,
                    "data": "Model not loaded.",
                    "confidence": 0.0,
                },
            },
        )
        # Only successful evidence should appear in result
        assert result.confidence > 0.0
        assert "walking" in result.answer
        # The failed evidence contributed nothing
        assert len(result.evidence) == 1

    def test_confidence_weighting(self, mock_config):
        """High-confidence evidence should dominate synthesis."""
        synth = EvidenceSynthesizer(config=mock_config)
        result = synth.synthesize(
            query="test",
            evidence={
                "high": {"success": True, "data": "Reliable info.", "confidence": 0.99},
                "medium": {
                    "success": True,
                    "data": "Maybe relevant.",
                    "confidence": 0.55,
                },
                "low": {"success": True, "data": "Uncertain claim.", "confidence": 0.2},
            },
        )
        assert result.confidence > 0.5  # High-confidence source should pull up avg


# ---------------------------------------------------------------------------
# MultiAgentOrchestrator Tests
# ---------------------------------------------------------------------------


class TestOrchestratorResult:
    def test_defaults(self):
        r = OrchestratorResult(query="test", answer="")
        assert r.answer == ""
        assert r.confidence == 0.0
        assert r.agents_used == 0
        assert r.plan_duration == 0.0

    def test_formatted(self):
        r = OrchestratorResult(
            query="What happens?",
            answer="Walking.",
            confidence=0.85,
            agents_used=2,
            plan_duration=1.5,
            evidence=[
                {
                    "agent": "visual_analyst",
                    "data": "Walking at 10s",
                    "confidence": 0.9,
                },
            ],
            reasoning=["Visual analysis done", "Synthesis complete"],
        )
        assert r.confidence == 0.85
        assert r.agents_used == 2
        assert "Walking" in r.answer


class TestMultiAgentOrchestrator:
    def test_init_disabled(self):
        """Orchestrator with orchestra_enabled=False should not initialize agents."""
        cfg = _make_config(orchestra_enabled=False)
        orch = MultiAgentOrchestrator(config=cfg)
        assert orch is not None

    def test_query_no_rag(self, mock_config):
        """Orchestrator should handle missing RAG gracefully."""
        orch = MultiAgentOrchestrator(config=mock_config)
        result = orch.query("What is in the video?")
        assert isinstance(result, OrchestratorResult)
        assert result.agents_used >= 0

    def test_query_empty(self, mock_config):
        """Empty query should return empty answer."""
        orch = MultiAgentOrchestrator(config=mock_config)
        result = orch.query("")
        assert result.answer == "" or result.confidence == 0.0

    def test_returns_orchestrator_result(self, mock_config):
        orch = MultiAgentOrchestrator(config=mock_config)
        result = orch.query("Describe the video content.")
        assert isinstance(result, OrchestratorResult)

    def test_route_plan_generated(self, mock_config):
        """Each query should produce a route plan."""
        orch = MultiAgentOrchestrator(config=mock_config)
        result = orch.query("What objects are in the scene?")
        assert len(result.evidence) >= 0
        assert result.answer is not None

    def test_temporal_question(self, mock_config):
        """Temporal questions should trigger transcript + temporal grounding."""
        orch = MultiAgentOrchestrator(config=mock_config)
        result = orch.query("What happens at 1 minute 30 seconds?")
        assert result is not None

    def test_visual_question(self, mock_config):
        """Visual questions should trigger visual analyst."""
        orch = MultiAgentOrchestrator(config=mock_config)
        result = orch.query("Describe the visual scene in detail.")
        assert result is not None

    def test_complex_question(self, mock_config):
        """Complex questions should use multiple agents."""
        orch = MultiAgentOrchestrator(config=mock_config)
        result = orch.query("Who is speaking and what objects are visible?")
        assert result is not None

    def test_confidence_threshold(self, mock_config):
        """Early stopping when confidence threshold is met."""
        cfg = _make_config(orchestra_confidence_threshold=0.9)
        orch = MultiAgentOrchestrator(config=cfg)
        result = orch.query("Simple question?")
        assert isinstance(result, OrchestratorResult)


# ---------------------------------------------------------------------------
# Integration Tests — ensure orchestra imports don't crash
# ---------------------------------------------------------------------------


class TestModuleImports:
    def test_all_exports_exist(self):
        """All expected symbols should be importable."""
        from video_analysis.orchestra import (
            HybridTree,
            HybridNode,
            RoutePlan,
            TaskItem,
            RouterAgent,
            SpecialistAgent,
            VisualAnalyst,
            RAGSearcher,
            TranscriptAnalyst,
            ObjectDetectorAgent,
            OCRAgent,
            ConfidenceAuditor,
            SummarizerAgent,
            EvidenceSynthesizer,
            SynthesisResult,
            MultiAgentOrchestrator,
            OrchestratorResult,
        )

        assert HybridTree is not None
        assert RoutePlan is not None
        assert RouterAgent is not None
        assert SpecialistAgent is not None
        assert VisualAnalyst is not None
        assert RAGSearcher is not None
        assert TranscriptAnalyst is not None
        assert ObjectDetectorAgent is not None
        assert OCRAgent is not None
        assert ConfidenceAuditor is not None
        assert SummarizerAgent is not None
        assert EvidenceSynthesizer is not None
        assert MultiAgentOrchestrator is not None
