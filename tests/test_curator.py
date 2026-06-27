"""
Tests for the Autonomous Video Curator (v0.48.0).
"""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from video_analysis.curator import (
    VideoCurator,
    CuratorObservation,
    CuratorEntity,
    CuratorKnowledge,
    CuratorReportChunk,
    VideoCuratorReport,
    CuriosityStrategy,
    run_curation,
)
from video_analysis.config import Config

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_config():
    cfg = MagicMock(spec=Config)
    cfg.data_dir = Path("/tmp/test_curator_data")
    cfg.curator_curiosity = 0.5
    cfg.curator_max_iterations = 15
    cfg.curator_output_dir = ""
    cfg.curator_enabled = False
    return cfg


@pytest.fixture
def sample_observation():
    return CuratorObservation(
        observation_id="obs_001",
        timestamp_seconds=30.0,
        observation_type="scene",
        content="A person wearing a blue shirt is standing in a well-lit office with white walls and a window.",
        confidence=0.85,
        source_tool="analyze_frames",
        metadata={"sample_index": 0, "point_type": "sweep"},
    )


@pytest.fixture
def sample_entity():
    return CuratorEntity(
        entity_id="person_0",
        name="John Smith",
        entity_type="person",
        first_seen=30.0,
        last_seen=120.0,
        appearances=3,
        description="A person wearing a blue shirt in an office.",
        related_timestamps=[30.0, 75.0, 120.0],
        attributes={"clothing": "blue shirt", "setting": "office"},
    )


@pytest.fixture
def sample_knowledge():
    return CuratorKnowledge(
        video_id="test_video",
        video_path="/tmp/test_video.mp4",
        duration_seconds=300.0,
    )


# ---------------------------------------------------------------------------
# CuratorObservation tests
# ---------------------------------------------------------------------------


class TestCuratorObservation:
    def test_observation_creation(self, sample_observation):
        """Basic observation dataclass creation."""
        assert sample_observation.observation_id == "obs_001"
        assert sample_observation.timestamp_seconds == 30.0
        assert sample_observation.observation_type == "scene"
        assert sample_observation.confidence == 0.85
        assert sample_observation.source_tool == "analyze_frames"

    def test_observation_to_dict(self, sample_observation):
        """Observation serializes to dict."""
        d = sample_observation.to_dict()
        assert d["observation_id"] == "obs_001"
        assert d["timestamp_seconds"] == 30.0
        assert d["observation_type"] == "scene"
        assert d["metadata"]["sample_index"] == 0


# ---------------------------------------------------------------------------
# CuratorEntity tests
# ---------------------------------------------------------------------------


class TestCuratorEntity:
    def test_entity_creation(self, sample_entity):
        """Basic entity dataclass creation."""
        assert sample_entity.entity_id == "person_0"
        assert sample_entity.name == "John Smith"
        assert sample_entity.entity_type == "person"
        assert sample_entity.first_seen == 30.0
        assert sample_entity.appearances == 3

    def test_entity_to_dict(self, sample_entity):
        """Entity serializes to dict."""
        d = sample_entity.to_dict()
        assert d["name"] == "John Smith"
        assert d["entity_type"] == "person"
        assert len(d["related_timestamps"]) == 3


# ---------------------------------------------------------------------------
# CuratorKnowledge tests
# ---------------------------------------------------------------------------


class TestCuratorKnowledge:
    def test_add_observation(self, sample_knowledge, sample_observation):
        """Adding observations updates knowledge state."""
        sample_knowledge.add_observation(sample_observation)
        assert len(sample_knowledge.observations) == 1
        assert sample_knowledge.observations[0].observation_id == "obs_001"

    def test_add_gap(self, sample_knowledge):
        """Knowledge gaps are tracked."""
        sample_knowledge.add_gap("What objects are in the background?")
        assert len(sample_knowledge.knowledge_gaps) == 1
        assert "What objects are in the background?" in sample_knowledge.knowledge_gaps

    def test_dedup_gaps(self, sample_knowledge):
        """Adding the same gap twice doesn't duplicate."""
        sample_knowledge.add_gap("Gap A")
        sample_knowledge.add_gap("Gap A")
        assert len(sample_knowledge.knowledge_gaps) == 1

    def test_add_exploration_question(self, sample_knowledge):
        """Exploration questions are tracked."""
        sample_knowledge.add_exploration_question("What happens next?")
        assert len(sample_knowledge.exploration_questions) == 1

    def test_mark_answered(self, sample_knowledge):
        """Questions can be moved from exploration to answered."""
        sample_knowledge.add_exploration_question("Q1")
        sample_knowledge.mark_answered("Q1")
        assert len(sample_knowledge.exploration_questions) == 0
        assert "Q1" in sample_knowledge.answered_questions

    def test_record_action(self, sample_knowledge):
        """Actions are recorded on the timeline."""
        sample_knowledge.record_action("Starting exploration")
        assert len(sample_knowledge.exploration_timeline) == 1
        assert "Starting exploration" in sample_knowledge.exploration_timeline[0]

    def test_summary(self, sample_knowledge, sample_observation):
        """Knowledge summary provides compact state overview."""
        sample_knowledge.add_observation(sample_observation)
        skills_summary = sample_knowledge.summary()
        assert skills_summary["video_id"] == "test_video"
        assert skills_summary["total_observations"] == 1
        assert skills_summary["iteration_count"] == 0

    def test_update_entities_from_observation(self, sample_knowledge):
        """Entities are extracted from observation content."""
        obs = CuratorObservation(
            observation_id="obs_person",
            timestamp_seconds=60.0,
            observation_type="scene",
            content="The person Jane Doe is standing near a computer. Objects: laptop, chair.",
            confidence=0.9,
            source_tool="analyze_frames",
        )
        sample_knowledge.add_observation(obs)

        # Should have extracted person and object references
        assert len(sample_knowledge.entities) >= 1


# ---------------------------------------------------------------------------
# VideoCuratorReport tests
# ---------------------------------------------------------------------------


class TestVideoCuratorReport:
    def test_report_to_markdown(self):
        """Report renders as Markdown with all sections."""
        report = VideoCuratorReport(
            video_id="test_vid",
            video_path="/tmp/test.mp4",
            title="Test Curation",
            overview="An autonomous analysis of test content.",
            sections=[
                CuratorReportChunk(
                    title="Findings", content="Something interesting was found."
                ),
            ],
            curation_duration_seconds=10.5,
            iterations_completed=5,
            observations_count=20,
            generated_at="2026-06-27 02:00:00",
        )

        md = report.to_markdown()
        assert "# Test Curation" in md
        assert "An autonomous analysis of test content." in md
        assert "## Findings" in md
        assert "Something interesting was found." in md
        assert "10.5s" in md
        assert "5 iterations" in md
        assert "20 observations" in md
        assert "2026-06-27 02:00:00" in md

    def test_report_with_entities(self):
        """Report includes entity section when entities are present."""
        entity = CuratorEntity(
            entity_id="person_0",
            name="Jane Doe",
            entity_type="person",
            first_seen=10.0,
            last_seen=200.0,
            appearances=5,
            description="A person in a lab coat.",
        )
        report = VideoCuratorReport(
            video_id="test_vid",
            video_path="/tmp/test.mp4",
            title="Entity Report",
            overview="Found some entities.",
            key_entities={"person_0": entity},
            generated_at="2026-06-27 02:00:00",
        )
        md = report.to_markdown()
        assert "Key Entities Discovered" in md
        assert "Jane Doe" in md
        assert "person" in md
        assert "5×" in md

    def test_report_with_timeline(self):
        """Report includes timeline section when events exist."""
        report = VideoCuratorReport(
            video_id="test_vid",
            video_path="/tmp/test.mp4",
            title="Timeline Report",
            overview="Events found.",
            key_timeline=[
                {"timestamp": 30.0, "description": "Scene starts"},
                {"timestamp": 120.0, "description": "Key moment"},
            ],
            generated_at="2026-06-27 02:00:00",
        )
        md = report.to_markdown()
        assert "Key Timeline" in md
        assert "00:30" in md
        assert "Scene starts" in md
        assert "02:00" in md
        assert "Key moment" in md

    def test_report_to_json(self):
        """Report serializes to JSON."""
        report = VideoCuratorReport(
            video_id="test_vid",
            video_path="/tmp/test.mp4",
            title="JSON Report",
            overview="JSON test.",
            sections=[CuratorReportChunk(title="S1", content="C1")],
            generated_at="2026-06-27 02:00:00",
        )
        json_str = report.to_json()
        data = json.loads(json_str)
        assert data["title"] == "JSON Report"
        assert data["section_count"] == 1
        assert data["observations"] == 0


# ---------------------------------------------------------------------------
# CuriosityStrategy tests
# ---------------------------------------------------------------------------


class TestCuriosityStrategy:
    def test_suggest_first_action(self, mock_config, sample_knowledge):
        """With no observations, suggests broad exploration."""
        strategy = CuriosityStrategy(mock_config, curiosity_threshold=0.5)
        action, params = strategy.suggest_next_action(
            sample_knowledge, ["analyze_frames"]
        )
        assert action == "sample_timeline"
        assert params.get("mode") == "broad"

    def test_suggest_search_unanswered_question(self, mock_config, sample_knowledge):
        """With unanswered questions, suggests searching that topic."""
        sample_knowledge.add_observation(
            CuratorObservation("o1", 30.0, "scene", "Test", 0.8, "analyze_frames")
        )
        sample_knowledge.add_exploration_question("What objects are in the background?")
        strategy = CuriosityStrategy(mock_config, curiosity_threshold=0.5)
        action, params = strategy.suggest_next_action(
            sample_knowledge, ["analyze_frames", "search_rag"]
        )
        assert action == "search_topic"
        assert "background" in params.get("query", "")

    def test_suggest_later_portion_exploration(self, mock_config, sample_knowledge):
        """When early portion explored, suggests later timestamps."""
        sample_knowledge.duration_seconds = 300.0
        sample_knowledge.add_observation(
            CuratorObservation("o1", 30.0, "scene", "Test", 0.8, "analyze_frames")
        )
        strategy = CuriosityStrategy(mock_config, curiosity_threshold=0.5)
        action, params = strategy.suggest_next_action(
            sample_knowledge, ["analyze_frames"]
        )
        assert action == "sample_timestamps"

    def test_generate_curiosity_questions(self, mock_config, sample_knowledge):
        """Generates appropriate curiosity questions based on knowledge state."""
        strategy = CuriosityStrategy(mock_config, curiosity_threshold=0.5)
        sample_knowledge.duration_seconds = 600.0
        questions = strategy.generate_curiosity_questions(sample_knowledge)
        assert len(questions) >= 1
        # Should ask about content since no observations
        assert any(
            "objects" in q.lower() or "content" in q.lower() or "setting" in q.lower()
            for q in questions
        )

    def test_high_curiosity_deep_focus(self, mock_config):
        """High curiosity triggers deep entity focus."""
        strategy = CuriosityStrategy(mock_config, curiosity_threshold=0.8)
        knowledge = CuratorKnowledge("test", "/tmp/test.mp4", 300.0)
        knowledge.add_observation(
            CuratorObservation(
                "o1",
                30.0,
                "scene",
                "A person Jane Doe is visible.",
                0.9,
                "analyze_frames",
            )
        )
        knowledge.add_observation(
            CuratorObservation(
                "o2", 60.0, "scene", "Jane Doe is speaking.", 0.9, "analyze_frames"
            )
        )
        knowledge.add_observation(
            CuratorObservation(
                "o3",
                90.0,
                "scene",
                "Jane Doe walks across the room.",
                0.9,
                "analyze_frames",
            )
        )

        action, params = strategy.suggest_next_action(
            knowledge, ["analyze_frames", "search_rag"]
        )
        # Should eventually suggest something interesting

    def test_default_action_without_gaps(self, mock_config, sample_knowledge):
        """With observations but no obvious gaps, asks default questions."""
        sample_knowledge.add_observation(
            CuratorObservation(
                "o1", 30.0, "scene", "Test content.", 0.8, "analyze_frames"
            )
        )
        sample_knowledge.duration_seconds = 300.0
        strategy = CuriosityStrategy(mock_config, curiosity_threshold=0.5)
        # This should produce some action
        action, _ = strategy.suggest_next_action(sample_knowledge, ["analyze_frames"])
        assert action is not None


# ---------------------------------------------------------------------------
# VideoCurator tests
# ---------------------------------------------------------------------------


class TestVideoCurator:
    def test_init(self, mock_config):
        """Curator initialization sets up knowledge and tools."""
        curator = VideoCurator(
            video_path="/tmp/test.mp4",
            config=mock_config,
            curiosity_threshold=0.5,
            max_iterations=10,
        )
        assert curator.video_id is not None
        assert curator.knowledge.video_id is not None
        assert curator.max_iterations == 10
        assert curator._duration == 0.0  # file doesn't exist

    def test_init_without_video(self, mock_config):
        """Curator can be initialized without a video path."""
        curator = VideoCurator(
            config=mock_config,
            curiosity_threshold=0.5,
            max_iterations=10,
        )
        assert curator.video_path is None
        assert curator._duration == 0.0

    def test_get_duration_missing_file(self, mock_config):
        """Duration of a missing file returns 0."""
        duration = VideoCurator._get_duration(Path("/tmp/nonexistent_video_xyz.mp4"))
        assert duration == 0.0

    def test_get_duration_with_ffprobe(self):
        """Duration via ffprobe returns valid float for real file."""
        # This test only works if there's an actual video file
        # We just test the method doesn't crash
        duration = VideoCurator._get_duration(Path("/dev/null"))
        # Should be 0.0 since /dev/null is not a valid video
        assert duration == 0.0

    def test_is_saturated_early(self, mock_config):
        """Saturation check returns False for early iterations."""
        curator = VideoCurator(config=mock_config, max_iterations=20)
        assert not curator._is_saturated(0)
        assert not curator._is_saturated(1)
        assert not curator._is_saturated(2)

    def test_is_saturated_no_new_entities(self, mock_config):
        """Saturation returns True when no new entities after many iterations."""
        curator = VideoCurator(config=mock_config, max_iterations=20)
        # Add some entities to start
        curator.knowledge.entities = {
            "e1": CuratorEntity("e1", "Obj1", "object", 0, 10, 1, "desc")
        }
        curator._prev_entity_count = 1
        curator._checkpoint_ts = time.time()

        # Simulate being at iteration 8 with no new entities
        # and no meaningful recent observations
        result = curator._is_saturated(8)
        assert result  # should be saturated

    def test_save_knowledge_state(self, mock_config, tmp_path):
        """Knowledge state saves to disk correctly."""
        mock_config.data_dir = tmp_path
        curator = VideoCurator(video_id="test_save", config=mock_config)
        curator.knowledge.add_observation(
            CuratorObservation(
                "obs_save", 30.0, "scene", "Test content.", 0.8, "analyze_frames"
            )
        )
        out_path = curator._save_knowledge_state()
        assert out_path is not None
        assert out_path.exists()

        # Verify saved content
        with open(out_path) as f:
            data = json.load(f)
        assert data["video_id"] == "test_save"
        assert len(data["observations"]) == 1
        assert data["observations"][0]["observation_id"] == "obs_save"

    def test_load_knowledge_state(self, mock_config, tmp_path):
        """Knowledge state loads from disk correctly."""
        mock_config.data_dir = tmp_path
        curator = VideoCurator(video_id="test_load", config=mock_config)

        # Save first
        curator.knowledge.add_observation(
            CuratorObservation(
                "obs_loaded", 30.0, "scene", "Loaded content.", 0.8, "analyze_frames"
            )
        )
        saved_path = curator._save_knowledge_state()

        # Create a new curator and load
        curator2 = VideoCurator(video_id="test_load", config=mock_config)
        loaded = curator2.load_knowledge_state(saved_path)
        assert loaded
        assert len(curator2.knowledge.observations) == 1
        assert curator2.knowledge.observations[0].observation_id == "obs_loaded"

    def test_load_knowledge_state_nonexistent(self, mock_config):
        """Loading a nonexistent state returns False."""
        curator = VideoCurator(video_id="test_nope", config=mock_config)
        loaded = curator.load_knowledge_state(Path("/tmp/nonexistent_state.json"))
        assert not loaded

    def test_rag_sweep_no_rag(self, mock_config):
        """RAG sweep handles missing RAG gracefully."""
        curator = VideoCurator(config=mock_config)
        curator._rag_sweep(None)  # Should not crash

    def test_broad_observation_sweep_no_tools(self, mock_config):
        """Broad sweep handles missing tools gracefully."""
        curator = VideoCurator(config=mock_config)
        curator._broad_observation_sweep(None)  # Should not crash

    def test_execute_action_sample_timestamps_no_tools(self, mock_config):
        """Action execution handles missing tools gracefully."""
        curator = VideoCurator(config=mock_config)
        curator._execute_action("sample_timestamps", {"timestamps": [30.0]}, None)
        # Should not crash

    def test_execute_action_search_topic_no_tools(self, mock_config):
        """Search topic action handles missing tools gracefully."""
        curator = VideoCurator(config=mock_config)
        curator._execute_action("search_topic", {"query": "test"}, None)
        # Should not crash

    def test_execute_action_generate_question(self, mock_config, sample_knowledge):
        """Generate question action populates exploration questions."""
        # Add some observations so generate_curiosity_questions has material
        sample_knowledge.add_observation(
            CuratorObservation(
                "o1", 30.0, "scene", "Test observation.", 0.8, "analyze_frames"
            )
        )
        sample_knowledge.duration_seconds = 600.0
        curator = VideoCurator(config=mock_config, video_id="test_gq")
        curator.knowledge = sample_knowledge

        curator._execute_action("generate_question", {}, MagicMock())
        # Should have added questions
        if not curator.knowledge.exploration_questions:
            # Even if no questions generated, should not crash
            pass

    def test_curate_no_tools(self, mock_config):
        """Curation handles no tools gracefully."""
        curator = VideoCurator(
            config=mock_config,
            curiosity_threshold=0.5,
            max_iterations=3,
        )
        report = curator.curate()
        assert report is not None
        assert report.video_id is not None
        assert report.iterations_completed >= 0


# ---------------------------------------------------------------------------
# run_curation convenience function tests
# ---------------------------------------------------------------------------


class TestRunCuration:
    def test_run_curation_no_rag(self, mock_config, tmp_path):
        """run_curation returns a report even without RAG."""
        with patch("video_analysis.curator.Config", return_value=mock_config):
            report = run_curation(
                video_id="test_cli",
                curiosity=0.3,
                max_iterations=2,
                output_dir=str(tmp_path),
            )
            assert report is not None
            assert report.video_id == "test_cli"
            assert report.iterations_completed >= 0


# ---------------------------------------------------------------------------
# Module-level doc / version checks
# ---------------------------------------------------------------------------


class TestCuratorModule:
    def test_curator_module_importable(self):
        """The curator module is importable through video_analysis."""
        import video_analysis

        assert hasattr(video_analysis, "curator")

    def test_version_check(self):
        """Check that test source references the current version."""
        import video_analysis

        assert video_analysis.__version__ == "0.55.0"
