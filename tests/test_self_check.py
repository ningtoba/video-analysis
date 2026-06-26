"""Tests for the Self-Check + Re-Retrieval module.

Tests cover:
- SelfCheckResult dataclass defaults
- SelfCheckRAG init with various configs
- _build_evidence_text formatting
- _parse_json_response (LLM output parsing)
- _merge_chunks deduplication and scoring
- _reformulate_query fallback behavior
- verify with empty chunks
- Integration with agentic_retrieve from rag
"""

import logging
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from video_analysis.config import Config
from video_analysis.rag import RetrievedChunk

logger = logging.getLogger(__name__)


def test_self_check_result_defaults():
    """Test SelfCheckResult dataclass defaults."""
    from video_analysis.self_check import SelfCheckResult

    result = SelfCheckResult(
        query="test query",
        draft_answer="test answer",
        verdict="supported",
        confidence_score=0.95,
    )
    assert result.query == "test query"
    assert result.draft_answer == "test answer"
    assert result.verdict == "supported"
    assert result.confidence_score == 0.95
    assert result.gaps == []
    assert result.corrected_answer == ""
    assert result.retrieval_rounds == 1
    assert result.total_chunks_used == 0


def test_self_check_result_partial():
    """Test SelfCheckResult with partial verdict and gaps."""
    from video_analysis.self_check import SelfCheckResult

    result = SelfCheckResult(
        query="test query",
        draft_answer="partial answer",
        verdict="partial",
        confidence_score=0.45,
        gaps=["Missing timestamps", "Incomplete coverage"],
        retrieval_rounds=2,
        total_chunks_used=5,
    )
    assert result.verdict == "partial"
    assert len(result.gaps) == 2
    assert result.retrieval_rounds == 2
    assert result.total_chunks_used == 5


def test_self_check_rag_init():
    """Test SelfCheckRAG initialization with various configs."""
    from video_analysis.self_check import SelfCheckRAG

    config = Config(data_dir="/tmp/va_test_selfcheck_init")
    checker = SelfCheckRAG(config=config)
    assert checker.config is not None
    assert checker.rag is None
    assert checker.config.self_check_enabled is True
    assert checker.config.self_check_max_rounds == 2

    import shutil

    shutil.rmtree("/tmp/va_test_selfcheck_init", ignore_errors=True)


def test_build_evidence_text():
    """Test _build_evidence_text formats chunks correctly."""
    from video_analysis.self_check import SelfCheckRAG

    config = Config(data_dir="/tmp/va_test_selfcheck_evid")
    checker = SelfCheckRAG(config=config)

    chunks = [
        RetrievedChunk(
            chunk_id="test_vid_scene_0000",
            video_id="test_vid",
            text="A person is speaking about Python programming.",
            timestamp=10.5,
            scene_id=0,
            score=0.85,
            chunk_type="scene",
        ),
        RetrievedChunk(
            chunk_id="test_vid_scene_0001",
            video_id="test_vid",
            text="The speaker demonstrates a code example.",
            timestamp=30.2,
            scene_id=1,
            score=0.72,
            chunk_type="scene",
        ),
    ]

    text = checker._build_evidence_text(chunks)
    assert "[1]" in text
    assert "[2]" in text
    assert "00:00:10.500" in text
    assert "00:00:30" in text
    assert "Python programming" in text
    assert "code example" in text

    import shutil

    shutil.rmtree("/tmp/va_test_selfcheck_evid", ignore_errors=True)


def test_parse_json_response():
    """Test JSON parsing via LLMProvider's utility method."""
    from video_analysis.llm_provider import HermesProvider
    from video_analysis.self_check import SelfCheckRAG
    from video_analysis.config import Config

    # Use the static method from HermesProvider
    config = Config(data_dir="/tmp/va_test_selfcheck_parse")
    checker = SelfCheckRAG(config=config)

    # Plain JSON
    result = HermesProvider._parse_json(
        '{"draft_answer": "test", "verdict": "supported", "confidence": 0.9, "gaps": []}'
    )
    assert result is not None
    assert result["verdict"] == "supported"
    assert result["draft_answer"] == "test"

    # Markdown code fence
    result = HermesProvider._parse_json(
        '```json\n{"draft_answer": "test2", "verdict": "partial", "confidence": 0.5, "gaps": ["gap1"]}\n```'
    )
    assert result is not None
    assert result["verdict"] == "partial"
    assert result["gaps"] == ["gap1"]

    # With surrounding text
    result = HermesProvider._parse_json(
        'Here is the result:\n{"draft_answer": "test3", "verdict": "unsupported", "confidence": 0.1, "gaps": ["gapA", "gapB"]}\nDone.'
    )
    assert result is not None
    assert result["verdict"] == "unsupported"
    assert len(result["gaps"]) == 2

    # Invalid JSON
    result = HermesProvider._parse_json("Invalid response text")
    assert result is None

    import shutil

    shutil.rmtree("/tmp/va_test_selfcheck_parse", ignore_errors=True)


def test_merge_chunks():
    """Test _merge_chunks deduplicates and boosts new chunks."""
    from video_analysis.self_check import SelfCheckRAG

    config = Config(data_dir="/tmp/va_test_selfcheck_merge")
    checker = SelfCheckRAG(config=config)

    existing = [
        RetrievedChunk(
            chunk_id="vid_scene_0000",
            video_id="vid",
            text="Existing chunk 1",
            timestamp=0.0,
            scene_id=0,
            score=0.8,
        ),
        RetrievedChunk(
            chunk_id="vid_scene_0001",
            video_id="vid",
            text="Existing chunk 2",
            timestamp=10.0,
            scene_id=1,
            score=0.7,
        ),
    ]

    new_chunks = [
        RetrievedChunk(
            chunk_id="vid_scene_0001",
            video_id="vid",
            text="Existing chunk 2 (duplicate)",
            timestamp=10.0,
            scene_id=1,
            score=0.75,  # higher score
        ),
        RetrievedChunk(
            chunk_id="vid_scene_0002",
            video_id="vid",
            text="New chunk 1",
            timestamp=20.0,
            scene_id=2,
            score=0.6,
        ),
    ]

    merged = checker._merge_chunks(existing, new_chunks)
    assert len(merged) == 3  # 2 existing + 1 new (1 duplicate)
    # The duplicate should have the higher score
    dup = [c for c in merged if c.chunk_id == "vid_scene_0001"]
    assert len(dup) == 1
    assert dup[0].score == 0.8  # 0.75 + 0.05 boost from new_chunks
    # New chunk should have slightly boosted score
    new = [c for c in merged if c.chunk_id == "vid_scene_0002"]
    assert len(new) == 1
    assert new[0].score > 0.6  # boosted

    import shutil

    shutil.rmtree("/tmp/va_test_selfcheck_merge", ignore_errors=True)


def test_verify_empty_chunks():
    """Test verify returns unsupported for empty chunk list."""
    from video_analysis.self_check import SelfCheckRAG

    config = Config(data_dir="/tmp/va_test_selfcheck_empty")
    checker = SelfCheckRAG(config=config)

    result = checker.verify("test query", [])
    assert result.verdict == "unsupported"
    assert result.confidence_score == 0.0
    assert len(result.gaps) > 0
    assert result.total_chunks_used == 0

    import shutil

    shutil.rmtree("/tmp/va_test_selfcheck_empty", ignore_errors=True)


def test_reformulate_query_fallback():
    """Test _reformulate_query falls back to original query on failure."""
    from video_analysis.self_check import SelfCheckRAG

    config = Config(data_dir="/tmp/va_test_selfcheck_qfmt")
    checker = SelfCheckRAG(config=config)

    # Without Hermes CLI available, should return original
    result = checker._reformulate_query(
        "What objects are visible?",
        ["Missing color details"],
        "Some objects are visible.",
    )
    # Should return the original query since Hermes won't be available
    assert result == "What objects are visible?"

    import shutil

    shutil.rmtree("/tmp/va_test_selfcheck_qfmt", ignore_errors=True)


def test_config_self_check_fields():
    """Test self-check config fields exist and have correct defaults."""
    config = Config(data_dir="/tmp/va_test_selfcheck_cfg")
    assert hasattr(config, "self_check_enabled")
    assert hasattr(config, "self_check_max_rounds")
    assert hasattr(config, "self_check_min_confidence")
    assert config.self_check_enabled is True
    assert config.self_check_max_rounds == 2
    assert config.self_check_min_confidence == 0.7
    assert config.agentic_max_rounds == 4  # updated to include self-check

    import shutil

    shutil.rmtree("/tmp/va_test_selfcheck_cfg", ignore_errors=True)


def test_self_check_importable():
    """Test that the self_check module can be imported cleanly."""
    from video_analysis.self_check import SelfCheckRAG, SelfCheckResult

    assert callable(SelfCheckRAG)
    assert SelfCheckResult.__name__ == "SelfCheckResult"


def test_raises_no_deps():
    """Test SelfCheckRAG init doesn't require any heavy deps."""
    from video_analysis.self_check import SelfCheckRAG

    config = Config(data_dir="/tmp/va_test_selfcheck_nodeps")
    checker = SelfCheckRAG(config=config)
    assert checker is not None
    assert checker.config.llm_model is not None

    import shutil

    shutil.rmtree("/tmp/va_test_selfcheck_nodeps", ignore_errors=True)
