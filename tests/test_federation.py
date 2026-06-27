"""Tests for the Federated Video Search module (v0.33.0)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from video_analysis.federation import (
    FederatedSearch,
    FederatedQueryResult,
    FederatedPeerResult,
    create_federated_search,
)
from video_analysis.rag import RetrievedChunk

# ====================================================================
# Fixtures
# ====================================================================


@pytest.fixture
def mock_rag():
    """Return a MagicMock that behaves like VideoRAG."""
    rag = MagicMock()
    rag.search_all.return_value = [
        RetrievedChunk(
            chunk_id="chunk_1",
            video_id="vid_a",
            text="A bridge over troubled water",
            timestamp=10.0,
            scene_id=0,
            score=0.95,
            chunk_type="scene",
        ),
        RetrievedChunk(
            chunk_id="chunk_2",
            video_id="vid_a",
            text="People walking across the bridge",
            timestamp=30.0,
            scene_id=1,
            score=0.88,
            chunk_type="scene",
        ),
    ]

    # Mock the _rerank method to just sort and truncate
    def mock_rerank(query, chunks, top_k):
        chunks.sort(key=lambda c: c.score, reverse=True)
        return chunks[:top_k]

    rag._rerank = mock_rerank
    return rag


@pytest.fixture
def sample_chunks():
    """Return a list of sample RetrievedChunks for peer simulation."""
    return [
        RetrievedChunk(
            chunk_id="remote_c1",
            video_id="vid_b",
            text="A cat sitting on a windowsill",
            timestamp=5.0,
            scene_id=0,
            score=0.92,
            chunk_type="scene",
        ),
        RetrievedChunk(
            chunk_id="remote_c2",
            video_id="vid_b",
            text="The cat jumps down gracefully",
            timestamp=15.0,
            scene_id=1,
            score=0.85,
            chunk_type="scene",
        ),
    ]


# ====================================================================
# FederatedSearch — constructor & peer management
# ====================================================================


class TestFederatedSearchInit:
    def test_empty_init(self):
        """Search with no peers initialises cleanly."""
        fs = FederatedSearch()
        assert fs.peers == []
        assert fs._rag is None

    def test_init_with_peers(self):
        """Peers list is accepted at construction."""
        fs = FederatedSearch(peers=["http://localhost:8000", "http://peer2:8000"])
        assert len(fs.peers) == 2
        assert "http://localhost:8000" in fs.peers
        assert "http://peer2:8000" in fs.peers

    def test_init_with_rag(self, mock_rag):
        """RAG instance is accepted at construction."""
        fs = FederatedSearch(rag=mock_rag)
        assert fs._rag is not None

    def test_create_federated_search(self, mock_rag):
        """Factory function creates a proper instance."""
        fs = create_federated_search(peers=["http://localhost:8000"], rag=mock_rag)
        assert isinstance(fs, FederatedSearch)
        assert len(fs.peers) == 1


class TestFederatedPeerManagement:
    def test_add_peer(self):
        """Adding a peer registers it and strips trailing slashes."""
        fs = FederatedSearch()
        fs.add_peer("http://example.com:8000/")
        assert fs.peers == ["http://example.com:8000"]

    def test_add_duplicate_peer(self):
        """Duplicate peers are silently ignored."""
        fs = FederatedSearch()
        fs.add_peer("http://localhost:8000")
        fs.add_peer("http://localhost:8000")
        assert len(fs.peers) == 1

    def test_remove_peer(self):
        """Removing a peer removes it from the list."""
        fs = FederatedSearch(peers=["http://a:8000", "http://b:8000"])
        fs.remove_peer("http://a:8000")
        assert fs.peers == ["http://b:8000"]

    def test_remove_missing_peer(self):
        """Removing a non-existent peer is a no-op."""
        fs = FederatedSearch(peers=["http://a:8000"])
        fs.remove_peer("http://nonexistent:8000")
        assert len(fs.peers) == 1

    def test_clear_peers(self):
        """Clearing removes all peers."""
        fs = FederatedSearch(peers=["http://a:8000", "http://b:8000"])
        fs.clear_peers()
        assert fs.peers == []


# ====================================================================
# FederatedSearch.query() — local only
# ====================================================================


class TestFederatedQueryLocal:
    def test_query_local_only(self, mock_rag):
        """Query with include_local=True and no peers returns local results."""
        fs = FederatedSearch(rag=mock_rag)
        result = fs.query("bridge water", top_k=5, include_peers=False)

        assert isinstance(result, FederatedQueryResult)
        assert result.query == "bridge water"
        assert result.peers_queried == 1
        assert result.peers_successful == 1
        assert len(result.merged_chunks) == 2
        assert result.merged_chunks[0].video_id == "vid_a"
        assert result.merged_chunks[0].score >= result.merged_chunks[1].score

    def test_query_local_empty(self):
        """Query with no RAG and no peers returns empty result."""
        fs = FederatedSearch()
        result = fs.query("anything", top_k=5, include_local=True, include_peers=False)

        assert isinstance(result, FederatedQueryResult)
        assert result.total_chunks == 0
        assert result.peers_queried == 0
        assert result.merged_chunks == []

    def test_query_local_with_rerank(self, mock_rag):
        """Query re-ranks results using cross-encoder."""
        # Add a lower-scored chunk that would be ranked up by rerank
        mock_rag.search_all.return_value = [
            RetrievedChunk(
                chunk_id="low",
                video_id="vid_a",
                text="Something else entirely",
                timestamp=0.0,
                scene_id=0,
                score=0.5,
                chunk_type="scene",
            ),
            RetrievedChunk(
                chunk_id="high",
                video_id="vid_a",
                text="Bridge crossing in the rain",
                timestamp=20.0,
                scene_id=2,
                score=0.99,
                chunk_type="scene",
            ),
        ]
        fs = FederatedSearch(rag=mock_rag)
        result = fs.query("bridge", top_k=1, include_peers=False)
        assert len(result.merged_chunks) == 1


# ====================================================================
# FederatedSearch.query() — deduplication
# ====================================================================


class TestFederatedDeduplication:
    def test_dedup_keeps_highest_score(self, mock_rag):
        """Duplicate (video_id, chunk_id) keeps the higher score."""
        fs = FederatedSearch(rag=mock_rag)

        # Simulate local + peer returning the same chunk with different scores
        local_chunks = [
            RetrievedChunk(
                chunk_id="dup_c1",
                video_id="vid_x",
                text="Some text",
                timestamp=0.0,
                scene_id=0,
                score=0.7,
                chunk_type="scene",
            ),
        ]
        peer_chunks = [
            RetrievedChunk(
                chunk_id="dup_c1",
                video_id="vid_x",
                text="Some text",
                timestamp=0.0,
                scene_id=0,
                score=0.95,
                chunk_type="scene",
            ),
        ]
        mock_rag.search_all.return_value = local_chunks
        fs._peers = ["http://test-peer:8000"]

        # We need to mock the HTTP call. Let's patch _query_peer.
        with patch.object(
            FederatedSearch, "_query_peer", return_value=peer_chunks
        ) as mock_query:
            result = fs.query("test", top_k=10)

        assert result.total_chunks == 1
        assert result.merged_chunks[0].score == 0.95  # higher score kept

    def test_no_duplicates_across_sources(self, mock_rag, sample_chunks):
        """Distinct chunks from different sources are all preserved."""
        local_chunks = [
            RetrievedChunk(
                chunk_id="local_only",
                video_id="vid_a",
                text="Local content",
                timestamp=0.0,
                scene_id=0,
                score=0.9,
                chunk_type="scene",
            ),
        ]
        mock_rag.search_all.return_value = local_chunks + sample_chunks

        fs = FederatedSearch(rag=mock_rag)
        # All chunks returned from local search only (no peers needed for this test)
        result = fs.query("test", top_k=10, include_peers=False)

        assert result.total_chunks == 3


# ====================================================================
# FederatedQueryResult model
# ====================================================================


class TestFederatedQueryResult:
    def test_result_model_fields(self):
        """FederatedQueryResult has all expected fields."""
        result = FederatedQueryResult(
            query="test query",
            total_chunks=5,
            peers_queried=3,
            peers_successful=2,
            merged_chunks=[],
        )
        assert result.query == "test query"
        assert result.total_chunks == 5
        assert result.peers_queried == 3
        assert result.peers_successful == 2

    def test_peer_result_error_field(self):
        """FederatedPeerResult stores error messages."""
        pr = FederatedPeerResult(
            peer_id="peer1",
            peer_url="http://peer1:8000",
            chunks=[],
            error="Connection refused",
            latency_ms=100.0,
        )
        assert pr.error == "Connection refused"
        assert pr.latency_ms == 100.0


# ====================================================================
# Config integration
# ====================================================================


class TestFederationConfig:
    def test_config_federation_fields(self):
        """Config has federation fields with defaults."""
        from video_analysis.config import Config

        cfg = Config(data_dir="/tmp/test_fed_config")
        assert cfg.federation_enabled is False
        assert cfg.federation_peers == ""
        assert cfg.federation_timeout == 30.0
        assert cfg.federation_include_local is True

    def test_config_federation_env_overrides(self, monkeypatch):
        """Env vars override federation config defaults."""
        monkeypatch.setenv("FEDERATION_ENABLED", "true")
        monkeypatch.setenv("FEDERATION_PEERS", "http://peer1:8000,http://peer2:8000")
        monkeypatch.setenv("FEDERATION_TIMEOUT", "60")
        monkeypatch.setenv("FEDERATION_INCLUDE_LOCAL", "false")

        from video_analysis.config import Config

        cfg = Config(data_dir="/tmp/test_fed_env")
        assert cfg.federation_enabled is True
        assert cfg.federation_peers == "http://peer1:8000,http://peer2:8000"
        assert cfg.federation_timeout == 60.0
        assert cfg.federation_include_local is False

    def test_config_federation_invalid_timeout(self, monkeypatch):
        """Invalid FEDERATION_TIMEOUT keeps the default."""
        monkeypatch.setenv("FEDERATION_TIMEOUT", "not_a_number")

        from video_analysis.config import Config

        cfg = Config(data_dir="/tmp/test_fed_bad_timeout")
        assert cfg.federation_timeout == 30.0  # default preserved


# ====================================================================
# Module import
# ====================================================================


class TestFederationModule:
    def test_module_import(self):
        """The federation module can be imported."""
        from video_analysis import federation

        assert hasattr(federation, "FederatedSearch")
        assert hasattr(federation, "FederatedQueryResult")
        assert hasattr(federation, "create_federated_search")

    def test_version_bumped(self):
        """Verify version is bumped to 0.33.0."""
        from video_analysis import __version__

        assert __version__ == "0.51.0"
