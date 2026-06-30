"""
Federated Video Search — cross-instance MCP-based video retrieval.

Query multiple video-analysis instances (remote peers) simultaneously
and merge the results into a unified, de-duplicated, cross-encoder
re-ranked result set.

Each peer is an independently running video-analysis MCP server (HTTP SSE
transport).  Federation is coordinated by the LOCAL instance, which:

  1. Sends a ``/api/federated/search`` request (or raw MCP tool call)
     to each known peer.
  2. Collects responses from all reachable peers.
  3. De-duplicates by (video_id, chunk_id) — keeps the higher score.
  4. Re-ranks all merged chunks using the local cross-encoder.
  5. Returns the top-K unified results.

Usage (programmatic)::

    from video_analysis.federation import FederatedSearch

    search = FederatedSearch()
    search.add_peer("http://192.168.1.50:8000")
    results = search.query("what objects are visible?")

Usage (config-driven)::

    FEDERATION_ENABLED=true
    FEDERATION_PEERS=http://192.168.1.50:8000,http://192.168.1.51:8000
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

from video_analysis.rag import RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass
class FederatedPeerResult:
    """Result from a single federated peer query."""

    peer_id: str
    peer_url: str
    chunks: List[RetrievedChunk]
    error: Optional[str] = None
    latency_ms: float = 0.0


@dataclass
class FederatedQueryResult:
    """Aggregated result from all queried peers."""

    query: str
    total_chunks: int
    peers_queried: int
    peers_successful: int
    merged_chunks: List[RetrievedChunk]
    peer_results: List[FederatedPeerResult] = field(default_factory=list)


class FederatedSearch:
    """Coordinate query across multiple video-analysis peers.

    Peers are identified by their MCP server URL (HTTP SSE transport).
    Each peer must expose:

      * ``GET /api/federated/search?query=...&top_k=...`` — REST search endpoint
      * Or the MCP tool ``search_videos`` (via the MCP client protocol).

    The local instance acts as the federation orchestrator.
    """

    def __init__(
        self,
        peers: Optional[List[str]] = None,
        rag: Any = None,
        timeout_s: float = 30.0,
    ) -> None:
        """Initialise with an optional list of peer URLs.

        Args:
            peers: List of peer MCP server URLs (e.g. ``http://host:8000``).
            rag: The local VideoRAG instance (used for cross-encoder re-ranking).
            timeout_s: HTTP request timeout per peer (default: 30s).
        """
        self._peers: list[str] = []
        self._rag = rag
        self._timeout = timeout_s

        if peers:
            for p in peers:
                self.add_peer(p)

    # ── Peer Management ────────────────────────────────────────────────

    @property
    def peers(self) -> List[str]:
        """Read-only peer list."""
        return list(self._peers)

    def add_peer(self, url: str) -> None:
        """Register a peer MCP server URL.

        Duplicates are silently ignored.  URL is cleaned (trailing slash
        removed).
        """
        url = url.rstrip("/")
        if url not in self._peers:
            self._peers.append(url)
            logger.info("Federated peer added: %s", url)

    def remove_peer(self, url: str) -> None:
        """Remove a peer by URL."""
        url = url.rstrip("/")
        if url in self._peers:
            self._peers.remove(url)
            logger.info("Federated peer removed: %s", url)

    def clear_peers(self) -> None:
        """Remove all registered peers."""
        self._peers.clear()
        logger.info("Federated peers cleared")

    # ── Query ──────────────────────────────────────────────────────────

    def query(
        self,
        query: str,
        top_k: int = 10,
        include_peers: bool = True,
        include_local: bool = True,
    ) -> FederatedQueryResult:
        """Query all peers and optionally the local index, then merge & re-rank.

        Args:
            query: Natural-language search query.
            top_k: Number of final merged results to return (after re-rank).
            include_peers: If True, query all registered remote peers.
            include_local: If True, also query the local ChromaDB index.

        Returns:
            A ``FederatedQueryResult`` with merged, de-duplicated chunks.
        """
        # 1. Collect results from local + each peer
        all_chunks: list[RetrievedChunk] = []
        peer_results: list[FederatedPeerResult] = []
        peers_successful = 0

        if include_local and self._rag is not None:
            try:
                local_chunks = self._rag.search_all(query=query, top_k=top_k * 2)
                peer_results.append(
                    FederatedPeerResult(
                        peer_id="local",
                        peer_url="local",
                        chunks=local_chunks,
                    )
                )
                all_chunks.extend(local_chunks)
                peers_successful += 1
            except Exception as e:
                logger.warning("Local search failed in federated query: %s", e)
                peer_results.append(
                    FederatedPeerResult(
                        peer_id="local",
                        peer_url="local",
                        chunks=[],
                        error=str(e),
                    )
                )

        if include_peers and self._peers:
            # Run peer queries in parallel via asyncio
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                peer_tasks = [self._query_peer(url, query, top_k) for url in self._peers]
                loop_results = loop.run_until_complete(
                    asyncio.gather(*peer_tasks, return_exceptions=True)
                )
                loop.close()
            except RuntimeError:
                # Fallback: serial execution
                loop_results = []
                for url in self._peers:
                    try:
                        import httpx

                        resp = httpx.get(
                            f"{url}/api/federated/search",
                            params={"query": query, "top_k": top_k * 2},
                            timeout=self._timeout,
                        )
                        resp.raise_for_status()
                        data = resp.json()
                        chunks = [
                            RetrievedChunk(
                                chunk_id=c["chunk_id"],
                                video_id=c["video_id"],
                                text=c["text"],
                                timestamp=c.get("timestamp", 0.0),
                                scene_id=c.get("scene_id", -1),
                                score=c.get("score", 0.0),
                                frame_path=c.get("frame_path"),
                                chunk_type=c.get("chunk_type", "scene"),
                            )
                            for c in data.get("chunks", [])
                        ]
                        loop_results.append(chunks)
                    except Exception as e:
                        logger.warning("Peer %s query failed: %s", url, e)
                        loop_results.append(e)

            for url, result in zip(self._peers, loop_results):
                if isinstance(result, Exception):
                    peer_results.append(
                        FederatedPeerResult(
                            peer_id=url,
                            peer_url=url,
                            chunks=[],
                            error=str(result),
                        )
                    )
                elif isinstance(result, list):
                    peer_results.append(
                        FederatedPeerResult(
                            peer_id=url,
                            peer_url=url,
                            chunks=result,
                        )
                    )
                    all_chunks.extend(result)
                    peers_successful += 1

        # 2. De-duplicate by (video_id, chunk_id) — keep higher score
        seen: dict[tuple[str, str], RetrievedChunk] = {}
        for chunk in all_chunks:
            key = (chunk.video_id, chunk.chunk_id)
            if key not in seen or chunk.score > seen[key].score:
                seen[key] = chunk
        deduped = list(seen.values())

        # 3. Re-rank with local cross-encoder if available
        if self._rag is not None and deduped:
            try:
                deduped = self._rag._rerank(query, deduped, top_k)
            except Exception as e:
                logger.warning("Federated re-rank failed, falling back to sort: %s", e)
                deduped.sort(key=lambda c: c.score, reverse=True)
                deduped = deduped[:top_k]
        else:
            deduped.sort(key=lambda c: c.score, reverse=True)
            deduped = deduped[:top_k]

        return FederatedQueryResult(
            query=query,
            total_chunks=len(seen),
            peers_queried=(1 if include_local and self._rag else 0) + len(self._peers),
            peers_successful=peers_successful,
            merged_chunks=deduped,
            peer_results=peer_results,
        )

    def query_peer_videos(self, peer_url: str, top_k: int = 20) -> List[dict[str, Any]]:
        """Query a single peer for its video library listing.

        Args:
            peer_url: The peer MCP server URL.
            top_k: Max videos to return.

        Returns:
            List of video info dicts, each with ``video_id``, ``filename``,
            ``num_scenes``, ``num_chunks``, ``duration``, ``has_sprite``.
        """
        try:
            import httpx

            resp = httpx.get(
                f"{peer_url}/api/library",
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("videos", [])
        except Exception as e:
            logger.warning("Failed to query peer library at %s: %s", peer_url, e)
            return []

    # ── Internal ───────────────────────────────────────────────────────

    @staticmethod
    async def _query_peer(url: str, query: str, top_k: int) -> list[RetrievedChunk]:
        """Query a single peer's federated search REST endpoint."""
        try:
            import httpx

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
                resp = await client.get(
                    f"{url}/api/federated/search",
                    params={"query": query, "top_k": top_k * 2},
                )
                resp.raise_for_status()
                data = resp.json()
                return [
                    RetrievedChunk(
                        chunk_id=c["chunk_id"],
                        video_id=c["video_id"],
                        text=c["text"],
                        timestamp=c.get("timestamp", 0.0),
                        scene_id=c.get("scene_id", -1),
                        score=c.get("score", 0.0),
                        frame_path=c.get("frame_path"),
                        chunk_type=c.get("chunk_type", "scene"),
                    )
                    for c in data.get("chunks", [])
                ]
        except Exception as e:
            logger.warning("Peer %s query failed: %s", url, e)
            raise


# ── Helper: Build FederatedSearch from Config ──────────────────────────


def create_federated_search(
    peers: Optional[List[str]] = None,
    rag: Any = None,
) -> FederatedSearch:
    """Factory: create a ``FederatedSearch`` with the given peer list and RAG.

    This is the recommended way to instantiate federation in application code.
    """
    return FederatedSearch(peers=peers, rag=rag)
