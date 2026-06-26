"""
MCP Tool Server — Expose video-analysis pipeline stages as MCP tools.

Provides a Model Context Protocol (MCP) server that exposes pipeline
operations as tools consumable by Hermes, Claude Code, and other MCP
hosts.  Tools:

  - process_video          Run the full analysis pipeline on a video file.
  - search_videos          Semantic search across indexed video library.
  - ask_question           Ask a natural-language question about a video.
  - extract_scenes         List detected scenes for a video.
  - detect_objects         Run YOLO object detection on a video.
  - list_library           Show indexed videos in the library.
  - delete_video           Remove a video from the library.

Usage (standalone server)::

    python -m video_analysis.mcp_server

Usage (stdio transport — for Hermes / Claude Code)::

    python -m video_analysis.mcp_server --stdio

Usage (HTTP SSE transport)::

    python -m video_analysis.mcp_server --port 8081
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from video_analysis.config import Config
from video_analysis.pipeline import VideoPipeline
from video_analysis.streaming import StreamingPipeline, StreamingChunkResult
from video_analysis.rag import VideoRAG
from video_analysis.chat import VideoChat
from video_analysis.federation import FederatedSearch

logger = logging.getLogger(__name__)

# ── Globals (lazy-initialized per tool call) ──────────────────────────

_config: Config | None = None
_pipeline: VideoPipeline | None = None
_rag: VideoRAG | None = None
_chat: VideoChat | None = None


def _ensure_services() -> tuple[VideoPipeline, VideoRAG, VideoChat]:
    """Lazy-init singletons so import alone doesn't load models."""
    global _config, _pipeline, _rag, _chat
    if _pipeline is None:
        _config = Config()
        _pipeline = VideoPipeline(_config)
        _rag = VideoRAG(_config)
        _chat = VideoChat(_rag, _config)
    assert _config is not None
    assert _pipeline is not None
    assert _rag is not None
    assert _chat is not None
    return _pipeline, _rag, _chat


# ── MCP Server ────────────────────────────────────────────────────────

mcp = FastMCP("video-analysis")


@mcp.tool(
    description="Run the full analysis pipeline on a video file (or YouTube URL). Returns a summary of the analysis results."
)
async def process_video(
    video: str,
    url: bool = False,
    processing_mode: str = "video_full",
) -> str:
    """Process a video file through the full analysis pipeline.

    Args:
        video: Path to a local video file, or YouTube URL when ``url=True``.
        url: When True, download the video from the given URL first.
        processing_mode: ``"video_full"``, ``"audio_only"``, or ``"auto"``.
    """
    pipeline, rag, _ = _ensure_services()

    # Override processing mode temporarily
    import os as _os

    _prev_mode = _os.environ.get("PROCESSING_MODE", "")
    _os.environ["PROCESSING_MODE"] = processing_mode

    try:
        path: str | Path
        if url:
            path = pipeline.download_from_url(video, pipeline.config.video_dir)
            if path is None:
                return f"❌ Failed to download: {video}"
        else:
            path = Path(video)
            if not path.exists():
                return f"❌ File not found: {video}"

        index = pipeline.process(str(path))
        rag.index_video(index)
        pipeline.cleanup()

        return json.dumps(
            {
                "video_id": index.video_id,
                "duration_s": round(index.duration, 1),
                "scenes": len(index.scenes),
                "transcript_segments": len(index.transcript),
                "objects_found": (
                    len(
                        {
                            o
                            for s in index.scenes
                            for f in s.key_frames
                            for o in (f.objects or [])
                        }
                    )
                ),
                "processing_mode": processing_mode,
            },
            indent=2,
        )
    finally:
        if _prev_mode:
            _os.environ["PROCESSING_MODE"] = _prev_mode
        else:
            _os.environ.pop("PROCESSING_MODE", None)


@mcp.tool(
    description="Semantic search across the indexed video library. Returns relevant scenes with timestamps."
)
async def search_videos(query: str, top_k: int = 5) -> str:
    """Search across all indexed videos using semantic (vector) search.

    Args:
        query: Natural-language search query.
        top_k: Number of top results to return (default: 5, max: 20).
    """
    _, rag, _ = _ensure_services()

    results = rag.search_all(query, top_k=min(top_k, 20))

    if not results:
        return "No results found."

    lines = [f"Found {len(results)} results for: {query}\n"]
    for r in results:
        lines.append(f"  [{r.video_id} @ {r.timestamp:.1f}s] {r.text[:180]}")
    return "\n".join(lines)


@mcp.tool(
    description="Ask a natural-language question about a specific video. Returns answer with timestamp citations."
)
async def ask_question(question: str, video_id: Optional[str] = None) -> str:
    """Ask a question about video content. Optionally scope to one video.

    Args:
        question: Natural-language question about the video content.
        video_id: Optional video ID to narrow the context.
    """
    _, _, chat = _ensure_services()

    response = chat.ask(question, video_id=video_id)
    parts = [f"Answer: {response.content}"]

    if response.sources:
        from video_analysis.models import format_timestamp

        parts.append(f"\nSources ({len(response.sources)}):")
        for s in response.sources[:5]:
            parts.append(
                f"  [{format_timestamp(s.timestamp)}] " f"[{s.video_id}] {s.text[:120]}"
            )

    return "\n".join(parts)


@mcp.tool(
    description="List detected scenes for a video by video_id. Includes scene timestamps and descriptions."
)
async def extract_scenes(video_id: str) -> str:
    """Retrieve scene-level metadata for a video.

    Args:
        video_id: The video ID to query (use ``list_library`` first).
    """
    _, rag, _ = _ensure_services()

    try:
        results = rag.collection.get(
            where={"video_id": video_id},
            limit=500,
        )
    except Exception:
        return f"No scenes found for video_id: {video_id}"

    if not results or not results.get("metadatas"):
        return f"No scenes found for video_id: {video_id}"

    lines = [f"Scenes for {video_id}:\n"]
    for i, md in enumerate(results["metadatas"]):
        ts = md.get("timestamp", 0)
        text = (md.get("text", "") or "")[:150]
        chunk_type = md.get("chunk_type", "scene")
        lines.append(f"  [{i}] @ {ts:.1f}s [{chunk_type}] {text}")

    return "\n".join(lines)


@mcp.tool(
    description="Run YOLO object detection on a video file and return detected objects by scene."
)
async def detect_objects(video_path: str) -> str:
    """Run YOLO object detection on a video file.

    Args:
        video_path: Path to a local video file.
    """
    pipeline, _, _ = _ensure_services()
    path = Path(video_path)
    if not path.exists():
        return f"❌ File not found: {video_path}"

    index = pipeline.process(str(path))

    lines = [f"Objects detected in {video_path}:\n"]
    for scene in index.scenes:
        for frame in scene.key_frames:
            if frame.objects:
                objects_str = ", ".join(
                    f"{o.get('label', '?')} ({o.get('confidence', 0):.0%})"
                    for o in frame.objects
                )
                lines.append(f"  @ {scene.start:.1f}s — {objects_str}")

    pipeline.cleanup()
    return "\n".join(lines) if len(lines) > 1 else "No objects detected."


@mcp.tool(
    description="List all indexed videos in the library with basic metadata (duration, scenes, timestamp)."
)
async def list_library() -> str:
    """List all videos that have been indexed."""
    _, rag, _ = _ensure_services()

    try:
        count = rag.collection.count()
    except Exception:
        return "RAG index is not initialized."

    if count == 0:
        return "Library is empty."

    # Deduplicate by video_id
    results = rag.collection.get(limit=count)
    videos: dict[str, dict[str, Any]] = {}
    for md in results.get("metadatas") or []:
        vid = md.get("video_id", "?")
        if vid not in videos:
            videos[vid] = {
                "video_id": vid,
                "scenes": 0,
                "first_seen": md.get("timestamp", 0),
            }
        videos[vid]["scenes"] += 1

    lines = [f"Library: {len(videos)} videos\n"]
    for v in sorted(videos.values(), key=lambda x: x["first_seen"]):
        lines.append(f"  {v['video_id']} — {v['scenes']} scenes")
    return "\n".join(lines)


@mcp.tool(description="Delete a video and its index from the library.")
async def delete_video(video_id: str) -> str:
    """Remove a video from the ChromaDB index.

    Args:
        video_id: The video ID to delete.
    """
    _, rag, _ = _ensure_services()

    try:
        count_before = rag.collection.count()
    except Exception:
        return "RAG index is not initialized."

    rag.collection.delete(where={"video_id": video_id})
    count_after = rag.collection.count()
    deleted = count_before - count_after
    return (
        f"✅ Deleted {deleted} entries for video_id: {video_id}"
        if deleted > 0
        else f"⚠️ No entries found for video_id: {video_id}"
    )


@mcp.tool(
    description="Process a video in streaming chunks (reduced latency to first result). "
    "Yields incremental results as each chunk is processed, rather than waiting for the full video."
)
async def stream_video(
    video: str,
    chunk_duration: float = 30.0,
    incremental_index: bool = True,
) -> str:
    """Process a video file in streaming chunks.

    Args:
        video: Path to a local video file.
        chunk_duration: Seconds per chunk (default: 30.0).
        incremental_index: If True, index each chunk to ChromaDB as it's processed.
    """
    path = Path(video)
    if not path.exists():
        return f"❌ File not found: {video}"

    pipeline = StreamingPipeline()

    results = []
    # Consume the generator and collect results
    try:
        for result in pipeline.process_streaming(
            video,
            chunk_duration=chunk_duration,
            incremental_index=incremental_index,
        ):
            results.append(
                {
                    "chunk_index": result.chunk_index,
                    "start_time": round(result.start_time, 1),
                    "end_time": round(result.end_time, 1),
                    "duration": round(result.duration, 1),
                    "scenes": len(result.scenes),
                    "transcript_segments": len(result.transcript_segments),
                    "objects_found": result.objects_found,
                }
            )
    except Exception as e:
        return f"❌ Streaming failed: {e}"

    stats = pipeline.stats
    return json.dumps(
        {
            "status": "completed",
            "chunks": results,
            "stats": {
                "chunks_processed": stats["chunks_processed"],
                "total_scenes": stats["total_scenes"],
                "total_transcript_segments": stats["total_transcript_segments"],
                "unique_objects": stats["unique_objects"],
            },
        },
        indent=2,
    )


@mcp.tool(
    description="Watch a recording file being written and process it live in streaming chunks. "
    "Useful for monitoring OBS recordings or other live video sources."
)
async def watch_video(
    source: str,
    chunk_duration: float = 10.0,
    incremental_index: bool = True,
    max_chunks: int = 5,
) -> str:
    """Watch a live recording source and process chunks incrementally.

    Args:
        source: Path to a file being written (e.g. OBS recording).
        chunk_duration: Seconds per processing chunk (default: 10.0).
        incremental_index: If True, index incrementally.
        max_chunks: Maximum chunks to process before stopping (default: 5).
    """
    path = Path(source)
    if not path.exists():
        return f"❌ Source not found: {source}"

    pipeline = StreamingPipeline()

    results = []
    chunk_count = 0
    try:
        for result in pipeline.process_live(
            source,
            chunk_duration=chunk_duration,
            incremental_index=incremental_index,
            poll_interval=1.0,
        ):
            results.append(
                {
                    "chunk_index": result.chunk_index,
                    "start_time": round(result.start_time, 1),
                    "end_time": round(result.end_time, 1),
                    "duration": round(result.duration, 1),
                    "scenes": len(result.scenes),
                    "transcript_segments": len(result.transcript_segments),
                    "objects_found": result.objects_found,
                }
            )
            chunk_count += 1
            if chunk_count >= max_chunks:
                break
    except Exception as e:
        return f"❌ Watch failed: {e}"

    stats = pipeline.stats
    return json.dumps(
        {
            "status": "completed",
            "chunks_watched": chunk_count,
            "chunks": results,
            "stats": {
                "chunks_processed": stats["chunks_processed"],
                "total_scenes": stats["total_scenes"],
                "total_transcript_segments": stats["total_transcript_segments"],
                "unique_objects": stats["unique_objects"],
            },
        },
        indent=2,
    )


# ── Federated Video Search Tools (v0.33.0) ───────────────────────────


@mcp.tool(
    description="Query multiple video-analysis instances and merge results. "
    "Searches local index + all configured remote peers, de-duplicates, "
    "and re-ranks via cross-encoder."
)
async def federated_search(
    query: str,
    top_k: int = 10,
    include_peers: bool = True,
    include_local: bool = True,
) -> str:
    """Federated query across all configured video-analysis peers.

    Args:
        query: Natural-language search query.
        top_k: Number of final merged results to return (default: 10).
        include_peers: If True, query all registered remote peers (default: True).
        include_local: If True, include results from the local index (default: True).
    """
    _, rag, _ = _ensure_services()

    # Build peer list from config
    config = _config
    peers_list: list[str] = []
    if config is not None and config.federation_peers:
        peers_list = [
            p.strip() for p in config.federation_peers.split(",") if p.strip()
        ]

    search = FederatedSearch(peers=peers_list, rag=rag)
    result = search.query(
        query=query,
        top_k=top_k,
        include_peers=include_peers,
        include_local=include_local,
    )

    if not result.merged_chunks:
        return json.dumps(
            {
                "query": query,
                "total_chunks": 0,
                "peers_queried": result.peers_queried,
                "peers_successful": result.peers_successful,
                "chunks": [],
            },
            indent=2,
        )

    return json.dumps(
        {
            "query": query,
            "total_chunks": result.total_chunks,
            "peers_queried": result.peers_queried,
            "peers_successful": result.peers_successful,
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "video_id": c.video_id,
                    "text": c.text[:300],
                    "timestamp": round(c.timestamp, 1),
                    "scene_id": c.scene_id,
                    "score": round(c.score, 4),
                    "frame_path": c.frame_path,
                    "chunk_type": c.chunk_type,
                }
                for c in result.merged_chunks
            ],
        },
        indent=2,
    )


@mcp.tool(description="Register a remote video-analysis peer for federated search.")
async def add_federation_peer(peer_url: str) -> str:
    """Register a peer for federated search.

    Args:
        peer_url: The peer MCP server URL (e.g. ``http://192.168.1.50:8000``).
    """
    _, rag, _ = _ensure_services()

    search = FederatedSearch(rag=rag)
    search.add_peer(peer_url)
    return json.dumps(
        {"status": "ok", "peer_url": peer_url, "total_peers": len(search.peers)},
        indent=2,
    )


@mcp.tool(description="List all registered federation peers.")
async def list_federation_peers() -> str:
    """List all currently registered federation peers."""
    _, rag, _ = _ensure_services()

    search = FederatedSearch(rag=rag)
    peers = search.peers
    return json.dumps(
        {"peers": peers, "count": len(peers)},
        indent=2,
    )


# ── Entry point ───────────────────────────────────────────────────────


def main() -> None:
    """Run the MCP server.

    Without arguments, starts on HTTP SSE transport at port 8000.
    Use ``--stdio`` for Hermes / Claude Code integration.
    Use ``--port N`` to specify an HTTP port.
    """
    import argparse

    parser = argparse.ArgumentParser(description="video-analysis MCP server")
    parser.add_argument("--stdio", action="store_true", help="Use stdio transport")
    parser.add_argument(
        "--port", type=int, default=8000, help="HTTP SSE port (default: 8000)"
    )
    args = parser.parse_args()

    if args.stdio:
        mcp.run(transport="stdio")
    else:
        print(
            f"Starting video-analysis MCP server on http://0.0.0.0:{args.port}",
            file=sys.stderr,
        )
        mcp.run(transport="sse", host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
