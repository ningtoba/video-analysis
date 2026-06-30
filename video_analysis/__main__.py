#!/usr/bin/env python3
"""
Video Analysis Platform — Entry point.

Usage:
    python -m video_analysis                          # Launch the web UI
    python -m video_analysis --cli --video FILE       # CLI mode (single video)
    python -m video_analysis --url URL                # Download & process YouTube URL
    python -m video_analysis --batch urls.txt         # Batch process from URL list
"""

import argparse
import logging
import signal
import threading
from pathlib import Path

from video_analysis.chat import VideoChat
from video_analysis.config import Config
from video_analysis.logging_setup import setup_logging as setup_structlog
from video_analysis.pipeline import VideoPipeline
from video_analysis.rag import VideoRAG

# Streaming pipeline (v0.32.0)
from video_analysis.streaming import StreamingPipeline

# Global shutdown event for graceful termination
_shutdown_event = threading.Event()


def _signal_handler(signum, frame):
    """Handle SIGTERM/SIGINT for graceful shutdown."""
    signal_name = signal.Signals(signum).name
    print(f"\n[{signal_name}] Shutting down gracefully... (press Ctrl+C again to force)")
    _shutdown_event.set()


def setup_logging(verbose: bool = False):
    """Configure logging for the application.

    Uses structlog-based structured logging by default, with fallback to
    stdlib logging if structured logging is disabled in config.
    """
    config = Config()
    if config.structured_logging_enabled:
        level = "DEBUG" if verbose else config.structured_logging_level
        setup_structlog(level=level, fmt=config.structured_logging_format)
    else:
        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )


def cli_mode(args):
    """CLI mode: process a video and optionally run Q&A."""
    config = Config()
    pipeline = VideoPipeline(config)
    rag = VideoRAG(config)

    if args.url:
        print(f"Downloading from URL: {args.url}")
        path = VideoPipeline.download_from_url(args.url, config.video_dir)
        if not path:
            print("❌ Download failed")
            return
        print(f"Downloaded: {path}")
        args.video = str(path)

    if not args.video:
        print("No video to process. Use --video or --url.")
        return

    print(f"Processing: {args.video}")
    index = pipeline.process(args.video)
    print(f"  Duration: {index.duration:.1f}s")
    print(f"  Scenes: {len(index.scenes)}")
    print(f"  Transcript: {len(index.transcript)} segments")

    if args.no_index:
        print("Skipping index (--no-index)")
        return

    rag.index_video(index)
    print("  Indexed: ready for questions")

    if args.query:
        chat = VideoChat(rag, config)
        response = chat.ask(args.query, video_id=index.video_id)
        print(f"\nQ: {args.query}")
        print(f"A: {response.content}")
        if response.sources:
            print(f"Sources: {len(response.sources)}")
            from video_analysis.models import format_timestamp

            for s in response.sources[:3]:
                print(f"  [{format_timestamp(s.timestamp)}] {s.text[:100]}...")
    elif args.batch:
        batch_process(args.batch, config, pipeline, rag)


def batch_process(manifest: str, config: Config, pipeline: VideoPipeline, rag: VideoRAG):
    """Process multiple videos from a manifest file."""
    manifest_path = Path(manifest)
    if not manifest_path.exists():
        print(f"❌ Manifest not found: {manifest}")
        return

    urls = [
        line.strip()
        for line in manifest_path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    print(f"Batch processing {len(urls)} videos...")

    for i, url in enumerate(urls):
        print(f"\n[{i + 1}/{len(urls)}] {'URL' if '://' in url else 'File'}: {url}")
        try:
            if "://" in url:
                path = VideoPipeline.download_from_url(url, config.video_dir)
                if not path:
                    print("  ❌ Download failed, skipping")
                    continue
            else:
                path = Path(url)
                if not path.exists():
                    print("  ❌ File not found, skipping")
                    continue

            index = pipeline.process(str(path))
            rag.index_video(index)
            print(
                f"  ✅ Processed: {index.video_id} ({index.duration:.0f}s, {len(index.scenes)} scenes)"
            )
        except Exception as e:
            print(f"  ❌ Error: {e}")


def url_mode(args):
    """Download a YouTube URL and process it."""
    config = Config()
    pipeline = VideoPipeline(config)
    rag = VideoRAG(config)

    print(f"Downloading from URL: {args.url}")
    downloaded = pipeline.download_from_url(args.url, config.video_dir)
    if downloaded is None:
        print("  ❌ Download failed")
        return

    print(f"  Downloaded to: {downloaded}")
    args.video = str(downloaded)
    cli_mode(args)


def batch_mode(args):
    """Batch process multiple videos from a URL list file."""
    config = Config()
    pipeline = VideoPipeline(config)
    rag = VideoRAG(config)

    with open(args.batch) as f:
        urls_or_paths = [line.strip() for line in f if line.strip()]

    print(f"Batch processing {len(urls_or_paths)} items...")
    success = 0
    for i, item in enumerate(urls_or_paths):
        print(f"\n[{i + 1}/{len(urls_or_paths)}] {item[:80]}...")
        try:
            if item.startswith(("http://", "https://")):
                downloaded = pipeline.download_from_url(item, config.video_dir)
                if downloaded is None:
                    print("  ❌ Download failed, skipping")
                    continue
                filepath = str(downloaded)
            else:
                filepath = item

            index = pipeline.process(filepath)
            rag.index_video(index)
            print(f"  ✅ {index.video_id}: {len(index.scenes)} scenes, {index.duration:.0f}s")
            success += 1
        except Exception as e:
            print(f"  ❌ Error: {e}")

    print(f"\n✅ Batch complete: {success}/{len(urls_or_paths)} succeeded")


def stream_mode(args):
    """Streaming mode: process video in chunks with incremental results."""
    config = Config()
    pipeline = StreamingPipeline(config)

    if args.live_source:
        # Live stream analysis mode (v0.40.0 — RTMP/RTSP/HLS)
        source_desc = args.live_source[:60] + ("..." if len(args.live_source) > 60 else "")
        print(
            f"📡 Live stream mode: {source_desc} "
            f"(source_type={args.source_type or 'auto'}, "
            f"chunk_duration={args.chunk_duration}s)"
        )
        if args.max_chunks:
            print(f"  Max chunks: {args.max_chunks}")
        chunk_count = 0
        for result in pipeline.process_live_stream(
            args.live_source,
            source_type=args.source_type,
            chunk_duration=args.chunk_duration,
            incremental_index=args.incremental,
        ):
            chunk_count += 1
            print(
                f"  Chunk {result.chunk_index}: "
                f"[{result.start_time:.1f}s - {result.end_time:.1f}s] "
                f"{len(result.scenes)} scenes, {len(result.transcript_segments)} transcript segs, "
                f"{len(result.objects_found)} objects"
            )
            if args.max_chunks and chunk_count >= args.max_chunks:
                print(f"  Reached max_chunks={args.max_chunks}, stopping.")
                break
    elif args.live:
        print(f"🌐 Live mode: watching {args.video} (chunk_duration={args.chunk_duration}s)")
        for result in pipeline.process_live(
            args.video,
            chunk_duration=args.chunk_duration,
            incremental_index=args.incremental,
        ):
            print(
                f"  Chunk {result.chunk_index}: "
                f"[{result.start_time:.1f}s - {result.end_time:.1f}s] "
                f"{len(result.scenes)} scenes, {len(result.transcript_segments)} transcript segs"
            )
    else:
        print(f"📹 Streaming file: {args.video} (chunk_duration={args.chunk_duration}s)")
        for result in pipeline.process_streaming(
            args.video,
            chunk_duration=args.chunk_duration,
            incremental_index=args.incremental,
        ):
            print(
                f"  Chunk {result.chunk_index}: "
                f"[{result.start_time:.1f}s - {result.end_time:.1f}s] "
                f"{len(result.scenes)} scenes, {len(result.transcript_segments)} transcript segs, "
                f"{len(result.objects_found)} objects"
            )

    stats = pipeline.stats
    print("\n✅ Streaming complete:")
    print(f"  Chunks processed: {stats['chunks_processed']}")
    print(f"  Total scenes: {stats['total_scenes']}")
    print(f"  Total transcript segments: {stats['total_transcript_segments']}")
    print(f"  Unique objects: {stats['unique_objects']}")


def main():
    parser = argparse.ArgumentParser(
        description="Video Analysis Platform — analyze videos and chat about their content",
    )
    parser.add_argument("--cli", action="store_true", help="Run in CLI mode instead of UI")
    parser.add_argument("--video", type=str, help="Video file to process")
    parser.add_argument("--url", type=str, help="YouTube URL to download and process")
    parser.add_argument("--batch", type=str, help="Path to file with URLs/paths (one per line)")
    parser.add_argument("--query", type=str, help="Question to ask about the video")
    parser.add_argument("--no-index", action="store_true", help="Skip RAG indexing")
    parser.add_argument("--host", type=str, default=None, help="UI host")
    parser.add_argument("--port", type=int, default=None, help="UI port")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    parser.add_argument(
        "--no-health",
        action="store_true",
        help="Disable health API and run Gradio standalone",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Process video in streaming/chunked mode",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Watch a recording file being written and process it live",
    )
    parser.add_argument(
        "--live-stream",
        dest="live_source",
        type=str,
        default=None,
        help="RTMP/RTSP/HLS live stream URL to capture and analyze (v0.40.0)",
    )
    parser.add_argument(
        "--source-type",
        type=str,
        default=None,
        choices=["rtmp", "rtsp", "hls"],
        help="Stream source type for --live-stream (default: auto-detect from URL)",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=0,
        help="Maximum number of chunks to process (0 = unlimited)",
    )
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=30.0,
        help="Seconds per streaming chunk (default: 30)",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        default=True,
        help="Index each chunk incrementally to ChromaDB (default: True)",
    )

    args = parser.parse_args()

    # Register graceful shutdown handler
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    if args.host or args.port:
        import os

        if args.host:
            os.environ["VIDEO_ANALYSIS_HOST"] = args.host
        if args.port:
            os.environ["VIDEO_ANALYSIS_PORT"] = str(args.port)

    setup_logging(args.verbose)

    if args.stream or args.live or args.live_source:
        if not args.video and not args.live_source:
            parser.error("--video is required in streaming mode, or use --live-stream <URL>")
        stream_mode(args)
    elif args.batch:
        batch_mode(args)
    elif args.url:
        url_mode(args)
    elif args.cli or args.video:
        if not args.video:
            parser.error("--video is required in CLI mode")
        cli_mode(args)
    else:
        import uvicorn

        from ui.server import create_app

        config = Config()
        app = create_app(config)

        uvicorn.run(
            app,
            host=config.ui_host,
            port=config.ui_port,
            log_level="info",
        )


if __name__ == "__main__":
    main()
