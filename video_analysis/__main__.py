#!/usr/bin/env python3
"""
Video Analysis Platform — Entry point.

Usage:
    python -m video_analysis          # Launch the web UI
    python -m video_analysis --cli    # CLI mode (batch process)
"""

import argparse
import logging
import sys

from video_analysis.config import Config
from video_analysis.pipeline import VideoPipeline
from video_analysis.rag import VideoRAG
from video_analysis.chat import VideoChat


def setup_logging(verbose: bool = False):
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

    print(f"Processing: {args.video}")
    index = pipeline.process(args.video)
    print(f"  Duration: {index.duration:.1f}s")
    print(f"  Scenes: {len(index.scenes)}")
    print(f"  Transcript: {len(index.transcript)} segments")

    if args.no_index:
        print("Skipping index (--no-index)")
        return

    rag.index_video(index)
    print(f"  Indexed: ready for questions")

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


def main():
    parser = argparse.ArgumentParser(
        description="Video Analysis Platform — analyze videos and chat about their content",
    )
    parser.add_argument(
        "--cli", action="store_true", help="Run in CLI mode instead of UI"
    )
    parser.add_argument("--video", type=str, help="Video file to process")
    parser.add_argument("--query", type=str, help="Question to ask about the video")
    parser.add_argument("--no-index", action="store_true", help="Skip RAG indexing")
    parser.add_argument("--host", type=str, default=None, help="UI host")
    parser.add_argument("--port", type=int, default=None, help="UI port")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.host or args.port:
        import os

        if args.host:
            os.environ["VIDEO_ANALYSIS_HOST"] = args.host
        if args.port:
            os.environ["VIDEO_ANALYSIS_PORT"] = str(args.port)

    setup_logging(args.verbose)

    if args.cli or args.video:
        if not args.video:
            parser.error("--video is required in CLI mode")
        cli_mode(args)
    else:
        from ui.app import launch

        launch()


if __name__ == "__main__":
    main()
