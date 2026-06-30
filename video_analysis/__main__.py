"""
Entry point for the video analysis platform.

Usage:
    python -m video_analysis                    # Start web UI
    python -m video_analysis --cli --video <file>  # CLI mode
    python -m video_analysis --url <youtube-url>   # Download + process
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
from pathlib import Path

from video_analysis.config import Config
from video_analysis.pipeline import VideoPipeline

_shutdown_event = threading.Event()

logger = logging.getLogger(__name__)


def _signal_handler(signum, frame):
    _shutdown_event.set()


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cli_mode(args):
    """CLI mode: process a video and optionally ask a question."""
    config = Config()
    pipeline = VideoPipeline(config)

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"❌ Video not found: {args.video}")
        sys.exit(1)

    print(f"Processing: {video_path.name}")
    analysis = pipeline.process_video(str(video_path), skip_llm_vision=args.no_vision)

    if analysis.error:
        print(f"❌ Error: {analysis.error}")
        sys.exit(1)

    print(f"  Duration: {analysis.duration:.1f}s")
    print(f"  Transcript: {len(analysis.transcript)} segments")
    print(f"  Scenes: {len(analysis.scenes)}")
    print(f"  Frames analyzed: {len([f for f in analysis.frames if f.llm_description])}")

    if args.query:
        from video_analysis.chat import VideoChat
        chat = VideoChat(config=config)
        answer = chat.ask(args.query, analysis)
        if answer:
            print(f"\nQ: {args.query}")
            print(f"A: {answer}")
        else:
            print("❌ Failed to get answer")


def url_mode(args):
    """Download a YouTube URL and process it."""
    try:
        import yt_dlp
    except ImportError:
        print("❌ yt-dlp not installed. Install with: pip install yt-dlp")
        sys.exit(1)

    config = Config()
    video_id = None

    print(f"Downloading: {args.url}")

    ydl_opts = {
        "format": "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
        "outtmpl": str(config.videos_dir / "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(args.url, download=True)
            video_id = info["id"]
            filename = f"{video_id}.{info['ext']}"
            video_path = config.videos_dir / filename
            print(f"  Downloaded: {info.get('title', filename)}")
    except Exception as e:
        print(f"❌ Download failed: {e}")
        sys.exit(1)

    # Process the downloaded video
    args.video = str(video_path)
    args.no_vision = False
    cli_mode(args)


def main():
    parser = argparse.ArgumentParser(
        description="Video Analysis Platform — analyze videos and chat about their content",
    )
    parser.add_argument("--cli", action="store_true", help="CLI mode instead of UI")
    parser.add_argument("--video", type=str, help="Video file to process")
    parser.add_argument("--url", type=str, help="YouTube URL to download and process")
    parser.add_argument("--query", type=str, help="Question to ask about the video")
    parser.add_argument("--no-vision", action="store_true", help="Skip LLM Vision analysis")
    parser.add_argument("--host", type=str, default=None, help="UI host")
    parser.add_argument("--port", type=int, default=None, help="UI port")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    setup_logging(args.verbose)

    if args.url:
        url_mode(args)
    elif args.cli or args.video:
        if not args.video:
            parser.error("--video is required in CLI mode")
        cli_mode(args)
    else:
        # Start web UI
        import uvicorn
        from ui.server import create_app

        config = Config()
        app = create_app(config)

        host = args.host or config.host
        port = args.port or config.port

        uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
