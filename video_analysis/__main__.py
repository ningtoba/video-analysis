"""
Entry point for the video analysis platform.

Modes:
  - (no args)          Start web UI with offline pipeline
  --cli --video <file> Process a video file (offline pipeline)
  --url <url>          Download YouTube URL and process
  --watch <source>     Real-time stream analysis (RTSP/webcam/file)
  --source <type>      Source type for --watch: rtsp, webcam, file
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading

from video_analysis.config import Config

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


def offline_cli(args):
    """Process an uploaded video file (offline pipeline)."""
    from video_analysis.pipeline import VideoPipeline

    config = Config()
    pipeline = VideoPipeline(config)

    from pathlib import Path
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Video not found: {args.video}")
        sys.exit(1)

    print(f"Processing: {video_path.name}")
    analysis = pipeline.process_video(str(video_path), skip_llm_vision=args.no_vision)

    if analysis.error:
        print(f"Error: {analysis.error}")
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
            print("Failed to get answer")


def url_mode(args):
    """Download YouTube URL and process it."""
    try:
        import yt_dlp
    except ImportError:
        print("yt-dlp not installed")
        sys.exit(1)

    config = Config()
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
            video_path = config.videos_dir / f"{video_id}.{info['ext']}"
            print(f"  Downloaded: {info.get('title', video_id)}")
    except Exception as e:
        print(f"Download failed: {e}")
        sys.exit(1)

    args.video = str(video_path)
    args.no_vision = False
    offline_cli(args)


def watch_mode(args):
    """Real-time stream analysis mode.

    Connects to an RTSP/webcam/file source, samples frames,
    detects motion, and sends periodic LLM Vision analysis.
    """
    from video_analysis.llm_provider import LLMProviderConfig, get_llm_provider
    from video_analysis.stream.engine import StreamEngine

    config = Config()

    # Set up LLM for vision
    llm_config = LLMProviderConfig(
        provider=config.llm_provider,
        api_key=config.llm_api_key,
        api_base=config.llm_api_base,
        model=config.llm_model,
        temperature=config.llm_temperature,
        max_tokens=config.llm_max_tokens,
    )
    llm = get_llm_provider(llm_config)

    def chat_fn(messages, images, system=None):
        """Wrapper for LLM Vision API."""
        return llm.chat_with_images(messages, images, system)

    # Determine source string
    source = args.watch
    if args.source == "webcam":
        source = "0"  # First webcam
    elif args.source == "file":
        source = args.watch
    # else rtsp — use URL as-is

    stream_id = args.stream_id or f"watch_{int(__import__('time').time())}"

    print(f"Starting stream: {source}")
    print(f"  Stream ID: {stream_id}")
    print(f"  Target FPS: {args.fps}")
    print(f"  Periodic analysis: every {args.interval}s")
    print(f"  Motion threshold: {args.motion_threshold}")
    print("Press Ctrl+C to stop")

    def on_event(event):
        ts = __import__('datetime').datetime.fromtimestamp(event.timestamp)
        print(f"  [{ts.strftime('%H:%M:%S')}] {event.description[:100]}...")

    engine = StreamEngine(
        source=source,
        llm_chat_fn=chat_fn,
        stream_id=stream_id,
        target_fps=args.fps,
        buffer_seconds=args.buffer,
        motion_strategy=args.motion_strategy,
        motion_threshold=args.motion_threshold,
        periodic_interval=args.interval,
        cooldown_seconds=args.cooldown,
        on_event=on_event,
    )

    engine.start(block=True)


def main():
    parser = argparse.ArgumentParser(
        description="Video Analysis Platform",
    )
    parser.add_argument("--cli", action="store_true", help="CLI mode (offline pipeline)")
    parser.add_argument("--video", type=str, help="Video file to process")
    parser.add_argument("--url", type=str, help="YouTube URL to download and process")
    parser.add_argument("--query", type=str, help="Question about the video")
    parser.add_argument("--no-vision", action="store_true", help="Skip LLM Vision")

    # Stream/watch mode
    parser.add_argument("--watch", type=str, metavar="SOURCE",
                        help="Watch a stream (RTSP URL, file path, or webcam index)")
    parser.add_argument("--source", type=str, default="auto",
                        choices=["auto", "rtsp", "webcam", "file"],
                        help="Source type for --watch")
    parser.add_argument("--stream-id", type=str, default="",
                        help="Stream identifier for event store")
    parser.add_argument("--fps", type=float, default=1.0,
                        help="Frame sampling rate (default: 1.0)")
    parser.add_argument("--buffer", type=float, default=300.0,
                        help="Circular buffer duration in seconds (default: 300)")
    parser.add_argument("--interval", type=float, default=30.0,
                        help="Periodic LLM analysis interval in seconds (default: 30)")
    parser.add_argument("--cooldown", type=float, default=15.0,
                        help="Minimum seconds between LLM calls (default: 15)")
    parser.add_argument("--motion-threshold", type=float, default=0.02,
                        help="Motion detection sensitivity (default: 0.02)")
    parser.add_argument("--motion-strategy", type=str, default="diff",
                        choices=["diff", "hist", "background"],
                        help="Motion detection strategy (default: diff)")

    parser.add_argument("--host", type=str, default=None, help="UI host")
    parser.add_argument("--port", type=int, default=None, help="UI port")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    setup_logging(args.verbose)

    if args.watch:
        watch_mode(args)
    elif args.url:
        url_mode(args)
    elif args.cli or args.video:
        if not args.video:
            parser.error("--video is required in CLI mode")
        offline_cli(args)
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
