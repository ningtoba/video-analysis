"""
Shared utility functions for the video analysis UI.

These are importable without gradio dependency, making them
testable in environments without the full UI stack.
"""

import re
from pathlib import Path
from typing import List, Optional


def parse_yt_url(url: str) -> bool:
    """
    Check if a URL looks like a YouTube or supported video URL.

    Args:
        url: URL string to check.

    Returns:
        True if the URL matches known video platform patterns.
    """
    patterns = [
        r"https?://(www\.)?youtube\.com/watch\?v=",
        r"https?://youtu\.be/",
        r"https?://(www\.)?youtube\.com/shorts/",
        r"https?://(www\.)?vimeo\.com/",
        r"https?://(www\.)?dailymotion\.com/",
        r"https?://(www\.)?twitch\.tv/",
    ]
    return any(re.match(p, url) for p in patterns)


def queue_html(items: list) -> str:
    """
    Render batch processing queue as HTML.

    Args:
        items: List of dicts with 'name' and 'status' keys.

    Returns:
        HTML string for the queue display.
    """
    if not items:
        return '<p style="color:var(--text-muted);font-size:0.85rem;">Queue is empty. Add videos above.</p>'
    html = '<div style="max-height:200px;overflow-y:auto;">'
    for item in items:
        name = item.get("name", "?")
        status = item.get("status", "pending")
        emoji = {"pending": "⏳", "active": "▶️", "done": "✅", "error": "❌"}.get(
            status, "⏳"
        )
        html += f'<div class="queue-item"><span class="q-status {status}">{emoji}</span><span>{name}</span></div>'
    html += "</div>"
    return html


def is_video_file(path: str) -> bool:
    """Check if a file path has a video extension."""
    video_exts = {".mp4", ".webm", ".mov", ".avi", ".mkv", ".flv", ".wmv"}
    return Path(path).suffix.lower() in video_exts
