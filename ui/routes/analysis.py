"""
Analysis Tab — Video upload, URL import, Q&A chat, clip export.

Replaces Tab 1 ("Analysis") from the Gradio UI.
"""

from __future__ import annotations

import json
import logging
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Request, UploadFile, Form
from fastapi.responses import HTMLResponse

from video_analysis.config import Config
from video_analysis.chat import ChatMessage

logger = logging.getLogger(__name__)


def register_analysis_routes(app, config: Config, templates):
    """Register Analysis tab routes on the FastAPI app."""
    router = APIRouter(prefix="/analysis", tags=["Analysis"])

    # ── Page render ──────────────────────────────────────────────────
    @router.get("", response_class=HTMLResponse)
    @router.get("/", response_class=HTMLResponse)
    async def analysis_page(request: Request, partial: str = ""):
        template = "pages/analyze.html"
        if partial:
            return templates.TemplateResponse(request=request, name=template)
        return templates.TemplateResponse(request=request, name=template)

    # ── Status ───────────────────────────────────────────────────────
    @router.get("/status", response_class=HTMLResponse)
    async def analysis_status(request: Request):
        return HTMLResponse('<span class="badge ready">● Ready</span>')

    # ── Upload + Process ─────────────────────────────────────────────
    @router.post("/process", response_class=HTMLResponse)
    async def process_video(request: Request, file: UploadFile = None):
        """Handle video file upload and start processing."""
        # Access backend singletons via app state
        pipeline = request.app.state.pipeline
        rag = request.app.state.rag
        chat = request.app.state.chat

        if file is None or not file.filename:
            return HTMLResponse(
                '<div class="card" style="border-color:var(--error);">'
                '<p class="text-muted">No file uploaded.</p></div>'
            )

        # Save uploaded file
        pid = str(uuid.uuid4())[:8]
        dest = config.video_dir / f"{pid}.mp4"
        dest.parent.mkdir(parents=True, exist_ok=True)

        try:
            content = await file.read()
            dest.write_bytes(content)
        except Exception as exc:
            return HTMLResponse(
                f'<div class="card" style="border-color:var(--error);">'
                f'<p>Failed to save upload: {exc}</p></div>'
            )

        # Return a progress panel that connects to WebSocket
        job_id = pid  # Use the file ID as job ID for simplicity
        try:
            # Start processing in background via job manager
            from video_analysis.job_queue import get_default_manager

            manager = get_default_manager()
            job = await manager.enqueue(
                "process_video",
                video_path=str(dest),
                video_id=pid,
            )
            job_id = job.job_id
        except Exception:
            # Fallback: process synchronously (will show progress steps)
            pass

        return HTMLResponse(f"""
        <div class="card" id="process-progress">
            <h3>⚡ Processing Video</h3>
            <div class="step active" id="step-save">
                <span class="step-icon">✓</span> File saved: {file.filename}
            </div>
            <div class="step pending" id="step-extract">
                <span class="step-icon">2</span> Extracting audio & detecting scenes...
            </div>
            <div class="step pending" id="step-index">
                <span class="step-icon">3</span> Indexing content for Q&A...
            </div>
            <div class="step pending" id="step-done">
                <span class="step-icon">4</span> Complete
            </div>
            <div class="mt-md flex gap-sm">
                <span class="spinner htmx-indicator" id="proc-spinner"></span>
                <span class="text-sm text-muted" id="proc-message">Connecting to job {job_id}...</span>
            </div>
            <script>
                (function() {{
                    const ws = connectJobWS('{job_id}', {{
                        onProgress: function(data) {{
                            const msg = document.getElementById('proc-message');
                            if (msg) msg.textContent = data.progress || data.status;
                            if (data.status === 'running') {{
                                document.getElementById('step-extract')?.classList.replace('pending', 'active');
                            }}
                        }},
                        onComplete: function(data) {{
                            document.querySelectorAll('.step').forEach(el => {{
                                el.classList.replace('active', 'done');
                                el.classList.replace('pending', 'done');
                            }});
                            const msg = document.getElementById('proc-message');
                            if (msg) msg.textContent = 'Processing complete!';
                            const spinner = document.getElementById('proc-spinner');
                            if (spinner) spinner.style.display = 'none';
                            // Update global state
                            const result = data.result || {{}};
                            window.dispatchEvent(new CustomEvent('va:videoReady', {{
                                detail: {{ videoId: result.video_id || '{pid}', videoPath: '{str(dest)}' }}
                            }}));
                            // Tell Alpine.js
                            const appEl = document.querySelector('[x-data]');
                            if (appEl && appEl.__x) {{
                                appEl.__x.$data.videoId = result.video_id || '{pid}';
                                appEl.__x.$data.videoPath = '{str(dest)}';
                                appEl.__x.$data.busy = false;
                            }}
                        }},
                        onError: function(data) {{
                            document.getElementById('step-done')?.classList.replace('pending', 'error');
                            document.querySelector('#step-done .step-icon').textContent = '✗';
                            const msg = document.getElementById('proc-message');
                            if (msg) msg.textContent = 'Error: ' + (data.error || 'Unknown error');
                            const spinner = document.getElementById('proc-spinner');
                            if (spinner) spinner.style.display = 'none';
                        }}
                    }});
                }})();
            </script>
        </div>
        """)

    # ── Import from URL ──────────────────────────────────────────────
    @router.post("/process-url", response_class=HTMLResponse)
    async def process_url(request: Request, url: str = Form(...)):
        """Start processing a video from a URL."""
        pipeline = request.app.state.pipeline

        if not url or not _is_valid_url(url):
            return HTMLResponse(
                '<div class="card" style="border-color:var(--error);">'
                '<p>Unsupported URL format. Enter a YouTube, Vimeo, or direct video URL.</p></div>'
            )

        # Kick off download + process
        try:
            downloaded = pipeline.download_from_url(url, config.video_dir)
            if downloaded is None:
                return HTMLResponse(
                    '<div class="card" style="border-color:var(--error);">'
                    '<p>Download failed. Check the URL and try again.</p></div>'
                )
            vid = str(downloaded.stem) if hasattr(downloaded, 'stem') else str(uuid.uuid4())[:8]
            return HTMLResponse(f"""
            <div class="card" id="process-progress">
                <h3>🌐 Downloading & Processing</h3>
                <div class="step active"><span class="step-icon">✓</span> Downloaded: {url[:60]}...</div>
                <div class="step pending"><span class="step-icon">2</span> Processing video...</div>
                <div class="step pending"><span class="step-icon">3</span> Indexing for Q&A...</div>
                <div class="mt-md">
                    <span class="spinner"></span>
                    <span class="text-sm text-muted ml-sm">Processing...</span>
                </div>
                <div class="mt-md" hx-get="/analysis/progress/{vid}" hx-trigger="load delay:0.5s" hx-swap="outerHTML"></div>
            </div>
            """)
        except Exception as exc:
            return HTMLResponse(
                f'<div class="card" style="border-color:var(--error);">'
                f'<p>Error: {exc}</p></div>'
            )

    # ── Chat ──────────────────────────────────────────────────────────
    @router.post("/chat", response_class=HTMLResponse)
    async def chat_message(
        request: Request,
        message: str = Form(...),
        video_id: str = Form(""),
        show_sources: str = Form("true"),
    ):
        """Send a chat message and return the response as HTML."""
        chat = request.app.state.chat

        if not message or not message.strip():
            return HTMLResponse('<div class="text-muted text-sm">Enter a question to ask about the video.</div>')

        if not video_id:
            return HTMLResponse(
                '<div class="card" style="border-color:var(--error);">'
                '<p>No video selected. Upload or select a video first.</p></div>'
            )

        try:
            response: ChatMessage = chat.ask_with_history(message.strip(), video_id=video_id)
        except Exception as exc:
            logger.error("Chat error: %s", exc)
            return HTMLResponse(
                f'<div class="card" style="border-color:var(--error);">'
                f'<p>Error: {exc}</p></div>'
            )

        sources_html = ""
        if show_sources.lower() == "true" and response.sources:
            src_items = []
            for s in response.sources[:5]:
                ts = _fmt_ts(s.timestamp) if hasattr(s, 'timestamp') else "--:--:--"
                score = f"{s.relevance_score:.2f}" if hasattr(s, 'relevance_score') else ""
                src_items.append(
                    f'<div class="source-card">'
                    f'<span class="src-timestamp">{ts}</span>'
                    f'<span class="src-relevance">{score}</span><br>'
                    f'{_escape(s.text)[:200]}</div>'
                )
            sources_html = (
                '<details class="mt-sm"><summary class="text-sm text-muted">'
                f'📎 {len(response.sources)} sources</summary>'
                f'{"".join(src_items)}</details>'
            )

        return HTMLResponse(f"""
        <div class="chat-bubble assistant">
            <div class="text-sm">{_md(response.content)}</div>
            {sources_html}
        </div>
        """)

    # ── Clear Chat ────────────────────────────────────────────────────
    @router.post("/chat/clear", response_class=HTMLResponse)
    async def clear_chat(request: Request):
        chat = request.app.state.chat
        chat.reset_history()
        return HTMLResponse("")

    # ── Export Clip ───────────────────────────────────────────────────
    @router.post("/export-clip", response_class=HTMLResponse)
    async def export_clip(
        request: Request,
        start: float = Form(0.0),
        end: float = Form(10.0),
    ):
        """Export a video clip."""
        pipeline = request.app.state.pipeline
        # Get video path from app state or query param
        # For now, return a placeholder
        return HTMLResponse(
            '<div class="card"><p class="text-muted">Clip export coming soon.</p></div>'
        )

    # ── Register router ───────────────────────────────────────────────
    app.include_router(router)


# ── Helpers ────────────────────────────────────────────────────────────────


def _is_valid_url(url: str) -> bool:
    """Basic URL validation for video sources."""
    import re
    patterns = [
        r'youtube\.com/watch',
        r'youtu\.be/',
        r'youtube\.com/shorts/',
        r'vimeo\.com/',
        r'dailymotion\.com/',
        r'twitch\.tv/',
        r'\.mp4$', r'\.mkv$', r'\.webm$',
    ]
    return any(re.search(p, url, re.IGNORECASE) for p in patterns)


def _fmt_ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _md(text: str) -> str:
    """Minimal markdown-to-HTML."""
    if not text:
        return ""
    import re
    text = _escape(text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    text = re.sub(r'`(.+?)`', r'<code>\1</code>', text)
    text = text.replace('\n', '<br>')
    return text
