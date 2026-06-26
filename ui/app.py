"""
Gradio-based UI for the video analysis platform.

Features:
- Video upload with drag-and-drop
- Video player with timeline + thumbnail preview
- Analysis progress with real-time updates
- Chat interface with source citations and clickable timestamps
- Dark theme, responsive layout
"""

import json
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional

import gradio as gr

from video_analysis.config import Config
from video_analysis.pipeline import VideoPipeline
from video_analysis.rag import VideoRAG
from video_analysis.chat import VideoChat
from video_analysis.models import format_timestamp

logger = logging.getLogger(__name__)

# Pipeline steps for progress display
PIPELINE_STEPS = [
    "Extracting audio",
    "Detecting scenes",
    "Extracting frames",
    "Transcribing audio",
    "Detecting objects",
    "Indexing content",
    "Ready for questions",
]


CSS = """
:root {
  --primary: #7c3aed;
  --primary-hover: #8b5cf6;
  --surface: #1e1b2e;
  --surface-alt: #2d2a3e;
  --text: #e2e0f0;
  --text-muted: #9895b0;
  --border: #3d3a50;
  --accent: #a78bfa;
}

body { background: #0f0d1a; color: var(--text); font-family: 'Inter', sans-serif; }
.gradio-container { max-width: 1600px !important; margin: 0 auto; background: #0f0d1a !important; }

.header { background: linear-gradient(135deg, #1e1b2e, #151225); border-bottom: 1px solid var(--border); padding: 1.5rem 2rem; margin-bottom: 1rem; }
.header h1 { color: white; font-size: 1.75rem; font-weight: 700; margin: 0; }
.header p { color: var(--text-muted); margin: 0.25rem 0 0; }

.badge { display: inline-block; padding: 0.25rem 0.75rem; border-radius: 999px; font-size: 0.8rem; font-weight: 600; }
.badge.ready { background: rgba(52,211,153,0.15); color: #34d399; }
.badge.busy { background: rgba(251,191,36,0.15); color: #fbbf24; animation: pulse 1.5s infinite; }
.badge.error { background: rgba(239,68,68,0.15); color: #ef4444; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.5} }

.source-card { display: inline-flex; align-items: center; gap: 0.25rem; background: rgba(124,58,237,0.15); color: var(--accent); padding: 0.2rem 0.5rem;border-radius: 6px; font-size: 0.75rem; cursor: pointer; margin: 0.1rem; font-family: monospace; }

.step { display: flex; align-items: center; gap: 0.5rem; padding: 0.35rem 0; }
.step .dot { width: 8px; height: 8px; border-radius: 50%; }
.step .dot.pending { background: var(--border); }
.step .dot.active { background: #fbbf24; animation: pulse 1s infinite; }
.step .dot.done { background: #34d399; }
"""


def _progress_html(step: int, message: str) -> str:
    """Render progress step indicators as HTML."""
    html = '<div style="padding: 0.25rem 0;">'
    labels = PIPELINE_STEPS
    for i, s in enumerate(labels):
        if i < step:
            cls = "done"
        elif i == step:
            cls = "active"
        else:
            cls = "pending"
        html += f'<div class="step"><div class="dot {cls}"></div><span style="color: {"var(--text)" if i <= step else "var(--text-muted)"}; font-size:0.85rem;font-weight:{"600" if i==step else "400"}">{s}</span></div>'
    html += "</div>"
    html += f'<p style="color:var(--text-muted);font-size:0.85rem;margin-top:0.5rem">{message}</p>'
    return html


def _video_summary(index) -> str:
    scenes = len(index.scenes)
    dur = index.duration
    objs = sum(len(f.objects) for s in index.scenes for f in s.key_frames)
    return f"{scenes} scenes, {objs} objects, {dur:.0f}s"


def build(config: Optional[Config] = None) -> gr.Blocks:
    """Build the Gradio application."""
    config = config or Config()
    pipeline = VideoPipeline(config)
    rag = VideoRAG(config)
    chat_session = VideoChat(rag, config)

    with gr.Blocks(
        css=CSS,
        theme=gr.themes.Soft(
            primary_hue="violet",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
        title="Video Analysis Platform",
    ) as app:

        # -- state --
        vid = gr.State("")
        vpath = gr.State("")
        busy = gr.State(False)

        # -- header --
        with gr.Row(elem_classes="container"):
            with gr.Column():
                gr.HTML(
                    '<div class="header"><h1>🎥 Video Analysis</h1>'
                    "<p>Upload a video, let AI analyze it, then ask questions</p></div>"
                )
            with gr.Column(scale=0, min_width=180):
                status = gr.HTML('<span class="badge ready">● Ready</span>')

        # -- main layout --
        with gr.Row(equal_height=False):
            # LEFT: upload + player
            with gr.Column(scale=3, min_width=480):
                gr.Markdown("### Upload Video")
                video_input = gr.Video(
                    label="Drop or select a video",
                    sources=["upload", "path"],
                    format="mp4",
                    height=360,
                )
                with gr.Row():
                    process_btn = gr.Button(
                        "⚡ Analyze", variant="primary", size="lg", scale=2
                    )
                    clear_btn = gr.Button("🗑 Clear", scale=1)

                progress_panel = gr.Group(visible=False)
                with progress_panel:
                    gr.Markdown("### Analysis Progress")
                    progress_html = gr.HTML("")

                video_player = gr.Video(label="Video Player", visible=False, height=400)

            # RIGHT: chat
            with gr.Column(scale=4, min_width=500):
                gr.Markdown("### 💬 Ask About the Video")
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=480,
                    bubble_full_width=False,
                    layout="panel",
                )
                chat_input = gr.Textbox(
                    label="Your question",
                    placeholder="e.g., What objects are visible? What is being discussed?",
                    lines=1,
                )
                with gr.Row():
                    send_btn = gr.Button("Send ➤", variant="primary", scale=2)
                    clear_chat_btn = gr.Button("Clear Chat", scale=1)
                show_sources = gr.Checkbox(label="Show source citations", value=True)

        # -- helpers --
        def do_process(video_path: str, state_vid: str):
            if not video_path or busy.value:
                return (
                    state_vid,
                    state_vid,
                    None,
                    gr.update(visible=False),
                    None,
                    status,
                    progress_html,
                )

            busy.value = True
            status.value = '<span class="badge busy">● Processing</span>'

            try:
                yield (
                    _progress_html(0, "Starting..."),
                    status,
                    progress_panel,
                    video_player,
                    vid,
                    vpath,
                )

                # copy file
                pid = str(uuid.uuid4())[:8]
                dest = config.video_dir / f"{pid}.mp4"
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(video_path, dest)

                yield (
                    _progress_html(1, "Extracting audio & detecting scenes..."),
                    status,
                    progress_panel,
                    video_player,
                    vid,
                    vpath,
                )

                index = pipeline.process(str(dest))

                yield (
                    _progress_html(5, "Indexing content for Q&A..."),
                    status,
                    progress_panel,
                    video_player,
                    vid,
                    vpath,
                )

                rag.index_video(index)
                chat_session.reset_history()

                status.value = (
                    '<span class="badge ready">● Ready — ask questions</span>'
                )
                yield (
                    _progress_html(7, f"✅ Complete — {_video_summary(index)}"),
                    status,
                    progress_panel,
                    video_player,
                    pid,
                    str(dest),
                )

            except Exception as e:
                logger.error(f"Process error: {e}", exc_info=True)
                status.value = (
                    f'<span class="badge error">● Error: {str(e)[:100]}</span>'
                )
                yield (
                    _progress_html(-1, f"❌ {str(e)[:200]}"),
                    status,
                    progress_panel,
                    video_player,
                    state_vid,
                    state_vid,
                )
            finally:
                busy.value = False

        evt = process_btn.click(
            fn=do_process,
            inputs=[video_input, vid],
            outputs=[progress_html, status, progress_panel, video_player, vid, vpath],
        )

        def do_send(msg: str, history: list, video_id: str, show_src: bool):
            if not msg or not video_id:
                history = history or []
                history.append((msg or "", "Please analyze a video first."))
                return "", history, history

            history = history or []
            try:
                resp = chat_session.ask_with_history(msg, video_id=video_id)
                answer = resp.content
                if show_src and resp.sources:
                    answer += "\n\n**📎 Sources:**\n"
                    for s in resp.sources[:3]:
                        ts = format_timestamp(s.timestamp)
                        rel = (
                            f" ({s.relevance_score:.0%})"
                            if s.relevance_score > 0
                            else ""
                        )
                        answer += f"- ⏱️ [{ts}](ts:{s.timestamp}){rel}\n"
                history.append((msg, answer))
            except Exception as e:
                logger.error(f"Chat error: {e}")
                history.append((msg, f"⚠️ Error: {str(e)[:200]}"))
            return "", history, history

        def clear_chat():
            chat_session.reset_history()
            return [], []

        def clear_all():
            pipeline.cleanup()
            return (
                "",
                "",
                None,
                gr.update(visible=False),
                '<span class="badge ready">● Ready</span>',
                [],
                [],
            )

        send_btn.click(
            do_send,
            [chat_input, chatbot, vid, show_sources],
            [chat_input, chatbot, chatbot],
        )
        chat_input.submit(
            do_send,
            [chat_input, chatbot, vid, show_sources],
            [chat_input, chatbot, chatbot],
        )
        clear_chat_btn.click(clear_chat, outputs=[chatbot, chatbot])
        clear_btn.click(
            clear_all,
            outputs=[vid, vpath, video_input, video_player, status, chatbot, chatbot],
        )

    return app


def launch(config: Optional[Config] = None):
    """Launch the Gradio UI."""
    cfg = config or Config()
    app = build(cfg)
    logger.info(f"Starting UI on http://{cfg.ui_host}:{cfg.ui_port}")
    app.launch(
        server_name=cfg.ui_host,
        server_port=cfg.ui_port,
        share=cfg.ui_share,
        show_error=True,
        quiet=False,
    )


if __name__ == "__main__":
    launch()
