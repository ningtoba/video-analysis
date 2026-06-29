"""
Gradio-based UI for the video analysis platform.

Features:
- Video upload with drag-and-drop
- YouTube URL import
- Batch video processing queue
- Video player with timeline + thumbnail preview
- Analysis progress with real-time updates
- Chat interface with source citations and clickable timestamps
- Clip export (jump to precise moments)
- Multi-video library management with search and delete
- Dark theme, responsive layout
"""

import logging
import shutil
import uuid
from pathlib import Path
from typing import List, Optional

import gradio as gr

from ui.camera import inject_camera_tab
from ui.comparison import inject_comparison_tab
from ui.event_timeline import inject_event_timeline_tab
from ui.knowledge_graph import inject_knowledge_graph_tab
from ui.monitor import inject_monitor_tab
from ui.utils import parse_yt_url, queue_html
from ui.workflow import inject_workflow_tab
from video_analysis.chat import VideoChat
from video_analysis.config import Config
from video_analysis.models import VideoIndex, format_timestamp
from video_analysis.pipeline import VideoPipeline
from video_analysis.rag import VideoRAG

logger = logging.getLogger(__name__)

# Pipeline steps for progress display
PIPELINE_STEPS = [
    "Extracting audio",
    "Detecting scenes",
    "Extracting frames",
    "Transcribing audio",
    "Detecting objects",
    "Describing scenes",
    "Generating sprite sheet",
    "Indexing content",
    "Event segmentation & indexing",
    "Ready for questions",
]

LIBRARY_CSS = """
.library-card { 
    background: var(--surface); 
    border-radius: 12px; 
    padding: 1rem; 
    margin-bottom: 0.75rem;
    border: 1px solid var(--border);
    transition: border-color 0.2s;
}
.library-card:hover {
    border-color: var(--primary);
    cursor: pointer;
}
.library-card .title { font-weight: 600; font-size: 1rem; }
.library-card .meta { color: var(--text-muted); font-size: 0.85rem; }
.library-card .badge-stats { 
    display: inline-flex; gap: 0.5rem; margin-top: 0.5rem; flex-wrap: wrap;
}
.library-card .stat {
    background: rgba(124,58,237,0.1);
    padding: 0.15rem 0.5rem;
    border-radius: 6px;
    font-size: 0.75rem;
    color: var(--accent);
}
.library-card .delete-btn {
    float: right;
    background: rgba(239,68,68,0.15);
    color: #ef4444;
    border: none;
    border-radius: 6px;
    padding: 0.25rem 0.5rem;
    font-size: 0.8rem;
    cursor: pointer;
    transition: background 0.2s;
}
.library-card .delete-btn:hover {
    background: rgba(239,68,68,0.3);
}
"""

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

/* Timeline hover preview */
.timeline-preview { position: relative; }
.timeline-preview .hover-card {
    display: none;
    position: absolute;
    bottom: 100%;
    left: 50%;
    transform: translateX(-50%);
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 4px;
    z-index: 100;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
}
.timeline-preview:hover .hover-card { display: block; }

/* Clip export */
.clip-preview { border: 1px solid var(--border); border-radius: 8px; padding: 0.75rem; margin-top: 0.5rem; }

/* Batch queue status */
.queue-item { display: flex; align-items: center; gap: 0.5rem; padding: 0.4rem 0.5rem; border-bottom: 1px solid var(--border); }
.queue-item .q-status { font-size: 0.8rem; padding: 0.15rem 0.4rem; border-radius: 4px; }
.queue-item .q-status.pending { background: rgba(251,191,36,0.15); color: #fbbf24; }
.queue-item .q-status.done { background: rgba(52,211,153,0.15); color: #34d399; }
.queue-item .q-status.error { background: rgba(239,68,68,0.15); color: #ef4444; }
.queue-item .q-status.active { background: rgba(124,58,237,0.15); color: #a78bfa; animation: pulse 1s infinite; }

/* Import tab progress */
.import-progress { margin-top: 1rem; }
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
        html += f'<div class="step"><div class="dot {cls}"></div><span style="color: {"var(--text)" if i <= step else "var(--text-muted)"}; font-size:0.85rem;font-weight:{"600" if i == step else "400"}">{s}</span></div>'
    html += "</div>"
    html += f'<p style="color:var(--text-muted);font-size:0.85rem;margin-top:0.5rem">{message}</p>'
    return html


def _video_summary(index: VideoIndex) -> str:
    scenes = len(index.scenes)
    dur = index.duration
    objs = sum(len(f.objects) for s in index.scenes for f in s.key_frames)
    descs = sum(1 for s in index.scenes for f in s.key_frames if f.description)
    return f"{scenes} scenes, {objs} objects, {descs} described frames, {dur:.0f}s"


def _library_html(
    video_ids: List[str],
    rag: VideoRAG,
    pipeline: Optional[VideoPipeline] = None,
    config: Optional[Config] = None,
) -> str:
    """Render library cards as HTML with click-to-select and rich metadata."""
    if not video_ids:
        return '<p style="color:var(--text-muted);padding:1rem;">No videos analyzed yet. Upload one above.</p>'
    html = '<div id="library-cards-container">'
    for vid in video_ids:
        html += f'<div class="library-card" onclick="window.__selectVideo(\'{vid}\')">'
        html += f'<div class="title">{vid}</div>'
        html += f'<div class="meta">ID: {vid}</div>'
        # Add stats if we have pipeline/config
        if pipeline and config:
            video_path = config.video_dir / f"{vid}.mp4"
            if video_path.exists():
                dur = pipeline._get_duration(video_path)
                parts = []
                if dur > 0:
                    parts.append(f"⏱ {dur:.0f}s")
                if parts:
                    html += (
                        '<div class="badge-stats">'
                        + "".join(f'<span class="stat">{p}</span>' for p in parts)
                        + "</div>"
                    )
        html += f'<button class="delete-btn" onclick="event.stopPropagation(); window.__deleteVideo(\'{vid}\')">🗑 Delete</button>'
        html += "</div>"
    html += "</div>"
    return html


def build(config: Optional[Config] = None) -> gr.Blocks:
    """Build the Gradio application."""
    config = config or Config()
    pipeline = VideoPipeline(config)
    rag = VideoRAG(config)
    chat_session = VideoChat(rag, config)

    with gr.Blocks(
        title="Video Analysis Platform",
    ) as app:
        # In Gradio 6, theme and css moved from Blocks() to launch().
        # Set as attributes to avoid deprecation warning while keeping
        # compatibility with gr.mount_gradio_app().
        app.theme = gr.themes.Soft(
            primary_hue="violet",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        )
        app.css = CSS + LIBRARY_CSS

        # -- state --
        vid = gr.State("")
        vpath = gr.State("")
        busy = gr.State(False)
        batch_queue = gr.State([])  # list of {name, filepath, url, status}

        # -- header --
        with gr.Row(elem_classes="container"):
            with gr.Column():
                gr.HTML(
                    '<div class="header"><h1>🎥 Video Analysis</h1>'
                    "<p>Upload a video, paste a URL, or batch process — let AI analyze it, then ask questions</p></div>"
                )
            with gr.Column(scale=0, min_width=180):
                status = gr.HTML('<span class="badge ready">● Ready</span>')

        with gr.Tabs():
            # ============ TAB 1: ANALYSIS ============
            with gr.TabItem("📹 Analysis", id="analysis"):
                with gr.Row(equal_height=False):
                    # LEFT: upload + player
                    with gr.Column(scale=3, min_width=480):
                        gr.Markdown("### Upload Video or Paste URL")
                        video_input = gr.Video(
                            label="Drop or select a video file",
                            sources=["upload"],
                            format="mp4",
                            height=320,
                        )
                        url_input = gr.Textbox(
                            label="Or paste a YouTube/URL",
                            placeholder="https://www.youtube.com/watch?v=...",
                            lines=1,
                        )
                        with gr.Row():
                            process_btn = gr.Button(
                                "⚡ Analyze Upload",
                                variant="primary",
                                size="lg",
                                scale=2,
                            )
                            import_url_btn = gr.Button(
                                "🌐 Download & Analyze URL",
                                variant="secondary",
                                size="lg",
                                scale=2,
                            )
                            clear_btn = gr.Button("🗑 Clear", scale=1)

                        # Progress panel
                        progress_panel = gr.Group(visible=False)
                        with progress_panel:
                            gr.Markdown("### Analysis Progress")
                            progress_html = gr.HTML("")

                        video_player = gr.Video(label="Video Player", visible=False, height=400)

                        # Sprite sheet + clip export
                        with gr.Group(visible=False) as export_group:
                            gr.Markdown("### 🎬 Export Clip")
                            with gr.Row():
                                clip_start = gr.Number(label="Start (seconds)", value=0, minimum=0)
                                clip_end = gr.Number(label="End (seconds)", value=10, minimum=0)
                            export_btn = gr.Button("✂️ Export Clip", variant="secondary", size="sm")
                            clip_output = gr.Video(
                                label="Exported Clip",
                                visible=False,
                                height=200,
                            )

                    # RIGHT: chat
                    with gr.Column(scale=4, min_width=500):
                        gr.Markdown("### 💬 Ask About the Video")
                        chatbot = gr.Chatbot(
                            label="Conversation",
                            height=480,
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

            # ============ TAB 2: IMPORT (YouTube/URL) ============
            with gr.TabItem("🌐 Import", id="import"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=2, min_width=480):
                        gr.Markdown("### Import a Video from URL")
                        import_url_input = gr.Textbox(
                            label="Video URL",
                            placeholder="https://www.youtube.com/watch?v=...",
                            lines=2,
                        )
                        with gr.Row():
                            import_download_btn = gr.Button(
                                "⬇️ Download & Analyze",
                                variant="primary",
                                size="lg",
                                scale=2,
                            )
                            import_clear_btn = gr.Button("🗑 Clear", scale=1)

                        # Import progress
                        import_progress_panel = gr.Group(
                            visible=False, elem_classes="import-progress"
                        )
                        with import_progress_panel:
                            gr.Markdown("### Import Progress")
                            import_progress_html = gr.HTML("")

                    with gr.Column(scale=3, min_width=500):
                        gr.Markdown("### Result")
                        import_video_player = gr.Video(
                            label="Analyzed Video", visible=False, height=350
                        )
                        import_video_id = gr.State("")
                        import_video_path = gr.State("")

            # ============ TAB 3: BATCH PROCESSING ============
            with gr.TabItem("📦 Batch", id="batch"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### Batch Import")
                        batch_urls = gr.Textbox(
                            label="URLs (one per line)",
                            placeholder="https://www.youtube.com/watch?v=...\nhttps://www.youtube.com/watch?v=...",
                            lines=6,
                        )
                        batch_files = gr.File(
                            label="Or upload video files",
                            file_count="multiple",
                            file_types=[".mp4", ".webm", ".mov", ".avi", ".mkv"],
                        )
                        with gr.Row():
                            add_batch_btn = gr.Button(
                                "➕ Add to Queue", variant="primary", size="lg", scale=2
                            )
                            process_batch_btn = gr.Button(
                                "▶️ Process All",
                                variant="secondary",
                                size="lg",
                                scale=2,
                            )
                            clear_batch_btn = gr.Button("🗑 Clear Queue", scale=1)
                    with gr.Column(scale=3):
                        gr.Markdown("### Queue Status")
                        batch_status = gr.HTML(
                            '<p style="color:var(--text-muted);padding:1rem;">Queue is empty.</p>'
                        )
                        gr.Markdown("### Batch Progress")
                        batch_overall_progress = gr.HTML(
                            '<p style="color:var(--text-muted);font-size:0.85rem;">Waiting for items to be added...</p>'
                        )

            # ============ TAB 5: SEARCH ALL VIDEOS (Cross-Video Semantic Search) ============
            with gr.TabItem("🔍 Video Search", id="search"):
                with gr.Row():
                    with gr.Column(scale=2, min_width=350):
                        gr.Markdown("### Search Across All Videos")
                        gr.Markdown(
                            "Search your entire video library with natural language. "
                            "Queries match against transcripts, frame descriptions, OCR text, "
                            "and detected objects from all indexed videos."
                        )
                        search_query = gr.Textbox(
                            label="Search Query",
                            placeholder='e.g., "people talking about Python" or "scenes with cars"',
                            lines=2,
                        )
                        with gr.Row():
                            search_btn = gr.Button("🔎 Search All", variant="primary", scale=2)
                            search_clear_btn = gr.Button("Clear", scale=1)
                        search_status = gr.HTML(
                            '<p style="color:var(--text-muted);padding:0.5rem 0;">'
                            "Enter a query and click Search to find relevant scenes.</p>"
                        )
                    with gr.Column(scale=3):
                        gr.Markdown("### Results")
                        search_results = gr.HTML(
                            '<p style="color:var(--text-muted);padding:1rem;">No results yet.</p>'
                        )
                        search_detail = gr.HTML("")
            with gr.TabItem("📚 Library", id="library"):
                with gr.Row():
                    with gr.Column(scale=2, min_width=350):
                        gr.Markdown("### Your Video Library")
                        lib_search = gr.Textbox(
                            label="🔍 Search videos",
                            placeholder="Filter by name...",
                            lines=1,
                        )
                        with gr.Row():
                            refresh_lib_btn = gr.Button("🔄 Refresh Library", size="sm", scale=2)
                            delete_all_lib_btn = gr.Button("🗑 Delete All", size="sm", scale=1)
                        library_list = gr.HTML(
                            '<p style="color:var(--text-muted);padding:1rem;">No videos analyzed yet. Upload one above.</p>'
                        )
                        delete_status = gr.HTML("")
                    with gr.Column(scale=3):
                        gr.Markdown("### Video Details")
                        lib_video_id = gr.State("")
                        lib_video_player = gr.Video(label="Selected Video", visible=False)
                        lib_info = gr.JSON(label="Video Info", visible=False)

            # ============ TAB 6: PIPELINE WORKFLOW (Gradio 6 Workflow) ============
            if config.workflow_enabled:
                inject_workflow_tab(config)

            # ============ TAB 7: CAMERA (Webcam Live Capture) ============
            inject_camera_tab(app, config)

            # ============ TAB 8: MONITOR (Dashboard & Evaluations) ============
            inject_monitor_tab(app, config)

            # ============ TAB 9: EVAL COMPARISON (Cross-Report Analysis) ============
            inject_comparison_tab(config)

            # ============ TAB 10: KNOWLEDGE GRAPH (Entity Explorer) ============
            inject_knowledge_graph_tab(app, config)

            # ============ TAB 11: EVENT-CAUSAL TIMELINE (v0.58.0) ============
            inject_event_timeline_tab(app, config)

        # ==================== EVENT HANDLERS ====================

        # --- Shared helpers ---
        def _persist_events_to_kg(video_id: str):
            """Persist indexed events to KnowledgeGraph if event-causal RAG is enabled."""
            if not config.event_causal_rag_enabled:
                return
            try:
                from video_analysis.knowledge_graph import KnowledgeGraph

                kg = KnowledgeGraph(config)
                kg.persist_events_from_rag(rag, video_id)
                kg.close()
            except Exception as kg_exc:
                logger.warning("KG event persistence failed: %s - continuing", kg_exc)

        # --- Process Video ---
        def do_process(video_path: str, state_vid: str):
            if not video_path or busy.value:
                return (
                    gr.update(),
                    gr.update(),
                    gr.update(visible=False),
                    gr.update(value=None),
                    gr.update(),
                    gr.update(),
                    gr.update(visible=False),
                )

            busy.value = True
            status.value = '<span class="badge busy">● Processing</span>'

            try:
                yield (
                    _progress_html(0, "Starting..."),
                    status,
                    gr.update(visible=True),
                    None,
                    state_vid,
                    state_vid,
                    gr.update(visible=False),
                )

                # copy file
                pid = str(uuid.uuid4())[:8]
                dest = config.video_dir / f"{pid}.mp4"
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(video_path, dest)

                yield (
                    _progress_html(1, "Extracting audio & detecting scenes..."),
                    status,
                    gr.update(visible=True),
                    None,
                    state_vid,
                    state_vid,
                    gr.update(visible=False),
                )

                index = pipeline.process(str(dest))
                video_path_str = str(dest)

                yield (
                    _progress_html(6, "Indexing content for Q&A..."),
                    status,
                    gr.update(visible=True),
                    None,
                    state_vid,
                    state_vid,
                    gr.update(visible=False),
                )

                rag.index_video(index)
                chat_session.reset_history()

                _persist_events_to_kg(index.video_id)

                status.value = '<span class="badge ready">● Ready - ask questions</span>'
                yield (
                    _progress_html(
                        9 if config.event_causal_rag_enabled else 8,
                        f"Complete - {_video_summary(index)}",
                    ),
                    status,
                    gr.update(visible=True),
                    video_path_str,
                    pid,
                    video_path_str,
                    gr.update(visible=True),
                )

            except Exception as e:
                logger.error(f"Process error: {e}", exc_info=True)
                status.value = f'<span class="badge error">● Error: {str(e)[:100]}</span>'
                yield (
                    _progress_html(-1, f"❌ {str(e)[:200]}"),
                    status,
                    gr.update(visible=True),
                    None,
                    state_vid,
                    state_vid,
                    gr.update(visible=False),
                )
            finally:
                busy.value = False

        # --- Import from URL ---
        def do_import_url(url: str, state_vid: str):
            if not url or busy.value:
                return (
                    gr.update(),
                    gr.update(),
                    gr.update(visible=False),
                    gr.update(value=None),
                    gr.update(),
                    gr.update(),
                    gr.update(visible=False),
                )

            if not parse_yt_url(url):
                status.value = '<span class="badge error">● Unsupported URL format</span>'
                return (
                    gr.update(),
                    status,
                    gr.update(visible=False),
                    gr.update(value=None),
                    gr.update(),
                    gr.update(),
                    gr.update(visible=False),
                )

            busy.value = True
            status.value = '<span class="badge busy">● Downloading...</span>'

            try:
                yield (
                    _progress_html(0, "Downloading video from URL..."),
                    status,
                    gr.update(visible=True),
                    None,
                    state_vid,
                    state_vid,
                    gr.update(visible=False),
                )

                downloaded = pipeline.download_from_url(url, config.video_dir)
                if downloaded is None:
                    status.value = '<span class="badge error">● Download failed</span>'
                    yield (
                        _progress_html(-1, "❌ Failed to download video"),
                        status,
                        gr.update(visible=True),
                        None,
                        state_vid,
                        state_vid,
                        gr.update(visible=False),
                    )
                    return

                logger.info(f"Downloaded to: {downloaded}")

                yield (
                    _progress_html(1, "Processing downloaded video..."),
                    status,
                    gr.update(visible=True),
                    None,
                    state_vid,
                    state_vid,
                    gr.update(visible=False),
                )

                index = pipeline.process(str(downloaded))

                yield (
                    _progress_html(6, "Indexing content for Q&A..."),
                    status,
                    gr.update(visible=True),
                    None,
                    state_vid,
                    state_vid,
                    gr.update(visible=False),
                )

                rag.index_video(index)
                chat_session.reset_history()

                _persist_events_to_kg(index.video_id)

                status.value = '<span class="badge ready">● Ready - ask questions</span>'
                yield (
                    _progress_html(
                        9 if config.event_causal_rag_enabled else 8,
                        f"Complete - {_video_summary(index)}",
                    ),
                    status,
                    gr.update(visible=True),
                    str(downloaded),
                    index.video_id,
                    str(downloaded),
                    gr.update(visible=True),
                )

            except Exception as e:
                logger.error(f"Import error: {e}", exc_info=True)
                status.value = f'<span class="badge error">● Error: {str(e)[:100]}</span>'
                yield (
                    _progress_html(-1, f"❌ {str(e)[:200]}"),
                    status,
                    gr.update(visible=True),
                    None,
                    state_vid,
                    state_vid,
                    gr.update(visible=False),
                )
            finally:
                busy.value = False

        # --- Import Tab: Download from URL (separate handler for Import tab) ---
        def do_import_tab_download(url: str):
            """Download from URL in the Import tab, process, and show result in the import tab."""
            if not url or busy.value:
                return (
                    gr.update(visible=False),
                    gr.update(),
                    "",
                    "",
                    status,
                )

            if not parse_yt_url(url):
                status.value = '<span class="badge error">● Unsupported URL format</span>'
                return (
                    gr.update(visible=False),
                    gr.update(
                        value='<p style="color:#ef4444;">❌ Unsupported URL format. Please enter a YouTube, Vimeo, DailyMotion, or Twitch URL.</p>'
                    ),
                    "",
                    "",
                    status,
                )

            busy.value = True
            status.value = '<span class="badge busy">● Downloading...</span>'

            try:
                yield (
                    gr.update(visible=True),
                    gr.update(value=_progress_html(0, "⬇️ Downloading video from URL...")),
                    "",
                    "",
                    status,
                )

                downloaded = pipeline.download_from_url(url, config.video_dir)
                if downloaded is None:
                    status.value = '<span class="badge error">● Download failed</span>'
                    yield (
                        gr.update(visible=True),
                        gr.update(
                            value=_progress_html(
                                -1, "❌ Download failed. Check the URL and try again."
                            )
                        ),
                        "",
                        "",
                        status,
                    )
                    return

                logger.info(f"Import tab: Downloaded to: {downloaded}")

                yield (
                    gr.update(visible=True),
                    gr.update(value=_progress_html(1, "⚙️ Processing downloaded video...")),
                    "",
                    "",
                    status,
                )

                index = pipeline.process(str(downloaded))

                yield (
                    gr.update(visible=True),
                    gr.update(value=_progress_html(6, "📝 Indexing content for Q&A...")),
                    "",
                    "",
                    status,
                )

                rag.index_video(index)

                _persist_events_to_kg(index.video_id)

                status.value = '<span class="badge ready">● Ready - ask questions</span>'

                video_path_str = str(downloaded)
                steps = 9 if config.event_causal_rag_enabled else 8
                yield (
                    gr.update(visible=True, value=video_path_str),
                    gr.update(value=_progress_html(steps, f"Complete - {_video_summary(index)}")),
                    index.video_id,
                    video_path_str,
                    status,
                )

            except Exception as e:
                logger.error(f"Import tab error: {e}", exc_info=True)
                status.value = f'<span class="badge error">● Error: {str(e)[:100]}</span>'
                yield (
                    gr.update(visible=False),
                    gr.update(value=_progress_html(-1, f"❌ {str(e)[:200]}")),
                    "",
                    "",
                    status,
                )
            finally:
                busy.value = False

        def clear_import_tab():
            return (
                "",
                gr.update(visible=False),
                gr.update(value=""),
                "",
                "",
            )

        # --- Send Chat Message (with streaming support) ---
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
                        rel = f" ({s.relevance_score:.0%})" if s.relevance_score > 0 else ""
                        answer += f"- ⏱️ [{ts}](ts:{s.timestamp}){rel}\n"
                history.append((msg, answer))
            except Exception as e:
                logger.error(f"Chat error: {e}")
                history.append((msg, f"⚠️ Error: {str(e)[:200]}"))
            return "", history, history

        # --- Export Clip ---
        def do_export_clip(start: float, end: float, video_path: str):
            if not video_path or start >= end:
                return gr.update(visible=False), None
            try:
                clip_path = pipeline.export_clip(video_path, start, end)
                return gr.update(visible=True), clip_path
            except Exception as e:
                logger.error(f"Export error: {e}")
                return gr.update(visible=False), None

        def clear_chat():
            chat_session.reset_history()
            return [], []

        def reset_ui():
            pipeline.cleanup()
            return (
                "",
                "",
                None,
                gr.update(visible=False),
                None,
                None,
                gr.update(visible=False),
                '<span class="badge ready">● Ready</span>',
                [],
                [],
                gr.update(visible=False),
                None,
            )

        def refresh_library():
            try:
                video_ids = rag.list_videos()
                html = _library_html(video_ids, rag, pipeline, config)
                return html, ""
            except Exception as e:
                logger.error(f"Library refresh error: {e}")
                return (
                    f'<p style="color:var(--text-muted);">Error loading library: {str(e)[:100]}</p>',
                    "",
                )

        def search_library(query: str):
            """Filter library by video name/ID keyword."""
            try:
                video_ids = rag.list_videos()
                if query:
                    query_lower = query.lower().strip()
                    video_ids = [v for v in video_ids if query_lower in v.lower()]
                html = _library_html(video_ids, rag, pipeline, config)
                if query and not video_ids:
                    html = f'<p style="color:var(--text-muted);padding:1rem;">No videos matching "{query}".</p>'
                return html
            except Exception as e:
                logger.error(f"Library search error: {e}")
                return f'<p style="color:var(--text-muted);">Error searching library: {str(e)[:100]}</p>'

        def delete_library_video(video_id: str):
            """Delete a video from the library (RAG index + video file)."""
            if not video_id:
                return (
                    gr.update(),
                    '<p style="color:var(--text-muted);">No video specified.</p>',
                )
            try:
                # Delete from RAG index
                rag.delete_video(video_id)

                # Also try to remove the video file
                video_path = config.video_dir / f"{video_id}.mp4"
                if video_path.exists():
                    video_path.unlink()

                # Also try to remove thumbnail data
                thumb_json = config.thumbnails_dir / f"{video_id}_sprite.json"
                thumb_jpg = config.thumbnails_dir / f"{video_id}_sprite.jpg"
                if thumb_json.exists():
                    thumb_json.unlink()
                if thumb_jpg.exists():
                    thumb_jpg.unlink()

                # Refresh library
                video_ids = rag.list_videos()
                html = _library_html(video_ids, rag, pipeline, config)
                return (
                    gr.update(value=html),
                    f'<p style="color:#34d399;">✅ Deleted "{video_id}"</p>',
                )
            except Exception as e:
                logger.error(f"Delete video error: {e}")
                return (
                    gr.update(),
                    f'<p style="color:#ef4444;">❌ Error deleting {video_id}: {str(e)[:100]}</p>',
                )

        def delete_all_library():
            """Delete all videos from the library."""
            try:
                video_ids = rag.list_videos()
                for vid in video_ids:
                    rag.delete_video(vid)
                    video_path = config.video_dir / f"{vid}.mp4"
                    if video_path.exists():
                        video_path.unlink()
                    for fname in [f"{vid}_sprite.json", f"{vid}_sprite.jpg"]:
                        fpath = config.thumbnails_dir / fname
                        if fpath.exists():
                            fpath.unlink()
                html = '<p style="color:var(--text-muted);padding:1rem;">Library cleared.</p>'
                return (
                    gr.update(value=html),
                    '<p style="color:#34d399;">✅ All videos deleted.</p>',
                    gr.update(visible=False),
                    gr.update(visible=False),
                    "",
                )
            except Exception as e:
                logger.error(f"Delete all error: {e}")
                return (
                    gr.update(),
                    f'<p style="color:#ef4444;">❌ Error: {str(e)[:100]}</p>',
                    gr.update(),
                    gr.update(),
                    "",
                )

        # --- Batch Queue ---
        def do_add_to_queue(urls: str, files: list, queue: list):
            """Add URLs and uploaded files to the batch queue."""
            queue = list(queue) if queue else []

            # Add URLs
            if urls:
                for url in urls.strip().split("\n"):
                    url = url.strip()
                    if url and parse_yt_url(url):
                        queue.append(
                            {
                                "name": url[:60] + "..." if len(url) > 60 else url,
                                "url": url,
                                "filepath": None,
                                "status": "pending",
                            }
                        )

            # Add files
            if files:
                for f in files:
                    if isinstance(f, str):
                        queue.append(
                            {
                                "name": Path(f).name,
                                "url": None,
                                "filepath": f,
                                "status": "pending",
                            }
                        )
                    elif hasattr(f, "name"):
                        queue.append(
                            {
                                "name": f.name,
                                "url": None,
                                "filepath": f.name,
                                "status": "pending",
                            }
                        )

            total = len(queue)
            pending = sum(1 for q in queue if q["status"] == "pending")
            progress_html_val = f'<p style="color:var(--text-muted);font-size:0.85rem;">{total} items in queue ({pending} pending)</p>'
            return queue, queue_html(queue), progress_html_val

        def do_process_batch(queue: list):
            """Process all items in the batch queue sequentially."""
            if not queue:
                return (
                    queue,
                    queue_html([]),
                    '<p style="color:var(--text-muted);font-size:0.85rem;">Queue is empty.</p>',
                    status,
                )

            busy.value = True
            status.value = '<span class="badge busy">● Batch processing...</span>'

            total = len(queue)
            for idx, item in enumerate(queue):
                if item["status"] != "pending":
                    continue

                item["status"] = "active"
                progress_str = f'<p style="color:var(--text-muted);font-size:0.85rem;">Processing {idx + 1}/{total}...</p>'
                yield queue, queue_html(queue), progress_str, status

                try:
                    filepath = item.get("filepath")
                    url = item.get("url")

                    if url:
                        # Download first
                        downloaded = pipeline.download_from_url(url, config.video_dir)
                        if downloaded is None:
                            item["status"] = "error"
                            yield queue, queue_html(queue), progress_str, status
                            continue
                        filepath = str(downloaded)
                    elif filepath:
                        # Copy file to video dir
                        pid = str(uuid.uuid4())[:8]
                        dest = config.video_dir / f"{pid}.mp4"
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(filepath, dest)
                        filepath = str(dest)

                    if not filepath:
                        item["status"] = "error"
                        yield queue, queue_html(queue), progress_str, status
                        continue

                    # Process
                    index = pipeline.process(filepath)
                    rag.index_video(index)
                    item["status"] = "done"

                except Exception as e:
                    logger.error(f"Batch item error: {e}")
                    item["status"] = "error"

                yield queue, queue_html(queue), progress_str, status

            busy.value = False
            status.value = '<span class="badge ready">● Batch complete</span>'
            final_progress = f'<p style="color:#34d399;font-size:0.85rem;">✅ Batch complete — {total} items processed.</p>'
            yield queue, queue_html(queue), final_progress, status

        def do_clear_batch():
            return (
                [],
                queue_html([]),
                '<p style="color:var(--text-muted);font-size:0.85rem;">Queue cleared.</p>',
            )

        # Wire events
        process_btn.click(
            fn=do_process,
            inputs=[video_input, vid],
            outputs=[
                progress_html,
                status,
                progress_panel,
                video_player,
                vid,
                vpath,
                export_group,
            ],
        )

        import_url_btn.click(
            fn=do_import_url,
            inputs=[url_input, vid],
            outputs=[
                progress_html,
                status,
                progress_panel,
                video_player,
                vid,
                vpath,
                export_group,
            ],
        )

        # Import tab events
        import_download_btn.click(
            fn=do_import_tab_download,
            inputs=[import_url_input],
            outputs=[
                import_video_player,
                import_progress_html,
                import_video_id,
                import_video_path,
                status,
            ],
        )

        import_clear_btn.click(
            fn=clear_import_tab,
            outputs=[
                import_url_input,
                import_video_player,
                import_progress_html,
                import_video_id,
                import_video_path,
            ],
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

        export_btn.click(
            do_export_clip,
            [clip_start, clip_end, vpath],
            [clip_output, clip_output],
        )

        clear_chat_btn.click(clear_chat, outputs=[chatbot, chatbot])
        clear_btn.click(
            reset_ui,
            outputs=[
                vid,
                vpath,
                video_input,
                video_player,
                clip_start,
                clip_end,
                export_group,
                status,
                chatbot,
                chatbot,
                clip_output,
                clip_output,
            ],
        )
        refresh_lib_btn.click(refresh_library, outputs=[library_list, delete_status])

        # --- Cross-video semantic search ---
        def do_search_all(query: str):
            """Cross-video semantic search across the entire library."""
            if not query or not query.strip():
                return (
                    '<p style="color:var(--text-muted);">Enter a query to search.</p>',
                    "",
                    '<p style="color:var(--text-muted);">Enter a query to search.</p>',
                )
            try:
                chunks = rag.search_all(query.strip(), top_k=15)
                if not chunks:
                    return (
                        '<p style="color:var(--text-muted);">No results found.</p>',
                        "",
                        '<p style="color:#f59e0b;">No matching scenes found.</p>',
                    )

                # Group by video_id
                from collections import defaultdict

                by_video = defaultdict(list)
                for c in chunks:
                    by_video[c.video_id].append(c)

                # Build HTML results
                html_parts = []
                total = len(chunks)
                html_parts.append(
                    f'<p style="color:#34d399;margin-bottom:1rem;">'
                    f"Found {total} matching chunk(s) across {len(by_video)} video(s)</p>"
                )
                for vid, vid_chunks in sorted(by_video.items()):
                    first = vid_chunks[0]
                    fname = first.metadata.get("filename", vid) if first.metadata else vid
                    html_parts.append(
                        f'<div class="library-card" style="margin-bottom:0.75rem;">'
                        f'<h4 style="margin:0 0 0.25rem 0;">📹 {fname}</h4>'
                        f'<p style="margin:0 0 0.5rem 0;font-size:0.85rem;color:var(--text-muted);">'
                        f"{len(vid_chunks)} match(es)</p>"
                    )
                    for c in vid_chunks[:5]:  # max 5 per video
                        ts = format_timestamp(c.timestamp)
                        score_pct = f"{c.score * 100:.0f}%"
                        preview = c.text[:200].replace("\n", " ")
                        html_parts.append(
                            f'<details style="margin:0.25rem 0;font-size:0.85rem;">'
                            f'<summary style="cursor:pointer;color:var(--link);">'
                            f"⏱ {ts} — relevance: {score_pct}</summary>"
                            f'<p style="margin:0.25rem 0 0 1rem;color:var(--text-muted);'
                            f'white-space:pre-wrap;">{preview}...</p>'
                            f"</details>"
                        )
                    if len(vid_chunks) > 5:
                        html_parts.append(
                            f'<p style="font-size:0.8rem;color:var(--text-muted);'
                            f'margin-left:1rem;">… and {len(vid_chunks) - 5} more</p>'
                        )
                    html_parts.append("</div>")

                return (
                    gr.update(value="\n".join(html_parts)),
                    "",
                    '<p style="color:#34d399;">✅ Search complete.</p>',
                )
            except Exception as e:
                logger.error(f"Search all error: {e}", exc_info=True)
                return (
                    gr.update(),
                    "",
                    f'<p style="color:#ef4444;">❌ Search failed: {str(e)[:200]}</p>',
                )

        def do_clear_search():
            return (
                '<p style="color:var(--text-muted);padding:1rem;">No results yet.</p>',
                "",
                '<p style="color:var(--text-muted);">Enter a query and click Search.</p>',
            )

        search_btn.click(
            fn=do_search_all,
            inputs=[search_query],
            outputs=[search_results, search_detail, search_status],
        )
        search_query.submit(
            fn=do_search_all,
            inputs=[search_query],
            outputs=[search_results, search_detail, search_status],
        )
        search_clear_btn.click(
            fn=do_clear_search,
            outputs=[search_results, search_detail, search_status],
        )

        # Library search
        lib_search.change(
            fn=search_library,
            inputs=[lib_search],
            outputs=[library_list],
        )

        # Library delete video
        def do_delete_video_js(video_id: str, lib_state: str):
            """Called from JS bridge when delete button clicked."""
            result_html, msg = delete_library_video(video_id)
            return (
                result_html,
                msg,
                gr.update(visible=False),
                gr.update(visible=False),
                "",
            )

        delete_all_lib_btn.click(
            fn=delete_all_library,
            outputs=[
                library_list,
                delete_status,
                lib_video_player,
                lib_info,
                lib_video_id,
            ],
        )

        # Batch events
        add_batch_btn.click(
            do_add_to_queue,
            inputs=[batch_urls, batch_files, batch_queue],
            outputs=[batch_queue, batch_status, batch_overall_progress],
        )
        process_batch_btn.click(
            do_process_batch,
            inputs=[batch_queue],
            outputs=[batch_queue, batch_status, batch_overall_progress, status],
        )
        clear_batch_btn.click(
            do_clear_batch,
            outputs=[batch_queue, batch_status, batch_overall_progress],
        )

        # --- Library video selection ---
        def do_select_video(video_id: str):
            """Load a selected library video's info into the library tab."""
            if not video_id:
                return gr.update(visible=False), gr.update(visible=False), ""
            try:
                # Find the video file
                video_path = config.video_dir / f"{video_id}.mp4"
                if not video_path.exists():
                    return gr.update(visible=False), gr.update(visible=False), ""
                # Get video info from RAG
                info = {"video_id": video_id, "filepath": str(video_path)}
                duration = pipeline._get_duration(video_path)
                if duration > 0:
                    info["duration_seconds"] = duration
                return (
                    gr.update(visible=True, value=str(video_path)),
                    gr.update(visible=True, value=info),
                    video_id,
                )
            except Exception as e:
                logger.error(f"Library select error: {e}")
                return gr.update(visible=False), gr.update(visible=False), ""

        # JS bridge: when library card is clicked, it calls __selectVideo(videoId)
        # which sets a hidden input to trigger the Gradio event
        lib_select_js = gr.HTML(
            """<script>
(function() {
  window.__selectVideo = function(videoId) {
    const el = document.querySelector('#lib-select-input input, #lib-select-input textarea');
    if (el) {
      const nativeInputValueSetter = Object.getOwnPropertyDescriptor(
        window.HTMLInputElement.prototype, 'value'
      ).set;
      nativeInputValueSetter.call(el, videoId);
      el.dispatchEvent(new Event('input', { bubbles: true }));
    }
  };
})();
</script>""",
            visible=True,
        )

        # JS bridge: delete video from library
        lib_delete_js = gr.HTML(
            """<script>
(function() {
  window.__deleteVideo = function(videoId) {
    const el = document.querySelector('#lib-delete-input input, #lib-delete-input textarea');
    if (el) {
      const nativeInputValueSetter = Object.getOwnPropertyDescriptor(
        window.HTMLInputElement.prototype, 'value'
      ).set;
      nativeInputValueSetter.call(el, videoId);
      el.dispatchEvent(new Event('input', { bubbles: true }));
    }
  };
})();
</script>""",
            visible=True,
        )

        lib_select_input = gr.Textbox(value="", visible=False, elem_id="lib-select-input")
        lib_select_input.change(
            do_select_video,
            inputs=[lib_select_input],
            outputs=[lib_video_player, lib_info, lib_video_id],
        )

        lib_delete_input = gr.Textbox(value="", visible=False, elem_id="lib-delete-input")
        lib_delete_input.change(
            fn=do_delete_video_js,
            inputs=[lib_delete_input, lib_video_id],
            outputs=[
                library_list,
                delete_status,
                lib_video_player,
                lib_info,
                lib_video_id,
            ],
        )

        # ==================== TIMELINE HOVER JAVASCRIPT ====================
        # Injects JS that observes the video element and shows a popup preview
        # on timeline hover, using the sprite sheet as CSS background offset.
        gr.HTML(
            """
<script>
(function() {
  'use strict';

  // ── State ──
  let spriteMeta = null;
  let spriteUrl = null;
  let previewEl = null;
  let videoEl = null;
  let cleanupFns = [];

  function formatTime(sec) {
    const h = Math.floor(sec / 3600);
    const m = Math.floor((sec % 3600) / 60);
    const s = Math.floor(sec % 60);
    const ms = Math.floor((sec % 1) * 1000);
    return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}.${String(ms).padStart(3, '0')}`;
  }

  function createPreview() {
    if (previewEl) return;
    previewEl = document.createElement('div');
    previewEl.className = 'timeline-preview';
    previewEl.innerHTML = '<div class="tp-img-wrap"><img alt="" style="width:160px;height:90px;" /></div><div class="tp-time" style="text-align:center;font-family:monospace;font-size:0.8rem;padding:2px 4px;">00:00:00.000</div>';
    previewEl.style.cssText = 'position:fixed;z-index:9999;background:#1e1b2e;border:1px solid #3d3a50;border-radius:8px;padding:4px;box-shadow:0 4px 20px rgba(0,0,0,0.6);pointer-events:none;display:none;';
    document.body.appendChild(previewEl);
  }

  function destroyPreview() {
    if (previewEl) { previewEl.remove(); previewEl = null; }
  }

  function getThumbnailIndex(timestamp) {
    if (!spriteMeta || !spriteMeta.thumbnails || spriteMeta.thumbnails.length === 0) return -1;
    const thumbnails = spriteMeta.thumbnails;
    let lo = 0, hi = thumbnails.length - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >>> 1;
      if (thumbnails[mid].timestamp <= timestamp) lo = mid;
      else hi = mid - 1;
    }
    return lo;
  }

  function updatePreview(timestamp) {
    if (!previewEl || !spriteMeta || !spriteUrl) return;
    const idx = getThumbnailIndex(timestamp);
    if (idx < 0) { previewEl.style.display = 'none'; return; }

    const thumb = spriteMeta.thumbnails[idx];
    const img = previewEl.querySelector('img');
    const timeEl = previewEl.querySelector('.tp-time');

    img.style.background = `url(${spriteUrl}) no-repeat`;
    img.style.backgroundSize = `${spriteMeta.num_columns * spriteMeta.thumbnail_width}px ${spriteMeta.num_rows * spriteMeta.thumbnail_height}px`;
    img.style.backgroundPosition = `-${thumb.x}px -${thumb.y}px`;
    img.style.width = spriteMeta.thumbnail_width + 'px';
    img.style.height = spriteMeta.thumbnail_height + 'px';
    img.src = '';
    timeEl.textContent = formatTime(timestamp);
    previewEl.style.display = 'block';
  }

  // ── Shadow DOM penetration: find <video> inside any shadow root ──
  function findVideoElements(root) {
    const videos = [];
    // Check this root
    if (root.querySelectorAll) {
      root.querySelectorAll('video').forEach(v => videos.push(v));
    }
    // Recurse into shadow roots
    const allElements = root.querySelectorAll ? root.querySelectorAll('*') : [];
    allElements.forEach(el => {
      if (el.shadowRoot) {
        videos.push(...findVideoElements(el.shadowRoot));
      }
    });
    // If root itself is a shadow root, check direct children too
    if (root instanceof ShadowRoot) {
      root.querySelectorAll('video').forEach(v => {
        if (!videos.includes(v)) videos.push(v);
      });
    }
    return videos;
  }

  // ── Observe for video elements with shadow DOM support ──
  function scanForVideo() {
    const videos = findVideoElements(document);
    // Among all videos, find the one currently visible (in a video player tab)
    for (const v of videos) {
      const rect = v.getBoundingClientRect();
      if (rect.width > 100 && rect.height > 50) {
        attachToVideo(v);
        return;
      }
    }
    // Fallback: attach to first video found
    if (videos.length > 0 && !videoEl) {
      attachToVideo(videos[0]);
    }
  }

  function attachToVideo(video) {
    if (!video || video === videoEl) return;
    videoEl = video;

    cleanupFns.forEach(fn => fn());
    cleanupFns = [];

    function loadSpriteData() {
      const src = video.querySelector('source')?.src || video.src;
      if (!src) return;
      const segments = src.split('/');
      const filename = segments[segments.length - 1];
      const videoId = filename.replace(/[.]mp4$/, '').replace(/[.]webm$/, '').replace(/[.]mov$/, '');

      const attempts = [
        `/file=data/thumbnails/${videoId}_sprite.json`,
        `/file=/app/data/thumbnails/${videoId}_sprite.json`,
      ];
      const spriteAttempts = [
        `/file=data/thumbnails/${videoId}_sprite.jpg`,
        `/file=/app/data/thumbnails/${videoId}_sprite.jpg`,
      ];

      function tryFetch(index) {
        if (index >= attempts.length) return;
        fetch(attempts[index])
          .then(r => r.json())
          .then(meta => {
            spriteMeta = meta;
            spriteUrl = spriteAttempts[index];
            console.log('[TimelinePreview] Sprite loaded:', meta.num_thumbnails, 'thumbnails');
          })
          .catch(() => tryFetch(index + 1));
      }
      tryFetch(0);
    }

    // Listen to source changes on video
    const srcObserver = new MutationObserver(() => loadSpriteData());
    srcObserver.observe(video, { attributes: true, attributeFilter: ['src'] });
    srcObserver.observe(video.querySelector('source') || video, { childList: true, subtree: true });
    cleanupFns.push(() => srcObserver.disconnect());

    setTimeout(loadSpriteData, 500);

    // ── Timeline hover detection ──
    function onTimelineHover(e) {
      if (!spriteMeta) { if (previewEl) previewEl.style.display = 'none'; return; }

      const rect = video.getBoundingClientRect();
      const timelineArea = {
        left: rect.left + rect.width * 0.05,
        right: rect.right - rect.width * 0.05,
        top: rect.bottom - 30,
        bottom: rect.bottom
      };

      if (e.clientY >= timelineArea.top && e.clientY <= timelineArea.bottom &&
          e.clientX >= timelineArea.left && e.clientX <= timelineArea.right) {
        const fraction = Math.max(0, Math.min(1,
          (e.clientX - timelineArea.left) / (timelineArea.right - timelineArea.left)
        ));
        const timestamp = fraction * (spriteMeta.duration || video.duration || 0);

        createPreview();
        previewEl.style.left = `${e.clientX}px`;
        previewEl.style.top = `${timelineArea.top - 12}px`;
        previewEl.style.transform = 'translate(-50%, -100%)';
        updatePreview(timestamp);
      } else {
        if (previewEl) previewEl.style.display = 'none';
      }
    }

    function onTimelineLeave() {
      if (previewEl) previewEl.style.display = 'none';
    }

    // Attach event listeners to the containing shadow DOM host or parent
    let container = video;
    // Walk up to find the Gradio video wrapper (could be shadow host or light DOM parent)
    let el = video;
    while (el && el !== document.body) {
      if (el.tagName && el.tagName.toLowerCase().includes('video')) {
        el = el.parentElement || el.getRootNode().host || document.body;
        continue;
      }
      container = el;
      break;
    }

    container.addEventListener('mousemove', onTimelineHover, { passive: true });
    container.addEventListener('mouseleave', onTimelineLeave);
    cleanupFns.push(() => {
      container.removeEventListener('mousemove', onTimelineHover);
      container.removeEventListener('mouseleave', onTimelineLeave);
    });
  }

  // ── Initial scan ──
  scanForVideo();

  // ── Periodic scan for new video elements (handles Gradio tab switches) ──
  const observer = new MutationObserver(() => {
    const cv = videoEl;
    const rect = cv ? cv.getBoundingClientRect() : { width: 0, height: 0 };
    // Only re-scan if the current video disappeared or was resized to 0
    if (!cv || rect.width === 0 || rect.height === 0) {
      scanForVideo();
    }
  });
  observer.observe(document.body, { childList: true, subtree: true, attributes: false });
  cleanupFns.push(() => observer.disconnect());

  // Also poll periodically for shadow DOM elements (Gradio lazy-renders tabs)
  let pollTimer = setInterval(scanForVideo, 2000);
  cleanupFns.push(() => clearInterval(pollTimer));

  // Cleanup on page unload
  window.addEventListener('beforeunload', () => {
    cleanupFns.forEach(fn => fn());
    destroyPreview();
  });
})();
</script>
""",
            visible=True,
        )

    return app


def launch(
    config: Optional[Config] = None,
    *,
    no_health: bool = False,
):
    """Launch the Gradio UI with FastAPI health/API endpoints.

    Uses gr.mount_gradio_app() to mount the Gradio interface at ``/`` on a
    FastAPI app, leaving ``/health`` and ``/api/*`` accessible directly.

    Args:
        config: Application configuration.
        no_health: If True, fall back to plain Gradio ``.launch()`` without
            the FastAPI wrapper.
    """
    cfg = config or Config()

    if no_health:
        logger.info(f"Starting UI (no-health) on http://{cfg.ui_host}:{cfg.ui_port}")
        gradio_app = build(cfg)
        gradio_app.launch(
            server_name=cfg.ui_host,
            server_port=cfg.ui_port,
            share=cfg.ui_share,
            show_error=True,
            quiet=False,
            css=gradio_app.css,
            theme=gradio_app.theme,
        )
        return

    from ui.health import create_health_app

    gradio_app = build(cfg)
    health_app = create_health_app(cfg)
    app = gr.mount_gradio_app(health_app, gradio_app, path="/")

    logger.info(f"Starting UI on http://{cfg.ui_host}:{cfg.ui_port} (health API at /health)")

    import uvicorn

    uvicorn.run(
        app,
        host=cfg.ui_host,
        port=cfg.ui_port,
        log_level="info",
    )


if __name__ == "__main__":
    launch()
