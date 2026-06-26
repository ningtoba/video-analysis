"""
Gradio-based UI for the video analysis platform.

Features:
- Video upload with drag-and-drop
- Video player with timeline + thumbnail preview
- Analysis progress with real-time updates
- Chat interface with source citations and clickable timestamps
- Clip export (jump to precise moments)
- Multi-video library management
- Dark theme, responsive layout
"""

import json
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional, List

import gradio as gr

from video_analysis.config import Config
from video_analysis.pipeline import VideoPipeline
from video_analysis.rag import VideoRAG
from video_analysis.chat import VideoChat
from video_analysis.models import format_timestamp, VideoIndex

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


def _video_summary(index: VideoIndex) -> str:
    scenes = len(index.scenes)
    dur = index.duration
    objs = sum(len(f.objects) for s in index.scenes for f in s.key_frames)
    descs = sum(1 for s in index.scenes for f in s.key_frames if f.description)
    return f"{scenes} scenes, {objs} objects, {descs} described frames, {dur:.0f}s"


def _library_html(video_ids: List[str], rag: VideoRAG) -> str:
    """Render library cards as HTML with click-to-select."""
    if not video_ids:
        return '<p style="color:var(--text-muted);padding:1rem;">No videos analyzed yet. Upload one above.</p>'
    html = ""
    for vid in video_ids:
        html += f'<div class="library-card" onclick="window.__selectVideo(\'{vid}\')">'
        html += f'<div class="title">{vid}</div>'
        html += f'<div class="meta">ID: {vid}</div>'
        html += "</div>"
    return html


def build(config: Optional[Config] = None) -> gr.Blocks:
    """Build the Gradio application."""
    config = config or Config()
    pipeline = VideoPipeline(config)
    rag = VideoRAG(config)
    chat_session = VideoChat(rag, config)

    with gr.Blocks(
        css=CSS + LIBRARY_CSS,
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

        with gr.Tabs():
            # ============ TAB 1: ANALYSIS ============
            with gr.TabItem("📹 Analysis", id="analysis"):
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

                        video_player = gr.Video(
                            label="Video Player", visible=False, height=400
                        )

                        # Timeline hover preview — JavaScript injected as HTML
                        timeline_preview_js = gr.HTML(
                            """<div id="timeline-hover-root"></div>""", visible=True
                        )

                        # Sprite sheet + clip export
                        with gr.Group(visible=False) as export_group:
                            gr.Markdown("### 🎬 Export Clip")
                            with gr.Row():
                                clip_start = gr.Number(
                                    label="Start (seconds)", value=0, minimum=0
                                )
                                clip_end = gr.Number(
                                    label="End (seconds)", value=10, minimum=0
                                )
                            export_btn = gr.Button(
                                "✂️ Export Clip", variant="secondary", size="sm"
                            )
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
                        show_sources = gr.Checkbox(
                            label="Show source citations", value=True
                        )

            # ============ TAB 2: LIBRARY ============
            with gr.TabItem("📚 Library", id="library"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### Your Video Library")
                        library_list = gr.HTML(
                            '<p style="color:var(--text-muted);padding:1rem;">No videos analyzed yet. Upload one above.</p>'
                        )
                        refresh_lib_btn = gr.Button("🔄 Refresh Library", size="sm")
                    with gr.Column(scale=3):
                        gr.Markdown("### Video Details")
                        lib_video_id = gr.State("")
                        lib_video_player = gr.Video(
                            label="Selected Video", visible=False
                        )
                        lib_info = gr.JSON(label="Video Info", visible=False)

        # ==================== EVENT HANDLERS ====================

        # --- Process Video ---
        def do_process(video_path: str, state_vid: str):
            if not video_path or busy.value:
                return (
                    state_vid,
                    state_vid,
                    None,
                    gr.update(visible=False),
                    None,
                    None,
                    gr.update(visible=False),
                    status,
                    progress_html,
                    gr.update(visible=False),
                    gr.update(visible=False),
                )

            busy.value = True
            status.value = '<span class="badge busy">● Processing</span>'

            try:
                yield (
                    _progress_html(0, "Starting..."),
                    status,
                    True,
                    None,
                    None,
                    None,
                    gr.update(visible=False),
                    vid,
                    vpath,
                    gr.update(visible=False),
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
                    True,
                    None,
                    None,
                    None,
                    gr.update(visible=False),
                    vid,
                    vpath,
                    gr.update(visible=False),
                    gr.update(visible=False),
                )

                index = pipeline.process(str(dest))
                video_path_str = str(dest)

                yield (
                    _progress_html(6, "Indexing content for Q&A..."),
                    status,
                    True,
                    None,
                    None,
                    None,
                    gr.update(visible=False),
                    vid,
                    vpath,
                    gr.update(visible=False),
                    gr.update(visible=False),
                )

                rag.index_video(index)
                chat_session.reset_history()

                status.value = (
                    '<span class="badge ready">● Ready — ask questions</span>'
                )
                yield (
                    _progress_html(8, f"✅ Complete — {_video_summary(index)}"),
                    status,
                    True,
                    video_path_str,
                    pid,
                    video_path_str,
                    gr.update(visible=True),
                    pid,
                    video_path_str,
                    gr.update(visible=True),
                    gr.update(visible=True),
                )

            except Exception as e:
                logger.error(f"Process error: {e}", exc_info=True)
                status.value = (
                    f'<span class="badge error">● Error: {str(e)[:100]}</span>'
                )
                yield (
                    _progress_html(-1, f"❌ {str(e)[:200]}"),
                    status,
                    True,
                    None,
                    state_vid,
                    state_vid,
                    gr.update(visible=False),
                    state_vid,
                    state_vid,
                    gr.update(visible=False),
                    gr.update(visible=False),
                )
            finally:
                busy.value = False

        # --- Send Chat Message ---
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
                html = _library_html(video_ids, rag)
                return html
            except Exception as e:
                logger.error(f"Library refresh error: {e}")
                return f'<p style="color:var(--text-muted);">Error loading library: {str(e)[:100]}</p>'

        # Wire events
        evt = process_btn.click(
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
                vid,
                vpath,
                progress_panel,
                export_group,
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
        refresh_lib_btn.click(refresh_library, outputs=[library_list])

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
        lib_select_input = gr.Textbox(
            value="", visible=False, elem_id="lib-select-input"
        )
        lib_select_input.change(
            do_select_video,
            inputs=[lib_select_input],
            outputs=[lib_video_player, lib_info, lib_video_id],
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
    previewEl.innerHTML = '<div class="tp-img-wrap"><img alt="" /></div><div class="tp-time">00:00:00.000</div>';
    document.body.appendChild(previewEl);
  }

  function destroyPreview() {
    if (previewEl) { previewEl.remove(); previewEl = null; }
  }

  function getThumbnailIndex(timestamp) {
    if (!spriteMeta || !spriteMeta.thumbnails || spriteMeta.thumbnails.length === 0) return -1;
    const thumbnails = spriteMeta.thumbnails;
    // Binary search for closest
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

    // Use sprite sheet as CSS background with offset
    img.style.background = `url(${spriteUrl}) no-repeat`;
    img.style.backgroundSize = `${spriteMeta.num_columns * spriteMeta.thumbnail_width}px ${spriteMeta.num_rows * spriteMeta.thumbnail_height}px`;
    img.style.backgroundPosition = `-${thumb.x}px -${thumb.y}px`;
    img.src = ''; // clear src to show background
    // Fallback: set src to sprite sheet (browsers will show top-left without bg-position)
    // We rely on background rendering via the div style above.
    timeEl.textContent = formatTime(timestamp);
    previewEl.style.display = 'block';
  }

  function attachToVideo(video) {
    if (!video || video === videoEl) return;
    videoEl = video;

    // Clean previous listeners
    cleanupFns.forEach(fn => fn());
    cleanupFns = [];

    // Find sprite metadata URL: look for JSON next to sprite
    // The sprite sheet URL is derived from the video source name
    function loadSpriteData() {
      const src = video.querySelector('source')?.src || video.src;
      if (!src) return;
      // Derive sprite URL from video path
      // Convention: /data/thumbnails/{video_id}_sprite.jpg
      // We look up via the video_id embedded in the filename
      const baseUrl = src.substring(0, src.lastIndexOf('/'));
      const segments = src.split('/');
      const filename = segments[segments.length - 1];
      const videoId = filename.replace(/[.]mp4$/, '').replace(/[.]webm$/, '').replace(/[.]mov$/, '');

      // Try to fetch the metadata JSON
      // The sprite and JSON live at: /data/thumbnails/{videoId}_sprite.json relative to app
      const metaUrl = `/file=${baseUrl}/../../thumbnails/${videoId}_sprite.json`.replace(/\\/+/g, '/');
      const spriteImgUrl = `/file=${baseUrl}/../../thumbnails/${videoId}_sprite.jpg`.replace(/\\/+/g, '/');

      fetch(metaUrl)
        .then(r => r.json())
        .then(meta => {
          spriteMeta = meta;
          spriteUrl = spriteImgUrl;
          console.log('[TimelinePreview] Sprite loaded:', spriteMeta.num_thumbnails, 'thumbnails');
        })
        .catch(() => {
          // Try alternative: relative to data dir
          const altMetaUrl = `/file=data/thumbnails/${videoId}_sprite.json`;
          const altSpriteUrl = `/file=data/thumbnails/${videoId}_sprite.jpg`;
          fetch(altMetaUrl)
            .then(r => r.json())
            .then(meta => {
              spriteMeta = meta;
              spriteUrl = altSpriteUrl;
            })
            .catch(e => console.warn('[TimelinePreview] No sprite data found:', e.message));
        });
    }

    // Listen to source changes
    const srcObserver = new MutationObserver(() => loadSpriteData());
    srcObserver.observe(video, { attributes: true, attributeFilter: ['src'] });
    srcObserver.observe(video.querySelector('source') || video, { childList: true, subtree: true });
    cleanupFns.push(() => srcObserver.disconnect());

    // Initial load
    setTimeout(loadSpriteData, 500);

    // ── Timeline hover detection ──
    // Gradio 6 uses a custom <gradio-video> web component.
    // Instead of looking for input[type=range], we listen for
    // mousemove on the video container and use the video duration
    // to compute the time position.
    function findVideoRect() {
      const vid = video;
      if (!vid) return null;
      const controls = vid.closest('gradio-video') || vid.parentElement;
      if (!controls) return vid.getBoundingClientRect();
      return controls.getBoundingClientRect();
    }

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

    // Attach to video container directly
    const videoContainer = video.closest('gradio-video') || video.parentElement || video;
    videoContainer.addEventListener('mousemove', onTimelineHover, { passive: true });
    videoContainer.addEventListener('mouseleave', onTimelineLeave);
    cleanupFns.push(() => {
      videoContainer.removeEventListener('mousemove', onTimelineHover);
      videoContainer.removeEventListener('mouseleave', onTimelineLeave);
    });
  }

  // ── Observe for video elements ──
  const observer = new MutationObserver(() => {
    const videos = document.querySelectorAll('.gradio-video video');
    if (videos.length > 0) {
      attachToVideo(videos[0]);
    }
  });
  observer.observe(document.body, { childList: true, subtree: true });
  cleanupFns.push(() => observer.disconnect());

  // Also check immediately
  setTimeout(() => {
    const videos = document.querySelectorAll('.gradio-video video');
    if (videos.length > 0) attachToVideo(videos[0]);
  }, 1000);

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
