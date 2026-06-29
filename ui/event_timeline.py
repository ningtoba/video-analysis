"""Event-Causal Timeline — Gradio tab for event timeline, causal chains, and streaming thinking (v0.58.0).

Provides a Gradio interface for browsing video events, exploring causal/temporal
chains, and visualising streaming thinking results — all surfaced from
EventCausalRAG and the KnowledgeGraph.

Usage:
    from ui.event_timeline import inject_event_timeline_tab

    with gr.Blocks() as app:
        with gr.Tabs():
            inject_event_timeline_tab(app, config)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import gradio as gr

from video_analysis.config import Config
from video_analysis.rag import VideoRAG

logger = logging.getLogger(__name__)

EVENT_CSS = """
.evt-card {
    background: var(--surface, #1e1b2e);
    border: 1px solid var(--border, #3d3a50);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    transition: border-color 0.2s;
}
.evt-card:hover { border-color: var(--primary, #7c3aed); }
.evt-card .label {
    color: var(--text-muted, #9895b0);
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.evt-card .event-row {
    padding: 0.5rem 0;
    border-bottom: 1px solid var(--border, #3d3a50);
    font-size: 0.9rem;
}
.evt-card .event-row:last-child { border-bottom: none; }
.evt-tag {
    display: inline-block;
    background: rgba(124,58,237,0.15);
    color: #a78bfa;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    font-size: 0.8rem;
    margin: 0.15rem;
}
.evt-causal {
    display: inline-block;
    background: rgba(239,68,68,0.15);
    color: #f87171;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    font-size: 0.8rem;
    margin: 0.15rem;
}
.evt-temporal {
    display: inline-block;
    background: rgba(59,130,246,0.15);
    color: #60a5fa;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    font-size: 0.8rem;
    margin: 0.15rem;
}
.evt-arrow {
    color: var(--text-muted, #9895b0);
    margin: 0 0.3rem;
}
"""


def _events_html(
    events: List[Dict[str, Any]],
    title: str = "Events",
) -> str:
    """Render events as an HTML timeline card."""
    if not events:
        return f'<div class="evt-card"><div class="label">{title}</div><p style="color:var(--text-muted)">No events available. Process a video with EVENT_CAUSAL_RAG_ENABLED=true to see events here.</p></div>'

    html = f'<div class="evt-card"><div class="label">{title} ({len(events)})</div><div style="margin-top:0.5rem">'
    for evt in events:
        title_txt = evt.get("title", "?")
        start = evt.get("start_time", 0)
        end = evt.get("end_time", 0)
        desc = evt.get("description", "")
        state_before = evt.get("state_before", "")
        state_after = evt.get("state_after", "")
        action = evt.get("action", "")
        entities = evt.get("entities", [])
        confidence = evt.get("confidence", 0.0)

        entity_tags = " ".join(
            f'<span class="evt-tag">{e}</span>' for e in entities[:8]
        )
        html += (
            f'<div class="event-row">'
            f"<strong>{title_txt}</strong> "
            f'<span style="color:var(--text-muted);font-size:0.8rem">'
            f"[{start:.1f}s - {end:.1f}s] (conf={confidence:.2f})</span><br>"
        )
        if desc:
            html += f'<span style="font-size:0.85rem;color:var(--text)">{desc[:200]}</span><br>'
        if action:
            html += f'<span style="font-size:0.85rem"><strong>Action:</strong> {action}</span><br>'
        if state_before:
            html += f'<span style="font-size:0.8rem;color:var(--text-muted)">State before: {state_before[:120]}</span><br>'
        if state_after:
            html += f'<span style="font-size:0.8rem;color:var(--text-muted)">State after: {state_after[:120]}</span><br>'
        if entity_tags:
            html += f'<div style="margin-top:0.25rem">{entity_tags}</div>'
        html += "</div>"
    html += "</div></div>"
    return html


def _causal_relations_html(
    relations: List[Dict[str, Any]],
    events_by_id: Dict[str, Dict[str, Any]],
) -> str:
    """Render causal/temporal relations as an HTML chain visualization."""
    if not relations:
        return '<div class="evt-card"><div class="label">Causal Relations</div><p style="color:var(--text-muted)">No causal relations yet.</p></div>'

    html = '<div class="evt-card"><div class="label">Causal / Temporal Chains</div><div style="margin-top:0.5rem">'
    for rel in relations:
        src_id = rel.get("source_event_id", "")
        tgt_id = rel.get("target_event_id", "")
        rel_type = rel.get("relation_type", "temporal")
        strength = rel.get("strength", 0.0)

        src_title = events_by_id.get(src_id, {}).get("title", src_id[-20:])
        tgt_title = events_by_id.get(tgt_id, {}).get("title", tgt_id[-20:])

        tag_class = "evt-causal" if rel_type == "causal" else "evt-temporal"
        arrow = "⏩" if rel_type == "causal" else "→"

        html += (
            f'<div class="event-row">'
            f"<strong>{src_title}</strong> "
            f'<span class="evt-arrow">{arrow}</span> '
            f"<strong>{tgt_title}</strong> "
            f'<span class="{tag_class}">{rel_type}</span> '
            f'<span style="color:var(--text-muted);font-size:0.8rem">strength={strength:.1f}</span>'
            f"</div>"
        )
    html += "</div></div>"
    return html


def _streaming_thoughts_html() -> str:
    """Placeholder for streaming thinking timeline visualization."""
    return (
        '<div class="evt-card"><div class="label">Streaming Thinking</div>'
        '<p style="color:var(--text-muted)">Streaming thinking timeline will be shown here '
        "when streaming_thinking_enabled=True and a video is being streamed.</p>"
        '<p style="font-size:0.85rem;color:var(--text-muted)">'
        "StreamingThinkingPipeline produces per-chunk thoughts, entity accumulations, "
        "causal observations, and partial answers. This tab will display them as a "
        "live-updating timeline.</p></div>"
    )


def inject_event_timeline_tab(app: gr.Blocks, config: Config) -> None:
    """Inject the Event-Causal Timeline tab into a Gradio app.

    Args:
        app: The Gradio Blocks app (with Tabs context open).
        config: Application configuration.
    """
    rag = VideoRAG(config)
    _kg = None
    if config.event_causal_rag_enabled:
        try:
            from video_analysis.knowledge_graph import KnowledgeGraph

            _kg = KnowledgeGraph(config)
        except Exception:
            pass

    with gr.TabItem("⏱ Events", id="event-timeline"):
        gr.HTML(EVENT_CSS, visible=False)

        # Video selector
        with gr.Row():
            video_dropdown = gr.Dropdown(
                choices=[""] + _get_video_list(rag),
                label="Select Video",
                value="",
                interactive=True,
            )
            refresh_videos_btn = gr.Button("🔄 Refresh Videos", size="sm", scale=1)
            refresh_events_btn = gr.Button(
                "🔄 Refresh Events", variant="primary", scale=1
            )

        with gr.Group():
            gr.Markdown("### 📋 Event Timeline")
            event_timeline_html = gr.HTML(_initial_events_html(config, _kg, rag))

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🔗 Causal / Temporal Chains")
                causal_html = gr.HTML(_initial_causal_html(_kg))
            with gr.Column(scale=1):
                gr.Markdown("### 💭 Streaming Thinking")
                thinking_html = gr.HTML(_streaming_thoughts_html())

        # Event-level query
        with gr.Group():
            gr.Markdown("### 🔍 Event-Level Query")
            with gr.Row():
                event_query_input = gr.Textbox(
                    label="Question",
                    placeholder="e.g., What caused this event? What happens next?",
                    lines=2,
                    scale=3,
                )
                event_query_btn = gr.Button(
                    "Ask (Event-Causal)", variant="primary", scale=1
                )
            event_query_output = gr.HTML(
                '<div class="evt-card"><p style="color:var(--text-muted)">'
                "Enter a question about events and their causal relationships.</p></div>"
            )

        # Wire up video dropdown change
        def _on_video_change(video_id: str):
            return _build_events_view(video_id, config, rag, _kg)

        video_dropdown.change(
            fn=_on_video_change,
            inputs=[video_dropdown],
            outputs=[event_timeline_html, causal_html],
        )

        # Wire up refresh buttons
        def _refresh_videos():
            return gr.update(choices=_get_video_list(rag))

        refresh_videos_btn.click(
            fn=_refresh_videos,
            inputs=[],
            outputs=[video_dropdown],
        )

        def _refresh_events(video_id: str):
            return _build_events_view(video_id, config, rag, _kg)

        refresh_events_btn.click(
            fn=_refresh_events,
            inputs=[video_dropdown],
            outputs=[event_timeline_html, causal_html],
        )

        # Wire up event-level query
        def _do_event_query(video_id: str, query: str):
            if not query.strip():
                return _empty_query_html()
            if not config.event_causal_rag_enabled:
                return _disabled_html()
            try:
                chunks = rag.event_retrieve(query, video_id=video_id)
                if not chunks:
                    return (
                        '<div class="evt-card"><p style="color:var(--text-muted)">'
                        "No relevant events found for this query.</p></div>"
                    )
                html = '<div class="evt-card"><div class="label">Event-Level Results</div><div style="margin-top:0.5rem">'
                for c in chunks[:10]:
                    meta = c.metadata or {}
                    evt_id = meta.get("event_id", c.chunk_id)
                    evt_title = meta.get("event_title", "Event")
                    retrieval_type = meta.get("retrieval_type", "semantic")
                    causal_summary = meta.get("causal_path_summary", "")
                    rel_tag = (
                        "evt-causal" if "causal" in retrieval_type else "evt-temporal"
                    )
                    html += (
                        f'<div class="event-row">'
                        f"<strong>{evt_title}</strong> "
                        f'<span class="{rel_tag}">{retrieval_type}</span> '
                        f'<span style="color:var(--text-muted);font-size:0.8rem">'
                        f"[{c.timestamp:.1f}s] score={c.score:.3f}</span><br>"
                        f'<span style="font-size:0.85rem">{c.text[:300]}</span>'
                    )
                    if causal_summary:
                        html += f'<br><span style="font-size:0.8rem;color:#f87171">Causal: {causal_summary}</span>'
                    html += "</div>"
                html += "</div></div>"
                return html
            except Exception as e:
                return f'<div class="evt-card"><p style="color:#ef4444">Error: {e}</p></div>'

        event_query_btn.click(
            fn=_do_event_query,
            inputs=[video_dropdown, event_query_input],
            outputs=[event_query_output],
        )


# ── Helpers ───────────────────────────────────────────────────────────────


def _get_video_list(rag: VideoRAG) -> List[str]:
    """Get sorted list of video IDs from the RAG index."""
    try:
        return rag.list_videos()
    except Exception:
        return []


def _initial_events_html(
    config: Config,
    kg,
    rag: VideoRAG,
) -> str:
    """Get initial events HTML (latest video or empty)."""
    if not config.event_causal_rag_enabled or kg is None:
        return (
            '<div class="evt-card"><div class="label">Event Timeline</div>'
            '<p style="color:var(--text-muted)">Event-Causal RAG is disabled. '
            "Set EVENT_CAUSAL_RAG_ENABLED=true and process a video to see events.</p></div>"
        )
    try:
        videos = rag.list_videos()
        if videos:
            return _build_events_view(videos[0], config, rag, kg)[0]
    except Exception as exc:
        logger.warning("Failed to load initial events: %s", exc)
    return (
        '<div class="evt-card"><div class="label">Event Timeline</div>'
        '<p style="color:var(--text-muted)">No videos indexed yet.</p></div>'
    )


def _initial_causal_html(kg) -> str:
    """Get initial causal relations HTML."""
    if kg is None:
        return _causal_relations_html([], {})
    try:
        rels = kg.get_causal_relations(limit=50)
        return _causal_relations_html(rels, {})
    except Exception as exc:
        logger.warning("Failed to load initial causal relations: %s", exc)
        return _causal_relations_html([], {})


def _build_events_view(
    video_id: str,
    config: Config,
    rag: VideoRAG,
    kg,
) -> tuple:
    """Build events and causal relations HTML for a video."""
    if not video_id or not config.event_causal_rag_enabled or kg is None:
        return (
            '<div class="evt-card"><div class="label">Event Timeline</div>'
            '<p style="color:var(--text-muted)">Select a video to view its events.</p></div>',
            _causal_relations_html([], {}),
        )

    events_data = []
    causal_relations = []
    try:
        events_data = kg.get_events_for_video(video_id)
        causal_relations = kg.get_causal_relations(video_id=video_id, limit=100)
    except Exception as e:
        logger.warning("Failed to load events for %s: %s", video_id, e)

    # Build event lookup
    events_by_id = {e["event_id"]: e for e in events_data}

    evt_html = _events_html(events_data, title=f"Events for {video_id}")
    causal_html = _causal_relations_html(causal_relations, events_by_id)
    return evt_html, causal_html


def _empty_query_html() -> str:
    return '<div class="evt-card"><p style="color:var(--text-muted)">Enter a question first.</p></div>'


def _disabled_html() -> str:
    return (
        '<div class="evt-card"><p style="color:var(--text-muted)">'
        "Event-Causal RAG is not enabled. Set EVENT_CAUSAL_RAG_ENABLED=true "
        "and process a video.</p></div>"
    )
