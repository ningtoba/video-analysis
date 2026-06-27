"""Knowledge Graph Explorer — Gradio tab for browsing the persistent knowledge graph.

Provides a Gradio interface for exploring entities, relationships, timelines,
and statistics from the cross-video knowledge graph (v0.52.0).

Usage:
    from ui.knowledge_graph import inject_knowledge_graph_tab

    with gr.Blocks() as app:
        with gr.Tabs():
            inject_knowledge_graph_tab(app, config)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import gradio as gr

from video_analysis.config import Config
from video_analysis.knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)

KG_DARK_CSS = """
.kg-card {
    background: var(--surface, #1e1b2e);
    border: 1px solid var(--border, #3d3a50);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    transition: border-color 0.2s;
}
.kg-card:hover { border-color: var(--primary, #7c3aed); }
.kg-card .label {
    color: var(--text-muted, #9895b0);
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.kg-card .value {
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--text, #e2e0f0);
    margin: 0.25rem 0;
}
.kg-card .entity-tag {
    display: inline-block;
    background: rgba(124,58,237,0.15);
    color: #a78bfa;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    font-size: 0.8rem;
    margin: 0.15rem;
}
.kg-card .rel-item {
    padding: 0.3rem 0;
    border-bottom: 1px solid var(--border, #3d3a50);
    font-size: 0.9rem;
}
"""


def _kg_stats_html(kg: KnowledgeGraph) -> str:
    """Render knowledge graph stats as HTML."""
    stats = kg.stats()
    html = '<div class="kg-card">'
    html += '<div class="label">Knowledge Graph Summary</div>'
    html += (
        f'<div class="value">{stats.get("entity_count", 0)}</div>'
        f'<div style="color:var(--text-muted)">Total Entities</div>'
    )
    html += '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-top:1rem">'
    html += f'<div><div class="label">Relationships</div><div style="font-size:1.25rem;font-weight:600">{stats.get("relationship_count", 0)}</div></div>'
    html += f'<div><div class="label">Videos Indexed</div><div style="font-size:1.25rem;font-weight:600">{stats.get("video_count", 0)}</div></div>'
    db_size = stats.get("database_size_bytes", 0)
    if db_size > 1024 * 1024:
        size_str = f"{db_size / 1024 / 1024:.1f} MB"
    elif db_size > 1024:
        size_str = f"{db_size / 1024:.1f} KB"
    else:
        size_str = f"{db_size} B"
    html += f'<div><div class="label">Database Size</div><div style="font-size:1.25rem;font-weight:600">{size_str}</div></div>'
    html += "</div>"

    # Type breakdown
    type_breakdown = stats.get("type_breakdown", {})
    if type_breakdown:
        html += '<div style="margin-top:1rem"><div class="label">Entity Types</div><div style="margin-top:0.5rem">'
        for etype, count in sorted(type_breakdown.items(), key=lambda x: -x[1]):
            html += f'<span class="entity-tag">{etype}: {count}</span> '
        html += "</div></div>"

    last_vid = stats.get("last_indexed_video")
    if last_vid:
        html += f'<div style="margin-top:0.75rem;color:var(--text-muted);font-size:0.8rem">Last indexed: {last_vid.get("video_id", "?")}</div>'
    html += "</div>"
    return html


def _entities_html(kg: KnowledgeGraph, entity_type: str = "", query: str = "") -> str:
    """Render top entities as HTML."""
    if query:
        entities = kg.cross_video_search(query, limit=50)
    elif entity_type:
        entities = kg.search_entities(entity_type=entity_type, limit=50)
    else:
        entities = kg.get_top_entities(limit=50)

    if not entities:
        return '<div class="kg-card"><p style="color:var(--text-muted)">No entities found.</p></div>'

    html = '<div class="kg-card">'
    html += f'<div class="label">Entities ({len(entities)})</div>'
    html += '<div style="margin-top:0.5rem">'
    for e in entities[:30]:
        html += (
            f'<div class="rel-item">'
            f"<strong>{e.name}</strong> "
            f'<span class="entity-tag">{e.entity_type}</span> '
            f'<span style="color:var(--text-muted);font-size:0.8rem">seen {e.frequency}x in {len(e.video_ids)} video(s)</span>'
            f"</div>"
        )
    if len(entities) > 30:
        html += f'<div style="color:var(--text-muted);font-size:0.8rem;margin-top:0.5rem">... and {len(entities) - 30} more</div>'
    html += "</div></div>"
    return html


def _timeline_html(kg: KnowledgeGraph) -> str:
    """Render video timeline as HTML."""
    timeline = kg.get_timeline(limit=50)
    if not timeline:
        return '<div class="kg-card"><p style="color:var(--text-muted)">No videos indexed yet.</p></div>'

    html = '<div class="kg-card">'
    html += f'<div class="label">Video Timeline ({len(timeline)} videos)</div>'
    html += '<div style="margin-top:0.5rem">'
    for item in timeline:
        top = item.get("top_entities", [])
        entity_tags = " ".join(f'<span class="entity-tag">{e}</span>' for e in top[:5])
        html += (
            f'<div class="rel-item">'
            f'<strong>{item.get("filename", "?")}</strong> '
            f'<span style="color:var(--text-muted);font-size:0.8rem">'
            f'({item.get("duration_seconds", 0):.0f}s, {item.get("entity_count", 0)} entities)'
            f"</span><br>"
            f"{entity_tags}"
            f"</div>"
        )
    html += "</div></div>"
    return html


def _relationships_html(kg: KnowledgeGraph) -> str:
    """Render top relationships as HTML."""
    rels = kg.get_top_relationships(limit=30)
    if not rels:
        return '<div class="kg-card"><p style="color:var(--text-muted)">No relationships yet.</p></div>'

    html = '<div class="kg-card">'
    html += f'<div class="label">Strongest Relationships ({len(rels)})</div>'
    html += '<div style="margin-top:0.5rem">'
    for r in rels:
        src = kg.get_entity(r.source_id)
        tgt = kg.get_entity(r.target_id)
        src_name = src.name if src else f"#{r.source_id}"
        tgt_name = tgt.name if tgt else f"#{r.target_id}"
        html += (
            f'<div class="rel-item">'
            f"<strong>{src_name}</strong> → <strong>{tgt_name}</strong> "
            f'<span class="entity-tag">{r.relation_type}</span> '
            f'<span style="color:var(--text-muted);font-size:0.8rem">strength {r.strength}</span>'
            f"</div>"
        )
    html += "</div></div>"
    return html


def _knowledge_context_html(kg: KnowledgeGraph) -> str:
    """Render the LLM-friendly knowledge context as formatted text."""
    context = kg.get_knowledge_context()
    if not context:
        return '<div class="kg-card"><p style="color:var(--text-muted)">No knowledge context available.</p></div>'
    return f'<div class="kg-card"><pre style="white-space:pre-wrap;font-size:0.85rem;color:var(--text);max-height:400px;overflow-y:auto">{context}</pre></div>'


def inject_knowledge_graph_tab(app: gr.Blocks, config: Config) -> None:
    """Inject the Knowledge Graph Explorer tab into a Gradio app.

    Args:
        app: The Gradio Blocks app (with Tabs context open).
        config: Application configuration.
    """
    kg = KnowledgeGraph(config)

    with gr.TabItem("🧠 Knowledge Graph", id="knowledge-graph"):
        gr.HTML(KG_DARK_CSS, visible=False)

        with gr.Row():
            with gr.Column(scale=2):
                stats_html = gr.HTML(_kg_stats_html(kg))
                timeline_html_display = gr.HTML(_timeline_html(kg))
            with gr.Column(scale=1):
                # Entity search controls
                with gr.Group():
                    gr.Markdown("### 🔍 Search Entities")
                    entity_type_input = gr.Dropdown(
                        choices=[
                            "",
                            "person",
                            "object",
                            "action",
                            "location",
                            "concept",
                            "event",
                        ],
                        label="Entity Type",
                        value="",
                    )
                    query_input = gr.Textbox(
                        label="Search Query (name or type)",
                        placeholder="e.g. Alice, car, running...",
                        value="",
                    )
                    search_btn = gr.Button("Search", variant="primary")

                relationships_html_display = gr.HTML(_relationships_html(kg))
                context_html_display = gr.HTML(_knowledge_context_html(kg))

        # Entity results area (initially empty)
        entities_html_display = gr.HTML(_entities_html(kg))

        # Wire up search button
        def _do_search(entity_type: str, query: str):
            return _entities_html(kg, entity_type, query)

        search_btn.click(
            fn=_do_search,
            inputs=[entity_type_input, query_input],
            outputs=[entities_html_display],
        )

        # Refresh all displays
        def _refresh():
            return (
                _kg_stats_html(kg),
                _timeline_html(kg),
                _entities_html(kg),
                _relationships_html(kg),
                _knowledge_context_html(kg),
            )

        refresh_btn = gr.Button("🔄 Refresh", size="sm")
        refresh_btn.click(
            fn=_refresh,
            inputs=[],
            outputs=[
                stats_html,
                timeline_html_display,
                entities_html_display,
                relationships_html_display,
                context_html_display,
            ],
        )
