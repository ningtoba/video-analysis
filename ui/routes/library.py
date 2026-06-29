"""Library Tab — renders video cards from RAG backend."""

from fastapi import APIRouter, Request, Query
from fastapi.responses import HTMLResponse


def register_library_routes(app, config, templates):
    router = APIRouter(prefix="/library", tags=["Library"])

    @router.get("", response_class=HTMLResponse)
    @router.get("/", response_class=HTMLResponse)
    async def library_page(request: Request, partial: str = ""):
        return templates.TemplateResponse(
            request=request, name="pages/library.html", context={"config": config}
        )

    @router.get("/list", response_class=HTMLResponse)
    async def library_list(request: Request, q: str = Query(default="")):
        """Return library cards as an HTMX partial."""
        rag = request.app.state.rag
        try:
            video_ids = rag.list_videos()
        except Exception:
            video_ids = []

        if q:
            video_ids = [v for v in video_ids if q.lower() in v.lower()]

        if not video_ids:
            return HTMLResponse(
                '<div class="card text-center" style="padding:var(--space-2xl);">'
                '<p class="text-muted">No videos indexed yet. Upload one on the Analysis tab.</p>'
                "</div>"
            )

        cards = []
        for vid in sorted(video_ids):
            try:
                info = rag.get_library_info(vid)
            except Exception:
                info = None
            filename = info.filename if info and info.filename else vid
            scenes = info.num_scenes if info else "?"
            duration = f"{info.duration:.0f}s" if info and info.duration else "?"
            cards.append(
                '<div class="library-card" style="display:flex;justify-content:space-between;align-items:center;"'
                f' onclick="document.querySelector(\'[x-data]\').__x.$data.switchTab(\'analysis\');'
                f'window.dispatchEvent(new CustomEvent(\'va:videoReady\',{{detail:{{videoId:\'{vid}\',videoPath:\'\'}}}}))">'
                f'<div>'
                f'<div class="lib-title">{_esc(filename)}</div>'
                f'<div class="lib-meta">'
                f'<span class="lib-stat">🆔 {vid[:12]}...</span>'
                f'<span class="lib-stat">🎬 {scenes} scenes</span>'
                f'<span class="lib-stat">⏱ {duration}</span>'
                f"</div></div>"
                f'<button class="btn btn-danger btn-sm" style="z-index:1;"'
                f' onclick="event.stopPropagation();if(confirm(\'Delete {vid[:12]}...?\')){{'
                f'fetch(\'/api/videos/{vid}\',{{method:\'DELETE\'}}).then(()=>location.reload())}}">'
                f"🗑</button></div>"
            )

        count = len(cards)
        header = (
            f'<div class="text-sm text-muted mb-sm">{count} video{"s" if count != 1 else ""} indexed'
            + (' (filtered)' if q else '')
            + "</div>"
        )
        return HTMLResponse(header + "".join(cards))

    app.include_router(router)


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
