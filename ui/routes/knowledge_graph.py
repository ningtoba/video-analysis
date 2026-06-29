"""Knowledge Graph Tab route handlers."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse


def register_kg_routes(app, config, templates):
    router = APIRouter(prefix="/knowledge-graph", tags=["Knowledge Graph"])

    @router.get("", response_class=HTMLResponse)
    @router.get("/", response_class=HTMLResponse)
    async def kg_page(request: Request, partial: str = ""):
        return templates.TemplateResponse(
            request=request, name="pages/knowledge_graph.html", context={"config": config}
        )

    @router.get("/status", response_class=HTMLResponse)
    async def kg_status(request: Request):
        return HTMLResponse(
            '<span class="text-sm text-muted">Knowledge graph explorer ready.</span>'
        )

    app.include_router(router)
