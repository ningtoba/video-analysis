"""Search Tab route handlers — cross-video semantic search."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse


def register_search_routes(app, config, templates):
    router = APIRouter(prefix="/search", tags=["Search"])

    @router.get("", response_class=HTMLResponse)
    @router.get("/", response_class=HTMLResponse)
    async def search_page(request: Request, partial: str = ""):
        return templates.TemplateResponse(
            request=request, name="pages/search.html", context={"config": config}
        )

    @router.get("/status", response_class=HTMLResponse)
    async def search_status(request: Request):
        return HTMLResponse(
            '<span class="text-sm text-muted">Search ready. Query across all indexed videos.</span>'
        )

    app.include_router(router)
