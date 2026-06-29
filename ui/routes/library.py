"""Library Tab route handlers."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse


def register_library_routes(app, config, templates):
    router = APIRouter(prefix="/library", tags=["Library"])

    @router.get("", response_class=HTMLResponse)
    @router.get("/", response_class=HTMLResponse)
    async def library_page(request: Request, partial: str = ""):
        return templates.TemplateResponse(
            request=request, name="pages/library.html", context={"config": config}
        )

    @router.get("/status", response_class=HTMLResponse)
    async def library_status(request: Request):
        return HTMLResponse(
            '<span class="text-sm text-muted">Video library ready.</span>'
        )

    app.include_router(router)
