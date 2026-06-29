"""Import Tab route handlers."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse


def register_import_routes(app, config, templates):
    router = APIRouter(prefix="/import", tags=["Import"])

    @router.get("", response_class=HTMLResponse)
    @router.get("/", response_class=HTMLResponse)
    async def import_page(request: Request, partial: str = ""):
        return templates.TemplateResponse(
            request=request, name="pages/import.html", context={"config": config}
        )

    @router.get("/status", response_class=HTMLResponse)
    async def import_status(request: Request):
        return HTMLResponse(
            '<span class="text-sm text-muted">Import tab ready.</span>'
        )

    app.include_router(router)
