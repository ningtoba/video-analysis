"""Batch Processing Tab route handlers."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse


def register_batch_routes(app, config, templates):
    router = APIRouter(prefix="/batch", tags=["Batch"])

    @router.get("", response_class=HTMLResponse)
    @router.get("/", response_class=HTMLResponse)
    async def batch_page(request: Request, partial: str = ""):
        return templates.TemplateResponse(
            request=request, name="pages/batch.html", context={"config": config}
        )

    @router.get("/status", response_class=HTMLResponse)
    async def batch_status(request: Request):
        return HTMLResponse(
            '<span class="text-sm text-muted">Batch processing queue ready.</span>'
        )

    app.include_router(router)
