"""Eval Comparison Tab route handlers."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse


def register_comparison_routes(app, config, templates):
    router = APIRouter(prefix="/comparison", tags=["Eval Comparison"])

    @router.get("", response_class=HTMLResponse)
    @router.get("/", response_class=HTMLResponse)
    async def comparison_page(request: Request, partial: str = ""):
        return templates.TemplateResponse(
            request=request, name="pages/comparison.html", context={"config": config}
        )

    @router.get("/status", response_class=HTMLResponse)
    async def comparison_status(request: Request):
        return HTMLResponse(
            '<span class="text-sm text-muted">Evaluation comparison ready.</span>'
        )

    app.include_router(router)
