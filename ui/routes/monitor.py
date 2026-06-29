"""Monitor Tab route handlers."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse


def register_monitor_routes(app, config, templates):
    router = APIRouter(prefix="/monitor", tags=["Monitor"])

    @router.get("", response_class=HTMLResponse)
    @router.get("/", response_class=HTMLResponse)
    async def monitor_page(request: Request, partial: str = ""):
        return templates.TemplateResponse(
            request=request, name="pages/monitor.html", context={"config": config}
        )

    @router.get("/status", response_class=HTMLResponse)
    async def monitor_status(request: Request):
        return HTMLResponse(
            '<span class="text-sm text-muted">Monitoring dashboard ready.</span>'
        )

    app.include_router(router)
