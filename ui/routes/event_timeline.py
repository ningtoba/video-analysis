"""Event Timeline Tab route handlers."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse


def register_event_routes(app, config, templates):
    router = APIRouter(prefix="/events", tags=["Event Timeline"])

    @router.get("", response_class=HTMLResponse)
    @router.get("/", response_class=HTMLResponse)
    async def events_page(request: Request, partial: str = ""):
        return templates.TemplateResponse(
            request=request, name="pages/event_timeline.html", context={"config": config}
        )

    @router.get("/status", response_class=HTMLResponse)
    async def events_status(request: Request):
        return HTMLResponse(
            '<span class="text-sm text-muted">Event timeline ready.</span>'
        )

    app.include_router(router)
