"""Camera Tab route handlers."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse


def register_camera_routes(app, config, templates):
    router = APIRouter(prefix="/camera", tags=["Camera"])

    @router.get("", response_class=HTMLResponse)
    @router.get("/", response_class=HTMLResponse)
    async def camera_page(request: Request, partial: str = ""):
        return templates.TemplateResponse(
            request=request, name="pages/camera.html", context={"config": config}
        )

    @router.get("/status", response_class=HTMLResponse)
    async def camera_status(request: Request):
        return HTMLResponse(
            '<span class="text-sm text-muted">Camera capture ready.</span>'
        )

    app.include_router(router)
