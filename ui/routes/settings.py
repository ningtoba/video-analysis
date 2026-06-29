"""Settings Tab — editable configuration via /api/config."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse


def register_settings_routes(app, config, templates):
    router = APIRouter(prefix="/settings", tags=["Settings"])

    @router.get("", response_class=HTMLResponse)
    @router.get("/", response_class=HTMLResponse)
    async def settings_page(request: Request):
        return templates.TemplateResponse(
            request=request, name="pages/settings.html", context={"config": config}
        )

    app.include_router(router)
