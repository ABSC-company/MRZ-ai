from __future__ import annotations

from contextlib import asynccontextmanager
import logging

from fastapi import Depends, FastAPI, File, Request, Response, UploadFile
from fastapi.responses import JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.api.v1.endpoints.mrz import health, process_scan
from app.api.v1.router import api_router
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.core.security import require_api_key, validate_security_configuration
from app.models.detection import ModelBundle


settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI):
    validate_security_configuration(settings)
    runtime = ModelBundle(settings)
    runtime.load()
    application.state.models = runtime
    logger.info(
        "document service ready",
        extra={"environment": settings.environment, "api_key_required": settings.api_key_required},
    )
    try:
        yield
    finally:
        runtime.close()
        application.state.models = None


def create_app() -> FastAPI:
    application = FastAPI(
        title="MRZ and identity field scanner",
        version="4.0",
        docs_url="/docs" if settings.enable_docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.enable_docs else None,
        lifespan=lifespan,
    )
    if settings.trusted_hosts != ("*",):
        application.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=list(settings.trusted_hosts),
        )

    @application.middleware("http")
    async def reject_oversized_request(request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                too_large = int(content_length) > settings.max_request_bytes
            except ValueError:
                too_large = True
            if too_large:
                return JSONResponse(
                    status_code=413,
                    content={"detail": {"code": "REQUEST_TOO_LARGE", "message": "Request is too large"}},
                )
        return await call_next(request)

    application.include_router(api_router, prefix="/api/v1")

    @application.post("/scan", include_in_schema=False)
    def legacy_scan(
        request: Request,
        response: Response,
        file: UploadFile = File(...),
        current_settings: Settings = Depends(get_settings),
        _: None = Depends(require_api_key),
    ) -> dict[str, object]:
        return process_scan(request, response, file, current_settings)

    @application.get("/health", include_in_schema=False)
    def legacy_health(request: Request) -> dict[str, object]:
        return health(request)

    return application


app = create_app()
