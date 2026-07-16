from __future__ import annotations

import logging
import re
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, Request, Response, UploadFile

from app.core.config import Settings, get_settings
from app.core.security import require_api_key
from app.models.detection import ModelBundle
from app.models.recognition import model_information
from app.schemas.response import ScanResponse
from app.services.document import scan_document
from app.utils.exceptions import UploadValidationError
from app.utils.image import read_upload_limited, validate_document_payload


logger = logging.getLogger(__name__)
router = APIRouter()


def _runtime(request: Request) -> ModelBundle:
    runtime = getattr(request.app.state, "models", None)
    if not isinstance(runtime, ModelBundle) or runtime.pipeline is None:
        raise HTTPException(status_code=503, detail="Document models are not ready")
    return runtime


def _request_id(request: Request) -> str:
    supplied = request.headers.get("X-Request-ID", "")
    if supplied and len(supplied) <= 64 and re.fullmatch(r"[A-Za-z0-9._-]+", supplied):
        return supplied
    return uuid4().hex


def process_scan(
    request: Request,
    response: Response,
    file: UploadFile,
    settings: Settings,
) -> dict[str, object]:
    request_id = _request_id(request)
    response.headers["X-Request-ID"] = request_id
    runtime = _runtime(request)
    try:
        data = read_upload_limited(file.file, settings.max_upload_bytes)
        detected_format = validate_document_payload(
            data=data,
            filename=file.filename or "upload",
            content_type=file.content_type,
            settings=settings,
        )
        with runtime.scan_lock:
            result = scan_document(
                pipeline=runtime.pipeline,
                fallback_provider=runtime.get_fallback,
                data=data,
                filename=file.filename or f"upload.{detected_format}",
            )
        logger.info(
            "document scan completed",
            extra={
                "request_id": request_id,
                "file_format": detected_format,
                "file_bytes": len(data),
                "source": result["source"],
                "status": result["status"],
                "processing_seconds": result["processing_seconds"],
            },
        )
        return result
    except UploadValidationError as exc:
        logger.warning(
            "document upload rejected",
            extra={"request_id": request_id, "error_code": exc.code},
        )
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": exc.message},
        ) from exc
    except ValueError as exc:
        logger.warning(
            "document could not be decoded",
            extra={"request_id": request_id, "error_code": "INVALID_DOCUMENT"},
        )
        raise HTTPException(status_code=400, detail="Document cannot be processed") from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(
            "document model failed",
            extra={"request_id": request_id, "error_code": type(exc).__name__},
        )
        raise HTTPException(status_code=503, detail="Document model is unavailable") from exc
    finally:
        file.file.close()


@router.post(
    "/mrz/recognize",
    response_model=ScanResponse,
    summary="Recognize MRZ or visual identity fields",
)
def recognize_document(
    request: Request,
    response: Response,
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
    _: None = Depends(require_api_key),
) -> dict[str, object]:
    return process_scan(request, response, file, settings)


@router.get("/health", summary="Service readiness")
def health(request: Request) -> dict[str, object]:
    runtime = getattr(request.app.state, "models", None)
    if not isinstance(runtime, ModelBundle):
        return {
            "status": "starting",
            "mrz_ready": False,
            "fallback_ready": False,
        }
    return runtime.health()


@router.get(
    "/models/info",
    summary="Loaded model information",
    dependencies=[Depends(require_api_key)],
)
def models_info(request: Request) -> dict[str, object]:
    return model_information(_runtime(request))
