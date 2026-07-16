from __future__ import annotations

import secrets

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.core.config import Settings, get_settings


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _expected_api_key(settings: Settings) -> str | None:
    if settings.api_key_file and settings.api_key_file.is_file():
        value = settings.api_key_file.read_text(encoding="utf-8").strip()
        print(value)
        return value or None
    return settings.api_key.strip() if settings.api_key else None


def validate_security_configuration(settings: Settings) -> None:
    if settings.api_key_required and not _expected_api_key(settings):
        raise RuntimeError(
            "API key protection is enabled, but API_KEY_FILE and API_KEY are empty"
        )


def require_api_key(
    provided: str | None = Security(api_key_header),
    settings: Settings = Depends(get_settings),
) -> None:
    if not settings.api_key_required:
        return
    expected = _expected_api_key(settings)
    if not provided or not expected or not secrets.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
