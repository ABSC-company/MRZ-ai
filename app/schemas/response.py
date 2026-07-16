from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class IdentityFields(BaseModel):
    birth_date: str | None = None
    expiry_date: str | None = None
    surname: str | None = None
    given_names: str | None = None
    middle_name: str | None = None
    document_number: str | None = None
    personal_number: str | None = None
    sex: str | None = None
    nationality: str | None = None


class DocumentMetadata(BaseModel):
    document_type: str | None = None
    issuing_state_code: str | None = None
    issue_date: str | None = None
    issuing_authority_code: str | None = None


class ScanResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: str
    success: bool
    source: Literal["mrz", "fallback"]
    fallback_used: bool
    fallback_reason: str | None
    fields: IdentityFields
    document: DocumentMetadata
    mrz: dict[str, Any]
    fallback: dict[str, Any] | None
    processing_seconds: float
    processing_time_ms: float
