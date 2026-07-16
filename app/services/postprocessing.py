from __future__ import annotations

from typing import Any

from app.services.validation import mrz_fields_for_api


def public_identity_fields(parsed_mrz: dict[str, Any]) -> dict[str, Any]:
    return mrz_fields_for_api(parsed_mrz)
