from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from app.services.fallback import FallbackFieldExtractor
from app.services.mrz_pipeline import MRZPipeline, MRZResult, final_mrz_score
from app.services.postprocessing import public_identity_fields
from app.services.preprocessing import decode_document_pages
from app.services.validation import parse_mrz


FallbackProvider = Callable[[], FallbackFieldExtractor]


def _best_primary_result(pipeline: MRZPipeline, pages: list[np.ndarray]) -> MRZResult:
    if not pages:
        raise ValueError("Unable to decode document")
    results = [pipeline.process_ndarray(page) for page in pages]
    return max(results, key=final_mrz_score)


def _run_primary(
    pipeline: MRZPipeline,
    data: bytes,
    filename: str,
    pages: list[np.ndarray] | None = None,
) -> tuple[MRZResult, list[np.ndarray] | None]:
    try:
        result = pipeline.process_file_bytes(data=data, filename=filename)
    except ValueError:
        result = None
    if result is not None:
        return result, pages
    pages = pages or decode_document_pages(data, filename)
    return _best_primary_result(pipeline, pages), pages


def _fallback_quality(result: dict[str, Any]) -> tuple[int, float, float]:
    details = result.get("field_details", {})
    values_found = sum(item.get("value") is not None for item in details.values())
    detector_sum = sum(float(item.get("detector_score") or 0.0) for item in details.values())
    ocr_sum = sum(float(item.get("ocr_score") or 0.0) for item in details.values())
    return values_found, detector_sum, ocr_sum


def _run_fallback(
    extractor: FallbackFieldExtractor,
    pages: list[np.ndarray],
) -> dict[str, Any]:
    if not pages:
        raise ValueError("Unable to decode document for fallback")
    candidates: list[dict[str, Any]] = []
    for page_index, page in enumerate(pages):
        result = extractor.extract(page)
        result["page_index"] = page_index
        candidates.append(result)
    return max(candidates, key=_fallback_quality)


def scan_document(
    pipeline: MRZPipeline,
    fallback_provider: FallbackProvider,
    data: bytes,
    filename: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    pages: list[np.ndarray] | None = None
    primary, pages = _run_primary(pipeline, data, filename, pages)
    parsed = parse_mrz(primary.texts)

    parsed["scanner_validation"] = {
        "format": primary.validation.format,
        "valid_digits": primary.validation.valid_digits,
        "total_digits": primary.validation.total_digits,
        "structural_ok": primary.validation.structural_ok,
    }
    parsed["scanner_message"] = primary.msg
    parsed["scanner_score"] = round(float(primary.score), 6)

    if parsed["mrz_valid"]:
        fields = public_identity_fields(parsed)
        document = parsed.get("parsed_fields", {})
        elapsed = round(time.perf_counter() - started, 6)
        return {
            "status": "ok",
            "success": True,
            "source": "mrz",
            "fallback_used": False,
            "fallback_reason": None,
            "fields": fields,
            "document": {
                "document_type": document.get("document_type"),
                "issuing_state_code": document.get("issuing_state_code"),
                "issue_date": document.get("issue_date"),
                "issuing_authority_code": document.get("issuing_authority_code"),
            },
            "mrz": parsed,
            "fallback": None,
            "processing_seconds": elapsed,
            "processing_time_ms": round(elapsed * 1000, 3),
        }

    fallback_reason = "mrz_invalid" if parsed["mrz_detected"] else "mrz_not_detected"
    pages = pages or decode_document_pages(data, filename)
    fallback = _run_fallback(fallback_provider(), pages)
    fields = fallback["values"]
    has_values = any(value is not None for value in fields.values())
    status = "ok" if has_values else "fields_not_found"
    elapsed = round(time.perf_counter() - started, 6)

    return {
        "status": status,
        "success": has_values,
        "source": "fallback",
        "fallback_used": True,
        "fallback_reason": fallback_reason,
        "fields": fields,
        "document": {
            "document_type": None,
            "issuing_state_code": None,
            "issue_date": None,
            "issuing_authority_code": None,
        },
        "mrz": parsed,
        "fallback": fallback,
        "processing_seconds": elapsed,
        "processing_time_ms": round(elapsed * 1000, 3),
    }
