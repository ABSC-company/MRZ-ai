from __future__ import annotations

from typing import Any

from app.models.detection import ModelBundle


def model_information(bundle: ModelBundle) -> dict[str, Any]:
    return {
        "mrz_detection": {
            "name": "unet_resnet34_document_crop_and_mrzscanner",
            "device": bundle.mrz_device,
            "ready": bundle.pipeline is not None,
        },
        "mrz_recognition": {
            "name": "MRZScanner_two_stage_with_EasyOCR_fallback",
            "ready": bundle.pipeline is not None,
        },
        "alternative_field_detection": {
            "name": "RT-DETRv2_R18",
            "device": bundle.fallback_device,
            "ready": bundle.fallback_extractor is not None,
            "classes": 9,
        },
    }
