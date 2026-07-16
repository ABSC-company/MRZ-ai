from __future__ import annotations

from dataclasses import dataclass, field
import logging
from threading import Lock
from typing import Any

import torch

from app.core.config import Settings
from app.services.fallback import FallbackFieldExtractor
from app.services.mrz_pipeline import MRZPipeline


logger = logging.getLogger(__name__)


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot use CUDA")
    return requested


@dataclass
class ModelBundle:
    settings: Settings
    pipeline: MRZPipeline | None = None
    fallback_extractor: FallbackFieldExtractor | None = None
    fallback_error: str | None = None
    scan_lock: Lock = field(default_factory=Lock)
    fallback_init_lock: Lock = field(default_factory=Lock)
    mrz_device: str = "cpu"
    fallback_device: str = "cpu"

    def load(self) -> None:
        self.mrz_device = _resolve_device(self.settings.mrz_device)
        self.fallback_device = _resolve_device(self.settings.fallback_device)
        if not self.settings.mrz_crop_model.is_file():
            raise FileNotFoundError(f"MRZ crop model not found: {self.settings.mrz_crop_model}")

        logger.info(
            "loading MRZ pipeline",
            extra={"device": self.mrz_device, "max_crops": self.settings.mrz_max_crops},
        )
        self.pipeline = MRZPipeline(
            crop_model=self.settings.mrz_crop_model,
            device=torch.device(self.mrz_device),
            debug=False,
            max_crop_candidates=self.settings.mrz_max_crops,
            try_upside_down=self.settings.mrz_try_upside_down,
            orientation_retry=self.settings.mrz_orientation_retry,
            use_easyocr=self.settings.mrz_use_easyocr,
            easyocr_aggressive=self.settings.mrz_easyocr_aggressive,
        )

        if self.settings.fallback_enabled:
            try:
                self.get_fallback()
            except Exception as exc:
                self.fallback_error = type(exc).__name__
                if self.settings.fallback_required:
                    raise RuntimeError("Fallback model initialization failed") from exc
                logger.exception("fallback model disabled after initialization failure")

    def get_fallback(self) -> FallbackFieldExtractor:
        if not self.settings.fallback_enabled:
            raise RuntimeError("Fallback field extraction is disabled")
        if self.fallback_extractor is not None:
            return self.fallback_extractor

        with self.fallback_init_lock:
            if self.fallback_extractor is not None:
                return self.fallback_extractor
            self.fallback_extractor = FallbackFieldExtractor(
                config_path=self.settings.fallback_config,
                weights_path=self.settings.fallback_weights,
                repo_dir=self.settings.fallback_repo,
                device=self.fallback_device,
                threshold=self.settings.fallback_threshold,
                crop_padding=self.settings.fallback_crop_padding,
                ocr_languages=self.settings.fallback_ocr_languages,
            )
            self.fallback_error = None
            return self.fallback_extractor

    @property
    def ready(self) -> bool:
        fallback_ready = self.fallback_extractor is not None
        fallback_ok = not self.settings.fallback_required or fallback_ready
        return self.pipeline is not None and fallback_ok

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok" if self.ready else "starting",
            "mrz_ready": self.pipeline is not None,
            "fallback_enabled": self.settings.fallback_enabled,
            "fallback_ready": self.fallback_extractor is not None,
            "fallback_required": self.settings.fallback_required,
            "fallback_error": self.fallback_error,
        }

    def close(self) -> None:
        self.pipeline = None
        self.fallback_extractor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
