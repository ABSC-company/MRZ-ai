from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int | None) -> int | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    parsed = int(value)
    return parsed if parsed > 0 else None


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None and value.strip() else default


def _env_path(name: str, default: Path | None) -> Path | None:
    value = os.getenv(name)
    return Path(value).expanduser() if value else default


def _env_list(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.getenv(name)
    if not value:
        return default
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    project_root: Path
    environment: str
    enable_docs: bool
    log_level: str
    trusted_hosts: tuple[str, ...]
    api_key_required: bool
    api_key_file: Path | None
    api_key: str | None
    max_upload_bytes: int
    max_image_pixels: int
    max_document_pages: int
    max_request_bytes: int
    mrz_crop_model: Path
    mrz_device: str
    mrz_max_crops: int | None
    mrz_try_upside_down: bool
    mrz_orientation_retry: bool
    mrz_use_easyocr: bool
    mrz_easyocr_aggressive: bool
    fallback_enabled: bool
    fallback_required: bool
    fallback_config: Path
    fallback_weights: Path
    fallback_repo: Path
    fallback_device: str
    fallback_threshold: float
    fallback_crop_padding: float
    fallback_ocr_languages: tuple[str, ...]

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @classmethod
    def from_environment(cls) -> "Settings":
        environment = os.getenv("ENVIRONMENT", "development").strip().lower()
        max_upload_mb = _env_int("MAX_UPLOAD_MB", 20) or 20
        max_upload_bytes = max_upload_mb * 1024 * 1024
        default_key_file = Path("/run/secrets/api_key.txt") if environment == "production" else None
        return cls(
            project_root=PROJECT_ROOT,
            environment=environment,
            enable_docs=_env_flag("ENABLE_DOCS", environment != "production"),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            trusted_hosts=_env_list("TRUSTED_HOSTS", ("*",)),
            api_key_required=_env_flag("API_KEY_REQUIRED", environment == "production"),
            api_key_file=_env_path("API_KEY_FILE", default_key_file),
            api_key=os.getenv("API_KEY"),
            max_upload_bytes=max_upload_bytes,
            max_image_pixels=_env_int("MAX_IMAGE_PIXELS", 60_000_000) or 60_000_000,
            max_document_pages=_env_int("MAX_DOCUMENT_PAGES", 10) or 10,
            max_request_bytes=max_upload_bytes + 1024 * 1024,
            mrz_crop_model=_env_path(
                "MRZ_CROP_MODEL", PROJECT_ROOT / "weights" / "unet_resnet34.pth"
            ),
            mrz_device=os.getenv("MRZ_DEVICE", "auto").strip().lower(),
            mrz_max_crops=_env_int("MRZ_MAX_CROPS", 2),
            mrz_try_upside_down=_env_flag("MRZ_TRY_UPSIDE_DOWN", True),
            mrz_orientation_retry=_env_flag("MRZ_ORIENTATION_RETRY", True),
            mrz_use_easyocr=_env_flag("MRZ_USE_EASYOCR", True),
            mrz_easyocr_aggressive=_env_flag("MRZ_EASYOCR_AGGRESSIVE", True),
            fallback_enabled=_env_flag("FALLBACK_ENABLED", True),
            fallback_required=_env_flag("FALLBACK_REQUIRED", True),
            fallback_config=_env_path(
                "FALLBACK_CONFIG", PROJECT_ROOT / "configs" / "cards_rtdetrv2_r18.yml"
            ),
            fallback_weights=_env_path(
                "FALLBACK_WEIGHTS", PROJECT_ROOT / "weights" / "cards_rtdetrv2_r18_best.pth"
            ),
            fallback_repo=_env_path(
                "FALLBACK_RTDETR_REPO", PROJECT_ROOT / "vendor" / "rtdetrv2_pytorch"
            ),
            fallback_device=os.getenv("FALLBACK_DEVICE", "auto").strip().lower(),
            fallback_threshold=_env_float("FALLBACK_THRESHOLD", 0.40),
            fallback_crop_padding=_env_float("FALLBACK_CROP_PADDING", 0.08),
            fallback_ocr_languages=_env_list("FALLBACK_OCR_LANGS", ("ru", "en")),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_environment()
