from __future__ import annotations

from io import BytesIO
from pathlib import Path

import fitz
from PIL import Image, ImageSequence, UnidentifiedImageError
from pillow_heif import register_heif_opener

from app.core.config import Settings
from app.utils.exceptions import UploadValidationError


register_heif_opener()

SUPPORTED_EXTENSIONS = {
    ".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png", ".bmp": "bmp",
    ".tif": "tiff", ".tiff": "tiff", ".webp": "webp", ".pdf": "pdf",
    ".heic": "heif", ".heif": "heif",
}
SUPPORTED_MIME_TYPES = {
    "image/jpeg": "jpeg", "image/png": "png", "image/bmp": "bmp",
    "image/tiff": "tiff", "image/webp": "webp", "application/pdf": "pdf",
    "image/heic": "heif", "image/heif": "heif", "application/octet-stream": None,
}
PIL_FORMATS = {
    "JPEG": "jpeg", "PNG": "png", "BMP": "bmp", "TIFF": "tiff",
    "WEBP": "webp", "HEIF": "heif", "HEIC": "heif",
}


def read_upload_limited(file_object, max_bytes: int, chunk_size: int = 1024 * 1024) -> bytes:
    payload = bytearray()
    while True:
        chunk = file_object.read(chunk_size)
        if not chunk:
            break
        payload.extend(chunk)
        if len(payload) > max_bytes:
            raise UploadValidationError(413, "FILE_TOO_LARGE", "Uploaded file is too large")
    if not payload:
        raise UploadValidationError(400, "EMPTY_FILE", "Uploaded file is empty")
    return bytes(payload)


def _magic_format(data: bytes) -> str | None:
    if data.startswith(b"%PDF-"):
        return "pdf"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith((b"II*\x00", b"MM\x00*")):
        return "tiff"
    if data.startswith(b"BM"):
        return "bmp"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if len(data) >= 12 and data[4:8] == b"ftyp":
        brand = data[8:12].lower()
        if brand in {b"heic", b"heix", b"hevc", b"hevx", b"mif1", b"msf1"}:
            return "heif"
    return None


def _validate_pdf(data: bytes, settings: Settings) -> None:
    try:
        document = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise UploadValidationError(400, "CORRUPT_FILE", "PDF cannot be decoded") from exc
    try:
        if document.page_count < 1:
            raise UploadValidationError(400, "EMPTY_DOCUMENT", "PDF has no pages")
        if document.page_count > settings.max_document_pages:
            raise UploadValidationError(413, "TOO_MANY_PAGES", "Document has too many pages")
        for page in document:
            rendered_pixels = int(page.rect.width * 3) * int(page.rect.height * 3)
            if rendered_pixels > settings.max_image_pixels:
                raise UploadValidationError(
                    413, "IMAGE_TOO_LARGE", "Rendered PDF page exceeds the pixel limit"
                )
    finally:
        document.close()


def _validate_image(data: bytes, settings: Settings) -> str:
    try:
        with Image.open(BytesIO(data)) as image:
            actual_format = PIL_FORMATS.get((image.format or "").upper())
            if actual_format is None:
                raise UploadValidationError(415, "UNSUPPORTED_FORMAT", "Image format is unsupported")
            frame_count = int(getattr(image, "n_frames", 1))
            if frame_count > settings.max_document_pages:
                raise UploadValidationError(413, "TOO_MANY_PAGES", "Document has too many pages")
            for frame in ImageSequence.Iterator(image):
                width, height = frame.size
                if width <= 0 or height <= 0:
                    raise UploadValidationError(400, "CORRUPT_FILE", "Image has invalid dimensions")
                if width * height > settings.max_image_pixels:
                    raise UploadValidationError(413, "IMAGE_TOO_LARGE", "Image exceeds the pixel limit")
                frame.load()
            return actual_format
    except UploadValidationError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise UploadValidationError(400, "CORRUPT_FILE", "Image cannot be decoded") from exc


def validate_document_payload(
    data: bytes,
    filename: str,
    content_type: str | None,
    settings: Settings,
) -> str:
    extension = Path(filename or "").suffix.lower()
    expected_from_extension = SUPPORTED_EXTENSIONS.get(extension)
    if expected_from_extension is None:
        raise UploadValidationError(415, "UNSUPPORTED_EXTENSION", "File extension is unsupported")

    normalized_mime = (content_type or "").split(";", 1)[0].strip().lower()
    if normalized_mime and normalized_mime not in SUPPORTED_MIME_TYPES:
        raise UploadValidationError(415, "UNSUPPORTED_MEDIA_TYPE", "Content-Type is unsupported")

    detected = _magic_format(data)
    if detected is None:
        raise UploadValidationError(415, "UNKNOWN_FILE_SIGNATURE", "File signature is unsupported")
    if detected != expected_from_extension:
        raise UploadValidationError(415, "FORMAT_MISMATCH", "File content does not match its extension")
    mime_format = SUPPORTED_MIME_TYPES.get(normalized_mime)
    if mime_format is not None and mime_format != detected:
        raise UploadValidationError(415, "MIME_MISMATCH", "File content does not match Content-Type")

    if detected == "pdf":
        _validate_pdf(data, settings)
    else:
        actual_format = _validate_image(data, settings)
        if actual_format != detected:
            raise UploadValidationError(415, "FORMAT_MISMATCH", "Decoded image format does not match")
    return detected
