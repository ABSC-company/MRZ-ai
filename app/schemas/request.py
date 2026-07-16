from __future__ import annotations

from pydantic import BaseModel


class UploadConstraints(BaseModel):
    max_upload_bytes: int
    max_image_pixels: int
    max_document_pages: int
    supported_extensions: list[str]
