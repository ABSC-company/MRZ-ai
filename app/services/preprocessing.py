from __future__ import annotations

from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from app.services.mrz_pipeline import pdf_bytes_to_images, tiff_bytes_to_images


def decode_document_pages(data: bytes, filename: str) -> list[np.ndarray]:
    suffix = Path(filename or "upload.jpg").suffix.lower()
    if suffix == ".pdf":
        return pdf_bytes_to_images(data)
    if suffix in {".tif", ".tiff"}:
        return tiff_bytes_to_images(data)

    try:
        with Image.open(BytesIO(data)) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            return [cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)]
    except Exception:
        array = np.frombuffer(data, dtype=np.uint8)
        image_bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)
        return [image_bgr] if image_bgr is not None else []
