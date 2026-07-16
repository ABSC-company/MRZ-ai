from __future__ import annotations

import sys
from types import ModuleType

import cv2
import numpy as np


class TurboJPEG:
    """OpenCV-backed subset used when libturbojpeg is unavailable on Windows."""

    def encode(self, image: np.ndarray, quality: int = 90, **_kwargs) -> bytes:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        ok, buffer = cv2.imencode(".jpg", image, params)
        if not ok:
            raise RuntimeError("OpenCV JPEG encode failed")
        return buffer.tobytes()

    def decode(self, data: bytes, **_kwargs) -> np.ndarray:
        array = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("OpenCV JPEG decode failed")
        return image


def install_turbojpeg_fallback() -> None:
    module = ModuleType("turbojpeg")
    module.TurboJPEG = TurboJPEG
    sys.modules["turbojpeg"] = module
