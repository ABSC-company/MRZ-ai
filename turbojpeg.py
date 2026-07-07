from __future__ import annotations

import cv2
import numpy as np


class TurboJPEG:
    """Small OpenCV-backed fallback for PyTurboJPEG.

    capybara imports TurboJPEG at module import time. The real PyTurboJPEG
    wrapper needs a system libturbojpeg DLL, which is not always present on
    Windows. MRZScanner only needs capybara to import and to run ONNX helpers
    here, so OpenCV encoding/decoding is enough.
    """

    def encode(self, img: np.ndarray, quality: int = 90, **_kwargs) -> bytes:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        ok, buf = cv2.imencode(".jpg", img, params)
        if not ok:
            raise RuntimeError("OpenCV JPEG encode failed")
        return buf.tobytes()

    def decode(self, data: bytes, **_kwargs) -> np.ndarray:
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("OpenCV JPEG decode failed")
        return img
