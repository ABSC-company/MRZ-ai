from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image


CLASS_NAMES = {
    0: "birth_date",
    1: "expiry_date",
    2: "surname",
    3: "given_names",
    4: "middle_name",
    5: "document_number",
    6: "personal_number",
    7: "sex",
    8: "nationality",
}

DATE_CLASSES = {"birth_date", "expiry_date"}
NAME_CLASSES = {"surname", "given_names", "middle_name"}


@dataclass(frozen=True)
class Detection:
    class_id: int
    score: float
    box: tuple[int, int, int, int]


def _normalize_date(value: str) -> str | None:
    clean = " ".join(value.split()).strip(" .,-/")
    patterns = (
        (r"^(\d{4})[.\-/](\d{1,2})[.\-/](\d{1,2})$", (1, 2, 3)),
        (r"^(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{4})$", (3, 2, 1)),
    )
    for pattern, order in patterns:
        match = re.match(pattern, clean)
        if not match:
            continue
        values = [match.group(index) for index in order]
        try:
            return datetime.strptime("-".join(values), "%Y-%m-%d").date().isoformat()
        except ValueError:
            return clean or None

    digits = re.sub(r"\D", "", clean)
    if len(digits) == 8:
        candidates = (digits, digits[4:8] + digits[2:4] + digits[0:2])
        for candidate in candidates:
            try:
                return datetime.strptime(candidate, "%Y%m%d").date().isoformat()
            except ValueError:
                continue
    return clean or None


def normalize_visual_value(class_name: str, value: str) -> str | None:
    clean = " ".join(value.replace("\n", " ").split()).strip()
    if not clean:
        return None
    if class_name in DATE_CLASSES:
        return _normalize_date(clean)
    if class_name in NAME_CLASSES:
        return clean.upper()
    if class_name == "personal_number":
        compact = re.sub(r"[^A-Za-z0-9]", "", clean).upper()
        return compact or None
    if class_name == "document_number":
        compact = re.sub(r"\s+", "", clean).upper()
        return compact or None
    if class_name == "sex":
        upper = clean.upper()
        female_markers = {"F", "Ж", "FEMALE", "ЖЕН", "ЖЕНСКИЙ"}
        male_markers = {"M", "М", "MALE", "МУЖ", "МУЖСКОЙ"}
        if upper in female_markers or upper.startswith("FEM") or upper.startswith("ЖЕН"):
            return "F"
        if upper in male_markers or upper.startswith("MAL") or upper.startswith("МУЖ"):
            return "M"
        return upper
    return clean.upper()


class FallbackFieldExtractor:
    def __init__(
        self,
        config_path: Path,
        weights_path: Path,
        repo_dir: Path,
        device: str,
        threshold: float = 0.40,
        crop_padding: float = 0.08,
        ocr_languages: tuple[str, ...] = ("ru", "en"),
    ) -> None:
        for name, path in {
            "config": config_path,
            "weights": weights_path,
            "RT-DETR repository": repo_dir,
        }.items():
            if not path.exists():
                raise FileNotFoundError(f"Fallback {name} not found: {path}")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Fallback threshold must be between 0 and 1")

        self.config_path = config_path.resolve()
        self.weights_path = weights_path.resolve()
        self.repo_dir = repo_dir.resolve()
        self.threshold = threshold
        self.crop_padding = max(0.0, crop_padding)
        self.device = device

        if str(self.repo_dir) not in sys.path:
            sys.path.insert(0, str(self.repo_dir))

        import torch
        from src.core import YAMLConfig

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for fallback, but PyTorch cannot use CUDA")

        cfg = YAMLConfig(str(self.config_path), resume=str(self.weights_path))
        try:
            checkpoint = torch.load(self.weights_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(self.weights_path, map_location="cpu")

        if "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
            checkpoint_source = "ema.module"
        elif "model" in checkpoint:
            state = checkpoint["model"]
            checkpoint_source = "model"
        else:
            state = checkpoint
            checkpoint_source = "state_dict"

        cfg.model.load_state_dict(state, strict=True)
        self.torch = torch
        self.model = cfg.model.deploy().to(device).eval()
        self.postprocessor = cfg.postprocessor.deploy().to(device).eval()
        configured_size = cfg.yaml_cfg.get("eval_spatial_size", [960, 960])
        if not isinstance(configured_size, (list, tuple)) or len(configured_size) != 2:
            configured_size = [960, 960]
        self.model_size = (int(configured_size[0]), int(configured_size[1]))

        import easyocr

        languages = list(dict.fromkeys(ocr_languages)) or ["en"]
        try:
            self.ocr_reader = easyocr.Reader(languages, gpu=device.startswith("cuda"))
            self.ocr_languages = languages
        except Exception as exc:
            if languages == ["en"]:
                raise
            print(f"Fallback OCR languages {languages} unavailable ({exc}); using ['en']")
            self.ocr_reader = easyocr.Reader(["en"], gpu=device.startswith("cuda"))
            self.ocr_languages = ["en"]

        print(
            "Fallback field model ready: "
            f"device={device} size={self.model_size} threshold={threshold} "
            f"weights={self.weights_path} source={checkpoint_source} "
            f"ocr_languages={','.join(self.ocr_languages)}"
        )

    def _preprocess(self, image: Image.Image):
        from torchvision.transforms import functional

        resized = functional.resize(image, list(self.model_size), antialias=True)
        tensor = functional.to_tensor(resized).unsqueeze(0).to(self.device)
        original_size = self.torch.tensor([[image.width, image.height]], device=self.device)
        return tensor, original_size

    @staticmethod
    def _valid_box(box: tuple[int, int, int, int], size: tuple[int, int]) -> bool:
        width, height = size
        x1, y1, x2, y2 = box
        return 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height

    def _best_per_class(self, labels, boxes, scores, size: tuple[int, int]) -> list[Detection]:
        width, height = size
        best: dict[int, Detection] = {}
        try:
            label_values = labels.detach().cpu().tolist()
            box_values = boxes.detach().cpu().tolist()
            score_values = scores.detach().cpu().tolist()
        except AttributeError:
            label_values = [item.item() for item in labels]
            box_values = [item.tolist() for item in boxes]
            score_values = [item.item() for item in scores]

        for class_value, raw_box, score_value in zip(label_values, box_values, score_values):
            class_id = int(class_value)
            score = float(score_value)
            if class_id not in CLASS_NAMES or score < self.threshold:
                continue
            raw = [float(value) for value in raw_box]
            box = (
                max(0, min(width, int(round(raw[0])))),
                max(0, min(height, int(round(raw[1])))),
                max(0, min(width, int(round(raw[2])))),
                max(0, min(height, int(round(raw[3])))),
            )
            if not self._valid_box(box, size):
                continue
            current = best.get(class_id)
            if current is None or score > current.score:
                best[class_id] = Detection(class_id, score, box)
        return [best[class_id] for class_id in sorted(best)]

    def _padded_box(self, box: tuple[int, int, int, int], size: tuple[int, int]) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = box
        width, height = size
        pad_x = round((x2 - x1) * self.crop_padding)
        pad_y = round((y2 - y1) * self.crop_padding)
        return max(0, x1 - pad_x), max(0, y1 - pad_y), min(width, x2 + pad_x), min(height, y2 + pad_y)

    @staticmethod
    def _ocr_allowlist(class_name: str) -> str | None:
        if class_name in DATE_CLASSES:
            return "0123456789.-/"
        if class_name == "personal_number":
            return "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if class_name == "document_number":
            return "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-"
        if class_name == "sex":
            return "MFМЖmfмж"
        return None

    def _read_crop(self, crop: Image.Image, class_name: str) -> tuple[str | None, float | None]:
        import cv2
        import numpy as np

        if crop.height < 96:
            scale = 96 / max(crop.height, 1)
            crop = crop.resize((max(1, round(crop.width * scale)), 96), Image.Resampling.LANCZOS)
        crop_array = cv2.cvtColor(np.asarray(crop), cv2.COLOR_RGB2BGR)
        results = self.ocr_reader.readtext(
            crop_array,
            detail=1,
            paragraph=False,
            allowlist=self._ocr_allowlist(class_name),
        )
        if not results:
            return None, None

        def position(item: Any) -> tuple[float, float]:
            polygon = item[0]
            return min(point[1] for point in polygon), min(point[0] for point in polygon)

        results = sorted(results, key=position)
        parts = [str(item[1]).strip() for item in results if str(item[1]).strip()]
        if not parts:
            return None, None
        weights = [max(len(str(item[1]).strip()), 1) for item in results if str(item[1]).strip()]
        scores = [float(item[2]) for item in results if str(item[1]).strip()]
        confidence = sum(score * weight for score, weight in zip(scores, weights)) / sum(weights)
        raw_value = " ".join(parts)
        return normalize_visual_value(class_name, raw_value), confidence

    def extract(self, image_bgr: Any) -> dict[str, Any]:
        import cv2

        image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        image_tensor, original_size = self._preprocess(image)
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize()
        started = time.perf_counter()
        with self.torch.inference_mode():
            outputs = self.model(image_tensor)
            labels, boxes, scores = self.postprocessor(outputs, original_size)
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - started

        detections = self._best_per_class(labels[0], boxes[0], scores[0], image.size)
        values: dict[str, str | None] = {name: None for name in CLASS_NAMES.values()}
        field_details: dict[str, dict[str, Any]] = {}
        for detection in detections:
            class_name = CLASS_NAMES[detection.class_id]
            crop_box = self._padded_box(detection.box, image.size)
            value, ocr_score = self._read_crop(image.crop(crop_box), class_name)
            values[class_name] = value
            field_details[class_name] = {
                "value": value,
                "detector_score": round(detection.score, 6),
                "ocr_score": round(ocr_score, 6) if ocr_score is not None else None,
                "bbox_xyxy": list(detection.box),
                "crop_bbox_xyxy": list(crop_box),
            }

        return {
            "values": values,
            "field_details": field_details,
            "detected_classes": len(detections),
            "inference_seconds": round(inference_seconds, 6),
            "threshold": self.threshold,
            "selection": "highest_score_per_class",
        }
