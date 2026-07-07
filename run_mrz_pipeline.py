#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from albumentations.pytorch import ToTensorV2
import fitz
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mrzscanner import ErrorCodes, MRZScanner, ModelType  # noqa: E402
import capybara as cb  # noqa: E402
import time

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".pdf", ".heic", ".heif",}

# ---------------------------------------------------------------------------
# Константы настройки кропа
# ---------------------------------------------------------------------------
COVERAGE_SKIP_THRESHOLD = 0.72
ANGLE_SKIP_THRESHOLD_DEG = 5.0
MIN_DOCUMENT_COVERAGE = 0.03
MIN_DOCUMENT_SIDE_RATIO = 0.12
MAX_DOCUMENT_ASPECT_RATIO = 3.5
WARP_BOX_EXPAND_RATIO = 1.08
MIN_WARP_SIDE_PX = 32

# ---------------------------------------------------------------------------
# Таблица весов для контрольных цифр MRZ (ИКАО 9303)
# ---------------------------------------------------------------------------
_MRZ_WEIGHTS = [7, 3, 1]
_MRZ_CHARS   = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<"
_MRZ_VALUES  = {ch: i for i, ch in enumerate(_MRZ_CHARS)}


def _check_digit(s: str) -> int:
    """Вычисляет контрольную цифру MRZ по алгоритму ИКАО 9303."""
    total = 0
    for i, ch in enumerate(s.upper()):
        total += _MRZ_VALUES.get(ch, 0) * _MRZ_WEIGHTS[i % 3]
    return total % 10


def _digit_ok(s: str, digit_char: str) -> bool:
    try:
        return _check_digit(s) == int(digit_char)
    except (ValueError, IndexError):
        return False


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MRZValidation:
    format: str           # "TD1", "TD2", "TD3", "unknown"
    valid_digits: int     # количество прошедших проверку контрольных цифр
    total_digits: int     # всего контрольных цифр в формате
    structural_ok: bool   # длины строк соответствуют формату
    score_bonus: float    # бонус к mrz_score


@dataclass
class CropResult:
    image: np.ndarray
    polygon: np.ndarray
    mask_confidence: tuple[float, float]
    ok: bool
    message: str


@dataclass
class MRZResult:
    texts: list[str]
    polygon: np.ndarray
    mrz_image: np.ndarray | None
    line_images: list[np.ndarray]
    rotation: int
    msg: str
    score: float
    validation: MRZValidation = field(
        default_factory=lambda: MRZValidation("unknown", 0, 0, False, 0.0)
    )


# ---------------------------------------------------------------------------
# MRZ валидация — проверка контрольных цифр и формата (ИКАО 9303)
# ---------------------------------------------------------------------------
MRZ_DOC_TYPES = set("PACDIRSVX")


def looks_like_icao_header(line: str) -> bool:

    if not line or len(line) < 5:
        return False

    if line[0] not in MRZ_DOC_TYPES:
        return False

    country = line[2:5]

    good = sum(
        ch.isalpha() or ch.isdigit()
        for ch in country
    )

    return good >= 2
def validate_mrz(texts: list[str]) -> MRZValidation:
    """
    Проверяет строки MRZ на соответствие форматам TD1/TD2/TD3.

    TD3 — паспорт:   2 строки по 44 символа
    TD1 — ID-карта:  3 строки по 30 символов
    TD2 — прочие:    2 строки по 36 символов

    Контрольные цифры по ИКАО 9303 Part 3:
      TD3: cd_doc_num(9), cd_dob(6), cd_expiry(6), cd_composite
      TD1: cd_doc_num(9), cd_dob(6), cd_expiry(6), cd_composite
      TD2: cd_doc_num(9), cd_dob(6), cd_expiry(6), cd_composite
    """
    lines = [t.strip() for t in texts if t and t.strip()]
    lengths = [len(x) for x in lines]

    is_td3 = (
            len(lines) == 2
            and min(lengths) >= 40
            and looks_like_icao_header(lines[0])
    )

    is_td2 = (
            len(lines) == 2
            and 30 <= min(lengths) < 40
            and looks_like_icao_header(lines[0])
    )

    is_td1 = (
            len(lines) == 3
            and looks_like_icao_header(lines[0])
    )

    # --- TD3 (паспорт, 2×44) ---
    if is_td3:
        l1 = lines[0].ljust(44, "<")[:44]
        l2 = lines[1].ljust(44, "<")[:44]
        # Составная строка для итоговой цифры: l2[0:10] + l2[13:20] + l2[21:43]
        composite = l2[0:10] + l2[13:20] + l2[21:43]
        checks = [
            _digit_ok(l2[0:9],   l2[9]),    # номер документа
            _digit_ok(l2[13:19], l2[19]),   # дата рождения
            _digit_ok(l2[21:27], l2[27]),   # дата истечения
            _digit_ok(composite,  l2[43]),   # итоговая
        ]
        valid = sum(checks)
        structural = (
            len(l1) == 44 and len(l2) == 44
            and l1[0] in "PAVCIB"
            and looks_like_icao_header(l1)
        )
        bonus = valid * 15.0 + (30.0 if structural else 0.0)
        return MRZValidation("TD3", valid, 4, structural, bonus)

    # --- TD1 (ID-карта, 3×30) ---
    if is_td1:
        l1 = lines[0].ljust(30, "<")[:30]
        l2 = lines[1].ljust(30, "<")[:30]
        l3 = lines[2].ljust(30, "<")[:30]
        # Составная строка по ИКАО 9303 Part 5:
        # l1[5:30] + l2[0:7] + l2[8:15] + l2[18:29]
        composite = l1[5:30] + l2[0:7] + l2[8:15] + l2[18:29]
        checks = [
            _digit_ok(l1[5:14], l1[14]),    # номер документа (9 символов)
            _digit_ok(l2[0:6],  l2[6]),     # дата рождения
            _digit_ok(l2[8:14], l2[14]),    # дата истечения
            _digit_ok(composite, l2[29]),    # итоговая
        ]
        valid = sum(checks)
        structural = (
            l1[0] in "IAVC"
            and looks_like_icao_header(l1)
        )
        bonus = valid * 15.0 + (30.0 if structural else 0.0)
        return MRZValidation("TD1", valid, 4, structural, bonus)

    # --- TD2 (2×36) ---
    if is_td2:
        l1 = lines[0].ljust(36, "<")[:36]
        l2 = lines[1].ljust(36, "<")[:36]
        composite = l2[0:10] + l2[13:20] + l2[21:35]
        checks = [
            _digit_ok(l2[0:9],   l2[9]),
            _digit_ok(l2[13:19], l2[19]),
            _digit_ok(l2[21:27], l2[27]),
            _digit_ok(composite,  l2[35]),
        ]
        valid = sum(checks)
        structural = (
            len(l1) == 36 and len(l2) == 36
            and l1[0] in "PAVCIB"
            and looks_like_icao_header(l1)
        )
        bonus = valid * 15.0 + (30.0 if structural else 0.0)
        return MRZValidation("TD2", valid, 4, structural, bonus)
    if len(lines) in (2, 3):

        joined = "".join(lines)

        filler_ratio = joined.count("<") / max(len(joined), 1)

        letters = sum(
            c.isalpha()
            for c in joined
        )

        if filler_ratio > 0.35 and letters >= 15:
            return MRZValidation(
                "NON_ICAO",
                0,
                0,
                True,
                20
            )

    unknown_bonus = 0.0

    if len(lines) in (2, 3):
        joined = "".join(lines)

        mrz_chars = sum(
            c.isalnum() or c == "<"
            for c in joined
        )

        ratio = mrz_chars / max(len(joined), 1)

        unknown_bonus = ratio * 10.0

    return MRZValidation("UNKNOWN", 0, 0, False, unknown_bonus)


def mrz_is_fully_valid(v: MRZValidation) -> bool:
    return v.structural_ok and v.total_digits > 0 and v.valid_digits == v.total_digits


# ---------------------------------------------------------------------------
# Вспомогательные функции изображений
# ---------------------------------------------------------------------------
from PIL import Image
from pillow_heif import register_heif_opener
from io import BytesIO
from PIL import Image, ImageSequence
register_heif_opener()

def imread_unicode(path: Path) -> np.ndarray | None:

    ext = path.suffix.lower()

    if ext in {".heic", ".heif"}:
        try:
            pil_img = Image.open(path)
            rgb = np.array(pil_img.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception:
            return None

    data = np.fromfile(str(path), dtype=np.uint8)

    if data.size == 0:
        return None

    return cv2.imdecode(
        data,
        cv2.IMREAD_COLOR
    )
def tiff_to_images(tiff_path: Path) -> list[np.ndarray]:

    pages = []

    try:
        img = Image.open(tiff_path)

        while True:
            rgb = np.array(img.convert("RGB"))

            pages.append(
                cv2.cvtColor(
                    rgb,
                    cv2.COLOR_RGB2BGR
                )
            )

            img.seek(img.tell() + 1)

    except EOFError:
        pass

    return pages

def tiff_bytes_to_images(
        data: bytes
) -> list[np.ndarray]:

    pages = []

    with Image.open(BytesIO(data)) as img:

        for frame in ImageSequence.Iterator(img):

            frame = frame.convert("RGB")

            arr = np.array(frame)

            arr = cv2.cvtColor(
                arr,
                cv2.COLOR_RGB2BGR
            )

            pages.append(arr)

    return pages
def pdf_to_images(pdf_path: Path) -> list[np.ndarray]:
    doc = fitz.open(str(pdf_path))
    pages = []
    for page in doc:
        pix = page.get_pixmap(
            matrix=fitz.Matrix(3, 3),  # 300+ dpi
            alpha=False
        )
        img = np.frombuffer(
            pix.samples,
            dtype=np.uint8
        ).reshape(
            pix.height,
            pix.width,
            3
        )
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        pages.append(img)
    return pages
def pdf_bytes_to_images(data: bytes) -> list[np.ndarray]:

    doc = fitz.open(
        stream=data,
        filetype="pdf"
    )

    pages = []

    for page in doc:

        pix = page.get_pixmap(
            matrix=fitz.Matrix(3, 3),
            alpha=False
        )

        img = np.frombuffer(
            pix.samples,
            dtype=np.uint8
        ).reshape(
            pix.height,
            pix.width,
            3
        )

        img = cv2.cvtColor(
            img,
            cv2.COLOR_RGB2BGR
        )

        pages.append(img)

    return pages

def extract_mrz_from_pdf_text(pdf_path: Path) -> list[str]:
    """
    Пытается извлечь строки MRZ напрямую из текстового слоя PDF.
    """
    try:
        doc = fitz.open(str(pdf_path))
        for page in doc:
            text = page.get_text()
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            candidate_lines = []
            for line in lines:
                clean_line = line.replace(" ", "")
                # MRZ состоит только из A-Z, 0-9 и заполнителя '<'.
                # Длина строк обычно от 30 до 44 символов.
                if len(clean_line) >= 28 and all(
                        c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<" for c in clean_line.upper()):
                    candidate_lines.append(clean_line.upper())

            # Если нашли 2 или 3 строки MRZ
            if len(candidate_lines) in (2, 3):
                return candidate_lines
    except Exception:
        pass
    return []


def imwrite_unicode(path: Path, image: np.ndarray) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".jpg"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def order_points(points):
    x_sorted = points[np.argsort(points[:, 0])]
    left = x_sorted[:2]
    right = x_sorted[2:]
    left = left[np.argsort(left[:, 1])]
    tl, bl = left
    right = right[np.argsort(right[:, 1])]
    tr, br = right
    return np.array(
        [tl, tr, br, bl],
        dtype=np.float32
    )


def four_point_transform(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    rect = order_points(points.astype(np.float32))
    tl, tr, br, bl = rect
    max_width  = max(1, int(round(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))))
    max_height = max(1, int(round(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))))
    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype=np.float32,
    )
    return cv2.warpPerspective(image, cv2.getPerspectiveTransform(rect, dst), (max_width, max_height))


def rotate_image(image: np.ndarray, degrees: int) -> np.ndarray:
    if degrees == 0:   return image
    if degrees == 90:  return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180: return cv2.rotate(image, cv2.ROTATE_180)
    if degrees == 270: return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"Unsupported rotation: {degrees}")


def _rect_angle_is_straight(rect: tuple) -> bool:
    angle      = rect[2]
    angle_norm = abs(angle) % 90.0
    return min(angle_norm, 90.0 - angle_norm) < ANGLE_SKIP_THRESHOLD_DEG


# ---------------------------------------------------------------------------
# Скоринг MRZ
# ---------------------------------------------------------------------------

def valid_mrz_chars_ratio(texts: Iterable[str]) -> float:
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")
    chars   = "".join(texts)
    if not chars:
        return 0.0
    return sum(ch in allowed for ch in chars) / len(chars)


def mrz_score(texts: list[str], msg: object, validation: MRZValidation | None = None) -> float:
    """
    Скоринг результата MRZ.
    - NO_ERROR от сканера даёт +5 (не доминирует над валидацией цифр)
    - каждая прошедшая контрольная цифра даёт +15
    - структурная корректность формата +30
    """
    line_count   = len([t for t in texts if t])
    lengths      = [len(t) for t in texts if t]
    joined       = "".join(texts)
    filler_ratio = joined.count("<") / max(len(joined), 1)

    score = (
        valid_mrz_chars_ratio(texts) * 10.0
        + sum(lengths) / 100.0
        + filler_ratio * 10.0
    )
    if msg == ErrorCodes.NO_ERROR or str(msg).endswith("NO_ERROR"):
        score += 5.0
    if line_count in (2, 3):
        score += 5.0
    if validation is not None:
        score += validation.score_bonus
    return score


def final_mrz_score(result):

    v = result.validation
    score = result.score
    if v.format != "UNKNOWN":
        score += v.valid_digits * 500
        if v.format in ("TD1", "TD2", "TD3"):
            score += v.valid_digits * 500

            if v.structural_ok and v.valid_digits > 0:
                score += 300

            if mrz_is_fully_valid(v):
                score += 5000
        elif v.format == "NON_ICAO":
            score += 200
    return score


# ---------------------------------------------------------------------------
# Перебор ориентаций MRZ-зоны — выбираем лучшую по score (ИСПРАВЛЕНИЕ)
# ---------------------------------------------------------------------------

def _orientation_score(texts: list[str]) -> float:
    """
    Скоринг ориентации MRZ — только по надёжным визуальным признакам.

    Намеренно НЕ использует контрольные цифры (могут случайно совпасть в мусоре).

    Главный критерий — доля символа '<':
      - Правильный MRZ содержит 40–90% символов '<'
      - Мусор (перевёрнутое/нечитаемое) содержит почти 0% символов '<'
    Это самый надёжный разделяющий признак.

    Дополнительные признаки (меньший вес):
      - Доля валидных MRZ-символов (A-Z, 0-9, <)
      - Строки правильной длины (30/36/44)
      - Последний символ строки == '<' (MRZ всегда заканчиваются на заполнители)
      - Первый символ строки — буква из строгого набора типов документа ИКАО
        (P=паспорт, I=ID, A/C/V=другие; намеренно НЕ включает D,S,X и пр.)
    """
    if not texts:
        return 0.0

    lines = [t for t in texts if t]
    if not lines:
        return 0.0

    joined = "".join(lines)
    if not joined:
        return 0.0

    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")

    # 1. Доля символа '<' — ГЛАВНЫЙ признак, максимальный вес
    #    Правильный MRZ: 0.4–0.9; мусор: ~0.0
    filler_ratio = joined.count("<") / len(joined)
    score = filler_ratio * 40.0

    # 2. Доля валидных MRZ-символов (второстепенный признак)
    valid_ratio = sum(ch in allowed for ch in joined) / len(joined)
    score += valid_ratio * 5.0

    # 3. Строки правильной длины
    valid_lengths = {30, 36, 44}
    for line in lines:
        if len(line) in valid_lengths:
            score += 3.0

    # 4. Последний символ строки 1 == '<'
    if lines[0] and lines[0][-1] == "<":
        score += 5.0

    # 5. Первый символ строки 1 — строгий набор типов документа ИКАО 9303:
    #    P=паспорт, I=ID-карта, A/C/V=другие официальные документы
    #    Намеренно НЕ включает D, S, X — это мусорные символы OCR.
    first_char = lines[0][0] if lines[0] else ""
    if first_char in "PIACV":
        score += 8.0

    return score

def _best_orientation_mrz(mrz_image: np.ndarray, scanner_func) -> tuple[np.ndarray, list[str]]:
    """
    Перебирает 0° и 180° ориентации MRZ-зоны, выбирает лучшую.
    Возвращает кортеж: (лучшее изображение MRZ, распознанный текст для этой ориентации).
    """
    if mrz_image is None:
        return mrz_image, []

    # Шаг 1: нормализовать к горизонтальному виду
    h, w = mrz_image.shape[:2]
    if h > w:
        mrz_image = cv2.rotate(mrz_image, cv2.ROTATE_90_CLOCKWISE)

    # Шаг 2: проверить 0° и 180°
    best_img   = mrz_image
    best_texts = []
    best_score = -float("inf")

    for angle, rotated in [
        (0,   mrz_image),
        (180, cv2.rotate(mrz_image, cv2.ROTATE_180)),
    ]:
        try:
            res   = scanner_func(rotated, do_center_crop=False, do_postprocess=True)
            texts = [str(t) for t in res.get("mrz_texts", []) if str(t)]
        except Exception:
            texts = []

        sc = _orientation_score(texts)
        # debug only

        if sc > best_score:
            best_score = sc
            best_img   = rotated
            best_texts = texts

    return best_img, best_texts


# ---------------------------------------------------------------------------
# DocumentCropper
# ---------------------------------------------------------------------------

class DocumentCropper:
    def __init__(self, model_path: Path, device: torch.device) -> None:
        self.device = device
        self.model  = smp.Unet(encoder_name="resnet34", encoder_weights=None, classes=1)
        checkpoint  = torch.load(model_path, map_location=device, weights_only=False)
        state = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        clean_state = {
            (k.replace("model.", "", 1) if k.startswith("model.") else k): v
            for k, v in state.items()
        }
        self.model.load_state_dict(clean_state, strict=False)
        self.model.to(device).eval()
        self.transform = A.Compose([
            A.LongestMaxSize(max_size=1024),
            A.PadIfNeeded(min_height=1024, min_width=1024, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    @staticmethod
    def _full_frame_polygon(h: int, w: int) -> np.ndarray:
        return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)

    @staticmethod
    def _bbox_crop(image: np.ndarray, points: np.ndarray) -> tuple[np.ndarray | None, np.ndarray]:
        h, w = image.shape[:2]
        x1 = max(0, int(np.floor(points[:, 0].min())))
        y1 = max(0, int(np.floor(points[:, 1].min())))
        x2 = min(w, int(np.ceil(points[:, 0].max())))
        y2 = int(np.ceil(points[:, 1].max()))

        extra_bottom = int(h * 0.08)

        y2 = min(h, y2 + extra_bottom)
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None, np.empty((0, 2), dtype=np.float32)
        poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        return image[y1:y2, x1:x2].copy(), poly

    def _predict_mask(self, image_bgr: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        sample    = self.transform(image=image_rgb)
        x         = sample["image"].unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.model(x)
        pred_mask = torch.sigmoid(pred)[0, 0].detach().cpu().numpy()
        conf      = (float(pred_mask.min()), float(pred_mask.max()))
        mask      = (pred_mask > 0.5).astype(np.uint8) * 255

        orig_h, orig_w = image_bgr.shape[:2]
        if orig_w > orig_h:
            scaled_w, scaled_h = 1024, int(round(orig_h * 1024 / orig_w))
        else:
            scaled_h, scaled_w = 1024, int(round(orig_w * 1024 / orig_h))

        pad_h, pad_w = 1024 - scaled_h, 1024 - scaled_w
        pt, pl       = pad_h // 2, pad_w // 2
        pb, pr       = pad_h - pt, pad_w - pl

        mask = cv2.resize(
            mask[pt : 1024 - pb, pl : 1024 - pr],
            (orig_w, orig_h),
            interpolation=cv2.INTER_NEAREST,
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        return mask, conf

    def crop_candidates(self, image_bgr: np.ndarray) -> list[CropResult]:
        """
        Возвращает список кандидатов кропа (лучший первым).
        Последним всегда стоит полный кадр как безопасный fallback.
        """
        mask, conf = self._predict_mask(image_bgr)
        orig_h, orig_w = image_bgr.shape[:2]
        full_box   = self._full_frame_polygon(orig_h, orig_w)
        full_frame = CropResult(image_bgr, full_box, conf, True, "full_frame_fallback")

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if num_labels <= 1:
            return [CropResult(image_bgr, full_box, conf, False, "fallback_mask_empty")]

        largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        clean_mask    = np.zeros_like(mask)
        clean_mask[labels == largest_label] = 255

        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_contour = None
        best_score = -1

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 5000:
                continue
            rect = cv2.minAreaRect(contour)
            w, h = rect[1]
            if min(w, h) < 50:
                continue
            aspect = max(w, h) / max(min(w, h), 1)
            aspect_penalty = abs(aspect - 1.45)
            score = area - aspect_penalty * 50000
            if score > best_score:
                best_score = score
                best_contour = contour

        if best_contour is None:
            return [CropResult(
                image_bgr,
                full_box,
                conf,
                False,
                "fallback_no_valid_contour"
            )]

        largest = best_contour
        document_area = cv2.contourArea(largest)
        coverage      = document_area / (orig_w * orig_h)

        if coverage >= COVERAGE_SKIP_THRESHOLD:
            return [CropResult(image_bgr, full_box, conf, True, f"skip_crop_coverage_{coverage:.3f}")]

        rect        = cv2.minAreaRect(cv2.convexHull(largest))
        rect_w, rect_h = rect[1]
        min_side    = min(rect_w, rect_h)
        aspect      = max(rect_w, rect_h) / max(min_side, 1e-6)

        if (
            coverage < MIN_DOCUMENT_COVERAGE
            or (min_side / max(min(orig_h, orig_w), 1)) < MIN_DOCUMENT_SIDE_RATIO
            or aspect > MAX_DOCUMENT_ASPECT_RATIO
        ):
            return [CropResult(
                image_bgr, full_box, conf, False,
                f"fallback_implausible_coverage_{coverage:.3f}_aspect_{aspect:.2f}",
            )]

        box        = order_points(cv2.boxPoints(rect).astype(np.float32))
        candidates: list[CropResult] = []

        if _rect_angle_is_straight(rect):
            cropped, straight_box = self._bbox_crop(image_bgr, box)
            if cropped is not None:
                candidates.append(CropResult(
                    cropped, straight_box, conf, True,
                    f"axis_crop_angle_{rect[2]:.1f}deg_coverage_{coverage:.3f}",
                ))
        else:
            exp_box = box.mean(axis=0) + (box - box.mean(axis=0)) * WARP_BOX_EXPAND_RATIO
            warped  = four_point_transform(image_bgr, exp_box)
            if min(warped.shape[:2]) >= MIN_WARP_SIDE_PX:
                candidates.append(CropResult(
                    warped, exp_box, conf, True,
                    f"warp_angle_{rect[2]:.1f}deg_coverage_{coverage:.3f}",
                ))
            bbox_crop, bbox_box = self._bbox_crop(image_bgr, box)
            if bbox_crop is not None:
                candidates.append(CropResult(
                    bbox_crop, bbox_box, conf, True,
                    f"bbox_backup_angle_{rect[2]:.1f}deg_coverage_{coverage:.3f}",
                ))

        candidates.append(full_frame)
        h, w = image_bgr.shape[:2]
        center_crop = image_bgr[
            int(h * 0.05):int(h * 0.95),
            int(w * 0.05):int(w * 0.95)
        ]
        candidates.append(
            CropResult(
                center_crop,
                full_box,
                conf,
                True,
                "center_crop"
            )
        )
        return candidates

    def crop(self, image_bgr: np.ndarray) -> CropResult:
        return self.crop_candidates(image_bgr)[0]


# ---------------------------------------------------------------------------
# Разбивка MRZ-зоны на строки
# ---------------------------------------------------------------------------

def _auto_rotate_mrz(mrz_image: np.ndarray) -> np.ndarray:
    """
    Если MRZ-зона вертикальная (высота > ширины * 1.5) — доворачивает на 90°.
    MRZ всегда горизонтальна: ширина >> высота.
    """
    h, w = mrz_image.shape[:2]
    if h > w * 1.5:
        return cv2.rotate(mrz_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return mrz_image


def detect_mrz_region(document: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(document, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # нижние 55% документа
    roi = gray[int(h * 0.45):, :]

    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
    sq_kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))

    blackhat = cv2.morphologyEx(roi, cv2.MORPH_BLACKHAT, rect_kernel)

    grad_x = cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=-1)
    grad_x = np.absolute(grad_x)

    min_val = grad_x.min()
    max_val = grad_x.max()
    if max_val - min_val > 0:
        grad_x = (255 * (grad_x - min_val) / (max_val - min_val))
    grad_x = grad_x.astype("uint8")

    grad_x = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, rect_kernel)
    thresh  = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh  = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sq_kernel)
    thresh  = cv2.erode(thresh, None, iterations=2)
    thresh  = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best_box  = None
    best_score = -1
    candidates = []
    for contour in contours:

        x, y, ww, hh = cv2.boundingRect(contour)

        area = ww * hh
        aspect = ww / max(hh, 1)

        if area < 5000:
            continue

        if aspect < 4:
            continue

        geom_score = area + aspect * 1000

        candidates.append(
            (geom_score, x, y, ww, hh)
        )

    if not candidates:
        return None

    candidates.sort(reverse=True)

    geom_score, x, y, ww, hh = candidates[0]

    best_box = (x, y, ww, hh)

    if best_box is None:
        return None

    x, y, ww, hh = best_box
    pad_x = int(ww * 0.05)
    pad_y = int(hh * 0.35)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(roi.shape[1], x + ww + pad_x)
    y2 = min(roi.shape[0], y + hh + pad_y)

    mrz = roi[y1:y2, x1:x2]
    return cv2.cvtColor(mrz, cv2.COLOR_GRAY2BGR)


def split_mrz_lines(mrz_image: np.ndarray, expected_lines: int) -> list[np.ndarray]:
    if mrz_image is None or expected_lines <= 0:
        return []

    mrz_image = _auto_rotate_mrz(mrz_image)

    gray = cv2.cvtColor(mrz_image, cv2.COLOR_BGR2GRAY) if mrz_image.ndim == 3 else mrz_image
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 15
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    projection = np.sum(binary > 0, axis=1).astype(np.float32)
    projection = cv2.GaussianBlur(projection.reshape(-1, 1), (1, 11), 0).flatten()

    threshold = projection.max() * 0.30
    active    = projection > threshold

    segments = []
    start = None
    for i, flag in enumerate(active):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start > 5:
                segments.append((start, i))
            start = None
    if start is not None:
        segments.append((start, len(active)))

    if len(segments) < expected_lines:
        h    = gray.shape[0]
        step = h // expected_lines
        segments = []
        for i in range(expected_lines):
            y1 = i * step
            y2 = h if i == expected_lines - 1 else (i + 1) * step
            segments.append((y1, y2))

    segments = sorted(segments, key=lambda x: x[1] - x[0], reverse=True)[:expected_lines]
    segments.sort(key=lambda x: x[0])

    result = []
    pad    = max(3, gray.shape[0] // 40)
    for y1, y2 in segments:
        y1 = max(0, y1 - pad)
        y2 = min(gray.shape[0], y2 + pad)
        result.append(mrz_image[y1:y2].copy())

    return result


def _clean_mrz_ocr_text(text: str) -> str:
    return "".join(ch for ch in text.replace(" ", "").upper() if ch in _MRZ_CHARS)


def _choose_mrz_ocr_variant(candidates: list[str]) -> str:
    cleaned = []
    seen = set()
    for candidate in candidates:
        text = _clean_mrz_ocr_text(candidate)
        if text and text not in seen:
            cleaned.append(text)
            seen.add(text)
    if not cleaned:
        return ""

    # If a second OCR pass only restores trailing filler characters, keep it.
    for longer in sorted(cleaned, key=len, reverse=True):
        for shorter in cleaned:
            if longer == shorter:
                continue
            if longer.startswith(shorter) and set(longer[len(shorter):]) <= {"<"}:
                return longer

    def score(text: str) -> tuple[int, int, int]:
        trailing_fillers = len(text) - len(text.rstrip("<"))
        return (trailing_fillers, text.count("<"), len(text))

    return max(cleaned, key=score)


def _read_easyocr_mrz_line(reader, line_img: np.ndarray) -> str:
    variants: list[str] = []
    read_configs = [
        {},
        {
            "text_threshold": 0.05,
            "low_text": 0.05,
            "link_threshold": 0.05,
            "mag_ratio": 2.0,
        },
    ]
    for config in read_configs:
        try:
            results = reader.readtext(
                line_img,
                detail=0,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<",
                **config,
            )
        except Exception:
            continue
        variants.append("".join(results))
    return _choose_mrz_ocr_variant(variants)


# ---------------------------------------------------------------------------
# MRZPipeline
# ---------------------------------------------------------------------------

class MRZPipeline:
    def __init__(self, crop_model: Path, device: torch.device, debug: bool = False) -> None:

        self.debug = debug

        self.cropper = DocumentCropper(crop_model, device)
        backend = (
            cb.Backend.cuda
            if torch.cuda.is_available()
            else cb.Backend.cpu
        )
        self.scanner = MRZScanner(model_type=ModelType.two_stage, backend=backend)
        import easyocr

        try:
            self.easyocr_reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
            self.has_easyocr = True

        except Exception:
            self.easyocr_reader = None
            self.has_easyocr = False

    def log(self, *args):

        if self.debug:
            print(*args)

    def _run_scanner(self, image: np.ndarray, rotation: int) -> MRZResult:
        self.log("RUN", rotation, image.shape)
        try:
            t0 = time.perf_counter()
            result = self.scanner(image, do_center_crop=False, do_postprocess=True)
            print(
                f"MRZScanner: "
                f"{time.perf_counter() - t0:.2f}s"
            )
        except Exception as exc:
            self.log("scanner error:", exc)
            return MRZResult(
                [],
                np.empty((0, 2), dtype=np.float32),
                None,
                [],
                rotation,
                f"scanner_error:{exc}",
                -1.0,
            )

        texts = [str(t) for t in result.get("mrz_texts", []) if str(t)]
        polygon = np.asarray(result.get("mrz_polygon", []), dtype=np.float32)
        msg = result.get("msg", "")

        # Извлекаем MRZ-зону из полигона
        mrz_image = four_point_transform(image, polygon) if polygon.shape == (4, 2) else None

        # -----------------------------------------------------------------
        # ИСПРАВЛЕНИЕ: перебираем ориентации MRZ-зоны (0° и 180°) и выбираем
        # ту, у которой сканер даёт лучший score.
        # Теперь мы сохраняем и обновляем корректные тексты (texts)!
        # -----------------------------------------------------------------
        if mrz_image is not None:
            mrz_image, texts = _best_orientation_mrz(mrz_image, self.scanner)

        # Вычисляем валидацию и скоринг на основе ФИНАЛЬНОГО корректного текста
        validation = validate_mrz(texts)
        score = mrz_score(texts, msg, validation)

        # Штраф за плохое соотношение сторон (MRZ всегда широкая)
        if mrz_image is not None:
            h, w = mrz_image.shape[:2]
            aspect = w / max(h, 1)
            if aspect < 4:
                score -= 40

        # Финальная авто-ротация на случай вертикального изображения
        if mrz_image is not None:
            mrz_image = _auto_rotate_mrz(mrz_image)

        line_images = split_mrz_lines(mrz_image, len(texts)) if mrz_image is not None else []

        # УЛУЧШЕННЫЙ FALLBACK: Используем универсальный OCR на каждую строку
        # -----------------------------------------------------------------
        if not mrz_is_fully_valid(validation) and line_images and self.has_easyocr:
            t0 = time.perf_counter()
            raw_fallback_texts = []
            for idx, line_img in enumerate(line_images):
                raw_fallback_texts.append(_read_easyocr_mrz_line(self.easyocr_reader, line_img))

            # Keep each OCR line at the length EasyOCR actually observed.
            fallback_texts = [text for text in raw_fallback_texts if text]

            if len(fallback_texts) == len(texts):
                validation_fb = validate_mrz(fallback_texts)
                score_fb = mrz_score(fallback_texts, msg, validation_fb)

                # Считаем количество букв (имя держателя)
                alpha_orig = sum(c.isalpha() for c in "".join(texts))
                alpha_fb = sum(c.isalpha() for c in "".join(fallback_texts))

                # Заменяем оригинальный текст на fallback, если:
                # 1. Fallback полностью валиден
                # 2. Score у fallback выше
                # 3. Или fallback содержит гораздо больше букв (считалось имя вместо цифр)
                if (
                        mrz_is_fully_valid(validation_fb)
                        or score_fb > score + 5.0
                ):
                    texts = fallback_texts
                    validation = validation_fb
                    score = score_fb
            print(
                f"EasyOCR fallback: "
                f"{time.perf_counter() - t0:.2f}s"
            )
        return MRZResult(texts, polygon, mrz_image, line_images, rotation, str(msg), score, validation)
    def scan_mrz(self, document_bgr: np.ndarray) -> MRZResult:
        start = time.perf_counter()
        best = MRZResult(
            [],
            np.empty((0, 2), dtype=np.float32),
            None,
            [],
            0,
            "not_run",
            -1.0
        )

        for rotation in (0, 180):
            img       = rotate_image(document_bgr, rotation)
            candidate = self._run_scanner(img, rotation)

            if final_mrz_score(candidate) > final_mrz_score(best):
                best = candidate

            if mrz_is_fully_valid(best.validation):
                return best
        print(
            f"scan_mrz: "
            f"{time.perf_counter() - start:.2f}s"
        )
        return best

    def scan_best_crop(self, image_bgr: np.ndarray) -> tuple[CropResult, MRZResult, list[CropResult]]:
        """
        Перебирает кандидатов кропа, для каждого запускает scan_mrz.

        Логика выбора:
        1. Полностью валидный MRZ → досрочный выход.
        2. Лучший по score; при равном score — больше валидных цифр.
        3. Если никто не дал структурно корректный MRZ — берём лучший по score.
        """
        start = time.perf_counter()
        crop_candidates = self.cropper.crop_candidates(image_bgr)

        for i, crop in enumerate(crop_candidates):
            h, w = crop.image.shape[:2]

            self.log(
                i,
                crop.message,
                f"{w}x{h}"
            )

        best_crop   = crop_candidates[0]
        best_result = self.scan_mrz(best_crop.image)



        if mrz_is_fully_valid(best_result.validation):
            return best_crop, best_result, crop_candidates

        for crop in crop_candidates[1:]:
            candidate_result = self.scan_mrz(crop.image)


            candidate_score = final_mrz_score(candidate_result)
            best_score = final_mrz_score(best_result)

            is_better = False

            if candidate_score > best_score:
                is_better = True

            elif abs(candidate_score - best_score) < 1.0:

                if (
                        candidate_result.validation.valid_digits
                        >
                        best_result.validation.valid_digits
                ):
                    is_better = True

                elif (
                        len("".join(candidate_result.texts))
                        >
                        len("".join(best_result.texts))
                ):
                    is_better = True
            if is_better:
                best_crop   = crop
                best_result = candidate_result

            if mrz_is_fully_valid(best_result.validation):
                break
        print(
            f"scan_best_crop: "
            f"{time.perf_counter() - start:.2f}s"
        )
        return best_crop, best_result, crop_candidates

    def process_image(self, path: Path, out_dir: Path, save_debug: bool = True) -> dict[str, object]:

        pdf_mrz_lines = []

        if path.suffix.lower() == ".pdf":

            pdf_mrz_lines = extract_mrz_from_pdf_text(path)

            pdf_pages = pdf_to_images(path)

            if not pdf_pages:
                return {
                    "image": str(path),
                    "status": "pdf_read_failed"
                }

            best_crop = None
            best_mrz = None
            best_score = -1
            crop_candidates = []

            for image in pdf_pages:

                crop, mrz, candidates = self.scan_best_crop(image)

                score = final_mrz_score(mrz)

                if score > best_score:
                    best_score = score
                    best_crop = crop
                    best_mrz = mrz
                    crop_candidates = candidates

            crop = best_crop
            mrz = best_mrz

            # только для debug-сохранения
            image = pdf_pages[0]
        elif path.suffix.lower() in {".tif", ".tiff"}:

            pages = tiff_to_images(path)

            if not pages:
                return {
                    "image": str(path),
                    "status": "tiff_read_failed"
                }

            best_crop = None
            best_mrz = None
            best_score = -1

            for image in pages:

                crop, mrz, candidates = self.scan_best_crop(image)

                score = final_mrz_score(mrz)

                if score > best_score:
                    best_score = score
                    best_crop = crop
                    best_mrz = mrz
                    crop_candidates = candidates

            crop = best_crop
            mrz = best_mrz
            image = pages[0]
            
        else:

            image = imread_unicode(path)

            if image is None:
                return {
                    "image": str(path),
                    "status": "read_failed",
                    "mrz_text": "",
                    "message": "cv2_imread_failed"
                }

            crop, mrz, crop_candidates = self.scan_best_crop(image)


        # 2. Если текст из PDF успешно извлечен, подменяем им результат распознавания
        if pdf_mrz_lines:
            mrz.texts = pdf_mrz_lines
            mrz.validation = validate_mrz(pdf_mrz_lines)
            mrz.score = 9999.0  # Даем максимальный приоритет правильному тексту

        stem      = path.stem
        debug_dir = out_dir / "debug" / stem
        if save_debug:
            imwrite_unicode(debug_dir / "input.jpg",         image)
            imwrite_unicode(debug_dir / "document_crop.jpg", crop.image)
            for idx, c in enumerate(crop_candidates, start=1):
                imwrite_unicode(debug_dir / f"document_candidate_{idx}.jpg", c.image)
            if mrz.mrz_image is not None:
                imwrite_unicode(debug_dir / "mrz_zone.jpg", mrz.mrz_image)
            for idx, line_img in enumerate(mrz.line_images, start=1):
                imwrite_unicode(debug_dir / f"mrz_line_{idx}.jpg", line_img)

        status = "ok" if mrz.texts else "mrz_failed"
        v = mrz.validation
        return {
            "image":             str(path),
            "status":            status,
            "crop_ok":           crop.ok,
            "crop_message":      crop.message,
            "crop_candidates":   len(crop_candidates),
            "mask_min":          f"{crop.mask_confidence[0]:.6f}",
            "mask_max":          f"{crop.mask_confidence[1]:.6f}",
            "rotation":          mrz.rotation,
            "scanner_msg":       mrz.msg,
            "score":             f"{mrz.score:.3f}",
            "line_count":        len(mrz.texts),
            "mrz_format":        v.format,
            "mrz_check_digits":  f"{v.valid_digits}/{v.total_digits}",
            "mrz_structural_ok": v.structural_ok,
            "mrz_fully_valid":   mrz_is_fully_valid(v),
            "mrz_text":          "\\n".join(mrz.texts),
        }

    def process_ndarray(
            self,
            image: np.ndarray
    ) -> MRZResult:

        crop, mrz, _ = self.scan_best_crop(image)

        return mrz

    def process_bytes(
            self,
            data: bytes
    ) -> MRZResult:

        nparr = np.frombuffer(data, dtype=np.uint8)

        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Unable to decode image")

        return self.process_ndarray(image)

    def process_file_bytes(
            self,
            data: bytes,
            filename: str
    ) -> MRZResult:
        start = time.perf_counter()
        suffix = Path(filename).suffix.lower()

        if suffix == ".pdf":

            # Сначала пытаемся достать MRZ из текстового слоя PDF
            try:
                doc = fitz.open(
                    stream=data,
                    filetype="pdf"
                )

                for page in doc:

                    text = page.get_text()

                    lines = [
                        line.strip().replace(" ", "").upper()
                        for line in text.splitlines()
                    ]

                    mrz_lines = [
                        line
                        for line in lines
                        if len(line) >= 28
                           and all(
                            c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
                            for c in line
                        )
                    ]

                    if len(mrz_lines) in (2, 3):
                        return MRZResult(
                            texts=mrz_lines,
                            polygon=np.empty((0, 2), dtype=np.float32),
                            mrz_image=None,
                            line_images=[],
                            rotation=0,
                            msg="pdf_text_layer",
                            score=9999.0,
                            validation=validate_mrz(mrz_lines)
                        )

            except Exception:
                pass

            # если текстового слоя нет — идём через картинки
            pages = pdf_bytes_to_images(data)

        elif suffix in {".tif", ".tiff"}:

            pages = tiff_bytes_to_images(data)
            if not pages:
                raise ValueError("No pages found")

            best_result = None
            best_score = -1

            for image in pages:

                result = self.process_ndarray(image)

                score = final_mrz_score(result)

                if score > best_score:
                    best_score = score
                    best_result = result

            return best_result

        else:

            return self.process_bytes(data)

# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------

def iter_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(p for p in input_path.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def write_results(out_dir: Path, rows: list[dict[str, object]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with (out_dir / "results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Document crop -> MRZ detect -> MRZ recognize pipeline.")
    parser.add_argument("--input",      type=Path, default=Path(__file__).parent / "local_samples")
    parser.add_argument("--out",        type=Path, default=Path("pipeline_results"))
    parser.add_argument("--crop-model", type=Path, default=Path(__file__).parent / "models" / "unet_resnet34.pth")
    parser.add_argument("--no-debug",   action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"input:  {args.input}")
    print(f"output: {args.out}")

    images = iter_images(args.input)
    if not images:
        print("No images found.")
        return 1

    pipeline = MRZPipeline(args.crop_model, device)
    rows: list[dict[str, object]] = []

    for idx, path in enumerate(images, start=1):
        print(f"\n[{idx}/{len(images)}] {path}")
        row = pipeline.process_image(path, args.out, save_debug=not args.no_debug)
        rows.append(row)
        print(
            f"  status:    {row['status']}\n"
            f"  scanner:   {row.get('scanner_msg')}\n"
            f"  crop:      {row.get('crop_message')}\n"
            f"  format:    {row.get('mrz_format')}  "
            f"check_digits: {row.get('mrz_check_digits')}  "
            f"fully_valid: {row.get('mrz_fully_valid')}\n"
            f"  score:     {row.get('score')}"
        )
        text = str(row.get("mrz_text", ""))
        print(text.replace("\\n", "\n") if text else "  MRZ: <empty>")

    write_results(args.out, rows)
    print(f"\nSaved: {args.out / 'results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
