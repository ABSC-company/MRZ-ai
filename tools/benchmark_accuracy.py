from pathlib import Path
import csv

import torch

from run_mrz_pipeline import MRZPipeline


# ============================================================
# НАСТРОЙКИ
# ============================================================

DATASET_ROOT = Path(
    r"D:\MRZ-dataset\MRZ-dataset-gh-pages\benchmark\dataset\Passports\BEL"
)

OUTPUT_CSV = "benchmark_report.csv"

DEBUG_DIR = Path("benchmark_debug")


# ============================================================
# LEVENSHTEIN
# ============================================================

def levenshtein(a: str, b: str) -> int:
    rows = len(a) + 1
    cols = len(b) + 1

    dist = [[0] * cols for _ in range(rows)]

    for i in range(rows):
        dist[i][0] = i

    for j in range(cols):
        dist[0][j] = j

    for i in range(1, rows):
        for j in range(1, cols):

            cost = 0 if a[i - 1] == b[j - 1] else 1

            dist[i][j] = min(
                dist[i - 1][j] + 1,
                dist[i][j - 1] + 1,
                dist[i - 1][j - 1] + cost,
            )

    return dist[-1][-1]


# ============================================================
# ЗАГРУЗКА ЭТАЛОНА
# ============================================================

def read_gt(txt_path: Path) -> str:
    return (
        txt_path.read_text(
            encoding="utf-8",
            errors="ignore"
        )
        .replace("\r", "")
        .strip()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    pipeline = MRZPipeline(
        Path("models/unet_resnet34.pth"),
        device
    )

    image_files = []

    for ext in (
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.tif",
        "*.tiff",
        "*.bmp",
        "*.webp",
    ):
        image_files.extend(DATASET_ROOT.rglob(ext))

    image_files = sorted(image_files)

    print("Images found:", len(image_files))

    total_images = 0
    detected_images = 0
    exact_matches = 0

    total_chars = 0
    correct_chars = 0
    total_edit_distance = 0

    rows = []

    for idx, image_path in enumerate(image_files, start=1):

        txt_path = image_path.with_suffix(".txt")

        if not txt_path.exists():
            continue

        total_images += 1

        print(
            f"[{idx}/{len(image_files)}] "
            f"{image_path.name}"
        )

        try:

            result = pipeline.process_image(
                image_path,
                DEBUG_DIR,
                save_debug=False
            )

            predicted = (
                str(result.get("mrz_text", ""))
                .replace("\\n", "\n")
                .replace("\r", "")
                .strip()
            )

            status = result.get("status", "")

        except Exception as e:

            predicted = ""
            status = f"exception:{e}"

        ground_truth = read_gt(txt_path)

        detected = len(predicted) > 0

        if detected:
            detected_images += 1

        exact = predicted == ground_truth

        if exact:
            exact_matches += 1

        distance = levenshtein(
            predicted,
            ground_truth
        )

        total_edit_distance += distance

        gt_len = len(ground_truth)

        total_chars += gt_len

        correct_chars += max(
            0,
            gt_len - distance
        )

        cer = (
            distance / gt_len
            if gt_len
            else 0
        )

        rows.append(
            {
                "image": str(image_path),
                "status": status,
                "detected": detected,
                "exact_match": exact,
                "edit_distance": distance,
                "cer": f"{cer:.6f}",
                "predicted": predicted,
                "ground_truth": ground_truth,
            }
        )

    # ========================================================
    # METRICS
    # ========================================================

    detection_rate = (
        detected_images / total_images * 100
        if total_images
        else 0
    )

    exact_accuracy = (
        exact_matches / total_images * 100
        if total_images
        else 0
    )

    char_accuracy = (
        correct_chars / total_chars * 100
        if total_chars
        else 0
    )

    cer_total = (
        total_edit_distance / total_chars
        if total_chars
        else 0
    )

    print()
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    print(
        f"Images:               {total_images}"
    )

    print(
        f"Detected:             {detected_images}"
    )

    print(
        f"Detection Rate:       "
        f"{detection_rate:.2f}%"
    )

    print(
        f"Exact Match Accuracy: "
        f"{exact_accuracy:.2f}%"
    )

    print(
        f"Character Accuracy:   "
        f"{char_accuracy:.2f}%"
    )

    print(
        f"CER:                  "
        f"{cer_total:.6f}"
    )

    # ========================================================
    # CSV
    # ========================================================

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("Report saved:")
    print(Path(OUTPUT_CSV).resolve())


if __name__ == "__main__":
    main()