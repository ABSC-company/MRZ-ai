from pathlib import Path
import sys

import torch

from run_mrz_pipeline import MRZPipeline


def main() -> int:
    if len(sys.argv) > 1:
        image_path = Path(sys.argv[1].strip().strip('"'))
    else:
        image_path = Path(input("Image path: ").strip().strip('"'))
    out_dir = Path("single_image_result")
    pipeline = MRZPipeline(
        crop_model=Path(__file__).parent / "models" / "unet_resnet34.pth",
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    row = pipeline.process_image(image_path, out_dir, save_debug=True)

    print()
    print("IMAGE:", image_path.name)
    print("STATUS:", row["status"])
    print("CROP:", row["crop_message"])
    print("SCANNER:", row["scanner_msg"])
    print("ROTATION:", row["rotation"])
    print()

    mrz_text = str(row.get("mrz_text", ""))
    if mrz_text:
        for idx, line in enumerate(mrz_text.split("\\n"), start=1):
            print(f"LINE {idx}:")
            print(line)
            print()
    else:
        print("MRZ: <empty>")

    print("Debug:", out_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
