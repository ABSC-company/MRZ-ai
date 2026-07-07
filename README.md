# MRZ Project

End-to-end MRZ extraction project for passport/ID photos and PDFs.

The project has two entry points:

- `run_mrz_pipeline.py` - CLI/batch pipeline.
- `main.py` - FastAPI service used by Docker/uvicorn.

The local `MRZScanner/` folder is vendored into this repository on purpose. It
contains the Docsaid MRZScanner package and ONNX checkpoints used by the
pipeline, so the project can be installed without relying on an editable GitHub
URL.

## Project Layout

```text
MRZ Project/
  main.py                 # FastAPI app: POST /scan
  run_mrz_pipeline.py     # CLI pipeline: crop document -> detect MRZ -> OCR
  turbojpeg.py            # OpenCV-backed PyTurboJPEG fallback for Windows
  Dockerfile
  .dockerignore
  .gitignore
  requirements.txt
  models/
    unet_resnet34.pth     # document cropper model
  MRZScanner/
    mrzscanner/           # vendored MRZScanner package + ONNX checkpoints
    setup.py
    setup.cfg
    LICENSE
    README.md
  tools/
    benchmark_accuracy.py
    test_real.py
```

Generated folders such as `.venv/`, `pipeline_results/`, `__pycache__/`, IDE
metadata, and `local_samples/` are ignored by git.

## Requirements

- Python 3.11
- NVIDIA GPU is optional; CPU works but is slower.
- For local development on Windows, keep the project root as the working
  directory so `turbojpeg.py` shadows the external `turbojpeg` import when
  needed.

Install:

```powershell
cd "D:\ML\MRZ Project"
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

If you already have a working `.venv`, just activate it:

```powershell
cd "D:\ML\MRZ Project"
.\.venv\Scripts\activate
```

Quick import check:

```powershell
python -c "import torch, cv2, fitz; from mrzscanner import MRZScanner; print('ok', torch.cuda.is_available())"
```

## CLI Usage

Process a folder or a single file:

```powershell
python run_mrz_pipeline.py --input local_samples --out pipeline_results
```

Run without debug image dumps:

```powershell
python run_mrz_pipeline.py --input "path\to\images" --out pipeline_results --no-debug
```

Use a custom crop model:

```powershell
python run_mrz_pipeline.py `
  --input "path\to\images" `
  --out pipeline_results `
  --crop-model models\unet_resnet34.pth
```

Supported input extensions include images, PDFs, HEIC/HEIF, and TIFF variants.

Outputs:

- `pipeline_results/results.csv`
- `pipeline_results/results.json`
- `pipeline_results/debug/<file_stem>/...` unless `--no-debug` is set

## API Usage

Start the API locally:

```powershell
python -m uvicorn main:app --host 127.0.0.1 --port 5000
```

Open:

```text
http://127.0.0.1:5000/docs
```

Endpoint:

```text
POST /scan
multipart/form-data file=<image-or-pdf>
```

The response is plain text with recognized MRZ lines.

## Docker

Build:

```powershell
docker build -t mrz-project .
```

Run:

```powershell
docker run --rm -p 8000:8000 mrz-project
```

Open:

```text
http://127.0.0.1:8000/docs
```

GPU Docker requires NVIDIA Container Toolkit. Example:

```powershell
docker run --rm --gpus all -p 8000:8000 mrz-project
```

## GitHub Notes

This repository intentionally includes:

- `run_mrz_pipeline.py`
- `turbojpeg.py`
- `main.py`
- `MRZScanner/`
- Docker files
- `models/unet_resnet34.pth`

Before pushing to GitHub, check large files:

```powershell
git ls-files | ForEach-Object {
  $item = Get-Item $_ -ErrorAction SilentlyContinue
  if ($item -and $item.Length -gt 90MB) {
    "{0:N1} MB`t{1}" -f ($item.Length / 1MB), $_
  }
}
```

GitHub rejects individual files larger than 100 MB. If `models/unet_resnet34.pth`
or any ONNX checkpoint crosses that limit in the future, move it to GitHub
Releases or Git LFS.

## Useful Tools

Run real-image smoke test:

```powershell
python tools\test_real.py
```

Run benchmark script:

```powershell
python tools\benchmark_accuracy.py
```
