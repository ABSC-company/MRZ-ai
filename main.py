from contextlib import asynccontextmanager
import os
from pathlib import Path

import torch
import time

from fastapi import FastAPI
from fastapi import UploadFile
from fastapi import File
from fastapi import HTTPException
from fastapi.responses import PlainTextResponse
from run_mrz_pipeline import MRZPipeline


pipeline = None


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int | None) -> int | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    parsed = int(value)
    return parsed if parsed > 0 else None


@asynccontextmanager
async def lifespan(app: FastAPI):

    global pipeline

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    max_crops = _env_int("MRZ_MAX_CROPS", 2)
    try_upside_down = _env_flag("MRZ_TRY_UPSIDE_DOWN", True)
    orientation_retry = _env_flag("MRZ_ORIENTATION_RETRY", True)
    use_easyocr = _env_flag("MRZ_USE_EASYOCR", True)
    easyocr_aggressive = _env_flag("MRZ_EASYOCR_AGGRESSIVE", True)

    print(
        "MRZ API config: "
        f"device={device} "
        f"max_crops={max_crops or 'all'} "
        f"try_upside_down={try_upside_down} "
        f"orientation_retry={orientation_retry} "
        f"use_easyocr={use_easyocr} "
        f"easyocr_aggressive={easyocr_aggressive}"
    )

    pipeline = MRZPipeline(
        crop_model=Path("models/unet_resnet34.pth"),
        device=device,
        debug=False,
        max_crop_candidates=max_crops,
        try_upside_down=try_upside_down,
        orientation_retry=orientation_retry,
        use_easyocr=use_easyocr,
        easyocr_aggressive=easyocr_aggressive,
    )

    yield

    pipeline = None


app = FastAPI(
    title="MRZ Scanner",
    version="1.0",
    lifespan=lifespan
)


@app.post("/scan", response_class=PlainTextResponse)
def scan(
        file: UploadFile = File(...)
):

    if pipeline is None:
        raise HTTPException(
            status_code=500,
            detail="Pipeline not initialized"
        )

    start = time.perf_counter()

    try:

        data = file.file.read()

        print(
            f"filename={file.filename} "
            f"bytes={len(data)}"
        )

        result = pipeline.process_file_bytes(
            data=data,
            filename=file.filename
        )

        print(
            f"TOTAL API TIME: "
            f"{time.perf_counter() - start:.2f}s"
        )

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    return "\n".join(result.texts)

