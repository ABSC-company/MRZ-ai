from contextlib import asynccontextmanager
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


@asynccontextmanager
async def lifespan(app: FastAPI):

    global pipeline

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    pipeline = MRZPipeline(
        crop_model=Path("models/unet_resnet34.pth"),
        device=device,
        debug=False
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

