from fastapi import APIRouter

from app.api.v1.endpoints.mrz import router as mrz_router


api_router = APIRouter()
api_router.include_router(mrz_router)
