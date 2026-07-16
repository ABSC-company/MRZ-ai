"""Stable public imports for the MRZ/document detection layer."""

from app.services.mrz_pipeline import CropResult, DocumentCropper

__all__ = ["CropResult", "DocumentCropper"]
