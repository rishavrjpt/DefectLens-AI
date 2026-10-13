"""Pydantic schemas for DefectLens AI API requests and responses."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., description="Server status ('healthy')")
    model_loaded: bool = Field(..., description="Whether a trained model is loaded and ready")
    model_path: Optional[str] = Field(None, description="Path to active model checkpoint")
    category: Optional[str] = Field(None, description="Active dataset category")
    device: str = Field(..., description="Compute device being used (mps/cuda/cpu)")


class ModelInfoResponse(BaseModel):
    model_name: str
    category: str
    backbone: str
    calibrated_threshold: Optional[float]
    layers: List[str]
    coreset_sampling_ratio: Optional[float]
    memory_bank_size: Optional[int]
    is_trained: bool
    evaluation_available: bool
    training_summary: Optional[Dict[str, Any]] = None
    evaluation_summary: Optional[Dict[str, Any]] = None


class PredictionResponse(BaseModel):
    status: str = Field(..., description="'Likely Normal' or 'Potential Anomaly'")
    is_anomaly: bool = Field(..., description="True if anomaly score >= calibrated threshold")
    anomaly_score: float = Field(..., description="Max patch distance to memory bank")
    threshold: float = Field(..., description="Calibrated decision threshold from validation data")
    score_ratio: float = Field(..., description="Ratio of anomaly score to threshold")
    anomalous_area_percentage: float = Field(..., description="Estimated percentage of surface area flagged")
    original_image_base64: str = Field(..., description="Base64 encoded original image preview")
    heatmap_base64: Optional[str] = Field(None, description="Base64 encoded anomaly localization heatmap")
    overlay_base64: Optional[str] = Field(None, description="Base64 encoded heatmap overlaid on input image")
    disclaimer: str = Field(..., description="Educational prototype disclaimer")


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    suggested_action: Optional[str] = None
