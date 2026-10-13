"""FastAPI application entrypoint for DefectLens AI.

Provides RESTful endpoints for model health, metadata, anomaly prediction,
and ML insight dashboards. Serves static assets for the web UI.
"""

import json
import os
from typing import Any, Dict
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.inference import ModelService
from app.schemas import (
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)

# App initialization
app = FastAPI(
    title="DefectLens AI",
    description="Industrial Surface Anomaly Detection & Visual Localization API",
    version="1.0.0",
)

# CORS: the bundled UI is same-origin, so cross-origin access is opt-in via a
# comma-separated ALLOWED_ORIGINS env var (e.g. "https://example.com,http://localhost:3000").
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ModelService singleton
model_service = ModelService.get_instance()

# Static files path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs("results/plots", exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/results", StaticFiles(directory="results"), name="results")


@app.get("/", include_in_schema=False)
async def serve_index():
    """Serve the single-page application frontend."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "DefectLens AI API is running. Web UI not found in static folder."}


@app.get("/api/health", response_model=HealthResponse)
async def get_health():
    """Healthcheck endpoint returning server and model readiness."""
    is_ready = model_service.is_ready()
    return HealthResponse(
        status="healthy",
        model_loaded=is_ready,
        model_path=model_service.model_path if is_ready else None,
        category=model_service.category,
        device=str(model_service.detector.device) if is_ready else "not_loaded",
    )


@app.get("/api/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """Return model configuration, backbone, and training metadata."""
    info = model_service.get_info()

    # Load evaluation summary if exists
    eval_file = "results/metrics.json"
    eval_summary = None
    if os.path.exists(eval_file):
        try:
            with open(eval_file, "r") as f:
                eval_summary = json.load(f)
        except Exception:
            pass

    return ModelInfoResponse(
        model_name=info["model_name"],
        category=info["category"],
        backbone=info["backbone"],
        calibrated_threshold=info.get("calibrated_threshold"),
        layers=info.get("layers", []),
        coreset_sampling_ratio=info.get("coreset_sampling_ratio"),
        memory_bank_size=info.get("memory_bank_size"),
        is_trained=info.get("is_trained", False),
        evaluation_available=eval_summary is not None,
        training_summary=info.get("training_summary"),
        evaluation_summary=eval_summary,
    )


@app.get("/api/insights")
async def get_ml_insights():
    """Return comprehensive dataset, training, and evaluation insights."""
    # 1. Dataset summary
    ds_file = "data/dataset_summary.json"
    dataset_summary = None
    if os.path.exists(ds_file):
        try:
            with open(ds_file, "r") as f:
                dataset_summary = json.load(f)
        except Exception:
            pass

    # 2. Training summary
    train_file = f"models/training_summary_{model_service.category}.json"
    training_summary = None
    if os.path.exists(train_file):
        try:
            with open(train_file, "r") as f:
                training_summary = json.load(f)
        except Exception:
            pass

    # 3. Evaluation metrics
    eval_file = "results/metrics.json"
    evaluation_metrics = None
    if os.path.exists(eval_file):
        try:
            with open(eval_file, "r") as f:
                evaluation_metrics = json.load(f)
        except Exception:
            pass

    return {
        "dataset_summary": dataset_summary,
        "training_summary": training_summary,
        "evaluation_metrics": evaluation_metrics,
        "model_status": {
            "is_trained": model_service.is_ready(),
            "category": model_service.category,
            "threshold": model_service.detector.threshold if model_service.is_ready() else None,
        },
    }


@app.get("/api/dataset/samples")
async def get_dataset_samples(limit_per_category: int = 8):
    """Return a curated gallery of real test images from the MVTec dataset."""
    base_dir = os.path.join("data", "mvtec", model_service.category)
    test_dir = os.path.join(base_dir, "test")
    gt_dir = os.path.join(base_dir, "ground_truth")

    if not os.path.exists(test_dir):
        return {"samples": [], "category": model_service.category, "available": False}

    samples = []
    subdirs = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])

    for defect_type in subdirs:
        folder_path = os.path.join(test_dir, defect_type)
        files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))])

        for fname in files[:limit_per_category]:
            is_anomaly = defect_type != "good"
            mask_url = None

            if is_anomaly:
                base_name = os.path.splitext(fname)[0]
                mask_candidates = [
                    f"{base_name}_mask.png",
                    f"{base_name}.png",
                ]
                for mc in mask_candidates:
                    if os.path.exists(os.path.join(gt_dir, defect_type, mc)):
                        mask_url = f"/api/dataset/image/ground_truth/{defect_type}/{mc}"
                        break

            samples.append({
                "id": f"{defect_type}_{fname}",
                "filename": fname,
                "defect_type": defect_type,
                "display_name": defect_type.replace("_", " ").title(),
                "is_anomaly": is_anomaly,
                "true_label": "Anomaly" if is_anomaly else "Normal",
                "image_url": f"/api/dataset/image/test/{defect_type}/{fname}",
                "mask_url": mask_url,
            })

    return {
        "category": model_service.category,
        "available": True,
        "total_samples": len(samples),
        "samples": samples,
    }


@app.get("/api/dataset/image/{folder}/{defect_type}/{filename}")
async def serve_dataset_image(folder: str, defect_type: str, filename: str):
    """Serve a dataset image safely from data/mvtec."""
    # Prevent directory traversal
    clean_folder = os.path.basename(folder)
    clean_defect = os.path.basename(defect_type)
    clean_filename = os.path.basename(filename)

    file_path = os.path.join(
        "data", "mvtec", model_service.category, clean_folder, clean_defect, clean_filename
    )
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Dataset image not found")

    return FileResponse(file_path, media_type="image/png")


@app.post("/api/dataset/test-sample")
async def test_dataset_sample(payload: Dict[str, str]):
    """Run model inference directly on an official dataset sample and compare with ground truth."""
    if not model_service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "ModelNotTrained",
                "detail": "Model is not trained yet.",
                "suggested_action": "Run: python src/train.py",
            },
        )

    defect_type = os.path.basename(payload.get("defect_type", "good"))
    filename = os.path.basename(payload.get("filename", "000.png"))

    img_path = os.path.join("data", "mvtec", model_service.category, "test", defect_type, filename)
    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail=f"Sample not found: {img_path}")

    # Read image bytes and ground truth mask
    with open(img_path, "rb") as f:
        img_bytes = f.read()

    is_true_anomaly = defect_type != "good"
    gt_mask_b64 = None

    if is_true_anomaly:
        base_name = os.path.splitext(filename)[0]
        gt_dir = os.path.join("data", "mvtec", model_service.category, "ground_truth", defect_type)
        candidate_masks = [
            os.path.join(gt_dir, f"{base_name}_mask.png"),
            os.path.join(gt_dir, f"{base_name}.png"),
        ]
        for c in candidate_masks:
            if os.path.exists(c):
                from PIL import Image
                with Image.open(c) as mask_img:
                    gt_mask_b64 = model_service._pil_to_base64(mask_img.convert("L").resize((224, 224)))
                break

    (
        stat,
        is_anom,
        score,
        thresh,
        ratio,
        area_pct,
        orig_b64,
        heat_b64,
        over_b64,
    ) = model_service.predict_image(img_bytes)

    # Verification of correctness against ground truth
    is_correct = (is_anom == is_true_anomaly)

    return {
        "filename": filename,
        "defect_type": defect_type,
        "true_label": "Anomaly" if is_true_anomaly else "Normal",
        "predicted_label": "Anomaly" if is_anom else "Normal",
        "is_correct": is_correct,
        "status": stat,
        "is_anomaly": is_anom,
        "anomaly_score": score,
        "threshold": thresh,
        "score_ratio": ratio,
        "anomalous_area_percentage": area_pct,
        "original_image_base64": orig_b64,
        "heatmap_base64": heat_b64,
        "overlay_base64": over_b64,
        "ground_truth_mask_base64": gt_mask_b64,
    }


@app.post(
    "/api/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image file format or size"},
        503: {"model": ErrorResponse, "description": "Model is not trained or loaded"},
    },
)
async def predict_anomaly(file: UploadFile = File(...)):
    """Analyze an uploaded product image and predict whether it is normal or anomalous.

    Validates file size (max 10MB) and format (JPEG, PNG). Runs PatchCore inference
    and returns anomaly score, decision, and localization heatmap.
    """
    # 1. Validate MIME type
    valid_content_types = ["image/jpeg", "image/png", "image/jpg", "image/bmp", "image/webp"]
    if file.content_type not in valid_content_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "InvalidFileType",
                "detail": f"Received content type '{file.content_type}'. Expected JPEG, PNG, or BMP.",
                "suggested_action": "Upload an image in PNG or JPEG format.",
            },
        )

    # 2. Validate file size (max 10 MB)
    max_size = 10 * 1024 * 1024
    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "EmptyFile",
                "detail": "Uploaded file contains 0 bytes.",
                "suggested_action": "Select a non-empty image file.",
            },
        )

    if len(image_bytes) > max_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "FileTooLarge",
                "detail": f"File size is {len(image_bytes) / 1024 / 1024:.2f}MB, exceeding 10MB limit.",
                "suggested_action": "Compress image or upload a smaller file.",
            },
        )

    # 3. Check if model is trained and ready
    if not model_service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "ModelNotTrained",
                "detail": f"No trained model found for category '{model_service.category}'.",
                "suggested_action": "Run dataset preparation and training first: python src/train.py",
            },
        )

    # 4. Execute inference
    try:
        (
            stat,
            is_anom,
            score,
            thresh,
            ratio,
            area_pct,
            orig_b64,
            heat_b64,
            over_b64,
        ) = model_service.predict_image(image_bytes)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "InferenceError",
                "detail": str(e),
                "suggested_action": "Ensure the image is a valid readable photo.",
            },
        )

    return PredictionResponse(
        status=stat,
        is_anomaly=is_anom,
        anomaly_score=score,
        threshold=thresh,
        score_ratio=ratio,
        anomalous_area_percentage=area_pct,
        original_image_base64=orig_b64,
        heatmap_base64=heat_b64,
        overlay_base64=over_b64,
        disclaimer=(
            "Educational prototype notice: DefectLens AI model scores represent distance "
            "in pretrained feature space relative to normal surface memory. This is not a "
            "certified industrial quality control decision."
        ),
    )
