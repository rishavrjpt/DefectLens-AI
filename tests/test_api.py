"""API integration tests for DefectLens AI endpoints."""

import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.inference import ModelService


@pytest.fixture
def client():
    return TestClient(app)


def test_api_health(client):
    """Test health check endpoint."""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data
    assert "device" in data


def test_api_model_info(client):
    """Test model-info endpoint schema."""
    resp = client.get("/api/model-info")
    assert resp.status_code == 200
    data = resp.json()
    assert "model_name" in data
    assert "category" in data
    assert "backbone" in data
    assert "is_trained" in data


def test_api_insights(client):
    """Test ML insights telemetry endpoint."""
    resp = client.get("/api/insights")
    assert resp.status_code == 200
    data = resp.json()
    assert "model_status" in data


def test_predict_rejects_invalid_file_type(client):
    """Ensure non-image files are rejected with 400 Bad Request."""
    file_content = b"Not an image file, just plain text."
    files = {"file": ("test.txt", io.BytesIO(file_content), "text/plain")}
    resp = client.post("/api/predict", files=files)
    assert resp.status_code == 400
    data = resp.json()
    assert data["detail"]["error"] == "InvalidFileType"


def test_predict_rejects_empty_file(client):
    """Ensure empty files are rejected with 400 Bad Request."""
    files = {"file": ("empty.png", io.BytesIO(b""), "image/png")}
    resp = client.post("/api/predict", files=files)
    assert resp.status_code == 400
    data = resp.json()
    assert data["detail"]["error"] == "EmptyFile"


def test_predict_endpoint_flow(client):
    """Test full predict flow with valid image."""
    # Generate simple test image
    img = Image.new("RGB", (100, 100), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    files = {"file": ("test_bottle.png", buf, "image/png")}
    resp = client.post("/api/predict", files=files)

    service = ModelService.get_instance()
    if not service.is_ready():
        # If model is not trained yet, expect 503
        assert resp.status_code == 503
        assert resp.json()["detail"]["error"] == "ModelNotTrained"
    else:
        # If model is trained, expect 200 and full schema
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("Likely Normal", "Potential Anomaly")
        assert isinstance(data["anomaly_score"], float)
        assert isinstance(data["threshold"], float)
        assert "original_image_base64" in data
        assert "heatmap_base64" in data
        assert "overlay_base64" in data
        # Threshold decision logic verification
        if data["anomaly_score"] >= data["threshold"]:
            assert data["is_anomaly"] is True
            assert data["status"] == "Potential Anomaly"
        else:
            assert data["is_anomaly"] is False
            assert data["status"] == "Likely Normal"


def test_dataset_samples_endpoint(client):
    """Test dataset samples retrieval endpoint."""
    resp = client.get("/api/dataset/samples?limit_per_category=2")
    assert resp.status_code == 200
    data = resp.json()
    assert "samples" in data
    assert "category" in data
    assert "available" in data
    if data["available"] and len(data["samples"]) > 0:
        sample = data["samples"][0]
        assert "filename" in sample
        assert "defect_type" in sample
        assert "image_url" in sample


def test_dataset_sample_direct_test(client):
    """Test direct inference on an official dataset sample."""
    resp = client.post("/api/dataset/test-sample", json={"defect_type": "good", "filename": "000.png"})
    if resp.status_code == 200:
        data = resp.json()
        assert data["true_label"] == "Normal"
        assert data["predicted_label"] == "Normal"
        assert data["is_correct"] is True
        assert "anomaly_score" in data
        assert "original_image_base64" in data
