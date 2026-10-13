"""Unit tests for dataset preparation and dataset loading."""

import os
import shutil
import tempfile
import pytest
import torch
from PIL import Image

from src.data.prepare_dataset import validate_dataset, create_synthetic_smoke_data
from src.data.dataset import MVTecDataset, get_transforms


@pytest.fixture
def temp_dataset_dir():
    """Create a temporary dataset folder with synthetic samples."""
    tmp_dir = tempfile.mkdtemp()
    category_dir = os.path.join(tmp_dir, "bottle")
    create_synthetic_smoke_data(category_dir)
    yield category_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_validate_dataset_structure(temp_dataset_dir):
    """Test validation script detects correct structure and counts."""
    summary = validate_dataset(temp_dataset_dir)
    assert summary["valid_structure"] is True
    assert summary["status"] == "verified"
    assert summary["train_good_count"] == 25
    assert summary["test_good_count"] == 10
    assert "broken_large" in summary["test_defect_counts"]
    assert "contamination" in summary["test_defect_counts"]
    assert summary["ground_truth_mask_count"] > 0
    assert len(summary["corrupted_files"]) == 0


def test_dataset_loader_splits(temp_dataset_dir):
    """Test MVTecDataset splits and data loading."""
    train_ds = MVTecDataset(temp_dataset_dir, split="train", val_ratio=0.2, seed=42)
    val_ds = MVTecDataset(temp_dataset_dir, split="val", val_ratio=0.2, seed=42)
    test_ds = MVTecDataset(temp_dataset_dir, split="test")

    # 25 total train good -> 20 train, 5 val with 0.2 ratio
    assert len(train_ds) == 20
    assert len(val_ds) == 5
    assert len(test_ds) == 10 + 8 + 8  # 10 good, 8 broken, 8 contamination

    # Verify tensor shapes
    img, label, mask, path, defect = train_ds[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 224, 224)
    assert label == 0
    assert mask.shape == (1, 224, 224)

    # Verify test anomaly sample has non-zero mask
    anomaly_samples = [test_ds[i] for i in range(len(test_ds)) if test_ds[i][1] == 1]
    assert len(anomaly_samples) > 0
    anom_img, anom_lbl, anom_mask, _, anom_defect = anomaly_samples[0]
    assert anom_lbl == 1
    assert anom_mask.sum().item() > 0


def test_invalid_dataset_path():
    """Test handling of non-existent dataset paths."""
    summary = validate_dataset("/tmp/non_existent_defectlens_path")
    assert summary["valid_structure"] is False
    assert "Missing train/good" in summary["status"]
