"""Unit tests for PatchCore model architecture and inference."""

import os
import tempfile
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.anomaly_detector import PatchCoreDetector


@pytest.fixture
def dummy_train_loader():
    """Create a dummy DataLoader with small random tensors simulating normal images."""
    torch.manual_seed(42)
    # 4 images of shape (3, 224, 224)
    x = torch.randn(4, 3, 224, 224)
    dataset = TensorDataset(x)
    return DataLoader(dataset, batch_size=2)


def test_detector_initialization():
    """Verify PatchCore backbone, hooks, and device configuration."""
    detector = PatchCoreDetector(backbone="resnet18", pretrained=False, device="cpu")
    assert detector.backbone_name == "resnet18"
    assert "layer2" in detector.hooks
    assert "layer3" in detector.hooks
    assert detector.device == torch.device("cpu")
    assert detector.memory_bank is None


def test_detector_fit_and_calibrate(dummy_train_loader):
    """Test model training and threshold calibration."""
    detector = PatchCoreDetector(
        backbone="resnet18",
        pretrained=False,
        coreset_sampling_ratio=0.1,
        device="cpu",
    )

    detector.fit(dummy_train_loader, max_patches_per_image=50, verbose=False)
    assert detector.memory_bank is not None
    assert detector.memory_bank.ndim == 2
    assert detector.memory_bank.shape[1] == 128 + 256  # layer2 (128) + layer3 (256)

    threshold = detector.calibrate_threshold(dummy_train_loader, percentile=95.0)
    assert isinstance(threshold, float)
    assert threshold > 0.0


def test_predict_and_heatmap_output(dummy_train_loader):
    """Test inference output shapes and ranges."""
    detector = PatchCoreDetector(
        backbone="resnet18",
        pretrained=False,
        coreset_sampling_ratio=0.1,
        device="cpu",
    )
    detector.fit(dummy_train_loader, max_patches_per_image=50, verbose=False)
    detector.calibrate_threshold(dummy_train_loader)

    # Test single query image
    query = torch.randn(1, 3, 224, 224)
    score, pred, heatmap = detector.predict(query)

    assert isinstance(score, float)
    assert pred in (0, 1)
    assert isinstance(heatmap, np.ndarray)
    assert heatmap.shape == (224, 224)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0


def test_save_and_load_persistence(dummy_train_loader):
    """Test saving checkpoint and restoring state."""
    detector = PatchCoreDetector(
        backbone="resnet18",
        pretrained=False,
        coreset_sampling_ratio=0.1,
        device="cpu",
    )
    detector.fit(dummy_train_loader, max_patches_per_image=50, verbose=False)
    detector.threshold = 1.42

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = os.path.join(tmp_dir, "test_model.pt")
        detector.save(save_path)
        assert os.path.exists(save_path)

        loaded = PatchCoreDetector.load(save_path, device="cpu")
        assert loaded.backbone_name == "resnet18"
        assert loaded.threshold == 1.42
        assert loaded.memory_bank.shape == detector.memory_bank.shape
