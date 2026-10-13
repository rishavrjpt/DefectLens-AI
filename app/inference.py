import base64
import io
import os
import threading
from typing import Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from PIL import Image
import numpy as np
import torch
import matplotlib.cm as cm

from src.data.dataset import get_transforms
from src.models.anomaly_detector import PatchCoreDetector


class ModelService:
    """Thread-safe model management and inference service."""

    _instance: Optional["ModelService"] = None
    _lock = threading.Lock()

    def __init__(self, model_dir: str = "models", default_category: str = "bottle"):
        self.model_dir = os.path.abspath(model_dir)
        self.category = default_category
        self.detector: Optional[PatchCoreDetector] = None
        self.model_path = os.path.join(self.model_dir, f"patchcore_{self.category}.pt")
        self.img_transform, _ = get_transforms(image_size=224)
        self.load_lock = threading.Lock()

        # Attempt initial load
        self.try_load_model()

    @classmethod
    def get_instance(cls, model_dir: str = "models", default_category: str = "bottle") -> "ModelService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(model_dir=model_dir, default_category=default_category)
        return cls._instance

    def try_load_model(self) -> bool:
        """Attempt to load or reload model checkpoint if it exists."""
        with self.load_lock:
            if os.path.exists(self.model_path):
                try:
                    print(f"[*] Loading model from {self.model_path}...")
                    self.detector = PatchCoreDetector.load(self.model_path)
                    print(f"[+] Loaded DefectLens AI model ({self.category}). Threshold: {self.detector.threshold}")
                    return True
                except Exception as e:
                    print(f"[!] Error loading model: {e}")
                    self.detector = None
                    return False
            else:
                self.detector = None
                return False

    def is_ready(self) -> bool:
        if self.detector is None:
            self.try_load_model()
        return self.detector is not None

    def get_info(self) -> dict:
        if not self.is_ready():
            return {
                "model_name": "DefectLens AI PatchCore",
                "category": self.category,
                "backbone": "resnet18",
                "calibrated_threshold": None,
                "layers": ["layer2", "layer3"],
                "coreset_sampling_ratio": 0.1,
                "memory_bank_size": 0,
                "is_trained": False,
                "evaluation_available": False,
            }

        return {
            "model_name": "DefectLens AI PatchCore",
            "category": self.category,
            "backbone": self.detector.backbone_name,
            "calibrated_threshold": float(self.detector.threshold),
            "layers": self.detector.layers,
            "coreset_sampling_ratio": self.detector.coreset_sampling_ratio,
            "memory_bank_size": int(self.detector.memory_bank.shape[0]) if self.detector.memory_bank is not None else 0,
            "is_trained": True,
            "evaluation_available": os.path.exists("results/metrics.json"),
            "training_summary": self.detector.metadata,
        }

    def predict_image(
        self, image_bytes: bytes
    ) -> Tuple[str, bool, float, float, float, float, str, str, str]:
        """Run anomaly detection and generate base64 visual responses.

        Args:
            image_bytes: Raw bytes of uploaded image file

        Returns:
            Tuple of:
              - status: "Likely Normal" or "Potential Anomaly"
              - is_anomaly: bool
              - score: float
              - threshold: float
              - score_ratio: float
              - area_percentage: float
              - orig_b64: base64 string of 224x224 input preview
              - heatmap_b64: base64 string of colorized anomaly map
              - overlay_b64: base64 string of blended overlay
        """
        if not self.is_ready():
            raise RuntimeError(
                "Model is not loaded. Please train the model using: python src/train.py"
            )

        # 1. Decode PIL image
        with Image.open(io.BytesIO(image_bytes)) as pil_img:
            rgb_img = pil_img.convert("RGB")
            resized_orig = rgb_img.resize((224, 224), Image.Resampling.BICUBIC)

        # 2. Transform and run inference
        img_tensor = self.img_transform(rgb_img)
        score, is_anomaly, anomaly_map = self.detector.predict(img_tensor)

        status = "Potential Anomaly" if is_anomaly else "Likely Normal"
        thresh = float(self.detector.threshold)
        score_ratio = round(score / thresh, 3) if thresh > 0 else 1.0

        # Estimate anomalous area (regions where heatmap exceeds 0.5 for anomalous surfaces)
        if is_anomaly:
            area_pct = round(float(np.mean(anomaly_map > 0.5)) * 100.0, 2)
        else:
            area_pct = 0.0

        # 3. Generate base64 images
        orig_b64 = self._pil_to_base64(resized_orig)

        # Colorize heatmap with 'jet' colormap
        try:
            import matplotlib
            cmap = matplotlib.colormaps["jet"]
        except Exception:
            from matplotlib import cm
            cmap = cm.get_cmap("jet")

        colored_map = cmap(anomaly_map)[:, :, :3]  # (H, W, 3) float [0, 1]
        heatmap_uint8 = (colored_map * 255.0).astype(np.uint8)
        heatmap_pil = Image.fromarray(heatmap_uint8)
        heatmap_b64 = self._pil_to_base64(heatmap_pil)

        # Generate blended overlay (60% original image + 40% heatmap)
        orig_np = np.array(resized_orig).astype(np.float32)
        blended = (0.55 * orig_np + 0.45 * (colored_map * 255.0)).clip(0, 255).astype(np.uint8)
        overlay_pil = Image.fromarray(blended)
        overlay_b64 = self._pil_to_base64(overlay_pil)

        return (
            status,
            bool(is_anomaly),
            round(float(score), 4),
            round(thresh, 4),
            score_ratio,
            area_pct,
            orig_b64,
            heatmap_b64,
            overlay_b64,
        )

    def _pil_to_base64(self, img: Image.Image, fmt: str = "PNG") -> str:
        buffer = io.BytesIO()
        img.save(buffer, format=fmt)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"
