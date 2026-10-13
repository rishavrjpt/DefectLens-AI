import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from PIL import Image
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.dataset import get_transforms
from src.models.anomaly_detector import PatchCoreDetector


def predict_single_image(
    image_path: str,
    model_path: str = "models/patchcore_bottle.pt",
    output_heatmap_path: Optional[str] = None,
    device: str = "auto",
) -> dict:
    """Run PatchCore anomaly inference on a single image.

    Args:
        image_path: Path to user query image
        model_path: Path to trained PatchCore checkpoint
        output_heatmap_path: Optional path to save visual overlay
        device: Compute device

    Returns:
        Dictionary with anomaly score, threshold, label, and metadata
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Please train a model first using:\n"
            f"  python src/train.py"
        )

    # 1. Load detector
    detector = PatchCoreDetector.load(model_path, device=device)

    # 2. Load and preprocess image
    with Image.open(image_path) as raw_img:
        rgb_img = raw_img.convert("RGB")
        orig_w, orig_h = rgb_img.size

    img_transform, _ = get_transforms(image_size=224)
    img_tensor = img_transform(rgb_img)

    # 3. Predict
    score, is_anomaly, anomaly_map = detector.predict(img_tensor)
    status_label = "Potential Anomaly" if is_anomaly else "Likely Normal"

    # Percentage of anomalous area based on heatmap threshold
    if is_anomaly:
        anomalous_pixel_ratio = float(np.mean(anomaly_map > 0.5))
    else:
        anomalous_pixel_ratio = 0.0

    result = {
        "image_path": image_path,
        "status": status_label,
        "is_anomaly": bool(is_anomaly),
        "anomaly_score": round(float(score), 4),
        "threshold": round(float(detector.threshold), 4),
        "score_to_threshold_ratio": round(float(score / detector.threshold), 3) if detector.threshold > 0 else 1.0,
        "anomalous_area_ratio": round(anomalous_pixel_ratio, 4),
        "disclaimer": (
            "Model score reflects patch feature distance to normal reference surfaces. "
            "It is not a calibrated probability or a certified industrial diagnosis."
        ),
    }

    # 4. Optional heatmap visualization
    if output_heatmap_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_heatmap_path)), exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(rgb_img.resize((224, 224)))
        axes[0].set_title("Input Image", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        im = axes[1].imshow(anomaly_map, cmap="jet", vmin=0.0, vmax=1.0)
        axes[1].set_title(f"Anomaly Heatmap\n(Score: {score:.3f})", fontsize=11, fontweight="bold")
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        # Grayscale overlay
        gray_img = np.array(rgb_img.convert("L").resize((224, 224))) / 255.0
        axes[2].imshow(gray_img, cmap="gray")
        axes[2].imshow(anomaly_map, cmap="jet", alpha=0.55)
        color = "#dc2626" if is_anomaly else "#059669"
        axes[2].set_title(f"Decision: {status_label}", fontsize=11, fontweight="bold", color=color)
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(output_heatmap_path, dpi=200)
        plt.close()
        result["heatmap_image_path"] = output_heatmap_path

    return result


def main():
    parser = argparse.ArgumentParser(description="Run DefectLens AI inference on an image")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model-path", type=str, default="models/patchcore_bottle.pt", help="Path to model checkpoint")
    parser.add_argument("--output-heatmap", type=str, default=None, help="Path to save output heatmap overlay")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (auto, cpu, mps, cuda)")
    args = parser.parse_args()

    result = predict_single_image(
        image_path=args.image,
        model_path=args.model_path,
        output_heatmap_path=args.output_heatmap,
        device=args.device,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
