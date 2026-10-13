"""Evaluation module for DefectLens AI PatchCore model.

Computes:
1. Image-level AUROC
2. Pixel-level AUROC (localization accuracy against ground-truth defect masks)
3. Precision, Recall, Specificity, and F1 score at the calibrated validation threshold
4. Confusion matrix (TP, FP, TN, FN)
5. Defect-type breakdown
6. Generates evaluation plots (ROC curves, confusion matrix, score distribution, sample heatmaps)
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    precision_recall_fscore_support,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.dataset import MVTecDataset
from src.models.anomaly_detector import PatchCoreDetector


def evaluate(
    model_path: str = "models/patchcore_bottle.pt",
    category: str = "bottle",
    data_dir: str = "data/mvtec",
    output_dir: str = "results",
    config_path: str = "configs/config.yaml",
    device: str = "auto",
) -> dict:
    """Run comprehensive evaluation on the official test set.

    Args:
        model_path: Path to trained model checkpoint
        category: MVTec category
        data_dir: Dataset root
        output_dir: Output directory for metrics and plots
        config_path: Path to config.yaml
        device: Device string

    Returns:
        Dictionary of computed metrics
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at {model_path}. Please train the model first:\n"
            f"  python src/train.py --category {category}"
        )

    category_dir = os.path.join(data_dir, category)
    test_dir = os.path.join(category_dir, "test")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory not found at {test_dir}")

    # Load configuration
    cfg = {}
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}

    image_size = cfg.get("dataset", {}).get("image_size", 224)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print(f"[*] Loading model from {model_path}...")
    detector = PatchCoreDetector.load(model_path, device=device)
    threshold = detector.threshold
    print(f"[+] Loaded model. Calibrated threshold: {threshold:.4f}")

    print(f"[*] Loading test dataset from {category_dir}...")
    test_dataset = MVTecDataset(
        category_dir=category_dir,
        split="test",
        image_size=image_size,
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    print(f"[+] Found {len(test_dataset)} test samples.")

    # Storage for metrics
    image_labels = []
    image_scores = []
    image_preds = []
    defect_types = []
    sample_vis_data = []

    # Pixel evaluation storage
    pixel_labels_flat = []
    pixel_scores_flat = []

    print("[*] Running inference across all test samples...")
    for idx, (img_tensor, label, mask_tensor, img_path, defect_type) in enumerate(test_loader):
        label_int = int(label.item())
        defect_type_str = defect_type[0]
        image_path_str = img_path[0]

        score, pred, anomaly_map = detector.predict(img_tensor)

        image_labels.append(label_int)
        image_scores.append(score)
        image_preds.append(pred)
        defect_types.append(defect_type_str)

        # Ground truth mask for pixel-level evaluation
        gt_mask = mask_tensor.squeeze().cpu().numpy()
        pixel_labels_flat.extend(gt_mask.flatten().astype(int))
        pixel_scores_flat.extend(anomaly_map.flatten())

        # Save a few sample visualizations (up to 2 per defect type + 2 normal)
        matching_vis = [s for s in sample_vis_data if s["defect_type"] == defect_type_str]
        if len(matching_vis) < 2:
            # Denormalize image for visualization
            raw_img = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            raw_img = np.clip((raw_img * std + mean) * 255.0, 0, 255).astype(np.uint8)

            sample_vis_data.append({
                "index": idx,
                "path": image_path_str,
                "defect_type": defect_type_str,
                "true_label": label_int,
                "pred_label": pred,
                "score": score,
                "raw_img": raw_img,
                "gt_mask": gt_mask,
                "anomaly_map": anomaly_map,
            })

    image_labels = np.array(image_labels)
    image_scores = np.array(image_scores)
    image_preds = np.array(image_preds)

    # 1. Image-level AUROC
    try:
        image_auroc = float(roc_auc_score(image_labels, image_scores))
        fpr_img, tpr_img, _ = roc_curve(image_labels, image_scores)
    except Exception as e:
        print(f"[!] Warning: Could not compute Image AUROC: {e}")
        image_auroc = 0.0
        fpr_img, tpr_img = np.array([0, 1]), np.array([0, 1])

    # 2. Pixel-level AUROC
    try:
        pixel_labels_arr = np.array(pixel_labels_flat)
        pixel_scores_arr = np.array(pixel_scores_flat)
        # Subsample for faster AUROC calculation if millions of pixels
        if len(pixel_labels_arr) > 200000:
            sample_idx = np.random.choice(len(pixel_labels_arr), size=200000, replace=False)
            pixel_labels_arr = pixel_labels_arr[sample_idx]
            pixel_scores_arr = pixel_scores_arr[sample_idx]

        if np.sum(pixel_labels_arr) > 0 and np.sum(1 - pixel_labels_arr) > 0:
            pixel_auroc = float(roc_auc_score(pixel_labels_arr, pixel_scores_arr))
            fpr_pix, tpr_pix, _ = roc_curve(pixel_labels_arr, pixel_scores_arr)
        else:
            pixel_auroc = None
            fpr_pix, tpr_pix = None, None
    except Exception as e:
        print(f"[!] Warning: Could not compute Pixel AUROC: {e}")
        pixel_auroc = None
        fpr_pix, tpr_pix = None, None

    # 3. Precision, Recall, Specificity, F1 at calibrated threshold
    cm = confusion_matrix(image_labels, image_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    precision, recall, f1, _ = precision_recall_fscore_support(
        image_labels, image_preds, average="binary", zero_division=0
    )
    accuracy = float((tp + tn) / len(image_labels))
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # 4. Per defect breakdown
    defect_breakdown = {}
    for d_type in set(defect_types):
        indices = [i for i, d in enumerate(defect_types) if d == d_type]
        sub_labels = image_labels[indices]
        sub_preds = image_preds[indices]
        sub_scores = image_scores[indices]
        defect_breakdown[d_type] = {
            "total": len(indices),
            "correct": int(np.sum(sub_labels == sub_preds)),
            "accuracy": float(np.mean(sub_labels == sub_preds)),
            "mean_anomaly_score": float(np.mean(sub_scores)),
        }

    # Format metrics dictionary
    metrics = {
        "model_name": "DefectLens AI PatchCore",
        "category": category,
        "backbone": detector.backbone_name,
        "calibrated_threshold": float(threshold),
        "test_samples_total": len(image_labels),
        "test_normal_count": int(np.sum(image_labels == 0)),
        "test_anomaly_count": int(np.sum(image_labels == 1)),
        "image_auroc": round(image_auroc, 4),
        "pixel_auroc": round(pixel_auroc, 4) if pixel_auroc is not None else "N/A",
        "accuracy": round(accuracy, 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "specificity": round(specificity, 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "defect_breakdown": defect_breakdown,
        "plots": {
            "roc_curves": "results/plots/roc_curves.png",
            "confusion_matrix": "results/plots/confusion_matrix.png",
            "score_distribution": "results/plots/score_distribution.png",
            "sample_predictions": "results/plots/sample_predictions.png",
        },
    }

    # 5. Generate and save plots
    _plot_roc_curves(fpr_img, tpr_img, image_auroc, fpr_pix, tpr_pix, pixel_auroc, plots_dir)
    _plot_confusion_matrix(cm, plots_dir)
    _plot_score_distribution(image_labels, image_scores, threshold, plots_dir)
    _plot_sample_predictions(sample_vis_data, plots_dir)

    metrics_file = os.path.join(output_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 55)
    print("           DEFECTLENS AI EVALUATION REPORT        ")
    print("=" * 55)
    print(f" Category:               {category}")
    print(f" Image AUROC:            {image_auroc:.4f}")
    print(f" Pixel AUROC:            {metrics['pixel_auroc']}")
    print(f" Calibrated Threshold:   {threshold:.4f}")
    print(f" Precision:              {precision:.4f}")
    print(f" Recall:                 {recall:.4f}")
    print(f" F1 Score:               {f1:.4f}")
    print(f" Specificity:            {specificity:.4f}")
    print(f" Accuracy:               {accuracy:.4f}")
    print(f" Confusion Matrix:       TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print("=" * 55)
    print(f"[+] Metrics saved to: {metrics_file}")
    print(f"[+] Plots saved under: {plots_dir}")

    return metrics


def _plot_roc_curves(fpr_img, tpr_img, auroc_img, fpr_pix, tpr_pix, auroc_pix, plots_dir):
    plt.figure(figsize=(7, 6))
    plt.plot(fpr_img, tpr_img, color="#2563eb", lw=2.5, label=f"Image-Level ROC (AUC = {auroc_img:.3f})")
    if fpr_pix is not None and tpr_pix is not None and auroc_pix is not None:
        plt.plot(fpr_pix, tpr_pix, color="#059669", lw=2.5, linestyle="--", label=f"Pixel-Level ROC (AUC = {auroc_pix:.3f})")
    plt.plot([0, 1], [0, 1], color="#94a3b8", lw=1.5, linestyle=":", label="Random Chance (0.50)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate (Recall)", fontsize=11, fontweight="bold")
    plt.title("DefectLens AI: ROC Curves", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "roc_curves.png"), dpi=200)
    plt.close()


def _plot_confusion_matrix(cm, plots_dir):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Image-Level Confusion Matrix", fontsize=13, fontweight="bold")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Normal", "Anomaly"], fontsize=10)
    plt.yticks(tick_marks, ["Normal", "Anomaly"], fontsize=10)

    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            plt.text(
                j, i, format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14,
                fontweight="bold"
            )

    plt.ylabel("True Surface State", fontsize=11, fontweight="bold")
    plt.xlabel("Predicted Surface State", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "confusion_matrix.png"), dpi=200)
    plt.close()


def _plot_score_distribution(labels, scores, threshold, plots_dir):
    plt.figure(figsize=(8, 5))
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]

    plt.hist(normal_scores, bins=15, alpha=0.6, color="#059669", label=f"Normal ({len(normal_scores)})", density=True)
    plt.hist(anomaly_scores, bins=15, alpha=0.6, color="#dc2626", label=f"Anomaly ({len(anomaly_scores)})", density=True)
    plt.axvline(threshold, color="#1e293b", linestyle="--", linewidth=2, label=f"Threshold ({threshold:.3f})")

    plt.xlabel("Model Anomaly Score", fontsize=11, fontweight="bold")
    plt.ylabel("Density", fontsize=11, fontweight="bold")
    plt.title("Anomaly Score Distribution & Calibrated Threshold", fontsize=13, fontweight="bold")
    plt.legend(frameon=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "score_distribution.png"), dpi=200)
    plt.close()


def _plot_sample_predictions(samples, plots_dir):
    if not samples:
        return

    n_samples = min(len(samples), 5)
    fig, axes = plt.subplots(n_samples, 4, figsize=(14, 3 * n_samples))
    if n_samples == 1:
        axes = np.expand_dims(axes, 0)

    for i in range(n_samples):
        item = samples[i]
        axes[i, 0].imshow(item["raw_img"])
        axes[i, 0].set_title(f"Input: {item['defect_type']}\nTrue: {'Anomaly' if item['true_label'] else 'Normal'}", fontsize=9)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(item["gt_mask"], cmap="gray")
        axes[i, 1].set_title("Ground Truth Mask", fontsize=9)
        axes[i, 1].axis("off")

        im = axes[i, 2].imshow(item["anomaly_map"], cmap="jet")
        axes[i, 2].set_title(f"PatchCore Heatmap\nScore: {item['score']:.3f}", fontsize=9)
        axes[i, 2].axis("off")

        # Overlay heatmap on grayscale input
        gray_bg = np.mean(item["raw_img"], axis=-1) / 255.0
        axes[i, 3].imshow(gray_bg, cmap="gray")
        axes[i, 3].imshow(item["anomaly_map"], cmap="jet", alpha=0.55)
        pred_text = "Anomaly" if item["pred_label"] else "Normal"
        axes[i, 3].set_title(f"Overlay\nPred: {pred_text}", fontsize=9)
        axes[i, 3].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "sample_predictions.png"), dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate DefectLens AI PatchCore model")
    parser.add_argument("--model-path", type=str, default="models/patchcore_bottle.pt", help="Path to checkpoint")
    parser.add_argument("--category", type=str, default="bottle", help="Category name")
    parser.add_argument("--data-dir", type=str, default="data/mvtec", help="Data directory")
    parser.add_argument("--output-dir", type=str, default="results", help="Results directory")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config file")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (auto, cpu, mps, cuda)")
    args = parser.parse_args()

    evaluate(
        model_path=args.model_path,
        category=args.category,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        config_path=args.config,
        device=args.device,
    )


if __name__ == "__main__":
    main()
