"""Training script for DefectLens AI PatchCore anomaly detector.

Trains on normal (defect-free) surface images, constructs the patch coreset memory bank,
and calibrates the anomaly threshold strictly on the normal validation partition.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import MVTecDataset
from src.models.anomaly_detector import PatchCoreDetector


def set_seed(seed: int = 42) -> None:
    """Set random seed across all libraries for deterministic reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(
    category: str = "bottle",
    data_dir: str = "data/mvtec",
    output_dir: str = "models",
    config_path: Optional[str] = "configs/config.yaml",
    coreset_ratio: Optional[float] = None,
    batch_size: int = 16,
    seed: int = 42,
    smoke_test: bool = False,
    device: str = "auto",
) -> str:
    """Run PatchCore training pipeline.

    Args:
        category: MVTec AD category name
        data_dir: Path to root data directory
        output_dir: Path to save model checkpoints
        config_path: Optional path to config.yaml
        coreset_ratio: Optional override for coreset sampling ratio
        batch_size: DataLoader batch size
        seed: Random seed
        smoke_test: Flag to execute quick verification run
        device: Device string ('auto', 'cpu', 'mps', 'cuda')

    Returns:
        Path to saved model checkpoint
    """
    start_time = time.time()
    set_seed(seed)

    cfg = {}
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}

    category = category or cfg.get("dataset", {}).get("category", "bottle")
    data_dir = data_dir or cfg.get("dataset", {}).get("root_dir", "data/mvtec")
    output_dir = output_dir or cfg.get("training", {}).get("save_dir", "models")
    image_size = cfg.get("dataset", {}).get("image_size", 224)
    val_split = cfg.get("dataset", {}).get("val_split", 0.15)
    backbone = cfg.get("model", {}).get("backbone", "resnet18")
    layers = tuple(cfg.get("model", {}).get("layers", ["layer2", "layer3"]))

    if coreset_ratio is None:
        coreset_ratio = cfg.get("model", {}).get("coreset_sampling_ratio", 0.1)

    if smoke_test:
        print("[!] Running in SMOKE TEST mode (fast verification).")
        coreset_ratio = min(coreset_ratio, 0.05)

    category_dir = os.path.join(data_dir, category)
    train_dir = os.path.join(category_dir, "train", "good")

    if not os.path.exists(train_dir):
        raise FileNotFoundError(
            f"Training data not found at {train_dir}. Please run dataset preparation first:\n"
            f"  python src/data/prepare_dataset.py --category {category} --download"
        )

    print(f"[*] Loading training and validation sets for category: '{category}'...")
    train_dataset = MVTecDataset(
        category_dir=category_dir,
        split="train",
        image_size=image_size,
        val_ratio=val_split,
        seed=seed,
    )

    val_dataset = MVTecDataset(
        category_dir=category_dir,
        split="val",
        image_size=image_size,
        val_ratio=val_split,
        seed=seed,
    )

    print(f"[+] Loaded {len(train_dataset)} normal train images and {len(val_dataset)} normal val images.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    print(f"[*] Initializing PatchCoreDetector (backbone: {backbone}, device: {device})...")
    detector = PatchCoreDetector(
        backbone=backbone,
        pretrained=True,
        layers=layers,
        coreset_sampling_ratio=coreset_ratio,
        device=device,
    )

    # Fit memory bank on normal training samples
    max_patches = 80 if smoke_test else 150
    detector.fit(train_loader, max_patches_per_image=max_patches, verbose=True)

    # Calibrate threshold on held-out normal validation set
    print("[*] Calibrating decision threshold on normal validation set...")
    threshold = detector.calibrate_threshold(val_loader, percentile=98.0, margin=0.05)
    print(f"[+] Calibrated Anomaly Threshold: {threshold:.4f} (from {len(val_dataset)} validation samples)")

    # Save model
    os.makedirs(output_dir, exist_ok=True)
    model_filename = f"patchcore_{category}.pt"
    model_path = os.path.join(output_dir, model_filename)

    elapsed_time = round(time.time() - start_time, 2)
    training_summary = {
        "model_name": "DefectLens AI PatchCore",
        "category": category,
        "backbone": backbone,
        "layers": list(layers),
        "coreset_sampling_ratio": coreset_ratio,
        "memory_bank_size": detector.memory_bank.shape[0] if detector.memory_bank is not None else 0,
        "feature_dim": detector.memory_bank.shape[1] if detector.memory_bank is not None else 0,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "calibrated_threshold": threshold,
        "val_score_min": float(np.min(detector.val_scores)) if detector.val_scores else 0.0,
        "val_score_max": float(np.max(detector.val_scores)) if detector.val_scores else 0.0,
        "val_score_mean": float(np.mean(detector.val_scores)) if detector.val_scores else 0.0,
        "device_used": str(detector.device),
        "training_time_seconds": elapsed_time,
        "smoke_test": smoke_test,
        "seed": seed,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    detector.metadata = training_summary
    detector.save(model_path)

    summary_path = os.path.join(output_dir, f"training_summary_{category}.json")
    with open(summary_path, "w") as f:
        json.dump(training_summary, f, indent=2)

    print(f"[+] Model checkpoint: {model_path}")
    print(f"[+] Training summary: {summary_path}")
    print(f"[+] Finished training in {elapsed_time}s.")
    return model_path


def main():
    parser = argparse.ArgumentParser(description="Train DefectLens AI PatchCore model")
    parser.add_argument("--category", type=str, default="bottle", help="Dataset category")
    parser.add_argument("--data-dir", type=str, default="data/mvtec", help="Data directory")
    parser.add_argument("--output-dir", type=str, default="models", help="Model output directory")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config file")
    parser.add_argument("--coreset-ratio", type=float, default=None, help="Coreset sampling ratio")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast smoke test")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (auto, cpu, mps, cuda)")
    args = parser.parse_args()

    train(
        category=args.category,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        config_path=args.config,
        coreset_ratio=args.coreset_ratio,
        batch_size=args.batch_size,
        seed=args.seed,
        smoke_test=args.smoke_test,
        device=args.device,
    )


if __name__ == "__main__":
    main()
