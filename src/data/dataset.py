"""PyTorch Dataset implementation for MVTec AD."""

from typing import Callable, List, Optional, Tuple
import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


# Standard ImageNet normalization for pretrained CNN backbones
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(image_size: int = 224) -> Tuple[Callable, Callable]:
    """Return standard image and ground-truth mask transforms.

    Args:
        image_size: Target square image dimension (default: 224 for ResNet)

    Returns:
        Tuple of (image_transform, mask_transform)
    """
    img_transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])

    return img_transform, mask_transform


class MVTecDataset(Dataset):
    """Dataset loader for MVTec Anomaly Detection category benchmarks."""

    def __init__(
        self,
        category_dir: str,
        split: str = "train",
        image_size: int = 224,
        val_ratio: float = 0.15,
        seed: int = 42,
    ):
        """Initialize MVTec dataset.

        Args:
            category_dir: Directory containing train, test, ground_truth (e.g. data/mvtec/bottle)
            split: 'train', 'val', or 'test'
            image_size: Target spatial size (224)
            val_ratio: Fraction of train/good held out for threshold calibration
            seed: Random seed for reproducible train/val partition
        """
        self.category_dir = os.path.abspath(category_dir)
        self.split = split
        self.image_size = image_size
        self.img_transform, self.mask_transform = get_transforms(image_size)

        self.samples: List[dict] = []
        self._load_samples(val_ratio, seed)

    def _load_samples(self, val_ratio: float, seed: int) -> None:
        train_good_dir = os.path.join(self.category_dir, "train", "good")
        test_dir = os.path.join(self.category_dir, "test")
        gt_dir = os.path.join(self.category_dir, "ground_truth")

        if self.split in ("train", "val"):
            if not os.path.exists(train_good_dir):
                raise FileNotFoundError(f"Training directory not found: {train_good_dir}")

            all_train_files = sorted(glob.glob(os.path.join(train_good_dir, "*.*")))
            valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
            all_train_files = [f for f in all_train_files if f.lower().endswith(valid_extensions)]

            # Deterministic train/val split using fixed random generator
            g = torch.Generator().manual_seed(seed)
            perm = torch.randperm(len(all_train_files), generator=g).tolist()
            val_size = int(len(all_train_files) * val_ratio)
            val_indices = set(perm[:val_size])

            for idx, img_path in enumerate(all_train_files):
                is_val = idx in val_indices
                if (self.split == "val" and is_val) or (self.split == "train" and not is_val):
                    self.samples.append({
                        "image_path": img_path,
                        "label": 0,  # 0: normal
                        "defect_type": "good",
                        "mask_path": None,
                    })

        elif self.split == "test":
            if not os.path.exists(test_dir):
                raise FileNotFoundError(f"Test directory not found: {test_dir}")

            subdirs = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
            valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

            for defect_type in subdirs:
                subdir_path = os.path.join(test_dir, defect_type)
                img_files = sorted(glob.glob(os.path.join(subdir_path, "*.*")))
                img_files = [f for f in img_files if f.lower().endswith(valid_extensions)]

                for img_path in img_files:
                    is_anomaly = defect_type != "good"
                    mask_path = None

                    if is_anomaly and os.path.exists(gt_dir):
                        # Locate corresponding mask file
                        base_name = os.path.splitext(os.path.basename(img_path))[0]
                        # Standard MVTec pattern: e.g. 000_mask.png
                        candidate_masks = [
                            os.path.join(gt_dir, defect_type, f"{base_name}_mask.png"),
                            os.path.join(gt_dir, defect_type, f"{base_name}.png"),
                        ]
                        for c in candidate_masks:
                            if os.path.exists(c):
                                mask_path = c
                                break

                    self.samples.append({
                        "image_path": img_path,
                        "label": 1 if is_anomaly else 0,
                        "defect_type": defect_type,
                        "mask_path": mask_path,
                    })

        else:
            raise ValueError(f"Unknown split: {self.split}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor, str, str]:
        item = self.samples[idx]
        img_path = item["image_path"]

        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img_tensor = self.img_transform(img)

        # Ground truth mask
        if item["mask_path"] is not None and os.path.exists(item["mask_path"]):
            with Image.open(item["mask_path"]) as mask_img:
                mask_img = mask_img.convert("L")
                mask_tensor = self.mask_transform(mask_img)
                mask_tensor = (mask_tensor > 0.5).float()
        else:
            # Normal or unannotated: zero mask
            mask_tensor = torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32)

        return img_tensor, item["label"], mask_tensor, img_path, item["defect_type"]
