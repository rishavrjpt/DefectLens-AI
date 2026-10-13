"""DefectLens AI: PatchCore Anomaly Detector implementation.

Based on Roth et al., 2022: 'Towards Total Recall in Industrial Anomaly Detection'.
Extracts multi-scale patch feature embeddings from a pretrained CNN backbone (ResNet-18),
subsamples a representative coreset memory bank from normal training samples,
and computes anomaly scores and localized heatmaps using nearest neighbor distances.
"""

from typing import Dict, List, Optional, Tuple, Union
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import numpy as np
from scipy.ndimage import gaussian_filter


class Hook:
    """Forward hook to capture intermediate activations."""
    def __init__(self):
        self.feature = None

    def __call__(self, module, inputs, output):
        self.feature = output


class PatchCoreDetector:
    """Industrial Surface Anomaly Detector based on the PatchCore framework."""

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        layers: Tuple[str, ...] = ("layer2", "layer3"),
        coreset_sampling_ratio: float = 0.1,
        n_neighbors: int = 3,
        device: Optional[str] = None,
    ):
        self.backbone_name = backbone
        self.pretrained = pretrained
        self.layers = list(layers)
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.n_neighbors = n_neighbors

        # Device selection: MPS > CUDA > CPU
        if device is None or device == "auto":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model = self._init_backbone()
        self.model.to(self.device)
        self.model.eval()

        # Register forward hooks on specified layers
        self.hooks = {}
        for layer_name in self.layers:
            layer = getattr(self.model, layer_name)
            hook = Hook()
            layer.register_forward_hook(hook)
            self.hooks[layer_name] = hook

        # Memory bank and calibrated threshold
        self.memory_bank: Optional[torch.Tensor] = None
        self.threshold: float = 0.5
        self.val_scores: List[float] = []
        self.patch_dim: Optional[int] = None
        self.metadata: Dict[str, Union[str, float, int]] = {}

    def _init_backbone(self) -> nn.Module:
        """Initialize pretrained CNN feature extractor without classification head."""
        if self.backbone_name == "resnet18":
            weights = ResNet18_Weights.DEFAULT if self.pretrained else None
            net = models.resnet18(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")

        # Freeze all backbone parameters
        for param in net.parameters():
            param.requires_grad = False
        return net

    def _extract_patch_embeddings(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Extract multi-scale patch embeddings for a batch of images.

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            Tuple of:
              - Patch embeddings of shape (B, H_patch * W_patch, D)
              - Spatial patch grid dimensions (H_patch, W_patch)
        """
        x = x.to(self.device)
        with torch.no_grad():
            _ = self.model(x)

        features = [self.hooks[name].feature for name in self.layers]
        # Target spatial size is the largest feature map (typically layer2)
        target_size = features[0].shape[-2:]  # (H2, W2)

        # Apply adaptive average pooling for local neighborhood aggregation
        agg_features = []
        for feat in features:
            # Local 3x3 context pooling
            pooled = F.avg_pool2d(feat, kernel_size=3, stride=1, padding=1)
            if pooled.shape[-2:] != target_size:
                pooled = F.interpolate(
                    pooled, size=target_size, mode="bilinear", align_corners=False
                )
            agg_features.append(pooled)

        # Concatenate along channel dimension: (B, C_total, H, W)
        concat_features = torch.cat(agg_features, dim=1)
        B, C, H, W = concat_features.shape

        # Reshape to (B, H*W, C)
        embeddings = concat_features.permute(0, 2, 3, 1).reshape(B, H * W, C)
        return embeddings, (H, W)

    def _greedy_coreset_subsampling(
        self, features: torch.Tensor, target_count: int
    ) -> torch.Tensor:
        """Minimax facility location greedy coreset subsampling.

        Selects an optimal representative subset of patch features from normal images
        to bound the maximum distance from any candidate patch to the selected memory bank.
        """
        n_samples = features.shape[0]
        if target_count >= n_samples:
            return features

        # Use CPU with float32 for deterministic coreset selection
        feats_cpu = features.cpu().float()

        # Optional Johnson-Lindenstrauss random projection for fast subsampling if dim > 128
        dim = feats_cpu.shape[1]
        if dim > 128 and n_samples > 20000:
            torch.manual_seed(42)
            proj = torch.randn(dim, 128) / (128**0.5)
            reduced = torch.matmul(feats_cpu, proj)
        else:
            reduced = feats_cpu

        # Initial point: sample closest to the mean center
        center = reduced.mean(dim=0, keepdim=True)
        init_idx = torch.argmin(torch.norm(reduced - center, dim=1)).item()

        selected_indices = [init_idx]
        min_distances = torch.norm(reduced - reduced[init_idx : init_idx + 1], dim=1)

        # Iteratively select point with maximum distance to current coreset
        for _ in range(1, target_count):
            farthest_idx = torch.argmax(min_distances).item()
            selected_indices.append(farthest_idx)
            new_dist = torch.norm(reduced - reduced[farthest_idx : farthest_idx + 1], dim=1)
            min_distances = torch.minimum(min_distances, new_dist)

        return features[selected_indices]

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        max_patches_per_image: int = 150,
        verbose: bool = True,
    ) -> None:
        """Train PatchCore by building the coreset memory bank from normal images.

        Args:
            train_loader: DataLoader returning normal training images
            max_patches_per_image: Max patch subsample per image before global coreset
            verbose: Print training progress
        """
        self.model.eval()
        all_embeddings = []

        if verbose:
            print(f"[*] Extracting normal patch features using device: {self.device}")

        total_images = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(train_loader):
                # Handle either tuple (images, labels) or images tensor
                images = batch[0] if isinstance(batch, (tuple, list)) else batch
                emb, _ = self._extract_patch_embeddings(images)
                B, N_patches, D = emb.shape
                total_images += B

                # Flatten batch into (B * N_patches, D)
                flat_emb = emb.reshape(-1, D)

                # Subsample locally if batch contains many patches
                if max_patches_per_image is not None and max_patches_per_image < N_patches:
                    indices = torch.randperm(flat_emb.shape[0])[: B * max_patches_per_image]
                    flat_emb = flat_emb[indices]

                all_embeddings.append(flat_emb.cpu())

        all_embeddings = torch.cat(all_embeddings, dim=0)
        total_extracted = all_embeddings.shape[0]
        self.patch_dim = all_embeddings.shape[1]

        target_coreset_size = max(
            50, int(total_extracted * self.coreset_sampling_ratio)
        )
        # Cap memory bank for lightweight laptop deployment (max 5000 patches)
        target_coreset_size = min(target_coreset_size, 5000)

        if verbose:
            print(
                f"[*] Collected {total_extracted} patches from {total_images} normal images. "
                f"Selecting coreset of {target_coreset_size} patches..."
            )

        self.memory_bank = self._greedy_coreset_subsampling(
            all_embeddings, target_coreset_size
        ).to(self.device)

        if verbose:
            print(f"[+] Memory bank constructed: shape {self.memory_bank.shape}")

    def calibrate_threshold(
        self,
        val_loader: torch.utils.data.DataLoader,
        percentile: float = 98.0,
        margin: float = 0.05,
    ) -> float:
        """Calibrate the decision threshold strictly using normal validation data.

        Does NOT touch the test set. Calibrated to allow only a small false alarm rate.

        Args:
            val_loader: DataLoader with normal validation images
            percentile: Percentile of normal validation scores to set threshold
            margin: Additional conservative margin above validation threshold

        Returns:
            Calibrated anomaly score threshold
        """
        scores = []
        with torch.no_grad():
            for batch in val_loader:
                images = batch[0] if isinstance(batch, (tuple, list)) else batch
                for img in images:
                    img = img.unsqueeze(0)
                    score, _, _ = self.predict(img)
                    scores.append(score)

        if not scores:
            self.threshold = 0.5
            return self.threshold

        self.val_scores = scores
        val_arr = np.array(scores)
        base_thresh = float(np.percentile(val_arr, percentile))
        std_val = float(np.std(val_arr)) if len(val_arr) > 1 else 0.05
        # Set threshold with a modest margin to prevent false alarms on unseen normals
        self.threshold = round(base_thresh + margin * std_val, 4)
        return self.threshold

    def predict(
        self,
        x: torch.Tensor,
        gaussian_sigma: float = 4.0,
    ) -> Tuple[float, int, np.ndarray]:
        """Run anomaly detection and localization on an input image tensor.

        Args:
            x: Input image tensor of shape (1, 3, H, W) or (3, H, W)
            gaussian_sigma: Sigma for spatial heatmap smoothing

        Returns:
            Tuple of:
              - anomaly_score: float (highest patch distance to memory bank)
              - prediction: int (0: Normal, 1: Anomaly)
              - anomaly_map: 2D numpy array normalized to [0, 1] of shape (H, W)
        """
        if self.memory_bank is None:
            raise RuntimeError("Model has not been trained yet. Call fit() first.")

        if x.ndim == 3:
            x = x.unsqueeze(0)

        orig_h, orig_w = x.shape[-2:]
        emb, (patch_h, patch_w) = self._extract_patch_embeddings(x)
        # emb: (1, patch_h * patch_w, D)
        query_patches = emb[0]  # (N_patches, D)

        # Compute pairwise Euclidean distances between query patches and memory bank
        # Distances: (N_patches, M)
        with torch.no_grad():
            dists = torch.cdist(query_patches, self.memory_bank, p=2.0)
            # Nearest neighbor distance for each patch
            min_dists, _ = torch.min(dists, dim=1)  # (N_patches,)

        patch_scores = min_dists.cpu().numpy()
        image_score = float(np.max(patch_scores))

        # Spatial anomaly heatmap: reshape to patch grid
        score_map = patch_scores.reshape(patch_h, patch_w)

        # Bilinearly interpolate to original image size
        score_map_tensor = (
            torch.tensor(score_map)
            .unsqueeze(0)
            .unsqueeze(0)
            .float()
        )
        interpolated = (
            F.interpolate(
                score_map_tensor,
                size=(orig_h, orig_w),
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .numpy()
        )

        # Apply Gaussian smoothing for smooth visualization
        smoothed = gaussian_filter(interpolated, sigma=gaussian_sigma)

        # Normalize heatmap to [0, 1]
        map_min, map_max = smoothed.min(), smoothed.max()
        if map_max - map_min > 1e-6:
            anomaly_map = (smoothed - map_min) / (map_max - map_min)
        else:
            anomaly_map = np.zeros_like(smoothed)

        is_anomaly = 1 if image_score >= self.threshold else 0
        return image_score, is_anomaly, anomaly_map

    def save(self, filepath: str) -> None:
        """Save model memory bank, threshold, and configuration to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        payload = {
            "backbone": self.backbone_name,
            "pretrained": self.pretrained,
            "layers": self.layers,
            "coreset_sampling_ratio": self.coreset_sampling_ratio,
            "n_neighbors": self.n_neighbors,
            "memory_bank": self.memory_bank.cpu() if self.memory_bank is not None else None,
            "threshold": self.threshold,
            "val_scores": self.val_scores,
            "metadata": self.metadata,
        }
        torch.save(payload, filepath)
        print(f"[+] Saved model checkpoint to: {filepath}")

    @classmethod
    def load(cls, filepath: str, device: Optional[str] = None) -> "PatchCoreDetector":
        """Load trained PatchCore model from checkpoint."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

        # Weights_only=False for legacy tensor dictionary or True if clean
        payload = torch.load(filepath, map_location="cpu", weights_only=False)
        detector = cls(
            backbone=payload.get("backbone", "resnet18"),
            pretrained=payload.get("pretrained", True),
            layers=tuple(payload.get("layers", ("layer2", "layer3"))),
            coreset_sampling_ratio=payload.get("coreset_sampling_ratio", 0.1),
            n_neighbors=payload.get("n_neighbors", 3),
            device=device,
        )
        if payload.get("memory_bank") is not None:
            detector.memory_bank = payload["memory_bank"].to(detector.device)
        detector.threshold = payload.get("threshold", 0.5)
        detector.val_scores = payload.get("val_scores", [])
        detector.metadata = payload.get("metadata", {})
        return detector
