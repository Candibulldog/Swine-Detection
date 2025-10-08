# src/transforms.py

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

IMG_SIZE = 640
DATASET_MEAN = [0.3615, 0.3564, 0.3543]
DATASET_STD = [0.3089, 0.3045, 0.3051]


# -----------------------------------------------------------------------------
# Albumentations Wrapper Class
# -----------------------------------------------------------------------------


class AlbumentationsTransform:
    """
    A wrapper class to streamline the use of Albumentations transforms.
    It is designed to ensure that the same transformation pipeline is applied
    consistently to both an image and its corresponding bounding boxes.
    """

    def __init__(self, transforms: list):
        """
        Initializes the transformation pipeline.

        Args:
            transforms (list): A list of Albumentations transformation objects.
        """
        self.transforms = A.Compose(
            transforms,
            bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["labels"],
                min_visibility=0.25,
                min_area=16,
            ),
        )

    def __call__(self, image, target):
        """
        Applies the defined transformation pipeline to an image and its target annotations.

        Args:
            image: The input image (e.g., a PIL Image).
            target (dict): A dictionary containing annotations like 'boxes' and 'labels'.

        Returns:
            A tuple containing the transformed image and its updated target dictionary.
        """
        transform_args = {
            "image": np.array(image),
            "bboxes": target.get("boxes", []),
            "labels": target.get("labels", []),
        }
        if isinstance(transform_args["bboxes"], torch.Tensor):
            transform_args["bboxes"] = transform_args["bboxes"].tolist()
        if isinstance(transform_args["labels"], torch.Tensor):
            transform_args["labels"] = transform_args["labels"].tolist()

        transformed = self.transforms(**transform_args)

        if "boxes" in target:
            if len(transformed["bboxes"]) > 0:
                target["boxes"] = torch.as_tensor(transformed["bboxes"], dtype=torch.float32)
            else:
                target["boxes"] = torch.empty((0, 4), dtype=torch.float32)
            target["labels"] = torch.as_tensor(transformed["labels"], dtype=torch.int64)

        return transformed["image"], target


# -----------------------------------------------------------------------------
# Transform Builders
# -----------------------------------------------------------------------------


def _get_base_transforms() -> list:
    """
    Returns base transforms for validation/testing.
    Only performs resizing, padding, normalization, and tensor conversion.
    """
    return [
        A.LongestMaxSize(max_size=IMG_SIZE, p=1.0),
        A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=0, p=1.0),
        A.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
        ToTensorV2(),
    ]


def _get_unified_train_transforms() -> list:
    """
    Returns unified training transforms suitable for all image types.

    This is the DEFAULT strategy that works well across:
    - Cluster 0: Color/normal lighting
    - Cluster 1: Infrared/grayscale
    - Cluster 2: Night vision/low-light

    Key improvements over original:
    - Removed ToGray (conflicts with grayscale images)
    - Added ColorJitter (auto-skips grayscale images)
    - Reduced brightness/contrast extremes (0.3 vs 0.6)
    """
    return [
        # === Geometric & Spatial Augmentations ===
        A.LongestMaxSize(max_size=IMG_SIZE, p=1.0),
        A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=0, p=1.0),
        A.RandomSizedBBoxSafeCrop(height=IMG_SIZE, width=IMG_SIZE, erosion_rate=0.2, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            scale=(0.9, 1.1),
            rotate=(-15, 15),
            p=0.7,
        ),
        # === Color & Lighting Augmentations ===
        # ColorJitter automatically skips grayscale images
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
        # Reduced from brightness_limit=0.6, contrast_limit=0.5
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
        # === Robustness & Occlusion Augmentations ===
        A.GridDropout(ratio=0.5, p=0.5),
        # === Final Processing ===
        A.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
        ToTensorV2(),
    ]


def _get_color_cluster_transforms() -> list:
    """
    Optimized transforms for Cluster 0: Color/normal lighting images.

    Strategy:
    - Leverage color information with more aggressive ColorJitter
    - Standard geometric augmentations
    - Normal dropout for occlusion robustness
    """
    return [
        A.LongestMaxSize(max_size=IMG_SIZE, p=1.0),
        A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=0, p=1.0),
        A.RandomSizedBBoxSafeCrop(height=IMG_SIZE, width=IMG_SIZE, erosion_rate=0.2, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            scale=(0.9, 1.1),
            rotate=(-15, 15),
            p=0.7,
        ),
        # More aggressive color augmentation for color images
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.8),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
        A.GridDropout(ratio=0.5, p=0.5),
        A.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
        ToTensorV2(),
    ]


def _get_infrared_cluster_transforms() -> list:
    """
    Optimized transforms for Cluster 1: Infrared/grayscale images.

    Strategy:
    - No color augmentation (grayscale images)
    - Focus on contrast enhancement
    - Optional CLAHE for better edge detection
    """
    return [
        A.LongestMaxSize(max_size=IMG_SIZE, p=1.0),
        A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=0, p=1.0),
        A.RandomSizedBBoxSafeCrop(height=IMG_SIZE, width=IMG_SIZE, erosion_rate=0.2, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            scale=(0.9, 1.1),
            rotate=(-15, 15),
            p=0.7,
        ),
        # Focus on brightness/contrast for infrared
        A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.5, p=0.9),
        # Optional: Contrast Limited Adaptive Histogram Equalization
        # A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        A.GridDropout(ratio=0.5, p=0.5),
        A.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
        ToTensorV2(),
    ]


def _get_nightvision_cluster_transforms() -> list:
    """
    Optimized transforms for Cluster 2: Night vision/low-light images.

    Strategy:
    - Very conservative geometric augmentations
    - Minimal brightness/contrast changes (already dark)
    - Reduced dropout (images already have noise)
    - Optional noise augmentation to improve robustness
    """
    return [
        A.LongestMaxSize(max_size=IMG_SIZE, p=1.0),
        A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=0, p=1.0),
        # More conservative crop for low-light images
        A.RandomSizedBBoxSafeCrop(height=IMG_SIZE, width=IMG_SIZE, erosion_rate=0.2, p=0.4),
        A.HorizontalFlip(p=0.5),
        # Reduced geometric augmentation
        A.Affine(
            translate_percent={"x": (-0.03, 0.03), "y": (-0.03, 0.03)},
            scale=(0.95, 1.05),
            rotate=(-10, 10),
            p=0.6,
        ),
        # Very conservative brightness/contrast
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.2, p=0.6),
        # Optional: Light noise augmentation
        # A.GaussNoise(var_limit=(5.0, 15.0), p=0.2),
        # Reduced dropout for already noisy images
        A.GridDropout(ratio=0.3, p=0.3),
        A.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
        ToTensorV2(),
    ]


# -----------------------------------------------------------------------------
# Factory Function
# -----------------------------------------------------------------------------


def get_transform(train: bool, cluster_id: int = -1, use_cluster_aware: bool = False) -> AlbumentationsTransform:
    """
    Factory function that returns the appropriate data augmentation pipeline.

    Args:
        train (bool): If True, returns training augmentations; else validation transforms.
        cluster_id (int): Image cluster ID (0=color, 1=infrared, 2=night_vision, -1=unknown).
        use_cluster_aware (bool): If True, uses cluster-specific augmentations.

    Returns:
        AlbumentationsTransform: Configured transform pipeline.

    Usage:
        # Default: Unified transforms (recommended to start)
        train_transform = get_transform(train=True)
        val_transform = get_transform(train=False)

        # Cluster-aware: Optimized for specific image types
        train_transform = get_transform(train=True, cluster_id=0, use_cluster_aware=True)
    """
    # Validation/Testing: Always use base transforms
    if not train:
        return AlbumentationsTransform(_get_base_transforms())

    # Training: Choose strategy
    if use_cluster_aware and cluster_id >= 0:
        # Cluster-specific augmentation
        if cluster_id == 0:
            transforms = _get_color_cluster_transforms()
        elif cluster_id == 1:
            transforms = _get_infrared_cluster_transforms()
        elif cluster_id == 2:
            transforms = _get_nightvision_cluster_transforms()
        else:
            # Fallback to unified
            transforms = _get_unified_train_transforms()
    else:
        # Default: Unified transforms (works well for all clusters)
        transforms = _get_unified_train_transforms()

    return AlbumentationsTransform(transforms)


# -----------------------------------------------------------------------------
# Configuration Presets (for easy experimentation)
# -----------------------------------------------------------------------------


class AugmentationConfig:
    """
    Configuration presets for different augmentation strategies.
    Useful for quick experimentation and hyperparameter tuning.
    """

    # Default configuration
    UNIFIED = {"use_cluster_aware": False}

    # Cluster-aware configuration
    CLUSTER_AWARE = {"use_cluster_aware": True}

    # Conservative configuration (less augmentation)
    CONSERVATIVE = {
        "use_cluster_aware": False,
        # You can add more parameters here if needed
    }

    # Aggressive configuration (more augmentation)
    AGGRESSIVE = {
        "use_cluster_aware": True,
        # You can add more parameters here if needed
    }
