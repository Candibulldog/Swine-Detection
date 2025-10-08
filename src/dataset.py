# src/dataset.py - Cluster-Aware Version

from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class PigDataset(Dataset):
    """
    Cluster-aware dataset for pig detection.
    """

    def __init__(
        self,
        annotations_df: pd.DataFrame,
        data_root: Path,
        frame_ids: list,
        is_train: bool = True,
        transforms=None,
        cluster_dict: Optional[dict] = None,
        use_cluster_aware_aug: bool = True,
    ):
        """
        Args:
            data_root: Root directory containing train/test folders
            frame_ids: List of frame IDs to include in this dataset
            is_train: Whether this is training or test dataset
            transforms: Transformation function/object
            cluster_dict: Dictionary mapping frame_id -> cluster_id
            use_cluster_aware_aug: Whether to use cluster-specific augmentation
        """
        self.data_root = Path(data_root)
        self.frame_ids = frame_ids
        self.is_train = is_train
        self.transforms = transforms
        self.cluster_dict = cluster_dict if cluster_dict else {}
        self.use_cluster_aware_aug = use_cluster_aware_aug

        # Load ground truth if training
        if is_train:
            self.annotations = annotations_df[annotations_df["frame"].isin(frame_ids)].copy()

            # Convert to (x1, y1, x2, y2) format
            self.annotations["x1"] = self.annotations["bb_left"]
            self.annotations["y1"] = self.annotations["bb_top"]
            self.annotations["x2"] = self.annotations["bb_left"] + self.annotations["bb_width"]
            self.annotations["y2"] = self.annotations["bb_top"] + self.annotations["bb_height"]

            # Group by frame for efficient lookup
            self.frame_annotations = self.annotations.groupby("frame")

    def __len__(self):
        return len(self.frame_ids)

    def _get_image_path(self, frame_id: int) -> Path:
        """Get the path to an image file."""
        if self.is_train:
            return self.data_root / "train" / "img" / f"{frame_id}.jpg"
        else:
            return self.data_root / "test" / "img" / f"{frame_id}.jpg"

    def __getitem__(self, idx: int):
        frame_id = self.frame_ids[idx]
        img_path = self._get_image_path(frame_id)

        # Load image
        image = Image.open(img_path).convert("RGB")

        target = {"image_id": torch.tensor([frame_id])}

        if self.is_train:
            # Get annotations for this frame
            if frame_id in self.frame_annotations.groups:
                frame_data = self.frame_annotations.get_group(frame_id)
                boxes = frame_data[["x1", "y1", "x2", "y2"]].values
                labels = torch.ones(len(boxes), dtype=torch.int64)  # All class 1 (pig)

                target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
                target["labels"] = labels
            else:
                # No annotations for this frame (should be rare after filtering)
                target["boxes"] = torch.empty((0, 4), dtype=torch.float32)
                target["labels"] = torch.empty((0,), dtype=torch.int64)

        # 🆕 Apply cluster-aware transforms if enabled
        if self.transforms:
            if self.use_cluster_aware_aug and hasattr(self.transforms, "__call__"):
                cluster_id = self.cluster_dict.get(frame_id, -1)
                # Check if transform supports cluster_id parameter
                try:
                    image, target = self.transforms(image, target, cluster_id=cluster_id)
                except TypeError:
                    # Fallback to regular transforms
                    image, target = self.transforms(image, target)
            else:
                image, target = self.transforms(image, target)

        return image, target


def create_cluster_dict(data_root: Path) -> dict:
    """
    Load cluster information from image_clusters.txt.

    Returns:
        Dictionary mapping frame_id -> cluster_id
    """
    cluster_path = data_root / "train" / "image_clusters.txt"
    if not cluster_path.exists():
        print(f"⚠️  Cluster file not found at {cluster_path}")
        return {}

    try:
        cluster_df = pd.read_csv(cluster_path, sep="\t")
        cluster_dict = dict(zip(cluster_df["frame_id"], cluster_df["cluster_id"]))

        # Print cluster statistics
        from collections import Counter

        cluster_counts = Counter(cluster_dict.values())
        print("📊 Cluster distribution:")
        for cluster_id in sorted(cluster_counts.keys()):
            count = cluster_counts[cluster_id]
            percentage = count / len(cluster_dict) * 100
            print(f"   Cluster {cluster_id}: {count} images ({percentage:.1f}%)")

        return cluster_dict
    except Exception as e:
        print(f"❌ Error loading cluster file: {e}")
        return {}
