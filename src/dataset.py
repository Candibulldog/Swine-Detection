# src/dataset.py - Cluster-Aware Version

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.transforms import get_transform


class PigDataset(Dataset):
    """
    Final cluster-aware dataset that correctly uses the transform factory.
    """

    def __init__(
        self,
        data_root: Path,
        frame_ids: Optional[List[int]],
        is_train: bool = True,
        annotations_df: Optional[pd.DataFrame] = None,
        cluster_dict: Optional[Dict[int, int]] = None,
        use_cluster_aware_aug: bool = False,
    ):
        self.data_root = Path(data_root)
        self.is_train = is_train
        self.cluster_dict = cluster_dict if cluster_dict else {}

        data_dir_name = "train" if self.is_train else "test"
        self.image_dir = self.data_root / data_dir_name / "img"

        # ✨ 1. 在初始化時，就創建好所有需要的 transform pipelines ✨
        if self.is_train:
            self.transforms = {}  # 創建一個字典來存放它們
            if use_cluster_aware_aug:
                if not self.cluster_dict:
                    raise ValueError("Cluster dict must be provided for cluster-aware training.")
                # 為每個 cluster 創建專用的 pipeline
                print("INFO: Initializing Dataset with CLUSTER-AWARE transforms.")
                self.transforms[0] = get_transform(train=True, cluster_id=0, use_cluster_aware=True)
                self.transforms[1] = get_transform(train=True, cluster_id=1, use_cluster_aware=True)
                self.transforms[2] = get_transform(train=True, cluster_id=2, use_cluster_aware=True)
            else:
                # 創建一個通用的 pipeline
                print("INFO: Initializing Dataset with UNIFIED transform.")
                self.transforms["unified"] = get_transform(train=True, use_cluster_aware=False)
        else:
            # 驗證/測試集只有一種 pipeline
            self.transforms = {"val": get_transform(train=False)}

        # ✨ (剩下的 __init__ 邏輯與我們之前修復 bug 時的版本完全一樣) ✨
        if is_train:
            # --- Training Mode ---
            if annotations_df is None:
                raise ValueError("annotations_df must be provided.")
            if frame_ids is None:
                raise ValueError("frame_ids must be provided.")
            self.frame_ids = frame_ids
            self.annotations = annotations_df[annotations_df["frame"].isin(self.frame_ids)].copy()
            self.annotations["x1"] = self.annotations["bb_left"]
            self.annotations["y1"] = self.annotations["bb_top"]
            self.annotations["x2"] = self.annotations["bb_left"] + self.annotations["bb_width"]
            self.annotations["y2"] = self.annotations["bb_top"] + self.annotations["bb_height"]
            self.frame_annotations = self.annotations.groupby("frame")
        else:
            # --- Test Mode ---
            if frame_ids is None:
                self.frame_ids = sorted([int(p.stem) for p in self.image_dir.glob("*.jpg") if p.stem.isdigit()])
            else:
                self.frame_ids = frame_ids
            self.annotations = pd.DataFrame(
                columns=["frame", "bb_left", "bb_top", "bb_width", "bb_height", "x1", "y1", "x2", "y2"]
            )
            self.frame_annotations = self.annotations.groupby("frame")

    def __len__(self):
        return len(self.frame_ids)

    def _get_image_path(self, frame_id: int) -> Path:
        filename = f"{frame_id:08d}.jpg"
        return self.image_dir / filename

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
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

                valid_indices = areas > 0
                boxes = boxes[valid_indices]
                labels = labels[valid_indices]
                areas = areas[valid_indices]

                target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
                target["labels"] = labels
                target["area"] = torch.as_tensor(areas, dtype=torch.float32)
                target["iscrowd"] = torch.zeros(len(boxes), dtype=torch.int64)
            else:
                # No annotations for this frame (should be rare after filtering)
                target["boxes"] = torch.empty((0, 4), dtype=torch.float32)
                target["labels"] = torch.empty((0,), dtype=torch.int64)
                target["area"] = torch.empty((0,), dtype=torch.float32)
                target["iscrowd"] = torch.empty((0,), dtype=torch.int64)

        # 🆕 Apply cluster-aware transforms if enabled
        transform_pipeline = None
        if self.is_train:
            if 0 in self.transforms:  # 檢查是否是 Cluster-Aware 模式
                cluster_id = self.cluster_dict.get(frame_id, 0)  # 默認 fallback 到 cluster 0
                transform_pipeline = self.transforms[cluster_id]
            else:  # Unified 模式
                transform_pipeline = self.transforms["unified"]
        else:  # Validation/Test 模式
            transform_pipeline = self.transforms["val"]

        image, target = transform_pipeline(image, target)

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
