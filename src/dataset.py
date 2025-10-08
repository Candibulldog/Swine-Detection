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
        is_train: bool,
        annotations_df: Optional[pd.DataFrame] = None,
        cluster_dict: Optional[Dict[int, int]] = None,
        use_cluster_aware_aug: bool = False,
    ):
        self.data_root = Path(data_root)
        self.is_train = is_train
        self.cluster_dict = cluster_dict if cluster_dict else {}

        # 根據是否提供標註來決定是 'train'/'val' 模式還是 'test' 模式
        # 驗證集的圖片也在 'train/img' 下，所以這個邏輯是正確的
        data_dir_name = "test" if annotations_df is None else "train"
        self.image_dir = self.data_root / data_dir_name / "img"

        # --- 核心修正: 標註加載邏輯 ---
        # 只要提供了 annotations_df (無論是訓練集還是驗證集)，就應該加載標註
        if annotations_df is not None:
            if frame_ids is None:
                raise ValueError("frame_ids must be provided when annotations_df is given.")
            self.frame_ids = frame_ids
            # 過濾出當前 split (train/val) 需要的標註
            self.annotations = annotations_df[annotations_df["frame"].isin(self.frame_ids)].copy()
            # 計算 x2, y2 坐標
            self.annotations["x1"] = self.annotations["bb_left"]
            self.annotations["y1"] = self.annotations["bb_top"]
            self.annotations["x2"] = self.annotations["bb_left"] + self.annotations["bb_width"]
            self.annotations["y2"] = self.annotations["bb_top"] + self.annotations["bb_height"]
            self.frame_annotations = self.annotations.groupby("frame")
        else:
            # 這是給 predict.py 使用的真正 Test 模式，沒有任何標註
            if frame_ids is None:
                self.frame_ids = sorted([int(p.stem) for p in self.image_dir.glob("*.jpg") if p.stem.isdigit()])
            else:
                self.frame_ids = frame_ids
            # 創建一個空的 annotations DataFrame
            self.annotations = pd.DataFrame(columns=["frame", "x1", "y1", "x2", "y2"])
            self.frame_annotations = self.annotations.groupby("frame")

        # --- Transform 初始化邏輯 (不變) ---
        # 這個邏輯由 self.is_train 正確控制，決定了是否應用數據增強
        if self.is_train:
            self.transforms = {}
            if use_cluster_aware_aug and self.cluster_dict:
                print("INFO: Initializing Dataset with CLUSTER-AWARE transforms.")
                self.transforms[0] = get_transform(train=True, cluster_id=0, use_cluster_aware=True)
                self.transforms[1] = get_transform(train=True, cluster_id=1, use_cluster_aware=True)
                self.transforms[2] = get_transform(train=True, cluster_id=2, use_cluster_aware=True)
            else:
                print("INFO: Initializing Dataset with UNIFIED transform.")
                self.transforms["unified"] = get_transform(train=True, use_cluster_aware=False)
        else:
            # 驗證/測試集只有一種 pipeline (Resize, Pad, Normalize, ToTensor)
            print("INFO: Initializing Dataset with VALIDATION/TEST transform.")
            self.transforms = {"val": get_transform(train=False)}

    def __len__(self):
        return len(self.frame_ids)

    def _get_image_path(self, frame_id: int) -> Path:
        filename = f"{frame_id:08d}.jpg"
        return self.image_dir / filename

    def __getitem__(self, idx: int):
        frame_id = self.frame_ids[idx]
        img_path = self._get_image_path(frame_id)
        image = Image.open(img_path).convert("RGB")

        target = {"image_id": torch.tensor([frame_id])}

        # --- 核心修正: Target 字典填充 ---
        # 移除 is_train 判斷，確保驗證集也能拿到含 'boxes' 的 target 字典
        if frame_id in self.frame_annotations.groups:
            frame_data = self.frame_annotations.get_group(frame_id)
            boxes = frame_data[["x1", "y1", "x2", "y2"]].values
            labels = torch.ones(len(boxes), dtype=torch.int64)
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

            # 過濾掉無效的 boxes
            valid_indices = areas > 0
            boxes = boxes[valid_indices]
            labels = labels[valid_indices]
            areas = areas[valid_indices]

            target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
            target["labels"] = labels
            target["area"] = torch.as_tensor(areas, dtype=torch.float32)
            target["iscrowd"] = torch.zeros(len(boxes), dtype=torch.int64)
        else:
            # 對於沒有標註的圖片 (train 或 val)，都提供空的 tensors
            target["boxes"] = torch.empty((0, 4), dtype=torch.float32)
            target["labels"] = torch.empty((0,), dtype=torch.int64)
            target["area"] = torch.empty((0,), dtype=torch.float32)
            target["iscrowd"] = torch.empty((0,), dtype=torch.int64)

        # --- Transform 應用邏輯 (不變) ---
        # 這裡的 is_train 判斷是正確且必須的，它決定了數據是否被增強
        transform_pipeline = None
        if self.is_train:
            if 0 in self.transforms:  # 檢查是否是 Cluster-Aware 模式
                cluster_id = self.cluster_dict.get(frame_id, 0)
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
