# src/dataset.py

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class PigDataset(Dataset):
    """
    用於載入豬隻偵測資料集的 PyTorch Dataset。

    根據模式 (train/val 或 test)，處理影像和對應的標註。
    """

    def __init__(self, root_dir: Path, frame_ids: Optional[List[int]], is_train: bool = True, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.is_train = is_train

        # 設定資料路徑
        data_dir_name = "train" if self.is_train else "test"
        self.image_dir = self.root_dir / data_dir_name / "img"

        if self.is_train:
            # 訓練/驗證模式：讀取標註並按 frame 分組，以加速後續查找
            annotations_path = self.root_dir / data_dir_name / "gt.txt"
            all_annotations = pd.read_csv(annotations_path, header=None, names=["frame", "x", "y", "w", "h"])

            # 只保留當前資料集劃分 (train/val) 所需的標註
            subset_annotations = all_annotations[all_annotations["frame"].isin(frame_ids)].copy()

            # 性能優化：將 DataFrame 按 frame 分組，方便快速查找
            self.annotations_map = {
                frame: group.to_numpy()[:, 1:] for frame, group in subset_annotations.groupby("frame")
            }
            self.image_frames = sorted(list(self.annotations_map.keys()))
        else:
            # 測試模式：掃描影像資料夾以獲取所有 frames
            self.annotations_map = {}
            self.image_frames = sorted([int(p.stem) for p in self.image_dir.glob("*.jpg") if p.stem.isdigit()])

        mode_str = "Train/Val" if self.is_train else "Test"
        print(f"Dataset in '{mode_str}' mode initialized with {len(self.image_frames)} images.")

    def __len__(self) -> int:
        """返回資料集中的影像總數。"""
        return len(self.image_frames)

    def __getitem__(self, idx: int):
        """
        根據索引獲取一個樣本，包含影像和其對應的標註。

        返回:
            - image (torch.Tensor): 經過轉換的影像張量。
            - target (dict): 包含標註資訊的字典。
        """
        frame_id = self.image_frames[idx]
        image_path = self.image_dir / f"{frame_id:08d}.jpg"
        image = Image.open(image_path).convert("RGB")

        target = {"image_id": torch.tensor([frame_id], dtype=torch.int64)}

        if self.is_train and frame_id in self.annotations_map:
            # 從預先處理好的 map 中高效獲取標註
            records = self.annotations_map[frame_id]

            # 使用向量化操作，快速計算 boxes 和 areas
            x1 = records[:, 0]
            y1 = records[:, 1]
            w = records[:, 2]
            h = records[:, 3]
            x2 = x1 + w
            y2 = y1 + h

            boxes = torch.as_tensor(np.stack([x1, y1, x2, y2], axis=1), dtype=torch.float32)
            areas = torch.as_tensor(w * h, dtype=torch.float32)

            num_boxes = boxes.shape[0]
            target["boxes"] = boxes
            target["labels"] = torch.ones((num_boxes,), dtype=torch.int64)  # Class 1 for 'pig'
            target["area"] = areas
            target["iscrowd"] = torch.zeros((num_boxes,), dtype=torch.int64)
        elif self.is_train:
            # 處理那些在 train_frames 中但可能沒有任何有效標註的圖片
            target["boxes"] = torch.empty((0, 4), dtype=torch.float32)
            target["labels"] = torch.empty((0,), dtype=torch.int64)
            target["area"] = torch.empty((0,), dtype=torch.float32)
            target["iscrowd"] = torch.empty((0,), dtype=torch.int64)

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target
