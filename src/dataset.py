import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class PigDataset(Dataset):
    """
    客製化 PyTorch Dataset 用於豬隻物件偵測。
    這個 class 負責讀取圖片和對應的標註。
    """

    def __init__(self, root_dir, transforms=None):
        """
        初始化 Dataset。

        Args:
            root_dir (string): 包含 'train/' 或 'test/' 的資料集根目錄路徑。
            transforms (callable, optional): 一個可選的轉換函式，用於對樣本進行增強。
        """
        self.root_dir = root_dir
        self.transforms = transforms

        # 建立 train/test 資料夾和標註檔的路徑
        self.data_dir = os.path.join(self.root_dir, "train")
        self.image_dir = os.path.join(self.data_dir, "img")
        annotations_path = os.path.join(self.data_dir, "gt.txt")

        # 讀取並儲存標註
        column_names = ["frame", "bb_left", "bb_top", "bb_width", "bb_height"]
        self.annotations = pd.read_csv(
            annotations_path, header=None, names=column_names
        )

        # 取得所有獨立的圖片 frame 編號，並排序
        self.image_frames = sorted(self.annotations["frame"].unique())

    def __len__(self):
        """
        回傳資料集中的圖片總數。
        """
        return len(self.image_frames)

    def __getitem__(self, idx):
        """
        根據索引 idx 獲取一張圖片及其對應的標註。

        Args:
            idx (int): 資料的索引 (從 0 到 len(dataset)-1)。

        Returns:
            tuple: (image, target)
                   image 是一個 PIL Image 物件。
                   target 是一個包含 'boxes' 和 'labels' 的字典。
        """
        # 1. 從索引 idx 映射到圖片的 frame 編號
        frame_id = self.image_frames[idx]

        # 2. 建立圖片路徑並讀取圖片
        image_name = f"{frame_id:08d}.jpg"
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")  # 確保是 RGB 格式

        # 3. 篩選出這張圖片對應的所有標註
        records = self.annotations[self.annotations["frame"] == frame_id]

        # 4. 將標註轉換成 PyTorch Tensors
        # 大部分的 PyTorch 模型要求 bounding box 格式為 [xmin, ymin, xmax, ymax]
        boxes = []
        for _, row in records.iterrows():
            xmin = row["bb_left"]
            ymin = row["bb_top"]
            xmax = xmin + row["bb_width"]
            ymax = ymin + row["bb_height"]
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # 建立 labels tensor
        # 我們的類別只有 "pig"，其 class id 設為 1 (0 通常留給背景)
        num_pigs = len(records)
        labels = torch.ones((num_pigs,), dtype=torch.int64)

        # 建立 target 字典
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        # 5. (可選) 應用資料增強
        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target
