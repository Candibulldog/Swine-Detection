import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class PigDataset(Dataset):
    def __init__(self, root_dir, transforms=None, annotations_df=None):  # <-- 新增 annotations_df 參數
        self.root_dir = root_dir
        self.transforms = transforms

        self.data_dir = os.path.join(self.root_dir, "train")
        self.image_dir = os.path.join(self.data_dir, "img")

        if annotations_df is not None:
            # 如果提供了 DataFrame，就直接使用它
            self.annotations = annotations_df
        else:
            # 否則，像以前一樣讀取完整的 gt.txt
            annotations_path = os.path.join(self.data_dir, "gt.txt")
            column_names = ["frame", "bb_left", "bb_top", "bb_width", "bb_height"]
            self.annotations = pd.read_csv(annotations_path, header=None, names=column_names)

        # 後續的邏輯保持不變，都基於 self.annotations
        self.image_frames = sorted(self.annotations["frame"].unique())

        print(f"Dataset 初始化成功，包含 {len(self.image_frames)} 筆有效資料。")

    def __len__(self):
        """
        回傳資料集中的圖片總數。
        """
        return len(self.image_frames)

    def __getitem__(self, idx):
        """
        根據索引 idx 獲取一張圖片及其對應的標註。
        """
        # 1. 從索引 idx 映射到圖片的 frame 編號
        frame_id = self.image_frames[idx]

        # 2. 建立圖片路徑並讀取圖片
        image_name = f"{frame_id:08d}.jpg"
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")

        # 3. 篩選出這張圖片對應的所有標註
        records = self.annotations[self.annotations["frame"] == frame_id]

        # 4. 初始化 lists 來存放有效的標註
        boxes = []
        areas = []

        # 5. 遍歷標註，過濾無效 box，並轉換格式
        for _, row in records.iterrows():
            # 過濾掉寬度或高度小於 1 像素的無效 box
            if row["bb_width"] < 1 or row["bb_height"] < 1:
                continue

            # 計算 [xmin, ymin, xmax, ymax]
            xmin = row["bb_left"]
            ymin = row["bb_top"]
            xmax = xmin + row["bb_width"]
            ymax = ymin + row["bb_height"]
            boxes.append([xmin, ymin, xmax, ymax])

            # 同步計算面積，確保與 boxes list 長度一致
            areas.append(row["bb_width"] * row["bb_height"])

        # 6. 將 lists 轉換為 PyTorch Tensors
        #    如果過濾後沒有任何有效的 box，我們需要回傳空的 Tensors
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            areas = torch.as_tensor(areas, dtype=torch.float32)
        else:
            # 確保即使沒有 box，Tensor 也有正確的 shape (0, 4)
            boxes = torch.empty((0, 4), dtype=torch.float32)
            areas = torch.empty((0,), dtype=torch.float32)

        # 7. 建立其他必要的 Tensors
        num_boxes = boxes.shape[0]
        labels = torch.ones((num_boxes,), dtype=torch.int64)
        image_id = torch.tensor([frame_id])
        iscrowd = torch.zeros((num_boxes,), dtype=torch.int64)

        # 8. 組裝最終的 target 字典
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = areas
        target["iscrowd"] = iscrowd

        # 9. (可選) 應用資料增強
        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target
