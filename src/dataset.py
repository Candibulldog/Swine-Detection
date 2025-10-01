import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class PigDataset(Dataset):
    # 在 __init__ 中加入一個 'is_train' 參數
    def __init__(self, root_dir, frame_ids, is_train=True, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.is_train = is_train

        # 根據 is_train 決定要讀取的資料夾
        if self.is_train:
            self.data_dir = os.path.join(self.root_dir, "train")
            annotations_path = os.path.join(self.data_dir, "gt.txt")
            column_names = ["frame", "bb_left", "bb_top", "bb_width", "bb_height"]
            full_annotations = pd.read_csv(annotations_path, header=None, names=column_names)
            self.annotations = full_annotations[full_annotations["frame"].isin(frame_ids)]
        else:
            # 測試模式下，沒有標註
            self.data_dir = os.path.join(self.root_dir, "test")
            self.annotations = None

        self.image_dir = os.path.join(self.data_dir, "img")

        # 如果是測試模式，frame_ids 就是所有圖片檔名
        if not self.is_train:
            self.image_frames = sorted([int(f.split(".")[0]) for f in os.listdir(self.image_dir)])
        else:
            self.image_frames = sorted(frame_ids)

        print(f"Dataset 初始化成功，模式: {'Train' if transforms else 'Val'}，包含 {len(self.image_frames)} 筆資料。")

    def __len__(self):
        """
        回傳資料集中的圖片總數。
        """
        return len(self.image_frames)

    def __getitem__(self, idx):
        frame_id = self.image_frames[idx]
        image_name = f"{frame_id:08d}.jpg"
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")

        # 建立 target 字典，無論是訓練還是測試都需要它
        target = {"image_id": torch.tensor([frame_id])}

        # 根據是否為訓練模式，決定是否要加載標註
        if self.is_train:
            records = self.annotations[self.annotations["frame"] == frame_id]

            boxes = []
            areas = []
            for _, row in records.iterrows():
                if row["bb_width"] < 1 or row["bb_height"] < 1:
                    continue
                xmin = row["bb_left"]
                ymin = row["bb_top"]
                xmax = xmin + row["bb_width"]
                ymax = ymin + row["bb_height"]
                boxes.append([xmin, ymin, xmax, ymax])
                areas.append(row["bb_width"] * row["bb_height"])

            if len(boxes) > 0:
                boxes = torch.as_tensor(boxes, dtype=torch.float32)
                areas = torch.as_tensor(areas, dtype=torch.float32)
            else:
                boxes = torch.empty((0, 4), dtype=torch.float32)
                areas = torch.empty((0,), dtype=torch.float32)

            num_boxes = boxes.shape[0]
            labels = torch.ones((num_boxes,), dtype=torch.int64)
            iscrowd = torch.zeros((num_boxes,), dtype=torch.int64)

            # 將標註資訊加入 target 字典
            target["boxes"] = boxes
            target["labels"] = labels
            target["area"] = areas
            target["iscrowd"] = iscrowd

        # --- !! 關鍵修正 !! ---
        # 現在，無論是訓練還是測試，我們都呼叫同樣的 transform 邏輯
        # 只是測試時的 transform 比較簡單 (只有 ToTensor)
        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target
