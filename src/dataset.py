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
        """
        根據索引 idx 獲取一張圖片及其對應的標註。
        """
        # 1. 從索引 idx 映射到圖片的 frame 編號
        frame_id = self.image_frames[idx]

        # 2. 建立圖片路徑並讀取圖片
        image_name = f"{frame_id:08d}.jpg"
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")

        # 對於測試集，我們只需要回傳圖片和 image_id
        if not self.is_train:
            # 建立一個只包含 image_id 的 target
            target = {"image_id": torch.tensor([frame_id])}
            if self.transforms:
                # 測試集只需要 ToTensor
                image = self.transforms(image)
            return image, target

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
