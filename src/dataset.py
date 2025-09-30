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
        """
        self.root_dir = root_dir
        self.transforms = transforms

        self.data_dir = os.path.join(self.root_dir, "train")
        self.image_dir = os.path.join(self.data_dir, "img")
        annotations_path = os.path.join(self.data_dir, "gt.txt")

        # 1. 讀取所有標註
        column_names = ["frame", "bb_left", "bb_top", "bb_width", "bb_height"]
        all_annotations = pd.read_csv(annotations_path, header=None, names=column_names)

        # 2. 取得實際存在的所有圖片檔名，並轉換為 frame ID
        all_image_files = os.listdir(self.image_dir)

        # === [修正長度問題 1: 將 set comprehension 包在括號中] ===
        # 我們將整個表達式轉換為一個產生器 (generator)，再傳入 set()
        # 這樣就可以利用括號內的隱式換行，非常 Pythonic！
        existing_frames_set = set(int(fname.split(".")[0]) for fname in all_image_files if fname.endswith(".jpg"))
        # =======================================================

        # 3. 取得標註檔中提到的所有 frame ID
        annotated_frames_set = set(all_annotations["frame"].unique())

        # 4. 找出兩者的交集
        valid_frames_set = existing_frames_set.intersection(annotated_frames_set)

        # 5. 過濾 annotations
        self.annotations = all_annotations[all_annotations["frame"].isin(valid_frames_set)]

        # 6. 建立圖片列表
        self.image_frames = sorted(list(valid_frames_set))

        # === [修正長度問題 2: 將 f-string 拆成多行] ===
        # Python 會自動合併相鄰的字串。我們可以利用這個特性和括號來換行。
        print(
            f"找到 {len(all_image_files)} 個圖片檔，"
            f"{len(annotated_frames_set)} 個被標註的影格，"
            f"有效資料共 {len(self.image_frames)} 筆。"
        )

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
