# src/transforms.py

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

# 定義一個固定的圖片尺寸，方便管理
IMG_SIZE = 640


class AlbumentationsTransform:
    """
    統一處理 Albumentations 的轉換流程。
    利用 A.Compose 的特性，單一 pipeline 即可處理有無標註的情況。
    """

    def __init__(self, transforms: list):
        self.transforms = A.Compose(
            transforms,
            bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["labels"],
                # 如果一個 Bbox 經過轉換後，其可見面積小於原始面積的 25%，則移除它
                min_visibility=0.25,
                # 如果一個 Bbox 經過轉換後，其像素面積小於 16，則移除它 (例如 4x4)
                min_area=16,
            ),
        )

    def __call__(self, image, target):
        # 準備傳給 Albumentations 的參數字典
        transform_args = {
            "image": np.array(image),
            "bboxes": target.get("boxes", []),
            "labels": target.get("labels", []),
        }

        # 將 PyTorch Tensor 轉為 list，如果它們存在的話
        if isinstance(transform_args["bboxes"], torch.Tensor):
            transform_args["bboxes"] = transform_args["bboxes"].tolist()
        if isinstance(transform_args["labels"], torch.Tensor):
            transform_args["labels"] = transform_args["labels"].tolist()

        transformed = self.transforms(**transform_args)

        # 更新 target (如果存在標註)
        if "boxes" in target:
            if len(transformed["bboxes"]) > 0:
                target["boxes"] = torch.as_tensor(transformed["bboxes"], dtype=torch.float32)
            else:
                target["boxes"] = torch.empty((0, 4), dtype=torch.float32)

            target["labels"] = torch.as_tensor(transformed["labels"], dtype=torch.int64)

        return transformed["image"], target


def get_transform(train: bool) -> AlbumentationsTransform:
    """根據模式返回對應的資料增強 pipeline。"""

    # 這是訓練和驗證/測試共享的基礎轉換：保持長寬比縮放並填充
    base_transforms = [
        # ✨ 技巧: 先將最長邊縮放到 IMG_SIZE，保持長寬比
        A.LongestMaxSize(max_size=IMG_SIZE, p=1.0),
        # ✨ 技巧: 再將圖片填充到 IMG_SIZE x IMG_SIZE，不足部分用黑色填充
        A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=0, p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]

    if train:
        # 在基礎轉換前，加入訓練模式獨有的隨機增強
        train_transforms = [
            # ✨ 技巧: 隨機安全裁切，強迫模型學習局部特徵
            A.RandomSizedBBoxSafeCrop(height=IMG_SIZE, width=IMG_SIZE, p=0.3),
            A.HorizontalFlip(p=0.5),
            # 1. 減弱 CoarseDropout
            #    - 降低觸發機率 (p=0.3)：不是一半的影像，而是約三分之一的影像會被遮擋。
            #    - 減小遮擋尺寸 (max_height/width=30)：遮擋塊變小，對影像的破壞性降低。
            #    - 減少遮擋數量 (max_holes=6)：遮擋塊的總數變少。
            A.CoarseDropout(max_holes=6, max_height=30, max_width=30, fill_value=0, p=0.3),
            # 2. 減弱 ColorJitter
            #    - 降低抖動範圍：將亮度、對比度等的變化範圍從 0.2 降至 0.15。
            #    - 降低色調變化：將色調變化範圍從 0.1 降至 0.08。
            #    - 觸發機率 (p=0.7) 可以保持不變，因為顏色變化是常見的真實場景。
            A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.08, p=0.7),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
            *base_transforms,
        ]
        return AlbumentationsTransform(train_transforms)
    else:
        # 驗證/測試模式只使用基礎轉換
        return AlbumentationsTransform(base_transforms)
