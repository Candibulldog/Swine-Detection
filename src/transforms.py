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
        self.transforms = A.Compose(transforms, bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]))

    def __call__(self, image, target):
        # 準備傳給 Albumentations 的參數字典
        transform_args = {
            "image": np.array(image),
        }
        if "boxes" in target:
            transform_args["bboxes"] = target["boxes"].tolist()
            transform_args["labels"] = target["labels"].tolist()

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
            # 整合的顏色抖動
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.7),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
            *base_transforms,
        ]
        return AlbumentationsTransform(train_transforms)
    else:
        # 驗證/測試模式只使用基礎轉換
        return AlbumentationsTransform(base_transforms)
