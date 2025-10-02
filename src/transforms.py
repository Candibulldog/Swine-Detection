# src/transforms.py

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2


class AlbumentationsCompose:
    def __init__(self, transforms):
        # --- !! 最終修正 !! ---
        # 我們在這裡建立兩個 pipeline，一個帶 bbox_params，一個不帶
        self.transform_with_boxes = A.Compose(
            transforms, bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"])
        )
        self.transform_without_boxes = A.Compose(transforms)

    def __call__(self, image, target):
        # 根據 target 是否有 'boxes' 鍵，來決定使用哪個 pipeline
        if "boxes" in target:
            # 訓練或驗證模式
            transformed = self.transform_with_boxes(
                image=np.array(image), bboxes=target["boxes"].tolist(), labels=target["labels"].tolist()
            )

            image = transformed["image"]

            if len(transformed["bboxes"]) > 0:
                target["boxes"] = torch.as_tensor(transformed["bboxes"], dtype=torch.float32)
            else:
                target["boxes"] = torch.empty((0, 4), dtype=torch.float32)

            target["labels"] = torch.as_tensor(transformed["labels"], dtype=torch.int64)
        else:
            # 純測試模式
            transformed = self.transform_without_boxes(image=np.array(image))
            image = transformed["image"]

        return image, target


def get_transform(train):
    if train:
        # 訓練模式的增強策略
        transform_list = [
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.7),
            A.HorizontalFlip(p=0.5),
            A.Resize(640, 640),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    else:
        # 驗證/測試模式的轉換 (無隨機增強)
        transform_list = [
            A.Resize(640, 640),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]

    return AlbumentationsCompose(transform_list)
