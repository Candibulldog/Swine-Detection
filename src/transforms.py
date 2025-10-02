# src/transforms.py

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

# 移除了舊的 AlbumentationsCompose 包裝類，因為新的 get_transform 會直接處理邏輯


def get_transform(train):
    """
    根據是訓練模式還是驗證/測試模式，返回對應的資料增強 pipeline。
    """
    if train:
        # --- 訓練模式下的增強策略 ---
        # 這個 pipeline 需要處理 bboxes 和 labels
        transform = A.Compose(
            [
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.7),
                A.HorizontalFlip(p=0.5),
                A.Resize(640, 640),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ],
            # 因為是訓練，所以必須有 bbox_params
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        )

        # --- !! 關鍵修正 !! ---
        # 返回一個 lambda 函式，它會處理 Dataset 和 Albumentations 之間的格式轉換
        return lambda image, target: {
            **target,  # 複製 target 字典中的所有鍵值對 (如 image_id)
            **transform(image=np.array(image), bboxes=target["boxes"].tolist(), labels=target["labels"].tolist()),
        }

    else:
        # --- 驗證/測試模式下的轉換 ---
        # 這個 pipeline 只需要處理 image
        transform = A.Compose(
            [
                A.Resize(640, 640),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )  # 注意：這裡沒有 bbox_params

        # 返回一個 lambda 函式，它會根據 target 是否有標註來決定如何轉換
        def test_val_transform(image, target):
            # 轉換圖片
            transformed_image = transform(image=np.array(image))["image"]

            # 檢查 target 是否包含標註 (用於驗證集)
            if "boxes" in target and "labels" in target:
                # 如果是驗證集，我們需要保持 target 結構的完整性
                # 注意：驗證集不做幾何增強，所以 box 座標不變，但需要轉為 Tensor
                return transformed_image, {
                    "image_id": target["image_id"],
                    "boxes": torch.as_tensor(target["boxes"], dtype=torch.float32),
                    "labels": torch.as_tensor(target["labels"], dtype=torch.int64),
                    "area": target["area"],
                    "iscrowd": target["iscrowd"],
                }
            else:
                # 如果是純測試集，target 只有 image_id
                return transformed_image, target

        return test_val_transform
