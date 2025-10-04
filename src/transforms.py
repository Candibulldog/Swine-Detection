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


# 泛化特化型增強
def get_transform(train: bool) -> AlbumentationsTransform:
    """根據模式返回對應的資料增強 pipeline。"""

    base_transforms = [
        A.LongestMaxSize(max_size=IMG_SIZE, p=1.0),
        A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=0, p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]

    if train:
        train_transforms = [
            # --- 1. 幾何與尺寸增強 ---
            # 首先進行大的、可能改變物體尺寸和位置的變換
            A.RandomSizedBBoxSafeCrop(height=IMG_SIZE, width=IMG_SIZE, p=0.3),
            A.HorizontalFlip(p=0.5),
            A.Affine(shift_limit_x=0.0625, shift_limit_y=0.0625, scale_limit=0.1, rotate=15, p=0.5),
            # --- 2. 像素級與特徵抹除增強 ---
            # A.OneOf 讓每次只隨機選擇一種顏色相關的增強，避免效果疊加過度
            A.OneOf(
                [
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                ],
                p=0.8,
            ),  # 80% 的機率執行其中一種
            # 【新增】模擬相機噪點與影像品質下降
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2),
            A.ImageCompression(quality_lower=75, quality_upper=95, p=0.2),
            # 【核心】使用 GridDropout 進行更強的特徵抹除
            # grid=(8, 8) 表示將圖片劃分為 8x8 的網格
            # ratio=0.6 表示隨機丟棄掉 60% 的網格塊
            # holes_number_x 和 holes_number_y 可以更精細地控制每行/列丟棄的數量
            A.GridDropout(ratio=0.6, unit_size_min=None, unit_size_max=None, holes_number_x=5, holes_number_y=5, p=0.5),
            # --- 3. 最終處理 ---
            # 將基礎轉換放在最後
            *base_transforms,
        ]
        return AlbumentationsTransform(train_transforms)
    else:
        # 驗證/測試模式只使用基礎轉換
        return AlbumentationsTransform(base_transforms)
