# src/transforms.py

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2


class AlbumentationsCompose:
    """
    一個包裝類，讓 albumentations 的 transform pipeline
    可以無縫地接收我們 Dataset 輸出的 (PIL Image, target_dict) 格式。
    """

    def __init__(self, transforms):
        # Albumentations 的核心，定義了轉換流程
        # 我們需要告訴它 bounding box 的格式以及如何處理標籤
        self.transforms = A.Compose(
            transforms,
            bbox_params=A.BboxParams(
                format="pascal_voc",  # [xmin, ymin, xmax, ymax], 這正是我們 Dataset 輸出的格式
                label_fields=["labels"],
            ),
        )

    def __call__(self, image, target):
        # --- !! 關鍵修正 !! ---
        # 判斷 target 字典中是否存在 'boxes' 鍵，來區分訓練/驗證模式和純測試模式
        if "boxes" in target and "labels" in target:
            # 這是訓練或驗證模式，target 包含標註
            transformed = self.transforms(
                image=np.array(image), bboxes=target["boxes"].tolist(), labels=target["labels"].tolist()
            )

            # 將轉換後的資料轉回我們的 target 字典格式
            image = transformed["image"]

            if len(transformed["bboxes"]) > 0:
                target["boxes"] = torch.as_tensor(transformed["bboxes"], dtype=torch.float32)
            else:
                target["boxes"] = torch.empty((0, 4), dtype=torch.float32)

            target["labels"] = torch.as_tensor(transformed["labels"], dtype=torch.int64)

        else:
            # 這是純測試模式，target 只有 image_id，沒有標註
            # 我們只需要對 image 進行轉換
            # 注意：這裡不能傳入 bboxes 和 labels 參數
            transformed = self.transforms(image=np.array(image))
            image = transformed["image"]
            # target 保持不變 (只包含 image_id)

        return image, target


def get_transform(train):
    """
    根據是訓練模式還是驗證/測試模式，返回對應的資料增強 pipeline。
    """
    if train:
        # --- 訓練模式下的增強策略 ---
        # 目標：提升 Bbox 回歸的精準度
        transform_list = [
            # 顏色增強：讓模型對光照變化更不敏感
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            # 幾何增強：讓模型學習不同姿態、大小、角度的豬
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.7),
            # 水平翻轉
            A.HorizontalFlip(p=0.5),
            # (可選) 隨機安全裁切，確保裁切後至少有一個 bbox 留下
            # A.RandomBBoxSafeCrop(p=0.3),
            # 將所有圖片 resize 到一個固定大小，有助於模型處理和批次化
            # 這個尺寸可以根據你的 GPU 記憶體和模型特性來調整
            A.Resize(640, 640),
            # 標準化 (使用 ImageNet 的均值和標準差)
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # 轉換為 PyTorch Tensor，這必須是最後一步！
            ToTensorV2(),
        ]
        return AlbumentationsCompose(transform_list)
    else:
        # --- 驗證/測試模式下的轉換 ---
        # 只需要做必要的尺寸調整、標準化和 Tensor 轉換，不做隨機增強
        transform_list = [
            A.Resize(640, 640),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
        return AlbumentationsCompose(transform_list)
