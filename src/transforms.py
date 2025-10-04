# src/transforms.py

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

# -----------------------------------------------------------------------------
# 常數定義 (Constants)
# -----------------------------------------------------------------------------

# 將所有影像統一處理到的目標尺寸，便於模型進行批次處理。
IMG_SIZE = 640

# 根據訓練集計算出的正規化統計數據。
# 使用數據集自身的均值和標準差，可以讓數據分佈更理想地中心化，有助於模型穩定收斂。
DATASET_MEAN = [0.3615, 0.3564, 0.3543]
DATASET_STD = [0.3089, 0.3045, 0.3051]


# -----------------------------------------------------------------------------
# Albumentations 轉換器封裝 (Wrapper Class)
# -----------------------------------------------------------------------------


class AlbumentationsTransform:
    """
    一個封裝類，用於統一處理 Albumentations 的轉換流程。
    其設計確保了同一轉換管線 (pipeline) 可以同時處理影像和其對應的邊界框 (bounding boxes)。
    """

    def __init__(self, transforms: list):
        # 建立轉換管線，並配置邊界框的處理參數。
        self.transforms = A.Compose(
            transforms,
            bbox_params=A.BboxParams(
                # 標註格式為 [x_min, y_min, x_max, y_max]。
                format="pascal_voc",
                # 邊界框對應的類別標籤字段名稱。
                label_fields=["labels"],
                # 若一個邊界框在轉換後，其可見面積小於原始面積的 25%，則將其移除。
                # 這有助於過濾掉那些在裁切後幾乎看不見的物體。
                min_visibility=0.25,
                # 若一個邊界框在轉換後，其像素面積小於 16 (例如 4x4)，則將其移除。
                # 這有助於過濾掉過小的物體，避免模型學習噪點。
                min_area=16,
            ),
        )

    def __call__(self, image, target):
        """應用轉換管線到影像和標註上。"""
        # 將 PIL 影像和 PyTorch 張量轉換為 Albumentations 所需的 Numpy 格式。
        transform_args = {
            "image": np.array(image),
            "bboxes": target.get("boxes", []),
            "labels": target.get("labels", []),
        }
        if isinstance(transform_args["bboxes"], torch.Tensor):
            transform_args["bboxes"] = transform_args["bboxes"].tolist()
        if isinstance(transform_args["labels"], torch.Tensor):
            transform_args["labels"] = transform_args["labels"].tolist()

        # 應用轉換
        transformed = self.transforms(**transform_args)

        # 將轉換後的結果重新打包為 PyTorch 張量格式。
        if "boxes" in target:
            if len(transformed["bboxes"]) > 0:
                target["boxes"] = torch.as_tensor(transformed["bboxes"], dtype=torch.float32)
            else:
                target["boxes"] = torch.empty((0, 4), dtype=torch.float32)
            target["labels"] = torch.as_tensor(transformed["labels"], dtype=torch.int64)

        return transformed["image"], target


# -----------------------------------------------------------------------------
# 獲取轉換管線的工廠函數 (Factory Function)
# -----------------------------------------------------------------------------


def get_transform(train: bool) -> AlbumentationsTransform:
    """
    根據是訓練模式還是驗證/測試模式，返回相應的資料增強管線。

    Args:
        train (bool): 若為 True，返回包含豐富資料增強的訓練管線；
                      否則，返回僅包含基礎尺寸調整與正規化的管線。
    """

    # --- 基礎轉換管線 (用於驗證與測試) ---
    # 此管線僅進行必要的尺寸統一與正規化，以確保模型輸入的一致性。
    base_transforms = [
        # 1. 保持長寬比縮放：將影像最長的一邊縮放到 IMG_SIZE。
        A.LongestMaxSize(max_size=IMG_SIZE, p=1.0),
        # 2. 填充至正方形：將縮放後的影像填充為 IMG_SIZE x IMG_SIZE 的正方形，
        #    使用黑色 (value=0) 進行填充，這對夜間紅外線影像的背景很自然。
        A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=0, value=0, p=1.0),
        # 3. 正規化：使用訓練集的統計數據進行正規化，使像素值分佈在零點附近，
        #    有助於加速模型收斂並提升訓練穩定性。
        A.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
        # 4. 轉換為張量：將 Numpy 陣列轉換為 PyTorch 張量。
        ToTensorV2(),
    ]

    if train:
        train_transforms = [
            # === 1. 幾何與空間增強 ===
            A.LongestMaxSize(max_size=IMG_SIZE, p=1.0),
            A.PadIfNeeded(
                min_height=IMG_SIZE,
                min_width=IMG_SIZE,
                border_mode=0,  # cv2.BORDER_CONSTANT
                value=0,  # 配合 border_mode=0，使用黑色填充
                p=1.0,
            ),
            A.RandomSizedBBoxSafeCrop(height=IMG_SIZE, width=IMG_SIZE, erosion_rate=0.2, p=0.5),
            A.HorizontalFlip(p=0.5),
            # 【修正】使用正確的參數名
            A.Affine(
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}, scale=(0.9, 1.1), rotate=(-15, 15), p=0.7
            ),
            # === 2. 核心領域適應增強 ===
            A.ToGray(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.6, contrast_limit=0.5, p=0.9),
            # === 3. 抗遮擋增強 ===
            # 【修正】移除不支援的參數，GridDropout 主要由 ratio 控制
            A.GridDropout(ratio=0.5, p=0.5),
            # === 4. 最終處理 ===
            A.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
            ToTensorV2(),
        ]
        return AlbumentationsTransform(train_transforms)
    else:
        # 驗證/測試模式下，只使用基礎轉換。
        return AlbumentationsTransform(base_transforms)
