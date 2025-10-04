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
        # --- 訓練專用增強管線 ---
        # 此管線在基礎轉換之上，加入了大量增強來提升模型的泛化能力與魯棒性。
        train_transforms = [
            # === 1. 幾何與空間增強 (Geometric & Spatial Augmentations) ===
            # 目標：使模型對豬隻的大小、位置、角度和部分遮擋不敏感。
            A.LongestMaxSize(max_size=IMG_SIZE, p=1.0),
            A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=0, value=0, p=1.0),
            # 隨機安全裁切：模擬「拉近鏡頭」的效果，讓模型學習偵測不同尺寸和
            # 部分可見的豬隻。erosion_rate 參數有助於在豬隻身體被輕微裁切時
            # 依然保留其邊界框。
            A.RandomSizedBBoxSafeCrop(height=IMG_SIZE, width=IMG_SIZE, erosion_rate=0.2, p=0.5),
            # 水平翻轉：豬是左右對稱的，此增強能讓數據集規模有效加倍。
            A.HorizontalFlip(p=0.5),
            # 仿射變換：組合了平移、縮放和旋轉，模擬了相機視角的輕微變化。
            A.Affine(shift_limit_x=0.05, shift_limit_y=0.05, scale_limit=0.1, rotate_limit=15, p=0.7),
            # === 2. 核心領域適應增強 (Core Domain Adaptation) ===
            # 目標：彌合訓練集(夜間)與測試集(日間)之間的光照與色彩鴻溝。
            # 隨機轉為灰階：強迫模型學習豬的形狀、輪廓和紋理等通用特徵，
            # 而非依賴顏色(例如粉紅色)，直接模擬了從彩色到紅外線影像的轉換。
            A.ToGray(p=0.5),
            # 隨機亮度和對比度：大範圍地調整亮度和對比度，用以模擬從深夜的
            # 微光到白天的強光等各種光照條件，是應對日夜變化的核心策略。
            A.RandomBrightnessContrast(brightness_limit=0.6, contrast_limit=0.5, p=0.9),
            # === 3. 抗遮擋增強 (Occlusion Robustness) ===
            # 目標：在豬隻高度擁擠和互相遮擋的場景下，提升模型的偵測能力。
            # 網格丟棄：隨機移除影像中的一塊塊網格區域，模擬嚴重的身體遮擋。
            # 這迫使模型必須根據豬隻的部分可見特徵（如一隻耳朵、一段背）來做出判斷。
            A.GridDropout(ratio=0.5, holes_number_x=5, holes_number_y=5, p=0.5),
            # === 4. 最終處理 (Finalization) ===
            A.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
            ToTensorV2(),
        ]
        return AlbumentationsTransform(train_transforms)
    else:
        # 驗證/測試模式下，只使用基礎轉換。
        return AlbumentationsTransform(base_transforms)
