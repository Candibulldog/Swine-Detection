# src/transforms.py

import random

import torchvision.transforms.functional as F


# Compose 類保持不變
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


# ToTensor 類保持不變
class ToTensor:
    def __call__(self, image, target):
        # 將 PIL Image 轉換為 PyTorch Tensor
        image = F.to_tensor(image)
        return image, target


# RandomHorizontalFlip 類進行修正
class RandomHorizontalFlip:
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            # --- !! 關鍵修正 !! ---
            # image 現在是一個 Tensor，形狀是 [C, H, W] (通道, 高, 寬)
            # 我們用 image.shape[-1] 來獲取寬度，而不是 image.size
            height, width = image.shape[-2:]

            # F.hflip 可以同時處理 PIL Image 和 Tensor
            image = F.hflip(image)

            # 手動翻轉 bounding box 的邏輯保持不變
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
        return image, target


def get_transform(train):
    transforms = []
    # 順序很重要：先做 Tensor 轉換
    transforms.append(ToTensor())
    if train:
        # 然後再做其他基於 Tensor 的操作
        transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)
