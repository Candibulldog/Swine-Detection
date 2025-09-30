# src/transforms.py

import torchvision.transforms.v2 as T


def get_transform(train):
    """
    定義訓練和驗證時的資料轉換流程。

    Args:
        train (bool): 如果是 True，則套用訓練時的資料增強；否則只做基本轉換。

    Returns:
        A torchvision transforms object.
    """
    transforms = []
    # 將 PIL Image 轉換為 PyTorch Tensor
    # 注意：v2 的 ToTensor 會自動將 bbox 和 label 格式化，且將像素值標準化到 [0, 1]
    transforms.append(T.ToTensor())

    if train:
        # 在訓練時，加入隨機水平翻轉的資料增強
        # p=0.5 代表有 50% 的機率會進行翻轉
        transforms.append(T.RandomHorizontalFlip(p=0.5))

    return T.Compose(transforms)
