# src/utils.py


def collate_fn(batch):
    """
    自定義 collate_fn 以處理標註數量不一的樣本。

    DataLoader 會將 (image, target) 的列表轉換為
    (images_tuple, targets_tuple) 的元組。
    """
    return tuple(zip(*batch))
