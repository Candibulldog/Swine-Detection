# src/utils.py


def collate_fn(batch):
    """
    自定義的 collate_fn，用於處理不同圖片標註框數量不同的情況。
    """
    return tuple(zip(*batch))
