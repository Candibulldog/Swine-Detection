# src/utils.py


def collate_fn(batch):
    """
    自定義的 collate_fn，用於處理物件偵測任務中，
    每個樣本標註數量不一致的情況。

    DataLoader 在打包一個 batch 時會呼叫這個函式。
    它接收一個 list of tuples (每個 tuple 是 (image, target))，
    並將它們重新組織成一個 tuple of lists (一個 list 是 images, 另一個是 targets)。
    """
    return tuple(zip(*batch))
