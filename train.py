# train.py

import torch

# 從 src 資料夾中引入我們寫好的模組
from src.dataset import PigDataset
from src.engine import evaluate, train_one_epoch
from src.model import create_model
from src.transforms import get_transform
from src.utils import collate_fn
from torch.utils.data import DataLoader

# ==================================
# 1. 超參數設定 (Hyperparameters)
# ==================================
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
NUM_CLASSES = 2  # 1 (pig) + 1 (background)
NUM_EPOCHS = 10
BATCH_SIZE = 4
LEARNING_RATE = 0.005
DATA_ROOT = "/content/data"  # 在 Colab 中的資料路徑


def main():
    # ==================================
    # 2. 準備資料 (Dataset & DataLoader) (最終修正版)
    # ==================================
    # 建立一個套用了訓練 transform 的完整資料集
    dataset = PigDataset(root_dir=DATA_ROOT, transforms=get_transform(train=True))
    # 建立一個套用了驗證 transform 的資料集副本
    dataset_val = PigDataset(root_dir=DATA_ROOT, transforms=get_transform(train=False))

    # 隨機打亂索引
    indices = torch.randperm(len(dataset)).tolist()

    # 按照索引切分兩個獨立的 dataset
    split_point = int(0.8 * len(dataset))
    train_dataset = torch.utils.data.Subset(dataset, indices[:split_point])
    val_dataset = torch.utils.data.Subset(dataset_val, indices[split_point:])

    # 建立 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    print(f"訓練集大小: {len(train_dataset)}")
    print(f"驗證集大小: {len(val_dataset)}")

    # ==================================
    # 3. 建立模型和優化器
    # ==================================
    model = create_model(NUM_CLASSES)
    model.to(DEVICE)

    # 設定優化器 (SGD 是一個穩健的選擇)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)

    print("\n--- 檢查設備 ---")
    print(f"DEVICE is set to: {DEVICE}")

    # ==================================
    # 4. 訓練迴圈 (Training Loop)
    # ==================================
    print("\n--- 開始訓練 ---")
    for epoch in range(NUM_EPOCHS):
        # 呼叫 engine 中的函式來進行訓練
        train_one_epoch(model, optimizer, train_loader, DEVICE, epoch)

        # 呼叫 engine 中的函式來進行驗證
        evaluate(model, val_loader, DEVICE)

    print("\n--- 訓練完成 ---")

    # 儲存模型權重
    torch.save(model.state_dict(), "fasterrcnn_pig_detector.pth")
    print("模型已儲存至 fasterrcnn_pig_detector.pth")


if __name__ == "__main__":
    main()
