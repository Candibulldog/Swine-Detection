# train.py

import os
import random

import pandas as pd
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
    # 2. 準備資料 (Dataset & DataLoader)
    # ==================================
    # 1. 先讀取一次完整的標註檔
    annotations_path = os.path.join(DATA_ROOT, "train", "gt.txt")
    column_names = ["frame", "bb_left", "bb_top", "bb_width", "bb_height"]
    full_annotations = pd.read_csv(annotations_path, header=None, names=column_names)

    # 2. 獲取所有獨一無二的圖片 frame ID，並打亂順序
    all_frames = full_annotations["frame"].unique()
    random.shuffle(all_frames)  # <-- 需要 import random

    # 3. 切分 frame ID 列表
    split_point = int(0.8 * len(all_frames))
    train_frames = all_frames[:split_point]
    val_frames = all_frames[split_point:]

    # 4. 根據切分好的 frame ID 來過濾 DataFrame
    train_df = full_annotations[full_annotations["frame"].isin(train_frames)]
    val_df = full_annotations[full_annotations["frame"].isin(val_frames)]

    # 5. 用切分好的 DataFrame 來初始化兩個獨立的 Dataset
    train_dataset = PigDataset(root_dir=DATA_ROOT, transforms=get_transform(train=True), annotations_df=train_df)
    val_dataset = PigDataset(root_dir=DATA_ROOT, transforms=get_transform(train=False), annotations_df=val_df)

    # 建立 DataLoader (這部分不變)
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
    best_map = 0.0  # 用來記錄目前最好的 mAP 分數

    print("\n--- 開始訓練 ---")
    for epoch in range(NUM_EPOCHS):
        train_one_epoch(model, optimizer, train_loader, DEVICE, epoch)

        # 呼叫 evaluate 並獲取評估結果
        coco_evaluator = evaluate(model, val_loader, DEVICE)

        # 從評估結果中提取 mAP_50:95 的分數 (它在 stats[0])
        current_map = coco_evaluator.coco_eval["bbox"].stats[0]

        # 檢查是否是目前最好的模型
        if current_map > best_map:
            best_map = current_map
            # 如果是，就儲存它！
            torch.save(model.state_dict(), "best_model.pth")
            print(f"🎉 New best model saved with mAP: {best_map:.4f} at epoch {epoch + 1}")

    print("\n--- 訓練完成 ---")
    print(f"整個訓練過程中最好的 mAP 分數是: {best_map:.4f}")

    # 儲存模型權重
    torch.save(model.state_dict(), "fasterrcnn_pig_detector.pth")
    print("模型已儲存至 fasterrcnn_pig_detector.pth")


if __name__ == "__main__":
    main()
