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
    # 1. 獲取所有有效的 Frame ID
    #    這段邏輯只執行一次，確保我們只使用有圖片且有標註的資料
    gt_path = os.path.join(DATA_ROOT, "train", "gt.txt")
    img_dir = os.path.join(DATA_ROOT, "train", "img")

    full_annotations = pd.read_csv(gt_path, header=None, names=["frame", "bb_left", "bb_top", "bb_width", "bb_height"])
    existing_files = {int(f.split(".")[0]) for f in os.listdir(img_dir)}
    annotated_frames = set(full_annotations["frame"].unique())

    valid_frames = sorted(list(existing_files.intersection(annotated_frames)))
    random.shuffle(valid_frames)

    # 2. 切分 Frame ID 列表
    split_point = int(0.8 * len(valid_frames))
    train_frames = valid_frames[:split_point]
    val_frames = valid_frames[split_point:]

    # 3. 用切分好的 Frame ID 列表來初始化兩個【完全獨立】的 Dataset
    #    不再使用 Subset 或 random_split！
    train_dataset = PigDataset(
        root_dir=DATA_ROOT,
        frame_ids=train_frames,
        is_train=True,
        transforms=get_transform(train=True),
    )

    val_dataset = PigDataset(
        root_dir=DATA_ROOT,
        frame_ids=val_frames,
        is_train=True,
        transforms=get_transform(train=False),
    )

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
