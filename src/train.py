# src/train.py
import argparse
import csv
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

# 從 src 資料夾中引入我們寫好的模組
from src.dataset import PigDataset
from src.engine import evaluate, train_one_epoch
from src.model import create_model
from src.transforms import get_transform
from src.utils import collate_fn  # ✅ 直接匯入函式本體

# --- 全域常數 ---
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
NUM_CLASSES = 2  # 1 (pig) + 1 (background)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(_):
    # 讓 DataLoader workers 的亂數可重現
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main():
    # --- 1. 設定與解析命令行參數 ---
    parser = argparse.ArgumentParser(description="Pig Detection Training Script")
    default_dr = "/content/data" if os.path.exists("/content/data") else "./data"
    parser.add_argument("--data_root", type=str, default=default_dr, help="Root path that contains train/ and test/")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.005, help="Initial learning rate")
    parser.add_argument("--output_dir", type=str, default="models", help="Directory to save the best model")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"DEVICE is set to: {DEVICE}")
    print(f"訓練參數: Epochs={args.epochs}, Batch Size={args.batch_size}, LR={args.lr}")

    # --- 2. 準備資料 ---
    DATA_ROOT = args.data_root
    gt_path = os.path.join(DATA_ROOT, "train", "gt.txt")
    img_dir = os.path.join(DATA_ROOT, "train", "img")

    if not os.path.isfile(gt_path):
        raise FileNotFoundError(f"找不到標註檔：{gt_path}")
    if not os.path.isdir(img_dir):
        raise NotADirectoryError(f"找不到影像資料夾：{img_dir}")

    full_annotations = pd.read_csv(gt_path, header=None, names=["frame", "bb_left", "bb_top", "bb_width", "bb_height"])

    # data analysis code (optional)
    """
    print("--- 開始數據探索 ---")

    # 確保輸出目錄存在
    output_dir = "data_analysis"  # 你可以自訂資料夾名稱
    os.makedirs(output_dir, exist_ok=True)

    # 計算面積和長寬比
    full_annotations["area"] = full_annotations["bb_width"] * full_annotations["bb_height"]
    full_annotations["aspect_ratio"] = full_annotations["bb_width"] / (full_annotations["bb_height"] + 1e-6)

    # --- 1. 繪製並儲存面積的直方圖 ---
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(full_annotations["area"], bins=50, kde=True)
    plt.title("Bbox 面積分佈")
    plt.xlabel("面積 (pixels)")
    plt.ylabel("數量")
    plt.yscale("log")

    # --- 2. 繪製並儲存長寬比的直方圖 ---
    plt.subplot(1, 2, 2)
    sns.histplot(full_annotations["aspect_ratio"], bins=50, kde=True)
    plt.title("Bbox 長寬比分佈")
    plt.xlabel("長寬比 (寬/高)")
    plt.xlim(0, 10)
    plt.ylabel("數量")
    plt.yscale("log")

    # --- !! 關鍵修改 !! ---
    # 將 plt.show() 改為 plt.savefig()
    plt.tight_layout()
    save_path = os.path.join(output_dir, "bbox_distribution.png")
    plt.savefig(save_path)
    print(f"✅ 圖表已儲存至: {save_path}")

    # 清除當前的圖形，以防萬一
    plt.close()

    # --- 3. 打印統計數據到終端機 ---
    print("\n面積統計數據 (Area Stats):")
    print(full_annotations["area"].describe())
    print("\n長寬比統計數據 (Aspect Ratio Stats):")
    print(full_annotations["aspect_ratio"].describe())

    print("\n--- 數據探索完成，程式即將退出 ---")
    sys.exit()  # 分析完畢，退出程式
    """

    # =================================================================
    # ✨✨✨ Data cleaning ✨✨✨
    # =================================================================
    print(f"原始標註數量: {len(full_annotations)}")

    # 1. 根據面積統計數據，過濾掉面積小於 500 的 Bbox
    MIN_AREA = 500
    full_annotations["area"] = full_annotations["bb_width"] * full_annotations["bb_height"]
    full_annotations = full_annotations[full_annotations["area"] > MIN_AREA]
    print(f"過濾掉面積過小 Bbox 後的數量: {len(full_annotations)}")

    # 2. 根據長寬比統計數據，過濾掉形狀畸形的 Bbox
    MAX_ASPECT_RATIO = 6.0
    full_annotations["aspect_ratio"] = full_annotations["bb_width"] / (full_annotations["bb_height"] + 1e-6)
    full_annotations = full_annotations[
        (full_annotations["aspect_ratio"] < MAX_ASPECT_RATIO)
        & (full_annotations["aspect_ratio"] > 1 / MAX_ASPECT_RATIO)
    ]
    print(f"過濾掉畸形 Bbox 後的數量: {len(full_annotations)}")

    # 移除輔助欄位，保持 DataFrame 乾淨
    full_annotations = full_annotations.drop(columns=["area", "aspect_ratio"])
    # ==================================================================

    # ✅ 更穩健的檔名解析（只收純數字檔名，如 00000001.jpg）
    existing_files = set()
    for f in os.listdir(img_dir):
        stem, _ = os.path.splitext(f)
        if stem.isdigit():
            existing_files.add(int(stem))

    annotated_frames = set(map(int, full_annotations["frame"].unique()))
    valid_frames = sorted(existing_files.intersection(annotated_frames))

    if len(valid_frames) < 2:
        raise RuntimeError("可用影像不足以切分 train/val，請檢查資料完整性。")

    # 固定隨機種子後再 shuffle，確保可重現
    rng = random.Random(args.seed)
    rng.shuffle(valid_frames)

    split_point = int(0.8 * len(valid_frames))
    # 至少留 1 張給驗證（以免 100%/0% 邊界）
    split_point = min(max(1, split_point), len(valid_frames) - 1)

    train_frames = valid_frames[:split_point]
    val_frames = valid_frames[split_point:]

    train_dataset = PigDataset(
        root_dir=DATA_ROOT,
        frame_ids=train_frames,
        is_train=True,  # 需要標註
        transforms=get_transform(train=True),
    )
    val_dataset = PigDataset(
        root_dir=DATA_ROOT,
        frame_ids=val_frames,
        is_train=True,  # 驗證集仍取自 train，有標註 → True
        transforms=get_transform(train=False),  # 驗證禁用隨機增強
    )

    # --- DataLoader（快又穩） ---
    cpu_cnt = os.cpu_count() or 2
    num_workers = max(1, cpu_cnt - 1)  # 至少 1，Colab/雲端通常這樣最穩
    g = torch.Generator()
    g.manual_seed(args.seed)

    dl_kwargs = dict(
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,  # ✅ 使用匯入的函式，不要寫 utils.collate_fn
        worker_init_fn=seed_worker,
        generator=g,
    )
    if num_workers > 0:
        dl_kwargs["persistent_workers"] = True
        dl_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **dl_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, **dl_kwargs)

    print(f"訓練集大小: {len(train_dataset)}, 驗證集大小: {len(val_dataset)}")

    # --- 3. 建立模型、優化器與學習率排程器 ---
    model = create_model(NUM_CLASSES)
    model.to(DEVICE)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)

    # 使用 CosineAnnealingLR 讓學習率平滑下降
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)

    # --- 4. 訓練與驗證迴圈 ---
    best_map = -1.0
    best_path = os.path.join(args.output_dir, "best_model.pth")
    log_file_path = os.path.join(args.output_dir, "training_log.csv")  # ✅ 日誌放在 output_dir

    with open(log_file_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "mAP_50:95", "AP_50"])

    print("\n--- 開始訓練 ---")
    for epoch in range(args.epochs):
        train_one_epoch(model, optimizer, train_loader, DEVICE, epoch)
        lr_scheduler.step()

        coco_evaluator = evaluate(model, val_loader, DEVICE)
        current_map = coco_evaluator.coco_eval["bbox"].stats[0]  # mAP_50:95
        current_ap50 = coco_evaluator.coco_eval["bbox"].stats[1]  # AP_50

        with open(log_file_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, f"{current_map:.4f}", f"{current_ap50:.4f}"])

        if current_map > best_map:
            best_map = current_map
            torch.save(model.state_dict(), best_path)
            print(f"🎉 New best model saved to {best_path} with mAP: {best_map:.4f} at epoch {epoch + 1}")

    print("\n--- 訓練完成 ---")
    print(f"整個訓練過程中最好的 mAP 分數是: {best_map:.4f}, 對應模型已儲存至 {best_path}")


if __name__ == "__main__":
    main()
