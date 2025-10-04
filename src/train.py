# src/train.py

import argparse
import csv
import os
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.dataset import PigDataset
from src.engine import evaluate, train_one_epoch
from src.model import create_model
from src.transforms import get_transform
from src.utils import collate_fn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 2  # 1 (pig) + 1 (background)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # 確保 cuDNN 的確定性，可能稍微影響效能，但可重現性更高
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(_):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# 資料清洗，移除無效的 bounding box 標註
def filter_annotations(df: pd.DataFrame) -> pd.DataFrame:
    print(f"原始標註數量: {len(df)}")
    initial_count = len(df)
    df_filtered = df[(df["bb_width"] > 1) & (df["bb_height"] > 1)].copy()
    removed_count = initial_count - len(df_filtered)
    if removed_count > 0:
        print(f"移除非法標註 (w/h <= 1): {removed_count} 個")
    print(f"過濾後的標註數量: {len(df_filtered)}")
    return df_filtered


def main():
    parser = argparse.ArgumentParser(description="Pig Detection Training Script")
    parser.add_argument("--data_root", type=Path, default=Path("./data"), help="Root path containing train/ and test/")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.005, help="Initial learning rate for AdamW")
    parser.add_argument("--output_dir", type=Path, default=Path("models"), help="Directory to save the best model")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(exist_ok=True)

    # 在日誌文件名中包含 seed，方便區分不同實驗
    log_filename = f"training_log_seed_{args.seed}.csv"
    log_path = args.output_dir / log_filename

    print(f"DEVICE is set to: {DEVICE}")
    print(f"訓練參數: {vars(args)}")

    # --- 1. 準備資料 ---
    gt_path = args.data_root / "train" / "gt.txt"
    img_dir = args.data_root / "train" / "img"

    full_annotations = pd.read_csv(gt_path, header=None, names=["frame", "bb_left", "bb_top", "bb_width", "bb_height"])

    # ✨ 執行資料清洗
    annotations = filter_annotations(full_annotations)

    # 找出實際存在且有標註的圖片 frames
    existing_files = {int(p.stem) for p in img_dir.glob("*.jpg") if p.stem.isdigit()}
    annotated_frames = set(map(int, annotations["frame"].unique()))
    valid_frames = sorted(list(existing_files.intersection(annotated_frames)))

    if len(valid_frames) < 2:
        raise RuntimeError("可用影像不足以切分 train/val，請檢查資料完整性。")

    # 可重現的 train/val 切分
    rng = random.Random(args.seed)
    rng.shuffle(valid_frames)
    split_point = int(0.8 * len(valid_frames))
    train_frames = valid_frames[:split_point]
    val_frames = valid_frames[split_point:]

    train_dataset = PigDataset(args.data_root, train_frames, is_train=True, transforms=get_transform(train=True))
    val_dataset = PigDataset(args.data_root, val_frames, is_train=True, transforms=get_transform(train=False))

    # --- 2. 建立 DataLoader ---
    # 伺服器環境下可適度增加 num_workers
    num_workers = min(int(os.cpu_count() * 0.75), 12)
    g = torch.Generator().manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=num_workers > 0,
    )
    print(f"訓練集: {len(train_dataset)} | 驗證集: {len(val_dataset)} | DataLoader Workers: {num_workers}")

    # --- 3. 建立模型與優化器 ---
    model = create_model(NUM_CLASSES).to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]

    # ✨ 使用 AdamW 優化器
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0005)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)

    # --- 4. 訓練迴圈 ---
    best_map = -1.0
    best_model_filename = f"best_model_seed_{args.seed}.pth"
    best_path = args.output_dir / best_model_filename

    # set of epochs to save intermediate checkpoints
    checkpoint_epochs = {40, 80, 120}

    with open(log_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "mAP_50:95", "AP_50", "Seed"])

    print("\n--- 開始訓練 ---")
    for epoch in range(args.epochs):
        train_one_epoch(model, optimizer, train_loader, DEVICE, epoch)
        lr_scheduler.step()

        coco_evaluator = evaluate(model, val_loader, DEVICE)
        # coco_evaluator.coco_eval['bbox'].stats 是一個 numpy array
        # [mAP@.5:.95, AP@.5, AP@.75, mAP_small, mAP_medium, mAP_large, ...]
        stats = coco_evaluator.coco_eval["bbox"].stats
        current_map = stats[0]
        current_ap50 = stats[1]

        with open(log_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, f"{current_map:.4f}", f"{current_ap50:.4f}", args.seed])

        if current_map > best_map:
            best_map = current_map
            torch.save(model.state_dict(), best_path)
            print(f"🎉 New best model saved to {best_path} with mAP: {best_map:.4f} at epoch {epoch + 1}")

        # 在特定 epoch 保存 checkpoint model，避免 overfitting
        if (epoch + 1) in checkpoint_epochs:
            if best_path.exists():
                checkpoint_path = args.output_dir / f"best_model_seed_{args.seed}_upto_epoch_{epoch + 1}.pth"
                shutil.copy2(best_path, checkpoint_path)
                print(f"✅ Checkpoint saved: Current best model copied to {checkpoint_path}")

    print(f"\n--- 訓練完成 ---\nBest mAP: {best_map:.4f} saved at {best_path}")


if __name__ == "__main__":
    main()
