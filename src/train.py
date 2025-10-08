# src/train.py - Optimized for ConvNeXt-Tiny

import argparse
import csv
import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from src.dataset import PigDataset
from src.engine import evaluate, train_one_epoch
from src.model import create_model
from src.transforms import get_transform
from src.utils import collate_fn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 2


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(_):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def filter_annotations(df: pd.DataFrame) -> pd.DataFrame:
    initial_count = len(df)
    print(f"Original number of annotations: {initial_count}")
    df_filtered = df[(df["bb_width"] > 1) & (df["bb_height"] > 1)].copy()
    removed_count = initial_count - len(df_filtered)
    if removed_count > 0:
        print(f"Removed {removed_count} invalid annotations (w/h <= 1).")
    print(f"Filtered number of annotations: {len(df_filtered)}")
    return df_filtered


def main():
    parser = argparse.ArgumentParser(description="Pig Detection Training Script (ConvNeXt-Tiny Optimized)")
    parser.add_argument("--data_root", type=Path, default=Path("./data"))
    parser.add_argument("--epochs", type=int, default=50, help="Increased for ConvNeXt (was 30)")
    parser.add_argument("--batch_size", type=int, default=8)
    # 🔥 KEY CHANGE: Lower learning rate for ConvNeXt
    parser.add_argument("--lr", type=float, default=0.0001, help="Reduced from 0.005 for ConvNeXt")
    parser.add_argument("--output_dir", type=Path, default=Path("models"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_epochs", type=int, nargs="+", default=[])
    # 🆕 NEW: Add backbone type to config
    parser.add_argument("--backbone", type=str, default="convnext_tiny", help="Backbone architecture")
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(exist_ok=True)

    config_filename = f"config_seed_{args.seed}.json"
    config_path = args.output_dir / config_filename
    args_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    with open(config_path, "w") as f:
        json.dump(args_dict, f, indent=4)
    print(f"✅ Experiment configuration saved to {config_path}")

    log_filename = f"training_log_seed_{args.seed}.csv"
    log_path = args.output_dir / log_filename

    print(f"🎯 DEVICE: {DEVICE}")
    print(f"🏗️  BACKBONE: {args.backbone}")
    print(f"📊 Training parameters: {vars(args)}")

    # --- Data Preparation ---
    gt_path = args.data_root / "train" / "gt.txt"
    img_dir = args.data_root / "train" / "img"

    full_annotations = pd.read_csv(gt_path, header=None, names=["frame", "bb_left", "bb_top", "bb_width", "bb_height"])
    annotations = filter_annotations(full_annotations)

    existing_files = {int(p.stem) for p in img_dir.glob("*.jpg") if p.stem.isdigit()}
    annotated_frames = set(map(int, annotations["frame"].unique()))
    valid_frames = sorted(list(existing_files.intersection(annotated_frames)))

    if len(valid_frames) < 2:
        raise RuntimeError("Not enough valid images for train/val splits.")

    rng = random.Random(args.seed)
    rng.shuffle(valid_frames)
    split_point = int(0.8 * len(valid_frames))
    train_frames = valid_frames[:split_point]
    val_frames = valid_frames[split_point:]

    train_dataset = PigDataset(args.data_root, train_frames, is_train=True, transforms=get_transform(train=True))
    val_dataset = PigDataset(args.data_root, val_frames, is_train=True, transforms=get_transform(train=False))

    # --- DataLoaders ---
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
    print(f"📚 Training: {len(train_dataset)} | Validation: {len(val_dataset)} | Workers: {num_workers}")

    # --- Model & Optimizer ---
    model = create_model(NUM_CLASSES).to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]

    # 🔥 KEY CHANGE: Higher weight decay for ConvNeXt
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.05)  # Increased from 0.0005

    # 🔥 KEY CHANGE: Adjusted warmup for lower learning rate
    warmup_epochs = 10
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,  # Changed from 0.01 to 0.1
        total_iters=warmup_epochs,
    )

    main_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs - warmup_epochs,
        eta_min=1e-6,  # Added minimum learning rate
    )

    lr_scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs])

    # --- Training Loop ---
    best_map = -1.0
    patience = 20  # 20 epochs 沒有改善就考慮停止
    epochs_no_improve = 0

    best_model_filename = f"best_model_seed_{args.seed}.pth"
    best_path = args.output_dir / best_model_filename

    checkpoint_epochs = set(args.checkpoint_epochs)

    header = [
        "epoch",
        "mAP_50:95",
        "AP_50",
        "AP_75",
        "train_loss",
        "val_loss",
        "train_loss_box_reg",
        "val_loss_box_reg",
        "lr",
        "duration_s",
    ]
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    print("\n🚀 Starting Training with ConvNeXt-Tiny ---")
    print(f"⚙️  Learning Rate: {args.lr} (optimized for ConvNeXt)")
    print("⚙️  Weight Decay: 0.05 (higher for ConvNeXt)")
    print(f"⚙️  Warmup: {warmup_epochs} epochs\n")

    for epoch in range(args.epochs):
        start_time = time.time()

        train_losses = train_one_epoch(model, optimizer, train_loader, DEVICE, epoch)
        current_lr = optimizer.param_groups[0]["lr"]
        lr_scheduler.step()

        coco_evaluator, val_losses = evaluate(model, val_loader, DEVICE)

        end_time = time.time()
        epoch_duration = end_time - start_time

        stats = coco_evaluator.coco_eval["bbox"].stats

        log_data = [
            epoch + 1,
            f"{stats[0]:.4f}",
            f"{stats[1]:.4f}",
            f"{stats[2]:.4f}",
            f"{train_losses.get('total_loss', -1):.4f}",
            f"{val_losses.get('val_total_loss', -1):.4f}",
            f"{train_losses.get('loss_box_reg', -1):.4f}",
            f"{val_losses.get('val_loss_box_reg', -1):.4f}",
            f"{current_lr:.6f}",
            f"{epoch_duration:.2f}",
        ]

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(log_data)

        current_map = stats[0]
        if current_map > best_map:
            best_map = current_map
            epochs_no_improve = 0  # 重置計數器
            torch.save(model.state_dict(), best_path)
            print(f"🎉 New best mAP: {best_map:.4f} at epoch {epoch + 1}")
        else:
            epochs_no_improve += 1
            print(f"⏳ No improvement for {epochs_no_improve} epochs (best: {best_map:.4f})")

            # 早停檢查
        if epochs_no_improve >= patience:
            print(f"\n⚠️ Early stopping triggered at epoch {epoch + 1}")
            print(f"   Best mAP: {best_map:.4f} was at epoch {epoch + 1 - epochs_no_improve}")
            break

        if (epoch + 1) in checkpoint_epochs:
            if best_path.exists():
                checkpoint_path = args.output_dir / f"best_model_seed_{args.seed}_epoch_{epoch + 1}.pth"
                shutil.copy2(best_path, checkpoint_path)
                print(f"💾 Checkpoint saved: {checkpoint_path}")

    print(f"\n✅ Training Complete! Best mAP: {best_map:.4f}")


if __name__ == "__main__":
    main()
