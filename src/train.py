# src/train.py

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

# Set the device for computation. Prefers CUDA if available.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Define the number of classes: 1 for 'pig' + 1 for the background.
NUM_CLASSES = 2


def set_seed(seed: int):
    """
    Sets the random seed for all relevant libraries to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # These settings ensure deterministic behavior from cuDNN, which can
        # slightly impact performance but is crucial for reproducible results.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(_):
    """
    A worker_init_fn for DataLoader to ensure that each worker process
    has a unique and reproducible seed.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def filter_annotations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the annotation DataFrame by removing invalid bounding boxes.
    A valid box must have a width and height greater than 1 pixel.
    """
    initial_count = len(df)
    print(f"Original number of annotations: {initial_count}")
    # Filter out boxes where width or height is non-positive.
    df_filtered = df[(df["bb_width"] > 1) & (df["bb_height"] > 1)].copy()
    removed_count = initial_count - len(df_filtered)
    if removed_count > 0:
        print(f"Removed {removed_count} invalid annotations (w/h <= 1).")
    print(f"Filtered number of annotations: {len(df_filtered)}")
    return df_filtered


def main():
    """Main function to orchestrate the model training and evaluation pipeline."""
    parser = argparse.ArgumentParser(description="Pig Detection Training Script")
    parser.add_argument("--data_root", type=Path, default=Path("./data"), help="Root path containing train/ and test/.")
    parser.add_argument("--epochs", type=int, default=30, help="Total number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=0.005, help="Initial learning rate for the AdamW optimizer.")
    parser.add_argument(
        "--output_dir", type=Path, default=Path("models"), help="Directory to save the best model and logs."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--checkpoint_epochs",
        type=int,
        nargs="+",
        default=[],
        help="A list of epochs at which to save model checkpoints.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(exist_ok=True)

    # ✨ NEW: Save experiment configuration to a separate JSON file for clarity.
    config_filename = f"config_seed_{args.seed}.json"
    config_path = args.output_dir / config_filename
    # Convert Path objects to strings for JSON serialization.
    args_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    with open(config_path, "w") as f:
        json.dump(args_dict, f, indent=4)
    print(f"✅ Experiment configuration saved to {config_path}")

    # Include the seed in the log filename to easily distinguish between experiments.
    log_filename = f"training_log_seed_{args.seed}.csv"
    log_path = args.output_dir / log_filename

    print(f"DEVICE is set to: {DEVICE}")
    print(f"Training parameters: {vars(args)}")

    # --- 1. Prepare Data ---
    gt_path = args.data_root / "train" / "gt.txt"
    img_dir = args.data_root / "train" / "img"

    full_annotations = pd.read_csv(gt_path, header=None, names=["frame", "bb_left", "bb_top", "bb_width", "bb_height"])

    # ✨ Perform data cleaning to remove invalid ground truth boxes.
    annotations = filter_annotations(full_annotations)

    # Cross-reference annotations with actual image files to get a list of valid frames.
    existing_files = {int(p.stem) for p in img_dir.glob("*.jpg") if p.stem.isdigit()}
    annotated_frames = set(map(int, annotations["frame"].unique()))
    valid_frames = sorted(list(existing_files.intersection(annotated_frames)))

    if len(valid_frames) < 2:
        raise RuntimeError("Not enough valid images to create train/val splits. Check data integrity.")

    # Create a reproducible train/validation split (80/20).
    rng = random.Random(args.seed)
    rng.shuffle(valid_frames)
    split_point = int(0.8 * len(valid_frames))
    train_frames = valid_frames[:split_point]
    val_frames = valid_frames[split_point:]

    # Instantiate datasets with appropriate transformations.
    train_dataset = PigDataset(args.data_root, train_frames, is_train=True, transforms=get_transform(train=True))
    val_dataset = PigDataset(args.data_root, val_frames, is_train=True, transforms=get_transform(train=False))

    # --- 2. Create DataLoaders ---
    # Set num_workers based on available CPU resources for efficient data loading.
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
    print(
        f"Training set: {len(train_dataset)} | Validation set: {len(val_dataset)} | DataLoader Workers: {num_workers}"
    )

    # --- 3. Create Model and Optimizer ---
    model = create_model(NUM_CLASSES).to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]

    # ✨ Use AdamW optimizer, which is a robust choice that often performs well.
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0005)

    # ✨ Implement a learning rate scheduler with a warm-up phase followed by cosine annealing.
    warmup_epochs = 5
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,  # 從 lr * 0.01 開始
        total_iters=warmup_epochs,  # 在 5 個 epoch 內完成 warmup
    )

    main_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - warmup_epochs, eta_min=0)

    # Combine the warm-up and main schedulers using SequentialLR.
    lr_scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs])

    # --- 4. Training Loop ---
    best_map = -1.0
    best_model_filename = f"best_model_seed_{args.seed}.pth"
    best_path = args.output_dir / best_model_filename

    # Define specific epochs at which to save a snapshot of the current best model.
    checkpoint_epochs = set(args.checkpoint_epochs)

    # 📊 NEW: Define a more detailed header for the training log.
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

    print("\n--- Starting Training ---")
    for epoch in range(args.epochs):
        start_time = time.time()  # ⏱️ NEW: Record start time.

        # ✨ MODIFIED: Capture the returned dictionary of training losses.
        train_losses = train_one_epoch(model, optimizer, train_loader, DEVICE, epoch)

        # Get the current learning rate *before* the scheduler steps.
        current_lr = optimizer.param_groups[0]["lr"]

        lr_scheduler.step()

        # ✨ MODIFIED: Capture both the evaluator and the validation losses.
        coco_evaluator, val_losses = evaluate(model, val_loader, DEVICE)

        end_time = time.time()  # ⏱️ NEW: Record end time.
        epoch_duration = end_time - start_time

        stats = coco_evaluator.coco_eval["bbox"].stats

        # 📊 NEW: Prepare a comprehensive row of data for logging.
        log_data = [
            epoch + 1,
            f"{stats[0]:.4f}",  # mAP_50:95
            f"{stats[1]:.4f}",  # AP_50
            f"{stats[2]:.4f}",  # AP_75
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

        # The logic for saving the best model and checkpoints remains the same.
        current_map = stats[0]
        if current_map > best_map:
            best_map = current_map
            torch.save(model.state_dict(), best_path)
            print(f"🎉 New best model saved to {best_path} with mAP: {best_map:.4f} at epoch {epoch + 1}")

        if (epoch + 1) in checkpoint_epochs:
            if best_path.exists():
                checkpoint_path = args.output_dir / f"best_model_seed_{args.seed}_upto_epoch_{epoch + 1}.pth"
                shutil.copy2(best_path, checkpoint_path)
                print(f"✅ Checkpoint saved: Current best model copied to {checkpoint_path}")

    print(f"\n--- Training Complete ---\nBest mAP: {best_map:.4f} saved at {best_path}")


if __name__ == "__main__":
    main()
