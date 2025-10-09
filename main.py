# main.py

import argparse
import random
import subprocess
import sys
from pathlib import Path

# ===================================================================
# ✨ Default Experiment Configuration ✨
# ===================================================================
# This dictionary now holds the configuration for our best-performing,
# high-resolution strategy.

CONFIG = {
    # --- Training Configuration ---
    "epochs": 200,
    "batch_size": 2,  # Physical batch size (small for VRAM)
    "accumulation_steps": 4,  # Effective batch size = 2 * 4 = 8
    "lr": 0.0001,
    "seed": None,  # Let the script generate a random one if not provided
    "use_cluster_aware": True,
    "checkpoint_epochs": [110, 120, 130, 140, 150, 160, 170, 180, 190, 200],
    # --- Prediction / Post-processing Configuration ---
    "conf_threshold": 0.01,  # Optimal for high-recall mAP submission
    "post_processing": "none",  # Options: "none", "nms", "soft_nms"
    "nms_iou_threshold": 0.45,
    "soft_nms_sigma": 0.5,
    # --- Path Configuration ---
    "data_root": Path("./data"),
    "output_dir": Path("./models"),
    "submission_dir": Path("./submissions"),
}
# ===================================================================


def run_command(cmd_list):
    """Executes a command and exits if it fails."""
    # Convert all parts of the command to string for subprocess
    cmd_str_list = [str(item) for item in cmd_list]

    print(f"\n▶️  Executing: {' '.join(cmd_str_list)}")
    try:
        subprocess.run(cmd_str_list, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with exit code {e.returncode}:\n   {' '.join(e.cmd)}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="CVPDL HW1 Main Runner: A one-key script to train and predict.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Dynamically create parser arguments from the CONFIG dictionary
    for key, value in CONFIG.items():
        if isinstance(value, bool):
            # Handle boolean flags correctly
            parser.add_argument(f"--{key}", action="store_true", default=value)
        elif isinstance(value, list):
            parser.add_argument(f"--{key}", type=type(value[0]), nargs="+", default=value)
        else:
            parser.add_argument(f"--{key}", type=type(value), default=value)

    args = parser.parse_args()

    # --- Setup ---
    if args.seed is None:
        args.seed = random.randint(0, 2**32 - 1)
        print(f"🌱 No seed provided. Generated a random seed: {args.seed}")

    args.output_dir.mkdir(exist_ok=True)
    args.submission_dir.mkdir(exist_ok=True)

    # Dynamic submission filename based on seed and post-processing method
    submission_filename = f"submission_seed_{args.seed}_pp_{args.post_processing}.csv"
    submission_path = args.submission_dir / submission_filename

    print("🚀 CVPDL HW1 | Starting Full Run")
    print("-" * 50)
    print("Configuration:")
    for key, value in vars(args).items():
        print(f"  - {key}: {value}")
    print(f"  - submission_path: {submission_path}")  # Show the auto-generated path
    print("-" * 50)

    # --- 1. Training ---
    print("\n[1/2] 🚀 Starting Training...")
    train_cmd = [
        sys.executable,
        "-m",
        "src.train",
        "--data_root",
        args.data_root,
        "--output_dir",
        args.output_dir,
        "--epochs",
        args.epochs,
        "--batch_size",
        args.batch_size,
        "--accumulation_steps",
        args.accumulation_steps,
        "--lr",
        args.lr,
        "--seed",
        args.seed,
        "--checkpoint_epochs",
        *args.checkpoint_epochs,
    ]
    if args.use_cluster_aware:
        train_cmd.append("--use_cluster_aware")

    run_command(train_cmd)
    print("✅ Training complete.")

    # --- 2. Prediction ---
    print("\n[2/2] 🔍 Starting Prediction...")
    best_model_path = args.output_dir / f"best_model_seed_{args.seed}.pth"
    if not best_model_path.is_file():
        raise FileNotFoundError(f"Best model not found at {best_model_path}. Please check if training was successful.")

    predict_cmd = [
        sys.executable,
        "-m",
        "src.predict",
        "--data_root",
        args.data_root,
        "--model_path",
        best_model_path,
        "--output_path",
        submission_path,
        "--conf_threshold",
        args.conf_threshold,
        "--seed",
        args.seed,
    ]
    # Logic to add post-processing arguments
    if args.post_processing == "none":
        predict_cmd.append("--no-nms")
    elif args.post_processing == "nms":
        predict_cmd.append("--nms_iou_threshold")
        predict_cmd.append(args.nms_iou_threshold)
    elif args.post_processing == "soft_nms":
        predict_cmd.append("--use_soft_nms")
        predict_cmd.append("--nms_iou_threshold")
        predict_cmd.append(args.nms_iou_threshold)
        predict_cmd.append("--soft_nms_sigma")
        predict_cmd.append(args.soft_nms_sigma)

    run_command(predict_cmd)
    print(f"\n🎉 All done! Submission file saved to {submission_path}")


if __name__ == "__main__":
    main()
