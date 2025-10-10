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
    "batch_size": 6,
    "accumulation_steps": 2,
    "lr": 0.0001,
    "seed": None,
    "use_cluster_aware": True,  # This flag will trigger the cluster generation
    "checkpoint_epochs": [110, 120, 130, 140, 150, 160, 170, 180, 190, 200],
    # --- Prediction / Post-processing Configuration ---
    "conf_threshold": 0.01,
    "post_processing": "none",
    "nms_iou_threshold": 0.7,
    "soft_nms_sigma": 0.5,
    # --- Path Configuration ---
    "data_root": Path("./data"),
    "output_dir": Path("./models"),
    "submission_dir": Path("./submissions"),
    "cluster_csv_path": Path("./image_clusters.csv"),
}
# ===================================================================


def run_command(cmd_list: list):
    """Executes a command and exits if it fails."""
    cmd_str_list = [str(item) for item in cmd_list]
    print(f"\n▶️  Executing: {' '.join(cmd_str_list)}")
    try:
        # We use sys.executable to ensure we're using the same Python interpreter
        # that is running main.py. This resolves module path issues.
        subprocess.run([sys.executable, "-m", *cmd_str_list], check=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with exit code {e.returncode}:\n   {' '.join(e.cmd)}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"❌ Error: Command '{cmd_str_list[0]}' not found. Is it in your PATH?")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="CVPDL HW1 Main Runner: A one-key script to train and predict.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Dynamically create parser arguments from the CONFIG dictionary
    for key, value in CONFIG.items():
        # Handle Path objects correctly
        arg_type = type(value)
        if isinstance(value, Path):
            arg_type = str

        if isinstance(value, bool):
            # Special handling for boolean flags to allow --no- flag
            parser.add_argument(f"--{key}", action=argparse.BooleanOptionalAction, default=value)
        elif isinstance(value, list):
            parser.add_argument(f"--{key}", type=type(value[0]), nargs="+", default=value)
        else:
            parser.add_argument(f"--{key}", type=arg_type, default=value)

    args = parser.parse_args()

    # Convert string paths back to Path objects after parsing
    for key, value in CONFIG.items():
        if isinstance(value, Path):
            setattr(args, key, Path(getattr(args, key)))

    # --- Setup ---
    if args.seed is None:
        args.seed = random.randint(0, 2**32 - 1)
        print(f"🌱 No seed provided. Generated a random seed: {args.seed}")

    args.output_dir.mkdir(exist_ok=True)
    args.submission_dir.mkdir(exist_ok=True)

    submission_filename = f"submission_seed_{args.seed}_pp_{args.post_processing}.csv"
    submission_path = args.submission_dir / submission_filename

    print("🚀 CVPDL HW1 | Starting Full Run")
    print("-" * 50)
    print("Configuration:")
    for key, value in vars(args).items():
        print(f"  - {key}: {value}")
    print(f"  - submission_path: {submission_path}")
    print("-" * 50)

    # --- 1. (Conditional) Data Clustering ---
    if args.use_cluster_aware:
        print("\n[Step 0/3] 🖼️  Checking for image cluster data...")
        if not args.cluster_csv_path.is_file():
            print(f"   - Cluster file '{args.cluster_csv_path}' not found. Generating it now...")
            cluster_cmd = [
                "scripts.build_clusters",
                "--image_dir",
                args.data_root / "train" / "img",
                "--output_csv",
                args.cluster_csv_path,
            ]
            run_command(cluster_cmd)
            print("   - ✅ Cluster file generated successfully.")
        else:
            print(f"   - ✅ Found existing cluster file: '{args.cluster_csv_path}'. Skipping generation.")

    # --- 2. Training ---
    print("\n[Step 1/3] 🚀 Starting Training...")
    train_cmd = [
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
        train_cmd.extend(["--use_cluster_aware", "--cluster_csv_path", args.cluster_csv_path])

    run_command(train_cmd)
    print("✅ Training complete.")

    # --- 3. Prediction ---
    print("\n[Step 2/3] 🔍 Starting Prediction...")
    best_model_path = args.output_dir / f"best_model_seed_{args.seed}.pth"
    if not best_model_path.is_file():
        raise FileNotFoundError(f"Best model not found at {best_model_path}. Please check if training was successful.")

    predict_cmd = [
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

    # Logic for post-processing arguments
    if args.post_processing == "nms":
        predict_cmd.extend(["--post_processing", "nms", "--nms_iou_threshold", args.nms_iou_threshold])
    elif args.post_processing == "soft_nms":
        predict_cmd.extend(
            [
                "--post_processing",
                "soft_nms",
                "--nms_iou_threshold",
                args.nms_iou_threshold,
                "--soft_nms_sigma",
                args.soft_nms_sigma,
            ]
        )

    run_command(predict_cmd)
    print(f"\n🎉 All done! Submission file saved to {submission_path}")


if __name__ == "__main__":
    main()
