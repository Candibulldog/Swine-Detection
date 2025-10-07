# main.py

import argparse
import random
import subprocess
import sys
from pathlib import Path

# ===================================================================
# ✨ Execute configuration ✨
# ===================================================================
USER_DEFAULTS = {
    "epochs": 120,  # 給予充分的訓練和微調時間
    "batch_size": 8,  # 可根據 VRAM 調整
    "lr": 0.0005,  # 配合 AdamW 和 CosineAnnealingLR 的較低學習率
    "seed": None,  # 確保實驗的可重現性
    "checkpoint_epochs": [70, 80, 90, 100, 110, 120],  # 在這些 epoch 保存模型檢查點
    "conf_threshold": 0.3,  # 預測時的信心度閾值，可後續調整
    # --- 路徑設定 ---
    "data_root": Path("./data"),
    "output_dir": Path("./models"),
    "submission_path": Path("submission.csv"),
}
# ===================================================================


def strtobool(val):
    """convert string to boolean (for argparse)."""
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return True
    elif val in ("n", "no", "f", "false", "off", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got '{val}'")


def run_command(cmd_list):
    """execute a command in subprocess and handle errors."""
    try:
        subprocess.run(cmd_list, check=True)
    except subprocess.CalledProcessError as e:
        print(f"命令執行失敗，返回碼 {e.returncode}:\n{' '.join(map(str, e.cmd))}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="CVPDL HW1 Runner: Train -> Predict", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # 從 USER_DEFAULTS 自動生成命令行參數
    for key, value in USER_DEFAULTS.items():
        arg_type = type(value) if not isinstance(value, bool) else lambda x: bool(strtobool(x))
        parser.add_argument(f"--{key}", type=arg_type, default=value, help=f"Override default {key}")

    args = parser.parse_args()

    # process random seed
    if args.seed is None:
        args.seed = random.randint(0, 2**32 - 1)
        print(f"INFO: No seed provided. Generated a random seed: {args.seed}")

    # 建立模型輸出路徑
    args.output_dir.mkdir(exist_ok=True)

    print("🚀 CVPDL HW1 | 訓練並預測")
    print("-" * 50)
    print("當前配置:")
    for key, value in vars(args).items():
        print(f"  - {key}: {value}")
    print("-" * 50)

    # --- 1. 訓練 ---
    print("\n[1/2] 🚀 開始訓練...")
    train_cmd = [
        sys.executable,
        "-m",
        "src.train",
        "--data_root",
        args.data_root,
        "--epochs",
        args.epochs,
        "--batch_size",
        args.batch_size,
        "--lr",
        args.lr,
        "--output_dir",
        args.output_dir,
        "--seed",
        args.seed,
    ]
    if args.checkpoint_epochs:
        train_cmd.append("--checkpoint_epochs")
        train_cmd.extend(map(str, args.checkpoint_epochs))

    run_command(map(str, train_cmd))
    print("✅ 訓練完成。")

    # --- 2. 推論 ---
    print("\n[2/2] 🔍 開始推論...")

    # 動態構建模型路徑，使其與 train.py 的輸出文件名匹配
    best_model_filename = f"best_model_seed_{args.seed}.pth"
    best_model_path = args.output_dir / best_model_filename

    if not best_model_path.is_file():
        raise FileNotFoundError(f"找不到最佳模型: {best_model_path} (請確認訓練是否成功存檔)")

    predict_cmd = [
        sys.executable,
        "-m",
        "src.predict",
        "--data_root",
        args.data_root,
        "--model_path",
        best_model_path,
        "--conf_threshold",
        args.conf_threshold,
        "--output_path",
        args.submission_path,
        "--seed",
        args.seed,
    ]
    run_command(map(str, predict_cmd))
    print(f"\n🎉 全部完成！提交檔案已儲存至 {args.submission_path}")


if __name__ == "__main__":
    main()
