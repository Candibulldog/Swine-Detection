# main.py

import argparse
import random
import subprocess
import sys
from pathlib import Path

# ===================================================================
# ✨ Execute configuration V2 (Optimized for Higher Performance) ✨
# ===================================================================

USER_DEFAULTS = {
    "epochs": 200,
    "batch_size": 8,
    "lr": 0.0001,
    "seed": None,
    "checkpoint_epochs": [100, 120, 150, 160, 170, 180, 190, 200],
    "conf_threshold": 0.3,
    "use_cluster_aware": True,
    # --- 預測後處理優化 ---
    "use_soft_nms": False,
    "use_nms": False,
    "nms_iou_threshold": 0.8,
    "soft_nms_sigma": 0.5,
    "soft_nms_min_score": 0.3,
    # --- 路徑設定 ---
    "data_root": Path("./data"),
    "output_dir": Path("./models"),
    "submission_path": None,
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
        # ✨ 修改 argparse 邏輯以支持動態檔名 ✨
        if key == "submission_path":
            parser.add_argument(
                f"--{key}", type=Path, default=value, help="Path to save submission file. (Auto-generated if not set)"
            )
        else:
            arg_type = type(value) if not isinstance(value, bool) else lambda x: bool(strtobool(x))
            parser.add_argument(f"--{key}", type=arg_type, default=value, help=f"Override default {key}")

    args = parser.parse_args()

    if args.seed is None:
        args.seed = random.randint(0, 2**32 - 1)
        print(f"INFO: No seed provided. Generated a random seed: {args.seed}")

    # --- ✨ 1. 建立 submissions 資料夾並動態生成檔名 ✨ ---
    # 確保 submissions 資料夾存在
    submissions_dir = Path("./submissions")
    submissions_dir.mkdir(exist_ok=True)

    # 如果使用者沒有從命令列手動指定 submission_path，則根據 seed 動態生成
    if args.submission_path is None:
        args.submission_path = submissions_dir / f"submission_seed_{args.seed}.csv"

    # 建立模型輸出路徑
    args.output_dir.mkdir(exist_ok=True)

    print("🚀 CVPDL HW1 | 訓練並預測 (Optimized Run)")
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

    if args.use_cluster_aware:
        train_cmd.append("--use_cluster_aware")

    run_command(list(map(str, train_cmd)))
    print("✅ 訓練完成。")

    # --- 2. 推論 ---
    print("\n[2/2] 🔍 開始推論...")

    # 動態構建模型路徑
    best_model_filename = f"best_model_seed_{args.seed}.pth"
    best_model_path = args.output_dir / best_model_filename

    if not best_model_path.is_file():
        raise FileNotFoundError(f"找不到最佳模型: {best_model_path} (請確認訓練是否成功存檔)")

    # ✨ 2. 將所有優化後的參數傳遞給 predict.py ✨
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
        "--nms_iou_threshold",
        args.nms_iou_threshold,
        "--soft_nms_sigma",
        args.soft_nms_sigma,
        "--soft_nms_min_score",
        args.soft_nms_min_score,
    ]
    if args.use_soft_nms:
        predict_cmd.append("--use_soft_nms")
    if args.use_nms:
        predict_cmd.append("--use_nms")

    run_command(list(map(str, predict_cmd)))
    print(f"\n🎉 全部完成！提交檔案已儲存至 {args.submission_path}")


if __name__ == "__main__":
    main()
