# main.py

import argparse
import os
import subprocess
import sys

# ==== User config ====
USER_DEFAULTS = {
    "epochs": 1,
    "batch_size": 4,
    "lr": 0.005,
    "conf_threshold": 0.5,
    # None 代表自動偵測：Colab -> /content/data；否則 ./data
    "data_root": None,
    "output_dir": "models",
    "best_model_path": "models/best_model.pth",
    "submission_path": "submission.csv",
}
# =========================================


def run(cmd_list):
    subprocess.run(cmd_list, check=True)


def parse_args():
    p = argparse.ArgumentParser(description="CVPDL HW1 minimal runner")
    # 所有參數 default=None，實際值用 USER_DEFAULTS 合併
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--conf_threshold", type=float, default=None)
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--best_model_path", type=str, default=None)
    p.add_argument("--submission_path", type=str, default=None)
    return p.parse_args()


def resolve_config(args):
    cfg = dict(USER_DEFAULTS)
    for k, v in vars(args).items():
        if v is not None:
            cfg[k] = v
    # 自動偵測 data_root
    if cfg["data_root"] is None:
        cfg["data_root"] = "/content/data" if os.path.exists("/content") else "./data"
    return cfg


def main():
    args = parse_args()
    cfg = resolve_config(args)

    # 顯示本次生效設定（方便對照）
    print("🚀 CVPDL HW1 | Train → Predict")
    print("有效設定：", {k: cfg[k] for k in sorted(cfg)})

    # 1) 訓練
    print("\n[1/2] 訓練中…")
    train_cmd = [
        sys.executable,
        "-m",
        "src.train",
        "--data_root",
        cfg["data_root"],
        "--epochs",
        str(cfg["epochs"]),
        "--batch_size",
        str(cfg["batch_size"]),
        "--lr",
        str(cfg["lr"]),
        "--output_dir",
        cfg["output_dir"],
    ]
    run(train_cmd)
    print("✅ 訓練完成。")

    # 2) 推論
    print("\n[2/2] 推論中…")
    if not os.path.isfile(cfg["best_model_path"]):
        raise FileNotFoundError(f"找不到最佳模型：{cfg['best_model_path']}（請確認訓練是否成功存檔）")

    predict_cmd = [
        sys.executable,
        "-m",
        "src.predict",
        "--data_root",
        cfg["data_root"],
        "--model_path",
        cfg["best_model_path"],
        "--conf_threshold",
        str(cfg["conf_threshold"]),
        "--output_path",
        cfg["submission_path"],
    ]
    run(predict_cmd)
    print(f"✅ 推論完成 → {cfg['submission_path']}")


if __name__ == "__main__":
    main()
