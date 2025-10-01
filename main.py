# main.py

import os
import subprocess
import sys

# --- 全域設定 ---
NUM_EPOCHS = 30
BATCH_SIZE = 4
LEARNING_RATE = 0.005
CONF_THRESHOLD = 0.5


def run_command(command):
    """執行 shell 指令，如果出錯則終止程式。"""
    print(f"--- 執行指令: {command} ---")
    try:
        if command.startswith("pip"):
            command = f"{sys.executable} -m {command}"

        # check=True 會在指令失敗時自動拋出異常
        subprocess.run(command, check=True, shell=True, text=True)

    except subprocess.CalledProcessError as e:
        print(f"\n❌ 指令 '{e.cmd}' 執行失敗，返回碼: {e.returncode}")
        sys.exit(1)


def main():
    print("🚀 ========== 開始執行 CVPDL HW1 完整流程 ========== 🚀")

    # --- 步驟 1: 環境設定 ---
    print("\n[步驟 1/4] 正在安裝所需套件...")
    run_command("pip install pandas opencv-python tqdm pycocotools -q")
    print("✅ 套件安裝完成。")

    # --- 步驟 2: 資料準備 ---
    print("\n[步驟 2/4] 正在準備資料集...")
    if not os.path.exists("/content/data"):
        run_command("kaggle competitions download -c ntu-cvpdl-2025-hw-1 -p /content/")
        run_command("mkdir -p /content/data")
        run_command("unzip -q /content/ntu-cvpdl-2025-hw-1.zip -d /content/data")
        run_command("rm /content/ntu-cvpdl-2025-hw-1.zip")
    else:
        print("資料夾 /content/data 已存在，跳過下載步驟。")
    print("✅ 資料集準備完畢。")

    # --- 步驟 3: 模型訓練 ---
    print("\n[步驟 3/4] 正在啟動模型訓練...")
    train_command = f"python -m src.train --epochs {NUM_EPOCHS} --batch_size {BATCH_SIZE} --lr {LEARNING_RATE}"
    run_command(train_command)
    print("✅ 模型訓練完成。")

    # --- 步驟 4: 產生提交檔案 ---
    print("\n[步驟 4/4] 正在使用最佳模型進行預測...")
    best_model_path = "best_model.pth"
    if os.path.exists(best_model_path):
        predict_command = f"python -m src.predict --model_path {best_model_path} --conf_threshold {CONF_THRESHOLD}"
        run_command(predict_command)
        print("✅ 預測完成！提交檔案已儲存至 submission.csv。")
    else:
        print(f"⚠️ 找不到 '{best_model_path}'，跳過預測步驟。")

    print("\n🎉🎉🎉 ========== 所有流程執行完畢！ ========== 🎉🎉🎉")


if __name__ == "__main__":
    main()
