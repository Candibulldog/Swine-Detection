# main.py (你的專案總指揮官)

import os
import subprocess
import sys

# --- 全域設定 ---
# 這些是你可能需要調整的參數
# 你也可以把 train.py 裡的超參數移到這裡，讓設定更集中
NUM_EPOCHS = 10
BATCH_SIZE = 4
LEARNING_RATE = 0.005
CONF_THRESHOLD = 0.5


def run_command(command):
    """執行 shell 指令，如果出錯則終止程式"""
    print(f"--- 執行指令: {command} ---")
    try:
        # 使用 sys.executable 確保我們用的是當前 Python 環境的 pip
        if command.startswith("pip"):
            command = f"{sys.executable} -m {command}"

        # 將指令輸出即時顯示在螢幕上
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end="")
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)

    except subprocess.CalledProcessError as e:
        print(f"\n❌ 指令執行失敗: {command}\n錯誤碼: {e.returncode}")
        sys.exit(1)


def main():
    print("🚀 ========== 開始執行 CVPDL HW1 完整流程 ========== 🚀")

    # --- 步驟 1: 環境設定 ---
    print("\n[步驟 1/4] 正在安裝/更新所需套件...")
    # run_command("pip install -r requirement.txt -q") # requirement.txt 拼寫錯誤
    run_command("pip install -r requirements.txt -q")
    run_command("pip install pycocotools -q")
    print("✅ 套件安裝完成。")

    # --- 步驟 2: 資料準備 ---
    print("\n[步驟 2/4] 正在從 Kaggle 下載並準備資料集...")
    # 這裡假設 Kaggle API token 已經設定好
    # 我們將資料下載到 Colab 的高速臨時空間 /content/data/
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
    # 使用命令行參數傳遞
    train_command = f"python src/train.py --epochs {NUM_EPOCHS} --batch_size {BATCH_SIZE} --lr {LEARNING_RATE}"
    run_command(train_command)
    print("✅ 模型訓練完成。")

    # --- 步驟 4: 產生提交檔案 ---
    print("\n[步驟 4/4] 正在使用最佳模型進行預測...")
    best_model_path = "./models/best_model.pth"
    if os.path.exists(best_model_path):
        # 使用命令行參數來設定模型路徑和信心閾值
        predict_command = f"python src/predict.py --model_path {best_model_path} --conf_threshold {CONF_THRESHOLD}"
        run_command(predict_command)
        print("✅ 預測完成！提交檔案已儲存至 submission.csv。")
    else:
        print(f"⚠️ 找不到 '{best_model_path}'，跳過預測步驟。請檢查訓練是否成功。")

    print("\n🎉🎉🎉 ========== 所有流程執行完畢！ ========== 🎉🎉🎉")


if __name__ == "__main__":
    main()
