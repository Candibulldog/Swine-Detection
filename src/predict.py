# src/predict.py

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# 從 src 引入我們的模組
from src.dataset import PigDataset
from src.model import create_model
from src.transforms import get_transform  # 我們需要用它來做 ToTensor
from src.utils import collate_fn  # DataLoader 仍然需要它

# --- 設定 ---
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
DATA_ROOT = "/content/data"
MODEL_PATH = "best_model.pth"  # 假設最佳模型儲存在這裡
OUTPUT_PATH = "submission.csv"
CONF_THRESHOLD = 0.5  # 只保留置信度高於 0.5 的預測框


def main():
    # --- 1. 建立測試集 Dataset 和 DataLoader ---
    # 測試集不需要 frame_ids 列表，它會自動讀取所有圖片
    # is_train=False 告訴 Dataset 進入測試模式
    # transforms=get_transform(train=False) 只會做 ToTensor
    test_dataset = PigDataset(root_dir=DATA_ROOT, frame_ids=None, is_train=False, transforms=get_transform(train=False))

    test_loader = DataLoader(
        test_dataset,
        batch_size=4,  # 可以根據你的 GPU 記憶體調整
        shuffle=False,  # 測試集不需要打亂
        collate_fn=collate_fn,
    )

    # --- 2. 載入模型 ---
    # NUM_CLASSES 必須和訓練時一樣
    model = create_model(num_classes=2)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(DEVICE)
    model.eval()  # **非常重要**：將模型設置為評估模式！

    # --- 3. 執行預測 ---
    results = []

    print("--- 開始預測測試集 ---")
    for images, targets in tqdm(test_loader):
        images = list(img.to(DEVICE) for img in images)

        with torch.no_grad():  # 在這個 block 中不計算梯度，節省記憶體和時間
            outputs = model(images)

        # 將預測結果從 GPU 移回 CPU
        outputs = [{k: v.to("cpu") for k, v in t.items()} for t in outputs]

        # 整理 image_id 和對應的 output
        for i, output in enumerate(outputs):
            image_id = targets[i]["image_id"].item()

            prediction_string = ""
            # 遍歷這個圖片的所有預測框
            for score, box, label in zip(output["scores"], output["boxes"], output["labels"]):
                # 過濾掉低置信度的預測和背景類別
                if score > CONF_THRESHOLD and label.item() == 1:
                    # 座標格式轉換：[xmin, ymin, xmax, ymax] -> [x, y, w, h]
                    x = box[0].item()
                    y = box[1].item()
                    w = (box[2] - box[0]).item()
                    h = (box[3] - box[1]).item()

                    # 格式化成 Kaggle 要求的字串
                    # <conf> <bb_left> <bb_top> <bb_width> <bb_height> <class>
                    prediction_string += f"{score.item():.4f} {x:.2f} {y:.2f} {w:.2f} {h:.2f} 0 "

            # 去掉最後一個多餘的空格
            prediction_string = prediction_string.strip()

            results.append({"Image_ID": image_id, "PredictionString": prediction_string})

    # --- 4. 寫入 CSV 檔案 ---
    submission_df = pd.DataFrame(results)
    # Kaggle 的 Image_ID 是檔名（不含 .jpg），所以我們需要格式化
    submission_df["Image_ID"] = submission_df["Image_ID"].apply(lambda x: f"{x:08d}")
    submission_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n✅ 預測完成！提交檔案已儲存至 {OUTPUT_PATH}")
    print("前5行預覽：")
    print(submission_df.head())


if __name__ == "__main__":
    main()
