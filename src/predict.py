# src/predict.py (argparse 修正版)

import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# 從 src 引入我們的模組
from src.dataset import PigDataset
from src.model import create_model
from src.transforms import get_transform
from src.utils import collate_fn

# --- 全域設定 (Global Settings) ---
# 這些是在腳本內部不太會改變的設定
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
DATA_ROOT = "/content/data"
NUM_CLASSES = 2  # 必須和訓練時一樣


def main():
    # ==================================
    # 1. 設定參數解析器 (Argument Parser)
    # ==================================
    parser = argparse.ArgumentParser(description="Pig Detection Prediction Script")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./models/best_model.pth",
        help="Path to the trained model weights (.pth file)",
    )
    parser.add_argument(
        "--output_path", type=str, default="submission.csv", help="Path to save the submission csv file"
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=0.5,
        help="Confidence threshold to filter out low-confidence predictions",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,  # 預測時通常可以用比訓練時更大的 batch size
        help="Batch size for prediction",
    )
    args = parser.parse_args()

    # ==================================
    # 2. 建立測試集 Dataset 和 DataLoader
    # ==================================
    test_dataset = PigDataset(root_dir=DATA_ROOT, frame_ids=None, is_train=False, transforms=get_transform(train=False))

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # ==================================
    # 3. 載入模型
    # ==================================
    model = create_model(num_classes=NUM_CLASSES)
    # 從解析出的參數載入模型路徑
    model.load_state_dict(torch.load(args.model_path))
    model.to(DEVICE)
    model.eval()  # **非常重要**：將模型設置為評估模式！

    results = []

    print(f"--- 開始預測測試集 (Confidence Threshold: {args.conf_threshold}) ---")
    for images, targets in tqdm(test_loader):
        images = list(img.to(DEVICE) for img in images)

        with torch.no_grad():
            outputs = model(images)

        outputs = [{k: v.to("cpu") for k, v in t.items()} for t in outputs]

        for i, output in enumerate(outputs):
            image_id = targets[i]["image_id"].item()
            prediction_string = ""

            for score, box, label in zip(output["scores"], output["boxes"], output["labels"]):
                # 使用解析出的參數來過濾
                if score > args.conf_threshold and label.item() == 1:
                    x = box[0].item()
                    y = box[1].item()
                    w = (box[2] - box[0]).item()
                    h = (box[3] - box[1]).item()
                    prediction_string += f"{score.item():.4f} {x:.2f} {y:.2f} {w:.2f} {h:.2f} 0 "

            prediction_string = prediction_string.strip()
            results.append({"Image_ID": image_id, "PredictionString": prediction_string})

    # ==================================
    # 4. 寫入 CSV 檔案
    # ==================================
    submission_df = pd.DataFrame(results)
    submission_df["Image_ID"] = submission_df["Image_ID"].apply(lambda x: f"{x:08d}")
    # 從解析出的參數設定輸出路徑
    submission_df.to_csv(args.output_path, index=False)

    print(f"\n✅ 預測完成！提交檔案已儲存至 {args.output_path}")
    print("前5行預覽：")
    print(submission_df.head())


if __name__ == "__main__":
    main()
