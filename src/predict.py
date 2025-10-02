# src/predict.py

import argparse
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# 從 src 引入我們的模組
from src.dataset import PigDataset
from src.model import create_model
from src.transforms import get_transform
from src.utils import collate_fn

# --- 全域設定 ---
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
NUM_CLASSES = 2  # 必須和訓練時一致（背景0、豬1）


def main():
    # =========================
    # 1) 參數
    # =========================
    parser = argparse.ArgumentParser(description="Pig Detection Prediction Script")

    default_dr = "/content/data" if os.path.exists("/content/data") else "./data"
    parser.add_argument(
        "--data_root", type=str, default=default_dr, help="Root path that contains test/ (expects test/img)"
    )

    parser.add_argument(
        "--model_path", type=str, default="models/best_model.pth", help="Path to the trained model weights (.pth file)"
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
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for prediction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for DataLoader determinism")
    args = parser.parse_args()

    # 路徑檢查
    test_img_dir = os.path.join(args.data_root, "test", "img")
    if not os.path.isdir(test_img_dir):
        raise NotADirectoryError(f"找不到測試影像資料夾：{test_img_dir}")
    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(f"找不到模型權重：{args.model_path}")

    # =========================
    # 2) Dataset / DataLoader
    # =========================
    test_dataset = PigDataset(
        root_dir=args.data_root,
        frame_ids=None,  # 由 Dataset 自行掃描 test/img
        is_train=False,
        transforms=get_transform(train=False),
    )

    # DataLoader 效能與可重現性（和 train.py 一致）
    g = torch.Generator()
    g.manual_seed(args.seed)
    cpu_cnt = os.cpu_count() or 2
    num_workers = max(1, cpu_cnt - 1)

    dl_kwargs = dict(
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        generator=g,
    )
    if num_workers > 0:
        dl_kwargs["persistent_workers"] = True
        dl_kwargs["prefetch_factor"] = 2

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **dl_kwargs)

    # =========================
    # 3) 載入模型
    # =========================
    model = create_model(num_classes=NUM_CLASSES)
    # map_location 確保在 CPU 也能載入 GPU 存的權重
    state = torch.load(args.model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    # =========================
    # 4) 推論
    # =========================
    results = []
    print(f"--- 開始預測測試集 (Confidence Threshold: {args.conf_threshold}) ---")

    # inference_mode 比 no_grad 更快更省記憶體（不可訓練）
    with torch.inference_mode():
        for images, targets in tqdm(test_loader):
            images = [img.to(DEVICE) for img in images]
            outputs = model(images)
            # 全轉回 CPU 便於後處理
            outputs = [{k: v.to("cpu") for k, v in out.items()} for out in outputs]

            for i, out in enumerate(outputs):
                # Dataset 應提供 image_id（int）
                image_id = targets[i]["image_id"].item()

                parts = []
                scores = out.get("scores", [])
                boxes = out.get("boxes", [])
                labels = out.get("labels", [])

                # torchvision 檢測模型已做 NMS，這裡只需要依閾值過濾
                for score, box, label in zip(scores, boxes, labels):
                    if score.item() >= args.conf_threshold and label.item() == 1:
                        x1, y1, x2, y2 = box.tolist()
                        w = max(0.0, x2 - x1)
                        h = max(0.0, y2 - y1)
                        # 兩位小數即可；class 固定 0（作業規範）
                        parts.append(f"{score.item():.4f} {x1:.2f} {y1:.2f} {w:.2f} {h:.2f} 0")

                prediction_string = " ".join(parts)
                results.append({"Image_ID": image_id, "PredictionString": prediction_string})

    # =========================
    # 5) 輸出提交檔
    # =========================
    df = pd.DataFrame(results, columns=["Image_ID", "PredictionString"])
    # 8 位數零補：00000001
    df["Image_ID"] = df["Image_ID"].apply(lambda x: f"{int(x):08d}")
    df.to_csv(args.output_path, index=False)

    print(f"\n✅ 預測完成！提交檔案已儲存至 {args.output_path}")
    print("前 5 行預覽：")
    print(df.head())


if __name__ == "__main__":
    main()
