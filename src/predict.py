# src/predict.py

import argparse
import os
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.ops import nms
from tqdm import tqdm

from src.dataset import PigDataset
from src.model import create_model
from src.transforms import IMG_SIZE, get_transform
from src.utils import collate_fn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 2  # 1 (pig) + 1 (background)


def get_original_image_size(image_id: int, root_dir: Path) -> tuple[int, int]:
    """根據 image_id 讀取原始圖片並返回其 (寬, 高)。"""
    image_path = root_dir / "test" / "img" / f"{image_id:08d}.jpg"
    with Image.open(image_path) as img:
        return img.size


def scale_boxes_to_original(boxes: torch.Tensor, current_size: int, original_size: tuple[int, int]) -> torch.Tensor:
    """將預測的 bounding boxes 從正方形尺寸縮放回原始圖片尺寸。"""
    orig_w, orig_h = original_size

    # 由於我們使用了 LongestMaxSize + PadIfNeeded，縮放因子由最長邊決定
    scale_factor = max(orig_w, orig_h) / current_size

    # 計算填充量 (padding)
    if orig_w >= orig_h:
        pad_h = (current_size - orig_h / scale_factor) / 2
        pad_w = 0
    else:
        pad_w = (current_size - orig_w / scale_factor) / 2
        pad_h = 0

    # 1. 減去 padding
    boxes[:, [0, 2]] -= pad_w
    boxes[:, [1, 3]] -= pad_h

    # 2. 乘以縮放因子
    boxes *= scale_factor

    # 3. 裁剪到原始圖片邊界內
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=orig_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=orig_h)

    return boxes


def main():
    parser = argparse.ArgumentParser(description="Pig Detection Prediction Script")
    parser.add_argument("--data_root", type=Path, default=Path("./data"), help="Root path containing test/")
    parser.add_argument(
        "--model_path", type=Path, default=Path("models/best_model.pth"), help="Path to trained model weights"
    )
    parser.add_argument("--output_path", type=Path, default=Path("submission.csv"), help="Path to save submission csv")
    parser.add_argument("--conf_threshold", type=float, default=0.5, help="Prediction confidence threshold")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for prediction")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--nms_iou_threshold", type=float, default=0.5, help="IoU threshold for Non-Maximum Suppression"
    )
    args = parser.parse_args()

    # --- 1. 準備資料 ---
    test_dataset = PigDataset(args.data_root, frame_ids=None, is_train=False, transforms=get_transform(train=False))

    # 與 train.py 一致的 DataLoader 設定
    num_workers = min(int(os.cpu_count() * 0.75), 12)
    g = torch.Generator().manual_seed(args.seed)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        generator=g,
        persistent_workers=num_workers > 0,
    )

    # --- 2. 載入模型 ---
    model = create_model(num_classes=NUM_CLASSES)
    state = torch.load(args.model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    # --- 3. 推論與後處理 ---
    results = []
    print(f"--- Predicting on test set (Confidence Threshold: {args.conf_threshold}) ---")

    for images, targets in tqdm(test_loader):
        images_gpu = [img.to(DEVICE) for img in images]

        with torch.inference_mode():
            outputs = model(images_gpu)

        outputs_cpu = [{k: v.to("cpu") for k, v in out.items()} for out in outputs]

        for i, out in enumerate(outputs_cpu):
            image_id = targets[i]["image_id"].item()

            scores = out["scores"]
            boxes = out["boxes"]
            labels = out["labels"]

            # 1. 先過濾 'pig' (class 1)
            pig_indices = labels == 1
            boxes = boxes[pig_indices]
            scores = scores[pig_indices]

            # 2. 再過濾低信心度的預測
            conf_indices = scores > args.conf_threshold
            boxes = boxes[conf_indices]
            scores = scores[conf_indices]

            # ✨ NMS
            # nms 會返回要保留的 box 的索引
            keep_indices = nms(boxes, scores, args.nms_iou_threshold)

            boxes = boxes[keep_indices]
            scores = scores[keep_indices]

            # ✨ 將 Bbox 座標縮放回原始圖片尺寸
            if boxes.shape[0] > 0:
                original_size = get_original_image_size(image_id, args.data_root)
                boxes = scale_boxes_to_original(boxes, current_size=IMG_SIZE, original_size=original_size)

            # 格式化為 Kaggle 提交字串
            parts = []
            for score, box in zip(scores, boxes):
                x1, y1, x2, y2 = box.tolist()
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                # Kaggle 要求 class 為 0
                parts.append(f"{score.item():.4f} {x1:.2f} {y1:.2f} {w:.2f} {h:.2f} 0")

            prediction_string = " ".join(parts)
            results.append({"Image_ID": image_id, "PredictionString": prediction_string})

    # --- 4. 輸出提交檔案 ---
    df = pd.DataFrame(results)
    df.to_csv(args.output_path, index=False)

    print(f"\n✅ Prediction complete! Submission file saved to {args.output_path}")
    print("Top 5 predictions:")
    print(df.head())


if __name__ == "__main__":
    main()
