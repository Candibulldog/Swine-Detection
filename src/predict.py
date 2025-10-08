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

# ✨ 1. 導入你的 soft_nms 模塊
from src.soft_nms import soft_nms
from src.transforms import IMG_SIZE
from src.utils import collate_fn

# Set the device for computation. Prefers CUDA if available.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Define the number of classes: 1 for 'pig' + 1 for the background.
NUM_CLASSES = 2


def get_original_image_size(image_id: int, root_dir: Path) -> tuple[int, int]:
    """
    Reads an original image from the test set and returns its dimensions.
    """
    image_path = root_dir / "test" / "img" / f"{image_id:08d}.jpg"
    with Image.open(image_path) as img:
        return img.size


def scale_boxes_to_original(boxes: torch.Tensor, current_size: int, original_size: tuple[int, int]) -> torch.Tensor:
    """
    Rescales bounding boxes from the model's square input size back to the original image dimensions.
    """
    orig_w, orig_h = original_size
    scale_factor = max(orig_w, orig_h) / current_size
    if orig_w >= orig_h:
        pad_h = (current_size - orig_h / scale_factor) / 2
        pad_w = 0
    else:
        pad_w = (current_size - orig_w / scale_factor) / 2
        pad_h = 0
    boxes[:, [0, 2]] -= pad_w
    boxes[:, [1, 3]] -= pad_h
    boxes *= scale_factor
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=orig_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=orig_h)
    return boxes


def main():
    """Main function to run the prediction and generate a submission file."""
    parser = argparse.ArgumentParser(description="Pig Detection Prediction Script")
    parser.add_argument(
        "--data_root", type=Path, default=Path("./data"), help="Root path containing the 'test/' directory."
    )
    parser.add_argument(
        "--model_path", type=Path, default=Path("models/best_model.pth"), help="Path to trained model weights."
    )
    parser.add_argument(
        "--output_path", type=Path, default=Path("submission.csv"), help="Path to save the submission CSV file."
    )
    parser.add_argument("--conf_threshold", type=float, default=0.3, help="Confidence score threshold for predictions.")
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size for inference.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--nms_iou_threshold", type=float, default=0.75, help="IoU threshold for Non-Maximum Suppression."
    )

    # ✨ 2. 添加新的命令行參數 ✨
    parser.add_argument("--use_soft_nms", action="store_true", help="Use Soft-NMS instead of standard NMS.")
    parser.add_argument("--soft_nms_sigma", type=float, default=0.6, help="Sigma for Gaussian Soft-NMS.")
    parser.add_argument("--soft_nms_min_score", type=float, default=0.2, help="Minimum score threshold for Soft-NMS.")

    args = parser.parse_args()

    # --- 1. Prepare Data ---
    test_dataset = PigDataset(
        data_root=args.data_root,
        frame_ids=None,
        is_train=False,
    )
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

    # --- 2. Load Model ---
    model = create_model(num_classes=NUM_CLASSES)
    state = torch.load(args.model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    # --- 3. Run Inference and Post-processing ---
    results = []

    # ✨ 更新打印信息 ✨
    nms_method_str = "Soft-NMS" if args.use_soft_nms else "Standard NMS"
    print(
        f"--- Predicting on test set (NMS Method: {nms_method_str}) ---\n"
        f"(Confidence Threshold: {args.conf_threshold}) and (NMS IoU Threshold: {args.nms_iou_threshold})"
    )
    if args.use_soft_nms:
        print(f"(Soft-NMS Sigma: {args.soft_nms_sigma}) and (Soft-NMS Min Score: {args.soft_nms_min_score})")

    for images, targets in tqdm(test_loader):
        images_gpu = [img.to(DEVICE) for img in images]

        with torch.inference_mode():
            outputs = model(images_gpu)

        outputs_cpu = [{k: v.to("cpu") for k, v in out.items()} for out in outputs]

        for i, out in enumerate(outputs_cpu):
            image_id = targets[i]["image_id"].item()
            scores, boxes, labels = out["scores"], out["boxes"], out["labels"]

            # Step 1: Filter for the 'pig' class (label 1).
            pig_indices = labels == 1
            boxes = boxes[pig_indices]
            scores = scores[pig_indices]

            # Step 2: Filter out predictions with low confidence scores.
            conf_indices = scores > args.conf_threshold
            boxes = boxes[conf_indices]
            scores = scores[conf_indices]

            # ✨ 3. 根據參數選擇 NMS 算法 ✨
            if args.use_soft_nms:
                # 使用 Soft-NMS。它返回要保留的框的索引和它們被更新後的分數。
                keep_indices, updated_scores = soft_nms(
                    boxes,
                    scores,
                    iou_threshold=args.nms_iou_threshold,
                    sigma=args.soft_nms_sigma,
                    score_threshold=args.soft_nms_min_score,  # 使用 soft_nms 自己的閾值
                    method="gaussian",
                )
                boxes = boxes[keep_indices]
                scores = updated_scores  # ✨ 使用被衰減後的新分數
            else:
                # 使用原始的標準 NMS
                if boxes.shape[0] > 0:  # 確保有框可供 nms 處理
                    keep_indices = nms(boxes, scores, args.nms_iou_threshold)
                    boxes = boxes[keep_indices]
                    scores = scores[keep_indices]

            # Step 4: Scale the final bounding box coordinates back to the original image size.
            if boxes.shape[0] > 0:
                original_size = get_original_image_size(image_id, args.data_root)
                boxes = scale_boxes_to_original(boxes, current_size=IMG_SIZE, original_size=original_size)

            # Format the predictions into the required Kaggle submission string.
            parts = []
            for score, box in zip(scores, boxes):
                x1, y1, x2, y2 = box.tolist()
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                parts.append(f"{score.item():.4f} {x1:.2f} {y1:.2f} {w:.2f} {h:.2f} 0")

            prediction_string = " ".join(parts)
            results.append({"Image_ID": image_id, "PredictionString": prediction_string})

    # --- 4. Generate Submission File ---
    df = pd.DataFrame(results)
    df.to_csv(args.output_path, index=False)
    print(f"\n✅ Prediction complete! Submission file saved to {args.output_path}")
    print("Top 5 predictions:")
    print(df.head())


if __name__ == "__main__":
    main()
