# src/predict.py (Refactored for Performance and Clarity)

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
from src.soft_nms import soft_nms
from src.transforms import IMG_SIZE
from src.utils import collate_fn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 2


# ✨ REFACTOR 1: Pre-load all image sizes at once to avoid repeated file I/O.
def preload_image_sizes(test_dir: Path) -> dict[int, tuple[int, int]]:
    """Scans the test directory once and caches all image dimensions."""
    print("Pre-loading original image sizes...")
    image_sizes = {}
    image_paths = list(test_dir.glob("*.jpg"))
    for image_path in tqdm(image_paths, desc="Caching image sizes"):
        try:
            image_id = int(image_path.stem)
            with Image.open(image_path) as img:
                image_sizes[image_id] = img.size
        except (ValueError, FileNotFoundError):
            continue  # Ignore non-numeric or invalid filenames
    return image_sizes


def scale_boxes_to_original(boxes: torch.Tensor, current_size: int, original_size: tuple[int, int]) -> torch.Tensor:
    """
    Rescales bounding boxes from the model's square input size back to the original image dimensions.
    NOTE: This logic is tightly coupled with the transforms in `transforms.py`
          (LongestMaxSize then PadIfNeeded). If transforms change, this must be updated.
    """
    orig_w, orig_h = original_size
    scale_factor = max(orig_w, orig_h) / current_size

    pad_w, pad_h = 0, 0
    if orig_w >= orig_h:
        pad_h = (current_size - orig_h / scale_factor) / 2
    else:
        pad_w = (current_size - orig_w / scale_factor) / 2

    boxes[:, [0, 2]] -= pad_w
    boxes[:, [1, 3]] -= pad_h
    boxes *= scale_factor
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=orig_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=orig_h)
    return boxes


def main():
    """Main function to run prediction and generate a submission file."""
    parser = argparse.ArgumentParser(description="Pig Detection Prediction Script")
    parser.add_argument("--data_root", type=Path, default=Path("./data"))
    parser.add_argument("--model_path", type=Path, default=Path("models/best_model.pth"))
    parser.add_argument("--output_path", type=Path, default=Path("submission.csv"))
    parser.add_argument("--conf_threshold", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=24)  # Increased for faster inference
    parser.add_argument("--seed", type=int, default=42)

    # ✨ REFACTOR 2: Consolidate post-processing logic into a single choice.
    parser.add_argument(
        "--post_processing",
        type=str,
        default="none",
        choices=["none", "nms", "soft_nms"],
        help="Post-processing method ('none' for raw output for mAP).",
    )
    parser.add_argument("--nms_iou_threshold", type=float, default=0.45)
    parser.add_argument("--soft_nms_sigma", type=float, default=0.5)

    args = parser.parse_args()

    # --- 1. Prepare Data & Pre-load metadata ---
    test_image_dir = args.data_root / "test" / "img"
    image_sizes = preload_image_sizes(test_image_dir)
    test_dataset = PigDataset(data_root=args.data_root, frame_ids=None, is_train=False)

    # ... (DataLoader setup remains the same) ...
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
    print(f"--- Predicting on test set (Post-processing: {args.post_processing.upper()}) ---")

    for images, targets in tqdm(test_loader):
        images_gpu = [img.to(DEVICE) for img in images]
        with torch.inference_mode():
            outputs = model(images_gpu)

        outputs_cpu = [{k: v.to("cpu") for k, v in out.items()} for out in outputs]

        for i, out in enumerate(outputs_cpu):
            image_id = targets[i]["image_id"].item()
            scores, boxes, labels = out["scores"], out["boxes"], out["labels"]

            # Step 1: Filter for 'pig' class and by confidence
            pig_mask = (labels == 1) & (scores > args.conf_threshold)
            boxes = boxes[pig_mask]
            scores = scores[pig_mask]

            # ✨ REFACTOR 3: Clean, linear post-processing logic.
            if boxes.shape[0] > 0:
                if args.post_processing == "nms":
                    keep_indices = nms(boxes, scores, args.nms_iou_threshold)
                    boxes, scores = boxes[keep_indices], scores[keep_indices]
                elif args.post_processing == "soft_nms":
                    keep_indices, updated_scores = soft_nms(
                        boxes,
                        scores,
                        iou_threshold=args.nms_iou_threshold,
                        sigma=args.soft_nms_sigma,
                        method="gaussian",
                    )
                    # Note: Soft-NMS might not filter any boxes out, just lower their scores.
                    # We still re-index in case the implementation does filter.
                    boxes, scores = boxes[keep_indices], updated_scores
                # if args.post_processing == "none", we do nothing.

            # Step 4: Scale boxes and format for submission
            if boxes.shape[0] > 0:
                original_size = image_sizes[image_id]
                boxes = scale_boxes_to_original(boxes, current_size=IMG_SIZE, original_size=original_size)

            parts = []
            # ✨ REFACTOR 4: Use strict=True for robustness.
            for score, box in zip(scores, boxes, strict=True):
                x1, y1, x2, y2 = box.tolist()
                w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
                parts.append(f"{score.item():.4f} {x1:.2f} {y1:.2f} {w:.2f} {h:.2f} 0")

            prediction_string = " ".join(parts)
            results.append({"Image_ID": image_id, "PredictionString": prediction_string})

    # --- 4. Generate Submission File ---
    # ... (remains the same) ...
    df = pd.DataFrame(results)
    df.to_csv(args.output_path, index=False)
    print(f"\n✅ Prediction complete! Submission file saved to {args.output_path}")
    print("Top 5 predictions:")
    print(df.head())


if __name__ == "__main__":
    main()
