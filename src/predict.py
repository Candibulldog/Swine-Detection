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
from src.soft_nms import soft_nms  # Assuming you have a soft_nms implementation here
from src.transforms import IMG_SIZE
from src.utils import collate_fn

# Constants for clarity and maintainability
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 2  # 1 (pig) + 1 (background)
PIG_LABEL_MODEL = 1  # The model was trained with label '1' for pigs
PIG_LABEL_SUBMISSION = 0  # Kaggle requires label '0' for pigs


def preload_image_sizes(test_dir: Path) -> dict[int, tuple[int, int]]:
    """
    Scans the test directory once and caches all image dimensions in memory.
    This avoids repeated, slow file I/O operations inside the inference loop.
    """
    print("Caching original image sizes for efficient coordinate scaling...")
    image_sizes = {}
    image_paths = sorted(list(test_dir.glob("*.jpg")))
    for image_path in tqdm(image_paths, desc="Reading image dimensions"):
        try:
            image_id = int(image_path.stem)
            with Image.open(image_path) as img:
                image_sizes[image_id] = img.size  # (width, height)
        except (ValueError, FileNotFoundError):
            # Ignore non-numeric filenames or broken symlinks
            continue
    return image_sizes


def scale_boxes_to_original(boxes: torch.Tensor, current_size: int, original_size: tuple[int, int]) -> torch.Tensor:
    """
    Rescales bounding boxes from the model's square input size back to the original image dimensions.
    This function is tightly coupled with the `LongestMaxSize` and `PadIfNeeded` transforms
    used during data pre-processing. If the transformation logic changes, this must be updated.
    """
    orig_w, orig_h = original_size

    # Determine the scaling factor used to fit the longest side to `current_size`
    scale = current_size / max(orig_w, orig_h)

    # Calculate the new dimensions after resizing
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)

    # Calculate the padding that was added to make the image square
    pad_x = (current_size - new_w) // 2
    pad_y = (current_size - new_h) // 2

    # Reverse the padding and scaling operations
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale

    # Clamp coordinates to ensure they are within the original image boundaries
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=orig_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=orig_h)

    return boxes


def main():
    """Main function to run inference and generate a Kaggle submission file."""
    parser = argparse.ArgumentParser(description="Pig Detection Prediction Script")
    parser.add_argument("--data_root", type=Path, default=Path("./data"), help="Root directory of the dataset.")
    parser.add_argument("--model_path", type=Path, required=True, help="Path to the trained model checkpoint (.pth).")
    parser.add_argument(
        "--output_path", type=Path, default=Path("submission.csv"), help="Path to save the output submission CSV."
    )
    parser.add_argument(
        "--conf_threshold", type=float, default=0.01, help="Confidence threshold for filtering detections."
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for inference (can be larger than training)."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--post_processing",
        type=str,
        default="none",
        choices=["none", "nms", "soft_nms"],
        help="Post-processing method. 'none' is often best for mAP calculation.",
    )
    parser.add_argument("--nms_iou_threshold", type=float, default=0.7, help="IoU threshold for NMS.")
    parser.add_argument("--soft_nms_sigma", type=float, default=0.5, help="Sigma value for Gaussian Soft-NMS.")
    args = parser.parse_args()

    # --- 1. Prepare Data & Pre-load Metadata ---
    test_image_dir = args.data_root / "test" / "img"
    image_sizes = preload_image_sizes(test_image_dir)
    test_dataset = PigDataset(data_root=args.data_root, is_train=False)

    num_workers = min(int(os.cpu_count() or 1), 8)
    g = torch.Generator().manual_seed(args.seed)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        generator=g,
    )

    # --- 2. Load Model ---
    print(f"Loading model from {args.model_path}...")
    model = create_model(num_classes=NUM_CLASSES)
    # Use weights_only=True for security against malicious code in checkpoints
    state = torch.load(args.model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    # --- 3. Run Inference and Post-processing ---
    results = []
    print(f"\n🚀 Starting inference on {len(test_dataset)} images (Post-processing: {args.post_processing.upper()})")

    for images, targets in tqdm(test_loader, desc="Predicting"):
        images_gpu = [img.to(DEVICE) for img in images]

        # Use torch.inference_mode() as it's the fastest context for evaluation
        with torch.inference_mode():
            outputs = model(images_gpu)

        outputs_cpu = [{k: v.to("cpu") for k, v in out.items()} for out in outputs]

        for i, output in enumerate(outputs_cpu):
            image_id = targets[i]["image_id"].item()
            scores, boxes, labels = output["scores"], output["boxes"], output["labels"]

            # Step A: Filter detections by confidence and for the 'pig' class
            # CRITICAL: The model was trained with 'pig' as label 1 (0 is background).
            keep_mask = (labels == PIG_LABEL_MODEL) & (scores > args.conf_threshold)
            boxes, scores = boxes[keep_mask], scores[keep_mask]

            # Step B: Apply selected post-processing if any boxes remain
            if boxes.shape[0] > 0:
                if args.post_processing == "nms":
                    keep_indices = nms(boxes, scores, args.nms_iou_threshold)
                    boxes, scores = boxes[keep_indices], scores[keep_indices]
                elif args.post_processing == "soft_nms":
                    # soft_nms returns updated scores and the original indices
                    boxes, scores = soft_nms(
                        boxes, scores, iou_threshold=args.nms_iou_threshold, sigma=args.soft_nms_sigma
                    )
                # If 'none', we pass through all filtered boxes.

            # Step C: Scale boxes back to original image size and format for submission
            if boxes.shape[0] > 0:
                original_size = image_sizes[image_id]
                boxes = scale_boxes_to_original(boxes, current_size=IMG_SIZE, original_size=original_size)

            prediction_parts = []
            for score, box in zip(scores, boxes, strict=True):
                x1, y1, x2, y2 = box.tolist()
                # Ensure width and height are non-negative
                w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)

                # Format: <conf> <bb_left> <bb_top> <bb_width> <bb_height> <class>
                # CRITICAL: Convert model's pig label (1) to submission format label (0).
                prediction_parts.append(f"{score.item():.4f} {x1:.2f} {y1:.2f} {w:.2f} {h:.2f} {PIG_LABEL_SUBMISSION}")

            prediction_string = " ".join(prediction_parts)
            results.append({"Image_ID": image_id, "PredictionString": prediction_string})

    # --- 4. Generate Submission File ---
    submission_df = pd.DataFrame(results)
    # Ensure the submission is sorted by Image_ID, a good practice for Kaggle
    submission_df = submission_df.sort_values(by="Image_ID").reset_index(drop=True)
    submission_df.to_csv(args.output_path, index=False)

    print(f"\n✅ Prediction complete! Submission file saved to: {args.output_path}")
    print("Top 5 predictions:")
    print(submission_df.head().to_string())


if __name__ == "__main__":
    main()
