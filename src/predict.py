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

# Set the device for computation. Prefers CUDA if available.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Define the number of classes: 1 for 'pig' + 1 for the background.
NUM_CLASSES = 2


def get_original_image_size(image_id: int, root_dir: Path) -> tuple[int, int]:
    """
    Reads an original image from the test set and returns its dimensions.

    Args:
        image_id (int): The frame ID of the image.
        root_dir (Path): The root directory of the dataset.

    Returns:
        tuple[int, int]: A tuple containing the (width, height) of the original image.
    """
    image_path = root_dir / "test" / "img" / f"{image_id:08d}.jpg"
    with Image.open(image_path) as img:
        return img.size


def scale_boxes_to_original(boxes: torch.Tensor, current_size: int, original_size: tuple[int, int]) -> torch.Tensor:
    """
    Rescales bounding boxes from the model's square input size back to the original image dimensions.
    This function reverses the `LongestMaxSize` and `PadIfNeeded` transformations.

    Args:
        boxes (torch.Tensor): Predicted bounding boxes on the padded, square image.
        current_size (int): The size of the square input to the model (e.g., IMG_SIZE).
        original_size (tuple[int, int]): The (width, height) of the original image.

    Returns:
        torch.Tensor: The bounding boxes scaled to the original image coordinates.
    """
    orig_w, orig_h = original_size

    # The transformations first scaled the longest side to `current_size` and then padded the shorter side.
    # The scale factor is therefore determined by the ratio of the longest original side to `current_size`.
    scale_factor = max(orig_w, orig_h) / current_size

    # Calculate the padding that was added to the shorter side to make the image square.
    if orig_w >= orig_h:  # Original image was landscape or square
        # Padded on top and bottom
        pad_h = (current_size - orig_h / scale_factor) / 2
        pad_w = 0
    else:  # Original image was portrait
        # Padded on left and right
        pad_w = (current_size - orig_w / scale_factor) / 2
        pad_h = 0

    # Reverse the transformation process:
    # 1. Subtract the padding from the coordinates.
    boxes[:, [0, 2]] -= pad_w  # Subtract horizontal padding from x-coordinates
    boxes[:, [1, 3]] -= pad_h  # Subtract vertical padding from y-coordinates

    # 2. Multiply by the scale factor to resize to original dimensions.
    boxes *= scale_factor

    # 3. Clip coordinates to ensure they are within the original image boundaries.
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
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--nms_iou_threshold", type=float, default=0.65, help="IoU threshold for Non-Maximum Suppression."
    )
    args = parser.parse_args()

    # --- 1. Prepare Data ---
    test_dataset = PigDataset(args.data_root, frame_ids=None, is_train=False, transforms=get_transform(train=False))

    # Configure DataLoader consistently with the training script.
    num_workers = min(int(os.cpu_count() * 0.75), 12)
    g = torch.Generator().manual_seed(args.seed)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # No need to shuffle for prediction
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        generator=g,
        persistent_workers=num_workers > 0,
    )

    # --- 2. Load Model ---
    model = create_model(num_classes=NUM_CLASSES)
    # Load weights with `weights_only=True` for security.
    state = torch.load(args.model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()  # Set model to evaluation mode

    # --- 3. Run Inference and Post-processing ---
    results = []
    print(
        f"--- Predicting on test set (Confidence Threshold: {args.conf_threshold}) "
        f"and (NMS IoU Threshold: {args.nms_iou_threshold}) ---"
    )

    for images, targets in tqdm(test_loader):
        images_gpu = [img.to(DEVICE) for img in images]

        with torch.inference_mode():  # Disables gradient calculation for efficiency
            outputs = model(images_gpu)

        # Move outputs to CPU for post-processing
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

            # Step 3: Apply Non-Maximum Suppression (NMS).
            # NMS removes redundant, overlapping bounding boxes for the same object,
            # keeping only the one with the highest score.
            keep_indices = nms(boxes, scores, args.nms_iou_threshold)
            boxes = boxes[keep_indices]
            scores = scores[keep_indices]

            # Step 4: Scale the final bounding box coordinates back to the original image size.
            if boxes.shape[0] > 0:
                original_size = get_original_image_size(image_id, args.data_root)
                boxes = scale_boxes_to_original(boxes, current_size=IMG_SIZE, original_size=original_size)

            # Format the predictions into the required Kaggle submission string.
            # Format: "score x1 y1 w h class_id" for each box, joined by spaces.
            parts = []
            for score, box in zip(scores, boxes):
                x1, y1, x2, y2 = box.tolist()
                w = max(0.0, x2 - x1)  # Ensure width is non-negative
                h = max(0.0, y2 - y1)  # Ensure height is non-negative
                # The competition requires the class to be 0 for the single category.
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
