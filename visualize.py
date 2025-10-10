# visualize.py (Refactored for Robustness and Extensibility)

import argparse
import itertools
import shutil
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle
from tqdm import tqdm

# A cycle of distinct colors for bounding boxes
COLORS = [
    "#FF3838",
    "#FF9D97",
    "#FF7566",
    "#FFA459",
    "#FFB243",
    "#FFD700",
    "#A8E4A0",
    "#32CD32",
    "#00A550",
    "#00BFFF",
    "#1E90FF",
    "#87CEEB",
    "#9370DB",
    "#C71585",
    "#FF1493",
]


# ✨ REFACTOR 1: Define a clear data structure for a prediction.
class Detection:
    def __init__(self, conf: float, x: float, y: float, w: float, h: float, cls_id: int):
        self.conf = conf
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.cls_id = cls_id

    def __repr__(self):
        return f"Detection(conf={self.conf:.2f}, cls={self.cls_id})"


def parse_prediction_string(pred_str: str) -> list[Detection]:
    """Parses the prediction string from Kaggle format into a list of Detection objects."""
    if pd.isna(pred_str) or not isinstance(pred_str, str) or pred_str.strip() == "":
        return []

    parts = pred_str.strip().split()
    detections = []

    # Process in chunks of 6
    for i in range(0, len(parts) - len(parts) % 6, 6):
        try:
            chunk = [float(p) for p in parts[i : i + 6]]
            detections.append(Detection(*chunk))
        except (ValueError, IndexError):
            print(f"⚠️ Warning: Could not parse prediction fragment: {' '.join(parts[i : i + 6])}")
            continue

    return detections


def draw_detections(
    image_path: Path, detections: list[Detection], class_names: dict[int, str], title: str
) -> plt.Figure:
    """Draws all detection boxes on a single image."""
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    dpi = 100
    height, width, _ = img_rgb.shape
    figsize = (width / dpi, height / dpi)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(img_rgb)
    ax.axis("off")

    color_cycle = itertools.cycle(COLORS)

    for det in detections:
        color = next(color_cycle)
        rect = Rectangle((det.x, det.y), det.w, det.h, edgecolor=color, facecolor="none", linewidth=3)
        ax.add_patch(rect)

        label = f"{class_names.get(det.cls_id, f'Class {det.cls_id}')}: {det.conf:.2f}"
        ax.text(
            det.x,
            det.y - 10,
            label,
            fontsize=14,
            weight="bold",
            color="white",
            bbox=dict(facecolor=color, alpha=0.8, edgecolor="none", pad=1.5),
        )

    ax.set_title(title, fontsize=16, weight="bold")
    fig.tight_layout(pad=0)
    return fig


def get_image_ids_to_visualize(df: pd.DataFrame, args: argparse.Namespace) -> list[str]:
    """Determines which image IDs to visualize based on command-line arguments."""
    # This calculation is now done once, preventing redundant computation.
    df["detection_count"] = df["PredictionString"].apply(lambda x: len(parse_prediction_string(x)))

    if args.fixed_ids:
        return [str(i) for i in args.fixed_ids]
    if args.random_k:
        return df["Image_ID"].sample(n=min(args.random_k, len(df)), random_state=args.seed).astype(str).tolist()
    if args.top_k_most:
        return df.nlargest(args.top_k_most, "detection_count")["Image_ID"].astype(str).tolist()
    if args.top_k_least:
        df_with_dets = df[df["detection_count"] > 0]
        return df_with_dets.nsmallest(args.top_k_least, "detection_count")["Image_ID"].astype(str).tolist()
    if args.no_detections:
        df_no_dets = df[df["detection_count"] == 0]
        if len(df_no_dets) == 0:
            print("ℹ️ No images with zero detections found.")
            return []
        return (
            df_no_dets.sample(n=min(args.no_detections, len(df_no_dets)), random_state=args.seed)["Image_ID"]
            .astype(str)
            .tolist()
        )

    # ✨ REFACTOR 2: Make the default behavior more explicit and informative.
    print("ℹ️ No selection mode specified. Visualizing a default set of representative images.")
    default_ids = ["25", "615", "950", "1255", "1805", "1864"]
    return [img_id for img_id in default_ids if img_id in df["Image_ID"].values]


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Visualize object detection results from a submission CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("csv_path", type=Path, help="Path to the submission.csv file to analyze.")
    parser.add_argument("--test_dir", type=Path, default=Path("./data/test/img"))
    parser.add_argument("--seed", type=int, default=42)

    # ... (mutually exclusive group remains the same) ...
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--fixed_ids", type=int, nargs="+", help="A specific list of Image IDs to visualize.")
    group.add_argument("--random_k", type=int, help="Visualize K randomly selected images.")
    group.add_argument("--top_k_most", type=int, help="Visualize the K images with the most detections.")
    group.add_argument("--top_k_least", type=int, help="Visualize the K images with the fewest detections (but > 0).")
    group.add_argument("--no_detections", type=int, help="Visualize K images with zero detections.")

    args = parser.parse_args()

    # --- 1. Setup ---
    if not args.csv_path.is_file():
        print(f"❌ Error: Submission file not found at '{args.csv_path}'.")
        return

    base_output_dir = Path("./visualizations/")
    output_dir_specific = base_output_dir / f"viz_{args.csv_path.stem}"

    # ✨ REFACTOR 3: Clean up old results for a fresh start.
    if output_dir_specific.exists():
        shutil.rmtree(output_dir_specific)
    output_dir_specific.mkdir(parents=True)

    print(f"✅ Visualizations will be saved to: {output_dir_specific.resolve()}")

    # --- 2. Load Data ---
    df = pd.read_csv(args.csv_path)
    df["Image_ID"] = df["Image_ID"].astype(str)

    image_ids = get_image_ids_to_visualize(df, args)
    if not image_ids:
        print("🤷 No images selected for visualization. Exiting.")
        return

    # --- 3. Main Visualization Loop ---
    # ✨ REFACTOR 4: Create a reusable dictionary for faster lookups.
    df_dict = df.set_index("Image_ID").to_dict()["PredictionString"]
    class_names = {0: "pig"}  # Define class names once

    print(f"\n🚀 Analyzing '{args.csv_path.name}'...")
    for image_id in tqdm(image_ids, desc="Visualizing"):
        pred_str = df_dict.get(image_id)
        if pred_str is None:
            print(f"⚠️ Warning: Image ID {image_id} not found in CSV's index.")
            continue

        preds = parse_prediction_string(pred_str)
        img_path = args.test_dir / f"{int(image_id):08d}.jpg"
        if not img_path.is_file():
            print(f"⚠️ Warning: Image file not found: {img_path}")
            continue

        title = f"Image ID: {image_id}\nSource: {args.csv_path.name}\nDetections: {len(preds)}"

        try:
            fig = draw_detections(img_path, preds, class_names, title)
            output_path = output_dir_specific / f"{image_id}.png"
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        except Exception as e:
            print(f"❌ An error occurred while processing Image ID {image_id}: {e}")
        finally:
            # Ensure figures are always closed to prevent memory leaks
            plt.close("all")

    print(f"\n🎉 Complete! Visualizations saved in '{output_dir_specific}'.")


if __name__ == "__main__":
    main()
