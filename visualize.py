# visualize.py

import argparse
import itertools
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle
from tqdm import tqdm

# ===================================================================
# ✨ Visualization Tool ✨
# ===================================================================

# A cycle of distinct colors for bounding boxes to improve visibility in crowded scenes.
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


def parse_prediction_string(pred_str: str) -> list[dict]:
    """Parses the prediction string from the Kaggle submission format."""
    preds = []
    if pd.isna(pred_str) or not isinstance(pred_str, str) or pred_str.strip() == "":
        return preds

    parts = pred_str.strip().split()
    if len(parts) % 6 != 0:
        print(f"Warning: Incomplete prediction string, trailing part will be ignored: {pred_str}")
        parts = parts[: (len(parts) // 6) * 6]

    for i in range(0, len(parts), 6):
        try:
            conf, x, y, w, h, cls_id = map(float, parts[i : i + 6])
            preds.append({"conf": conf, "x": x, "y": y, "w": w, "h": h, "cls": int(cls_id)})
        except (ValueError, IndexError):
            print(f"Warning: Could not parse prediction fragment: {' '.join(parts[i : i + 6])}")
            continue
    return preds


def draw_detections(
    image_path: Path, detections: list[dict], conf_threshold: float, class_names: dict, title: str
) -> plt.Figure:
    """Draws all detection boxes on a single image."""
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # --- Intelligently adjust canvas size to match the image's original aspect ratio ---
    dpi = 100
    height, width, _ = img_rgb.shape
    figsize = (width / dpi, height / dpi)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(img_rgb)
    ax.axis("off")

    filtered_dets = [d for d in detections if d["conf"] >= conf_threshold]
    color_cycle = itertools.cycle(COLORS)

    for det in filtered_dets:
        x, y, w, h, cls_id, conf = det["x"], det["y"], det["w"], det["h"], det["cls"], det["conf"]
        color = next(color_cycle)

        rect = Rectangle((x, y), w, h, edgecolor=color, facecolor="none", linewidth=2.5)
        ax.add_patch(rect)

        label = f"{class_names.get(cls_id, f'Class {cls_id}')}: {conf:.2f}"
        ax.text(
            x, y - 5, label, fontsize=12, color="white", bbox=dict(facecolor=color, alpha=0.8, edgecolor="none", pad=1)
        )

    ax.set_title(title, fontsize=14)
    plt.tight_layout(pad=0)
    return fig


def get_image_ids_to_visualize(df: pd.DataFrame, args: argparse.Namespace) -> list[str]:
    """Determines which image IDs to visualize based on command-line arguments."""
    df["detection_count"] = df["PredictionString"].apply(lambda x: len(parse_prediction_string(x)))

    if args.fixed_ids:
        print(f"🔍 Using specified fixed Image IDs: {args.fixed_ids}")
        return [str(i) for i in args.fixed_ids]
    if args.random_k:
        print(f"🎲 Selecting {args.random_k} random images...")
        return df["Image_ID"].sample(n=min(args.random_k, len(df)), random_state=args.seed).tolist()
    if args.top_k_most:
        print(f"📈 Selecting the {args.top_k_most} images with the most detections...")
        return df.nlargest(args.top_k_most, "detection_count")["Image_ID"].tolist()
    if args.top_k_least:
        df_with_dets = df[df["detection_count"] > 0]
        print(f"📉 Selecting the {args.top_k_least} images with the fewest detections (but > 0)...")
        return df_with_dets.nsmallest(args.top_k_least, "detection_count")["Image_ID"].tolist()
    if args.no_detections:
        df_no_dets = df[df["detection_count"] == 0]
        print(f"👻 Selecting {args.no_detections} images with no detections...")
        return df_no_dets.sample(n=min(args.no_detections, len(df_no_dets)), random_state=args.seed)[
            "Image_ID"
        ].tolist()

    # Default behavior: use a predefined set of representative IDs.
    print("ℹ️ No selection mode specified. Using default fixed IDs.")
    return ["1000", "1859", "832", "664", "1", "817", "1360", "1258"]


def main(args):
    """Main execution function."""
    csv_path = Path(args.csv_path)
    test_dir = Path(args.test_dir)

    # --- 1. Create Output Directory ---
    # ✨ MODIFIED: All visualizations are now saved under a unified './visualizations' directory.
    base_output_dir = Path("./visualizations")
    output_dir_specific = base_output_dir / f"viz_{csv_path.stem}"
    output_dir_specific.mkdir(parents=True, exist_ok=True)
    print(f"✅ Setup complete. Visualization results will be saved to: {output_dir_specific.resolve()}")

    # --- 2. Load and Prepare Data ---
    try:
        df = pd.read_csv(csv_path)
        df["Image_ID"] = df["Image_ID"].astype(str)
    except FileNotFoundError:
        print(f"❌ Error: Submission file not found at '{csv_path}'.")
        return

    # --- 3. Select Images to Process ---
    image_ids = get_image_ids_to_visualize(df, args)

    # --- 4. Loop Through and Process Each Image ---
    print(f"\n🚀 Analyzing '{csv_path.name}'...")
    for image_id in tqdm(image_ids, desc="Visualizing"):
        row = df[df["Image_ID"] == image_id]

        if row.empty:
            print(f"⚠️ Warning: Image_ID not found in CSV: {image_id}")
            continue

        pred_str = row["PredictionString"].iloc[0]
        preds = parse_prediction_string(pred_str)

        img_path = test_dir / f"{int(image_id):08d}.jpg"
        if not img_path.exists():
            print(f"⚠️ Warning: Image file not found: {img_path}")
            continue

        filtered_count = sum(1 for p in preds if p["conf"] >= args.conf_threshold)
        title = (
            f"Image ID: {image_id}\n"
            f"Source: {csv_path.name}\n"
            f"Detections (conf ≥ {args.conf_threshold}): {filtered_count}"
        )

        try:
            fig = draw_detections(img_path, preds, args.conf_threshold, {0: "pig"}, title)
            output_path = output_dir_specific / f"{image_id}.png"
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            print(f"❌ An error occurred while processing Image ID {image_id}: {e}")
            plt.close("all")

    print(f"\n🎉 Complete! Visualization results saved in '{output_dir_specific}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Object Detection Results")

    parser.add_argument("--csv_path", type=str, required=True, help="Path to the submission.csv file to analyze.")
    parser.add_argument(
        "--test_dir", type=str, default="/content/data/test/img", help="Directory containing the test images."
    )
    parser.add_argument("--conf_threshold", type=float, default=0.5, help="Confidence threshold for visualization.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random sampling.")

    # --- Mutually exclusive group for selecting which images to visualize ---
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--fixed_ids", type=int, nargs="+", help="A specific list of Image IDs to visualize.")
    group.add_argument("--random_k", type=int, help="Visualize K randomly selected images.")
    group.add_argument("--top_k_most", type=int, help="Visualize the K images with the most detections.")
    group.add_argument("--top_k_least", type=int, help="Visualize the K images with the fewest detections (but > 0).")
    group.add_argument("--no_detections", type=int, help="Visualize K images with zero detections.")

    # ✨ MODIFIED: Correctly parse arguments and then call the main function.
    args = parser.parse_args()
    main(args)
