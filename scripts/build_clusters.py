# scripts/build_clusters.py

import argparse
import random
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def extract_image_features(image_path: Path) -> tuple[float, ...] | None:
    """
    Extracts a feature vector from a single image based on macroscopic properties.

    The feature vector includes metrics for brightness, contrast, color saturation,
    and sharpness, which are effective for differentiating scene types (e.g.,
    daytime color vs. nighttime infrared).

    Args:
        image_path: Path to the input image file.

    Returns:
        A tuple of floats representing the feature vector, or None if processing fails.
    """
    try:
        # Use OpenCV for fast image reading and processing.
        image = cv2.imread(str(image_path))
        if image is None:
            return None

        # a. Brightness and Contrast Features
        # Convert to grayscale to analyze luminance properties.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)  # Standard deviation of brightness represents contrast.

        # b. Colorfulness Feature
        # Convert to HSV color space; Saturation (S) is a good proxy for color intensity.
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mean_saturation = np.mean(hsv[:, :, 1])

        # c. Sharpness/Clarity Feature
        # Variance of the Laplacian operator is a common metric for image sharpness.
        # Higher variance implies more pronounced edges and thus a sharper image.
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

        return (mean_brightness, std_brightness, mean_saturation, sharpness)
    except Exception as e:
        print(f"Warning: Could not process image {image_path.name}. Error: {e}", file=sys.stderr)
        return None


def visualize_clusters(df: pd.DataFrame, image_dir: Path, n_clusters: int, save_path: Path):
    """Generates and saves a grid of sample images for each cluster for verification."""
    print("\n🖼️  Generating cluster visualization...")
    frame_to_path = {int(p.stem): p for p in image_dir.glob("*.jpg")}

    fig, axes = plt.subplots(n_clusters, 5, figsize=(20, 3 * n_clusters))
    if n_clusters == 1:  # Handle case where there's only one cluster
        axes = np.array([axes])
    fig.suptitle("Image Clustering Results Verification", fontsize=20)

    for i in range(n_clusters):
        cluster_frames = df[df["cluster_id"] == i]["frame_id"].tolist()
        if not cluster_frames:
            continue

        sample_frames = random.sample(cluster_frames, min(len(cluster_frames), 5))
        axes[i, 0].set_ylabel(f"Cluster {i}\n({len(cluster_frames)} images)", fontsize=14)

        for j, frame_id in enumerate(sample_frames):
            try:
                img = Image.open(frame_to_path[frame_id])
                axes[i, j].imshow(img)
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
            except Exception as e:
                print(f"Warning: Failed to load image for visualization {frame_id}. Error: {e}", file=sys.stderr)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path)
    print(f"✅ Visualization saved to: {save_path}")


def main(args):
    """
    Main execution function to run the image clustering pipeline.
    """
    image_dir = Path(args.image_dir)
    output_path = Path(args.output_csv)
    vis_path = output_path.with_name(f"{output_path.stem}_visualization.png")

    if not image_dir.is_dir():
        print(f"❌ Error: Image directory not found at '{image_dir}'", file=sys.stderr)
        sys.exit(1)

    image_paths = sorted(list(image_dir.glob("*.jpg")))
    if not image_paths:
        print(f"❌ Error: No .jpg images found in '{image_dir}'", file=sys.stderr)
        sys.exit(1)

    # --- 1. Feature Extraction ---
    print(f"[1/4] 📊 Extracting features from {len(image_paths)} images...")
    features_list, valid_frame_ids = [], []
    for path in tqdm(image_paths, desc="Extracting Features"):
        features = extract_image_features(path)
        if features:
            features_list.append(features)
            valid_frame_ids.append(int(path.stem))

    if not features_list:
        print("❌ Error: Failed to extract features from any image.", file=sys.stderr)
        sys.exit(1)

    features_array = np.array(features_list)

    # --- 2. Feature Scaling ---
    # StandardScaler is crucial for K-Means as it ensures all features
    # (which have different scales) contribute equally to the distance metric.
    print("[2/4] ⚖️  Standardizing features...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_array)

    # --- 3. K-Means Clustering ---
    # n_init='auto' is the modern default to handle future changes in scikit-learn.
    print(f"[3/4] 🤖 Performing K-Means clustering (K={args.n_clusters})...")
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=args.random_state, n_init="auto")
    cluster_labels = kmeans.fit_predict(features_scaled)

    # --- 4. Save Results ---
    print(f"[4/4] 💾 Saving cluster assignments to {output_path}...")
    df = pd.DataFrame({"frame_id": valid_frame_ids, "cluster_id": cluster_labels})
    df.to_csv(output_path, index=False)
    print("\n✅ Success! Clustering complete.")
    print("Preview of results:")
    print(df.head().to_string())

    # --- 5. (Optional) Visualization ---
    if args.visualize:
        visualize_clusters(df, image_dir, args.n_clusters, vis_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform feature-based clustering on swine images to identify different scene types.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Set default path relative to a typical project structure
    # where this script is in `scripts/` and data is in `data/`.
    default_img_dir = Path(__file__).resolve().parent.parent / "data" / "train" / "img"

    parser.add_argument(
        "--image_dir", type=str, default=str(default_img_dir), help="Path to the directory containing training images."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="image_clusters.csv",
        help="Path to save the output CSV file with cluster assignments.",
    )
    parser.add_argument("--n_clusters", type=int, default=3, help="The number of clusters (scene types) to identify.")
    parser.add_argument(
        "--random_state", type=int, default=42, help="Seed for the random number generator to ensure reproducibility."
    )
    parser.add_argument(
        "--visualize", action="store_true", help="If set, generate and save a visualization of the clusters."
    )

    args = parser.parse_args()
    main(args)
