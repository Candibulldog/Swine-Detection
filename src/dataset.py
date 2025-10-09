# src/dataset.py

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.transforms import get_transform


class PigDataset(Dataset):
    """
    A comprehensive PyTorch Dataset for the pig detection task.
    It supports training, validation, and test modes, and features
    optional cluster-aware data augmentation for training.
    """

    def __init__(
        self,
        data_root: Path,
        is_train: bool,
        annotations_df: pd.DataFrame | None = None,
        frame_ids: list[int] | None = None,
        cluster_dict: dict[int, int] | None = None,
        use_cluster_aware_aug: bool = False,
    ):
        self.data_root = Path(data_root)
        self.is_train = is_train

        # Determine the image directory based on the mode (train/val vs. test)
        data_dir_name = "train" if annotations_df is not None else "test"
        self.image_dir = self.data_root / data_dir_name / "img"

        # If frame_ids are not provided in test mode, scan the directory.
        if frame_ids is None and annotations_df is None:
            self.frame_ids = sorted([int(p.stem) for p in self.image_dir.glob("*.jpg") if p.stem.isdigit()])
        elif frame_ids is not None:
            self.frame_ids = frame_ids
        else:
            raise ValueError("`frame_ids` must be provided for train/val modes.")

        # Pre-process annotations for efficient lookups.
        self.annotations_map = self._prepare_annotations(annotations_df, self.frame_ids)

        # Initialize the appropriate transformation pipelines.
        self.transforms = self._initialize_transforms(cluster_dict, use_cluster_aware_aug)

    def _prepare_annotations(self, df: pd.DataFrame | None, frame_ids: list[int]) -> dict:
        """
        Processes the annotations DataFrame into a dictionary for fast access in __getitem__.
        Returns a dictionary mapping `frame_id -> annotations_for_that_frame`.
        """
        if df is None:
            return {}

        # Filter for relevant frames and calculate derived columns.
        df_filtered = df[df["frame"].isin(frame_ids)].copy()
        df_filtered["x2"] = df_filtered["bb_left"] + df_filtered["bb_width"]
        df_filtered["y2"] = df_filtered["bb_top"] + df_filtered["bb_height"]

        # Group by frame and convert to a dictionary for O(1) lookup.
        return {
            frame: frame_df[["bb_left", "bb_top", "x2", "y2"]].values
            for frame, frame_df in df_filtered.groupby("frame")
        }

    def _initialize_transforms(self, cluster_dict: dict | None, use_cluster_aware: bool):
        """
        Initializes and returns the correct transformation pipeline(s) in an extensible way.
        """
        if self.is_train:
            if use_cluster_aware and cluster_dict:
                unique_cluster_ids = sorted(list(set(cluster_dict.values())))
                print(
                    f"INFO: Initializing CLUSTER-AWARE transforms for {len(unique_cluster_ids)} unique clusters: {unique_cluster_ids}"
                )

                self.cluster_dict = cluster_dict
                return {
                    cluster_id: get_transform(train=True, cluster_id=cluster_id, use_cluster_aware=True)
                    for cluster_id in unique_cluster_ids
                }
            else:
                print("INFO: Initializing Dataset with UNIFIED transform.")
                self.cluster_dict = {}
                return {"unified": get_transform(train=True, use_cluster_aware=False)}
        else:
            print("INFO: Initializing Dataset with VALIDATION/TEST transform.")
            self.cluster_dict = {}
            return {"val_test": get_transform(train=False)}

    def __len__(self) -> int:
        return len(self.frame_ids)

    def __getitem__(self, idx: int):
        frame_id = self.frame_ids[idx]
        image_path = self.image_dir / f"{frame_id:08d}.jpg"
        image = Image.open(image_path).convert("RGB")

        # Prepare the target dictionary.
        target = {"image_id": torch.tensor([frame_id])}
        boxes = self.annotations_map.get(frame_id)

        if boxes is not None and len(boxes) > 0:
            # This logic is for train/val sets which have annotations.
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

            # Filter out invalid boxes with zero or negative area.
            valid_indices = areas > 0
            boxes = boxes[valid_indices]
            areas = areas[valid_indices]

            target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
            target["labels"] = torch.ones(len(boxes), dtype=torch.int64)  # All pigs are class 1
            target["area"] = torch.as_tensor(areas, dtype=torch.float32)
            target["iscrowd"] = torch.zeros(len(boxes), dtype=torch.int64)
        else:
            # This logic handles the test set and any train/val images without annotations.
            target["boxes"] = torch.empty((0, 4), dtype=torch.float32)
            target["labels"] = torch.empty((0,), dtype=torch.int64)
            target["area"] = torch.empty((0,), dtype=torch.float32)
            target["iscrowd"] = torch.empty((0,), dtype=torch.int64)

        # Apply the appropriate transformation pipeline.
        if self.is_train:
            if "unified" in self.transforms:
                image, target = self.transforms["unified"](image, target)
            else:
                cluster_id = self.cluster_dict.get(frame_id, 0)  # Default to cluster 0 if missing
                image, target = self.transforms[cluster_id](image, target)
        else:
            image, target = self.transforms["val_test"](image, target)

        return image, target
