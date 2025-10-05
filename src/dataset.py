# src/dataset.py

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class PigDataset(Dataset):
    """
    A custom PyTorch Dataset for loading the pig detection dataset.

    This class handles loading images and their corresponding annotations
    based on the specified mode (train/validation or test).
    """

    def __init__(self, root_dir: Path, frame_ids: Optional[List[int]], is_train: bool = True, transforms=None):
        """
        Initializes the dataset.

        Args:
            root_dir (Path): The root directory of the dataset.
            frame_ids (Optional[List[int]]): A list of frame IDs to be included in this dataset
                                             split (used for creating train/val sets).
            is_train (bool): If True, the dataset is in training/validation mode and will load annotations.
                             If False, it's in test mode and will only load images.
            transforms: The data augmentation and transformation pipeline to be applied.
        """
        self.root_dir = root_dir
        self.transforms = transforms
        self.is_train = is_train

        # Set data paths based on the mode.
        data_dir_name = "train" if self.is_train else "test"
        self.image_dir = self.root_dir / data_dir_name / "img"

        if self.is_train:
            # --- Training/Validation Mode ---
            # Load annotations and group them by frame for efficient retrieval.
            annotations_path = self.root_dir / data_dir_name / "gt.txt"
            all_annotations = pd.read_csv(annotations_path, header=None, names=["frame", "x", "y", "w", "h"])

            # Filter to retain only the annotations for the frames in the current split (train or val).
            subset_annotations = all_annotations[all_annotations["frame"].isin(frame_ids)].copy()

            # Performance Optimization: Group annotations by frame into a dictionary for fast lookups.
            # This avoids searching the entire DataFrame in __getitem__.
            self.annotations_map = {
                frame: group.to_numpy()[:, 1:] for frame, group in subset_annotations.groupby("frame")
            }
            # The list of images to be loaded is derived from the frames that have annotations.
            self.image_frames = sorted(list(self.annotations_map.keys()))
        else:
            # --- Test Mode ---
            # In test mode, we do not have ground truth annotations.
            # We simply scan the image directory to get the list of all image frames.
            self.annotations_map = {}
            self.image_frames = sorted([int(p.stem) for p in self.image_dir.glob("*.jpg") if p.stem.isdigit()])

        mode_str = "Train/Val" if self.is_train else "Test"
        print(f"Dataset in '{mode_str}' mode initialized with {len(self.image_frames)} images.")

    def __len__(self) -> int:
        """Returns the total number of images in the dataset."""
        return len(self.image_frames)

    def __getitem__(self, idx: int):
        """
        Retrieves a single sample (image and its target annotations) from the dataset.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing:
                - image (torch.Tensor): The transformed image tensor.
                - target (dict): A dictionary with annotation information.
        """
        frame_id = self.image_frames[idx]
        image_path = self.image_dir / f"{frame_id:08d}.jpg"
        image = Image.open(image_path).convert("RGB")

        # Initialize the target dictionary. It's structured to be compatible with
        # models like Faster R-CNN in torchvision.
        target = {"image_id": torch.tensor([frame_id], dtype=torch.int64)}

        # Initialize default empty tensors for annotations. This ensures consistent
        # return types even when there are no objects in an image.
        boxes = torch.empty((0, 4), dtype=torch.float32)
        labels = torch.empty((0,), dtype=torch.int64)
        areas = torch.empty((0,), dtype=torch.float32)
        iscrowd = torch.empty((0,), dtype=torch.int64)

        if self.is_train and frame_id in self.annotations_map:
            records = self.annotations_map[frame_id]

            # ✨ --- CRITICAL FIX: Filter out invalid bounding boxes before processing --- ✨
            # Ensures that both width (records[:, 2]) and height (records[:, 3]) are strictly positive.
            # This prevents errors during transformation and training.
            valid_indices = (records[:, 2] > 0) & (records[:, 3] > 0)
            records = records[valid_indices]

            if records.shape[0] > 0:
                # Use vectorized operations for efficient conversion and calculation.
                # Convert from [x, y, w, h] format to [x_min, y_min, x_max, y_max].
                x1 = records[:, 0]
                y1 = records[:, 1]
                w = records[:, 2]
                h = records[:, 3]
                x2 = x1 + w
                y2 = y1 + h

                boxes = torch.as_tensor(np.stack([x1, y1, x2, y2], axis=1), dtype=torch.float32)
                areas = torch.as_tensor(w * h, dtype=torch.float32)
                num_boxes = boxes.shape[0]

                # Assuming a single class (pig), so all labels are 1.
                labels = torch.ones((num_boxes,), dtype=torch.int64)
                # 'iscrowd' is used for evaluating object detection on complex scenes (e.g., COCO).
                # Here, we assume no instances are part of a crowd.
                iscrowd = torch.zeros((num_boxes,), dtype=torch.int64)

        # Populate the target dictionary with the processed annotations.
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = areas
        target["iscrowd"] = iscrowd

        # Apply the specified data augmentations and transformations.
        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target
