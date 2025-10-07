# src/transforms.py

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

# The target size to which all images will be resized.
# This ensures consistent input dimensions for batch processing in the model.
IMG_SIZE = 640

# Normalization statistics calculated from the training dataset.
# Using the dataset's own mean and standard deviation helps to center the
# data distribution around zero, which can lead to more stable and faster training.
DATASET_MEAN = [0.3615, 0.3564, 0.3543]
DATASET_STD = [0.3089, 0.3045, 0.3051]


# -----------------------------------------------------------------------------
# Albumentations Wrapper Class
# -----------------------------------------------------------------------------


class AlbumentationsTransform:
    """
    A wrapper class to streamline the use of Albumentations transforms.
    It is designed to ensure that the same transformation pipeline is applied
    consistently to both an image and its corresponding bounding boxes.
    """

    def __init__(self, transforms: list):
        """
        Initializes the transformation pipeline.

        Args:
            transforms (list): A list of Albumentations transformation objects.
        """
        # Create a composition of transforms with specific parameters for bounding boxes.
        self.transforms = A.Compose(
            transforms,
            bbox_params=A.BboxParams(
                # The format of the bounding boxes, [x_min, y_min, x_max, y_max].
                format="pascal_voc",
                # Specifies the field name that contains class labels for the boxes.
                label_fields=["labels"],
                # If a bounding box's visibility is less than 25% after a transform
                # (e.g., cropping), it will be removed. This prevents corrupted labels.
                min_visibility=0.25,
                # If a bounding box's area is less than 16 pixels after a transform,
                # it will be removed. This helps filter out objects that become too
                # small to be meaningful, preventing the model from learning noise.
                min_area=16,
            ),
        )

    def __call__(self, image, target):
        """
        Applies the defined transformation pipeline to an image and its target annotations.

        Args:
            image: The input image (e.g., a PIL Image).
            target (dict): A dictionary containing annotations like 'boxes' and 'labels'.

        Returns:
            A tuple containing the transformed image and its updated target dictionary.
        """
        # Convert input data to the NumPy format required by Albumentations.
        # This handles cases where images are PIL objects and annotations are PyTorch tensors.
        transform_args = {
            "image": np.array(image),
            "bboxes": target.get("boxes", []),
            "labels": target.get("labels", []),
        }
        if isinstance(transform_args["bboxes"], torch.Tensor):
            transform_args["bboxes"] = transform_args["bboxes"].tolist()
        if isinstance(transform_args["labels"], torch.Tensor):
            transform_args["labels"] = transform_args["labels"].tolist()

        # Apply the transformation pipeline.
        transformed = self.transforms(**transform_args)

        # Repackage the transformed data back into the expected PyTorch Tensor format.
        if "boxes" in target:
            if len(transformed["bboxes"]) > 0:
                # Convert list of boxes back to a float tensor.
                target["boxes"] = torch.as_tensor(transformed["bboxes"], dtype=torch.float32)
            else:
                # Handle the edge case where all boxes are removed by the transform.
                target["boxes"] = torch.empty((0, 4), dtype=torch.float32)

            # Convert list of labels back to an integer tensor.
            target["labels"] = torch.as_tensor(transformed["labels"], dtype=torch.int64)

        return transformed["image"], target


# -----------------------------------------------------------------------------
# Factory Function to Get Transforms
# -----------------------------------------------------------------------------


def get_transform(train: bool) -> AlbumentationsTransform:
    """
    A factory function that returns the appropriate data augmentation pipeline
    based on whether it's for training or validation/testing.

    Args:
        train (bool): If True, returns a pipeline with extensive data augmentation.
                      Otherwise, returns a basic pipeline with only resizing and normalization.
    """

    # --- Base Transforms (for Validation & Testing) ---
    # This pipeline performs only the essential steps to prepare an image for the model,
    # ensuring consistent evaluation without random alterations.
    base_transforms = [
        # 1. Resize while preserving aspect ratio: Scales the longest side of the
        #    image to IMG_SIZE without distorting the content.
        A.LongestMaxSize(max_size=IMG_SIZE, p=1.0),
        # 2. Pad to a square: Pads the resized image to a square shape (IMG_SIZE x IMG_SIZE).
        #    'border_mode=0' uses constant padding (black).
        A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=0, p=1.0),
        # 3. Normalize: Applies normalization using the pre-calculated dataset statistics.
        #    This helps the model converge faster and more reliably.
        A.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
        # 4. Convert to Tensor: Converts the NumPy array to a PyTorch tensor and
        #    rearranges dimensions from HWC (Height, Width, Channels) to CHW.
        ToTensorV2(),
    ]

    if train:
        # --- Augmentation Transforms (for Training) ---
        # This pipeline includes a variety of augmentations to increase data diversity,
        # which helps the model generalize better and reduces overfitting.
        train_transforms = [
            # === 1. Geometric & Spatial Augmentations ===
            # These alter the image's geometry to make the model robust to changes in
            # object scale, position, and orientation.
            A.LongestMaxSize(max_size=IMG_SIZE, p=1.0),
            A.PadIfNeeded(
                min_height=IMG_SIZE,
                min_width=IMG_SIZE,
                border_mode=0,  # cv2.BORDER_CONSTANT
                p=1.0,
            ),
            # Randomly crops a part of the image, ensuring that at least one bounding box remains valid.
            A.RandomSizedBBoxSafeCrop(height=IMG_SIZE, width=IMG_SIZE, erosion_rate=0.2, p=0.5),
            # Flips the image horizontally.
            A.HorizontalFlip(p=0.5),
            # Applies a combination of translation, scaling, and rotation.
            A.Affine(
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}, scale=(0.9, 1.1), rotate=(-15, 15), p=0.7
            ),
            # === 2. Color & Style Augmentations ===
            # These change the visual properties of the image, simulating different
            # environmental conditions.
            # Converts the image to grayscale, forcing the model to learn shape and texture features.
            A.ToGray(p=0.3),
            # Randomly alters brightness and contrast to simulate different lighting conditions.
            A.RandomBrightnessContrast(brightness_limit=0.6, contrast_limit=0.5, p=0.9),
            # === 3. Robustness & Occlusion Augmentations ===
            # These techniques help the model become more robust to partially hidden objects.
            # Removes rectangular regions from the image, mimicking occlusion.
            A.GridDropout(ratio=0.5, p=0.5),
            # === 4. Final Processing ===
            # These steps are mandatory to prepare the augmented image for the model.
            A.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
            ToTensorV2(),
        ]
        return AlbumentationsTransform(train_transforms)
    else:
        # For validation or testing, use only the basic, deterministic transforms.
        return AlbumentationsTransform(base_transforms)
