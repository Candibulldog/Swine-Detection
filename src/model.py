# src/model.py

from torchvision.models import ResNet50_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def create_model(num_classes: int):
    """
    Creates a Faster R-CNN model with a ResNet-50 FPN backbone, configured for transfer learning.

    This function adheres to specific transfer learning rules:
    - The backbone (ResNet-50) is initialized with weights pre-trained on ImageNet.
    - The rest of the model, including the RPN, RoI heads, and the final classification/regression
      layers, is randomly initialized and must be trained from scratch on the target dataset.
    - No weights pre-trained on COCO for the full Faster R-CNN model are used.

    Args:
        num_classes (int): The number of classes for the classification head.
                           This should include the background class, so for a single-object
                           detection task (e.g., "pig"), num_classes should be 2 (1 for pig + 1 for background).

    Returns:
        A PyTorch model instance (Faster R-CNN).
    """

    # Instantiate the Faster R-CNN model with a ResNet50 FPN backbone.
    model = fasterrcnn_resnet50_fpn_v2(
        # `weights=None`: This is critical. It ensures that the entire detector head
        # (RPN, RoI heads, box predictor) is NOT loaded with pre-trained weights from COCO.
        # Its weights will be randomly initialized.
        weights=None,
        # `weights_backbone`: This specifies that ONLY the feature extractor part of the model
        # (the ResNet-50 backbone) should be loaded with its default pre-trained weights from ImageNet.
        # This is a standard and effective practice for transfer learning.
        weights_backbone=ResNet50_Weights.DEFAULT,
    )

    # --- Customize the model head for the target dataset ---
    # 1. Get the number of input features for the classifier.
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # 2. Replace the pre-trained head with a new, randomly initialized one.
    #    The new head is a `FastRCNNPredictor` layer with the correct number of output classes
    #    (our `num_classes`). This is necessary because the original head was designed for a
    #    different dataset (like COCO) with a different number of classes.
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
