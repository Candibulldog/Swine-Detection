# src/model.py

from torchvision.models import ResNet50_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def create_model(num_classes: int):
    """
    Create a Faster R-CNN model with a ResNet50-FPN backbone.

    IMPORTANT for HW1 rules:
    - We are ONLY using ImageNet-pretrained weights for the backbone (ResNet50).
    - The detection head (classifier and box regressor) is randomly initialized.
    - We do NOT load any COCO-pretrained Faster R-CNN weights.
    """

    model = fasterrcnn_resnet50_fpn_v2(
        weights=None,  # <-- Detector head: NO pretrained weights (must train from scratch)
        weights_backbone=ResNet50_Weights.DEFAULT,  # <-- Backbone: YES, allowed to use ImageNet-pretrained ResNet50
    )

    # Replace the classification head with a new one (num_classes = background + pig)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
