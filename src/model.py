# src/model.py

import torch
from torchvision.models import Swin_V2_T_Weights, swin_v2_t
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool


class SwinBackboneWithFPN(torch.nn.Module):
    """
    Swin Transformer V2 backbone with Feature Pyramid Network (FPN).
    """

    def __init__(self, backbone, return_layers, in_channels_list, out_channels):
        super().__init__()
        self.body = backbone
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool(),
        )
        self.out_channels = out_channels

    def forward(self, x):
        features = self.body(x)
        features = self.fpn(features)
        return features


def create_model(num_classes: int):
    """
    Creates a Faster R-CNN model with a Swin-V2-T backbone and FPN, configured for transfer learning.

    This function adheres to specific transfer learning rules:
    - The backbone (Swin-V2-T) is initialized with weights pre-trained on ImageNet.
    - The rest of the model, including the RPN, RoI heads, and the final classification/regression
      layers, is randomly initialized and must be trained from scratch on the target dataset.

    Args:
        num_classes (int): The number of classes for the classification head.
                           This should include the background class, so for a single-object
                           detection task (e.g., "pig"), num_classes should be 2 (1 for pig + 1 for background).

    Returns:
        A PyTorch model instance (Faster R-CNN with Swin-V2-T backbone).
    """

    # Load Swin-V2-T backbone with ImageNet pre-trained weights
    swin_backbone = swin_v2_t(weights=Swin_V2_T_Weights.DEFAULT)

    # Remove the classification head (we only need the feature extractor)
    swin_backbone.head = torch.nn.Identity()

    # Create a feature extractor that returns intermediate features
    class SwinV2FeatureExtractor(torch.nn.Module):
        def __init__(self, swin_model):
            super().__init__()
            self.features = swin_model.features
            self.norm = swin_model.norm
            self.permute = swin_model.permute

        def forward(self, x):
            # Extract features at different stages
            features = {}

            # Stage 0 & 1 combined
            x = self.features[0](x)
            x = self.features[1](x)

            # Stage 2 (downsample + blocks)
            x = self.features[2](x)
            features["0"] = self.permute(x)  # [B, 96, H/4, W/4]

            # Stage 3
            x = self.features[3](x)
            features["1"] = self.permute(x)  # [B, 192, H/8, W/8]

            # Stage 4
            x = self.features[4](x)
            features["2"] = self.permute(x)  # [B, 384, H/16, W/16]

            # Stage 5
            x = self.features[5](x)
            features["3"] = self.permute(x)  # [B, 768, H/32, W/32]

            return features

    backbone = SwinV2FeatureExtractor(swin_backbone)

    # Define the channels for each feature map from Swin-V2-T
    in_channels_list = [96, 192, 384, 768]
    out_channels = 256

    # Create backbone with FPN
    backbone_with_fpn = SwinBackboneWithFPN(
        backbone=backbone,
        return_layers={"0": "0", "1": "1", "2": "2", "3": "3"},
        in_channels_list=in_channels_list,
        out_channels=out_channels,
    )

    # Define anchor generator
    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,), (512,)), aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )

    # Create the Faster R-CNN model
    model = FasterRCNN(
        backbone_with_fpn,
        num_classes=91,  # Temporary, will be replaced
        rpn_anchor_generator=anchor_generator,
    )

    # Replace the box predictor with the correct number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
