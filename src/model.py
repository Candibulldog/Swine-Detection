# src/model.py

import torch.nn as nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool


class ConvNeXtBackboneWithFPN(nn.Module):
    """
    Wrapper to connect ConvNeXt backbone with a Feature Pyramid Network (FPN).
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
        # The backbone (self.body) returns a dictionary of features
        features = self.body(x)
        # The FPN takes this dictionary and returns its own feature dictionary
        features = self.fpn(features)
        return features


def create_model(num_classes: int):
    """
    Creates a Faster R-CNN model with a ConvNeXt-Tiny backbone and FPN.
    """
    print("✅ INFO: Creating model with ConvNeXt-Tiny backbone.")

    # 1. Load ConvNeXt-Tiny backbone with ImageNet pre-trained weights
    convnext_backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)

    # 2. Create a feature extractor to get intermediate features.
    #    ConvNeXt's `features` attribute is a Sequential module. We need to
    #    intercept the outputs of each stage.
    class ConvNeXtFeatureExtractor(nn.Module):
        def __init__(self, convnext_model):
            super().__init__()
            # We directly take the feature extractor part of ConvNeXt
            self.features = convnext_model.features

        def forward(self, x):
            # ConvNeXt-Tiny has 4 stages, with downsampling at the beginning of each.
            # The stages are at indices 1, 3, 5, 7.
            outputs = {}
            # Stage 1 output (after self.features[1])
            x = self.features[0](x)
            x = self.features[1](x)
            outputs["0"] = x  # Output channels: 96

            # Stage 2 output (after self.features[3])
            x = self.features[2](x)
            x = self.features[3](x)
            outputs["1"] = x  # Output channels: 192

            # Stage 3 output (after self.features[5])
            x = self.features[4](x)
            x = self.features[5](x)
            outputs["2"] = x  # Output channels: 384

            # Stage 4 output (after self.features[7])
            x = self.features[6](x)
            x = self.features[7](x)
            outputs["3"] = x  # Output channels: 768

            return outputs

    backbone = ConvNeXtFeatureExtractor(convnext_backbone)

    # 3. Define the channels for FPN.
    #    These are the output channels of each stage of ConvNeXt-Tiny.
    in_channels_list = [96, 192, 384, 768]
    out_channels = 256  # Standard FPN output channels

    # 4. Create the backbone with FPN.
    backbone_with_fpn = ConvNeXtBackboneWithFPN(
        backbone=backbone,
        return_layers={"0": "0", "1": "1", "2": "2", "3": "3"},
        in_channels_list=in_channels_list,
        out_channels=out_channels,
    )

    # 5. Define anchor generator (using the standard FPN setup).
    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,), (512,)), aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )

    # 6. Create the Faster R-CNN model.
    model = FasterRCNN(
        backbone_with_fpn,
        num_classes=91,  # Temporary, will be replaced
        rpn_anchor_generator=anchor_generator,
    )

    # 7. Replace the box predictor head.
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
