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

    def __init__(self, backbone, in_channels_list, out_channels):
        super().__init__()
        self.body = backbone
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool(),
        )
        self.out_channels = out_channels

    def forward(self, x):
        # backbone 返回字典: {'0': tensor, '1': tensor, '2': tensor, '3': tensor}
        features = self.body(x)
        # FPN 接收字典,返回新的字典: {'0': tensor, '1': tensor, '2': tensor, '3': tensor, 'pool': tensor}
        features = self.fpn(features)
        return features


def create_model(num_classes: int):
    """
    Creates a Faster R-CNN model with a ConvNeXt-Tiny backbone and FPN.
    """
    print("✅ INFO: Creating model with ConvNeXt-Tiny backbone.")

    # 1. Load ConvNeXt-Tiny backbone with ImageNet pre-trained weights
    convnext_backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)

    # 2. Create a feature extractor to get intermediate features
    class ConvNeXtFeatureExtractor(nn.Module):
        def __init__(self, convnext_model):
            super().__init__()
            self.features = convnext_model.features

        def forward(self, x):
            outputs = {}
            for i, layer in enumerate(self.features):
                x = layer(x)
                # Extract features from the 4 main stages
                if i in [1, 3, 5, 7]:
                    stage_idx = [1, 3, 5, 7].index(i)
                    outputs[str(stage_idx)] = x
            return outputs

    backbone = ConvNeXtFeatureExtractor(convnext_backbone)

    # 3. Define the channels for FPN
    in_channels_list = [96, 192, 384, 768]
    out_channels = 256

    # 4. Create the backbone with FPN
    backbone_with_fpn = ConvNeXtBackboneWithFPN(
        backbone=backbone,
        in_channels_list=in_channels_list,
        out_channels=out_channels,
    )

    # 5. Define anchor generator
    anchor_generator = AnchorGenerator(
        sizes=((32,), (48,), (64,), (96,), (128,)),  # 更密集的尺寸，移除了無用的大尺寸
        aspect_ratios=((0.4, 1.0, 2.5),) * 5,  # 稍微擴大長寬比範圍
    )

    # 6. Create the Faster R-CNN model
    model = FasterRCNN(
        backbone_with_fpn,
        num_classes=91,  # Temporary, will be replaced
        rpn_anchor_generator=anchor_generator,
    )

    # 7. Replace the box predictor head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
