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
            # 我們直接引用 swin_model 的 features 列表
            self.features = swin_model.features
            # permute 操作是將 [B, H*W, C] 轉為 [B, C, H, W]
            self.permute = swin_model.permute

        def forward(self, x):
            # 存儲不同 stage 的輸出
            outputs = {}
            feature_idx = 0

            # 遍歷 features 中的每一個模塊 (Conv, LayerNorm, SwinTransformerBlock, etc.)
            for i, layer in enumerate(self.features):
                x = layer(x)
                # Swin-V2 的結構是 downsample -> blocks
                # 我們在每個 stage 的 block 序列之後提取特徵
                # 根據 torchvision swin_v2_t 的源碼, stage 的分界點在索引 1, 3, 5, 7
                if i in [1, 3, 5, 7]:
                    # 提取特徵並進行 permute
                    outputs[str(feature_idx)] = self.permute(x)
                    feature_idx += 1

            return outputs

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
