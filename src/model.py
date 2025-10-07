# src/model.py

import torch
from torchvision.models import Swin_V2_T_Weights, swin_v2_t
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.feature_extraction import create_feature_extractor  # ✨ 1. 導入新工具
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool


def create_model(num_classes: int):
    """
    Creates a Faster R-CNN model with a Swin-V2-T backbone and FPN, configured for transfer learning.
    This version uses the official torchvision feature_extraction utility for robustness.
    """
    print("✅ INFO: Creating model with Swin Transformer (Swin-V2-T) backbone.")

    # --- 1. 加載預訓練的 Swin-V2-T Backbone ---
    backbone = swin_v2_t(weights=Swin_V2_T_Weights.DEFAULT)

    # --- 2. 使用官方工具創建特徵提取器 ---
    # 我們需要告訴工具，我們想要從 Swin-V2-T 的哪幾層提取特徵。
    # 對於 FPN，我們通常需要 4 個不同尺度的特徵圖。
    # Swin-T 的特徵層通常在 'features' 模塊中，編號 2, 3, 4, 5 對應 4 個 stage 的輸出。
    # 注意：層的名稱可能因 torchvision 版本而異，需要確認。
    # 一個常見的 return_nodes 字典：
    return_nodes = {
        "features.2": "0",
        "features.3": "1",
        "features.4": "2",
        "features.5": "3",
    }
    # 讓 torchvision 自動幫我們創建一個只返回這四層輸出的新模型
    body = create_feature_extractor(backbone, return_nodes=return_nodes)

    # 該工具創建的模型會自動處理 permute 等操作，使輸出兼容 FPN

    # --- 3. 創建 FPN ---
    # Swin-V2-T 在這四個 stage 的輸出通道數是 [96, 192, 384, 768]
    in_channels_list = [96, 192, 384, 768]
    fpn = FeaturePyramidNetwork(
        in_channels_list=in_channels_list,
        out_channels=256,
        extra_blocks=LastLevelMaxPool(),
    )

    # --- 4. 將 Backbone 和 FPN 組合起來 ---
    # 我們需要一個簡單的包裝類來按順序調用它們
    class BackboneWithFPN(torch.nn.Module):
        def __init__(self, body, fpn):
            super().__init__()
            self.body = body
            self.fpn = fpn
            self.out_channels = fpn.out_channels

        def forward(self, x):
            x = self.body(x)
            x = self.fpn(x)
            return x

    backbone_with_fpn = BackboneWithFPN(body, fpn)

    # --- 5. 創建 Faster R-CNN 模型 (和之前一樣) ---
    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,), (512,)), aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )

    model = FasterRCNN(
        backbone_with_fpn,
        num_classes=91,  # 臨時值
        rpn_anchor_generator=anchor_generator,
    )

    # --- 6. 替換頭部 (和之前一樣) ---
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
