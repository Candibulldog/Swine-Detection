# src/model.py

# 引入 ResNet50 的權重
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def create_model(num_classes):
    """
    建立一個客製化的 Faster R-CNN 模型。
    """
    # =================================================================
    # 修正：我們需要明確地提供 Backbone (ResNet50) 的權重，
    # 而不是整個 Faster R-CNN 的權重。
    # =================================================================
    backbone_weights = ResNet50_Weights.DEFAULT

    # 將正確的 backbone 權重傳入 weights_backbone 參數
    model = fasterrcnn_resnet50_fpn_v2(weights_backbone=backbone_weights)

    # 獲取分類器的輸入特徵數
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # 替換掉預訓練的分類頭
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
