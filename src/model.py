# src/model.py

from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def create_model(num_classes):
    """
    建立一個客製化的 Faster R-CNN 模型。

    Args:
        num_classes (int): 類別數量 (包含背景)。

    Returns:
        torch.nn.Module: 一個準備好進行訓練的 Faster R-CNN 模型。
    """
    # 使用 TorchVision 0.13+ 的新 API
    # weights_backbone 參數指定只載入 backbone 的權重，其他層隨機初始化
    # 這完全符合作業規定
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights_backbone=weights)

    # 獲取分類器的輸入特徵數
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # 替換掉預訓練的分類頭
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
