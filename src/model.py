from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def create_model(num_classes):
    """
    建立一個客製化的 Faster R-CNN 模型 (修正版)。

    Args:
        num_classes (int): 類別數量 (包含背景)。

    Returns:
        torch.nn.Module: 一個準備好進行訓練的 Faster R-CNN 模型。
    """
    # =================================================================
    # 1. 載入一個在 COCO 上預訓練的 Faster R-CNN 模型架構
    # =================================================================
    # 這次我們使用更高階的 API，它會自動處理好 backbone 的載入和 FPN 的設定。
    # weights_backbone 參數會確保只有 backbone (ResNet-50) 的權重被載入，
    # 而模型的其他部分 (RPN, RoI heads) 則是隨機初始化的。
    # 這完全符合作業規定！
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)

    # =================================================================
    # 2. 替換掉預訓練的分類頭 (Classifier Head)
    # =================================================================
    # a. 取得分類器輸入的特徵維度
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # b. 用我們新的分類頭替換掉舊的
    #    新的分類頭是一個線性層 (nn.Linear)，輸出維度是我們的 `num_classes`
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# --- 如何在你的 Notebook 中使用它 ---

# 我們的類別有 'pig' 和 'background'，所以總共是 2 個類別
NUM_CLASSES = 2

# 建立模型
model = create_model(NUM_CLASSES)

# 把它移到 GPU (如果有的話)
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# model.to(device)

print("✅ Faster R-CNN 模型建立成功！")
print(model)
