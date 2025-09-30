import torchvision
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator


def create_model(num_classes):
    """
    建立一個客製化的 Faster R-CNN 模型。

    Args:
        num_classes (int): 類別數量 (包含背景)。

    Returns:
        torch.nn.Module: 一個準備好進行訓練的 Faster R-CNN 模型。
    """
    # =================================================================
    # 1. 載入一個預訓練的 Backbone (特徵提取器)
    # =================================================================
    # 作業規定：可以使用預訓練的 "特徵提取器 (feature extractors)"
    # 這裡我們載入在 ImageNet 上預訓練的 ResNet-50 作為 backbone。
    # 我們只載入 backbone 的權重，整個 Faster R-CNN 的其他部分是隨機初始化的。
    weights_backbone = ResNet50_Weights.DEFAULT
    backbone = torchvision.models.resnet50(weights=weights_backbone)

    # --- 去掉 ResNet-50 最後的全連接層，我們只需要特徵提取的部分 ---
    # Faster R-CNN 需要知道 backbone 輸出的 channel 數量
    # 對於 ResNet-50，最後一個卷積 block 輸出的 channel 數是 2048
    backbone.out_channels = 2048

    # =================================================================
    # 2. 建立 RPN (Region Proposal Network)
    # =================================================================
    # RPN 需要知道在哪幾個尺寸和長寬比的 anchor box 上進行預測。
    # 這裡我們使用常見的設定，適用於大多數情況。
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),)
    )

    # =================================================================
    # 3. 建立 RoI (Region of Interest) Head
    # =================================================================
    # RoI head 負責從 RPN 提出的候選區域中，提取特徵並進行最終分類和邊界框回歸。
    # box_roi_pool 參數定義了要用哪幾層 feature map 來提取特徵。
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=["0"],  # '0' 代表 backbone 輸出的第一層 feature map
        output_size=7,
        sampling_ratio=2,
    )

    # =================================================================
    # 4. 組合所有組件，建立 Faster R-CNN 模型
    # =================================================================
    # !! 關鍵 !! 這裡不載入任何預訓練權重，模型是從頭開始的
    model = FasterRCNN(
        backbone,
        num_classes=91,  # 暫時用 COCO 的 91 類初始化
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
    )

    # =================================================================
    # 5. 替換掉預訓練的分類頭 (Classifier Head)
    # =================================================================
    # 這是遷移學習的關鍵步驟。
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
