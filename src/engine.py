# src/engine.py


import torch
from tqdm import tqdm  # 一個好用的進度條工具

# 為了計算 mAP，我們需要從 torchvision 引入評估工具
# 如果你沒有安裝 pycocotools，Colab 會提示你安裝：!pip install pycocotools
from .coco_eval import CocoEvaluator
from .coco_utils import get_coco_api_from_dataset


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """
    在一個 epoch 上訓練模型。

    Args:
        model (torch.nn.Module): 要訓練的模型。
        optimizer (torch.optim.Optimizer): 優化器。
        data_loader (torch.utils.data.DataLoader): 訓練資料的 DataLoader。
        device (torch.device): 訓練設備 (CPU 或 GPU)。
        epoch (int): 當前的 epoch 編號。
    """
    model.train()  # 將模型設置為訓練模式

    # 使用 tqdm 建立一個進度條
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1}")

    for images, targets in progress_bar:
        # 1. 將圖片和標註移動到指定的 device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 2. 前向傳播 (Forward pass)
        #    在訓練模式下，模型會直接回傳一個包含所有損失的字典
        #    例如: {'loss_classifier': tensor, 'loss_box_reg': tensor, ...}
        loss_dict = model(images, targets)

        # 3. 計算總損失
        #    我們將所有回傳的損失加總
        losses = sum(loss for loss in loss_dict.values())

        # 4. 反向傳播 (Backward pass)
        optimizer.zero_grad()  # 清除舊的梯度
        losses.backward()  # 計算梯度
        optimizer.step()  # 更新模型權重

        # 5. (可選) 在進度條上顯示當前的總損失
        progress_bar.set_postfix(loss=losses.item())

    print(f"Epoch {epoch + 1} training finished. Total Loss: {losses.item():.4f}")


@torch.no_grad()  # 在這個函式中，我們不需要計算梯度
def evaluate(model, data_loader, device):
    """
    在驗證集上評估模型。

    Args:
        model (torch.nn.Module): 要評估的模型。
        data_loader (torch.utils.data.DataLoader): 驗證資料的 DataLoader。
        device (torch.device): 評估設備 (CPU 或 GPU)。

    Returns:
        coco_evaluator object: 包含所有評估結果的物件。
    """
    model.eval()  # 將模型設置為評估模式

    # 建立 CocoEvaluator 物件，這是 TorchVision 官方推薦的評估工具
    # 它可以處理 AP, AP50, AP75 等多種指標的計算
    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = ["bbox"]
    coco_evaluator = CocoEvaluator(coco, iou_types)

    print("\nEvaluating...")
    progress_bar = tqdm(data_loader, desc="Validation")

    for images, targets in progress_bar:
        # 1. 將圖片移動到指定的 device
        images = list(img.to(device) for img in images)

        # 2. 進行預測
        #    在評估模式下，模型會回傳每個圖片的預測結果
        #    格式: [{'boxes': tensor, 'labels': tensor, 'scores': tensor}, ...]
        outputs = model(images)

        # 3. 將 PyTorch Tensors 轉換為 CPU 上的 NumPy arrays，以便評估
        outputs = [
            {k: v.to(torch.device("cpu")).numpy() for k, v in t.items()}
            for t in outputs
        ]

        # 4. 格式化 targets 以符合評估器要求
        #    評估器需要 targets 包含一個 'image_id' 鍵
        res = {
            target["image_id"].item(): output
            for target, output in zip(targets, outputs)
        }

        # 5. 將這一批次的預測結果餵給評估器
        coco_evaluator.update(res)

    # 累積所有批次的結果並進行最終計算
    coco_evaluator.accumulate()
    coco_evaluator.summarize()  # 這一步會在 console 印出 mAP_50:95 等結果

    return coco_evaluator
