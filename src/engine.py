# src/engine.py

import math
from collections import defaultdict

import torch
from tqdm import tqdm

# 引入 COCO 評估工具
from .coco_eval import CocoEvaluator
from .coco_utils import get_coco_api_from_dataset


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """
    在一個 epoch 上訓練模型，包含混合精度訓練和梯度裁剪。
    """
    model.train()

    # 用於混合精度訓練的 GradScaler
    scaler = torch.cuda.amp.GradScaler()

    # 用於記錄和平均各項損失
    loss_accumulator = defaultdict(float)

    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1} [Training]")

    for i, (images, targets) in enumerate(progress_bar):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 使用 autocast 上下文管理器進行混合精度前向傳播
        with torch.cuda.amp.autocast():
            loss_dict = model(images, targets)
            total_loss = sum(loss for loss in loss_dict.values())

        # 檢查 loss 是否有效，防止 NaN 導致訓練崩潰
        if not math.isfinite(total_loss.item()):
            print(f"!!! Loss is {total_loss.item()}, stopping training at iteration {i} to prevent model corruption.")
            # 在這裡可以選擇退出或跳過這個 batch
            continue

        # 反向傳播
        optimizer.zero_grad()
        # scaler.scale 會將損失乘以一個縮放因子，防止梯度下溢
        scaler.scale(total_loss).backward()

        # 可選：梯度裁剪，防止梯度爆炸
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # scaler.step 會自動 unscale 梯度，然後調用 optimizer.step()
        scaler.step(optimizer)
        # 更新縮放因子
        scaler.update()

        # 記錄損失
        for k, v in loss_dict.items():
            loss_accumulator[k] += v.item()
        loss_accumulator["total_loss"] += total_loss.item()

        # 更新進度條顯示
        avg_loss = loss_accumulator["total_loss"] / (i + 1)
        progress_bar.set_postfix(loss=f"{avg_loss:.4f}")

    # 打印 epoch 的平均損失
    num_batches = len(data_loader)
    print(f"Epoch {epoch + 1} training finished. Average losses:")
    for k, v in loss_accumulator.items():
        print(f"  - {k}: {v / num_batches:.4f}")


@torch.no_grad()
def evaluate(model, data_loader, device):
    """
    在驗證集上評估模型。
    """
    model.eval()

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = ["bbox"]
    coco_evaluator = CocoEvaluator(coco, iou_types)

    progress_bar = tqdm(data_loader, desc="Validation")

    for images, targets in progress_bar:
        images = [img.to(device) for img in images]

        # 在評估時也使用 autocast，可以加速推論
        with torch.cuda.amp.autocast():
            outputs = model(images)

        outputs_cpu = [{k: v.to("cpu") for k, v in t.items()} for t in outputs]
        res = {target["image_id"].item(): output for target, output in zip(targets, outputs_cpu)}

        coco_evaluator.update(res)

    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    return coco_evaluator
