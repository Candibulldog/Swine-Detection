# src/soft_nms.py

import torch
from torchvision.ops import box_iou


def soft_nms(boxes, scores, iou_threshold=0.5, sigma=0.5, score_threshold=0.001, method="gaussian"):
    """
    Soft-NMS: 改進的 NMS，不直接刪除重疊框，而是降低其分數

    Args:
        boxes: [N, 4] 檢測框 (x1, y1, x2, y2)
        scores: [N] 置信度分數
        iou_threshold: IoU 閾值
        sigma: Gaussian 函數的參數
        score_threshold: 最終保留框的分數閾值
        method: 'linear' 或 'gaussian'

    Returns:
        keep: 保留的框的索引
        scores: 更新後的分數
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device), torch.empty(
            (0,), dtype=torch.float, device=boxes.device
        )

    # 確保是浮點數且 clone scores（避免修改原始值）
    boxes = boxes.float()
    scores = scores.float().clone()

    keep = []
    order = scores.argsort(descending=True)

    while order.numel() > 0:
        # 選擇當前分數最高的框
        i = order[0]
        keep.append(i.item())

        if order.numel() == 1:
            break

        # 計算與其他框的 IoU
        ious = box_iou(boxes[i : i + 1], boxes[order[1:]])[0]

        # 根據 IoU 計算權重
        if method == "linear":
            weight = torch.ones_like(ious)
            weight[ious > iou_threshold] = 1 - ious[ious > iou_threshold]
        elif method == "gaussian":
            weight = torch.exp(-(ious**2) / sigma)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'linear' or 'gaussian'")

        # 更新分數
        scores[order[1:]] *= weight

        # 移除分數過低的框並重新排序
        valid_mask = scores[order[1:]] > score_threshold
        order = order[1:][valid_mask]

        if order.numel() > 0:
            order = order[scores[order].argsort(descending=True)]

    keep = torch.tensor(keep, dtype=torch.long, device=boxes.device)
    return keep, scores[keep]


def apply_soft_nms_to_predictions(predictions, iou_threshold=0.5, sigma=0.5, score_threshold=0.05, method="gaussian"):
    """
    將 Soft-NMS 應用到 Faster R-CNN 的預測結果

    Args:
        predictions: List of dict, 每個 dict 包含:
            - 'boxes': [N, 4]
            - 'scores': [N]
            - 'labels': [N]
        iou_threshold: IoU 閾值
        sigma: Gaussian decay 參數
        score_threshold: 最終保留閾值
        method: 'linear' 或 'gaussian'

    Returns:
        filtered_predictions: 過濾後的預測結果
    """
    filtered_predictions = []

    for pred in predictions:
        boxes = pred["boxes"]
        scores = pred["scores"]
        labels = pred["labels"]

        if boxes.numel() == 0:
            filtered_predictions.append(pred)
            continue

        # 對每個類別分別應用 Soft-NMS
        unique_labels = labels.unique()
        keep_indices = []
        updated_scores = scores.clone()

        for label in unique_labels:
            mask = labels == label
            label_boxes = boxes[mask]
            label_scores = scores[mask]

            keep, new_scores = soft_nms(
                label_boxes,
                label_scores,
                iou_threshold=iou_threshold,
                sigma=sigma,
                score_threshold=score_threshold,
                method=method,
            )

            # 轉換回原始索引
            original_indices = torch.where(mask)[0][keep]
            keep_indices.append(original_indices)
            updated_scores[original_indices] = new_scores

        if keep_indices:
            keep_all = torch.cat(keep_indices)
            filtered_pred = {"boxes": boxes[keep_all], "scores": updated_scores[keep_all], "labels": labels[keep_all]}
        else:
            filtered_pred = {
                "boxes": torch.empty((0, 4), dtype=boxes.dtype, device=boxes.device),
                "scores": torch.empty((0,), dtype=scores.dtype, device=scores.device),
                "labels": torch.empty((0,), dtype=labels.dtype, device=labels.device),
            }

        filtered_predictions.append(filtered_pred)

    return filtered_predictions
