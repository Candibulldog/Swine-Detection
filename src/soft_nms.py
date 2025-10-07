# src/soft_nms.py

import torch
from torchvision.ops import box_iou


def soft_nms(boxes, scores, iou_threshold=0.5, sigma=0.5, score_threshold=0.001, method="gaussian"):
    """
    Soft-NMS: 改進的 NMS，不直接刪除重疊框，而是降低其分數
    對密集物體檢測特別有效！

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
        return torch.empty((0,), dtype=torch.long), torch.empty((0,), dtype=torch.float)

    # 確保是浮點數
    boxes = boxes.float()
    scores = scores.float().clone()

    # 初始化
    keep = []

    # 按分數降序排序
    order = scores.argsort(descending=True)

    while order.numel() > 0:
        # 選擇當前分數最高的框
        i = order[0]
        keep.append(i.item())

        if order.numel() == 1:
            break

        # 計算與其他框的 IoU
        ious = box_iou(boxes[i : i + 1], boxes[order[1:]])[0]

        # Soft-NMS: 根據 IoU 降低分數而不是直接刪除
        if method == "linear":
            # Linear decay
            weight = torch.ones_like(ious)
            weight[ious > iou_threshold] = 1 - ious[ious > iou_threshold]
        elif method == "gaussian":
            # Gaussian decay
            weight = torch.exp(-(ious**2) / sigma)
        else:
            raise ValueError(f"Unknown method: {method}")

        # 更新分數
        scores[order[1:]] *= weight

        # 移除分數過低的框
        valid = scores[order[1:]] > score_threshold
        order = order[1:][valid]

        # 重新排序
        order = order[scores[order].argsort(descending=True)]

    keep = torch.tensor(keep, dtype=torch.long, device=boxes.device)
    return keep, scores[keep]


def batched_soft_nms(boxes_list, scores_list, iou_threshold=0.5, sigma=0.5, score_threshold=0.001, method="gaussian"):
    """
    批量處理的 Soft-NMS

    Args:
        boxes_list: List of [N_i, 4] 每張圖的檢測框
        scores_list: List of [N_i] 每張圖的分數
        其他參數同 soft_nms

    Returns:
        filtered_boxes: 過濾後的框列表
        filtered_scores: 過濾後的分數列表
    """
    filtered_boxes = []
    filtered_scores = []

    for boxes, scores in zip(boxes_list, scores_list):
        keep, new_scores = soft_nms(
            boxes, scores, iou_threshold=iou_threshold, sigma=sigma, score_threshold=score_threshold, method=method
        )
        filtered_boxes.append(boxes[keep])
        filtered_scores.append(new_scores)

    return filtered_boxes, filtered_scores


def apply_soft_nms_to_predictions(predictions, iou_threshold=0.5, sigma=0.5, score_threshold=0.05, method="gaussian"):
    """
    將 Soft-NMS 應用到 Faster R-CNN 的預測結果

    Args:
        predictions: List of dict, 每個 dict 包含:
            - 'boxes': [N, 4]
            - 'scores': [N]
            - 'labels': [N]
        其他參數同 soft_nms

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
        keep_all = []

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
            original_indices = torch.where(mask)[0]
            keep_all.append(original_indices[keep])
            scores[original_indices[keep]] = new_scores

        if keep_all:
            keep_all = torch.cat(keep_all)

            filtered_pred = {"boxes": boxes[keep_all], "scores": scores[keep_all], "labels": labels[keep_all]}
        else:
            filtered_pred = {
                "boxes": torch.empty((0, 4), device=boxes.device),
                "scores": torch.empty((0,), device=scores.device),
                "labels": torch.empty((0,), dtype=torch.long, device=labels.device),
            }

        filtered_predictions.append(filtered_pred)

    return filtered_predictions


# ===== 使用範例 =====
if __name__ == "__main__":
    # 測試 Soft-NMS
    boxes = torch.tensor(
        [
            [10, 10, 50, 50],
            [15, 15, 55, 55],  # 與第一個框重疊
            [100, 100, 150, 150],
            [105, 105, 155, 155],  # 與第三個框重疊
        ],
        dtype=torch.float,
    )

    scores = torch.tensor([0.9, 0.85, 0.88, 0.82], dtype=torch.float)

    print("原始框數:", len(boxes))
    print("原始分數:", scores)

    # 標準 NMS (使用 torchvision)
    from torchvision.ops import nms

    keep_nms = nms(boxes, scores, iou_threshold=0.5)
    print("\n標準 NMS 保留:", len(keep_nms), "個框")
    print("保留的分數:", scores[keep_nms])

    # Soft-NMS
    keep_soft, scores_soft = soft_nms(boxes, scores, iou_threshold=0.5, method="gaussian")
    print("\nSoft-NMS 保留:", len(keep_soft), "個框")
    print("保留的分數:", scores_soft)
