from __future__ import annotations

from typing import Dict, List, Sequence

import torch

from .geometry import cxcywh_to_xyxy


def classification_metrics(preds: Sequence[int], targets: Sequence[int], num_classes: int) -> Dict[str, float]:
    if not preds:
        return {"accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0, "balanced_accuracy": 0.0, "support": 0.0}
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.float32)
    for pred, target in zip(preds, targets):
        confusion[int(target), int(pred)] += 1.0
    correct = float(torch.trace(confusion))
    total = float(confusion.sum().clamp_min(1.0))
    per_class_f1: List[float] = []
    per_class_recall: List[float] = []
    supports: List[float] = []
    for index in range(num_classes):
        tp = confusion[index, index]
        fp = confusion[:, index].sum() - tp
        fn = confusion[index, :].sum() - tp
        denom = (2 * tp) + fp + fn
        support = float(confusion[index, :].sum().item())
        per_class_f1.append(float((2 * tp / denom).item()) if denom > 0 else 0.0)
        recall_denom = tp + fn
        per_class_recall.append(float((tp / recall_denom).item()) if recall_denom > 0 else 0.0)
        supports.append(support)

    weighted_f1 = 0.0
    if total > 0:
        weighted_f1 = sum(f1 * support for f1, support in zip(per_class_f1, supports)) / total
    balanced_accuracy = sum(per_class_recall) / max(len(per_class_recall), 1)
    metrics = {
        "accuracy": correct / total,
        "macro_f1": sum(per_class_f1) / max(len(per_class_f1), 1),
        "weighted_f1": weighted_f1,
        "balanced_accuracy": balanced_accuracy,
        "support": total,
    }
    for index, f1 in enumerate(per_class_f1):
        metrics[f"f1_class_{index}"] = f1
    return metrics


def segmentation_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mode: str,
    num_classes: int = 2,
) -> Dict[str, float]:
    if mode == "binary":
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        targets = targets.float()
        intersection = (preds * targets).sum().item()
        pred_sum = preds.sum().item()
        target_sum = targets.sum().item()
        union = pred_sum + target_sum - intersection
        dice = (2.0 * intersection + 1.0) / (pred_sum + target_sum + 1.0)
        iou = (intersection + 1.0) / (union + 1.0)
        return {"dice": float(dice), "iou": float(iou)}

    preds = torch.argmax(logits, dim=1)
    dice_scores: List[float] = []
    iou_scores: List[float] = []
    for class_index in range(1, num_classes):
        pred_mask = (preds == class_index).float()
        true_mask = (targets == class_index).float()
        intersection = (pred_mask * true_mask).sum().item()
        pred_sum = pred_mask.sum().item()
        target_sum = true_mask.sum().item()
        union = pred_sum + target_sum - intersection
        dice_scores.append((2.0 * intersection + 1.0) / (pred_sum + target_sum + 1.0))
        iou_scores.append((intersection + 1.0) / (union + 1.0))
    return {
        "dice": float(sum(dice_scores) / max(len(dice_scores), 1)),
        "iou": float(sum(iou_scores) / max(len(iou_scores), 1)),
    }


def bbox_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(inter_x2 - inter_x1, 0.0)
    inter_h = max(inter_y2 - inter_y1, 0.0)
    inter_area = inter_w * inter_h
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def detection_metrics(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> Dict[str, float]:
    ious: List[float] = []
    for pred_box, target_box in zip(pred_boxes, target_boxes):
        ious.append(bbox_iou(cxcywh_to_xyxy(pred_box.tolist()), cxcywh_to_xyxy(target_box.tolist())))
    return {"mean_iou": float(sum(ious) / max(len(ious), 1))}


def regression_metrics(preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    if preds.numel() == 0:
        return {"mae": 0.0, "rmse": 0.0}
    errors = preds - targets
    mae = torch.abs(errors).mean().item()
    rmse = torch.sqrt((errors ** 2).mean()).item()
    return {"mae": float(mae), "rmse": float(rmse)}
