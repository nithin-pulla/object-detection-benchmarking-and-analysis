from __future__ import annotations


def iou_xyxy(box_a: list[float], box_b: list[float]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    denom = area_a + area_b - inter
    if denom <= 0.0:
        return 0.0
    return inter / denom


def greedy_match(
    pred_boxes: list[list[float]],
    pred_scores: list[float],
    gt_boxes: list[list[float]],
    iou_threshold: float,
) -> tuple[list[int], list[int], int]:
    order = sorted(range(len(pred_boxes)), key=lambda i: pred_scores[i], reverse=True)
    gt_used = [False] * len(gt_boxes)
    tp, fp = [], []

    for idx in order:
        best_iou = 0.0
        best_gt = -1
        for j, gt in enumerate(gt_boxes):
            if gt_used[j]:
                continue
            value = iou_xyxy(pred_boxes[idx], gt)
            if value > best_iou:
                best_iou = value
                best_gt = j
        if best_gt >= 0 and best_iou >= iou_threshold:
            gt_used[best_gt] = True
            tp.append(idx)
        else:
            fp.append(idx)

    fn = gt_used.count(False)
    return tp, fp, fn
