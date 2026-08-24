from __future__ import annotations

import numpy as np

from benchmark.data.class_map import VOC_CLASSES
from benchmark.eval.matching import greedy_match


def _ap_11_point(recalls: np.ndarray, precisions: np.ndarray) -> float:
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        if np.any(recalls >= t):
            ap += np.max(precisions[recalls >= t])
    return ap / 11.0


def _class_ap(
    preds: list[dict],
    gts: list[dict],
    class_id: int,
    iou_threshold: float,
) -> float:
    class_preds = [p for p in preds if p["class_id"] == class_id]
    class_preds.sort(key=lambda x: x["score"], reverse=True)

    gt_by_image = {}
    total_gt = 0
    for gt in gts:
        entries = [o for o in gt["objects"] if o["class_id"] == class_id]
        gt_by_image[gt["image_id"]] = {"boxes": [x["bbox_xyxy"] for x in entries], "matched": [False] * len(entries)}
        total_gt += len(entries)

    if total_gt == 0:
        return 0.0

    tps = np.zeros(len(class_preds), dtype=np.float32)
    fps = np.zeros(len(class_preds), dtype=np.float32)

    for i, pred in enumerate(class_preds):
        image_gt = gt_by_image.get(pred["image_id"], {"boxes": [], "matched": []})
        gt_boxes = image_gt["boxes"]
        if not gt_boxes:
            fps[i] = 1
            continue

        best_iou = 0.0
        best_idx = -1
        for j, gt_box in enumerate(gt_boxes):
            if image_gt["matched"][j]:
                continue
            # imported lazily to avoid cycle warnings in static checks
            from benchmark.eval.matching import iou_xyxy

            value = iou_xyxy(pred["bbox_xyxy"], gt_box)
            if value > best_iou:
                best_iou = value
                best_idx = j

        if best_idx >= 0 and best_iou >= iou_threshold:
            image_gt["matched"][best_idx] = True
            tps[i] = 1
        else:
            fps[i] = 1

    cum_tps = np.cumsum(tps)
    cum_fps = np.cumsum(fps)
    precisions = cum_tps / np.maximum(cum_tps + cum_fps, 1e-6)
    recalls = cum_tps / max(total_gt, 1)
    return float(_ap_11_point(recalls, precisions))


def compute_map(preds: list[dict], gts: list[dict], iou_thresholds: list[float]) -> dict:
    per_class = {}
    maps = []

    for class_id, class_name in enumerate(VOC_CLASSES):
        ap_at_thresholds = [_class_ap(preds, gts, class_id, thr) for thr in iou_thresholds]
        per_class[class_name] = {
            "ap@0.5": ap_at_thresholds[0] if iou_thresholds else 0.0,
            "ap_mean": float(np.mean(ap_at_thresholds)) if ap_at_thresholds else 0.0,
        }

    for thr in iou_thresholds:
        class_aps = [_class_ap(preds, gts, class_id, thr) for class_id in range(len(VOC_CLASSES))]
        maps.append(float(np.mean(class_aps)))

    return {
        "mAP@0.5": maps[0] if maps else 0.0,
        "mAP@0.5:0.95": float(np.mean(maps)) if maps else 0.0,
        "per_class": per_class,
    }


def compute_precision_recall(
    preds: list[dict],
    gts: list[dict],
    match_iou: float,
) -> dict[str, float]:
    tp_total = 0
    fp_total = 0
    fn_total = 0

    gt_by_image = {gt["image_id"]: gt for gt in gts}

    for class_id in range(len(VOC_CLASSES)):
        for image_id, gt in gt_by_image.items():
            gt_boxes = [o["bbox_xyxy"] for o in gt["objects"] if o["class_id"] == class_id]
            class_preds = [p for p in preds if p["image_id"] == image_id and p["class_id"] == class_id]
            pred_boxes = [p["bbox_xyxy"] for p in class_preds]
            pred_scores = [p["score"] for p in class_preds]

            tp, fp, fn = greedy_match(pred_boxes, pred_scores, gt_boxes, match_iou)
            tp_total += len(tp)
            fp_total += len(fp)
            fn_total += fn

    precision = tp_total / max(tp_total + fp_total, 1)
    recall = tp_total / max(tp_total + fn_total, 1)
    return {"precision": float(precision), "recall": float(recall)}


def evaluate_predictions(preds: list[dict], gts: list[dict], config: dict) -> dict:
    ap_thresholds = config.get("iou_thresholds_ap", [0.5 + i * 0.05 for i in range(10)])
    match_iou = float(config.get("precision_recall_match_iou", 0.5))

    map_metrics = compute_map(preds, gts, ap_thresholds)
    pr_metrics = compute_precision_recall(preds, gts, match_iou)

    return {
        "aggregate": {
            "mAP@0.5": map_metrics["mAP@0.5"],
            "mAP@0.5:0.95": map_metrics["mAP@0.5:0.95"],
            "precision": pr_metrics["precision"],
            "recall": pr_metrics["recall"],
        },
        "per_class": map_metrics["per_class"],
    }
