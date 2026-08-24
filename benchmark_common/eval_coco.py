"""Single authoritative COCO-format evaluation pipeline (P0 fix).

Audit finding this addresses: FasterRCNN/SSD300 computed metrics via
pycocotools.COCOeval directly, while YOLOv5's reported precision/recall/
mAP came from Ultralytics' own results.csv -- a different evaluation
implementation not proven numerically equivalent. Every model's
predictions must now be converted to the COCO detection format below and
evaluated through this one function:

    [{"image_id": int, "category_id": int, "bbox": [x, y, w, h], "score": float}, ...]
"""
import json
import os
import tempfile

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def _xywh_to_xyxy(box):
    x, y, w, h = box
    return x, y, x + w, y + h


def _iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = _xywh_to_xyxy(box_a)
    bx1, by1, bx2, by2 = _xywh_to_xyxy(box_b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _fixed_point_precision_recall(coco_gt, predictions, iou_thresh=0.5):
    """Ordinary detection precision/recall/F1 at ONE fixed operating point
    (confidence threshold already applied by the caller, IoU>=iou_thresh
    for a match). This is deliberately not COCO's multi-threshold
    Average Recall -- it's the plain PR statistic the README's
    "Precision"/"Recall" line items actually describe.
    """
    preds_by_img_cat = {}
    for p in predictions:
        preds_by_img_cat.setdefault((p["image_id"], p["category_id"]), []).append(p)

    tp = fp = fn = 0
    for img_id in coco_gt.getImgIds():
        gts = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))
        gt_by_cat = {}
        for g in gts:
            gt_by_cat.setdefault(g["category_id"], []).append(g)

        cat_ids = set(gt_by_cat) | {cid for (iid, cid) in preds_by_img_cat if iid == img_id}
        for cat_id in cat_ids:
            gt_boxes = gt_by_cat.get(cat_id, [])
            preds = sorted(preds_by_img_cat.get((img_id, cat_id), []), key=lambda p: -p["score"])
            matched_gt = set()
            for pred in preds:
                best_iou, best_idx = 0.0, -1
                for gi, g in enumerate(gt_boxes):
                    if gi in matched_gt:
                        continue
                    iou = _iou(pred["bbox"], g["bbox"])
                    if iou > best_iou:
                        best_iou, best_idx = iou, gi
                if best_iou >= iou_thresh:
                    tp += 1
                    matched_gt.add(best_idx)
                else:
                    fp += 1
            fn += len(gt_boxes) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def evaluate_coco(predictions, coco_gt_path, voc_classes, conf_threshold=None):
    """
    predictions: list of COCO-format detection dicts, or a path to a JSON
        file containing them.
    coco_gt_path: path to a COCO-format ground-truth JSON (see
        benchmark_common.voc_coco.convert_voc_to_coco).
    voc_classes: ordered class-name list; category_id = index + 1.
    conf_threshold: if given, predictions below this score are dropped
        before evaluation (the shared operating point,
        benchmark_common.protocol.CONF_THRESHOLD).
    """
    if isinstance(predictions, str):
        with open(predictions, "r") as f:
            predictions = json.load(f)

    if conf_threshold is not None:
        predictions = [p for p in predictions if p["score"] >= conf_threshold]

    if not predictions:
        raise ValueError("No predictions to evaluate (all filtered out or input was empty)")

    coco_gt = COCO(coco_gt_path)
    if "info" not in coco_gt.dataset:
        coco_gt.dataset["info"] = {}
    if "licenses" not in coco_gt.dataset:
        coco_gt.dataset["licenses"] = []

    fd, tmp_pred_path = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(predictions, f)
        coco_pred = coco_gt.loadRes(tmp_pred_path)

        coco_eval = COCOeval(coco_gt, coco_pred, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats = coco_eval.stats
        metrics = {
            "mAP_0.5_0.95": float(stats[0]),
            "mAP_0.5": float(stats[1]),
            "mAP_0.75": float(stats[2]),
            "AP_small": float(stats[3]),
            "AP_medium": float(stats[4]),
            "AP_large": float(stats[5]),
            "AR_1": float(stats[6]),
            "AR_10": float(stats[7]),
            "AR_100": float(stats[8]),
            "AR_small": float(stats[9]),
            "AR_medium": float(stats[10]),
            "AR_large": float(stats[11]),
        }

        # Per-class AP@0.5: precision array is [T(iou), R(recall), K(cls), A(area), M(maxDet)];
        # T=0 -> IoU 0.5 (pycocotools' default iouThrs starts at .50), A=0 -> area='all',
        # M=2 -> maxDet=100.
        precision_arr = coco_eval.eval["precision"]
        per_class_ap = {}
        for k, class_name in enumerate(voc_classes):
            p = precision_arr[0, :, k, 0, 2]
            p = p[p > -1]
            per_class_ap[class_name] = float(np.mean(p)) if p.size else float("nan")
        metrics["per_class_AP_0.5"] = per_class_ap

        precision_pt, recall_pt, f1_pt = _fixed_point_precision_recall(coco_gt, predictions, iou_thresh=0.5)
        metrics["precision"] = precision_pt
        metrics["recall"] = recall_pt
        metrics["f1"] = f1_pt

        return metrics
    finally:
        if os.path.exists(tmp_pred_path):
            os.remove(tmp_pred_path)
