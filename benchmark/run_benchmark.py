from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from benchmark.adapters import build_adapter
from benchmark.config import config_hash, load_config
from benchmark.data.voc_loader import load_ground_truth, load_pil_image
from benchmark.eval.metrics import evaluate_predictions
from benchmark.io_utils import (
    build_experiment_metadata,
    ensure_schema,
    get_environment_metadata,
    save_json,
)
from benchmark.timing.profiler import profile_single_image, summarize_timings, warmup
from benchmark.viz.plots import plot_accuracy_speed


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _gt_to_eval_payload(gt_items):
    payload = []
    for item in gt_items:
        payload.append(
            {
                "image_id": item.image_id,
                "objects": [
                    {
                        "class_id": obj.class_id,
                        "class_name": obj.class_name,
                        "bbox_xyxy": obj.bbox_xyxy,
                    }
                    for obj in item.objects
                ],
            }
        )
    return payload


def run(config: dict):
    _set_seeds(int(config.get("random_seed", 42)))

    voc_root = config["dataset"]["voc_root"]
    split = config["dataset"].get("split", "val")
    gt_items = load_ground_truth(voc_root, split)
    gt_eval = _gt_to_eval_payload(gt_items)

    cfg_hash = config_hash(config)
    exp_metadata = build_experiment_metadata(config, cfg_hash)
    output_root = Path(config.get("output_root", "results")) / exp_metadata["run_id"]

    all_results = {}
    for model_name in config.get("models", []):
        adapter = build_adapter(model_name)
        model = adapter.load_model(config)

        warmup(adapter, model, load_pil_image(gt_items[0].image_path), config)

        all_preds = []
        per_image_timings = []

        benchmark_runs = int(config.get("benchmark_runs", len(gt_items)))
        for item in gt_items[:benchmark_runs]:
            image = load_pil_image(item.image_path)
            detections, timing = profile_single_image(adapter, model, image, config)
            preds = adapter.to_standard_prediction(item.image_id, detections)
            all_preds.extend([p.to_dict() for p in preds])
            per_image_timings.append(timing.to_dict())

        acc_metrics = evaluate_predictions(all_preds, gt_eval[:benchmark_runs], config)
        timing_metrics = summarize_timings(per_image_timings)

        result = {
            "experiment_metadata": exp_metadata,
            "environment": get_environment_metadata(),
            "dataset_info": {
                "name": "VOC2012",
                "split": split,
                "count": benchmark_runs,
                "class_map": "VOC20",
            },
            "model_info": adapter.get_model_metadata(),
            "accuracy_metrics": acc_metrics,
            "timing_metrics": timing_metrics,
            "predictions": all_preds,
            "comparison20_summary": {},
            "artifacts_index": {},
        }
        ensure_schema(result)

        model_out = output_root / model_name / "metrics.json"
        save_json(result, model_out)
        all_results[model_name] = result

    tradeoff_path = output_root / "accuracy_speed_tradeoff.png"
    plot_accuracy_speed(all_results, tradeoff_path)

    summary_path = output_root / "summary.json"
    save_json(
        {
            "run_id": exp_metadata["run_id"],
            "models": list(all_results.keys()),
            "tradeoff_figure": str(tradeoff_path),
        },
        summary_path,
    )

    print(json.dumps({"output_root": str(output_root), "summary": str(summary_path)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Run standardized object detection benchmark")
    parser.add_argument("--config", required=True, help="Path to benchmark YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
