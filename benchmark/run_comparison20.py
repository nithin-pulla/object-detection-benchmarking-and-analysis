from __future__ import annotations

import argparse
from pathlib import Path

from benchmark.adapters import build_adapter
from benchmark.config import load_config
from benchmark.data.comparison_loader import load_comparison_image_ids
from benchmark.data.voc_loader import load_ground_truth, load_pil_image
from benchmark.io_utils import save_json
from benchmark.timing.profiler import profile_single_image, warmup
from benchmark.viz.qualitative import save_comparison_panel


def run(config: dict):
    image_ids = load_comparison_image_ids(config["comparison_images_path"])

    voc_root = config["dataset"]["voc_root"]
    split = config["dataset"].get("split", "val")
    gt_items = load_ground_truth(voc_root, split, image_ids=image_ids)
    gt_by_id = {item.image_id: item for item in gt_items}

    out_root = Path(config.get("output_root", "results")) / config.get("run_name", "comparison20") / "comparison20"
    preds_by_model = {model_name: {} for model_name in config["models"]}
    summary_rows = []

    for model_name in config["models"]:
        adapter = build_adapter(model_name)
        model = adapter.load_model(config)
        warmup(adapter, model, load_pil_image(gt_items[0].image_path), config)

        for item in gt_items:
            image = load_pil_image(item.image_path)
            detections, timing = profile_single_image(adapter, model, image, config)
            preds = [p.to_dict() for p in adapter.to_standard_prediction(item.image_id, detections)]
            preds_by_model[model_name][item.image_id] = {"predictions": preds, "timing": timing.to_dict()}

    for image_id in image_ids:
        gt = [
            {
                "class_id": obj.class_id,
                "class_name": obj.class_name,
                "bbox_xyxy": obj.bbox_xyxy,
            }
            for obj in gt_by_id[image_id].objects
        ]
        panel_models = {
            model_name: preds_by_model[model_name][image_id]["predictions"]
            for model_name in config["models"]
        }
        panel_path = out_root / "panels" / f"{image_id}.png"
        save_comparison_panel(gt_by_id[image_id].image_path, gt, panel_models, panel_path)

        row = {"image_id": image_id, "gt_count": len(gt), "panel": str(panel_path)}
        for model_name in config["models"]:
            entry = preds_by_model[model_name][image_id]
            row[f"{model_name}_detections"] = len(entry["predictions"])
            row[f"{model_name}_latency_ms"] = entry["timing"]["total_ms"]
        summary_rows.append(row)

    save_json({"rows": summary_rows}, out_root / "comparison20_summary.json")

    for model_name in config["models"]:
        save_json(preds_by_model[model_name], out_root / model_name / "predictions_per_image.json")

    print(f"Comparison20 outputs written to {out_root}")


def main():
    parser = argparse.ArgumentParser(description="Run standardized 20-image comparison")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    run(config)


if __name__ == "__main__":
    main()
