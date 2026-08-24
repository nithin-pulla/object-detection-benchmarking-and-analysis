from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run(results_root: str):
    root = Path(results_root)
    rows = []

    for metrics_file in root.glob("*/metrics.json"):
        model_name = metrics_file.parent.name
        data = _read_json(metrics_file)
        row = {
            "model": model_name,
            "mAP@0.5": data["accuracy_metrics"]["aggregate"]["mAP@0.5"],
            "mAP@0.5:0.95": data["accuracy_metrics"]["aggregate"]["mAP@0.5:0.95"],
            "precision": data["accuracy_metrics"]["aggregate"]["precision"],
            "recall": data["accuracy_metrics"]["aggregate"]["recall"],
            "pre_ms": data["timing_metrics"]["preprocessing_ms"],
            "infer_ms": data["timing_metrics"]["inference_ms"],
            "post_ms": data["timing_metrics"]["postprocessing_ms"],
            "total_ms": data["timing_metrics"]["total_ms"],
            "fps_total": data["timing_metrics"]["fps_total"],
        }
        rows.append(row)

    if not rows:
        raise ValueError(f"No metrics.json files found in {root}")

    df = pd.DataFrame(rows).sort_values("model")
    csv_path = root / "aggregate_metrics.csv"
    json_path = root / "aggregate_metrics.json"

    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"Wrote {csv_path} and {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate standardized benchmark outputs")
    parser.add_argument("--results-root", required=True)
    args = parser.parse_args()
    run(args.results_root)


if __name__ == "__main__":
    main()
