from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_accuracy_speed(results_by_model: dict, out_path: str | Path) -> str:
    plt.figure(figsize=(7, 5))
    for model, result in results_by_model.items():
        acc = result["accuracy_metrics"]["aggregate"]["mAP@0.5:0.95"]
        latency = result["timing_metrics"]["total_ms"]
        plt.scatter(latency, acc, label=model)
        plt.text(latency, acc, model)

    plt.xlabel("Total latency (ms)")
    plt.ylabel("mAP@0.5:0.95")
    plt.title("Accuracy-speed tradeoff")
    plt.grid(True, alpha=0.2)
    plt.legend()
    out_path = str(out_path)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path
