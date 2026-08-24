from __future__ import annotations

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image


def _draw_boxes(ax, boxes, color, label_prefix):
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box["bbox_xyxy"]
        score = box.get("score")
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        text = f"{label_prefix}:{box['class_name']}"
        if score is not None:
            text += f" {score:.2f}"
        ax.text(x1, max(0, y1 - 3), text, color=color, fontsize=7, backgroundcolor="black")


def save_comparison_panel(
    image_path: str,
    gt: list[dict],
    preds_by_model: dict[str, list[dict]],
    out_path: str | Path,
) -> str:
    image = Image.open(image_path).convert("RGB")

    cols = 1 + len(preds_by_model)
    fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 5))
    if cols == 1:
        axes = [axes]

    axes[0].imshow(image)
    axes[0].set_title("Ground Truth")
    _draw_boxes(axes[0], gt, "lime", "GT")
    axes[0].axis("off")

    for idx, (model_name, preds) in enumerate(preds_by_model.items(), start=1):
        axes[idx].imshow(image)
        axes[idx].set_title(model_name)
        _draw_boxes(axes[idx], preds, "red", "P")
        axes[idx].axis("off")

    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path
