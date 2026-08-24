from __future__ import annotations

import json
from pathlib import Path


def load_comparison_image_ids(comparison_images_path: str | Path) -> list[str]:
    with open(comparison_images_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        image_ids = data
    elif isinstance(data, dict):
        image_ids = data.get("image_ids") or data.get("images") or []
    else:
        raise ValueError("comparison_images.json must be a list or object")

    normalized: list[str] = []
    for item in image_ids:
        if isinstance(item, str):
            normalized.append(item)
        elif isinstance(item, dict) and "image_id" in item:
            normalized.append(item["image_id"])
        else:
            raise ValueError("Invalid comparison image entry")

    if len(normalized) != 20:
        raise ValueError(f"Expected exactly 20 comparison images, found {len(normalized)}")
    return normalized
