"""Single, fail-loud loader for the standardized comparison-image manifest (P0 fix).

Audit finding this addresses: YOLOv5's comparison-testing cell never read
comparison_images.json at all -- it was hardcoded to one image that isn't
even in the manifest. FasterRCNN/SSD300 read the manifest correctly but
via a multi-path guess-and-silently-fall-back pattern that could load a
smaller default set without the run's output making that obvious. This
loader replaces both: one deterministic search, and a hard failure
instead of a silent partial/default set.
"""
import json
from pathlib import Path


def load_comparison_manifest(search_paths):
    for path in search_paths:
        path = Path(path)
        if path.exists():
            with open(path, "r") as f:
                data = json.load(f)
            images = data.get("images", [])
            expected = data.get("total_images", len(images))
            print(f"Loaded comparison manifest: {path} -- {len(images)}/{expected} images")
            if len(images) != expected:
                raise ValueError(
                    f"Comparison manifest at {path} is malformed: expected "
                    f"{expected} images, found {len(images)}"
                )
            return images

    raise FileNotFoundError(
        "comparison_images.json not found in any of: "
        + ", ".join(str(p) for p in search_paths)
        + ". All three models must evaluate on the exact same image set -- "
        "silently falling back to a smaller default set is not permitted."
    )
