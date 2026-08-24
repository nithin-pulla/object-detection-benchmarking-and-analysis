"""Benchmark-controlled protocol constants (P0 scope).

These values MUST be identical across SSD300, Faster R-CNN, and YOLOv5s
for any run that feeds the primary comparison table. Model-specific
parameters (architecture, optimizer, input resolution, batch size, etc.)
stay in each notebook's own CONFIG dict and are intentionally NOT part of
this file -- see the "Model configuration and training" section of the
benchmarking roadmap for the full benchmark-controlled vs. model-specific
split.
"""

GLOBAL_SEED = 42

# Shared evaluation-time operating point. Every model's headline
# precision/recall/F1 numbers (benchmark_common.eval_coco) must be
# computed at this single confidence/IoU threshold. Per-model "optimal"
# thresholds belong in a separate, explicitly-labeled sensitivity
# ablation -- never the default reported number.
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5

# Shared speed-benchmark methodology (benchmark_common.speed_bench).
WARMUP_ITERS = 10
TIMED_ITERS = 100

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
