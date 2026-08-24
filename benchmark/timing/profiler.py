from __future__ import annotations

import statistics
import time
from typing import Any

import torch

from benchmark.schemas import TimingBreakdown


def _sync_if_cuda(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def profile_single_image(
    adapter,
    model,
    image,
    config: dict[str, Any],
) -> tuple[list, TimingBreakdown]:
    device = str(config.get("device", "cpu"))

    t0 = time.perf_counter()
    model_input, meta = adapter.preprocess(image, config)
    _sync_if_cuda(device)
    t1 = time.perf_counter()

    raw = adapter.infer(model_input)
    _sync_if_cuda(device)
    t2 = time.perf_counter()

    detections = adapter.postprocess(raw, config, meta)
    _sync_if_cuda(device)
    t3 = time.perf_counter()

    timing = TimingBreakdown(
        preprocessing_ms=(t1 - t0) * 1000.0,
        inference_ms=(t2 - t1) * 1000.0,
        postprocessing_ms=(t3 - t2) * 1000.0,
    )
    return detections, timing


def warmup(adapter, model, image, config: dict[str, Any]) -> None:
    for _ in range(int(config.get("warmup_runs", 3))):
        profile_single_image(adapter, model, image, config)


def summarize_timings(per_image: list[dict[str, float]]) -> dict[str, float]:
    def avg(key: str) -> float:
        return float(statistics.mean(item[key] for item in per_image)) if per_image else 0.0

    def std(key: str) -> float:
        values = [item[key] for item in per_image]
        return float(statistics.pstdev(values)) if len(values) > 1 else 0.0

    total_ms = avg("total_ms")
    infer_ms = avg("inference_ms")
    return {
        "preprocessing_ms": avg("preprocessing_ms"),
        "inference_ms": infer_ms,
        "postprocessing_ms": avg("postprocessing_ms"),
        "total_ms": total_ms,
        "fps_total": (1000.0 / total_ms) if total_ms > 0 else 0.0,
        "fps_inference_only": (1000.0 / infer_ms) if infer_ms > 0 else 0.0,
        "total_ms_std": std("total_ms"),
        "inference_ms_std": std("inference_ms"),
    }
