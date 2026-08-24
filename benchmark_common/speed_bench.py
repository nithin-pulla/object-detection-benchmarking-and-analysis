"""Shared inference-speed timing harness (P0 fix).

Audit finding this addresses: SSD300 had no speed benchmark at all,
FasterRCNN's aggregate FPS bundled DataLoader/disk-I/O time into
"inference" time, and only YOLOv5 isolated preprocessing/inference/
postprocessing with a warm-up loop. This harness generalizes YOLOv5's
(already-correct) approach so all three models are timed identically --
no model can gain an FPS advantage from a looser timing methodology.

Every caller supplies three callables operating on ONE already-loaded,
in-memory image (no disk I/O inside the timed loop):
    preprocess_fn(image) -> model_input
    infer_fn(model_input) -> raw_output
    postprocess_fn(raw_output) -> detections
"""
import time

import numpy as np
import torch


def _is_cuda(device):
    if isinstance(device, torch.device):
        return device.type == "cuda"
    return str(device) == "cuda"


def benchmark_speed(preprocess_fn, infer_fn, postprocess_fn, images, device,
                     warmup_iters=10, timed_iters=100):
    images = list(images)
    if not images:
        raise ValueError("benchmark_speed requires at least one image")

    if len(images) < timed_iters:
        # Cycle through the available images rather than silently timing
        # fewer iterations than the shared protocol specifies.
        reps = (timed_iters + len(images) - 1) // len(images)
        images = (images * reps)[:timed_iters]
    else:
        images = images[:timed_iters]

    def _sync():
        if _is_cuda(device):
            torch.cuda.synchronize()

    # Warm-up (discarded): excludes CUDA kernel JIT/cache-warming effects
    # from the reported numbers.
    for img in images[:min(warmup_iters, len(images))]:
        model_input = preprocess_fn(img)
        _sync()
        raw = infer_fn(model_input)
        _sync()
        _ = postprocess_fn(raw)
        _sync()

    prep_times, infer_times, post_times, total_times = [], [], [], []
    for img in images:
        t0 = time.perf_counter()
        model_input = preprocess_fn(img)
        _sync()
        t1 = time.perf_counter()

        raw = infer_fn(model_input)
        _sync()
        t2 = time.perf_counter()

        _ = postprocess_fn(raw)
        _sync()
        t3 = time.perf_counter()

        prep_times.append(t1 - t0)
        infer_times.append(t2 - t1)
        post_times.append(t3 - t2)
        total_times.append(t3 - t0)

    total_ms_arr = np.array(total_times) * 1000.0
    total_wall_time = float(sum(total_times))
    mean_total_ms = float(np.mean(total_ms_arr))

    return {
        "preprocessing_ms": float(np.mean(prep_times) * 1000.0),
        "inference_ms": float(np.mean(infer_times) * 1000.0),
        "postprocessing_ms": float(np.mean(post_times) * 1000.0),
        "total_ms": mean_total_ms,
        "latency_percentiles_ms": {
            "p50": float(np.percentile(total_ms_arr, 50)),
            "p95": float(np.percentile(total_ms_arr, 95)),
        },
        "fps_latency": float(1000.0 / mean_total_ms) if mean_total_ms > 0 else 0.0,
        "fps_throughput": float(len(images) / total_wall_time) if total_wall_time > 0 else 0.0,
        "num_warmup_iters": warmup_iters,
        "num_timed_iters": len(images),
    }
