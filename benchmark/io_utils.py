from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def get_git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
            .strip()
        )
    except Exception:
        return "unknown"


def get_environment_metadata() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": str(torch.backends.cudnn.version()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "os": platform.platform(),
    }


def build_experiment_metadata(config: dict[str, Any], cfg_hash: str) -> dict[str, Any]:
    run_name = config.get("run_name", "default_run")
    timestamp = datetime.now(timezone.utc).isoformat()
    run_id = f"{run_name}_{timestamp.replace(':', '').replace('-', '')}"
    return {
        "run_id": run_id,
        "timestamp": timestamp,
        "git_commit": get_git_commit(),
        "config_hash": cfg_hash,
    }


def ensure_schema(result: dict[str, Any]) -> None:
    required = {
        "experiment_metadata",
        "environment",
        "dataset_info",
        "model_info",
        "accuracy_metrics",
        "timing_metrics",
        "predictions",
        "comparison20_summary",
        "artifacts_index",
    }
    missing = required.difference(result)
    if missing:
        raise ValueError(f"Result schema missing fields: {sorted(missing)}")


def save_json(data: dict[str, Any], output_path: str | Path) -> str:
    output_path = str(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return output_path
