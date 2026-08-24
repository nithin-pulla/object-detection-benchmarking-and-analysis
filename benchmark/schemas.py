from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class StandardPrediction:
    image_id: str
    class_id: int
    class_name: str
    bbox_xyxy: list[float]
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TimingBreakdown:
    preprocessing_ms: float
    inference_ms: float
    postprocessing_ms: float

    @property
    def total_ms(self) -> float:
        return self.preprocessing_ms + self.inference_ms + self.postprocessing_ms

    def to_dict(self) -> dict[str, float]:
        return {
            "preprocessing_ms": self.preprocessing_ms,
            "inference_ms": self.inference_ms,
            "postprocessing_ms": self.postprocessing_ms,
            "total_ms": self.total_ms,
        }


def validate_prediction_dict(pred: dict[str, Any]) -> None:
    required = {"image_id", "class_id", "class_name", "bbox_xyxy", "score"}
    missing = required.difference(pred)
    if missing:
        raise ValueError(f"Missing prediction fields: {sorted(missing)}")
    if len(pred["bbox_xyxy"]) != 4:
        raise ValueError("bbox_xyxy must contain 4 values")
