from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from benchmark.schemas import StandardPrediction


class BaseModelAdapter(ABC):
    model_name: str

    @abstractmethod
    def load_model(self, config: dict[str, Any]) -> Any:
        raise NotImplementedError

    @abstractmethod
    def preprocess(self, image, config: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def infer(self, model_input: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def postprocess(
        self, raw_output: Any, config: dict[str, Any], preprocess_meta: dict[str, Any]
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def to_standard_prediction(self, image_id: str, detections: Any) -> list[StandardPrediction]:
        raise NotImplementedError

    @abstractmethod
    def get_model_metadata(self) -> dict[str, Any]:
        raise NotImplementedError
