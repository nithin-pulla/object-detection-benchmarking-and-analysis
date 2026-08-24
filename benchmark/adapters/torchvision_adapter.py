from __future__ import annotations

from typing import Any, Callable

import torch
import torchvision
from PIL import Image
from torchvision.transforms import ToTensor

from benchmark.adapters.base_adapter import BaseModelAdapter
from benchmark.data.class_map import VOC_CLASSES
from benchmark.schemas import StandardPrediction


class TorchVisionDetectorAdapter(BaseModelAdapter):
    def __init__(self, model_name: str, model_builder: Callable[..., Any], weights_enum: Any):
        self.model_name = model_name
        self._model_builder = model_builder
        self._weights_enum = weights_enum
        self.model = None
        self.device = "cpu"
        self.to_tensor = ToTensor()

    def load_model(self, config: dict[str, Any]) -> Any:
        model_cfg = config.get("model_configs", {}).get(self.model_name, {})
        use_pretrained = model_cfg.get("pretrained", True)
        weights = self._weights_enum.DEFAULT if use_pretrained else None
        self.model = self._model_builder(weights=weights)
        self.device = config.get("device", "cpu")
        self.model.to(self.device)
        self.model.eval()
        return self.model

    def preprocess(self, image: Image.Image, config: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        tensor = self.to_tensor(image).to(self.device)
        return [tensor], {"orig_size": image.size}

    def infer(self, model_input: Any) -> Any:
        with torch.no_grad():
            return self.model(model_input)

    def postprocess(
        self, raw_output: Any, config: dict[str, Any], preprocess_meta: dict[str, Any]
    ) -> list[dict[str, Any]]:
        conf_threshold = float(config.get("conf_threshold", 0.25))
        output = raw_output[0]
        boxes = output["boxes"].detach().cpu()
        labels = output["labels"].detach().cpu()
        scores = output["scores"].detach().cpu()

        detections: list[dict[str, Any]] = []
        for box, label, score in zip(boxes, labels, scores):
            if float(score) < conf_threshold:
                continue
            detections.append(
                {
                    "bbox_xyxy": [float(v) for v in box.tolist()],
                    "class_id": int(label) - 1,
                    "score": float(score),
                }
            )
        return detections

    def to_standard_prediction(self, image_id: str, detections: Any) -> list[StandardPrediction]:
        preds: list[StandardPrediction] = []
        for det in detections:
            class_id = det["class_id"]
            if class_id < 0 or class_id >= len(VOC_CLASSES):
                continue
            preds.append(
                StandardPrediction(
                    image_id=image_id,
                    class_id=class_id,
                    class_name=VOC_CLASSES[class_id],
                    bbox_xyxy=det["bbox_xyxy"],
                    score=det["score"],
                )
            )
        return preds

    def get_model_metadata(self) -> dict[str, Any]:
        return {
            "name": self.model_name,
            "framework": "torchvision",
            "checkpoint": "torchvision default weights",
            "input_policy": "native torchvision transform",
        }
