from __future__ import annotations

from typing import Any

from ultralytics import YOLO

from benchmark.adapters.base_adapter import BaseModelAdapter
from benchmark.data.class_map import VOC_CLASSES
from benchmark.schemas import StandardPrediction


class YOLOv5Adapter(BaseModelAdapter):
    model_name = "yolov5"

    def __init__(self):
        self.model = None

    def load_model(self, config: dict[str, Any]) -> Any:
        model_cfg = config.get("model_configs", {}).get(self.model_name, {})
        weights_path = model_cfg.get("weights", "yolov5su.pt")
        self.model = YOLO(weights_path)
        return self.model

    def preprocess(self, image, config: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        return image, {"orig_size": image.size}

    def infer(self, model_input: Any) -> Any:
        return self.model.predict(model_input, verbose=False)

    def postprocess(
        self, raw_output: Any, config: dict[str, Any], preprocess_meta: dict[str, Any]
    ) -> list[dict[str, Any]]:
        conf_threshold = float(config.get("conf_threshold", 0.25))
        result = raw_output[0]

        detections: list[dict[str, Any]] = []
        for box in result.boxes:
            score = float(box.conf.item())
            if score < conf_threshold:
                continue
            cls = int(box.cls.item())
            if cls < 0 or cls >= len(VOC_CLASSES):
                continue
            xyxy = [float(v) for v in box.xyxy[0].tolist()]
            detections.append({"bbox_xyxy": xyxy, "class_id": cls, "score": score})
        return detections

    def to_standard_prediction(self, image_id: str, detections: Any) -> list[StandardPrediction]:
        return [
            StandardPrediction(
                image_id=image_id,
                class_id=det["class_id"],
                class_name=VOC_CLASSES[det["class_id"]],
                bbox_xyxy=det["bbox_xyxy"],
                score=det["score"],
            )
            for det in detections
        ]

    def get_model_metadata(self) -> dict[str, Any]:
        return {
            "name": self.model_name,
            "framework": "ultralytics",
            "checkpoint": str(self.model.ckpt_path) if self.model is not None else "unloaded",
            "input_policy": "ultralytics internal preprocessing",
        }
