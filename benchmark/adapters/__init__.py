from benchmark.adapters.fasterrcnn_adapter import FasterRCNNAdapter
from benchmark.adapters.ssd300_adapter import SSD300Adapter
from benchmark.adapters.yolov5_adapter import YOLOv5Adapter


def build_adapter(model_name: str):
    adapters = {
        "ssd300": SSD300Adapter,
        "fasterrcnn": FasterRCNNAdapter,
        "yolov5": YOLOv5Adapter,
    }
    if model_name not in adapters:
        raise ValueError(f"Unsupported model: {model_name}")
    return adapters[model_name]()
