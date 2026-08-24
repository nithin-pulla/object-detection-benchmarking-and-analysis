import torchvision

from benchmark.adapters.torchvision_adapter import TorchVisionDetectorAdapter


class FasterRCNNAdapter(TorchVisionDetectorAdapter):
    def __init__(self):
        super().__init__(
            model_name="fasterrcnn",
            model_builder=torchvision.models.detection.fasterrcnn_resnet50_fpn,
            weights_enum=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights,
        )
