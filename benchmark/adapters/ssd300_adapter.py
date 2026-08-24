import torchvision

from benchmark.adapters.torchvision_adapter import TorchVisionDetectorAdapter


class SSD300Adapter(TorchVisionDetectorAdapter):
    def __init__(self):
        super().__init__(
            model_name="ssd300",
            model_builder=torchvision.models.detection.ssd300_vgg16,
            weights_enum=torchvision.models.detection.SSD300_VGG16_Weights,
        )
