from .onnx_base_model import ONNXBaseModel
from .onnx_vgg16_model import ONNXVGGModel
from .onnx_resnet18_model import ONNXResNet18Model
from .onnx_mobilenetv3_small_model import ONNXMobileNetV3SmallModel

__all__ = [
    "ONNXBaseModel",
    "ONNXVGGModel",
    "ONNXResNet18Model",
    "ONNXMobileNetV3SmallModel",
]

try:
    from .base_model import BaseModel
    from .vgg16_model import VGG16Model
    from .resnet18_model import ResNet18Model
    from .mobilenet_v3_small_model import MobileNetV3SmallModel

    __all__ += [
        "BaseModel",
        "VGG16Model",
        "ResNet18Model",
        "MobileNetV3SmallModel",
    ]
except ModuleNotFoundError:
    pass
