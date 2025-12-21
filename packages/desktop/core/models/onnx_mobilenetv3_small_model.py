from typing import Tuple

from .onnx_base_model import ONNXBaseModel


class ONNXMobileNetV3SmallModel(ONNXBaseModel):
    """MobileNetV3-Small ONNX model implementation."""

    def get_onnx_path(self) -> str:
        return str(self._models_dir() / "mobilenetv3_small.onnx")

    def get_preprocessing_fn(self):
        return self._imagenet_preprocess

    def get_feature_dim(self) -> int:
        return 576

    def get_model_name(self) -> str:
        return "mobilenetv3_small"

    def get_input_shape(self) -> Tuple[int, int, int]:
        return (3, 224, 224)
