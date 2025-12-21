from typing import Tuple

from .onnx_base_model import ONNXBaseModel


class ONNXVGGModel(ONNXBaseModel):
    """VGG16 ONNX model implementation."""

    def get_onnx_path(self) -> str:
        return str(self._models_dir() / "vgg16.onnx")

    def get_preprocessing_fn(self):
        return self._imagenet_preprocess

    def get_feature_dim(self) -> int:
        return 512

    def get_model_name(self) -> str:
        return "vgg16"

    def get_input_shape(self) -> Tuple[int, int, int]:
        return (3, 224, 224)
