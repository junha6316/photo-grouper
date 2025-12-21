from abc import ABC, abstractmethod
import sys
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
from PIL import Image


class ONNXBaseModel(ABC):
    """Abstract base class for ONNX feature extraction models."""

    @abstractmethod
    def get_onnx_path(self) -> str:
        """Return the ONNX model path."""
        raise NotImplementedError

    @abstractmethod
    def get_preprocessing_fn(self) -> Callable[[Image.Image], np.ndarray]:
        """Return the image preprocessing function."""
        raise NotImplementedError

    @abstractmethod
    def get_feature_dim(self) -> int:
        """Return the feature dimension."""
        raise NotImplementedError

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model name for caching."""
        raise NotImplementedError

    @abstractmethod
    def get_input_shape(self) -> Tuple[int, int, int]:
        """Return the expected input shape (C, H, W)."""
        raise NotImplementedError

    def _models_dir(self) -> Path:
        base_path = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2]))
        return base_path / "models"

    def _imagenet_preprocess(self, image: Image.Image) -> np.ndarray:
        _, height, width = self.get_input_shape()
        image = image.convert("RGB").resize((width, height), Image.BILINEAR)

        img_array = np.asarray(image).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_array = (img_array - mean) / std
        img_array = np.transpose(img_array, (2, 0, 1))

        return np.ascontiguousarray(img_array, dtype=np.float32)
