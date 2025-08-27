from abc import ABC, abstractmethod
import torch.nn as nn
from torchvision import transforms


class BaseModel(ABC):
    """Abstract base class for feature extraction models."""
    
    @abstractmethod
    def get_model(self) -> nn.Module:
        """Return the PyTorch model."""
        pass
    
    @abstractmethod
    def get_transforms(self) -> transforms.Compose:
        """Return the image preprocessing transforms."""
        pass
    
    @abstractmethod
    def get_feature_dim(self) -> int:
        """Return the feature dimension."""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model name for caching."""
        pass
