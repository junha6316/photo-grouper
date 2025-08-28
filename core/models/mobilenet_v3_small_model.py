import torch.nn as nn
from torchvision import models, transforms
from .base_model import BaseModel


class MobileNetV3SmallModel(BaseModel):
    """MobileNetV3-Small model implementation for efficient feature extraction."""
    
    def get_model(self) -> nn.Module:
        # Load pre-trained MobileNetV3-Small
        base_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        
        # MobileNetV3 structure: features -> avgpool -> classifier
        # We want features + global average pooling, but not the classifier
        features = base_model.features
        avgpool = base_model.avgpool
        
        return nn.Sequential(
            features,      # Convolutional features
            avgpool,       # Global average pooling (7x7 -> 1x1)
            nn.Flatten()   # Flatten to 1D vector
        )
    
    def get_transforms(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def get_feature_dim(self) -> int:
        # MobileNetV3-Small outputs 576-dimensional features after global avg pooling
        return 576
    
    def get_model_name(self) -> str:
        return "mobilenetv3_small"
