import torch.nn as nn
from torchvision import models, transforms
from .base_model import BaseModel


class VGG16Model(BaseModel):
    """VGG16 model implementation."""
    
    def get_model(self) -> nn.Module:
        
        base_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        return nn.Sequential(
            base_model.features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
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
        return 512
    
    def get_model_name(self) -> str:
        return "vgg16"

