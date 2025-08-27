import torch.nn as nn
from torchvision import models, transforms
from .base_model import BaseModel


class ResNet18Model(BaseModel):
    """ResNet18 model implementation."""
    
    def get_model(self) -> nn.Module:
        base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # ResNet doesn't have 'features' attribute like VGG
        # Remove the final classification layer and use all layers except fc
        layers = list(base_model.children())[:-1]  # Remove final fc layer
        return nn.Sequential(
            *layers,
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
        return "resnet18"
