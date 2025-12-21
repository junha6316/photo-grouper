#!/usr/bin/env python3
"""
Export PyTorch VGG16 model to ONNX format.

This script extracts the VGG16 model weights and exports them to ONNX format,
which can be used with ONNX Runtime for more efficient inference.
"""

import torch
import torch.nn as nn
from torchvision import models
import os

def export_vgg16_to_onnx(output_path: str = "models/vgg16.onnx"):
    """
    Export VGG16 model to ONNX format.

    Args:
        output_path: Path where the ONNX model will be saved
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("Loading VGG16 model from torchvision...")
    # Load the same model architecture as used in vgg16_model.py
    base_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

    # Create the same model structure as in VGG16Model
    model = nn.Sequential(
        base_model.features,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten()
    )

    model.eval()

    # Create dummy input (batch_size=1, channels=3, height=224, width=224)
    dummy_input = torch.randn(1, 3, 224, 224)

    print(f"Exporting model to {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,  # Use opset 14 for better compatibility
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"✅ Model exported successfully to {output_path}")

    # Verify the exported model
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX model validation passed")

    # Print model info
    print(f"\nModel info:")
    print(f"  Input shape: {onnx_model.graph.input[0].type.tensor_type.shape}")
    print(f"  Output shape: {onnx_model.graph.output[0].type.tensor_type.shape}")

    # Test inference with ONNX Runtime
    try:
        import onnxruntime as ort
        import numpy as np

        print("\nTesting ONNX Runtime inference...")
        session = ort.InferenceSession(output_path)

        # Run inference with dummy data
        dummy_input_np = np.random.randn(1, 3, 224, 224).astype(np.float32)
        outputs = session.run(None, {'input': dummy_input_np})

        print(f"✅ ONNX Runtime inference successful")
        print(f"  Output shape: {outputs[0].shape}")

    except ImportError:
        print("\n⚠️  onnxruntime not installed, skipping inference test")
        print("   Install with: pip install onnxruntime")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export VGG16 model to ONNX")
    parser.add_argument(
        "--output",
        type=str,
        default="models/vgg16.onnx",
        help="Output path for ONNX model (default: models/vgg16.onnx)"
    )

    args = parser.parse_args()
    export_vgg16_to_onnx(args.output)
