#!/usr/bin/env python3
"""
Export PyTorch models to ONNX format.

This script exports VGG16, ResNet18, and MobileNetV3-Small to ONNX format,
which can be used with ONNX Runtime for efficient inference.
"""

import os

import torch
import torch.nn as nn
from torchvision import models

OPSET_VERSION = 18


def _validate_onnx_model(output_path: str) -> None:
    # Verify the exported model
    import onnx

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX model validation passed")

    # Print model info
    print("\nModel info:")
    print(f"  Input shape: {onnx_model.graph.input[0].type.tensor_type.shape}")
    print(f"  Output shape: {onnx_model.graph.output[0].type.tensor_type.shape}")

    # Test inference with ONNX Runtime
    try:
        import numpy as np
        import onnxruntime as ort

        print("\nTesting ONNX Runtime inference...")
        session = ort.InferenceSession(output_path)

        # Run inference with dummy data
        dummy_input_np = np.random.randn(1, 3, 224, 224).astype(np.float32)
        outputs = session.run(None, {"input": dummy_input_np})

        print("✅ ONNX Runtime inference successful")
        print(f"  Output shape: {outputs[0].shape}")

    except ImportError:
        print("\n⚠️  onnxruntime not installed, skipping inference test")
        print("   Install with: pip install onnxruntime")


def _export_model(model: nn.Module, output_path: str) -> None:
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    model.eval()

    # Create dummy input (batch_size=1, channels=3, height=224, width=224)
    dummy_input = torch.randn(1, 3, 224, 224)

    print(f"Exporting model to {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=OPSET_VERSION,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    print(f"✅ Model exported successfully to {output_path}")
    _validate_onnx_model(output_path)


def export_vgg16_to_onnx(output_path: str = "models/vgg16.onnx") -> None:
    """
    Export VGG16 model to ONNX format.

    Args:
        output_path: Path where the ONNX model will be saved
    """
    print("Loading VGG16 model from torchvision...")
    # Load the same model architecture as used in vgg16_model.py
    base_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

    # Create the same model structure as in VGG16Model
    model = nn.Sequential(
        base_model.features,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
    )

    _export_model(model, output_path)


def export_resnet18_to_onnx(output_path: str = "models/resnet18.onnx") -> None:
    """
    Export ResNet18 model to ONNX format.

    Args:
        output_path: Path where the ONNX model will be saved
    """
    print("Loading ResNet18 model from torchvision...")
    base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    model = nn.Sequential(
        *list(base_model.children())[:-1],
        nn.Flatten(),
    )

    _export_model(model, output_path)


def export_mobilenetv3_small_to_onnx(output_path: str = "models/mobilenetv3_small.onnx") -> None:
    """
    Export MobileNetV3-Small model to ONNX format.

    Args:
        output_path: Path where the ONNX model will be saved
    """
    print("Loading MobileNetV3-Small model from torchvision...")
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    base_model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)

    model = nn.Sequential(
        base_model.features,
        base_model.avgpool,
        nn.Flatten(),
    )

    _export_model(model, output_path)


def export_all(output_dir: str = "models") -> None:
    export_vgg16_to_onnx(os.path.join(output_dir, "vgg16.onnx"))
    export_resnet18_to_onnx(os.path.join(output_dir, "resnet18.onnx"))
    export_mobilenetv3_small_to_onnx(os.path.join(output_dir, "mobilenetv3_small.onnx"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export PyTorch models to ONNX")
    parser.add_argument(
        "--model",
        type=str,
        choices=["vgg16", "resnet18", "mobilenetv3_small"],
        default="vgg16",
        help="Model name to export (default: vgg16)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for ONNX model (overrides --output-dir)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Output directory for ONNX models (default: models)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Export all models",
    )

    args = parser.parse_args()

    if args.all:
        export_all(args.output_dir)
    else:
        output_path = args.output or os.path.join(args.output_dir, f"{args.model}.onnx")
        if args.model == "vgg16":
            export_vgg16_to_onnx(output_path)
        elif args.model == "resnet18":
            export_resnet18_to_onnx(output_path)
        elif args.model == "mobilenetv3_small":
            export_mobilenetv3_small_to_onnx(output_path)
