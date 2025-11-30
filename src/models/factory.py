from typing import Tuple
import torch
import torch.nn as nn
from torchvision import models


def build_model(name: str, num_classes: int) -> Tuple[nn.Module, int]:
    """Build a pretrained classification model for chest X-ray analysis.
    
    Loads a model with ImageNet pretrained weights and replaces the final
    classification layer to match the target number of classes.
    
    Args:
        name: Model architecture name. Supported:
            - 'resnet18', 'resnet50': ResNet variants
            - 'efficientnet_b0', 'efficientnet_b2': EfficientNet variants
            - 'densenet121': DenseNet-121
        num_classes: Number of output classes (typically 2 for NORMAL/PNEUMONIA)
    
    Returns:
        model: Initialized PyTorch model with modified classifier
        img_size: Recommended input image size for this architecture
    
    Raises:
        ValueError: If model name is not recognized
    
    Example:
        >>> model, img_size = build_model('resnet18', num_classes=2)
        >>> print(img_size)  # 224
    """
    name = name.lower()
    if name in ["resnet18", "resnet-18"]:
        net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_feats = net.fc.in_features
        net.fc = nn.Linear(in_feats, num_classes)
        return net, 224
    if name in ["resnet50", "resnet-50"]:
        net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_feats = net.fc.in_features
        net.fc = nn.Linear(in_feats, num_classes)
        return net, 224
    if name in ["efficientnet_b0", "effb0", "efficientnet-b0"]:
        try:
            net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            in_feats = net.classifier[-1].in_features
            net.classifier[-1] = nn.Linear(in_feats, num_classes)
            return net, 224
        except (ImportError, AttributeError) as e:
            # Fallback to resnet18 if efficientnet is unavailable
            print(f"[WARNING] EfficientNet-B0 not available ({e}), falling back to ResNet18")
            net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            in_feats = net.fc.in_features
            net.fc = nn.Linear(in_feats, num_classes)
            return net, 224
    if name in ["efficientnet_b2", "effb2", "efficientnet-b2"]:
        try:
            net = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
            in_feats = net.classifier[-1].in_features
            net.classifier[-1] = nn.Linear(in_feats, num_classes)
            return net, 260  # EfficientNet-B2 default input size
        except (ImportError, AttributeError) as e:
            # Fallback to efficientnet_b0
            print(f"[WARNING] EfficientNet-B2 not available ({e}), falling back to EfficientNet-B0")
            net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            in_feats = net.classifier[-1].in_features
            net.classifier[-1] = nn.Linear(in_feats, num_classes)
            return net, 224
    if name in ["densenet121", "densenet-121"]:
        net = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        in_feats = net.classifier.in_features
        net.classifier = nn.Linear(in_feats, num_classes)
        return net, 224
    raise ValueError(f"Unknown model name: {name}")
