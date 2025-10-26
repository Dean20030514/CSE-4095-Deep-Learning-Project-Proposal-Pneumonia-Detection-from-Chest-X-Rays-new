from typing import Tuple
import torch
import torch.nn as nn
from torchvision import models


def build_model(name: str, num_classes: int) -> Tuple[nn.Module, int]:
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
        except Exception:
            # Fallback to resnet18 if efficientnet is unavailable
            net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            in_feats = net.fc.in_features
            net.fc = nn.Linear(in_feats, num_classes)
            return net, 224
    raise ValueError(f"Unknown model name: {name}")
