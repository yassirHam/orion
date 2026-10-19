"""Baseline models (O1.3): a fine-tuned ResNet18 or a small CNN."""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


def build_model(name: str, num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    name = name.lower()
    if name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.resnet18(weights=weights)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        return net
    if name == "small_cnn":
        return SmallCNN(num_classes=num_classes)
    raise ValueError(f"unknown model '{name}' (use resnet18 or small_cnn)")


class SmallCNN(nn.Module):
    """A compact CNN usable with no pretrained weights / no network access."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
