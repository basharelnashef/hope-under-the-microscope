"""VGG-8 architecture described in HOPE Appendix G.2.

This module defines only the network structure. A dataset, training schedule,
checkpoint, and evaluation protocol are separate experimental choices. Using
this class does not by itself reproduce a reported paper result.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VGG8(nn.Module):
    """Appendix G.2 VGG-style network with a configurable output head."""

    def __init__(self, n_classes: int = 10) -> None:
        super().__init__()

        def block(cin: int, cout: int) -> list[nn.Module]:
            return [
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            ]

        layers: list[nn.Module] = []
        layers += block(3, 128) + [nn.MaxPool2d(2)]
        layers += block(128, 256) + [nn.MaxPool2d(2)]
        layers += block(256, 512) + block(512, 512) + [nn.MaxPool2d(2)]
        layers += block(512, 512) + block(512, 512) + [nn.MaxPool2d(2)]
        self.features = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.bottleneck = nn.Linear(512, 512, bias=False)
        self.bn_b = nn.BatchNorm1d(512)
        self.act_b = nn.ReLU(inplace=True)
        self.head = nn.Linear(512, n_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        features = self.gap(features).flatten(1)
        features = self.act_b(self.bn_b(self.bottleneck(features)))
        return self.head(features)
