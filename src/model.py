from __future__ import annotations

import torch
from torch import nn


class SmallPatchCNN(nn.Module):
    """Compact CNN for 13x13 multispectral patches."""

    def __init__(self, in_channels: int, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 13->6
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MetaMLP(nn.Module):
    def __init__(self, in_dim: int = 2, out_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNNMetaClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        meta_dim: int = 2,
        cnn_dim: int = 128,
        meta_out_dim: int = 32,
        head_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.cnn = SmallPatchCNN(in_channels=in_channels, out_dim=cnn_dim)
        self.meta = MetaMLP(in_dim=meta_dim, out_dim=meta_out_dim)

        self.head = nn.Sequential(
            nn.Linear(cnn_dim + meta_out_dim, head_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_dim, num_classes),
        )

    def forward(self, patch: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        x_img = self.cnn(patch)
        x_meta = self.meta(meta)
        x = torch.cat([x_img, x_meta], dim=1)
        return self.head(x)
