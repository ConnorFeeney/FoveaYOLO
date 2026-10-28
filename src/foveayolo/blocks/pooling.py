# Extracted from Ultralytics 8.4.191, AGPL-3.0: https://ultralytics.com/license
"""Spatial pyramid pooling."""

from typing import cast, override

import torch
from torch import nn

from .conv import Conv


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(
        self, c1: int, c2: int, k: int = 5, n: int = 3, shortcut: bool = False
    ) -> None:
        """Initialize the SPPF layer with given input/output channels and kernel size."""
        super().__init__()
        c_ = c1 // 2
        self.cv1: Conv = Conv(c1, c_, 1, 1, act=False)
        self.cv2: Conv = Conv(c_ * (n + 1), c2, 1, 1)
        self.m: nn.MaxPool2d = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n: int = n
        self.add: bool = shortcut and c1 == c2

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sequential pooling operations to input and return concatenated feature maps."""
        y = [cast(torch.Tensor, self.cv1(x))]
        y.extend(
            cast(torch.Tensor, self.m(y[-1])) for _ in range(getattr(self, "n", 3))
        )
        y = cast(torch.Tensor, self.cv2(torch.cat(y, 1)))
        return y + x if getattr(self, "add", False) else y
