# Extracted from Ultralytics 8.4.191, AGPL-3.0: https://ultralytics.com/license
"""Distribution-to-distance projection used by the shared detection head."""

from typing import cast, override

import torch
from torch import nn


class DFL(nn.Module):
    """Integral module of Distribution Focal Loss (DFL)."""

    def __init__(self, c1: int = 16) -> None:
        """Initialize the DFL module with a fixed, non-trainable integral convolution."""
        super().__init__()
        self.conv: nn.Conv2d = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1: int = c1

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode distributions of shape (B, 4 * c1, A) into expected box distances of shape (B, 4, A)."""
        b, _, a = x.shape
        return cast(
            torch.Tensor, self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1))
        ).view(b, 4, a)
