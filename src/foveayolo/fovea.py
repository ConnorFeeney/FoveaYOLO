from typing import cast, override

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import Conv


class Saliency(nn.Module):
    """Combine feature maps into spatial probability maps; largest input first."""

    def __init__(self, in_channels: list[int], out_channels: int) -> None:
        super().__init__()
        if not in_channels:
            raise ValueError("in_channels must contain at least one channel count")
        self.convs: nn.ModuleList = nn.ModuleList(
            Conv(c, out_channels) for c in in_channels
        )

    @override
    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Return [B, out_channels, H, W], normalized over H*W per channel."""
        if len(x) != len(self.convs):
            raise ValueError("Expected one input tensor per convolution")
        target_size = x[0].shape[-2:]
        maps = [
            F.interpolate(
                cast(torch.Tensor, conv(feature)), size=target_size, mode="nearest"
            )
            for conv, feature in zip(self.convs, x, strict=True)
        ]
        combined = torch.stack(maps, dim=0).sum(dim=0)
        return combined.flatten(2).softmax(dim=-1).reshape_as(combined)
