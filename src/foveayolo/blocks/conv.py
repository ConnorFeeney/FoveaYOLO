# Extracted from Ultralytics 8.4.191, AGPL-3.0: https://ultralytics.com/license
"""Convolution, depth-wise convolution, and concatenation."""

import math
from typing import cast, override

import torch
from torch import nn

type Size2 = int | tuple[int, int]


def autopad(k: Size2, p: Size2 | None = None, d: int = 1) -> Size2:
    """Compute padding for 'same' shape outputs."""
    if d > 1:
        k = (
            d * (k - 1) + 1
            if isinstance(k, int)
            else (d * (k[0] - 1) + 1, d * (k[1] - 1) + 1)
        )
    if p is None:
        p = k // 2 if isinstance(k, int) else (k[0] // 2, k[1] // 2)
    return p


class Conv(nn.Module):
    """Standard convolution module with batch normalization and activation."""

    default_act: nn.Module = nn.SiLU()

    def __init__(
        self,
        c1: int,
        c2: int,
        k: Size2 = 1,
        s: Size2 = 1,
        p: Size2 | None = None,
        g: int = 1,
        d: int = 1,
        act: bool | nn.Module = True,
    ) -> None:
        super().__init__()
        self.conv: nn.Conv2d = nn.Conv2d(
            c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False
        )
        self.bn: nn.BatchNorm2d = nn.BatchNorm2d(c2)
        self.act: nn.Module = (
            self.default_act
            if act is True
            else act
            if isinstance(act, nn.Module)
            else nn.Identity()
        )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution, batch normalization, and activation."""
        convolved = cast(torch.Tensor, self.conv(x))
        normalized = cast(torch.Tensor, self.bn(convolved))
        return cast(torch.Tensor, self.act(normalized))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution and activation without batch normalization."""
        return cast(torch.Tensor, self.act(cast(torch.Tensor, self.conv(x))))


class DWConv(Conv):
    """Depth-wise convolution module."""

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        d: int = 1,
        act: bool | nn.Module = True,
    ) -> None:
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class Concat(nn.Module):
    """Concatenate a list of tensors along a specified dimension."""

    def __init__(self, dimension: int = 1) -> None:
        super().__init__()
        self.d: int = dimension

    @override
    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(x, self.d)
