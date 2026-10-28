# Extracted from Ultralytics 8.4.191, AGPL-3.0: https://ultralytics.com/license
"""CSP bottlenecks and the C3k2 block family."""

from typing import cast, override

import torch
from torch import nn

from .attention import PSABlock
from .conv import Conv, Size2


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        k: tuple[Size2, Size2] = (3, 3),
        e: float = 0.5,
    ) -> None:
        """Initialize a standard bottleneck module."""
        super().__init__()
        c_ = int(c2 * e)
        self.cv1: Conv = Conv(c1, c_, k[0], 1)
        self.cv2: Conv = Conv(c_, c2, k[1], 1, g=g)
        self.add: bool = shortcut and c1 == c2

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bottleneck with optional shortcut connection."""
        return (
            x + cast(torch.Tensor, self.cv2(cast(torch.Tensor, self.cv1(x))))
            if self.add
            else cast(torch.Tensor, self.cv2(cast(torch.Tensor, self.cv1(x))))
        )


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
    ) -> None:
        """Initialize a CSP bottleneck with 2 convolutions."""
        super().__init__()
        self.c: int = int(c2 * e)
        self.cv1: Conv = Conv(c1, 2 * self.c, 1, 1)
        self.cv2: Conv = Conv((2 + n) * self.c, c2, 1)
        self.m: nn.ModuleList = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f layer."""
        y = list(cast(torch.Tensor, self.cv1(x)).chunk(2, 1))
        y.extend(cast(torch.Tensor, m(y[-1])) for m in self.m)
        return cast(torch.Tensor, self.cv2(torch.cat(y, 1)))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = torch.split(cast(torch.Tensor, self.cv1(x)), [self.c, self.c], 1)
        y = [y[0], y[1]]
        y.extend(cast(torch.Tensor, m(y[-1])) for m in self.m)
        return cast(torch.Tensor, self.cv2(torch.cat(y, 1)))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
    ) -> None:
        """Initialize the CSP Bottleneck with 3 convolutions."""
        super().__init__()
        c_ = int(c2 * e)
        self.cv1: Conv = Conv(c1, c_, 1, 1)
        self.cv2: Conv = Conv(c1, c_, 1, 1)
        self.cv3: Conv = Conv(2 * c_, c2, 1)
        self.m: nn.Sequential = nn.Sequential(
            *(
                Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0)
                for _ in range(n)
            )
        )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CSP bottleneck with 3 convolutions."""
        return cast(
            torch.Tensor,
            self.cv3(
                torch.cat(
                    (
                        cast(torch.Tensor, self.m(cast(torch.Tensor, self.cv1(x)))),
                        cast(torch.Tensor, self.cv2(x)),
                    ),
                    1,
                )
            ),
        )


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
        k: int = 3,
    ) -> None:
        """Initialize C3k module."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m: nn.Sequential = nn.Sequential(
            *(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n))
        )


class C3k2(C2f):
    """C2f variant whose inner blocks are Bottleneck, C3k, or Bottleneck + PSABlock attention modules."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ) -> None:
        """Initialize C3k2 module."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m: nn.ModuleList = nn.ModuleList(
            nn.Sequential(
                Bottleneck(self.c, self.c, shortcut, g),
                PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)),
            )
            if attn
            else C3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )
