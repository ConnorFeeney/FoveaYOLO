# Extracted from Ultralytics 8.4.191, AGPL-3.0: https://ultralytics.com/license
"""Position-sensitive attention blocks used by YOLO26."""

from typing import cast, override

import torch
from torch import nn
from torch.nn import functional as F

from .conv import Conv


class Attention(nn.Module):
    """Attention module that performs self-attention on the input tensor."""

    format: str | None = None

    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5) -> None:
        """Initialize multi-head attention module."""
        super().__init__()
        self.num_heads: int = num_heads
        self.head_dim: int = dim // num_heads
        self.key_dim: int = int(self.head_dim * attn_ratio)
        self.scale: float = cast(float, self.key_dim ** (-0.5))
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv: Conv = Conv(dim, h, 1, act=False)
        self.proj: Conv = Conv(dim, dim, 1, act=False)
        self.pe: Conv = Conv(dim, dim, 3, 1, g=dim, act=False)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Attention module."""
        B, C, H, W = x.shape
        N = H * W
        qkv = cast(torch.Tensor, self.qkv(x))
        q, k, v = torch.split(
            qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N),
            [self.key_dim, self.key_dim, self.head_dim],
            dim=2,
        )
        if self.format == "coreml" and hasattr(F, "scaled_dot_product_attention"):
            x = F.scaled_dot_product_attention(
                q.transpose(-2, -1), k.transpose(-2, -1), v.transpose(-2, -1)
            )
            x = x.transpose(-2, -1).reshape(B, C, H, W) + cast(
                torch.Tensor, self.pe(v.reshape(B, C, H, W))
            )
        else:
            attn = ((q * self.scale).transpose(-2, -1) @ k).softmax(dim=-1)
            x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + cast(
                torch.Tensor, self.pe(v.reshape(B, C, H, W))
            )
        x = cast(torch.Tensor, self.proj(x))
        return x


class PSABlock(nn.Module):
    """PSABlock class implementing a Position-Sensitive Attention block for neural networks."""

    def __init__(
        self, c: int, attn_ratio: float = 0.5, num_heads: int = 4, shortcut: bool = True
    ) -> None:
        """Initialize the PSABlock."""
        super().__init__()
        self.attn: Attention = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn: nn.Sequential = nn.Sequential(
            Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False)
        )
        self.add: bool = shortcut

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute a forward pass through PSABlock."""
        x = (
            x + cast(torch.Tensor, self.attn(x))
            if self.add
            else cast(torch.Tensor, self.attn(x))
        )
        x = (
            x + cast(torch.Tensor, self.ffn(x))
            if self.add
            else cast(torch.Tensor, self.ffn(x))
        )
        return x


class C2PSA(nn.Module):
    """C2PSA module with attention mechanism for enhanced feature extraction and processing."""

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5) -> None:
        """Initialize C2PSA module."""
        super().__init__()
        assert c1 == c2
        self.c: int = int(c1 * e)
        self.cv1: Conv = Conv(c1, 2 * self.c, 1, 1)
        self.cv2: Conv = Conv(2 * self.c, c1, 1)
        self.m: nn.Sequential = nn.Sequential(
            *(
                PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1))
                for _ in range(n)
            )
        )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process the input tensor through a series of PSA blocks."""
        a, b = torch.split(cast(torch.Tensor, self.cv1(x)), [self.c, self.c], dim=1)
        b = cast(torch.Tensor, self.m(b))
        return cast(torch.Tensor, self.cv2(torch.cat((a, b), 1)))
