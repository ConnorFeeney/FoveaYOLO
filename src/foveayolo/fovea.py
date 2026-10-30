"""Saliency maps and the separable FOVEA warp (https://arxiv.org/abs/2108.12102)."""

import math
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


def foveate(
    image: torch.Tensor,
    saliency: torch.Tensor,
    *,
    sigma: float = 0.18,
) -> torch.Tensor:
    """Magnify salient regions, returning an image with the input shape/dtype.

    image is floating-point [B, C, H, W], with H and W at least two.
    saliency is finite, nonnegative [B, 1, h, w]; it need not sum to one.
    It is bilinearly resized to (H, W) before computing the warp. Uniform or
    all-zero saliency gives an identity warp. Each batch item warps separately.

    sigma controls the Gaussian attraction radius as a fraction of the shorter
    image axis (standard deviation, not kernel width). The default approximates
    the paper's 5.5-cell standard deviation on a 31-cell axis. The Gaussian is
    truncated at three standard deviations or the available reflection extent.
    A tiny uniform floor prevents undefined sampling in zero-saliency regions.

    Gradients flow to both inputs. Half/bfloat16 inputs compute in float32;
    float64 images retain double precision. This only warps the image, not
    detection boxes or other labels.
    """
    if image.ndim != 4 or min(image.shape) < 1 or min(image.shape[-2:]) < 2:
        raise ValueError("image must have shape [B, C, H, W] with H, W >= 2")
    if (
        saliency.ndim != 4
        or saliency.shape[:2] != (image.shape[0], 1)
        or min(saliency.shape[-2:]) < 1
    ):
        raise ValueError("saliency must have shape [B, 1, h, w] matching image batch")
    if not image.is_floating_point() or not saliency.is_floating_point():
        raise TypeError("image and saliency must be floating-point tensors")
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    if not bool(torch.isfinite(saliency).all()) or bool((saliency < 0).any()):
        raise ValueError("saliency must be finite and nonnegative")

    batch, _, height, width = image.shape
    dtype = torch.float64 if image.dtype == torch.float64 else torch.float32
    density = F.interpolate(
        saliency.to(device=image.device, dtype=dtype),
        size=(height, width),
        mode="bilinear",
        align_corners=True,
    )
    # Scale invariance also keeps the small background floor meaningful.
    maximum = density.amax(dim=(-2, -1), keepdim=True)
    density = density / maximum.clamp_min(torch.finfo(dtype).tiny) + 1e-6

    std = sigma * (min(height, width) - 1)
    radius = min(math.ceil(3 * std), min(height, width) - 1)
    offsets = torch.arange(-radius, radius + 1, device=image.device, dtype=dtype)
    gaussian = torch.exp(-0.5 * (offsets / std).square())
    kernel = gaussian.reshape(1, 1, -1)
    moment_kernel = (gaussian * offsets).reshape(1, 1, -1)

    def axis_coordinates(marginal: torch.Tensor) -> torch.Tensor:
        length = marginal.shape[-1]
        padded = F.pad(marginal, (radius, radius), mode="reflect")
        mass = F.conv1d(padded, kernel)
        displacement = F.conv1d(padded, moment_kernel) / mass
        positions = torch.arange(length, device=image.device, dtype=dtype)
        # Positions extend beyond the boundary; only saliency is reflected.
        return ((positions + displacement) * (2 / (length - 1)) - 1).clamp(-1, 1)

    x = axis_coordinates(density.sum(dim=2))
    y = axis_coordinates(density.sum(dim=3))
    grid = torch.stack(
        (
            x.reshape(batch, 1, width).expand(batch, height, width),
            y.reshape(batch, height, 1).expand(batch, height, width),
        ),
        dim=-1,
    )
    warped = F.grid_sample(
        image.to(dtype=dtype),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return warped.to(dtype=image.dtype)
