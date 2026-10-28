# Extracted from Ultralytics 8.4.191, AGPL-3.0: https://ultralytics.com/license
"""Anchor-point generation and bounding-box decoding."""

from typing import cast

import torch


def make_anchors(
    feats: list[torch.Tensor] | torch.Tensor,
    strides: torch.Tensor | list[float],
    grid_cell_offset: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate anchor points and stride tensors from feature maps."""
    anchor_points: list[torch.Tensor] = []
    stride_tensor: list[torch.Tensor] = []
    assert feats is not None
    dtype = feats[0].dtype
    for i in range(
        len(feats)
    ):  # use len(feats) to avoid TracerWarning from iterating over strides tensor
        stride = strides[i]
        h, w = (
            feats[i].shape[2:]
            if isinstance(feats, list)
            else (int(feats[i][0]), int(feats[i][1]))
        )
        # no cumsum (nondeterministic on CUDA), no device= (baked into traces), no out= (does not convert to CoreML)
        sx = torch.arange(w).type_as(feats[0]) + grid_cell_offset  # shift x
        sy = torch.arange(h).type_as(feats[0]) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(
            feats[0].new_full((h * w, 1), cast(float, stride), dtype=dtype)
        )
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(
    distance: torch.Tensor,
    anchor_points: torch.Tensor,
    xywh: bool = True,
    dim: int = -1,
) -> torch.Tensor:
    """Transform distance (ltrb) to box (xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat([c_xy, wh], dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox
