# Extracted from Ultralytics 8.4.191, AGPL-3.0: https://ultralytics.com/license
"""YOLO26 detection branches, decoding, and prediction selection."""

import copy
import logging
import math
from typing import NotRequired, TypedDict, cast, override

import torch
from torch import nn

from .blocks import DFL, Conv, DWConv
from .geometry import dist2bbox, make_anchors

LOGGER = logging.getLogger(__name__)


class RawPredictions(TypedDict):
    """One detection branch's raw tensors and feature maps."""

    boxes: torch.Tensor
    scores: torch.Tensor
    feats: list[torch.Tensor]
    angle: NotRequired[torch.Tensor]


class DualPredictions(TypedDict):
    """Raw predictions from both detection branches."""

    one2many: RawPredictions
    one2one: RawPredictions


class HeadBranches(TypedDict):
    box_head: nn.ModuleList | None
    cls_head: nn.ModuleList | None


type ModelOutput = (
    RawPredictions
    | DualPredictions
    | torch.Tensor
    | tuple[torch.Tensor, RawPredictions | DualPredictions]
)


class Detect(nn.Module):
    """YOLO Detect head for object detection models."""

    dynamic: bool = False  # force grid reconstruction
    export: bool = False  # export mode
    format: str | None = None  # export format
    max_det: int = 300  # max_det
    agnostic_nms: bool = False
    shape: torch.Size | None = None
    anchors: torch.Tensor = torch.empty(0)  # init
    strides: torch.Tensor = torch.empty(0)  # init
    legacy: bool = False  # backward compatibility for v3/v5/v8/v9 models
    xyxy: bool = False  # xyxy or xywh output

    @staticmethod
    def _grouped_topk(
        x: torch.Tensor, k: int, groups: int = 8
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select exact top-k values through smaller grouped selections."""
        n = x.shape[1]
        while groups > 1 and (n % groups or n // groups < k):
            groups //= 2
        if (
            groups == 1
        ):  # nothing to gain, e.g. a short axis or one that does not divide evenly
            return x.topk(k, dim=1)
        size = n // groups
        values, index = x.reshape(x.shape[0], groups, size).topk(k, dim=-1)
        values, winners = values.flatten(1).topk(k, dim=1)
        return values, winners // k * size + index.flatten(1).gather(1, winners)

    def _gather(self, x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """Select index (batch, k) rows of x (batch, n, channels) along dim 1."""
        return x.gather(
            1, index if x.ndim == 2 else index[..., None].expand(-1, -1, x.shape[-1])
        )

    def __init__(
        self,
        nc: int = 80,
        reg_max: int = 16,
        end2end: bool = False,
        ch: list[int] | tuple[int, ...] = (),
    ) -> None:
        """Initialize the YOLO detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc: int = nc  # number of classes
        self.nl: int = len(ch)  # number of detection layers
        self.reg_max: int = reg_max  # DFL channels
        self.no: int = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride: torch.Tensor = torch.zeros(
            self.nl
        )  # strides computed during build
        c2, c3 = (
            max((16, ch[0] // 4, self.reg_max * 4)),
            max(ch[0], min(self.nc, 100)),
        )  # channels
        self.cv2: nn.ModuleList | None = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)
            )
            for x in ch
        )
        self.cv3: nn.ModuleList | None = (
            nn.ModuleList(
                nn.Sequential(
                    Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)
                )
                for x in ch
            )
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc, 1),
                )
                for x in ch
            )
        )
        self.dfl: nn.Module = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        self._end2end: bool = False
        self.inplace: bool = True
        self.one2one_cv2: nn.ModuleList | None = None
        self.one2one_cv3: nn.ModuleList | None = None
        if end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    @property
    def one2many(self) -> HeadBranches:
        """Return the one-to-many head components, here for v3/v5/v8/v9/v11 backward compatibility."""
        return {"box_head": self.cv2, "cls_head": self.cv3}

    @property
    def one2one(self) -> HeadBranches:
        """Return the one-to-one head components."""
        return {"box_head": self.one2one_cv2, "cls_head": self.one2one_cv3}

    @property
    def end2end(self) -> bool:
        """Select one-to-one inference when requested or when fusion has removed the one-to-many head."""
        return getattr(self, "one2one_cv2", None) is not None and (
            getattr(self, "_end2end", False) or self.cv2 is None
        )

    @end2end.setter
    def end2end(self, value: bool) -> None:
        """Select the inference head without changing dual-head training."""
        if value and getattr(self, "one2one_cv2", None) is None:
            LOGGER.warning(
                "This model has no one-to-one head; using one-to-many outputs."
            )
        elif not value and self.cv2 is None:
            LOGGER.warning(
                "The one-to-many head was removed by fusion; using the remaining one-to-one head."
            )
        self._end2end = value

    def forward_head(
        self,
        x: list[torch.Tensor],
        box_head: nn.ModuleList | None = None,
        cls_head: nn.ModuleList | None = None,
    ) -> RawPredictions:
        """Concatenate and return predicted bounding boxes and class probabilities."""
        if box_head is None or cls_head is None:  # for fused inference
            # Upstream represents a removed, unused inference branch with an empty dict.
            return cast(RawPredictions, cast(object, {}))
        bs = x[0].shape[0]  # batch size
        boxes = torch.cat(
            [
                cast(torch.Tensor, box_head[i](x[i])).view(bs, 4 * self.reg_max, -1)
                for i in range(self.nl)
            ],
            dim=-1,
        )
        scores = torch.cat(
            [
                cast(torch.Tensor, cls_head[i](x[i])).view(bs, self.nc, -1)
                for i in range(self.nl)
            ],
            dim=-1,
        )
        return {"boxes": boxes, "scores": scores, "feats": x}

    @override
    def forward(self, x: list[torch.Tensor]) -> ModelOutput:
        """Run the detection head on multi-level feature maps."""
        raw = self.forward_head(x, **self.one2many)
        preds: RawPredictions | DualPredictions = raw
        inference_raw = raw
        if getattr(self, "one2one_cv2", None) is not None:
            x_detach = [xi.detach() for xi in x] if self.training else x
            one2one = self.forward_head(x_detach, **self.one2one)
            preds = {"one2many": raw, "one2one": one2one}
            inference_raw = one2one if self.end2end else raw
        if self.training:
            return preds
        y = self._inference(inference_raw)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def _inference(self, x: RawPredictions) -> torch.Tensor:
        """Decode predicted bounding boxes and class probabilities based on multiple-level feature maps."""
        # Inference path
        dbox = self._get_decode_boxes(x)
        return torch.cat((dbox, x["scores"].sigmoid()), 1)

    def _get_decode_boxes(self, x: RawPredictions) -> torch.Tensor:
        """Get decoded boxes based on anchors and strides."""
        shape = x["feats"][0].shape  # BCHW
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (
                a.transpose(0, 1) for a in make_anchors(x["feats"], self.stride, 0.5)
            )
            self.shape = shape

        dbox = (
            self.decode_bboxes(
                cast(torch.Tensor, self.dfl(x["boxes"])),
                self.anchors.unsqueeze(0),
                x.get("angle"),
            )
            * self.strides
        )
        return dbox

    def bias_init(self) -> None:
        """Initialize detection biases after strides have been calculated."""
        branches = [self.one2many]
        if getattr(self, "one2one_cv2", None) is not None:
            branches.append(self.one2one)
        for branch in branches:
            box_head, cls_head = branch["box_head"], branch["cls_head"]
            assert box_head is not None and cls_head is not None
            for i, (a, b) in enumerate(zip(box_head, cls_head, strict=False)):
                box_conv = cast(nn.Conv2d, cast(nn.Sequential, a)[-1])
                cls_conv = cast(nn.Conv2d, cast(nn.Sequential, b)[-1])
                assert box_conv.bias is not None and cls_conv.bias is not None
                box_conv.bias.data[:] = 2.0
                cls_conv.bias.data[: self.nc] = math.log(
                    5 / self.nc / (640 / self.stride[i]) ** 2
                )

    def decode_bboxes(
        self,
        bboxes: torch.Tensor,
        anchors: torch.Tensor,
        angle: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode bounding boxes from predictions, angle is only used by the OBB head."""
        _ = angle  # Accepted for compatibility with the shared upstream head signature.
        return dist2bbox(
            bboxes, anchors, xywh=not self.end2end and not self.xyxy, dim=1
        )

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """Post-processes YOLO model predictions."""
        # Segment, Pose and OBB carry task channels after the class scores, Detect has none
        parts = torch.split(
            preds,
            [s for s in (4, self.nc, preds.shape[-1] - 4 - self.nc) if s],
            dim=-1,
        )
        boxes, scores, *extra = parts
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        return torch.cat(
            [
                self._gather(boxes, idx),
                scores,
                conf,
                *(self._gather(e, idx) for e in extra),
            ],
            dim=-1,
        )

    def get_topk_index(
        self, scores: torch.Tensor, max_det: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get top-k indices from scores."""
        anchors, nc = scores.shape[1:]  # i.e. shape(16,8400,80)
        k = min(max_det, anchors)
        if self.agnostic_nms:
            scores, labels = scores.max(dim=-1)
            scores, index = self._grouped_topk(scores, k, 1)
            return (
                scores[..., None],
                self._gather(labels[..., None].float(), index),
                index,
            )
        groups = (
            8 if self.export and self.format == "engine" and not self.dynamic else 1
        )
        ori_index = self._grouped_topk(scores.max(dim=-1)[0], k, groups)[1]
        scores = self._gather(scores, ori_index)
        scores, index = self._grouped_topk(scores.flatten(1), k, groups)
        return (
            scores[..., None],
            (index % nc)[..., None].float(),
            self._gather(ori_index, index // nc),
        )

    def fuse(self) -> None:
        """Remove the unused detection branch for inference."""
        end2end = self.end2end
        for name in tuple(self._modules):
            if name.startswith("one2one_"):
                setattr(self, name[8:] if end2end else name, None)
