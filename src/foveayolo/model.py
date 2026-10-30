# Extracted from Ultralytics 8.4.191, AGPL-3.0: https://ultralytics.com/license
"""Standalone YOLO26 detection models, with Ultralytics-compatible parameter names."""

import math
from collections import OrderedDict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Self, cast, override

import torch
from torch import nn

from .blocks import C2PSA, SPPF, C3k2, Concat, Conv
from .checkpoint import LoadReport, read_checkpoint
from .head import Detect, DualPredictions, ModelOutput

SCALES: dict[str, tuple[float, float, int]] = {
    "n": (0.50, 0.25, 1024),
    "s": (0.50, 0.50, 1024),
    "m": (0.50, 1.00, 512),
    "l": (1.00, 1.00, 512),
    "x": (1.00, 1.50, 512),
}

# Rows from cfg/models/26/yolo26.yaml: (from, repeats, module, arguments).
# The YAML's "head" section includes the feature aggregation neck.
type Source = int | tuple[int, ...]
type ArchitectureRow = tuple[Source, int, Callable[..., nn.Module], tuple[object, ...]]

BACKBONE_ARCHITECTURE: tuple[ArchitectureRow, ...] = (
    # Backbone: layers 0-10.
    (-1, 1, Conv, (64, 3, 2)),
    (-1, 1, Conv, (128, 3, 2)),
    (-1, 2, C3k2, (256, False, 0.25)),
    (-1, 1, Conv, (256, 3, 2)),
    (-1, 2, C3k2, (512, False, 0.25)),
    (-1, 1, Conv, (512, 3, 2)),
    (-1, 2, C3k2, (512, True)),
    (-1, 1, Conv, (1024, 3, 2)),
    (-1, 2, C3k2, (1024, True)),
    (-1, 1, SPPF, (1024, 5, 3, True)),
    (-1, 2, C2PSA, (1024,)),
)

# Sources keep their global indices across the stage boundaries.
NECK_ARCHITECTURE: tuple[ArchitectureRow, ...] = (
    (-1, 1, nn.Upsample, (None, 2, "nearest")),
    ((-1, 6), 1, Concat, (1,)),
    (-1, 2, C3k2, (512, True)),
    (-1, 1, nn.Upsample, (None, 2, "nearest")),
    ((-1, 4), 1, Concat, (1,)),
    (-1, 2, C3k2, (256, True)),
    (-1, 1, Conv, (256, 3, 2)),
    ((-1, 13), 1, Concat, (1,)),
    (-1, 2, C3k2, (512, True)),
    (-1, 1, Conv, (512, 3, 2)),
    ((-1, 10), 1, Concat, (1,)),
    (-1, 1, C3k2, (1024, True, 0.5, True)),
)

HEAD_ARCHITECTURE: tuple[ArchitectureRow, ...] = (((16, 19, 22), 1, Detect, ()),)


def _build(
    size: str, num_classes: int, in_channels: int
) -> tuple[dict[str, nn.Sequential], list[int]]:
    """Apply the YOLO26 subset of Ultralytics' parse_model scaling rules.

    Table arguments omit input channels and, for C3k2/C2PSA, internal repeats.
    For example, nano's (256, False, 0.25) C3k2 row becomes constructor args
    (32, 64, 1, False, 0.25): c1, scaled c2, scaled n, c3k, expansion.
    """
    depth, width, max_channels = SCALES[size]
    channels = [in_channels]
    layers: list[nn.Module] = []
    save: list[int] = []
    sections = (
        ("backbone", BACKBONE_ARCHITECTURE),
        ("neck", NECK_ARCHITECTURE),
        ("head", HEAD_ARCHITECTURE),
    )
    architecture = tuple(row for _, section in sections for row in section)
    for i, (source, repeats, module, arguments) in enumerate(architecture):
        args = list(arguments)
        repeats = max(round(repeats * depth), 1) if repeats > 1 else repeats
        out_channels = (
            channels[source]
            if isinstance(source, int)
            else sum(channels[j] for j in source)
        )
        if module in {Conv, C3k2, SPPF, C2PSA}:
            assert isinstance(source, int)
            # Scale the requested output channels; infer input channels from the source.
            out_channels = (
                math.ceil(min(cast(int, args[0]), max_channels) * width / 8) * 8
            )
            args = [channels[source], out_channels, *args[1:]]
            if module in {C3k2, C2PSA}:
                # These blocks repeat internally rather than duplicating whole layers.
                args.insert(2, repeats)
                repeats = 1
            if module is C3k2 and size in {"m", "l", "x"}:
                # The upstream parser selects C3k internals for the larger variants.
                args[3:4] = [True]
        elif module is Concat:
            assert isinstance(source, tuple)
            out_channels = sum(channels[j] for j in source)
        elif module is Detect:
            assert isinstance(source, tuple)
            # nc, reg_max, construct dual branches, input channels of all three maps.
            args = [num_classes, 1, True, [channels[j] for j in source]]

        factory = cast(Callable[..., nn.Module], module)
        layer = (
            nn.Sequential(*(factory(*args) for _ in range(repeats)))
            if repeats > 1
            else factory(*args)
        )
        layer.__dict__.update(i=i, f=source)
        save.extend(
            j % i for j in ([source] if isinstance(source, int) else source) if j != -1
        )
        layers.append(layer)
        if i == 0:
            channels = []
        channels.append(out_channels)
    saved = sorted(set(save))
    stages: dict[str, nn.Sequential] = {}
    start = 0
    for name, section in sections:
        end = start + len(section)
        modules = OrderedDict((str(i), layers[i]) for i in range(start, end))
        stages[name] = nn.Sequential(modules)
        start = end
    return stages, saved


class YOLO26(nn.Module):
    """YOLO26 n/s/m/l/x detector; tensor forwards preserve upstream output formats.

    Training mode returns raw head dictionaries, without computing a loss.
    Evaluation returns (decoded predictions, raw predictions), or just decoded
    predictions when model.detect.export is True. Inputs are BCHW tensors,
    normally normalized to [0, 1], with height and width divisible by 32.
    """

    def __init__(
        self, size: str = "n", num_classes: int = 80, in_channels: int = 3
    ) -> None:
        super().__init__()
        if size not in SCALES:
            raise ValueError(
                f"Unknown YOLO26 size {size!r}; expected one of {tuple(SCALES)}"
            )
        if num_classes < 1 or in_channels < 1:
            raise ValueError("num_classes and in_channels must be positive")
        self.size: str = size
        stages, save_indices = _build(size, num_classes, in_channels)
        self.backbone: nn.Sequential = stages["backbone"]
        self.neck: nn.Sequential = stages["neck"]
        self.head: nn.Sequential = stages["head"]
        self.save_indices: list[int] = save_indices
        self.names: dict[int, str] = {i: str(i) for i in range(num_classes)}
        self.inplace: bool = True
        # Keep graph metadata available when serializing the standalone model.
        self.yaml: dict[str, str | int] = {
            "scale": size,
            "nc": num_classes,
            "channels": in_channels,
        }

        head = self.detect
        head.inplace = self.inplace
        _ = self.eval()
        head.training = True
        with torch.no_grad():
            predictions = cast(
                DualPredictions,
                self._forward_graph(torch.zeros(1, in_channels, 256, 256)),
            )
        features = predictions["one2many"]["feats"]
        head.stride = torch.tensor([256 / feature.shape[-2] for feature in features])
        self.stride: torch.Tensor = head.stride
        _ = self.train()
        head.bias_init()
        for module in self.modules():
            if type(module) is nn.BatchNorm2d:
                module.eps = 1e-3
                module.momentum = 0.03

    @property
    def detect(self) -> Detect:
        """The final detection module, with its concrete type available to callers."""
        return cast(Detect, self.head[-1])

    @property
    def end2end(self) -> bool:
        """Select the one-to-one inference branch, retaining both training branches."""
        return self.detect.end2end

    @end2end.setter
    def end2end(self, value: bool) -> None:
        self.detect.end2end = value

    @override
    def forward(self, x: torch.Tensor) -> ModelOutput:
        """Execute the three stages, preserving the detection head's output format."""
        return self._forward_graph(x)

    def fuse(self) -> Self:
        """Fold Conv BatchNorm layers into convolutions in place for inference.

        Call eval() and load weights first. Repeated calls are harmless. Fusion
        is irreversible: keep an unfused model/checkpoint for training and for
        save()/from_pretrained(), which expect the original parameter layout.
        Detection branches and output formats are preserved.
        """
        if any(module.training for module in self.modules()):
            raise RuntimeError("Call model.eval() before model.fuse()")
        for module in self.modules():
            if isinstance(module, Conv) and isinstance(
                getattr(module, "bn", None), nn.BatchNorm2d
            ):
                module.conv = nn.utils.fuse_conv_bn_eval(module.conv, module.bn)
                del module.bn
                module.forward = module.forward_fuse
        return self

    def _forward_graph(self, x: torch.Tensor) -> ModelOutput:
        """Keep stride initialization independent of a subclass's custom forward."""
        outputs: list[object] = []
        value, index = self.forward_stage(self.backbone, x, 0, outputs)
        print(f"P5: {cast(torch.Tensor, outputs[10]).shape}\nP4: {cast(torch.Tensor, outputs[6]).shape}\nP3: {cast(torch.Tensor, outputs[4]).shape}")
        value, index = self.forward_stage(self.neck, value, index, outputs)
        value, _ = self.forward_stage(self.head, value, index, outputs)
        return cast(ModelOutput, value)

    def forward_stage(
        self,
        stage: nn.Sequential,
        x: object,
        index: int,
        outputs: list[object],
    ) -> tuple[object, int]:
        """Run a plain sequential's layers using a global index and output cache.

        Pass the returned index and the same outputs list to the next stage.
        Start with index=0 and a new list for each independent backbone pass.
        """
        value = x
        for layer in stage:
            source = cast(Source, cast(object, layer.f))
            if source != -1:
                value = (
                    outputs[source]
                    if isinstance(source, int)
                    else [value if j == -1 else outputs[j] for j in source]
                )
            value = cast(object, layer(value))
            outputs.append(value if index in self.save_indices else None)
            index += 1
        return value, index

    @override
    def _apply(
        self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True
    ) -> Self:
        """Move/cast the head's non-buffer geometry tensors along with parameters."""
        apply = cast(Callable[..., Self], super()._apply)
        _ = apply(fn, recurse=recurse)
        head = self.detect
        head.stride = fn(head.stride)
        head.anchors = fn(head.anchors)
        head.strides = fn(head.strides)
        self.stride = head.stride
        return self

    def load_pretrained(
        self,
        path: str | Path,
        *,
        key_map: Mapping[str, str] | None = None,
    ) -> LoadReport:
        """Overlay native or trusted Ultralytics weights, keeping this model's graph.

        Only matching tensor names and shapes are copied. New or incompatible
        parameters retain their current values. key_map maps source state-dict
        keys to destination keys when modules have been renamed or reindexed.
        Device, dtype, mode, metadata, and requires_grad flags are preserved.
        """
        return self._overlay_weights(read_checkpoint(path).state_dict, key_map)

    def save(self, path: str | Path) -> None:
        """Save all parameters/buffers and constructor metadata, including added modules.

        Reload into the same modified architecture with load_pretrained(), or use
        that architecture's class.from_pretrained(). Python forward code remains
        defined by your model class, rather than embedded in this weights file.
        """
        torch.save({"state_dict": self.state_dict(), "config": self.yaml}, path)

    def _overlay_weights(
        self,
        weights: Mapping[str, torch.Tensor],
        key_map: Mapping[str, str] | None = None,
    ) -> LoadReport:
        target = cast(dict[str, torch.Tensor], self.state_dict())
        matched: dict[str, torch.Tensor] = {}
        unexpected: list[str] = []
        mismatched: list[str] = []
        destinations: set[str] = set()
        layer_stages = {
            index: name
            for name, stage in (
                ("backbone", self.backbone),
                ("neck", self.neck),
                ("head", self.head),
            )
            for index, _ in stage.named_children()
        }
        for source_key, tensor in weights.items():
            key = (
                key_map.get(source_key, source_key)
                if key_map is not None
                else source_key
            )
            if key.startswith("model."):
                parts = key.split(".", 2)
                if len(parts) == 3 and parts[1] in layer_stages:
                    key = f"{layer_stages[parts[1]]}.{parts[1]}.{parts[2]}"
            if key in destinations:
                raise ValueError(f"Multiple checkpoint tensors map to {key!r}")
            destinations.add(key)
            if key not in target:
                unexpected.append(source_key)
            elif tensor.shape != target[key].shape:
                mismatched.append(key)
            else:
                matched[key] = tensor
        if not matched:
            raise ValueError(
                "No checkpoint tensors match this model; check size and key_map"
            )
        _ = self.load_state_dict(matched, strict=False)
        return LoadReport(
            loaded_keys=tuple(matched),
            missing_keys=tuple(key for key in target if key not in matched),
            unexpected_keys=tuple(unexpected),
            mismatched_keys=tuple(mismatched),
        )

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        *,
        size: str | None = None,
        num_classes: int | None = None,
        in_channels: int | None = None,
        device: str | torch.device = "cpu",
    ) -> Self:
        """Build the current architecture, then overlay a trusted checkpoint's weights.

        Unspecified size/classes/channels are inferred from the checkpoint.
        For an already modified instance, use load_pretrained to retain that instance.
        Ultralytics .pt files pickle Python objects; only load files you trust.
        """
        weights = read_checkpoint(path)
        size = weights.size if size is None else size
        num_classes = weights.num_classes if num_classes is None else num_classes
        in_channels = weights.in_channels if in_channels is None else in_channels
        if size is None or num_classes is None or in_channels is None:
            raise ValueError(
                "Checkpoint lacks constructor metadata; pass size, num_classes and in_channels"
            )
        model = cls(
            size=size,
            num_classes=num_classes,
            in_channels=in_channels,
        )
        _ = model._overlay_weights(weights.state_dict)
        return model.eval().to(device)
