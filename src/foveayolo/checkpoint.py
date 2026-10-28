# Extracted from Ultralytics 8.4.191, AGPL-3.0: https://ultralytics.com/license
"""Extract weights from trusted YOLO26 checkpoints without importing Ultralytics."""

import pickle
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import cast, override

import torch
from torch import nn

from .blocks import (
    C2PSA,
    C3,
    DFL,
    SPPF,
    Attention,
    Bottleneck,
    C2f,
    C3k,
    C3k2,
    Concat,
    Conv,
    DWConv,
    PSABlock,
)
from .head import Detect


@dataclass(frozen=True)
class LoadReport:
    """Destination tensor keys loaded/missing/mismatched; unmatched source keys."""

    loaded_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    mismatched_keys: tuple[str, ...]


@dataclass(frozen=True)
class CheckpointWeights:
    """Weights and constructor metadata, never the checkpoint's executable graph."""

    state_dict: dict[str, torch.Tensor]
    size: str | None = None
    num_classes: int | None = None
    in_channels: int | None = None


class _CheckpointModel(nn.Module):
    """Temporary container for serialized module attributes; never used for inference."""


_BLOCKS: dict[str, type[nn.Module]] = {
    layer.__name__: layer
    for layer in (
        Attention,
        Bottleneck,
        C2f,
        C2PSA,
        C3,
        C3k,
        C3k2,
        Concat,
        Conv,
        DFL,
        DWConv,
        PSABlock,
        SPPF,
    )
}


class _Unpickler(pickle.Unpickler):
    @override
    def find_class(self, module: str, name: str) -> object:
        if module == "foveayolo.model" and name == "RoutedSequential":
            return nn.Sequential
        if module == "ultralytics.nn.tasks" and name == "DetectionModel":
            return _CheckpointModel
        if module == "ultralytics.nn.modules.head" and name == "Detect":
            return Detect
        if module in {"ultralytics.nn.modules.conv", "ultralytics.nn.modules.block"}:
            layer = _BLOCKS.get(name)
            if layer is not None:
                return layer
        if module.startswith("ultralytics"):
            raise ValueError(f"Unsupported YOLO26 checkpoint class: {module}.{name}")
        return cast(object, super().find_class(module, name))


# torch.load accepts a pickle-module interface; this does not alter sys.modules.
_PICKLE = ModuleType("foveayolo_checkpoint")
_PICKLE.__dict__.update(Unpickler=_Unpickler, load=pickle.load)


def read_checkpoint(path: str | Path) -> CheckpointWeights:
    """Read native weights or a trusted model checkpoint without restoring its graph.

    Native files may contain a plain state_dict, a state_dict/config dictionary,
    a saved nn.Module, or an Ultralytics-style model/ema dictionary. Stock graph
    validation applies only to models serialized under Ultralytics' class path.
    """
    raw = cast(
        object,
        torch.load(path, map_location="cpu", pickle_module=_PICKLE, weights_only=False),
    )
    config: dict[str, object] = {}
    if isinstance(raw, nn.Module):
        model: object = raw
    elif isinstance(raw, dict):
        checkpoint = cast(dict[str, object], raw)
        if checkpoint.get("modelopt"):
            raise ValueError("Quantized checkpoints are not supported")
        if "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
            raw_config = checkpoint.get("config", {})
            if not isinstance(raw_config, dict):
                raise TypeError("Checkpoint config must be a dictionary")
            config = cast(dict[str, object], raw_config)
            return _native_weights(state, config)
        if checkpoint and all(
            isinstance(value, torch.Tensor) for value in checkpoint.values()
        ):
            return _native_weights(checkpoint, config)
        model = checkpoint.get("ema") or checkpoint.get("model")
    else:
        raise TypeError("Expected a model, state_dict, or checkpoint dictionary")
    if not isinstance(model, nn.Module):
        raise TypeError("Checkpoint does not contain a model or state_dict")
    config = cast(dict[str, object], getattr(model, "yaml", {}))
    if not isinstance(model, _CheckpointModel):
        return _native_weights(model.state_dict(), config)

    # Ultralytics uses DetectionModel for multiple generations. Keep this check
    # for imported Ultralytics models, while allowing arbitrary native graphs.
    size = config.get("scale")
    layers = cast(nn.Sequential, model.model)
    head = layers[-1]
    final_block = layers[22] if len(layers) > 22 else None
    if (
        not isinstance(size, str)
        or size not in {"n", "s", "m", "l", "x"}
        or len(layers) != 24
        or not isinstance(head, Detect)
        or head.reg_max != 1
        or not isinstance(final_block, C3k2)
        or not any(isinstance(m, PSABlock) for m in final_block.modules())
    ):
        raise ValueError("Checkpoint architecture is not YOLO26 detection")
    first_conv = next(
        module for module in model.modules() if isinstance(module, nn.Conv2d)
    )
    return CheckpointWeights(
        dict(model.state_dict()), size, head.nc, first_conv.in_channels
    )


def _native_weights(state: object, config: dict[str, object]) -> CheckpointWeights:
    """Extract tensor weights without imposing a YOLO26 layer layout."""
    if not isinstance(state, dict):
        raise TypeError("state_dict must be a dictionary of named tensors")
    entries = cast(dict[object, object], state)
    weights: dict[str, torch.Tensor] = {}
    for key, value in entries.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise TypeError("state_dict must be a dictionary of named tensors")
        weights[key] = value
    size, classes, channels = (
        config.get("scale"),
        config.get("nc"),
        config.get("channels"),
    )
    return CheckpointWeights(
        weights,
        size if isinstance(size, str) else None,
        classes if isinstance(classes, int) else None,
        channels if isinstance(channels, int) else None,
    )
