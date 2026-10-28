"""Focused construction, forward, geometry, and checkpoint checks for YOLO26."""

import os
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest
import torch

from foveayolo import YOLO26
from foveayolo.blocks import C3k2, Conv
from foveayolo.head import DualPredictions

type EvalOutput = tuple[torch.Tensor, DualPredictions]
PARAMETERS = {"n": 2572280, "s": 10009784, "m": 21896248, "l": 26299704, "x": 58993368}


class ModifiedYOLO26(YOLO26):
    """A changed graph with a learned block inserted before the detection head."""

    def __init__(
        self, size: str = "n", num_classes: int = 3, in_channels: int = 3
    ) -> None:
        super().__init__(size, num_classes, in_channels)
        head = self.detect
        channels = cast(C3k2, self.neck[-1]).cv2.conv.out_channels
        added = Conv(channels, channels, 1)
        added.__dict__.update(i=23, f=-1)
        head.__dict__.update(i=24, f=(16, 19, 23))
        self.neck.add_module("23", added)
        del self.head[-1]
        self.head.add_module("24", head)
        self.save_indices.append(23)


@pytest.fixture(scope="session", autouse=True)
def cpu_threads() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("size", PARAMETERS)
def test_family_shapes_and_parameter_counts(size: str) -> None:
    model = YOLO26(size).eval()
    assert (
        sum(parameter.numel() for parameter in model.parameters()) == PARAMETERS[size]
    )
    torch.testing.assert_close(model.stride, torch.tensor([8.0, 16.0, 32.0]))
    with torch.no_grad():
        decoded, raw = cast(EvalOutput, model(torch.zeros(1, 3, 64, 96)))
        assert decoded.shape == (1, 84, 126)
        assert torch.isfinite(decoded).all()
        for branch in ("one2many", "one2one"):
            assert raw[branch]["boxes"].shape == (1, 4, 126)
            assert raw[branch]["scores"].shape == (1, 80, 126)
            assert [feature.shape[-2:] for feature in raw[branch]["feats"]] == [
                (8, 12),
                (4, 6),
                (2, 3),
            ]
        model.end2end = True
        detections, _ = cast(EvalOutput, model(torch.zeros(1, 3, 64, 96)))
        assert detections.shape == (1, 126, 6)
        assert torch.isfinite(detections).all()
        assert ((detections[..., 4] >= 0) & (detections[..., 4] <= 1)).all()
        assert ((detections[..., 5] >= 0) & (detections[..., 5] < 80)).all()


def test_custom_channels_classes_and_raw_outputs() -> None:
    model = YOLO26("n", num_classes=3, in_channels=1)
    raw = cast(DualPredictions, model(torch.rand(2, 1, 64, 64)))
    assert set(raw) == {"one2many", "one2one"}
    assert raw["one2many"]["scores"].shape == (2, 3, 84)
    assert raw["one2many"]["feats"][0].requires_grad
    assert not raw["one2one"]["feats"][0].requires_grad


def test_decoding_matches_raw_head_and_refreshes_geometry() -> None:
    model = YOLO26("n", num_classes=3).eval()
    head = model.detect
    with torch.no_grad():
        for height, width in ((64, 96), (96, 128)):
            decoded, raw = cast(EvalOutput, model(torch.rand(1, 3, height, width)))
            branch = raw["one2many"]
            points: list[torch.Tensor] = []
            strides: list[torch.Tensor] = []
            for feature, stride in zip(branch["feats"], (8, 16, 32), strict=True):
                h, w = feature.shape[-2:]
                y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
                points.append(torch.stack((x, y), dim=0).reshape(2, -1) + 0.5)
                strides.append(torch.full((1, h * w), stride))
            anchors = torch.cat(points, dim=1).unsqueeze(0)
            stride_tensor = torch.cat(strides, dim=1).unsqueeze(0)
            left_top = anchors - branch["boxes"][:, :2]
            right_bottom = anchors + branch["boxes"][:, 2:]
            expected_boxes = (
                torch.cat(
                    ((left_top + right_bottom) / 2, right_bottom - left_top), dim=1
                )
                * stride_tensor
            )
            torch.testing.assert_close(decoded[:, :4], expected_boxes)
            torch.testing.assert_close(decoded[:, 4:], branch["scores"].sigmoid())
            assert head.anchors.shape[1] == decoded.shape[-1]
        _ = model.double()
        decoded, _ = cast(
            EvalOutput, model(torch.rand(1, 3, 96, 128, dtype=torch.float64))
        )
        assert decoded.dtype == head.anchors.dtype == head.stride.dtype == torch.float64
        assert model.stride is head.stride


def test_checkpoint_prefers_ema_and_preserves_predictions(tmp_path: Path) -> None:
    model = YOLO26("n", num_classes=3).eval()
    model.names = {0: "a", 1: "b", 2: "c"}
    path = tmp_path / "checkpoint.pt"
    torch.save({"ema": model, "model": None}, path)
    loaded = YOLO26.from_pretrained(path)
    assert loaded.size == "n"
    assert loaded.detect.nc == 3
    assert not loaded.training
    images = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        torch.testing.assert_close(
            cast(EvalOutput, loaded(images))[0],
            cast(EvalOutput, model(images))[0],
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("format", ("native", "state_dict", "module", "model_dict"))
def test_modified_architecture_checkpoint_round_trip(
    tmp_path: Path, format: str
) -> None:
    source = ModifiedYOLO26().eval()
    path = tmp_path / "modified.pt"
    if format == "native":
        source.save(path)
    elif format == "state_dict":
        torch.save(source.state_dict(), path)
    elif format == "module":
        torch.save(source, path)
    else:
        torch.save({"model": source}, path)

    target = ModifiedYOLO26().eval()
    stages = (target.backbone, target.neck, target.head)
    report = target.load_pretrained(path)
    assert (target.backbone, target.neck, target.head) == stages
    assert sum(len(stage) for stage in stages) == 25
    assert not report.missing_keys
    assert not report.unexpected_keys
    assert not report.mismatched_keys
    torch.testing.assert_close(target.state_dict(), source.state_dict(), rtol=0, atol=0)
    images = torch.rand(1, 3, 64, 96)
    with torch.no_grad():
        torch.testing.assert_close(
            cast(EvalOutput, target(images)),
            cast(EvalOutput, source(images)),
            rtol=0,
            atol=0,
        )


def test_modified_architecture_from_saved_configuration(tmp_path: Path) -> None:
    source = ModifiedYOLO26(num_classes=5).eval()
    path = tmp_path / "modified.pt"
    source.save(path)
    loaded = ModifiedYOLO26.from_pretrained(path)
    assert len(loaded.neck) == 13
    assert loaded.detect.nc == 5
    torch.testing.assert_close(loaded.state_dict(), source.state_dict(), rtol=0, atol=0)


def test_state_dict_without_metadata_requires_constructor_arguments(
    tmp_path: Path,
) -> None:
    source = ModifiedYOLO26().eval()
    path = tmp_path / "weights.pt"
    torch.save(source.state_dict(), path)
    with pytest.raises(ValueError, match="constructor metadata"):
        _ = ModifiedYOLO26.from_pretrained(path)
    loaded = ModifiedYOLO26.from_pretrained(
        path, size="n", num_classes=3, in_channels=3
    )
    torch.testing.assert_close(loaded.state_dict(), source.state_dict(), rtol=0, atol=0)


@pytest.mark.parametrize("size", PARAMETERS)
def test_official_ultralytics_checkpoint(size: str) -> None:
    """Set YOLO26_CHECKPOINT_DIR to run against local official .pt files, offline."""
    directory = os.environ.get("YOLO26_CHECKPOINT_DIR")
    if directory is None:
        pytest.skip("Set YOLO26_CHECKPOINT_DIR to test official Ultralytics .pt files")
    path = Path(directory) / f"yolo26{size}.pt"
    assert path.is_file(), f"Missing requested checkpoint: {path}"
    loaded = YOLO26.from_pretrained(path)
    fresh = YOLO26(size).eval()
    _ = fresh.load_state_dict(
        cast(dict[str, torch.Tensor], loaded.state_dict()), strict=True
    )
    assert loaded.size == size
    assert (
        sum(parameter.numel() for parameter in loaded.parameters()) == PARAMETERS[size]
    )
    images = torch.rand(1, 3, 64, 96)
    with torch.no_grad():
        for end2end in (False, True):
            loaded.end2end = fresh.end2end = end2end
            torch.testing.assert_close(
                cast(EvalOutput, loaded(images)),
                cast(EvalOutput, fresh(images)),
                rtol=0,
                atol=0,
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_forward_and_geometry() -> None:
    model = YOLO26("n").eval()
    with torch.no_grad():
        cpu_prediction = cast(EvalOutput, model(torch.zeros(1, 3, 64, 96)))[0]
        _ = model.cuda()
        prediction = cast(EvalOutput, model(torch.zeros(1, 3, 64, 96, device="cuda")))[
            0
        ]
    assert model.stride.device.type == "cuda"
    assert model.detect.anchors.device.type == "cuda"
    torch.testing.assert_close(
        prediction.cpu(), cpu_prediction, rtol=0.0001, atol=0.0001
    )


def test_pretrained_overlay_keeps_new_parameters_and_model_state(
    tmp_path: Path,
) -> None:
    source = YOLO26("n")
    path = tmp_path / "source.pt"
    torch.save({"model": source}, path)
    target = YOLO26("n", num_classes=3)
    target.add_module("fovea", torch.nn.Conv2d(3, 3, 1))
    new_weights = cast(dict[str, torch.Tensor], target.state_dict())[
        "fovea.weight"
    ].clone()
    classifier = cast(dict[str, torch.Tensor], target.state_dict())[
        "head.23.cv3.0.2.weight"
    ].clone()
    stages = (target.backbone, target.neck, target.head)
    stem = target.backbone[0]
    report = target.load_pretrained(path)
    assert (target.backbone, target.neck, target.head) == stages
    assert target.backbone[0] is stem
    assert target.training
    assert target.detect.nc == 3
    assert "fovea.weight" in report.missing_keys
    assert "head.23.cv3.0.2.weight" in report.mismatched_keys
    torch.testing.assert_close(
        cast(dict[str, torch.Tensor], target.state_dict())["fovea.weight"],
        new_weights,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        cast(dict[str, torch.Tensor], target.state_dict())["head.23.cv3.0.2.weight"],
        classifier,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        cast(dict[str, torch.Tensor], target.state_dict())["backbone.0.conv.weight"],
        cast(dict[str, torch.Tensor], source.state_dict())["backbone.0.conv.weight"],
        rtol=0,
        atol=0,
    )


def test_pretrained_key_mapping_for_wrapped_layer(tmp_path: Path) -> None:
    from collections import OrderedDict

    source = YOLO26("n").eval()
    path = tmp_path / "source.pt"
    torch.save({"model": source}, path)
    target = YOLO26("n").eval()
    wrapper = torch.nn.Sequential(
        OrderedDict(stem=target.backbone[0], fovea=torch.nn.Identity())
    )
    wrapper.__dict__.update(i=0, f=-1)
    target.backbone[0] = wrapper
    key_map = {
        key: key.replace("backbone.0.", "backbone.0.stem.", 1)
        for key in cast(dict[str, torch.Tensor], source.state_dict())
        if key.startswith("backbone.0.")
    }
    report = target.load_pretrained(path, key_map=key_map)
    assert not report.missing_keys
    assert not report.mismatched_keys
    images = torch.rand(1, 3, 64, 96)
    with torch.no_grad():
        torch.testing.assert_close(
            cast(EvalOutput, target(images)),
            cast(EvalOutput, source(images)),
            rtol=0,
            atol=0,
        )


def test_named_stages_and_shared_backbone_output_caches() -> None:
    model = YOLO26("n").eval()
    assert type(model.backbone) is torch.nn.Sequential
    assert type(model.neck) is torch.nn.Sequential
    assert type(model.head) is torch.nn.Sequential
    assert tuple(name for name, _ in model.backbone.named_children()) == tuple(
        str(i) for i in range(11)
    )
    assert tuple(name for name, _ in model.neck.named_children()) == tuple(
        str(i) for i in range(11, 23)
    )
    assert tuple(name for name, _ in model.head.named_children()) == ("23",)
    image = torch.rand(1, 3, 64, 96)
    crop = torch.rand(1, 3, 96, 128)
    global_outputs: list[object] = []
    foveal_outputs: list[object] = []
    with torch.no_grad():
        global_result, global_index = model.forward_stage(
            model.backbone, image, 0, global_outputs
        )
        global_value = cast(torch.Tensor, global_result)
        saved_global = tuple(global_outputs)
        foveal_result, foveal_index = model.forward_stage(
            model.backbone, crop, 0, foveal_outputs
        )
        foveal_value = cast(torch.Tensor, foveal_result)
        assert global_index == foveal_index == 11
        assert all(a is b for a, b in zip(global_outputs, saved_global, strict=True))
        assert len(global_outputs) == len(foveal_outputs) == 11
        assert global_value.shape[-2:] == (2, 3)
        assert foveal_value.shape[-2:] == (3, 4)
        value, global_index = model.forward_stage(
            model.neck, global_value, global_index, global_outputs
        )
        assert global_index == 23
        result, global_index = model.forward_stage(
            model.head, value, global_index, global_outputs
        )
        prediction = cast(EvalOutput, result)
        assert global_index == 24
        torch.testing.assert_close(
            prediction, cast(EvalOutput, model(image)), rtol=0, atol=0
        )
        assert len(global_outputs) == 24
        assert len(foveal_outputs) == 11


def test_flat_checkpoint_keys_load_into_named_stages(tmp_path: Path) -> None:
    source = YOLO26("n", num_classes=3).eval()
    state = cast(dict[str, torch.Tensor], source.state_dict())
    flat = {"model." + key.split(".", 1)[1]: value for key, value in state.items()}
    assert len(flat) == len(state)
    path = tmp_path / "legacy-flat.pt"
    torch.save(flat, path)
    target = YOLO26("n", num_classes=3).eval()
    report = target.load_pretrained(path)
    assert not report.missing_keys
    assert not report.unexpected_keys
    assert not report.mismatched_keys
    torch.testing.assert_close(target.state_dict(), source.state_dict(), rtol=0, atol=0)
