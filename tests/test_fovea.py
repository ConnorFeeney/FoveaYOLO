"""Check the separable FOVEA warp against its discrete weighted-average formula."""

import math
from collections.abc import Iterator

import pytest
import torch
from torch.nn import functional as F

from foveayolo.fovea import foveate


@pytest.fixture(scope="module", autouse=True)
def cpu_threads() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("value", (0.0, 1.0))
def test_uniform_saliency_is_identity(value: float) -> None:
    image = torch.rand(2, 3, 17, 25, dtype=torch.float64)
    saliency = torch.full((2, 1, 3, 5), value, dtype=torch.float64)
    warped = foveate(image, saliency)
    assert warped.shape == image.shape
    assert warped.dtype == image.dtype
    torch.testing.assert_close(warped, image, rtol=0, atol=1e-12)


def test_warp_matches_paper_weighted_coordinate_average() -> None:
    height, width = 9, 13
    y, x = torch.meshgrid(
        torch.linspace(0, 1, height, dtype=torch.float64),
        torch.linspace(0, 1, width, dtype=torch.float64),
        indexing="ij",
    )
    # A coordinate-ramp image reveals the source coordinates sampled by the warp.
    image = torch.stack((x, y)).unsqueeze(0)
    saliency = torch.arange(1, 13, dtype=torch.float64).reshape(1, 1, 3, 4)
    density = F.interpolate(
        saliency, size=(height, width), mode="bilinear", align_corners=True
    )
    density = density / density.amax() + 1e-6
    sigma = 0.18
    std = sigma * (height - 1)
    radius = min(math.ceil(3 * std), height - 1)

    def reference(marginal: torch.Tensor) -> torch.Tensor:
        length = marginal.shape[-1]
        padded = F.pad(marginal, (radius, radius), mode="reflect")[0, 0]
        source = torch.arange(-radius, length + radius, dtype=torch.float64)
        target = torch.arange(length, dtype=torch.float64)
        distances = source.unsqueeze(0) - target.unsqueeze(1)
        weights = torch.exp(-0.5 * (distances / std).square())
        weights = weights * (distances.abs() <= radius) * padded
        return (weights @ source) / weights.sum(dim=1) / (length - 1)

    expected_x = reference(density.sum(dim=2)).expand(height, width)
    expected_y = reference(density.sum(dim=3)).unsqueeze(1).expand(height, width)
    expected = torch.stack((expected_x, expected_y)).unsqueeze(0)
    torch.testing.assert_close(foveate(image, saliency), expected, rtol=0, atol=1e-12)
    torch.testing.assert_close(
        foveate(image, saliency * 100), expected, rtol=0, atol=1e-12
    )


def test_salient_region_attracts_samples_and_preserves_boundaries() -> None:
    y, x = torch.meshgrid(
        torch.linspace(0, 1, 33), torch.linspace(0, 1, 49), indexing="ij"
    )
    image = torch.stack((x, y)).unsqueeze(0).expand(2, -1, -1, -1)
    saliency = torch.stack(
        [
            torch.exp(-((x - center).square() + (y - 0.5).square()) / 0.005)
            for center in (0.3, 0.7)
        ]
    ).unsqueeze(1)
    warped = foveate(image, saliency)
    assert warped[0, 0, 16, 24] < 0.5
    assert warped[1, 0, 16, 24] > 0.5
    torch.testing.assert_close(
        warped[:, 0, :, 0], torch.zeros(2, 33), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(
        warped[:, 0, :, -1], torch.ones(2, 33), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(
        warped[:, 1, 0, :], torch.zeros(2, 49), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(
        warped[:, 1, -1, :], torch.ones(2, 49), atol=1e-6, rtol=0
    )


def test_gradients_reach_image_and_lower_resolution_saliency() -> None:
    image = torch.rand(1, 3, 17, 25, requires_grad=True)
    saliency = torch.rand(1, 1, 3, 5, requires_grad=True)
    warped = foveate(image, saliency)
    gradients = torch.autograd.grad(warped.square().mean(), (image, saliency))
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_low_precision_images_work_on_cpu(dtype: torch.dtype) -> None:
    image = torch.rand(1, 3, 11, 15).to(dtype)
    warped = foveate(image, torch.ones(1, 1, 3, 5))
    assert warped.dtype == dtype
    torch.testing.assert_close(warped, image)


@pytest.mark.parametrize("value", (-1.0, float("nan"), float("inf")))
def test_invalid_saliency_is_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        _ = foveate(torch.zeros(1, 3, 8, 8), torch.full((1, 1, 2, 2), value))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_warp_matches_cpu() -> None:
    image = torch.rand(2, 3, 17, 25)
    saliency = torch.rand(2, 1, 3, 5)
    expected = foveate(image, saliency)
    actual = foveate(image.cuda(), saliency.cuda())
    assert actual.device.type == "cuda"
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-4)
