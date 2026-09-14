# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the SuperADD model.

Verifies that :class:`PatchedExecution` reassembles per-patch backbone outputs
into an exact full-resolution token map. A mock backbone encodes the global
pixel coordinates of every token, so any error in the patch layout, the
overlap splitting, or the batch/patch reshaping shows up as a nonzero
difference from the expected coordinate ramp.
"""

import itertools
from dataclasses import dataclass

import pytest
import torch
from torch import nn
from torchvision.transforms.v2 import Normalize, Resize

from anomalib.models.image.super_add import SuperADD
from anomalib.models.image.super_add.components import RadioBackbone
from anomalib.models.image.super_add.post_processor import SuperADDPostProcessor
from anomalib.models.image.super_add.torch_model import PatchedExecution

MODEL_PATCH_SIZE = 16


class CoordinateBackbone(nn.Module):
    """Mock backbone returning the mean input value of each token cell.

    When fed an image whose channels contain global coordinates, every output
    token holds the exact center coordinate of its 16x16 pixel cell, allowing
    bit-exact verification of the stitched result.
    """

    @staticmethod
    def forward(x: torch.Tensor) -> list[torch.Tensor]:
        """Average-pool each token cell and return NLC tokens."""
        pooled = torch.nn.functional.avg_pool2d(x, MODEL_PATCH_SIZE)
        return [pooled.flatten(2).transpose(1, 2)]


def _coordinate_image(batch_size: int, height: int, width: int) -> torch.Tensor:
    ys = torch.arange(height).float().view(1, 1, height, 1).expand(batch_size, 1, height, width)
    xs = torch.arange(width).float().view(1, 1, 1, width).expand(batch_size, 1, height, width)
    ids = torch.arange(batch_size).float().view(batch_size, 1, 1, 1).expand(batch_size, 1, height, width) * 10000
    return torch.cat([ids, ys, xs], dim=1)


def _expected_tokens(batch_size: int, height: int, width: int) -> torch.Tensor:
    tokens_y, tokens_x = height // MODEL_PATCH_SIZE, width // MODEL_PATCH_SIZE
    center = (MODEL_PATCH_SIZE - 1) / 2
    ey = (torch.arange(tokens_y).float() * MODEL_PATCH_SIZE + center).view(tokens_y, 1).expand(tokens_y, tokens_x)
    ex = (torch.arange(tokens_x).float() * MODEL_PATCH_SIZE + center).view(1, tokens_x).expand(tokens_y, tokens_x)
    expected = []
    for batch_idx in range(batch_size):
        ids = torch.full((tokens_y, tokens_x), batch_idx * 10000.0)
        expected.append(torch.stack([ids, ey, ex], dim=-1))
    return torch.stack(expected)


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize(
    ("height", "width", "patch_size", "patch_overlap"),
    [
        (1024, 1024, 512, 128),  # 3x3 grid, evenly spaced patches
        (896, 896, 448, 64),  # 3x3 grid, uneven strides
        (1024, 768, 448, 64),  # non-square image
        (1024, 1224, 512, 128),  # non-square, width not divisible by 16
    ],
)
def test_patched_execution_stitching_is_exact(
    batch_size: int,
    height: int,
    width: int,
    patch_size: int,
    patch_overlap: int,
) -> None:
    """Stitched token maps must exactly match a single-pass coordinate ramp."""
    patch_exec = PatchedExecution(
        CoordinateBackbone(),
        patch_size=patch_size,
        patch_overlap=patch_overlap,
        model_patch_size=MODEL_PATCH_SIZE,
    )
    image = _coordinate_image(batch_size, height, width)

    result = patch_exec(image)[0]

    expected = _expected_tokens(batch_size, height, width)
    assert result.shape == expected.shape
    torch.testing.assert_close(result, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("dim_size", [448, 535, 1024, 1224])
def test_axis_patch_split_covers_axis_without_gaps(dim_size: int) -> None:
    """Result ROIs must tile the token axis contiguously and completely."""
    patch_exec = PatchedExecution(
        CoordinateBackbone(),
        patch_size=448,
        patch_overlap=64,
        model_patch_size=MODEL_PATCH_SIZE,
    )
    input_rois, prediction_rois, result_rois = patch_exec.axis_patch_split(dim_size)

    assert result_rois[0][0] == 0
    assert result_rois[-1][1] == dim_size // MODEL_PATCH_SIZE
    for previous, current in itertools.pairwise(result_rois):
        assert previous[1] == current[0]
    for (pred_start, pred_end), (res_start, res_end) in zip(prediction_rois, result_rois, strict=True):
        assert pred_end - pred_start == res_end - res_start
    for in_start, in_end in input_rois:
        assert in_end - in_start == patch_exec.patch_size


@dataclass
class DummyValidationBatch:
    """Minimal batch carrying the fields the post-processor consumes."""

    anomaly_map: torch.Tensor
    pred_score: torch.Tensor


def test_percentile_post_processor_thresholds() -> None:
    """Thresholds must equal the configured percentile of validation scores times the factor."""
    post_processor = SuperADDPostProcessor(
        pixel_threshold_percentile=95.0,
        pixel_threshold_factor=1.421,
        image_threshold_percentile=95.0,
        image_threshold_factor=1.0,
    )

    torch.manual_seed(0)
    batches = [DummyValidationBatch(anomaly_map=torch.rand(2, 32, 32), pred_score=torch.rand(2)) for _ in range(3)]
    for batch in batches:
        post_processor.on_validation_batch_end(None, None, batch)
    post_processor.on_validation_epoch_end(None, None)

    all_pixels = torch.cat([batch.anomaly_map.flatten() for batch in batches])
    all_scores = torch.cat([batch.pred_score for batch in batches])
    expected_pixel = torch.quantile(all_pixels, 0.95) * 1.421
    expected_image = torch.quantile(all_scores, 0.95)

    torch.testing.assert_close(post_processor._pixel_threshold, expected_pixel)  # noqa: SLF001
    torch.testing.assert_close(post_processor._image_threshold, expected_image)  # noqa: SLF001
    # normalization statistics must still be computed by the base class
    assert post_processor.pixel_max.item() == all_pixels.max().item()
    assert post_processor.image_min.item() == all_scores.min().item()


def test_radio_backbone_geometry_without_weights() -> None:
    """The geometry and default taps must resolve from the spec table, with no checkpoint."""
    backbone = RadioBackbone("c-radio_v4-h", pretrained=False)

    assert backbone.model_patch_size == 16
    assert backbone.depth == 32
    # the same taps the DINOv3 `huge` preset uses, so the two encoders are compared like for like
    assert backbone.layers == [7, 15, 23, 31]
    # a smaller backbone spaces its taps to its own depth
    assert RadioBackbone("c-radio_v4-so400m", pretrained=False).layers == [5, 12, 19, 26]
    # explicit layers win
    assert RadioBackbone("c-radio_v4-h", layers=[31], pretrained=False).layers == [31]

    with pytest.raises(RuntimeError, match="holds no weights"):
        backbone(torch.zeros(1, 3, 32, 32))
    with pytest.raises(ValueError, match="Unknown RADIO backbone"):
        RadioBackbone("vit_huge_plus_patch16_dinov3", pretrained=False)
    with pytest.raises(ValueError, match="must be non-empty indices"):
        RadioBackbone("c-radio_v4-h", layers=[32], pretrained=False)


#: Validation map with an exact median (0.30), maximum (0.60) and 95th percentile (0.57), so the
#: arithmetic below is closed-form. The original rule gives 0.57 * 1.421 = 0.80997.
CLEAN_VALIDATION_MAP = torch.linspace(0.0, 0.6, 128).reshape(2, 8, 8)
SCALED_PERCENTILE_THRESHOLD = 0.57 * 1.421


@pytest.mark.parametrize(
    ("cap_k", "expected"),
    [
        # 0.30 + 0.8 * (0.60 - 0.30) = 0.54, below the original rule, so the cap binds
        (0.8, 0.54),
        (0.98, 0.30 + 0.98 * 0.30),
        # the cap sits above the original rule, which therefore stands unchanged
        (2.0, SCALED_PERCENTILE_THRESHOLD),
    ],
)
def test_capped_pixel_threshold(cap_k: float, expected: float) -> None:
    """The cap is one-sided: it may lower the original threshold, never raise it."""
    post_processor = SuperADDPostProcessor(pixel_threshold_method="capped", pixel_threshold_cap_k=cap_k)

    batch = DummyValidationBatch(anomaly_map=CLEAN_VALIDATION_MAP.clone(), pred_score=torch.rand(2))
    post_processor.on_validation_batch_end(None, None, batch)
    post_processor.on_validation_epoch_end(None, None)

    assert post_processor._pixel_threshold.item() == pytest.approx(expected, abs=1e-5)  # noqa: SLF001
    assert post_processor._pixel_threshold.item() <= SCALED_PERCENTILE_THRESHOLD + 1e-6  # noqa: SLF001


def test_capped_threshold_anchors_on_the_exact_maximum() -> None:
    """The cap's anchor must be the true maximum over full maps, not the random pixel subsample."""
    post_processor = SuperADDPostProcessor(
        pixel_threshold_method="capped",
        samples_per_batch=8,  # far below the 256 pixels per batch, so the subsample misses most
    )

    quiet = torch.full((1, 16, 16), 0.5)
    spike = torch.full((1, 16, 16), 0.5)
    spike[0, 0, 0] = 1.0
    for anomaly_map in (spike, quiet):
        post_processor.on_validation_batch_end(None, None, DummyValidationBatch(anomaly_map, torch.rand(1)))

    # exact, and carried across batches rather than reset by the later quiet one
    assert post_processor._pixel_score_max.item() == 1.0  # noqa: SLF001
    # and it is released once consumed, so a second validation run starts clean
    post_processor.on_validation_epoch_end(None, None)
    assert post_processor._pixel_score_max is None  # noqa: SLF001


def test_invalid_threshold_method_is_rejected() -> None:
    """An unknown method must fail at construction rather than silently fall back."""
    with pytest.raises(ValueError, match="Unknown pixel_threshold_method"):
        SuperADDPostProcessor(pixel_threshold_method="aug_capped")


def test_radio_pre_processor_skips_imagenet_normalization() -> None:
    """RADIO standardizes its own input, so the default pre-processor must leave it in [0, 1]."""
    default_transforms = SuperADD.configure_pre_processor().transform.transforms
    radio_transforms = SuperADD.configure_pre_processor(normalize=False).transform.transforms

    assert any(isinstance(transform, Normalize) for transform in default_transforms)
    assert not any(isinstance(transform, Normalize) for transform in radio_transforms)
    # the resize is kept either way
    assert any(isinstance(transform, Resize) for transform in radio_transforms)
