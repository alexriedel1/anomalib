# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""NVIDIA RADIO backbone for SuperADD.

RADIO checkpoints are not distributed through timm, so they are loaded through ``torch.hub`` and
wrapped to present the same surface as
:class:`~anomalib.models.image.super_add.torch_model.DinoV3Backbone`.
"""

import torch
from torch import nn

RADIO_HUB_REPOSITORY = "NVlabs/RADIO"

#: ``(patch_size, depth)`` per checkpoint, so the geometry resolves without downloading weights.
RADIO_SPECS = {
    "c-radio_v4-h": (16, 32),
    "c-radio_v4-so400m": (16, 27),
}


def is_radio_backbone(backbone: str) -> bool:
    """Whether ``backbone`` names a RADIO checkpoint rather than a timm model."""
    return backbone in RADIO_SPECS


class RadioBackbone(nn.Module):
    """NVIDIA RADIO feature extractor.

    Args:
        backbone (str): A key of :data:`RADIO_SPECS`, e.g. ``"c-radio_v4-h"``.
        layers (list[int] | None): Indices of the blocks to return. Defaults to four evenly spaced
            taps ending at the last block, matching the DINOv3 presets.
        pretrained (bool): Whether to load weights. ``False`` resolves the geometry from
            :data:`RADIO_SPECS` only, and makes :meth:`forward` raise.

    Note:
        RADIO normalizes its own input, so it expects images in ``[0, 1]``. ``SuperADD`` configures
        a matching pre-processor automatically.
    """

    def __init__(self, backbone: str, layers: list[int] | None = None, pretrained: bool = True) -> None:
        super().__init__()
        if not is_radio_backbone(backbone):
            msg = f"Unknown RADIO backbone '{backbone}'. Known backbones: {sorted(RADIO_SPECS)}."
            raise ValueError(msg)

        self.backbone_name = backbone
        patch_size, depth = RADIO_SPECS[backbone]
        self.model = None
        if pretrained:
            # `trust_repo` keeps torch.hub from prompting on stdin.
            self.model = torch.hub.load(RADIO_HUB_REPOSITORY, "radio_model", version=backbone, trust_repo=True)  # nosec B614
            patch_size, depth = self.model.patch_size, len(self.model.blocks)
            # A plain attribute, so `.half()` would miss it and every batch would be re-cast back.
            self.model.input_conditioner.dtype = None
            self.model.requires_grad_(requires_grad=False)
        self.model_patch_size = patch_size
        self.depth = depth

        self.layers = layers if layers is not None else [(i + 1) * depth // 4 - 1 for i in range(4)]
        if not self.layers or any(not 0 <= index < depth for index in self.layers):
            msg = f"`layers` must be non-empty indices in 0..{depth - 1} for '{backbone}', got {self.layers}."
            raise ValueError(msg)
        self.eval()

    def train(self, mode: bool = True) -> "RadioBackbone":
        """Stay in eval mode.

        SuperADD builds its memory bank inside Lightning's training loop, and in train mode RADIO's
        cropped position embedding samples a random viewport per image.
        """
        del mode
        return super().train(mode=False)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract intermediate-layer token features from a ``[0, 1]`` image batch.

        Returns:
            list[torch.Tensor]: One tensor per requested layer, in ``NLC`` format.
        """
        if self.model is None:
            msg = f"RadioBackbone('{self.backbone_name}') holds no weights (pretrained=False)."
            raise RuntimeError(msg)

        out_dtype = x.dtype
        with torch.inference_mode():
            result = self.model.forward_intermediates(
                x.to(next(self.model.parameters()).dtype),
                indices=self.layers,
                norm=True,  # applies RADIO's own per-layer inter_feature_normalizer
                output_fmt="NLC",  # its NCHW path reshapes with a patch_size the patch generator never sets
                intermediates_only=True,
            )
        return [tensor.to(out_dtype) for tensor in result]
