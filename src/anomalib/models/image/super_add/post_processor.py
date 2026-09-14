# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Percentile-based post-processor for the SuperADD model.

The default anomalib :class:`~anomalib.post_processing.PostProcessor` derives
its thresholds with :class:`~anomalib.metrics.F1AdaptiveThreshold`, which needs
anomalous validation samples. On datasets whose validation split contains only
normal images (e.g. MVTec AD 2), that threshold degenerates to the maximum
validation score — a max-statistic that grows with the number of pixels, so
high-resolution multi-patch configurations end up with a threshold far beyond
any defect score and F1 collapses even though the anomaly maps are good.

The original SuperADD implementation instead calibrates its threshold on
held-out *normal* images only::

    threshold = percentile(anomaly_map_pixels, 95) * 1.421

A percentile is independent of the input resolution, which keeps the threshold
meaningful for any patch configuration. This module ports that scheme to
anomalib: the percentile is computed over the anomaly maps of the validation
images (normal by design on MVTec AD 2), replacing the degenerate adaptive
threshold. Score normalization is inherited unchanged from the base class.

That margin assumes the anomaly map starts near zero. Backbones whose maps sit on
a large offset with a narrow spread (e.g. RADIO) push it above *every* normal
pixel, so nothing crosses the threshold. The ``"capped"`` method bounds the
original rule by a point near the brightest normal validation pixel::

    threshold = min(p95 * 1.421, p50 + k * (max - p50))

The cap can only lower the original threshold, never raise it, so it leaves
backbones the original rule already suits alone.

See Also:
    - Original implementation: https://github.com/LukasRoom/SuperADD
"""

import torch

from anomalib.post_processing import PostProcessor

PIXEL_THRESHOLD_METHODS = ("scaled_percentile", "capped")


class SuperADDPostProcessor(PostProcessor):
    """Post-processor with percentile-based thresholds from normal validation data.

    Instead of the F1-adaptive threshold (which requires anomalous validation
    samples), thresholds are set to a percentile of the validation scores of
    normal images, scaled by a factor, following the original SuperADD
    implementation.

    Args:
        pixel_threshold_percentile (float): Percentile of validation anomaly map
            pixel scores used as the base pixel threshold. Defaults to ``95.0``
            (original implementation value).
        pixel_threshold_factor (float): Multiplicative factor applied to the
            pixel percentile. Defaults to ``1.421`` (original implementation
            value).
        image_threshold_percentile (float): Percentile of validation image
            scores used as the base image threshold. Defaults to ``95.0``.
        image_threshold_factor (float): Multiplicative factor applied to the
            image percentile. Defaults to ``1.0``.
        samples_per_batch (int): Maximum number of anomaly map pixels sampled
            per validation batch for the percentile estimate. Defaults to
            ``100000``.
        pixel_threshold_method (str): One of :data:`PIXEL_THRESHOLD_METHODS`.
            Defaults to ``"scaled_percentile"``, the original rule.
        pixel_threshold_cap_k (float): Position of the cap between the median and
            the largest normal validation pixel, used by ``"capped"`` only.
            Defaults to ``0.98``.
        **kwargs: Passed to :class:`~anomalib.post_processing.PostProcessor`.

    Example:
        >>> from anomalib.models.image.super_add import SuperADDPostProcessor
        >>> post_processor = SuperADDPostProcessor(pixel_threshold_method="capped")
    """

    def __init__(
        self,
        pixel_threshold_percentile: float = 95.0,
        pixel_threshold_factor: float = 1.421,
        image_threshold_percentile: float = 95.0,
        image_threshold_factor: float = 1.0,
        samples_per_batch: int = 100000,
        pixel_threshold_method: str = "scaled_percentile",
        pixel_threshold_cap_k: float = 0.98,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if pixel_threshold_method not in PIXEL_THRESHOLD_METHODS:
            msg = f"Unknown pixel_threshold_method {pixel_threshold_method!r}. Expected {PIXEL_THRESHOLD_METHODS}."
            raise ValueError(msg)
        self.pixel_threshold_percentile = pixel_threshold_percentile
        self.pixel_threshold_factor = pixel_threshold_factor
        self.image_threshold_percentile = image_threshold_percentile
        self.image_threshold_factor = image_threshold_factor
        self.samples_per_batch = samples_per_batch
        self.pixel_threshold_method = pixel_threshold_method
        self.pixel_threshold_cap_k = pixel_threshold_cap_k

        self._pixel_score_samples: list[torch.Tensor] = []
        self._image_score_samples: list[torch.Tensor] = []
        # The cap anchors on the largest normal pixel, which the random subsample above would
        # underestimate, so it is tracked exactly over the full maps.
        self._pixel_score_max: torch.Tensor | None = None

    def on_validation_batch_end(self, trainer, pl_module, outputs, *args, **kwargs) -> None:  # noqa: ANN001
        """Collect normalization statistics and score samples for the percentile thresholds.

        Unlike the base class, the F1-adaptive threshold metrics are not
        updated; the thresholds are derived from the collected score samples in
        :meth:`on_validation_epoch_end` instead.
        """
        del trainer, pl_module, args, kwargs  # Unused arguments.
        if self.enable_normalization:
            self._image_min_max_metric.update(outputs)
            self._pixel_min_max_metric.update(outputs)
        if self.enable_thresholding:
            anomaly_map = getattr(outputs, "anomaly_map", None)
            if anomaly_map is not None:
                pixel_scores = anomaly_map.detach().flatten().float()
                batch_max = pixel_scores.max().cpu()
                self._pixel_score_max = (
                    batch_max if self._pixel_score_max is None else torch.maximum(self._pixel_score_max, batch_max)
                )
                if pixel_scores.numel() > self.samples_per_batch:
                    indices = torch.randint(pixel_scores.numel(), (self.samples_per_batch,), device=pixel_scores.device)
                    pixel_scores = pixel_scores[indices]
                self._pixel_score_samples.append(pixel_scores.cpu())
            pred_score = getattr(outputs, "pred_score", None)
            if pred_score is not None:
                self._image_score_samples.append(pred_score.detach().flatten().float().cpu())

    def on_validation_epoch_end(self, trainer, pl_module) -> None:  # noqa: ANN001
        """Compute the percentile-based thresholds and the normalization statistics."""
        # The base class computes the min/max normalization statistics; its F1
        # threshold metrics were never updated, so it leaves the thresholds alone.
        super().on_validation_epoch_end(trainer, pl_module)
        if self.enable_thresholding:
            if self._image_score_samples:
                image_scores = torch.cat(self._image_score_samples)
                image_threshold = torch.quantile(image_scores, self.image_threshold_percentile / 100.0)
                self._image_threshold.copy_(image_threshold * self.image_threshold_factor)
                self._image_score_samples = []
            if self._pixel_score_samples:
                pixel_scores = torch.cat(self._pixel_score_samples)
                pixel_threshold = torch.quantile(pixel_scores, self.pixel_threshold_percentile / 100.0)
                pixel_threshold = pixel_threshold * self.pixel_threshold_factor
                if self.pixel_threshold_method == "capped" and self._pixel_score_max is not None:
                    # An interpolating 50th percentile, not `torch.median`, which returns the lower
                    # of the two middle values for an even count and so would not reproduce the
                    # calibration the default `pixel_threshold_cap_k` was chosen under.
                    median = torch.quantile(pixel_scores, 0.5)
                    cap = median + self.pixel_threshold_cap_k * (self._pixel_score_max - median)
                    pixel_threshold = torch.minimum(pixel_threshold, cap)
                self._pixel_threshold.copy_(pixel_threshold)
                self._pixel_score_samples = []
                self._pixel_score_max = None
