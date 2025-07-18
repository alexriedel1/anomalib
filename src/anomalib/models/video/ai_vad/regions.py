# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Regions extraction module of AI-VAD model implementation.

This module implements the region extraction stage of the AI-VAD model. It extracts
regions of interest from video frames using object detection and foreground
detection.

Example:
    >>> from anomalib.models.video.ai_vad.regions import RegionExtractor
    >>> import torch
    >>> extractor = RegionExtractor()
    >>> frames = torch.randn(32, 2, 3, 256, 256)  # (N, L, C, H, W)
    >>> regions = extractor(frames)

The module provides the following components:
    - :class:`RegionExtractor`: Main class that handles region extraction using
      object detection and foreground detection
"""

import torch
from torch import nn
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2
from torchvision.ops import box_area, clip_boxes_to_image
from torchvision.transforms.functional import gaussian_blur, rgb_to_grayscale

from anomalib.data.utils.boxes import boxes_to_masks, masks_to_boxes

PERSON_LABEL = 1


class RegionExtractor(nn.Module):
    """Region extractor for AI-VAD.

    This class extracts regions of interest from video frames using object detection and
    foreground detection. It uses a Mask R-CNN model for object detection and can
    optionally detect foreground regions based on frame differences.

    Args:
        box_score_thresh (float, optional): Confidence threshold for bounding box
            predictions. Defaults to ``0.8``.
        persons_only (bool, optional): When enabled, only regions labeled as person are
            included. Defaults to ``False``.
        min_bbox_area (int, optional): Minimum bounding box area. Regions with a surface
            area lower than this value are excluded. Defaults to ``100``.
        max_bbox_overlap (float, optional): Maximum allowed overlap between bounding
            boxes. Defaults to ``0.65``.
        enable_foreground_detections (bool, optional): Add additional foreground
            detections based on pixel difference between consecutive frames.
            Defaults to ``True``.
        foreground_kernel_size (int, optional): Gaussian kernel size used in foreground
            detection. Defaults to ``3``.
        foreground_binary_threshold (int, optional): Value between 0 and 255 which acts
            as binary threshold in foreground detection. Defaults to ``18``.

    Example:
        >>> import torch
        >>> from anomalib.models.video.ai_vad.regions import RegionExtractor
        >>> extractor = RegionExtractor()
        >>> first_frame = torch.randn(2, 3, 256, 256)  # (N, C, H, W)
        >>> last_frame = torch.randn(2, 3, 256, 256)  # (N, C, H, W)
        >>> regions = extractor(first_frame, last_frame)
        >>> # Returns list of dicts with keys: boxes, labels, scores, masks
    """

    def __init__(
        self,
        box_score_thresh: float = 0.8,
        persons_only: bool = False,
        min_bbox_area: int = 100,
        max_bbox_overlap: float = 0.65,
        enable_foreground_detections: bool = True,
        foreground_kernel_size: int = 3,
        foreground_binary_threshold: int = 18,
    ) -> None:
        super().__init__()

        self.persons_only = persons_only
        self.min_bbox_area = min_bbox_area
        self.max_bbox_overlap = max_bbox_overlap
        self.enable_foreground_detections = enable_foreground_detections
        self.foreground_kernel_size = foreground_kernel_size
        self.foreground_binary_threshold = foreground_binary_threshold

        weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.backbone = maskrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=box_score_thresh, rpn_nms_thresh=0.3)

    def forward(self, first_frame: torch.Tensor, last_frame: torch.Tensor) -> list[dict]:
        """Perform forward-pass through region extractor.

        The forward pass consists of:
        1. Object detection on the last frame using Mask R-CNN
        2. Optional foreground detection by comparing first and last frames
        3. Post-processing to filter and refine detections

        Args:
            first_frame (torch.Tensor): Batch of input images of shape ``(N, C, H, W)``
                forming the first frames in the clip.
            last_frame (torch.Tensor): Batch of input images of shape ``(N, C, H, W)``
                forming the last frame in the clip.

        Returns:
            list[dict]: List of Mask R-CNN predictions for each image in the batch. Each
                dict contains:
                - boxes (torch.Tensor): Detected bounding boxes
                - labels (torch.Tensor): Class labels for each detection
                - scores (torch.Tensor): Confidence scores for each detection
                - masks (torch.Tensor): Instance segmentation masks
        """
        with torch.no_grad():
            regions = self.backbone(last_frame)

        if self.enable_foreground_detections:
            regions = self._add_foreground_boxes(
                regions=regions,
                first_frame=first_frame,
                last_frame=last_frame,
                kernel_size=self.foreground_kernel_size,
                binary_threshold=self.foreground_binary_threshold,
            )

        return self.post_process_bbox_detections(regions)

    @staticmethod
    def _add_foreground_boxes(
        regions: list[dict[str, torch.Tensor]],
        first_frame: torch.Tensor,
        last_frame: torch.Tensor,
        kernel_size: int,
        binary_threshold: int,
    ) -> list[dict[str, torch.Tensor]]:
        """Add any foreground regions that were not detected by the region extractor.

        This method adds regions that likely belong to the foreground of the video
        scene, but were not detected by the region extractor module. The foreground
        pixels are determined by taking the pixel difference between two consecutive
        video frames and applying a binary threshold. The final detections consist of
        all connected components in the foreground that do not fall in one of the
        bounding boxes predicted by the region extractor.

        Args:
            regions (list[dict[str, torch.Tensor]]): Region detections for a batch of
                images, generated by the region extraction module.
            first_frame (torch.Tensor): Video frame at time t-1
            last_frame (torch.Tensor): Video frame at time t
            kernel_size (int): Kernel size for Gaussian smoothing applied to input
                frames
            binary_threshold (int): Binary threshold used in foreground detection,
                should be in range ``[0, 255]``

        Returns:
            list[dict[str, torch.Tensor]]: Region detections with foreground regions
                appended. Each dict contains:
                - boxes (torch.Tensor): Updated bounding boxes
                - labels (torch.Tensor): Updated class labels
                - scores (torch.Tensor): Updated confidence scores
                - masks (torch.Tensor): Updated instance masks
        """
        # apply gaussian blur to first and last frame
        first_frame = gaussian_blur(first_frame, [kernel_size, kernel_size])
        last_frame = gaussian_blur(last_frame, [kernel_size, kernel_size])

        # take the abs diff between the blurred images and convert to grayscale
        pixel_diff = torch.abs(first_frame - last_frame)
        pixel_diff = rgb_to_grayscale(pixel_diff).squeeze(1)

        # apply binary threshold to the diff
        foreground_map = (pixel_diff > binary_threshold / 255).int()

        # remove regions already detected by region extractor
        boxes_list = [im_regions["boxes"] for im_regions in regions]
        boxes_list = [
            clip_boxes_to_image(boxes + torch.tensor([-2, -2, 2, 2]).to(boxes.device), foreground_map.shape[-2:])
            for boxes in boxes_list
        ]  # extend boxes by 2 in all directions to ensure full object is included
        boxes_mask = boxes_to_masks(boxes_list, foreground_map.shape[-2:]).int()
        foreground_map *= -boxes_mask + 1  # invert mask

        # find boxes from foreground map
        batch_boxes, _ = masks_to_boxes(foreground_map)

        # append foreground detections to region extractor detections
        for im_regions, boxes, pixel_mask in zip(regions, batch_boxes, foreground_map, strict=True):
            if boxes.shape[0] == 0:
                continue

            # append boxes, labels and scores
            im_regions["boxes"] = torch.cat([im_regions["boxes"], boxes])
            im_regions["labels"] = torch.cat(
                [im_regions["labels"], torch.zeros(boxes.shape[0], device=boxes.device)],
            )  # set label as background, in accordance with region extractor predictions
            im_regions["scores"] = torch.cat(
                [im_regions["scores"], torch.ones(boxes.shape[0], device=boxes.device) * 0.5],
            )  # set confidence to 0.5

            # append masks
            im_boxes_as_list = [box.unsqueeze(0) for box in boxes]  # list with one box per element
            boxes_mask = boxes_to_masks(im_boxes_as_list, pixel_mask.shape[-2:]).int()
            new_masks = pixel_mask.repeat((len(im_boxes_as_list), 1, 1)) * boxes_mask
            im_regions["masks"] = torch.cat([im_regions["masks"], new_masks.unsqueeze(1)])

        return regions

    def post_process_bbox_detections(self, regions: list[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
        """Post-process the region detections.

        The region detections are filtered based on class label, bbox area and overlap
        with other regions.

        Args:
            regions (list[dict[str, torch.Tensor]]): Region detections for a batch of
                images, generated by the region extraction module.

        Returns:
            list[dict[str, torch.Tensor]]: Filtered regions containing only valid
                detections based on the filtering criteria.
        """
        filtered_regions_list = []
        for img_regions in regions:
            filtered_regions = self._keep_only_persons(img_regions) if self.persons_only else img_regions
            filtered_regions = self._filter_by_area(filtered_regions, self.min_bbox_area)
            filtered_regions = self._delete_overlapping_boxes(filtered_regions, self.max_bbox_overlap)
            filtered_regions_list.append(filtered_regions)
        return filtered_regions_list

    def _keep_only_persons(self, regions: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Remove all region detections that are not labeled as a person.

        Args:
            regions (dict[str, torch.Tensor]): Region detections for a single image in
                the batch.

        Returns:
            dict[str, torch.Tensor]: Region detections from which non-person objects
                have been removed.
        """
        keep = torch.where(regions["labels"] == PERSON_LABEL)
        return self.subsample_regions(regions, keep)

    def _filter_by_area(self, regions: dict[str, torch.Tensor], min_area: int) -> dict[str, torch.Tensor]:
        """Remove all regions with a surface area smaller than the specified value.

        Args:
            regions (dict[str, torch.Tensor]): Region detections for a single image in
                the batch.
            min_area (int): Minimum bounding box area. Regions with a surface area
                lower than this value are excluded.

        Returns:
            dict[str, torch.Tensor]: Region detections from which small regions have
                been removed.
        """
        areas = box_area(regions["boxes"])
        keep = torch.where(areas > min_area)
        return self.subsample_regions(regions, keep)

    def _delete_overlapping_boxes(self, regions: dict[str, torch.Tensor], threshold: float) -> dict[str, torch.Tensor]:
        """Delete overlapping bounding boxes.

        For each bounding box, the overlap with all other bounding boxes relative to
        their own surface area is computed. When the relative overlap with any other
        box is higher than the specified threshold, the box is removed. When both boxes
        have a relative overlap higher than the threshold, only the smaller box is
        removed.

        Args:
            regions (dict[str, torch.Tensor]): Region detections for a single image in
                the batch.
            threshold (float): Maximum allowed overlap between bounding boxes.

        Returns:
            dict[str, torch.Tensor]: Region detections from which overlapping regions
                have been removed.
        """
        # sort boxes by area
        areas = box_area(regions["boxes"])
        indices = areas.argsort()

        keep = []
        for idx in range(len(indices)):
            overlap_coords = torch.hstack(
                [
                    torch.max(regions["boxes"][indices[idx], :2], regions["boxes"][indices[idx + 1 :], :2]),  # x1, y1
                    torch.min(regions["boxes"][indices[idx], 2:], regions["boxes"][indices[idx + 1 :], 2:]),  # x2, y2
                ],
            )
            mask = torch.all(overlap_coords[:, :2] < overlap_coords[:, 2:], dim=1)  # filter non-overlapping
            overlap = box_area(overlap_coords) * mask.int()
            overlap_ratio = overlap / areas[indices[idx]]

            if not any(overlap_ratio > threshold):
                keep.append(indices[idx])

        return self.subsample_regions(regions, torch.tensor(keep, dtype=torch.int64))

    @staticmethod
    def subsample_regions(regions: dict[str, torch.Tensor], indices: torch.Tensor) -> dict[str, torch.Tensor]:
        """Subsample the items in a region dictionary based on a Tensor of indices.

        Args:
            regions (dict[str, torch.Tensor]): Region detections for a single image in
                the batch.
            indices (torch.Tensor): Indices of region detections that should be kept.

        Returns:
            dict[str, torch.Tensor]: Subsampled region detections containing only the
                specified indices.
        """
        new_regions_dict = {}
        for key, value in regions.items():
            new_regions_dict[key] = value[indices]
        return new_regions_dict
