# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Test OpenVINO inference entrypoint script."""

import sys
from collections.abc import Callable
from importlib.util import find_spec
from pathlib import Path

import pytest

from anomalib import TaskType
from anomalib.models import Padim

sys.path.append("tools/inference")


class TestOpenVINOInferenceEntrypoint:
    """This tests whether the entrypoints run without errors without quantitative measure of the outputs."""

    @pytest.fixture(scope="module")
    @staticmethod
    def get_functions() -> tuple[Callable, Callable]:
        """Get functions from openvino_inference.py."""
        if find_spec("openvino_inference") is not None:
            from tools.inference.openvino_inference import get_parser, infer
        else:
            msg = "Unable to import openvino_inference.py for testing"
            raise ImportError(msg)
        return get_parser, infer

    @staticmethod
    def test_openvino_inference(
        get_functions: tuple[Callable, Callable],
        ckpt_path: Callable[[str], Path],
        get_dummy_inference_image: str,
    ) -> None:
        """Test openvino_inference.py."""
        get_parser, infer = get_functions
        checkpoint_path = ckpt_path("Padim")
        model = Padim.load_from_checkpoint(checkpoint_path)

        # export OpenVINO model
        model.to_openvino(
            export_root=checkpoint_path.parent.parent.parent,
            ov_kwargs={},
            task=TaskType.SEGMENTATION,
        )

        arguments = get_parser().parse_args(
            [
                "--weights",
                str(checkpoint_path.parent.parent) + "/openvino/model.bin",
                "--input",
                get_dummy_inference_image,
                "--output",
                str(checkpoint_path.parent.parent) + "/output.png",
            ],
        )
        infer(arguments)
