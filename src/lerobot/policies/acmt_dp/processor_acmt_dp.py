# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.processor.pipeline import ObservationProcessorStep, ProcessorStepRegistry
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from .configuration_acmt_dp import ACMTDPConfig, depth_key, rgb_key


@dataclass
@ProcessorStepRegistry.register(name="acmt_dp_wrist_roi_processor")
class ACMTDPWristROIProcessorStep(ObservationProcessorStep):
    camera_keys: tuple[str, str]
    roi: tuple[int, int, int, int]

    @staticmethod
    def _channels_first(key: str, value: torch.Tensor, channels: int) -> torch.Tensor:
        if value.ndim != 4:
            raise ValueError(f"{key} must be BCHW or BHWC, got {tuple(value.shape)}")
        if value.shape[1] == channels:
            return value
        if value.shape[-1] == channels:
            return value.permute(0, 3, 1, 2).contiguous()
        raise ValueError(f"{key} has no {channels}-channel axis: {tuple(value.shape)}")

    def _prepare_image(self, key: str, value: Any, channels: int, rgb: bool) -> torch.Tensor:
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        tensor = self._channels_first(key, tensor, channels)
        if tensor.shape[-2:] == (128, 128):
            cropped = tensor
        else:
            y0, y1, x0, x1 = self.roi
            if tensor.shape[-2] < y1 or tensor.shape[-1] < x1:
                raise ValueError(f"{key} is too small for ROI {self.roi}: {tuple(tensor.shape)}")
            cropped = tensor[..., y0:y1, x0:x1]
        if cropped.shape[-2:] != (128, 128):
            raise ValueError(f"{key} ROI must be 128x128, got {tuple(cropped.shape[-2:])}")
        if rgb:
            cropped = cropped.float()
            if cropped.numel() and cropped.detach().max() > 2.0:
                cropped = cropped / 255.0
        else:
            cropped = cropped.float()
        return cropped.contiguous()

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        result = dict(observation)
        for camera in self.camera_keys:
            rgb = rgb_key(camera)
            depth = depth_key(camera)
            if rgb in result:
                result[rgb] = self._prepare_image(rgb, result[rgb], 3, rgb=True)
            if depth in result:
                result[depth] = self._prepare_image(depth, result[depth], 1, rgb=False)
        return result

    def get_config(self) -> dict[str, Any]:
        return {
            "camera_keys": list(self.camera_keys),
            "roi": list(self.roi),
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        transformed = {kind: dict(bucket) for kind, bucket in features.items()}
        observations = transformed.get(PipelineFeatureType.OBSERVATION, {})
        for camera in self.camera_keys:
            observations[rgb_key(camera)] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128))
            observations[depth_key(camera)] = PolicyFeature(type=FeatureType.VISUAL, shape=(1, 128, 128))
        return transformed


def make_acmt_dp_pre_post_processors(
    config: ACMTDPConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    del dataset_stats
    preprocessor = PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
        steps=[
            RenameObservationsProcessorStep(rename_map={}),
            AddBatchDimensionProcessorStep(),
            ACMTDPWristROIProcessorStep(
                camera_keys=config.wrist_camera_keys,
                roi=config.wrist_roi,
            ),
            DeviceProcessorStep(device=config.device),
        ],
        name=POLICY_PREPROCESSOR_DEFAULT_NAME,
    )
    postprocessor = PolicyProcessorPipeline[PolicyAction, PolicyAction](
        steps=[DeviceProcessorStep(device="cpu")],
        name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return preprocessor, postprocessor
