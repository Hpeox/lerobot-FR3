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
from .visual_preprocess import prepare_for_frozen_encoder


@dataclass
@ProcessorStepRegistry.register(name="acmt_dp_center480_processor")
class ACMTDPCenter480ProcessorStep(ObservationProcessorStep):
    camera_keys: tuple[str, str]
    visual_preprocess: str = "center480"

    def _prepare_image(self, key: str, value: Any, channels: int, rgb: bool) -> torch.Tensor:
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        if self.visual_preprocess != "center480":
            raise ValueError("ACMT-DP v3 requires visual_preprocess='center480'")
        if tensor.ndim == 3:
            if tensor.shape[0] == channels or tensor.shape[-1] == channels:
                tensor = tensor.unsqueeze(0)
            else:
                raise ValueError(f"{key} has no {channels}-channel axis: {tuple(tensor.shape)}")
        if tensor.ndim != 4:
            raise ValueError(f"{key} must be BCHW or BHWC, got {tuple(tensor.shape)}")
        if tensor.shape[1] != channels and tensor.shape[-1] != channels:
            raise ValueError(f"{key} has no {channels}-channel axis: {tuple(tensor.shape)}")
        channels_first = tensor if tensor.shape[1] == channels else tensor.permute(0, 3, 1, 2).contiguous()
        if tuple(channels_first.shape[-2:]) == (128, 128):
            processed = channels_first
        else:
            # Keep RGB-D processing paired so both streams receive the exact
            # same crop. The depth result is selected below for this key.
            dummy = torch.zeros(
                channels_first.shape[0],
                1,
                channels_first.shape[-2],
                channels_first.shape[-1],
                dtype=channels_first.dtype,
                device=channels_first.device,
            )
            if rgb:
                processed, _ = prepare_for_frozen_encoder(channels_first, dummy)
            else:
                dummy_rgb = torch.zeros(
                    channels_first.shape[0],
                    3,
                    channels_first.shape[-2],
                    channels_first.shape[-1],
                    dtype=channels_first.dtype,
                    device=channels_first.device,
                )
                _, processed = prepare_for_frozen_encoder(dummy_rgb, channels_first)
        if rgb:
            processed = processed.float()
            if processed.numel() and processed.detach().max() > 2.0:
                processed = processed / 255.0
        else:
            # Preserve millimetre values and integer sensor dtype until the
            # model/DFormer boundary; never normalize depth in the processor.
            processed = processed.contiguous()
        return processed.contiguous()

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
            "visual_preprocess": self.visual_preprocess,
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


# Import compatibility for callers that only need to discover the processor
# class. Passing the old ``roi`` argument is intentionally unsupported by the
# v3 constructor/configuration validation.
ACMTDPWristROIProcessorStep = ACMTDPCenter480ProcessorStep


def make_acmt_dp_pre_post_processors(
    config: ACMTDPConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    del dataset_stats
    preprocessor = PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
        steps=[
            RenameObservationsProcessorStep(rename_map={}),
            AddBatchDimensionProcessorStep(),
            ACMTDPCenter480ProcessorStep(
                camera_keys=config.wrist_camera_keys,
                visual_preprocess=config.visual_preprocess,
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
