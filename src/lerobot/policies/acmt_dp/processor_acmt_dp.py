# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LeRobot input processing for Native-DP v4.

Unlike v3, the policy RGB path remains raw here.  The model owns the exact
256-resize/224-center-crop/quantization/ImageNet transform, while raw wrist
RGB-D is retained for the TactiGen side branch.
"""

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


ACMT_DP_DEFAULT_SOURCE_CAMERA_KEYS = (
    "camera.cam4",
    "camera.cam3",
    "camera.cam1",
    "camera.cam2",
)


def _as_bchw(value: Any, channels: int, key: str) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.ndim == 3:
        if tensor.shape[0] == channels or tensor.shape[-1] == channels:
            tensor = tensor.unsqueeze(0)
        else:
            raise ValueError(f"{key} has no {channels}-channel axis: {tuple(tensor.shape)}")
    if tensor.ndim != 4:
        raise ValueError(f"{key} must be BCHW or BHWC, got {tuple(tensor.shape)}")
    if tensor.shape[1] == channels:
        result = tensor
    elif tensor.shape[-1] == channels:
        result = tensor.permute(0, 3, 1, 2).contiguous()
    else:
        raise ValueError(f"{key} has no {channels}-channel axis: {tuple(tensor.shape)}")
    spatial = tuple(result.shape[-2:])
    allowed = {(480, 640), (224, 224)} if channels == 3 else {(480, 640), (128, 128)}
    if spatial not in allowed:
        raise ValueError(f"{key} has unsupported resolution {spatial}; expected one of {sorted(allowed)}")
    return result.contiguous()


@dataclass
@ProcessorStepRegistry.register(name="acmt_dp_native_v4_processor")
class ACMTDPNativeV4ProcessorStep(ObservationProcessorStep):
    camera_keys: tuple[str, ...]
    source_camera_keys: tuple[str, ...] = ACMT_DP_DEFAULT_SOURCE_CAMERA_KEYS
    wrist_camera_keys: tuple[str, str] = ("camera.cam3", "camera.cam4")
    tactile_source: str = "none"
    visual_preprocess: str = "resize256_center224_imagenet"

    def __post_init__(self) -> None:
        self.camera_keys = tuple(self.camera_keys)
        self.source_camera_keys = tuple(self.source_camera_keys)
        self.wrist_camera_keys = tuple(self.wrist_camera_keys)
        if len(self.camera_keys) != 4 or len(set(self.camera_keys)) != 4:
            raise ValueError("ACMT-DP v4 camera_keys must contain four distinct cameras")
        if len(self.source_camera_keys) != 4 or len(set(self.source_camera_keys)) != 4:
            raise ValueError("ACMT-DP v4 source_camera_keys must contain four distinct cameras")
        if set(self.source_camera_keys) != set(self.camera_keys):
            raise ValueError("ACMT-DP v4 source_camera_keys must match camera_keys")
        if len(self.wrist_camera_keys) != 2 or len(set(self.wrist_camera_keys)) != 2:
            raise ValueError("ACMT-DP v4 wrist_camera_keys must contain two distinct cameras")
        if not set(self.wrist_camera_keys).issubset(self.camera_keys):
            raise ValueError("ACMT-DP v4 wrist_camera_keys must be included in camera_keys")

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        source = dict(observation)
        result = dict(observation)
        for target_camera, source_camera in zip(self.camera_keys, self.source_camera_keys):
            source_key = rgb_key(source_camera)
            target_key = rgb_key(target_camera)
            if source_key not in source:
                raise KeyError(f"ACMT-DP v4 observation is missing {source_key}")
            result[target_key] = _as_bchw(source[source_key], 3, source_key)
        if self.tactile_source == "tactigen":
            source_by_target = dict(zip(self.camera_keys, self.source_camera_keys))
            for target_camera in self.wrist_camera_keys:
                source_camera = source_by_target[target_camera]
                source_key = depth_key(source_camera)
                target_key = depth_key(target_camera)
                if source_key not in source:
                    raise KeyError(f"tactigen observation is missing generator depth {source_key}")
                result[target_key] = _as_bchw(source[source_key], 1, source_key)
        return result

    def get_config(self) -> dict[str, Any]:
        return {
            "camera_keys": list(self.camera_keys),
            "source_camera_keys": list(self.source_camera_keys),
            "wrist_camera_keys": list(self.wrist_camera_keys),
            "tactile_source": self.tactile_source,
            "visual_preprocess": self.visual_preprocess,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        transformed = {kind: dict(bucket) for kind, bucket in features.items()}
        observations = transformed.setdefault(PipelineFeatureType.OBSERVATION, {})
        for camera in self.camera_keys:
            observations[rgb_key(camera)] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640))
        if self.tactile_source == "tactigen":
            for camera in self.wrist_camera_keys:
                observations[depth_key(camera)] = PolicyFeature(type=FeatureType.VISUAL, shape=(1, 480, 640))
        return transformed


# Keep import compatibility for applications that used the v3 class name. It
# now deliberately implements the v4 raw-input ABI.
ACMTDPCenter480ProcessorStep = ACMTDPNativeV4ProcessorStep
ACMTDPWristROIProcessorStep = ACMTDPNativeV4ProcessorStep


def make_acmt_dp_pre_post_processors(
    config: ACMTDPConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    del dataset_stats
    preprocessor = PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
        steps=[
            RenameObservationsProcessorStep(rename_map={}),
            AddBatchDimensionProcessorStep(),
            ACMTDPNativeV4ProcessorStep(
                camera_keys=config.camera_keys,
                source_camera_keys=ACMT_DP_DEFAULT_SOURCE_CAMERA_KEYS,
                wrist_camera_keys=config.wrist_camera_keys,
                tactile_source=config.tactile_source,
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
