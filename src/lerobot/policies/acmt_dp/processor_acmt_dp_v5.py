"""Input processors for the native-DP v5 raw RGB 4:3 ABI."""

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

from .configuration_acmt_dp_v5 import ACMTDPV5Config, depth_key, rgb_key


def _as_bchw(value: Any, channels: int, key: str) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.ndim == 3:
        if tensor.shape[0] == channels:
            tensor = tensor.unsqueeze(0)
        elif tensor.shape[-1] == channels:
            tensor = tensor.unsqueeze(0).movedim(-1, 1)
    elif tensor.ndim == 4 and tensor.shape[-1] == channels and tensor.shape[1] != channels:
        tensor = tensor.movedim(-1, 1)
    if tensor.ndim != 4 or tensor.shape[1] != channels:
        raise ValueError(f"{key} must be BCHW or BHWC with {channels} channels, got {tuple(tensor.shape)}")
    expected = {(480, 640), (240, 320)} if channels == 3 else {(480, 640)}
    if tuple(tensor.shape[-2:]) not in expected:
        raise ValueError(
            f"{key} has unsupported resolution {tuple(tensor.shape[-2:])}; expected {sorted(expected)}"
        )
    return tensor.contiguous()


@dataclass
@ProcessorStepRegistry.register(name="acmt_dp_native_v5_processor")
class ACMTDPV5ProcessorStep(ObservationProcessorStep):
    camera_keys: tuple[str, ...]
    wrist_camera_keys: tuple[str, str] = ("camera.cam3", "camera.cam4")
    tactile_source: str = "none"
    visual_preprocess: str = "resize240_center216_range"

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        result = dict(observation)
        for camera in self.camera_keys:
            key = rgb_key(camera)
            if key not in result:
                raise KeyError(f"Native-DP v5 observation is missing {key}")
            result[key] = _as_bchw(result[key], 3, key)
        if self.tactile_source == "tactigen":
            for camera in self.wrist_camera_keys:
                key = depth_key(camera)
                if key not in result:
                    raise KeyError(f"tactigen observation is missing {key}")
                result[key] = _as_bchw(result[key], 1, key)
        return result

    def get_config(self) -> dict[str, Any]:
        return {
            "camera_keys": list(self.camera_keys),
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


def make_acmt_dp_v5_pre_post_processors(
    config: ACMTDPV5Config, dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    del dataset_stats
    preprocessor = PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
        steps=[
            RenameObservationsProcessorStep(rename_map={}),
            AddBatchDimensionProcessorStep(),
            ACMTDPV5ProcessorStep(
                camera_keys=config.camera_keys,
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
