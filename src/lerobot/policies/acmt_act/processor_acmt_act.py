"""LeRobot processors for ACMT-ACT.

The policy receives cropped RGB and normalized robot state.  In substitution
mode this step also preserves an untouched RGB-D/proprioceptive copy for the
external causal ACMT generator; those private keys never enter the ACT model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.processor.pipeline import ObservationProcessorStep, ProcessorStepRegistry
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from .configuration_acmt_act import (
    DEFAULT_SOURCE_CAMERA_KEYS,
    DQ,
    FT300,
    GRIPPER_GPO,
    O_T_EE,
    TAU_J,
    XENSE0,
    XENSE1,
    ACMTACTConfig,
    depth_key,
    rgb_key,
)
from ..acmt_dp.gripper_mapping import ACMTDPGripperGPOProcessorStep


# Private fields are intentionally not part of the ACT model input.  They are
# copied before the policy crop/normalizer and are consumed only by the
# substitution inference callback.
GEN_RGB = "_acmt_act.generator.rgb"
GEN_DEPTH = "_acmt_act.generator.depth"
GEN_LOWDIM = "_acmt_act.generator.lowdim"
GEN_POSE = "_acmt_act.generator.pose"
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _tensor(value: Any, key: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    return torch.as_tensor(value)


def _rgb_bchw(value: Any, key: str, *, allow_precropped: bool = False) -> torch.Tensor:
    tensor = _tensor(value, key)
    if tensor.ndim == 3:
        if tensor.shape[0] == 3:
            tensor = tensor.unsqueeze(0)
        elif tensor.shape[-1] == 3:
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        else:
            raise ValueError(f"{key} must be CHW or HWC RGB, got {tuple(tensor.shape)}")
    elif tensor.ndim == 4:
        if tensor.shape[1] == 3:
            pass
        elif tensor.shape[-1] == 3:
            tensor = tensor.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"{key} must be BCHW or BHWC RGB, got {tuple(tensor.shape)}")
    else:
        raise ValueError(f"{key} must be a 3D/4D RGB tensor, got {tuple(tensor.shape)}")
    spatial = tuple(tensor.shape[-2:])
    if spatial != (480, 640) and not (allow_precropped and spatial == (320, 580)):
        raise ValueError(
            f"{key} must be 480x640 before ACMT-ACT crop (or 320x580 memmap input), got {spatial}"
        )
    tensor = tensor.to(dtype=torch.float32)
    if tensor.numel() and float(tensor.detach().amax()) > 1.5:
        tensor = tensor / 255.0
    return tensor.clamp(0.0, 1.0).contiguous()


def _depth_bchw(value: Any, key: str) -> torch.Tensor:
    tensor = _tensor(value, key)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 3:
        if tensor.shape[0] == 1:
            tensor = tensor.unsqueeze(0)
        elif tensor.shape[-1] == 1:
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        else:
            tensor = tensor.unsqueeze(1)
    elif tensor.ndim == 4:
        if tensor.shape[1] == 1:
            pass
        elif tensor.shape[-1] == 1:
            tensor = tensor.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"{key} must be BCHW or BHWC depth, got {tuple(tensor.shape)}")
    else:
        raise ValueError(f"{key} must be a 2D/3D/4D depth tensor, got {tuple(tensor.shape)}")
    if tuple(tensor.shape[-2:]) != (480, 640):
        raise ValueError(f"{key} must be 480x640, got {tuple(tensor.shape[-2:])}")
    return tensor.to(dtype=torch.float32).contiguous()


def _force_bchw(value: Any, key: str) -> torch.Tensor:
    tensor = _tensor(value, key)
    if tensor.ndim == 3:
        if tensor.shape[0] == 3:
            tensor = tensor.unsqueeze(0)
        elif tensor.shape[-1] == 3:
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        else:
            raise ValueError(f"{key} must be CHW or HWC force field, got {tuple(tensor.shape)}")
    elif tensor.ndim == 4:
        if tensor.shape[1] == 3:
            pass
        elif tensor.shape[-1] == 3:
            tensor = tensor.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"{key} must be BCHW or BHWC force field, got {tuple(tensor.shape)}")
    else:
        raise ValueError(f"{key} must be 3D/4D force field, got {tuple(tensor.shape)}")
    if tuple(tensor.shape[-3:]) != (3, 35, 20):
        raise ValueError(f"{key} must end in [3,35,20], got {tuple(tensor.shape)}")
    return tensor.to(dtype=torch.float32).contiguous()


def _vector(value: Any, key: str, width: int) -> torch.Tensor:
    tensor = _tensor(value, key).float()
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2 or tensor.shape[1] != width:
        raise ValueError(f"{key} must be [B,{width}], got {tuple(tensor.shape)}")
    return tensor.contiguous()


def _pose(value: Any, key: str) -> torch.Tensor:
    tensor = _tensor(value, key).float()
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3 or tuple(tensor.shape[1:]) != (4, 4):
        raise ValueError(f"{key} must be [B,4,4], got {tuple(tensor.shape)}")
    return tensor.contiguous()


@dataclass
@ProcessorStepRegistry.register(name="acmt_act_observation_processor")
class ACMTACTObservationProcessorStep(ObservationProcessorStep):
    camera_keys: tuple[str, ...]
    camera_names: tuple[str, ...]
    crop_params: dict[str, tuple[int, int, int, int]]
    source_camera_keys: tuple[str, ...] = DEFAULT_SOURCE_CAMERA_KEYS
    tactile_source: str = "none"
    image_mean: tuple[float, float, float] = _IMAGENET_MEAN
    image_std: tuple[float, float, float] = _IMAGENET_STD

    def __post_init__(self) -> None:
        self.camera_keys = tuple(self.camera_keys)
        self.camera_names = tuple(self.camera_names)
        self.source_camera_keys = tuple(self.source_camera_keys)
        if len(self.camera_keys) != 4 or len(set(self.camera_keys)) != 4:
            raise ValueError("ACMT-ACT camera_keys must contain four distinct cameras")
        if len(self.camera_names) != 4 or len(set(self.camera_names)) != 4:
            raise ValueError("ACMT-ACT camera_names must contain four distinct names")
        if len(self.source_camera_keys) != 4 or len(set(self.source_camera_keys)) != 4:
            raise ValueError("ACMT-ACT source_camera_keys must contain four distinct cameras")
        if set(self.source_camera_keys) != set(self.camera_keys):
            raise ValueError("ACMT-ACT source_camera_keys must match camera_keys")

    def get_config(self) -> dict[str, Any]:
        return {
            "camera_keys": list(self.camera_keys),
            "camera_names": list(self.camera_names),
            "source_camera_keys": list(self.source_camera_keys),
            "crop_params": {key: list(value) for key, value in self.crop_params.items()},
            "tactile_source": self.tactile_source,
            "image_mean": list(self.image_mean),
            "image_std": list(self.image_std),
        }

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        source = dict(observation)
        result = dict(observation)
        rgb_values: dict[str, torch.Tensor] = {}
        for target_camera, source_camera, name in zip(
            self.camera_keys, self.source_camera_keys, self.camera_names, strict=True
        ):
            source_key = rgb_key(source_camera)
            target_key = rgb_key(target_camera)
            if source_key not in source:
                raise KeyError(f"ACMT-ACT observation is missing {source_key}")
            raw = _rgb_bchw(source[source_key], source_key, allow_precropped=self.tactile_source != "substitution")
            if self.tactile_source == "substitution" and name in {"wrist_left", "wrist_right"}:
                rgb_values[name] = raw
            if tuple(raw.shape[-2:]) == (480, 640):
                y, x, height, width = self.crop_params[name]
                cropped = raw[..., y : y + height, x : x + width]
            else:
                # The training-only ACMT-ACT memmap already contains the
                # exact crop.  Deployment still supplies the raw 480x640
                # frame and takes the branch above.
                cropped = raw
            mean = cropped.new_tensor(self.image_mean).view(1, 3, 1, 1)
            std = cropped.new_tensor(self.image_std).view(1, 3, 1, 1)
            result[target_key] = (cropped - mean) / std

        state = _vector(result.get("observation.state"), "observation.state", 8)
        # FR3's canonical state ABI stores the seventh joint followed by the
        # gripper opening in [0,1].  Prefer the raw gPO field when available;
        # otherwise accept legacy recordings that kept gPO on the uint8 wire
        # scale and normalize only that final component.
        if GRIPPER_GPO in result:
            gpo_state = _vector(result[GRIPPER_GPO], GRIPPER_GPO, 1) / 255.0
            state = torch.cat([state[:, :7], gpo_state], dim=-1)
        elif state[:, 7].abs().amax().item() > 1.5:
            state = torch.cat([state[:, :7], state[:, 7:8] / 255.0], dim=-1)
        result["observation.state"] = state

        # Normalize/standardize force fields inside the model so the private
        # substitution path never receives values altered by the LeRobot
        # state normalizer.
        if self.tactile_source == "real":
            if XENSE0 not in result or XENSE1 not in result:
                raise KeyError("real ACMT-ACT mode requires both Xense force fields")
            result[XENSE0] = _force_bchw(result[XENSE0], XENSE0)
            result[XENSE1] = _force_bchw(result[XENSE1], XENSE1)
        else:
            result[XENSE0] = torch.zeros(state.shape[0], 3, 35, 20, dtype=torch.float32, device=state.device)
            result[XENSE1] = torch.zeros_like(result[XENSE0])

        if self.tactile_source == "substitution":
            required = (DQ, TAU_J, FT300, O_T_EE, GRIPPER_GPO)
            missing = [key for key in required if key not in result]
            source_by_target = dict(zip(self.camera_keys, self.source_camera_keys, strict=True))
            missing.extend(
                depth_key(source_by_target[camera])
                for camera in self.camera_keys[2:]
                if depth_key(source_by_target[camera]) not in result
            )
            if missing:
                raise KeyError(f"substitution ACMT-ACT observation is missing {sorted(set(missing))}")
            left, right = (rgb_values["wrist_left"], rgb_values["wrist_right"])
            result[GEN_RGB] = torch.stack([left, right], dim=1)
            result[GEN_DEPTH] = torch.stack(
                [
                    _depth_bchw(
                        result[depth_key(source_by_target[camera])],
                        depth_key(source_by_target[camera]),
                    )
                    for camera in self.camera_keys[2:]
                ],
                dim=1,
            )
            q = state[:, :7]
            dq = _vector(result[DQ], DQ, 7)
            tau = _vector(result[TAU_J], TAU_J, 7)
            ft = _vector(result[FT300], FT300, 6)
            gpo = _vector(result[GRIPPER_GPO], GRIPPER_GPO, 1) / 255.0
            result[GEN_LOWDIM] = torch.cat([q, dq, tau, ft, gpo], dim=-1)
            result[GEN_POSE] = _pose(result[O_T_EE], O_T_EE)
        return result

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        transformed = {kind: dict(bucket) for kind, bucket in features.items()}
        observations = transformed.setdefault(PipelineFeatureType.OBSERVATION, {})
        for camera, name in zip(self.camera_keys, self.camera_names, strict=True):
            observations[rgb_key(camera)] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 320, 580))
        observations[XENSE0] = PolicyFeature(type=FeatureType.STATE, shape=(3, 35, 20))
        observations[XENSE1] = PolicyFeature(type=FeatureType.STATE, shape=(3, 35, 20))
        return transformed


def make_acmt_act_pre_post_processors(
    config: ACMTACTConfig, dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """Create serializable LeRobot pre/post processors for ACMT-ACT."""

    if config.input_features is None or config.output_features is None:
        raise ValueError("ACMT-ACT config features must be initialized before building processors")
    normalize_keys = set(config.image_features) | {"observation.state"}
    features = {**config.input_features, **config.output_features}
    preprocessor = PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
        steps=[
            RenameObservationsProcessorStep(rename_map={}),
            AddBatchDimensionProcessorStep(),
            ACMTACTObservationProcessorStep(
                camera_keys=config.camera_keys,
                camera_names=config.camera_names,
                crop_params=config.crop_params,
                source_camera_keys=config.source_camera_keys,
                tactile_source=config.tactile_source,
                image_mean=config.image_mean,
                image_std=config.image_std,
            ),
            DeviceProcessorStep(device=config.device),
            NormalizerProcessorStep(
                features=features,
                norm_map=config.normalization_mapping,
                stats=dataset_stats,
                device=config.device,
                normalize_observation_keys=normalize_keys,
            ),
        ],
        name=POLICY_PREPROCESSOR_DEFAULT_NAME,
    )
    postprocessor = PolicyProcessorPipeline[PolicyAction, PolicyAction](
        steps=[
            UnnormalizerProcessorStep(
                features=config.output_features,
                norm_map=config.normalization_mapping,
                stats=dataset_stats,
            ),
            DeviceProcessorStep(device="cpu"),
            ACMTDPGripperGPOProcessorStep(),
        ],
        name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return preprocessor, postprocessor


__all__ = [
    "ACMTACTObservationProcessorStep",
    "DEFAULT_SOURCE_CAMERA_KEYS",
    "GEN_DEPTH",
    "GEN_LOWDIM",
    "GEN_POSE",
    "GEN_RGB",
    "make_acmt_act_pre_post_processors",
]
