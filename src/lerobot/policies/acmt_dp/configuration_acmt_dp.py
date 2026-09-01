# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LeRobot configuration for the Native-DP v4 ACMT-DP policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import LRSchedulerConfig, OptimizerConfig
from lerobot.utils.constants import ACTION, OBS_STATE

DQ = "observation.fr3.dq"
TAU_J = "observation.fr3.tau_J"
O_T_EE = "observation.fr3.O_T_EE"
FT300 = "observation.ft300s.wrench"
GRIPPER_GPO = "observation.gripper.gPO"
XENSE0 = "observation.xense.sensor0.force_field"
XENSE1 = "observation.xense.sensor1.force_field"
# Eight steps are the production real-time setting; one hundred steps are
# retained for slower synchronous experiments using the same model weights.
SUPPORTED_DIFFUSION_INFERENCE_STEPS = (8, 100)


def rgb_key(camera: str) -> str:
    return f"observation.images.{camera}.rgb"


def depth_key(camera: str) -> str:
    return f"observation.images.{camera}.depth"


def _coerce_features(features: dict[str, Any] | None) -> dict[str, PolicyFeature] | None:
    if features is None:
        return None
    result: dict[str, PolicyFeature] = {}
    for key, value in features.items():
        if isinstance(value, PolicyFeature):
            result[key] = value
        else:
            result[key] = PolicyFeature(type=FeatureType(value["type"]), shape=tuple(value["shape"]))
    return result


@PreTrainedConfig.register_subclass("acmt_dp")
@dataclass
class ACMTDPConfig(PreTrainedConfig):
    """Inference-only Native-DP v4 configuration."""

    tactile_source: str = "none"
    checkpoint_tactile_source: str | None = None
    task_variant: str = "peg"
    checkpoint_task_variant: str | None = None
    checkpoint_schema_version: int = 4
    checkpoint_schema: str = "acmt_dp.native_dp_v4"
    vision_mode: str = "scratch"
    visual_preprocess: str = "resize256_center224_imagenet"

    # This order is part of the v4 checkpoint ABI.
    camera_keys: tuple[str, str, str, str] = (
        "camera.cam1",
        "camera.cam2",
        "camera.cam3",
        "camera.cam4",
    )
    camera_names: tuple[str, str, str, str] = ("top", "side", "wrist_left", "wrist_right")
    wrist_camera_keys: tuple[str, str] = ("camera.cam3", "camera.cam4")
    rgb_camera_keys: tuple[str, ...] | None = None

    obs_horizon: int = 4
    n_obs_steps: int = 4
    pred_horizon: int = 16
    action_horizon: int = 16
    n_action_steps: int = 8
    action_execution_horizon: int = 8
    tactile_history: int = 4
    control_hz: float = 30.0
    state_dim: int = 8
    action_dim: int = 8
    feature_dim: int = 512
    tactile_dim: int = 160

    diffusion_train_steps: int = 100
    diffusion_inference_steps: int = 8
    unet_dims: tuple[int, ...] = (256, 512, 1024)
    unet_kernel_size: int = 5
    diffusion_step_embed_dim: int = 128
    cond_predict_scale: bool = True

    state_mean: tuple[float, ...] = field(default_factory=lambda: (0.0,) * 8)
    state_std: tuple[float, ...] = field(default_factory=lambda: (1.0,) * 8)
    action_min: tuple[float, ...] = field(default_factory=lambda: (-1.0,) * 7 + (0.0,))
    action_max: tuple[float, ...] = field(default_factory=lambda: (1.0,) * 8)
    force_mean: tuple[float, ...] = field(default_factory=lambda: (0.0,) * 3)
    force_std: tuple[float, ...] = field(default_factory=lambda: (1.0,) * 3)

    generator_model_config: dict[str, Any] | None = None
    input_features: dict[str, PolicyFeature] | None = None
    output_features: dict[str, PolicyFeature] | None = None
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )
    action_feature_names: list[str] = field(
        default_factory=lambda: [*(f"fr3_joint{i}.pos" for i in range(1, 8)), "gripper.pos"]
    )

    # Legacy fields are retained only so generic config loading can produce a
    # targeted migration message instead of an opaque unknown-field failure.
    wrist_roi: tuple[int, int, int, int] | None = None
    lowdim_mean: tuple[float, ...] | None = None
    lowdim_std: tuple[float, ...] | None = None
    visual_encoder_name: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        self.camera_keys = tuple(self.camera_keys)
        self.camera_names = tuple(self.camera_names)
        if self.rgb_camera_keys is not None:
            self.camera_keys = tuple(self.rgb_camera_keys)
        self.rgb_camera_keys = tuple(self.camera_keys)
        self.wrist_camera_keys = tuple(self.wrist_camera_keys)
        self.unet_dims = tuple(int(value) for value in self.unet_dims)
        self.state_mean = tuple(float(value) for value in self.state_mean)
        self.state_std = tuple(float(value) for value in self.state_std)
        self.action_min = tuple(float(value) for value in self.action_min)
        self.action_max = tuple(float(value) for value in self.action_max)
        self.force_mean = tuple(float(value) for value in self.force_mean)
        self.force_std = tuple(float(value) for value in self.force_std)
        self.input_features = _coerce_features(self.input_features)
        self.output_features = _coerce_features(self.output_features)

        if self.tactile_source == "generated":
            raise ValueError(
                "tactile_source='generated' is no longer supported; reconvert a v4 scratch "
                "real checkpoint and use tactile_source='tactigen'"
            )
        if self.tactile_source not in {"none", "real", "tactigen"}:
            raise ValueError("tactile_source must be one of: none, real, tactigen")
        if self.checkpoint_tactile_source is None:
            self.checkpoint_tactile_source = (
                "real" if self.tactile_source == "tactigen" else self.tactile_source
            )
        if self.checkpoint_tactile_source == "generated":
            raise ValueError("checkpoint_tactile_source='generated' is no longer supported; use 'real'")
        expected_source = "real" if self.tactile_source == "tactigen" else self.tactile_source
        if self.checkpoint_tactile_source != expected_source:
            raise ValueError(
                "ACMT-DP checkpoint/runtime tactile mode mismatch: "
                f"checkpoint={self.checkpoint_tactile_source!r}, requested={self.tactile_source!r}; "
                f"expected checkpoint source={expected_source!r}"
            )
        if self.task_variant not in {"peg", "gear"}:
            raise ValueError("task_variant must be 'peg' or 'gear'")
        if self.checkpoint_task_variant is None:
            self.checkpoint_task_variant = self.task_variant
        if self.checkpoint_task_variant != self.task_variant:
            raise ValueError(
                "ACMT-DP checkpoints are task-specific: "
                f"checkpoint={self.checkpoint_task_variant!r}, requested={self.task_variant!r}"
            )
        if self.checkpoint_schema_version != 4 or self.checkpoint_schema != "acmt_dp.native_dp_v4":
            raise ValueError(
                "ACMT-DP v3/legacy checkpoints are incompatible with Native-DP v4; "
                "reconvert a schema=acmt_dp.native_dp_v4 scratch checkpoint"
            )
        if self.vision_mode != "scratch":
            raise ValueError(
                "LeRobot ACMT-DP v4 only supports scratch checkpoints; "
                "frozen/finetune artifacts must be reconverted from scratch"
            )
        if self.visual_preprocess != "resize256_center224_imagenet":
            raise ValueError(
                "ACMT-DP v4 requires visual_preprocess='resize256_center224_imagenet'; "
                "the old center480 ROI is incompatible"
            )
        if self.wrist_roi is not None:
            raise ValueError("wrist_roi/center480 is a v3 ABI; reconvert the checkpoint for v4")
        if self.visual_encoder_name not in (None, "resnet18"):
            raise ValueError("DFormer policy visual encoders are v3-only; v4 uses shared scratch ResNet18")
        if self.lowdim_mean is not None or self.lowdim_std is not None:
            raise ValueError("28-dimensional v3 lowdim statistics are incompatible; use v4 state statistics")
        if len(self.camera_keys) != 4 or len(set(self.camera_keys)) != 4:
            raise ValueError("ACMT-DP v4 camera_keys must contain four distinct camera names")
        if self.camera_names != ("top", "side", "wrist_left", "wrist_right"):
            raise ValueError("ACMT-DP v4 camera_names must be top, side, wrist_left, wrist_right")
        if len(self.wrist_camera_keys) != 2 or len(set(self.wrist_camera_keys)) != 2:
            raise ValueError("wrist_camera_keys must contain two distinct camera names")
        if self.wrist_camera_keys != self.camera_keys[2:]:
            raise ValueError("wrist_camera_keys must be camera.cam3 and camera.cam4 in v4")
        if (self.obs_horizon, self.n_obs_steps, self.tactile_history) != (4, 4, 4):
            raise ValueError("ACMT-DP v4 fixes obs_horizon=n_obs_steps=tactile_history=4")
        if (
            self.pred_horizon,
            self.action_horizon,
            self.n_action_steps,
            self.action_execution_horizon,
            self.action_dim,
            self.state_dim,
        ) != (16, 16, 8, 8, 8, 8):
            raise ValueError("ACMT-DP v4 fixes action/state protocol to 16/8 and dimensions 8")
        if self.control_hz != 30.0:
            raise ValueError("ACMT-DP v4 fixes control_hz=30")
        if self.diffusion_inference_steps not in SUPPORTED_DIFFUSION_INFERENCE_STEPS:
            supported = ", ".join(str(value) for value in SUPPORTED_DIFFUSION_INFERENCE_STEPS)
            raise ValueError(f"ACMT-DP v4 diffusion_inference_steps must be one of: {supported}")
        if self.feature_dim != 512 or self.tactile_dim != 160:
            raise ValueError("ACMT-DP v4 fixes ResNet feature_dim=512 and tactile_dim=160")
        if self.unet_dims != (256, 512, 1024) or self.unet_kernel_size != 5:
            raise ValueError("ACMT-DP v4 requires U-Net dims=(256,512,1024) and kernel_size=5")
        if self.diffusion_step_embed_dim != 128 or not self.cond_predict_scale:
            raise ValueError("ACMT-DP v4 requires step embedding 128 and cond_predict_scale=true")
        expected_lengths = {
            "state_mean": (self.state_mean, 8),
            "state_std": (self.state_std, 8),
            "action_min": (self.action_min, 8),
            "action_max": (self.action_max, 8),
            "force_mean": (self.force_mean, 3),
            "force_std": (self.force_std, 3),
        }
        for name, (value, expected) in expected_lengths.items():
            if len(value) != expected:
                raise ValueError(f"{name} must contain {expected} values")
        if self.tactile_source == "tactigen" and self.generator_model_config is None:
            raise ValueError("tactigen checkpoints require generator_model_config")

        if self.input_features is None:
            self.input_features = self._default_input_features()
        if self.output_features is None:
            self.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(8,))}
        self.validate_features()

    @property
    def global_cond_dim(self) -> int:
        return self.obs_horizon * (4 * self.feature_dim + self.state_dim + self.tactile_dim)

    def _default_input_features(self) -> dict[str, PolicyFeature]:
        features: dict[str, PolicyFeature] = {
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
            GRIPPER_GPO: PolicyFeature(type=FeatureType.STATE, shape=(1,)),
        }
        for camera in self.camera_keys:
            features[rgb_key(camera)] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640))
        if self.tactile_source == "real":
            features[XENSE0] = PolicyFeature(type=FeatureType.STATE, shape=(3, 35, 20))
            features[XENSE1] = PolicyFeature(type=FeatureType.STATE, shape=(3, 35, 20))
        if self.tactile_source == "tactigen":
            for camera in self.wrist_camera_keys:
                features[depth_key(camera)] = PolicyFeature(type=FeatureType.VISUAL, shape=(1, 480, 640))
            features.update(
                {
                    DQ: PolicyFeature(type=FeatureType.STATE, shape=(7,)),
                    TAU_J: PolicyFeature(type=FeatureType.STATE, shape=(7,)),
                    FT300: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
                    O_T_EE: PolicyFeature(type=FeatureType.STATE, shape=(4, 4)),
                }
            )
        return features

    def validate_features(self) -> None:
        if self.input_features is None or self.output_features is None:
            raise ValueError("ACMT-DP input_features and output_features are required")
        expected = self._default_input_features()
        missing = sorted(set(expected) - set(self.input_features))
        if missing:
            raise ValueError(f"ACMT-DP v4 is missing required input features: {missing}")
        for key, feature in expected.items():
            actual = self.input_features[key]
            valid_shape = actual.shape == feature.shape
            if key == OBS_STATE:
                # FR3 adapters commonly expose q as a 7-D observation and
                # gPO as its separate feature; v4 concatenates them to the
                # required 8-D policy state.
                valid_shape = actual.shape in {(7,), (8,)}
            if key.startswith("observation.images.") and key.endswith(".rgb"):
                valid_shape = actual.shape in {feature.shape, (3, 224, 224)}
            if key.startswith("observation.images.") and key.endswith(".depth"):
                valid_shape = actual.shape in {feature.shape, (1, 128, 128)}
            if not valid_shape or actual.type is not feature.type:
                raise ValueError(
                    f"ACMT-DP v4 feature {key!r} must be {feature.type.value}{feature.shape}, "
                    f"got {actual.type.value}{actual.shape}"
                )
        action = self.output_features.get(ACTION)
        if action is None or action.type is not FeatureType.ACTION or action.shape != (8,):
            raise ValueError("ACMT-DP output feature 'action' must have shape (8,)")

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> None:
        return None

    @property
    def reward_delta_indices(self) -> None:
        return None

    def get_optimizer_preset(self) -> OptimizerConfig:
        raise NotImplementedError("Native-DP v4 ACMT-DP checkpoints are inference-only")

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        return None
