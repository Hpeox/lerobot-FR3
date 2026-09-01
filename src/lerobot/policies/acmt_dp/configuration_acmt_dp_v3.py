# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
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


@PreTrainedConfig.register_subclass("acmt_dp_v3")
@dataclass
class ACMTDPV3Config(PreTrainedConfig):
    """Inference-only configuration for the migrated ACMT-DP policy."""

    tactile_source: str = "none"
    checkpoint_tactile_source: str | None = None
    task_variant: str = "peg"
    checkpoint_task_variant: str | None = None
    wrist_camera_keys: tuple[str, str] = ("camera.cam1", "camera.cam2")
    # v3 replaces the old fixed 128px ROI with a raw-frame center crop.
    # Keep the field only so loading an old JSON can fail with a useful message.
    wrist_roi: tuple[int, int, int, int] | None = None
    visual_preprocess: str = "center480"
    checkpoint_schema_version: int = 3
    checkpoint_schema: str = "v3_temporal_center480"
    tactile_history: int = 4
    action_execution_horizon: int = 8
    control_hz: float = 30.0

    n_obs_steps: int = 4
    obs_horizon: int = 4
    # Keep the upstream name alongside LeRobot's historical ``action_horizon``
    # spelling. Both are fixed and serialized for unambiguous v3 manifests.
    pred_horizon: int = 16
    action_horizon: int = 16
    n_action_steps: int = 8
    action_dim: int = 8
    diffusion_train_steps: int = 100
    diffusion_inference_steps: int = 100
    unet_dims: tuple[int, ...] = (256, 512, 1024)
    visual_encoder_name: str = "dformerv2"
    generator_model_config: dict[str, Any] | None = None

    lowdim_mean: tuple[float, ...] = field(default_factory=lambda: (0.0,) * 28)
    lowdim_std: tuple[float, ...] = field(default_factory=lambda: (1.0,) * 28)
    action_min: tuple[float, ...] = field(default_factory=lambda: (-1.0,) * 7 + (0.0,))
    action_max: tuple[float, ...] = field(default_factory=lambda: (1.0,) * 8)
    force_mean: tuple[float, ...] = field(default_factory=lambda: (0.0,) * 3)
    force_std: tuple[float, ...] = field(default_factory=lambda: (1.0,) * 3)

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )
    input_features: dict[str, PolicyFeature] | None = None
    output_features: dict[str, PolicyFeature] | None = None
    action_feature_names: list[str] = field(
        default_factory=lambda: [*(f"fr3_joint{i}.pos" for i in range(1, 8)), "gripper.pos"]
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.wrist_camera_keys = tuple(self.wrist_camera_keys)
        if self.wrist_roi is not None:
            self.wrist_roi = tuple(self.wrist_roi)
        self.unet_dims = tuple(self.unet_dims)
        self.lowdim_mean = tuple(self.lowdim_mean)
        self.lowdim_std = tuple(self.lowdim_std)
        self.action_min = tuple(self.action_min)
        self.action_max = tuple(self.action_max)
        self.force_mean = tuple(self.force_mean)
        self.force_std = tuple(self.force_std)
        self.input_features = _coerce_features(self.input_features)
        self.output_features = _coerce_features(self.output_features)

        if self.tactile_source == "generated":
            raise ValueError(
                "tactile_source='generated' is deprecated: use 'tactigen' with a real "
                "policy checkpoint and an embedded TactiGen generator"
            )
        if self.tactile_source not in {"none", "real", "tactigen"}:
            raise ValueError("tactile_source must be one of: none, real, tactigen")
        if self.checkpoint_tactile_source is None:
            self.checkpoint_tactile_source = (
                "real" if self.tactile_source == "tactigen" else self.tactile_source
            )
        if self.checkpoint_tactile_source == "generated":
            raise ValueError("checkpoint_tactile_source='generated' is deprecated: use 'real'")
        if self.checkpoint_tactile_source not in {"none", "real"}:
            raise ValueError("checkpoint_tactile_source must be 'none' or 'real'")
        expected_checkpoint_source = "real" if self.tactile_source == "tactigen" else self.tactile_source
        if self.checkpoint_tactile_source != expected_checkpoint_source:
            raise ValueError(
                "ACMT-DP checkpoints are mode-specific: "
                f"checkpoint={self.checkpoint_tactile_source!r}, requested={self.tactile_source!r}; "
                f"expected checkpoint source={expected_checkpoint_source!r}"
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
        if len(self.wrist_camera_keys) != 2 or len(set(self.wrist_camera_keys)) != 2:
            raise ValueError("wrist_camera_keys must contain two distinct camera names")
        if self.wrist_roi is not None:
            raise ValueError(
                "ACMT-DP v3 no longer supports wrist_roi=(176,304,256,384); "
                "use visual_preprocess='center480' and reconvert the checkpoint"
            )
        if self.checkpoint_schema_version != 3:
            raise ValueError(
                "ACMT-DP only supports checkpoint schema v3 (temporal tactile + center480); "
                "reconvert the checkpoint"
            )
        if self.checkpoint_schema != "v3_temporal_center480":
            raise ValueError("ACMT-DP v3 requires checkpoint_schema='v3_temporal_center480'")
        if self.visual_preprocess != "center480":
            raise ValueError("ACMT-DP v3 requires visual_preprocess='center480'")
        if self.tactile_history != 4:
            raise ValueError("ACMT-DP v3 fixes tactile_history=4")
        if self.action_execution_horizon != 8 or self.control_hz != 30.0:
            raise ValueError("ACMT-DP v3 fixes action_execution_horizon=8 and control_hz=30")
        if (
            self.n_obs_steps,
            self.obs_horizon,
            self.pred_horizon,
            self.action_horizon,
            self.n_action_steps,
            self.action_dim,
        ) != (4, 4, 16, 16, 8, 8):
            raise ValueError("ACMT-DP v3 fixes obs/action horizons to 4/16/8 and action_dim to 8")
        if self.visual_encoder_name not in {"dformerv2", "tiny"}:
            raise ValueError("visual_encoder_name must be dformerv2 or tiny")
        expected_lengths = {
            "lowdim_mean": (self.lowdim_mean, 28),
            "lowdim_std": (self.lowdim_std, 28),
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
            self.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(self.action_dim,))}
        self.validate_features()

    def _default_input_features(self) -> dict[str, PolicyFeature]:
        features: dict[str, PolicyFeature] = {
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
            DQ: PolicyFeature(type=FeatureType.STATE, shape=(7,)),
            TAU_J: PolicyFeature(type=FeatureType.STATE, shape=(7,)),
            FT300: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
            GRIPPER_GPO: PolicyFeature(type=FeatureType.STATE, shape=(1,)),
        }
        for camera in self.wrist_camera_keys:
            features[rgb_key(camera)] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640))
            features[depth_key(camera)] = PolicyFeature(type=FeatureType.VISUAL, shape=(1, 480, 640))
        if self.tactile_source == "real":
            features[XENSE0] = PolicyFeature(type=FeatureType.STATE, shape=(3, 35, 20))
            features[XENSE1] = PolicyFeature(type=FeatureType.STATE, shape=(3, 35, 20))
        if self.tactile_source == "tactigen":
            features[O_T_EE] = PolicyFeature(type=FeatureType.STATE, shape=(4, 4))
        return features

    def validate_features(self) -> None:
        expected_checkpoint_source = "real" if self.tactile_source == "tactigen" else self.tactile_source
        if self.checkpoint_tactile_source != expected_checkpoint_source:
            raise ValueError(
                "ACMT-DP checkpoint/runtime tactile mode mismatch: "
                f"checkpoint={self.checkpoint_tactile_source!r}, requested={self.tactile_source!r}; "
                f"expected checkpoint source={expected_checkpoint_source!r}"
            )
        if self.checkpoint_task_variant != self.task_variant:
            raise ValueError(
                "ACMT-DP checkpoint/runtime task mismatch: "
                f"checkpoint={self.checkpoint_task_variant!r}, requested={self.task_variant!r}"
            )
        if self.input_features is None or self.output_features is None:
            raise ValueError("ACMT-DP input_features and output_features are required")
        expected = self._default_input_features()
        missing = sorted(set(expected) - set(self.input_features))
        if missing:
            raise ValueError(f"ACMT-DP is missing required input features: {missing}")
        for key, feature in expected.items():
            actual = self.input_features[key]
            # The policy accepts raw camera features before the processor and
            # 128px features for direct offline calls.
            camera_shape_ok = key.startswith("observation.images.") and (
                actual.shape
                in {
                    feature.shape,
                    (3, 128, 128) if key.endswith(".rgb") else (1, 128, 128),
                }
            )
            if (not camera_shape_ok and actual.shape != feature.shape) or actual.type is not feature.type:
                raise ValueError(
                    f"ACMT-DP feature {key!r} must be {feature.type.value}{feature.shape}, "
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
        raise NotImplementedError("Migrated ACMT-DP checkpoints are inference-only")

    def _save_pretrained(self, save_directory: Path) -> None:
        super()._save_pretrained(save_directory)
        config_path = save_directory / "config.json"
        serialized = json.loads(config_path.read_text(encoding="utf-8"))
        serialized["type"] = "acmt_dp_v3"
        config_path.write_text(json.dumps(serialized, indent=4) + "\n", encoding="utf-8")

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        return None
