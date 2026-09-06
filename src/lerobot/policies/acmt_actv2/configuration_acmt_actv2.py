"""Configuration for the four-camera RGB-D ACMT-ACTv2 policy."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig

from lerobot.policies.acmt_act.configuration_acmt_act import (
    DQ,
    FT300,
    GRIPPER_GPO,
    GOAL_XYZ,
    O_T_EE,
    TAU_J,
    XENSE0,
    XENSE1,
    ACMTACTConfig,
    depth_key,
    rgb_key,
)


CAMERA_KEYS = ("camera.cam1", "camera.cam2", "camera.cam3", "camera.cam4")
CAMERA_NAMES = ("top", "side", "wrist_left", "wrist_right")
DEFAULT_CROP_PARAMS = {
    "top": (80, 30, 320, 580),
    "side": (140, 60, 320, 580),
    "wrist_left": (80, 30, 320, 580),
    "wrist_right": (80, 30, 320, 580),
}
DEFAULT_DFORMER_CHECKPOINT = (
    "/cym/TactiGen/ACMTv4/checkpoints/pretrained/"
    "DFormerv2/pretrained/DFormerv2_Small_pretrained.pth"
)
DEFAULT_DFORMER_SHA256 = "19116988fc86dc9f3e879282237941e11b9b1b5c480edb51e92807311dbc11a6"


def _coerce_features(
    features: dict[str, PolicyFeature] | dict[str, Any] | None,
) -> dict[str, PolicyFeature] | None:
    if features is None:
        return None
    return {
        key: value
        if isinstance(value, PolicyFeature)
        else PolicyFeature(type=FeatureType(value["type"]), shape=tuple(value["shape"]))
        for key, value in features.items()
    }


@PreTrainedConfig.register_subclass("acmt_actv2")
@dataclass
class ACMTACTV2Config(ACMTACTConfig):
    """ACMT-ACTv2 with four independent DFormerv2 spatial encoders."""

    checkpoint_schema: str = "acmt_actv2.dformerv2_spatial.v1"
    checkpoint_schema_version: int = 2
    visual_encoder_mode: str = "dformerv2_s_stage3"
    vision_backbone: str = "dformerv2_s"
    pretrained_backbone_weights: str | None = None
    dformer_checkpoint: str = DEFAULT_DFORMER_CHECKPOINT
    dformer_checkpoint_sha256: str = DEFAULT_DFORMER_SHA256
    require_dformer_checkpoint: bool = True
    dformer_training_phase: str = "frozen"
    use_dformer_depth: bool = True
    camera_keys: tuple[str, str, str, str] = CAMERA_KEYS
    camera_names: tuple[str, str, str, str] = CAMERA_NAMES
    crop_params: dict[str, tuple[int, int, int, int]] = field(
        default_factory=lambda: dict(DEFAULT_CROP_PARAMS)
    )

    def __post_init__(self) -> None:
        if self.input_features is None:
            self.input_features = self._default_input_features()
        if self.output_features is None:
            self.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(8,))}
        self.camera_keys = tuple(self.camera_keys)
        self.camera_names = tuple(self.camera_names)
        self.crop_params = {
            str(name): tuple(int(value) for value in values)
            for name, values in self.crop_params.items()
        }
        self.force_mean = tuple(float(value) for value in self.force_mean)
        self.force_std = tuple(float(value) for value in self.force_std)
        self.image_mean = tuple(float(value) for value in self.image_mean)
        self.image_std = tuple(float(value) for value in self.image_std)
        self.goal_mean = tuple(float(value) for value in self.goal_mean)
        self.goal_std = tuple(float(value) for value in self.goal_std)
        self.action_mean = tuple(float(value) for value in self.action_mean)
        self.action_std = tuple(float(value) for value in self.action_std)
        self.input_features = _coerce_features(self.input_features)
        self.output_features = _coerce_features(self.output_features)

        # Keep all common device/AMP initialization but bypass ACT's ResNet-
        # only vision_backbone check.
        PreTrainedConfig.__post_init__(self)

        if self.tactile_source not in {"none", "real", "substitution"}:
            raise ValueError("tactile_source must be one of: none, real, substitution")
        expected_source = "real" if self.tactile_source == "substitution" else self.tactile_source
        if self.checkpoint_tactile_source is None:
            self.checkpoint_tactile_source = expected_source
        if self.checkpoint_tactile_source != expected_source:
            raise ValueError("ACMT-ACTv2 checkpoint/runtime tactile mismatch")
        if self.task_variant not in {"peg", "gear"}:
            raise ValueError("task_variant must be peg or gear")
        if self.checkpoint_task_variant is None:
            self.checkpoint_task_variant = self.task_variant
        if self.checkpoint_task_variant != self.task_variant:
            raise ValueError("ACMT-ACTv2 checkpoints are task-specific")
        if self.tactile_source == "substitution" and not self.generator_checkpoint:
            raise ValueError("substitution mode requires generator_checkpoint")
        if self.generator_task_variant is None:
            self.generator_task_variant = self.task_variant
        if self.generator_task_variant != self.task_variant:
            raise ValueError("ACMT generator and policy task variants must match")

        if self.checkpoint_schema != "acmt_actv2.dformerv2_spatial.v1" or self.checkpoint_schema_version != 2:
            raise ValueError("ACMT-ACTv2 requires the DFormer spatial checkpoint schema")
        if self.training_contract != "residual_joint_physical_gripper_visual_goal_v1":
            raise ValueError("ACMT-ACTv2 requires the corrected residual-action contract")
        if self.visual_encoder_mode != "dformerv2_s_stage3" or self.vision_backbone != "dformerv2_s":
            raise ValueError("ACMT-ACTv2 requires visual_encoder_mode=dformerv2_s_stage3")
        if self.camera_backbone_mode != "independent":
            raise ValueError("ACMT-ACTv2 requires four independent DFormer encoders")
        if self.dformer_training_phase not in {"frozen", "stage3"}:
            raise ValueError("dformer_training_phase must be frozen or stage3")
        if self.require_dformer_checkpoint and not Path(self.dformer_checkpoint).is_file():
            raise FileNotFoundError(f"DFormer checkpoint not found: {self.dformer_checkpoint}")
        if self.dformer_checkpoint_sha256 and Path(self.dformer_checkpoint).is_file():
            digest = hashlib.sha256(Path(self.dformer_checkpoint).read_bytes()).hexdigest()
            if digest != self.dformer_checkpoint_sha256.lower():
                raise ValueError("DFormer checkpoint SHA256 does not match the configuration")
        if self.n_obs_steps != 1 or self.chunk_size != 16 or self.n_action_steps != 8:
            raise ValueError("ACMT-ACTv2 fixes n_obs_steps=1, chunk_size=16 and n_action_steps=8")
        if (self.action_execution_horizon, self.pred_horizon, self.action_dim, self.state_dim) != (8, 16, 8, 8):
            raise ValueError("ACMT-ACTv2 fixes the 16-predict/8-execute 8D action protocol")
        if self.tactile_history != 4 or self.control_hz != 30.0:
            raise ValueError("ACMT-ACTv2 fixes a four-frame causal ACMT ring at 30 Hz")
        if self.camera_keys != CAMERA_KEYS or self.camera_names != CAMERA_NAMES:
            raise ValueError("ACMT-ACTv2 camera order must be top, side, wrist_left, wrist_right")
        if set(self.crop_params) != set(CAMERA_NAMES):
            raise ValueError(f"crop_params must contain exactly {sorted(CAMERA_NAMES)}")
        for name, crop in self.crop_params.items():
            if len(crop) != 4 or any(value < 0 for value in crop):
                raise ValueError(f"invalid crop for {name}: {crop}")
            y, x, height, width = crop
            if (y + height, x + width) > (480, 640) or (height, width) != (320, 580):
                raise ValueError(f"{name} crop must be inside 480x640 and have size 320x580")
        if self.tactile_feature_dim != 160:
            raise ValueError("ACMT-ACTv2 tactile_feature_dim is fixed at 160")
        if len(self.force_mean) != 3 or len(self.force_std) != 3 or any(value <= 0 for value in self.force_std):
            raise ValueError("force_mean/force_std must contain three finite channels")
        if len(self.image_mean) != 3 or len(self.image_std) != 3 or any(value <= 0 for value in self.image_std):
            raise ValueError("image_mean/image_std must contain three positive channels")
        if len(self.action_mean) != 8 or len(self.action_std) != 8 or any(value <= 0 for value in self.action_std):
            raise ValueError("action statistics must contain eight positive channels")

    def _default_input_features(self) -> dict[str, PolicyFeature]:
        features: dict[str, PolicyFeature] = {
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
            GOAL_XYZ: PolicyFeature(type=FeatureType.STATE, shape=(3,)),
            XENSE0: PolicyFeature(type=FeatureType.STATE, shape=(3, 35, 20)),
            XENSE1: PolicyFeature(type=FeatureType.STATE, shape=(3, 35, 20)),
        }
        for camera in self.camera_keys:
            features[rgb_key(camera)] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640))
            features[depth_key(camera)] = PolicyFeature(type=FeatureType.STATE, shape=(1, 480, 640))
        if self.tactile_source == "substitution":
            features[DQ] = PolicyFeature(type=FeatureType.STATE, shape=(7,))
            features[TAU_J] = PolicyFeature(type=FeatureType.STATE, shape=(7,))
            features[FT300] = PolicyFeature(type=FeatureType.STATE, shape=(6,))
            features[O_T_EE] = PolicyFeature(type=FeatureType.STATE, shape=(4, 4))
            features[GRIPPER_GPO] = PolicyFeature(type=FeatureType.STATE, shape=(1,))
        return features

    def validate_features(self) -> None:
        if not self.image_features or set(self.image_features) != {rgb_key(camera) for camera in CAMERA_KEYS}:
            raise ValueError("ACMT-ACTv2 requires exactly four RGB camera features")
        if self.robot_state_feature is None or tuple(self.robot_state_feature.shape) != (8,):
            raise ValueError("ACMT-ACTv2 requires observation.state shape (8,)")
        for camera in CAMERA_KEYS:
            key = depth_key(camera)
            feature = self.input_features.get(key) if self.input_features else None
            if feature is None or tuple(feature.shape) != (1, 480, 640):
                raise ValueError(f"ACMT-ACTv2 requires depth feature {key} with shape (1,480,640)")
        if self.action_feature is None or tuple(self.action_feature.shape) != (8,):
            raise ValueError("ACMT-ACTv2 requires action shape (8,)")


__all__ = [
    "ACMTACTV2Config",
    "CAMERA_KEYS",
    "CAMERA_NAMES",
    "DEFAULT_CROP_PARAMS",
    "DEFAULT_DFORMER_CHECKPOINT",
    "DEFAULT_DFORMER_SHA256",
]
