"""Configuration for the three-camera ACMT-ACT policy variant."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig

from lerobot.policies.acmt_act.configuration_acmt_act import (
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


CAMERA_KEYS = ("camera.cam2", "camera.cam3", "camera.cam4")
CAMERA_NAMES = ("side", "wrist_left", "wrist_right")
DEFAULT_SOURCE_CAMERA_KEYS = CAMERA_KEYS
DEFAULT_CROP_PARAMS = {
    "side": (140, 60, 320, 580),
    "wrist_left": (80, 30, 320, 580),
    "wrist_right": (80, 30, 320, 580),
}


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
    """ACMT-ACT v2 with side and two wrist cameras only.

    The v2 checkpoint has an explicit ABI and is intentionally not loadable by
    the four-camera ``acmt_act`` policy.  It retains the ACT/tactile/action
    protocol and all causal substitution fields from the parent config.
    """

    checkpoint_schema: str = "acmt_actv2.v1"
    checkpoint_schema_version: int = 1
    camera_keys: tuple[str, ...] = CAMERA_KEYS
    camera_names: tuple[str, ...] = CAMERA_NAMES
    source_camera_keys: tuple[str, ...] = DEFAULT_SOURCE_CAMERA_KEYS
    crop_params: dict[str, tuple[int, int, int, int]] = field(
        default_factory=lambda: dict(DEFAULT_CROP_PARAMS)
    )

    def __post_init__(self) -> None:
        # Draccus can deserialize a CLI ``None`` as the literal string.
        if isinstance(self.pretrained_backbone_weights, str) and self.pretrained_backbone_weights.lower() in {
            "none",
            "null",
        }:
            self.pretrained_backbone_weights = None
        if self.input_features is None:
            self.input_features = self._default_input_features()
        if self.output_features is None:
            self.output_features = {
                "action": PolicyFeature(type=FeatureType.ACTION, shape=(8,))
            }
        self.camera_keys = tuple(self.camera_keys)
        self.camera_names = tuple(self.camera_names)
        self.source_camera_keys = tuple(self.source_camera_keys)
        self.crop_params = {
            str(name): tuple(int(value) for value in values)
            for name, values in self.crop_params.items()
        }
        self.force_mean = tuple(float(value) for value in self.force_mean)
        self.force_std = tuple(float(value) for value in self.force_std)
        self.image_mean = tuple(float(value) for value in self.image_mean)
        self.image_std = tuple(float(value) for value in self.image_std)
        self.input_features = _coerce_features(self.input_features)
        self.output_features = _coerce_features(self.output_features)

        if self.tactile_source not in {"none", "real", "substitution"}:
            raise ValueError("tactile_source must be one of: none, real, substitution")
        expected_source = "real" if self.tactile_source == "substitution" else self.tactile_source
        if self.checkpoint_tactile_source is None:
            self.checkpoint_tactile_source = expected_source
        if self.checkpoint_tactile_source != expected_source:
            raise ValueError(
                "ACMT-ACTv2 checkpoint/runtime tactile mismatch: "
                f"checkpoint={self.checkpoint_tactile_source!r}, requested={self.tactile_source!r}"
            )
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
            raise ValueError("ACMT generator and ACMT-ACTv2 task variants must match")
        if self.generator_checkpoint_sha256 is not None:
            digest = str(self.generator_checkpoint_sha256).lower()
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError("generator_checkpoint_sha256 must be a 64-character hexadecimal digest")
            self.generator_checkpoint_sha256 = digest

        # The parent ACMT config deliberately validates a four-camera v3 ABI.
        # Call ACTConfig directly, then apply the corresponding v2 checks.
        ACTConfig.__post_init__(self)

        if self.checkpoint_schema != "acmt_actv2.v1" or self.checkpoint_schema_version != 1:
            raise ValueError("ACMT-ACTv2 checkpoint schema must be acmt_actv2.v1")
        if self.camera_backbone_mode != "independent":
            raise ValueError("ACMT-ACTv2 requires independent camera backbones")
        if self.vision_backbone != "resnet50":
            raise ValueError("ACMT-ACTv2 requires vision_backbone=resnet50")
        if isinstance(self.pretrained_backbone_weights, str):
            short_name = self.pretrained_backbone_weights.rsplit(".", 1)[-1]
            if not (
                self.pretrained_backbone_weights.startswith("ResNet50_Weights.")
                or short_name in {"DEFAULT", "IMAGENET1K_V1", "IMAGENET1K_V2"}
            ):
                raise ValueError("ACMT-ACTv2 pretrained_backbone_weights must be a ResNet50_Weights value")
        if self.n_obs_steps != 1 or self.chunk_size != 16 or self.n_action_steps != 8:
            raise ValueError("ACMT-ACTv2 fixes n_obs_steps=1, chunk_size=16 and n_action_steps=8")
        if (self.action_execution_horizon, self.pred_horizon, self.action_dim, self.state_dim) != (
            8,
            16,
            8,
            8,
        ):
            raise ValueError("ACMT-ACTv2 fixes the 16-predict/8-execute 8D action protocol")
        if self.tactile_history != 4 or self.control_hz != 30.0:
            raise ValueError("ACMT-ACTv2 fixes a four-frame causal ACMT ring at 30 Hz")
        if self.camera_keys != CAMERA_KEYS or self.camera_names != CAMERA_NAMES:
            raise ValueError("ACMT-ACTv2 camera order must be side, wrist_left, wrist_right")
        if (
            len(self.source_camera_keys) != len(self.camera_keys)
            or len(set(self.source_camera_keys)) != len(self.source_camera_keys)
            or set(self.source_camera_keys) != set(self.camera_keys)
        ):
            raise ValueError("ACMT-ACTv2 source_camera_keys must match the three camera keys")
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
        if len(self.force_mean) != 3 or len(self.force_std) != 3:
            raise ValueError("force_mean and force_std must contain three values")
        if any(value <= 0 for value in self.force_std):
            raise ValueError("force_std values must be positive")
        if len(self.image_mean) != 3 or len(self.image_std) != 3:
            raise ValueError("image_mean and image_std must contain three values")
        if any(value <= 0 for value in self.image_std):
            raise ValueError("image_std values must be positive")

    def _default_input_features(self) -> dict[str, PolicyFeature]:
        features: dict[str, PolicyFeature] = {
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
            XENSE0: PolicyFeature(type=FeatureType.STATE, shape=(3, 35, 20)),
            XENSE1: PolicyFeature(type=FeatureType.STATE, shape=(3, 35, 20)),
        }
        for camera in self.camera_keys:
            features[rgb_key(camera)] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640))
        if self.tactile_source == "substitution":
            wrist_keys = tuple(
                camera
                for camera, name in zip(self.camera_keys, self.camera_names, strict=True)
                if name in {"wrist_left", "wrist_right"}
            )
            source_by_target = dict(zip(self.camera_keys, self.source_camera_keys, strict=True))
            for camera in wrist_keys:
                source = source_by_target[camera]
                features[depth_key(source)] = PolicyFeature(type=FeatureType.STATE, shape=(1, 480, 640))
            features[DQ] = PolicyFeature(type=FeatureType.STATE, shape=(7,))
            features[TAU_J] = PolicyFeature(type=FeatureType.STATE, shape=(7,))
            features[FT300] = PolicyFeature(type=FeatureType.STATE, shape=(6,))
            features[O_T_EE] = PolicyFeature(type=FeatureType.STATE, shape=(4, 4))
            features[GRIPPER_GPO] = PolicyFeature(type=FeatureType.STATE, shape=(1,))
        return features

    def validate_features(self) -> None:
        ACTConfig.validate_features(self)
        if set(self.image_features) != {rgb_key(camera) for camera in self.camera_keys}:
            raise ValueError("ACMT-ACTv2 requires exactly side and two wrist RGB camera features")
        if self.robot_state_feature is None or tuple(self.robot_state_feature.shape) != (8,):
            raise ValueError("ACMT-ACTv2 requires observation.state with shape (8,)")
        if self.action_feature is None or tuple(self.action_feature.shape) != (8,):
            raise ValueError("ACMT-ACTv2 requires action with shape (8,)")


__all__ = [
    "ACMTACTV2Config",
    "CAMERA_KEYS",
    "CAMERA_NAMES",
    "DEFAULT_SOURCE_CAMERA_KEYS",
    "DEFAULT_CROP_PARAMS",
]
