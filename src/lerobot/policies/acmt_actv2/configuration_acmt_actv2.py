"""Configuration for the native ACT + DINOv2 ACMT-ACTv2 policy."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.policies.acmt_act.configuration_acmt_act import (
    DQ,
    FT300,
    GRIPPER_GPO,
    O_T_EE,
    TAU_J,
    XENSE0,
    XENSE1,
    ACMTACTConfig,
    CAMERA_KEYS,
    CAMERA_NAMES,
    DEFAULT_CROP_PARAMS,
    DEFAULT_SOURCE_CAMERA_KEYS,
    depth_key,
    rgb_key,
)
from lerobot.utils.constants import ACTION, OBS_STATE


DINOV2_MODEL_NAME = "vit_small_patch14_dinov2.lvd142m"
DINOV2_PATCH_SIZE = 14
DINOV2_TOKEN_DIM = 384
DINOV2_PROJECTION_DIM = 512
DINOV2_IMAGE_SIZE = (336, 448)
DINOV2_GRID_SIZE = (24, 32)
DINOV2_TOKEN_COUNT = 768


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
    """Native ACT contract with one shared frozen DINOv2-S/14 encoder.

    The v2 name remains registered for LeRobot compatibility, while the schema
    rejects the old ResNet50 and DFormerv2 checkpoints.  The inherited ACMT
    generator fields are retained only for substitution runtime support.
    """

    checkpoint_schema: str = "acmt_actv2.dinov2_spatial.v1"
    checkpoint_schema_version: int = 3
    training_contract: str = "native_absolute_physical_gripper_v1"

    visual_encoder_mode: str = "dinov2_spatial"
    camera_backbone_mode: str = "shared"
    vision_backbone: str = "dinov2_vits14"
    pretrained_backbone_weights: str | None = None
    dinov2_model_name: str = DINOV2_MODEL_NAME
    dinov2_checkpoint: str | None = None
    dinov2_checkpoint_sha256: str | None = None
    # Scratch training may request timm's pretrained weights.  The local
    # checkpoint loader disables this because the policy safetensors contains
    # the complete DINO parameters and deployment must not access the network.
    dinov2_pretrained: bool = True
    require_dinov2_checkpoint: bool = False
    dinov2_patch_size: int = DINOV2_PATCH_SIZE
    dinov2_token_dim: int = DINOV2_TOKEN_DIM
    dinov2_projection_dim: int = DINOV2_PROJECTION_DIM
    dinov2_image_size: tuple[int, int] = DINOV2_IMAGE_SIZE
    dinov2_grid_size: tuple[int, int] = DINOV2_GRID_SIZE
    dinov2_token_count: int = DINOV2_TOKEN_COUNT
    optimizer_lr_visual_projection: float = 1e-4

    n_obs_steps: int = 1
    chunk_size: int = 100
    pred_horizon: int = 100
    n_action_steps: int = 1
    action_execution_horizon: int = 1
    temporal_ensemble_coeff: float | None = 0.01
    action_dim: int = 8
    state_dim: int = 8
    tactile_history: int = 4
    control_hz: float = 30.0

    dinov2_spatial_preprocess: bool = True
    use_relative_actions: bool = False
    use_dformer_depth: bool = False
    camera_keys: tuple[str, str, str, str] = CAMERA_KEYS
    camera_names: tuple[str, str, str, str] = CAMERA_NAMES
    source_camera_keys: tuple[str, str, str, str] = DEFAULT_SOURCE_CAMERA_KEYS
    crop_params: dict[str, tuple[int, int, int, int]] = field(
        default_factory=lambda: dict(DEFAULT_CROP_PARAMS)
    )

    # Kept as inert inherited fields so old command-line config parsers do not
    # fail on a shared config file.  v2 has no goal branch or gripper BCE.
    goal_loss_weight: float = 0.0
    gripper_loss_weight: float = 0.0
    state_noise_std: float = 0.0
    state_token_dropout: float = 0.0

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )
    input_features: dict[str, PolicyFeature] | None = None
    output_features: dict[str, PolicyFeature] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.pretrained_backbone_weights, str) and self.pretrained_backbone_weights.lower() in {
            "none",
            "null",
        }:
            self.pretrained_backbone_weights = None
        if self.input_features is None:
            self.input_features = self._default_input_features()
        if self.output_features is None:
            self.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(8,))}

        self.camera_keys = tuple(self.camera_keys)
        self.camera_names = tuple(self.camera_names)
        self.source_camera_keys = tuple(self.source_camera_keys)
        self.crop_params = {
            str(name): tuple(int(value) for value in values)
            for name, values in self.crop_params.items()
        }
        self.dinov2_image_size = tuple(int(value) for value in self.dinov2_image_size)
        self.dinov2_grid_size = tuple(int(value) for value in self.dinov2_grid_size)
        self.force_mean = tuple(float(value) for value in self.force_mean)
        self.force_std = tuple(float(value) for value in self.force_std)
        self.image_mean = tuple(float(value) for value in self.image_mean)
        self.image_std = tuple(float(value) for value in self.image_std)
        self.action_mean = tuple(float(value) for value in self.action_mean)
        self.action_std = tuple(float(value) for value in self.action_std)
        self.input_features = _coerce_features(self.input_features)
        self.output_features = _coerce_features(self.output_features)

        # Bypass ACMTACTConfig/ACTConfig's ResNet-only validator.
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

        if (self.checkpoint_schema, self.checkpoint_schema_version) != (
            "acmt_actv2.dinov2_spatial.v1",
            3,
        ):
            raise ValueError("ACMT-ACTv2 requires the DINOv2 spatial schema version 3")
        if self.training_contract != "native_absolute_physical_gripper_v1":
            raise ValueError("ACMT-ACTv2 requires the native absolute-action contract")
        if self.visual_encoder_mode != "dinov2_spatial" or self.vision_backbone != "dinov2_vits14":
            raise ValueError("ACMT-ACTv2 requires visual_encoder_mode=dinov2_spatial")
        if self.camera_backbone_mode != "shared":
            raise ValueError("ACMT-ACTv2 requires one shared DINOv2 encoder")
        if self.use_dformer_depth:
            raise ValueError("DINOv2 policy input is RGB-only; depth is private to substitution ACMT")
        if (self.n_obs_steps, self.chunk_size, self.pred_horizon, self.n_action_steps) != (1, 100, 100, 1):
            raise ValueError("ACMT-ACTv2 fixes n_obs_steps=1, chunk_size=pred_horizon=100 and n_action_steps=1")
        if (self.action_execution_horizon, self.action_dim, self.state_dim) != (1, 8, 8):
            raise ValueError("ACMT-ACTv2 fixes one-step execution and an 8D action")
        if self.temporal_ensemble_coeff is None or self.temporal_ensemble_coeff <= 0:
            raise ValueError("ACMT-ACTv2 requires a positive temporal_ensemble_coeff")
        if self.tactile_history < 1 or self.control_hz <= 0:
            raise ValueError("invalid tactile history/control frequency")
        if self.camera_keys != CAMERA_KEYS or self.camera_names != CAMERA_NAMES:
            raise ValueError("ACMT-ACTv2 camera order must be top, side, wrist_left, wrist_right")
        if (
            len(self.source_camera_keys) != 4
            or len(set(self.source_camera_keys)) != 4
            or set(self.source_camera_keys) != set(self.camera_keys)
        ):
            raise ValueError("source_camera_keys must be a permutation of camera_keys")
        if set(self.crop_params) != set(CAMERA_NAMES):
            raise ValueError(f"crop_params must contain exactly {sorted(CAMERA_NAMES)}")
        for name, crop in self.crop_params.items():
            if len(crop) != 4 or any(value < 0 for value in crop):
                raise ValueError(f"invalid crop for {name}: {crop}")
            y, x, height, width = crop
            if (y + height, x + width) > (480, 640) or (height, width) != (320, 580):
                raise ValueError(f"{name} crop must be inside 480x640 and have size 320x580")
        if tuple(self.dinov2_image_size) != (336, 448) or tuple(self.dinov2_grid_size) != (24, 32):
            raise ValueError("ACMT-ACTv2 fixes DINOv2 input 336x448 and a 24x32 patch grid")
        if self.dinov2_patch_size != 14 or self.dinov2_token_dim != 384 or self.dinov2_token_count != 768:
            raise ValueError("ACMT-ACTv2 requires DINOv2-S/14's 384D, 768-token output")
        if self.dinov2_projection_dim != self.dim_model:
            raise ValueError("DINOv2 projection dimension must equal ACT dim_model")
        if self.tactile_feature_dim != 160:
            raise ValueError("ACMT-ACTv2 tactile_feature_dim is fixed at 160")
        if len(self.force_mean) != 3 or len(self.force_std) != 3 or any(value <= 0 for value in self.force_std):
            raise ValueError("force_mean/force_std must contain three finite channels")
        if len(self.image_mean) != 3 or len(self.image_std) != 3 or any(value <= 0 for value in self.image_std):
            raise ValueError("image_mean/image_std must contain three positive channels")
        if len(self.action_mean) != 8 or len(self.action_std) != 8 or any(value <= 0 for value in self.action_std):
            raise ValueError("action statistics must contain eight positive channels")
        if self.dinov2_checkpoint_sha256 is not None:
            digest = str(self.dinov2_checkpoint_sha256).lower()
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError("dinov2_checkpoint_sha256 must be a 64-character hexadecimal digest")
            self.dinov2_checkpoint_sha256 = digest
            checkpoint = Path(str(self.dinov2_checkpoint or ""))
            if checkpoint.is_file():
                actual = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                if actual != digest:
                    raise ValueError("DINOv2 checkpoint SHA256 does not match the configuration")

    def _default_input_features(self) -> dict[str, PolicyFeature]:
        features: dict[str, PolicyFeature] = {
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
            XENSE0: PolicyFeature(type=FeatureType.STATE, shape=(3, 35, 20)),
            XENSE1: PolicyFeature(type=FeatureType.STATE, shape=(3, 35, 20)),
        }
        for camera in self.camera_keys:
            features[rgb_key(camera)] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640))
        if self.tactile_source == "substitution":
            # Private fields selected by the rollout frame builder for ACMT;
            # the DINO/ACT model ignores them.
            for camera in self.camera_keys:
                features[depth_key(camera)] = PolicyFeature(type=FeatureType.STATE, shape=(1, 480, 640))
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
        if self.action_feature is None or tuple(self.action_feature.shape) != (8,):
            raise ValueError("ACMT-ACTv2 requires action shape (8,)")


__all__ = [
    "ACMTACTV2Config",
    "CAMERA_KEYS",
    "CAMERA_NAMES",
    "DEFAULT_SOURCE_CAMERA_KEYS",
    "DEFAULT_CROP_PARAMS",
    "DINOV2_MODEL_NAME",
    "DINOV2_PATCH_SIZE",
    "DINOV2_TOKEN_DIM",
    "DINOV2_PROJECTION_DIM",
    "DINOV2_IMAGE_SIZE",
    "DINOV2_GRID_SIZE",
    "DINOV2_TOKEN_COUNT",
]
