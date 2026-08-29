"""LeRobot configuration for the strict Native-DP v5 Real-Hybrid policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
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
    return {
        key: value
        if isinstance(value, PolicyFeature)
        else PolicyFeature(type=FeatureType(value["type"]), shape=tuple(value["shape"]))
        for key, value in features.items()
    }


@PreTrainedConfig.register_subclass("acmt_dp_v5")
@dataclass
class ACMTDPV5Config(PreTrainedConfig):
    tactile_source: str = "none"
    checkpoint_tactile_source: str | None = None
    task_variant: str = "peg"
    checkpoint_task_variant: str | None = None
    checkpoint_schema_version: int = 5
    checkpoint_schema: str = "acmt_dp.native_dp_v5_hybrid"
    visual_preprocess: str = "resize240_center216_range"
    vision_mode: str = "scratch"
    resize_height: int = 240
    resize_width: int = 320
    random_crop: bool = True
    use_group_norm: bool = True

    camera_keys: tuple[str, str, str, str] = ("camera.cam1", "camera.cam2", "camera.cam3", "camera.cam4")
    camera_names: tuple[str, str, str, str] = ("top", "side", "wrist_left", "wrist_right")
    wrist_camera_keys: tuple[str, str] = ("camera.cam3", "camera.cam4")
    rgb_camera_keys: tuple[str, ...] | None = None

    obs_horizon: int = 4
    n_obs_steps: int = 4
    pad_before: int = 3
    internal_horizon: int = 19
    pred_horizon: int = 16
    public_pred_horizon: int = 16
    action_horizon: int = 16
    n_action_steps: int = 8
    action_execution_horizon: int = 8
    tactile_history: int = 4
    control_hz: float = 30.0
    state_dim: int = 8
    action_dim: int = 8
    feature_dim: int = 64
    tactile_dim: int = 160
    crop_height: int = 216
    crop_width: int = 288
    spatial_num_keypoints: int = 32

    diffusion_train_steps: int = 100
    diffusion_inference_steps: int = 8
    unet_dims: tuple[int, ...] = (256, 512, 1024)
    unet_kernel_size: int = 5
    diffusion_step_embed_dim: int = 128
    cond_predict_scale: bool = True

    state_min: tuple[float, ...] = field(default_factory=lambda: (0.0,) * 8)
    state_max: tuple[float, ...] = field(default_factory=lambda: (1.0,) * 8)
    action_min: tuple[float, ...] = field(default_factory=lambda: (-1.0,) * 8)
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

    def __post_init__(self) -> None:
        super().__post_init__()
        self.camera_keys = tuple(self.rgb_camera_keys or self.camera_keys)
        self.rgb_camera_keys = tuple(self.camera_keys)
        self.camera_names = tuple(self.camera_names)
        self.wrist_camera_keys = tuple(self.wrist_camera_keys)
        self.unet_dims = tuple(int(value) for value in self.unet_dims)
        for name in ("state_min", "state_max", "action_min", "action_max", "force_mean", "force_std"):
            setattr(self, name, tuple(float(value) for value in getattr(self, name)))
        self.input_features = _coerce_features(self.input_features)
        self.output_features = _coerce_features(self.output_features)
        if self.tactile_source == "generated":
            raise ValueError("generated is not a v5 runtime mode; use tactigen with a real checkpoint")
        if self.tactile_source not in {"none", "real", "tactigen"}:
            raise ValueError("ACMT-DP v5 tactile_source must be none, real or tactigen")
        expected_source = "real" if self.tactile_source == "tactigen" else self.tactile_source
        if self.checkpoint_tactile_source is None:
            self.checkpoint_tactile_source = expected_source
        if self.checkpoint_tactile_source == "generated":
            raise ValueError("checkpoint_tactile_source='generated' is obsolete; use a real v5 checkpoint")
        if self.checkpoint_tactile_source != expected_source:
            raise ValueError(
                f"v5 checkpoint/runtime tactile mismatch: checkpoint={self.checkpoint_tactile_source!r}, requested={self.tactile_source!r}"
            )
        if self.task_variant not in {"peg", "gear"}:
            raise ValueError("task_variant must be peg or gear")
        if self.checkpoint_task_variant is None:
            self.checkpoint_task_variant = self.task_variant
        if self.checkpoint_task_variant != self.task_variant:
            raise ValueError("v5 checkpoints are task-specific")
        if self.checkpoint_schema_version != 5 or self.checkpoint_schema != "acmt_dp.native_dp_v5_hybrid":
            raise ValueError("checkpoint schema is not Native-DP v5")
        if self.visual_preprocess != "resize240_center216_range":
            raise ValueError("v5 requires resize240_center216_range preprocessing")
        if self.vision_mode != "scratch":
            raise ValueError("v5 deployment only supports scratch ResNet18 checkpoints")
        if (self.resize_height, self.resize_width) != (240, 320):
            raise ValueError("v5 requires raw RGB resize to 240x320")
        if not self.use_group_norm:
            raise ValueError("v5 visual encoders require GroupNorm")
        if self.camera_names != ("top", "side", "wrist_left", "wrist_right") or len(self.camera_keys) != 4:
            raise ValueError("v5 camera order must be top, side, wrist_left, wrist_right")
        if self.wrist_camera_keys != self.camera_keys[2:]:
            raise ValueError("v5 wrist cameras must be camera.cam3 and camera.cam4")
        if (self.obs_horizon, self.n_obs_steps, self.tactile_history, self.pad_before) != (4, 4, 4, 3):
            raise ValueError("v5 fixes four-frame history and pad_before=3")
        if (self.internal_horizon, self.pred_horizon, self.public_pred_horizon, self.action_horizon) != (
            19,
            16,
            16,
            16,
        ):
            raise ValueError("v5 fixes internal/public horizons to 19/16")
        if (self.n_action_steps, self.action_execution_horizon, self.state_dim, self.action_dim) != (
            8,
            8,
            8,
            8,
        ):
            raise ValueError("v5 fixes action/state protocol to 16/8 and dimensions 8")
        if self.feature_dim != 64 or self.tactile_dim != 160 or self.spatial_num_keypoints != 32:
            raise ValueError("v5 fixes 64 visual features, 160 tactile features and 32 keypoints")
        if self.unet_dims != (256, 512, 1024):
            raise ValueError("v5 requires U-Net down_dims=(256,512,1024)")
        if self.unet_kernel_size != 5:
            raise ValueError("v5 requires U-Net kernel_size=5")
        if self.diffusion_step_embed_dim != 128:
            raise ValueError("v5 requires diffusion step embedding dim 128")
        if not self.cond_predict_scale:
            raise ValueError("v5 requires cond_predict_scale=true")
        if self.diffusion_train_steps != 100 or self.diffusion_inference_steps != 8:
            raise ValueError("v5 fixes diffusion_train_steps=100 and diffusion_inference_steps=8")
        if self.control_hz != 30.0:
            raise ValueError("v5 fixes control_hz=30")
        for name, value, size in (
            ("state_min", self.state_min, 8),
            ("state_max", self.state_max, 8),
            ("action_min", self.action_min, 8),
            ("action_max", self.action_max, 8),
            ("force_mean", self.force_mean, 3),
            ("force_std", self.force_std, 3),
        ):
            if len(value) != size:
                raise ValueError(f"{name} must contain {size} values")
        if self.tactile_source == "tactigen" and self.generator_model_config is None:
            raise ValueError("tactigen mode requires generator_model_config")
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
            raise ValueError("Native-DP v5 input_features and output_features are required")
        expected = self._default_input_features()
        missing = sorted(set(expected) - set(self.input_features))
        if missing:
            raise ValueError(f"Native-DP v5 is missing required input features: {missing}")
        for key, feature in expected.items():
            actual = self.input_features[key]
            valid_shape = actual.shape == feature.shape
            if key == OBS_STATE:
                valid_shape = actual.shape in {(7,), (8,)}
            if key.startswith("observation.images.") and key.endswith(".rgb"):
                valid_shape = actual.shape in {feature.shape, (3, 240, 320)}
            if key.startswith("observation.images.") and key.endswith(".depth"):
                valid_shape = actual.shape in {feature.shape, (1, 240, 320)}
            if not valid_shape or actual.type is not feature.type:
                raise ValueError(
                    f"Native-DP v5 feature {key!r} must be {feature.type.value}{feature.shape}, got {actual.type.value}{actual.shape}"
                )
        action = self.output_features.get(ACTION)
        if action is None or action.type is not FeatureType.ACTION or action.shape != (8,):
            raise ValueError("Native-DP v5 output feature 'action' must have shape (8,)")

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> None:
        return None

    @property
    def reward_delta_indices(self) -> None:
        return None

    def get_optimizer_preset(self):
        raise NotImplementedError("Native-DP v5 migrated checkpoints are inference-only")

    def get_scheduler_preset(self):
        return None
