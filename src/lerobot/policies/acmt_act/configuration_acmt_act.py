"""Configuration for the ACMT-ACT downstream policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.utils.constants import ACTION, OBS_STATE


XENSE0 = "observation.xense.sensor0.force_field"
XENSE1 = "observation.xense.sensor1.force_field"
DQ = "observation.fr3.dq"
TAU_J = "observation.fr3.tau_J"
FT300 = "observation.ft300s.wrench"
O_T_EE = "observation.fr3.O_T_EE"
GRIPPER_GPO = "observation.gripper.gPO"


def rgb_key(camera: str) -> str:
    return f"observation.images.{camera}.rgb"


def depth_key(camera: str) -> str:
    return f"observation.images.{camera}.depth"


CAMERA_KEYS = (
    "camera.cam1",
    "camera.cam2",
    "camera.cam3",
    "camera.cam4",
)
CAMERA_NAMES = ("top", "side", "wrist_left", "wrist_right")
DEFAULT_SOURCE_CAMERA_KEYS = (
    "camera.cam4",
    "camera.cam3",
    "camera.cam1",
    "camera.cam2",
)

# (top, left, height, width), in the original 480x640 frame.
DEFAULT_CROP_PARAMS = {
    "top": (80, 30, 320, 580),
    "side": (140, 60, 320, 580),
    "wrist_left": (80, 30, 320, 580),
    "wrist_right": (80, 30, 320, 580),
}


def _coerce_features(
    features: dict[str, PolicyFeature] | dict[str, Any] | None,
) -> dict[str, PolicyFeature] | None:
    """Accept both dataclass features and JSON-shaped feature dictionaries."""

    if features is None:
        return None
    return {
        key: value
        if isinstance(value, PolicyFeature)
        else PolicyFeature(type=FeatureType(value["type"]), shape=tuple(value["shape"]))
        for key, value in features.items()
    }


@PreTrainedConfig.register_subclass("acmt_act")
@dataclass
class ACMTACTConfig(ACTConfig):
    """ACT configuration with one causal tactile token.

    The ACT visual/state path remains single-step and follows LeRobot's ACT
    implementation. The tactile field is an extra input modality; it is not
    folded into the robot-state feature and is encoded by the policy model.
    """

    tactile_source: str = "none"
    checkpoint_tactile_source: str | None = None
    task_variant: str = "peg"
    checkpoint_task_variant: str | None = None
    # v3 makes the four camera backbones explicit and requires ResNet50.  The
    # schema is part of the serialized ABI so a deployment process cannot
    # silently load an older ResNet18/shared-backbone checkpoint.
    checkpoint_schema: str = "acmt_act.v3"
    checkpoint_schema_version: int = 3
    camera_backbone_mode: str = "independent"
    vision_backbone: str = "resnet50"
    pretrained_backbone_weights: str | None = "ResNet50_Weights.IMAGENET1K_V2"

    # ACMT generator is deliberately external to the policy state_dict. The
    # path is used only by substitution inference.
    generator_checkpoint: str | None = None
    generator_model_config: dict[str, Any] | None = None
    generator_checkpoint_sha256: str | None = None
    generator_task_variant: str | None = None

    # Observation/action protocol.
    n_obs_steps: int = 1
    chunk_size: int = 16
    n_action_steps: int = 8
    action_execution_horizon: int = 8
    pred_horizon: int = 16
    action_dim: int = 8
    state_dim: int = 8
    tactile_history: int = 4  # causal ACMT ring; ACT consumes its latest frame
    control_hz: float = 30.0
    # Train through Accelerate's autocast context.  This is deliberately a
    # config field rather than an environment-only switch so checkpoints keep
    # the precision used for their training run.
    dtype: str = "float16"

    camera_keys: tuple[str, str, str, str] = CAMERA_KEYS
    camera_names: tuple[str, str, str, str] = CAMERA_NAMES
    source_camera_keys: tuple[str, str, str, str] = DEFAULT_SOURCE_CAMERA_KEYS
    crop_params: dict[str, tuple[int, int, int, int]] = field(
        default_factory=lambda: dict(DEFAULT_CROP_PARAMS)
    )

    tactile_feature_dim: int = 160
    force_mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    force_std: tuple[float, float, float] = (1.0, 1.0, 1.0)
    image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    action_feature_names: list[str] = field(
        default_factory=lambda: [*(f"fr3_joint{i}.pos" for i in range(1, 8)), "gripper.pos"]
    )
    # Pixels use the ACT/torchvision ImageNet transform in the custom
    # observation step.  State and action still use LeRobot dataset stats.
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
        # draccus represents a CLI value such as
        # ``--policy.pretrained_backbone_weights=None`` as the literal string
        # ``"None"``. Normalize that spelling before torchvision's ACT base
        # class constructs its first backbone.
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
            str(name): tuple(int(v) for v in values) for name, values in self.crop_params.items()
        }
        self.force_mean = tuple(float(v) for v in self.force_mean)
        self.force_std = tuple(float(v) for v in self.force_std)
        self.image_mean = tuple(float(v) for v in self.image_mean)
        self.image_std = tuple(float(v) for v in self.image_std)
        self.input_features = _coerce_features(self.input_features)
        self.output_features = _coerce_features(self.output_features)

        if self.tactile_source not in {"none", "real", "substitution"}:
            raise ValueError("tactile_source must be one of: none, real, substitution")
        expected_source = "real" if self.tactile_source == "substitution" else self.tactile_source
        if self.checkpoint_tactile_source is None:
            self.checkpoint_tactile_source = expected_source
        if self.checkpoint_tactile_source != expected_source:
            raise ValueError(
                "ACMT-ACT checkpoint/runtime tactile mismatch: "
                f"checkpoint={self.checkpoint_tactile_source!r}, requested={self.tactile_source!r}"
            )
        if self.task_variant not in {"peg", "gear"}:
            raise ValueError("task_variant must be peg or gear")
        if self.checkpoint_task_variant is None:
            self.checkpoint_task_variant = self.task_variant
        if self.checkpoint_task_variant != self.task_variant:
            raise ValueError("ACMT-ACT checkpoints are task-specific")
        if self.tactile_source == "substitution" and not self.generator_checkpoint:
            raise ValueError("substitution mode requires generator_checkpoint")
        if self.generator_task_variant is None:
            self.generator_task_variant = self.task_variant
        if self.generator_task_variant != self.task_variant:
            raise ValueError("ACMT generator and ACMT-ACT policy must use the same task variant")
        if self.generator_checkpoint_sha256 is not None:
            digest = str(self.generator_checkpoint_sha256).lower()
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError("generator_checkpoint_sha256 must be a 64-character hexadecimal digest")
            self.generator_checkpoint_sha256 = digest

        super().__post_init__()

        if self.checkpoint_schema != "acmt_act.v3" or self.checkpoint_schema_version != 3:
            raise ValueError("ACMT-ACT checkpoint schema must be acmt_act.v3")
        if self.camera_backbone_mode != "independent":
            raise ValueError("ACMT-ACT v3 requires four independent camera backbones")
        if self.vision_backbone != "resnet50":
            raise ValueError("ACMT-ACT v3 requires vision_backbone=resnet50")
        if isinstance(self.pretrained_backbone_weights, str) and self.pretrained_backbone_weights not in {
            "IMAGENET1K_V2",
            "ResNet50_Weights.IMAGENET1K_V2",
        }:
            raise ValueError("ACMT-ACT v3 pretrained_backbone_weights must be the ResNet50 IMAGENET1K_V2 value")
        if self.n_obs_steps != 1 or self.chunk_size != 16 or self.n_action_steps != 8:
            raise ValueError("ACMT-ACT fixes n_obs_steps=1, chunk_size=16 and n_action_steps=8")
        if (self.action_execution_horizon, self.pred_horizon, self.action_dim, self.state_dim) != (
            8,
            16,
            8,
            8,
        ):
            raise ValueError("ACMT-ACT fixes the 16-predict/8-execute 8D action protocol")
        if self.tactile_history != 4 or self.control_hz != 30.0:
            raise ValueError("ACMT-ACT fixes a four-frame causal ACMT ring at 30 Hz")
        if self.camera_keys != CAMERA_KEYS or self.camera_names != CAMERA_NAMES:
            raise ValueError("ACMT-ACT camera order must be top, side, wrist_left, wrist_right")
        if (
            len(self.source_camera_keys) != 4
            or len(set(self.source_camera_keys)) != 4
            or set(self.source_camera_keys) != set(self.camera_keys)
        ):
            raise ValueError("ACMT-ACT source_camera_keys must be a permutation of camera_keys")
        expected_crops = set(CAMERA_NAMES)
        if set(self.crop_params) != expected_crops:
            raise ValueError(f"crop_params must contain exactly {sorted(expected_crops)}")
        for name, crop in self.crop_params.items():
            if len(crop) != 4 or any(v < 0 for v in crop):
                raise ValueError(f"invalid crop for {name}: {crop}")
            y, x, height, width = crop
            if (y + height, x + width) > (480, 640) or (height, width) != (320, 580):
                raise ValueError(f"{name} crop must be inside 480x640 and have size 320x580")
        if self.tactile_feature_dim != 160:
            raise ValueError("ACMT-ACT tactile_feature_dim is fixed at 160")
        if len(self.force_mean) != 3 or len(self.force_std) != 3:
            raise ValueError("force_mean and force_std must contain three values")
        if any(v <= 0 for v in self.force_std):
            raise ValueError("force_std values must be positive")
        if len(self.image_mean) != 3 or len(self.image_std) != 3:
            raise ValueError("image_mean and image_std must contain three values")
        if any(v <= 0 for v in self.image_std):
            raise ValueError("image_std values must be positive")

    def _default_input_features(self) -> dict[str, PolicyFeature]:
        features: dict[str, PolicyFeature] = {
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
            XENSE0: PolicyFeature(type=FeatureType.STATE, shape=(3, 35, 20)),
            XENSE1: PolicyFeature(type=FeatureType.STATE, shape=(3, 35, 20)),
        }
        for camera in self.camera_keys:
            features[rgb_key(camera)] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640))
        if self.tactile_source == "substitution":
            # These are auxiliary, non-policy fields consumed only by the
            # frozen causal ACMT generator. Marking them STATE prevents the
            # ACT visual encoder from treating depth as another camera.
            for camera in self.camera_keys[2:]:
                features[depth_key(camera)] = PolicyFeature(type=FeatureType.STATE, shape=(1, 480, 640))
            features[DQ] = PolicyFeature(type=FeatureType.STATE, shape=(7,))
            features[TAU_J] = PolicyFeature(type=FeatureType.STATE, shape=(7,))
            features[FT300] = PolicyFeature(type=FeatureType.STATE, shape=(6,))
            features[O_T_EE] = PolicyFeature(type=FeatureType.STATE, shape=(4, 4))
            features[GRIPPER_GPO] = PolicyFeature(type=FeatureType.STATE, shape=(1,))
        return features

    def validate_features(self) -> None:
        super().validate_features()
        if set(self.image_features) != {rgb_key(camera) for camera in self.camera_keys}:
            raise ValueError("ACMT-ACT requires exactly four RGB camera features")
        if self.robot_state_feature is None or tuple(self.robot_state_feature.shape) != (8,):
            raise ValueError("ACMT-ACT requires observation.state with shape (8,)")
        if self.action_feature is None or tuple(self.action_feature.shape) != (8,):
            raise ValueError("ACMT-ACT requires action with shape (8,)")


__all__ = [
    "ACMTACTConfig",
    "CAMERA_KEYS",
    "CAMERA_NAMES",
    "DEFAULT_SOURCE_CAMERA_KEYS",
    "DEFAULT_CROP_PARAMS",
    "DQ",
    "FT300",
    "GRIPPER_GPO",
    "O_T_EE",
    "TAU_J",
    "XENSE0",
    "XENSE1",
    "depth_key",
    "rgb_key",
]
