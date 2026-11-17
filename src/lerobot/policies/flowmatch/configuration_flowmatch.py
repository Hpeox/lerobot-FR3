"""Native LeRobot 0.6 config. Physical camera/action mappings have no defaults."""

import re
from dataclasses import asdict, dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig

from ._core.config import ModelConfig

CAMERAS = ("side", "top", "wrist1", "wrist2")
ACTION_NAMES = [*(f"fr3_joint{i}.pos" for i in range(1, 8)), "gripper.pos"]


@PreTrainedConfig.register_subclass("flowmatch")
@dataclass
class FlowMatchConfig(PreTrainedConfig):
    model: dict = field(default_factory=lambda: asdict(ModelConfig()))
    stats: dict = field(default_factory=dict)
    tasks: list[str] = field(default_factory=list)
    trained: bool = False
    source_step: int = 100000
    source_bundle_id: str = ""
    execution_mode: str = "whole"
    vision_device: str | None = None
    action_device: str | None = None
    camera_map: dict[str, str] = field(default_factory=dict)
    camera_mapping_source: str = ""
    action_mapping: dict | None = None
    default_task_id: int | None = None
    raw_image_height: int = 480
    raw_image_width: int = 640
    action_feature_names: list[str] = field(default_factory=lambda: list(ACTION_NAMES))
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "STATE": NormalizationMode.IDENTITY,
            "VISUAL": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    def __post_init__(self):
        super().__post_init__()
        cfg = ModelConfig(**self.model)
        if cfg.cameras != CAMERAS or self.n_obs_steps != 1:
            raise ValueError("Expected one observation with side/top/wrist1/wrist2")
        if self.execution_mode not in {"whole", "split"}:
            raise ValueError("execution_mode must be whole or split")
        if self.source_step != 100000:
            raise ValueError("This integration targets the trained 100k checkpoint")
        if self.camera_map:
            if set(self.camera_map) != set(CAMERAS) or len(set(self.camera_map.values())) != 4:
                raise ValueError("camera_map must map all four model cameras to distinct FR3 image keys")
            if any(
                not re.fullmatch(r"observation\.images\.camera\.cam[1-9]\d*\.rgb", key)
                for key in self.camera_map.values()
            ):
                raise ValueError("camera_map values must be native observation.images.camera.camN.rgb keys")
        if self.default_task_id is not None and (
            isinstance(self.default_task_id, bool)
            or not isinstance(self.default_task_id, int)
            or not 0 <= self.default_task_id < len(self.tasks)
        ):
            raise ValueError("default_task_id must index the checkpoint task vocabulary")
        if self.stats:
            from .preprocessing import Normalizer

            Normalizer(self.stats)
        if not self.input_features:
            self.input_features = {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,))}
            for name in CAMERAS:
                self.input_features[self.image_key(name)] = PolicyFeature(
                    type=FeatureType.VISUAL, shape=(3, self.raw_image_height, self.raw_image_width)
                )
        if not self.output_features:
            self.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(8,))}

    def image_key(self, camera):
        return self.camera_map.get(camera, f"observation.images.{camera}")

    def validate_features(self):
        required = {"observation.state", *(self.image_key(name) for name in CAMERAS)}
        if not required.issubset(self.input_features):
            raise ValueError(f"Missing features: {required - set(self.input_features)}")
        if self.input_features["observation.state"].shape != (8,) or self.output_features["action"].shape != (
            8,
        ):
            raise ValueError("Expected eight state/action dimensions")
        if self.action_feature_names != ACTION_NAMES:
            raise ValueError("FR3 action feature names/order must match the driver")
        for camera in CAMERAS:
            feature = self.input_features[self.image_key(camera)]
            if feature.type != FeatureType.VISUAL or feature.shape != (
                3,
                self.raw_image_height,
                self.raw_image_width,
            ):
                raise ValueError("FlowMatch requires raw RGB visual features with the configured resolution")

    def require_deployment_mapping(self):
        from .mapping import ActionMapping

        if not self.camera_map or not self.camera_mapping_source.strip():
            raise ValueError("Physical camera mapping is unconfigured; use raw offline inference only")
        return ActionMapping.from_dict(self.action_mapping)

    def get_optimizer_preset(self):
        raise NotImplementedError("Inference integration only; use the original training project")

    def get_scheduler_preset(self):
        return None

    @property
    def observation_delta_indices(self):
        return None

    @property
    def action_delta_indices(self):
        return list(range(ModelConfig(**self.model).horizon))

    @property
    def reward_delta_indices(self):
        return None

    @property
    def chunk_size(self):
        return ModelConfig(**self.model).horizon

    @property
    def n_action_steps(self):
        return ModelConfig(**self.model).n_action_steps
