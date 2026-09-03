from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from lerobot.policies.acmt_act.configuration_acmt_act import (
    XENSE0,
    XENSE1,
    ACMTACTConfig,
    rgb_key,
)
from lerobot.policies.acmt_act.modeling_acmt_act import ACMTACTPolicy
from lerobot.policies.acmt_act.processor_acmt_act import (
    ACMTACTObservationProcessorStep,
    DEFAULT_SOURCE_CAMERA_KEYS,
    GEN_DEPTH,
    GEN_LOWDIM,
    GEN_POSE,
    GEN_RGB,
    make_acmt_act_pre_post_processors,
)
from lerobot.policies.acmt_dp.gripper_mapping import ACMTDPGripperGPOProcessorStep
from lerobot.policies.factory import get_policy_class, make_policy_config, make_pre_post_processors
from lerobot.utils.constants import OBS_STATE


CAMERAS = ("camera.cam1", "camera.cam2", "camera.cam3", "camera.cam4")


def _config(mode: str = "none") -> ACMTACTConfig:
    return ACMTACTConfig(
        device="cpu",
        pretrained_backbone_weights=None,
        tactile_source=mode,
        generator_checkpoint="/tmp/acmt-act-test-generator.pt" if mode == "substitution" else None,
    )


def _rgb_batch(config: ACMTACTConfig, size: int = 32) -> dict[str, torch.Tensor]:
    batch: dict[str, torch.Tensor] = {OBS_STATE: torch.zeros(1, 8)}
    for key in config.image_features:
        batch[key] = torch.zeros(1, 3, size, size)
    return batch


def test_factory_and_config_protocol() -> None:
    config = make_policy_config("acmt_act", device="cpu", pretrained_backbone_weights=None)
    assert isinstance(config, ACMTACTConfig)
    assert config.type == "acmt_act"
    assert config.n_obs_steps == 1
    assert config.chunk_size == 16
    assert config.n_action_steps == 8
    assert get_policy_class("acmt_act") is ACMTACTPolicy


def test_v3_accepts_serialized_short_resnet34_weight_name() -> None:
    config = ACMTACTConfig(
        device="cpu",
        pretrained_backbone_weights="IMAGENET1K_V1",
        tactile_source="none",
    )
    assert config.pretrained_backbone_weights == "IMAGENET1K_V1"


def test_crop_boxes_are_exact_and_reject_wrong_resolution() -> None:
    config = _config()
    preprocessor, _ = make_acmt_act_pre_post_processors(config)
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[..., 0] = np.arange(480, dtype=np.uint8)[:, None]
    image[..., 1] = np.arange(640, dtype=np.uint8)[None, :]
    raw = {OBS_STATE: np.zeros(8, dtype=np.float32)}
    for key in config.image_features:
        raw[key] = image.copy()
    processed = preprocessor(raw)
    top = processed[rgb_key(CAMERAS[0])]
    side = processed[rgb_key(CAMERAS[1])]
    assert top.shape == (1, 3, 320, 580)
    assert side.shape == (1, 3, 320, 580)
    mean = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
    top_rgb = top[0] * std + mean
    side_rgb = side[0] * std + mean
    torch.testing.assert_close(top_rgb[:, 0, 0], torch.tensor((80, 30, 0), dtype=torch.float32) / 255)
    torch.testing.assert_close(side_rgb[:, 0, 0], torch.tensor((140, 60, 0), dtype=torch.float32) / 255)

    bad = dict(raw)
    bad[next(iter(config.image_features))] = np.zeros((479, 640, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="480x640"):
        preprocessor(bad)


def test_runtime_camera_mapping_uses_a_snapshot_without_cyclic_overwrite() -> None:
    config = _config()
    preprocessor, _ = make_acmt_act_pre_post_processors(config)
    raw = {OBS_STATE: np.zeros(8, dtype=np.float32)}
    for index, key in enumerate(config.image_features, start=1):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[..., 0] = index * 16
        raw[key] = image

    processed = preprocessor(raw)
    mean = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
    for target, source in zip(CAMERAS, DEFAULT_SOURCE_CAMERA_KEYS, strict=True):
        value = processed[rgb_key(target)][0] * std + mean
        expected = float(int(source.removeprefix("camera.cam")) * 16) / 255.0
        torch.testing.assert_close(value[:, 0, 0], torch.tensor((expected, 0.0, 0.0)))

    assert config.source_camera_keys == DEFAULT_SOURCE_CAMERA_KEYS


def test_independent_camera_backbones_and_output_shape() -> None:
    policy = ACMTACTPolicy(_config())
    assert len(policy.model.backbone) == 4
    assert len(policy.model.encoder_img_feat_input_proj) == 4
    assert len({id(module) for module in policy.model.backbone}) == 4
    assert len({id(module) for module in policy.model.encoder_img_feat_input_proj}) == 4
    assert policy.config.checkpoint_schema == "acmt_act.v3"
    assert policy.config.checkpoint_schema_version == 3
    assert policy.config.vision_backbone == "resnet50"
    assert not any(isinstance(module, nn.BatchNorm2d) for module in policy.model.modules())
    batch = _rgb_batch(policy.config, size=32)
    batch[XENSE0] = torch.zeros(1, 3, 35, 20)
    batch[XENSE1] = torch.zeros(1, 3, 35, 20)
    batch["action"] = torch.zeros(1, 16, 8)
    batch["action_is_pad"] = torch.zeros(1, 16, dtype=torch.bool)
    loss, metrics = policy(batch)
    assert loss.ndim == 0
    assert "l1_loss" in metrics and "kld_loss" in metrics
    with torch.no_grad():
        actions = policy.predict_action_chunk(
            {key: value for key, value in batch.items() if key not in {"action", "action_is_pad"}}
        )
    assert actions.shape == (1, 16, 8)


def test_none_and_real_have_identical_parameters() -> None:
    torch.manual_seed(7)
    none = ACMTACTPolicy(_config("none"))
    torch.manual_seed(7)
    real = ACMTACTPolicy(_config("real"))
    assert list(none.state_dict()) == list(real.state_dict())
    assert all(torch.equal(none.state_dict()[key], real.state_dict()[key]) for key in none.state_dict())


def test_processor_serialization_and_real_shapes(tmp_path) -> None:
    config = _config("real")
    preprocessor, postprocessor = make_acmt_act_pre_post_processors(config)
    raw = {
        OBS_STATE: np.zeros(8, dtype=np.float32),
        XENSE0: np.zeros((35, 20, 3), dtype=np.float32),
        XENSE1: np.zeros((3, 35, 20), dtype=np.float32),
    }
    for key in config.image_features:
        raw[key] = np.zeros((480, 640, 3), dtype=np.uint8)
    result = preprocessor(raw)
    assert result[XENSE0].shape == (1, 3, 35, 20)
    assert result[XENSE1].shape == (1, 3, 35, 20)
    preprocessor.save_pretrained(tmp_path)
    from lerobot.processor import PolicyProcessorPipeline

    restored = PolicyProcessorPipeline.from_pretrained(tmp_path, config_filename="policy_preprocessor.json")
    assert restored(raw)[rgb_key(CAMERAS[0])].shape == (1, 3, 320, 580)
    step = next(item for item in restored.steps if isinstance(item, ACMTACTObservationProcessorStep))
    assert step.source_camera_keys == DEFAULT_SOURCE_CAMERA_KEYS
    assert isinstance(postprocessor.steps[-1], ACMTDPGripperGPOProcessorStep)


def test_gripper_postprocessor_maps_policy_opening_to_fr3_gpo_direction() -> None:
    step = ACMTDPGripperGPOProcessorStep()
    action = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    mapped = step.action(action)
    torch.testing.assert_close(mapped[:, :7], action[:, :7])
    torch.testing.assert_close(mapped[:, 7], torch.tensor([1.0, 3.0 / 255.0]))


def test_substitution_overrides_serialized_real_observation_step(tmp_path) -> None:
    """Keep real normalizer stats while switching only the tactile source."""

    real = _config("real")
    real_preprocessor, real_postprocessor = make_pre_post_processors(real)
    real_preprocessor.save_pretrained(tmp_path)
    real_postprocessor.save_pretrained(tmp_path)

    substitution = ACMTACTConfig(
        device="cpu",
        pretrained_backbone_weights=None,
        tactile_source="substitution",
        generator_checkpoint="/tmp/acmt-act-test-generator.pt",
    )
    loaded, _ = make_pre_post_processors(substitution, pretrained_path=tmp_path)
    step = next(item for item in loaded.steps if isinstance(item, ACMTACTObservationProcessorStep))
    assert step.tactile_source == "substitution"


def test_loader_rejects_cross_source_checkpoint_before_weight_load(tmp_path) -> None:
    config = _config("none")
    config.save_pretrained(tmp_path)
    with pytest.raises(ValueError, match="checkpoint tactile source"):
        ACMTACTPolicy.from_pretrained(
            tmp_path,
            config=ACMTACTConfig(
                device="cpu",
                pretrained_backbone_weights=None,
                tactile_source="real",
            ),
        )


class _FakeGenerator:
    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]] = []

    def reset(self) -> None:
        return None

    def predict_next(self, observation, pose, action):
        self.calls.append((observation, pose.clone(), action.clone()))
        return torch.ones(action.shape[0], 2, 35, 20, 3)


def test_substitution_uses_only_actual_action_and_keeps_generator_external() -> None:
    policy = ACMTACTPolicy(_config("none"))
    policy.config.tactile_source = "substitution"
    fake = _FakeGenerator()
    policy._generator_runtime = fake
    policy.reset()
    policy._generator_runtime = fake
    batch = _rgb_batch(policy.config)
    batch.update(
        {
            GEN_RGB: torch.zeros(1, 4, 2, 3, 480, 640),
            GEN_DEPTH: torch.zeros(1, 4, 2, 1, 480, 640),
            GEN_LOWDIM: torch.zeros(1, 4, 28),
            GEN_POSE: torch.eye(4).reshape(1, 4, 4),
        }
    )
    window = policy.observe(batch)
    policy.notify_action_executed(torch.arange(8, dtype=torch.float32).reshape(1, 8), window)
    assert len(fake.calls) == 1
    torch.testing.assert_close(fake.calls[0][2], torch.arange(8, dtype=torch.float32).reshape(1, 8))
    assert policy._generator_runtime is fake
    assert not any(key.startswith("_generator_runtime") for key in policy.state_dict())
    assert policy._latest_window["tactile"].shape == (1, 4, 2, 3, 35, 20)
