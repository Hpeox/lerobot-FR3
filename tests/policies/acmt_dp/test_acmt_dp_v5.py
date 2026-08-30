from __future__ import annotations

import json
from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.configs import PreTrainedConfig
from lerobot.policies.acmt_dp.configuration_acmt_dp_v5 import (
    DQ,
    FT300,
    GRIPPER_GPO,
    O_T_EE,
    TAU_J,
    XENSE0,
    XENSE1,
    ACMTDPV5Config,
    depth_key,
    rgb_key,
)
from lerobot.policies.acmt_dp.modeling_acmt_dp_v5 import ACMTDPV5Policy
from lerobot.policies.acmt_dp.modeling_native_v5 import (
    FrameTactileEncoder,
    NativeV5VisionEncoder,
)
from lerobot.policies.acmt_dp.processor_acmt_dp_v5 import ACMTDPV5ProcessorStep
from lerobot.policies.acmt_dp.processor_acmt_dp import ACMT_DP_DEFAULT_SOURCE_CAMERA_KEYS
from lerobot.policies.acmt_dp.processor_acmt_dp_v5 import make_acmt_dp_v5_pre_post_processors
from lerobot.policies.factory import get_policy_class, make_policy_config
from lerobot.utils.constants import OBS_STATE

CAMERAS = ("camera.cam1", "camera.cam2", "camera.cam3", "camera.cam4")


def _config(mode: str = "none") -> ACMTDPV5Config:
    return ACMTDPV5Config(
        tactile_source=mode,
        checkpoint_tactile_source="real" if mode == "tactigen" else mode,
        generator_model_config={"test": True} if mode == "tactigen" else None,
        device="cpu",
    )


def _batch(mode: str = "none", *, raw_rgb: bool = False) -> dict[str, torch.Tensor]:
    height, width = (480, 640) if raw_rgb or mode == "tactigen" else (240, 320)
    batch: dict[str, torch.Tensor] = {
        OBS_STATE: torch.zeros(1, 8),
        GRIPPER_GPO: torch.zeros(1, 1),
    }
    for camera in CAMERAS:
        batch[rgb_key(camera)] = torch.zeros(1, 3, height, width, dtype=torch.uint8)
    if mode == "real":
        batch[XENSE0] = torch.zeros(1, 35, 20, 3)
        batch[XENSE1] = torch.zeros(1, 3, 35, 20)
    if mode == "tactigen":
        for camera in CAMERAS[2:]:
            batch[depth_key(camera)] = torch.zeros(1, 1, 480, 640, dtype=torch.uint16)
        batch[DQ] = torch.zeros(1, 7)
        batch[TAU_J] = torch.zeros(1, 7)
        batch[FT300] = torch.zeros(1, 6)
        batch[O_T_EE] = torch.eye(4).unsqueeze(0)
    return batch


def _bare_policy(mode: str, generator: object | None = None) -> ACMTDPV5Policy:
    policy = object.__new__(ACMTDPV5Policy)
    nn.Module.__init__(policy)
    policy.config = _config(mode)
    policy.model_horizon = ((policy.config.internal_horizon + 7) // 8) * 8
    policy.tactile_generator = generator
    policy.reset()
    return policy


def test_v5_registration_and_serialization(tmp_path) -> None:
    config = make_policy_config("acmt_dp_v5")
    assert isinstance(config, ACMTDPV5Config)
    assert get_policy_class("acmt_dp_v5").name == "acmt_dp_v5"
    assert config.global_cond_dim == 1696
    config.save_pretrained(tmp_path)
    restored = PreTrainedConfig.from_pretrained(tmp_path)
    assert isinstance(restored, ACMTDPV5Config)
    assert restored.checkpoint_schema == "acmt_dp.native_dp_v5_hybrid"
    assert restored.camera_keys == CAMERAS
    assert restored.random_crop is True


def test_v5_rejects_legacy_schema_modes_and_networks() -> None:
    with pytest.raises(ValueError, match="generated"):
        _config("generated")
    with pytest.raises(ValueError, match="schema"):
        ACMTDPV5Config(checkpoint_schema_version=4, device="cpu")
    with pytest.raises(ValueError, match="scratch"):
        ACMTDPV5Config(vision_mode="frozen", device="cpu")
    with pytest.raises(ValueError, match="GroupNorm"):
        ACMTDPV5Config(use_group_norm=False, device="cpu")
    with pytest.raises(ValueError, match="down_dims"):
        ACMTDPV5Config(unet_dims=(32, 64), device="cpu")


def test_v5_fixes_the_native_diffusion_step_budget() -> None:
    assert _config().diffusion_train_steps == 100
    assert _config().diffusion_inference_steps == 8
    with pytest.raises(ValueError, match="diffusion_inference_steps=8"):
        ACMTDPV5Config(diffusion_inference_steps=100, device="cpu")
    with pytest.raises(ValueError, match="diffusion_inference_steps=8"):
        ACMTDPV5Config(diffusion_inference_steps=7, device="cpu")



def test_v5_loader_rejects_legacy_config_before_model_construction(tmp_path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"checkpoint_schema": "acmt_dp.native_dp_v4", "checkpoint_schema_version": 4}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="v3/v4"):
        ACMTDPV5Policy.from_pretrained(tmp_path)


def test_v5_visual_preprocess_matches_resize_crop_and_range() -> None:
    value = torch.zeros(1, 4, 3, 480, 640, dtype=torch.uint8)
    value[:, :, 0] = 255
    processed = NativeV5VisionEncoder.preprocess(value)
    assert processed.shape == (1, 4, 3, 216, 288)
    torch.testing.assert_close(processed[:, :, 0].mean(), torch.tensor(1.0))

    dark = torch.ones(1, 4, 3, 240, 320, dtype=torch.uint8)
    assert float(NativeV5VisionEncoder.preprocess(dark).max()) < 0.0


def test_v5_visual_and_tactile_component_shapes() -> None:
    vision = NativeV5VisionEncoder()
    assert not any(isinstance(module, nn.modules.batchnorm._BatchNorm) for module in vision.modules())
    assert all(camera.spatial.num_kp == 32 for camera in vision.camera_encoders)
    tactile = FrameTactileEncoder((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    output = tactile(torch.zeros(2, 4, 2, 35, 20, 3))
    assert output.shape == (2, 4, 160)


def test_v5_processor_maps_runtime_camera_order_to_semantic_slots() -> None:
    step = ACMTDPV5ProcessorStep(camera_keys=CAMERAS)
    observation = {
        rgb_key(camera): torch.full((480, 640, 3), value, dtype=torch.uint8)
        for value, camera in enumerate(CAMERAS, start=1)
    }
    original = {key: value.clone() for key, value in observation.items()}
    processed = step.observation(observation)
    expected_values = {
        "camera.cam1": 4,
        "camera.cam2": 3,
        "camera.cam3": 1,
        "camera.cam4": 2,
    }
    assert step.source_camera_keys == ACMT_DP_DEFAULT_SOURCE_CAMERA_KEYS
    for camera, expected in expected_values.items():
        value = processed[rgb_key(camera)]
        assert value.shape == (1, 3, 480, 640)
        assert value.dtype is torch.uint8
        assert torch.all(value == expected)
    for key, value in original.items():
        assert torch.equal(observation[key], value)


def test_v5_processor_validates_and_serializes_source_camera_order() -> None:
    with pytest.raises(ValueError, match="source_camera_keys"):
        ACMTDPV5ProcessorStep(camera_keys=CAMERAS, source_camera_keys=(CAMERAS[0],) * 4)
    step = ACMTDPV5ProcessorStep(camera_keys=CAMERAS)
    saved = step.get_config()
    assert saved["source_camera_keys"] == list(ACMT_DP_DEFAULT_SOURCE_CAMERA_KEYS)
    legacy = dict(saved)
    legacy.pop("source_camera_keys")
    restored = ACMTDPV5ProcessorStep(**legacy)
    assert restored.source_camera_keys == ACMT_DP_DEFAULT_SOURCE_CAMERA_KEYS


def test_v5_postprocessor_maps_gripper_to_wire_gpo() -> None:
    _, postprocessor = make_acmt_dp_v5_pre_post_processors(_config("none"))
    processed = postprocessor.process_action(torch.tensor([[0.0] * 8]))
    assert processed[0, 7].item() == 1.0


def test_v5_tactigen_depth_uses_the_same_camera_permutation() -> None:
    step = ACMTDPV5ProcessorStep(camera_keys=CAMERAS, tactile_source="tactigen")
    observation = {
        rgb_key(camera): torch.zeros(480, 640, 3, dtype=torch.uint8)
        for camera in CAMERAS
    }
    for value, camera in enumerate(CAMERAS, start=1):
        observation[depth_key(camera)] = torch.full((1, 480, 640), value, dtype=torch.uint16)
    processed = step.observation(observation)
    assert torch.all(processed[depth_key("camera.cam3")] == 1)
    assert torch.all(processed[depth_key("camera.cam4")] == 2)


def test_v5_none_and_real_keep_four_frame_tactile_history() -> None:
    none = _bare_policy("none")
    none_window = none.observe(_batch("none"))
    assert none_window["tactile"].shape == (1, 4, 2, 35, 20, 3)
    assert torch.count_nonzero(none_window["tactile"]) == 0

    real = _bare_policy("real")
    batch = _batch("real")
    batch[XENSE0].fill_(1.0)
    batch[XENSE1].fill_(2.0)
    real_window = real.observe(batch)
    assert real_window["tactile"].shape == (1, 4, 2, 35, 20, 3)
    assert torch.all(real_window["tactile"][:, :, 0, ..., 0] == 1)
    assert torch.all(real_window["tactile"][:, :, 1, ..., 0] == 2)


class _FakeGenerator:
    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]] = []
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def predict_next(self, previous, pose, action):
        self.calls.append((previous, pose.clone(), action.clone()))
        return torch.ones(action.shape[0], 2, 35, 20, 3)


def test_v5_tactigen_consumes_only_successful_action_and_resets() -> None:
    generator = _FakeGenerator()
    policy = _bare_policy("tactigen", generator)
    tactile_seen: list[torch.Tensor] = []

    def fake_plan(self, observation, noise=None, inference_steps=None):
        del noise, inference_steps
        tactile_seen.append(observation["tactile"].clone())
        return torch.zeros(1, 16, 8)

    policy._plan = MethodType(fake_plan, policy)
    batch = _batch("tactigen")
    first = policy.predict_action_chunk(batch)
    assert first.shape == (1, 16, 8)
    assert torch.count_nonzero(tactile_seen[0]) == 0

    executed = torch.full((1, 8), 3.0)
    policy.notify_action_executed(executed)
    policy.predict_action_chunk(batch)
    assert torch.count_nonzero(tactile_seen[1][:, :3]) == 0
    assert torch.count_nonzero(tactile_seen[1][:, 3]) == 2 * 35 * 20 * 3
    torch.testing.assert_close(generator.calls[0][2], executed)

    policy._fixed_initial_noise = torch.ones(1, 19, 8)
    policy.reset()
    assert policy._fixed_initial_noise is None
    assert generator.reset_count == 2
    policy.predict_action_chunk(batch)
    assert torch.count_nonzero(tactile_seen[-1]) == 0


def _stubbed_plan_policy(unnormalize) -> ACMTDPV5Policy:
    policy = _bare_policy("none")
    policy.encode_observation = lambda _observation: torch.zeros(1, policy.config.global_cond_dim)
    policy.noise_scheduler = SimpleNamespace(
        timesteps=(),
        set_timesteps=lambda _steps, device=None: None,
    )
    policy.normalizer = {"action": SimpleNamespace(unnormalize=unnormalize)}
    return policy


def test_v5_seed42_noise_is_reused_within_and_across_episodes() -> None:
    policy = _stubbed_plan_policy(lambda value: value)
    first = policy._plan({})
    second = policy._plan({})
    torch.testing.assert_close(first, second)
    policy.reset()
    third = policy._plan({})
    torch.testing.assert_close(first, third)


def test_v5_plan_clips_every_action_dimension_to_checkpoint_bounds(caplog) -> None:
    policy = _stubbed_plan_policy(lambda value: torch.full_like(value, 2.0))
    with caplog.at_level("WARNING", logger="lerobot.policies.acmt_dp.modeling_acmt_dp_v5"):
        action = policy._plan({})
    assert torch.all(action == 1.0)
    assert "action clipped to training bounds" in caplog.text
    assert "dimensions=[0, 1, 2, 3, 4, 5, 6, 7]" in caplog.text


def test_v5_action_public_and_execution_shapes() -> None:
    policy = _bare_policy("none")
    chunk = torch.zeros(1, 16, 8)
    assert policy.action_execution_slice(chunk).shape == (1, 8, 8)
    with pytest.raises(ValueError, match=r"\[16,8\]"):
        policy.action_execution_slice(torch.zeros(1, 19, 8))
