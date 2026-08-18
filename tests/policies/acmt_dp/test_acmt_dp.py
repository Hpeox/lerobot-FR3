# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

from __future__ import annotations

from types import MethodType

import pytest
import torch

from lerobot.configs import PreTrainedConfig
from lerobot.policies.acmt_dp.configuration_acmt_dp import (
    DQ,
    FT300,
    GRIPPER_GPO,
    O_T_EE,
    TAU_J,
    XENSE0,
    XENSE1,
    ACMTDPConfig,
    depth_key,
    rgb_key,
)
from lerobot.policies.acmt_dp.modeling_acmt_dp import ACMTDPPolicy
from lerobot.policies.acmt_dp.processor_acmt_dp import ACMTDPNativeV4ProcessorStep
from lerobot.policies.factory import get_policy_class, make_policy_config
from lerobot.utils.constants import OBS_STATE


def _generator_config() -> dict:
    return {
        "t_obs": 4,
        "t_pred": 1,
        "visual_dim": 160,
        "shared_dim": 160,
        "decoder_dim": 192,
        "num_heads": 4,
        "conv_layers": 1,
        "decoder_layout": "drifting_fz_xy_heads_v2",
        "field_order": ["fz", "fx", "fy"],
        "visual_encoder_name": "tiny",
        "view_dropout": 0.0,
    }


def _config(mode: str = "none") -> ACMTDPConfig:
    return ACMTDPConfig(
        tactile_source=mode,
        checkpoint_tactile_source="real" if mode == "tactigen" else mode,
        generator_model_config=_generator_config() if mode == "tactigen" else None,
        device="cpu",
    )


def _batch(mode: str = "none", batch_size: int = 1) -> dict[str, torch.Tensor]:
    batch = {
        OBS_STATE: torch.zeros(batch_size, 8),
        GRIPPER_GPO: torch.zeros(batch_size, 1),
    }
    for camera in ("camera.cam1", "camera.cam2", "camera.cam3", "camera.cam4"):
        batch[rgb_key(camera)] = torch.zeros(batch_size, 3, 224, 224)
    if mode == "real":
        batch[XENSE0] = torch.zeros(batch_size, 35, 20, 3)
        batch[XENSE1] = torch.zeros(batch_size, 3, 35, 20)
    if mode == "tactigen":
        for camera in ("camera.cam3", "camera.cam4"):
            batch[depth_key(camera)] = torch.zeros(batch_size, 1, 128, 128)
        batch[DQ] = torch.zeros(batch_size, 7)
        batch[TAU_J] = torch.zeros(batch_size, 7)
        batch[FT300] = torch.zeros(batch_size, 6)
        batch[O_T_EE] = torch.eye(4).repeat(batch_size, 1, 1)
    return batch


def test_registration_and_v4_config_serialization(tmp_path) -> None:
    config = make_policy_config("acmt_dp")
    assert isinstance(config, ACMTDPConfig)
    assert get_policy_class("acmt_dp") is ACMTDPPolicy
    config.save_pretrained(tmp_path)
    restored = PreTrainedConfig.from_pretrained(tmp_path)
    assert isinstance(restored, ACMTDPConfig)
    assert restored.checkpoint_schema == "acmt_dp.native_dp_v4"
    assert restored.camera_keys == ("camera.cam1", "camera.cam2", "camera.cam3", "camera.cam4")
    assert restored.wrist_camera_keys == ("camera.cam3", "camera.cam4")
    assert restored.visual_preprocess == "resize256_center224_imagenet"
    assert restored.global_cond_dim == 8864


def test_v4_rejects_old_modes_and_abis() -> None:
    with pytest.raises(ValueError, match="generated"):
        ACMTDPConfig(tactile_source="generated", device="cpu")
    with pytest.raises(ValueError, match="center480"):
        ACMTDPConfig(visual_preprocess="center480", device="cpu")
    with pytest.raises(ValueError, match="schema"):
        ACMTDPConfig(checkpoint_schema_version=3, device="cpu")
    with pytest.raises(ValueError, match="scratch"):
        ACMTDPConfig(vision_mode="frozen", device="cpu")


def test_native_processor_preserves_raw_four_camera_rgb() -> None:
    config = _config()
    step = ACMTDPNativeV4ProcessorStep(camera_keys=config.camera_keys)
    observation = {
        rgb_key(camera): torch.zeros(480, 640, 3, dtype=torch.uint8) for camera in config.camera_keys
    }
    processed = step.observation(observation)
    assert processed[rgb_key("camera.cam1")].shape == (1, 3, 480, 640)
    assert processed[rgb_key("camera.cam1")].dtype is torch.uint8


def test_real_tactile_layout_and_v4_inputs() -> None:
    policy = ACMTDPPolicy(_config("real"))
    batch = _batch("real")
    batch[XENSE0][..., 0] = 1
    batch[XENSE1][:, 1] = 2
    current = policy._extract_current(batch)
    assert current["rgb"].shape == (1, 4, 3, 224, 224)
    assert current["state"].shape == (1, 8)
    assert current["tactile"].shape == (1, 2, 35, 20, 3)
    assert torch.all(current["tactile"][0, 0, ..., 0] == 1)
    assert torch.all(current["tactile"][0, 1, ..., 1] == 2)


def test_tactigen_causal_reset_and_action_shape() -> None:
    policy = ACMTDPPolicy(_config("tactigen"))
    tactile_seen: list[torch.Tensor] = []
    actions_seen: list[torch.Tensor] = []

    def fake_plan(self, observation, noise=None):
        del noise
        tactile_seen.append(observation["tactile"].clone())
        return torch.zeros(1, 16, 8)

    def fake_generate(self, previous, action):
        assert previous["gen_lowdim"].shape == (1, 4, 28)
        actions_seen.append(action.clone())
        return torch.ones(1, 2, 35, 20, 3)

    policy._plan = MethodType(fake_plan, policy)
    policy._generate_tactile = MethodType(fake_generate, policy)
    batch = _batch("tactigen")
    first = policy.predict_action_chunk(batch)
    assert first.shape == (1, 16, 8)
    assert torch.count_nonzero(tactile_seen[0]) == 0
    policy.notify_action_executed(first[:, 0])
    policy.predict_action_chunk(batch)
    assert torch.count_nonzero(tactile_seen[1][:, :3]) == 0
    assert torch.count_nonzero(tactile_seen[1][:, 3]) == 2 * 35 * 20 * 3
    torch.testing.assert_close(actions_seen[0], first[:, 0])
    policy.reset()
    policy.predict_action_chunk(batch)
    assert torch.count_nonzero(tactile_seen[-1]) == 0


def test_missing_mode_specific_inputs_are_explicit() -> None:
    real = ACMTDPPolicy(_config("real"))
    with pytest.raises(KeyError, match="xense"):
        real._extract_current(_batch("none"))
    tactigen = ACMTDPPolicy(_config("tactigen"))
    missing_pose = _batch("tactigen")
    del missing_pose[O_T_EE]
    with pytest.raises(KeyError, match="O_T_EE"):
        tactigen._extract_current(missing_pose)


def test_training_interfaces_are_disabled() -> None:
    policy = ACMTDPPolicy(_config())
    with pytest.raises(NotImplementedError, match="inference-only"):
        policy.forward({})
    with pytest.raises(NotImplementedError, match="inference-only"):
        policy.get_optim_params()
