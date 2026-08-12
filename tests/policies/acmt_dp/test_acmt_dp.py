# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
from lerobot.policies.acmt_dp.processor_acmt_dp import ACMTDPWristROIProcessorStep
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
    checkpoint_source = "real" if mode == "tactigen" else mode
    return ACMTDPConfig(
        tactile_source=mode,
        checkpoint_tactile_source=checkpoint_source,
        visual_encoder_name="tiny",
        unet_dims=(32, 64),
        diffusion_inference_steps=2,
        generator_model_config=_generator_config() if mode == "tactigen" else None,
        device="cpu",
    )


def _batch(mode: str = "none", batch_size: int = 1) -> dict[str, torch.Tensor]:
    batch = {
        OBS_STATE: torch.zeros(batch_size, 8),
        DQ: torch.zeros(batch_size, 7),
        TAU_J: torch.zeros(batch_size, 7),
        FT300: torch.zeros(batch_size, 6),
        GRIPPER_GPO: torch.zeros(batch_size, 1),
    }
    for camera in ("camera.cam1", "camera.cam2"):
        batch[rgb_key(camera)] = torch.zeros(batch_size, 3, 128, 128)
        batch[depth_key(camera)] = torch.zeros(batch_size, 1, 128, 128)
    if mode == "real":
        batch[XENSE0] = torch.zeros(batch_size, 35, 20, 3)
        batch[XENSE1] = torch.zeros(batch_size, 3, 35, 20)
    if mode == "tactigen":
        batch[O_T_EE] = torch.eye(4).repeat(batch_size, 1, 1)
    return batch


def test_registration_and_config_serialization(tmp_path) -> None:
    config = make_policy_config("acmt_dp")
    assert isinstance(config, ACMTDPConfig)
    assert get_policy_class("acmt_dp") is ACMTDPPolicy

    config.save_pretrained(tmp_path)
    restored = PreTrainedConfig.from_pretrained(tmp_path)
    assert isinstance(restored, ACMTDPConfig)
    assert restored.wrist_camera_keys == ("camera.cam1", "camera.cam2")
    assert restored.wrist_roi == (176, 304, 256, 384)

    tactigen = _config("tactigen")
    tactigen.save_pretrained(tmp_path / "tactigen")
    restored_tactigen = PreTrainedConfig.from_pretrained(tmp_path / "tactigen")
    assert restored_tactigen.tactile_source == "tactigen"
    assert restored_tactigen.checkpoint_tactile_source == "real"


def test_mode_specific_config_validation() -> None:
    with pytest.raises(ValueError, match="mode-specific"):
        ACMTDPConfig(tactile_source="real", checkpoint_tactile_source="none", device="cpu")
    with pytest.raises(ValueError, match="task-specific"):
        ACMTDPConfig(task_variant="gear", checkpoint_task_variant="peg", device="cpu")
    with pytest.raises(ValueError, match="generator_model_config"):
        ACMTDPConfig(tactile_source="tactigen", device="cpu")
    with pytest.raises(ValueError, match="deprecated"):
        ACMTDPConfig(tactile_source="generated", device="cpu")
    config = ACMTDPConfig(device="cpu")
    config.task_variant = "gear"
    with pytest.raises(ValueError, match="checkpoint/runtime task mismatch"):
        config.validate_features()


def test_roi_processor_maps_rgb_and_depth() -> None:
    step = ACMTDPWristROIProcessorStep(camera_keys=("camera.cam1", "camera.cam2"), roi=(176, 304, 256, 384))
    rows = torch.arange(480, dtype=torch.uint8)[None, :, None, None]
    rgb = rows.expand(1, 480, 640, 3).clone()
    depth = torch.arange(480, dtype=torch.float32)[None, :, None, None].expand(1, 480, 640, 1)
    observation = {}
    for camera in step.camera_keys:
        observation[rgb_key(camera)] = rgb
        observation[depth_key(camera)] = depth

    processed = step.observation(observation)
    for camera in step.camera_keys:
        assert processed[rgb_key(camera)].shape == (1, 3, 128, 128)
        assert processed[depth_key(camera)].shape == (1, 1, 128, 128)
        assert processed[rgb_key(camera)].dtype == torch.float32
        assert processed[depth_key(camera)][0, 0, 0, 0] == 176
        assert processed[rgb_key(camera)][0, 0, 0, 0] == pytest.approx(176 / 255)
    assert step.get_config() == {
        "camera_keys": ["camera.cam1", "camera.cam2"],
        "roi": [176, 304, 256, 384],
    }


def test_lowdim_order_and_real_tactile_layout() -> None:
    policy = ACMTDPPolicy(_config("real"))
    batch = _batch("real")
    batch[OBS_STATE][0] = torch.arange(8)
    batch[DQ][0] = 10 + torch.arange(7)
    batch[TAU_J][0] = 20 + torch.arange(7)
    batch[FT300][0] = 30 + torch.arange(6)
    batch[GRIPPER_GPO][0] = 127.5
    batch[XENSE0][..., 0] = 1
    batch[XENSE1][:, 1] = 2

    current = policy._extract_current(batch)
    expected = torch.cat(
        [
            torch.arange(7),
            10 + torch.arange(7),
            20 + torch.arange(7),
            30 + torch.arange(6),
            torch.tensor([0.5]),
        ]
    )
    torch.testing.assert_close(current["lowdim"][0], expected)
    assert current["tactile"].shape == (1, 2, 35, 20, 3)
    assert torch.all(current["tactile"][0, 0, ..., 0] == 1)
    assert torch.all(current["tactile"][0, 1, ..., 1] == 2)


def test_tactigen_causal_state_replans_and_reset() -> None:
    policy = ACMTDPPolicy(_config("tactigen"))
    tactile_seen: list[torch.Tensor] = []
    generator_inputs: list[torch.Tensor] = []
    plan_count = 0

    def fake_plan(self, observation, noise=None):
        nonlocal plan_count
        del noise
        plan_count += 1
        tactile_seen.append(observation["tactile"].clone())
        return torch.full((1, 16, 8), float(plan_count))

    def fake_generate(self, previous, action_chunk):
        assert previous["lowdim"].shape == (1, 4, 28)
        generator_inputs.append(action_chunk[:, 0].clone())
        return torch.full((1, 2, 35, 20, 3), 7.0)

    policy._plan = MethodType(fake_plan, policy)
    policy._generate_tactile = MethodType(fake_generate, policy)
    batch = _batch("tactigen")

    first = policy.predict_action_chunk(batch)
    policy.predict_action_chunk(batch)
    assert plan_count == 2
    assert first.shape == (1, 16, 8)
    assert torch.count_nonzero(tactile_seen[0]) == 0
    assert torch.all(tactile_seen[1] == 7)
    torch.testing.assert_close(generator_inputs[0], first[:, 0])
    torch.testing.assert_close(policy.select_action(batch), torch.full((1, 8), 3.0))
    assert plan_count == 3

    policy.reset()
    policy.predict_action_chunk(batch)
    assert torch.count_nonzero(tactile_seen[-1]) == 0
    assert policy.causal_state_dict() == {"history_length": 4, "has_previous_plan": True}


def test_mode_specific_missing_and_shape_errors() -> None:
    real = ACMTDPPolicy(_config("real"))
    missing_real = _batch("none")
    with pytest.raises(KeyError, match="xense"):
        real._extract_current(missing_real)
    invalid_real = _batch("real")
    invalid_real[XENSE0] = torch.zeros(1, 35, 20)
    with pytest.raises(ValueError, match="four dimensions"):
        real._extract_current(invalid_real)

    tactigen = ACMTDPPolicy(_config("tactigen"))
    missing_pose = _batch("none")
    with pytest.raises(KeyError, match="O_T_EE"):
        tactigen._extract_current(missing_pose)
    invalid_pose = _batch("tactigen")
    invalid_pose[O_T_EE] = torch.zeros(1, 16)
    with pytest.raises(ValueError, match=r"\[B,4,4\]"):
        tactigen._extract_current(invalid_pose)


def test_training_interfaces_are_disabled() -> None:
    policy = ACMTDPPolicy(_config())
    with pytest.raises(NotImplementedError, match="inference-only"):
        policy.forward({})
    with pytest.raises(NotImplementedError, match="inference-only"):
        policy.get_optim_params()
