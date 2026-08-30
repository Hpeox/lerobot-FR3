# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

from __future__ import annotations

import json
from pathlib import Path
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
    SUPPORTED_DIFFUSION_INFERENCE_STEPS,
    depth_key,
    rgb_key,
)
from lerobot.policies.acmt_dp.modeling_acmt_dp import ACMTDPPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.policies.acmt_dp.processor_acmt_dp import (
    ACMT_DP_DEFAULT_SOURCE_CAMERA_KEYS,
    ACMTDPNativeV4ProcessorStep,
)
from lerobot.policies.factory import get_policy_class, make_policy_config
from lerobot.scripts.convert_acmt_dp_checkpoint import _make_config, _validate_v4_scratch
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
    assert 100 in SUPPORTED_DIFFUSION_INFERENCE_STEPS
    assert ACMTDPConfig(diffusion_inference_steps=100, device="cpu").diffusion_inference_steps == 100
    with pytest.raises(ValueError, match="must be one of"):
        ACMTDPConfig(diffusion_inference_steps=7, device="cpu")


def test_converter_rejects_unsupported_inference_steps() -> None:
    config = {
        "obs_horizon": 4,
        "pred_horizon": 16,
        "action_execution_horizon": 8,
        "state_dim": 8,
        "action_dim": 8,
        "tactile_dim": 160,
        "feature_dim": 512,
        "unet_kernel_size": 5,
        "diffusion_step_embed_dim": 128,
        "diffusion_inference_steps": 7,
        "control_hz": 30.0,
        "camera_names": ["top", "side", "wrist_left", "wrist_right"],
        "vision_mode": "scratch",
        "vision_weights": None,
    }
    checkpoint = {
        "schema": "acmt_dp.native_dp_v4",
        "stage": "scratch",
        "config": config,
        "statistics": {},
        "model_state_dict": {},
        "ema_state_dict": {},
    }
    with pytest.raises(ValueError, match="diffusion_inference_steps must be one of"):
        _validate_v4_scratch(checkpoint, Path("best.pt"))


def test_converter_can_override_inference_steps() -> None:
    checkpoint = {
        "config": {
            "tactile_source": "none",
            "diffusion_train_steps": 100,
            "diffusion_inference_steps": 8,
            "unet_dims": [256, 512, 1024],
            "unet_kernel_size": 5,
            "diffusion_step_embed_dim": 128,
            "cond_predict_scale": True,
        },
        "statistics": {
            "state_mean": [0.0] * 8,
            "state_std": [1.0] * 8,
            "action_min": [-1.0] * 8,
            "action_max": [1.0] * 8,
            "force_mean": [0.0] * 3,
            "force_std": [1.0] * 3,
        },
    }

    config = _make_config("peg", "none", checkpoint, None, diffusion_inference_steps=100)

    assert config.diffusion_inference_steps == 100


def test_native_processor_maps_runtime_camera_order_to_v4_semantic_order() -> None:
    config = _config()
    step = ACMTDPNativeV4ProcessorStep(camera_keys=config.camera_keys)
    runtime_cameras = ("camera.cam1", "camera.cam2", "camera.cam3", "camera.cam4")
    observation = {
        rgb_key(camera): torch.full((224, 224, 3), value, dtype=torch.uint8)
        for value, camera in enumerate(runtime_cameras, start=1)
    }
    original = {key: value.clone() for key, value in observation.items()}

    processed = step.observation(observation)

    expected_values = {
        "camera.cam1": 4,
        "camera.cam2": 3,
        "camera.cam3": 2,
        "camera.cam4": 1,
    }
    assert step.source_camera_keys == ACMT_DP_DEFAULT_SOURCE_CAMERA_KEYS
    for camera, expected in expected_values.items():
        value = processed[rgb_key(camera)]
        assert value.shape == (1, 3, 224, 224)
        assert value.dtype is torch.uint8
        assert torch.all(value == expected)
    for key, value in original.items():
        assert torch.equal(observation[key], value)


def test_native_processor_validates_source_camera_mapping() -> None:
    with pytest.raises(ValueError, match="source_camera_keys"):
        ACMTDPNativeV4ProcessorStep(
            camera_keys=("camera.cam1", "camera.cam2", "camera.cam3", "camera.cam4"),
            source_camera_keys=("camera.cam1",) * 4,
        )


def test_native_processor_serializes_and_defaults_source_camera_mapping() -> None:
    step = ACMTDPNativeV4ProcessorStep(
        camera_keys=("camera.cam1", "camera.cam2", "camera.cam3", "camera.cam4"),
    )
    saved = step.get_config()
    assert saved["source_camera_keys"] == list(ACMT_DP_DEFAULT_SOURCE_CAMERA_KEYS)

    legacy = dict(saved)
    legacy.pop("source_camera_keys")
    restored = ACMTDPNativeV4ProcessorStep(**legacy)
    assert restored.source_camera_keys == ACMT_DP_DEFAULT_SOURCE_CAMERA_KEYS


def test_legacy_processor_json_loads_without_source_camera_mapping(tmp_path) -> None:
    pipeline = PolicyProcessorPipeline(
        steps=[ACMTDPNativeV4ProcessorStep(camera_keys=_config().camera_keys)],
        name="policy_preprocessor",
    )
    pipeline.save_pretrained(tmp_path, config_filename="policy_preprocessor.json")
    config_path = tmp_path / "policy_preprocessor.json"
    payload = json.loads(config_path.read_text())
    payload["steps"][0]["config"].pop("source_camera_keys")
    config_path.write_text(json.dumps(payload))

    loaded = PolicyProcessorPipeline.from_pretrained(
        tmp_path,
        config_filename="policy_preprocessor.json",
    )

    assert loaded.steps[0].source_camera_keys == ACMT_DP_DEFAULT_SOURCE_CAMERA_KEYS


def test_tactigen_processor_maps_wrist_depth_sources() -> None:
    step = ACMTDPNativeV4ProcessorStep(
        camera_keys=("camera.cam1", "camera.cam2", "camera.cam3", "camera.cam4"),
        tactile_source="tactigen",
    )
    observation = {
        rgb_key(camera): torch.zeros(224, 224, 3, dtype=torch.uint8)
        for camera in ("camera.cam1", "camera.cam2", "camera.cam3", "camera.cam4")
    }
    observation[depth_key("camera.cam1")] = torch.full((128, 128, 1), 1.0)
    observation[depth_key("camera.cam2")] = torch.full((128, 128, 1), 2.0)

    processed = step.observation(observation)

    assert torch.all(processed[depth_key("camera.cam3")] == 2.0)
    assert torch.all(processed[depth_key("camera.cam4")] == 1.0)


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
