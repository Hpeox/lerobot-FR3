from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from lerobot.configs import PreTrainedConfig
from lerobot.policies.acmt_dp.configuration_acmt_dp_v3 import (
    DQ,
    FT300,
    GRIPPER_GPO,
    TAU_J,
    ACMTDPV3Config,
    depth_key,
    rgb_key,
)
from lerobot.policies.acmt_dp.modeling_acmt_dp_v3 import ACMTDPV3Policy
from lerobot.policies.acmt_dp.processor_acmt_dp_v3 import (
    ACMTDPV3Center480ProcessorStep,
    make_acmt_dp_v3_pre_post_processors,
)
from lerobot.policies.factory import get_policy_class, make_policy_config
from lerobot.utils.constants import OBS_STATE


def _config(*, visual_encoder_name: str = "tiny") -> ACMTDPV3Config:
    return ACMTDPV3Config(
        tactile_source="none",
        checkpoint_tactile_source="none",
        task_variant="peg",
        checkpoint_task_variant="peg",
        visual_encoder_name=visual_encoder_name,
        device="cpu",
    )


def _batch() -> dict[str, torch.Tensor]:
    batch: dict[str, torch.Tensor] = {
        OBS_STATE: torch.zeros(1, 8),
        DQ: torch.zeros(1, 7),
        TAU_J: torch.zeros(1, 7),
        FT300: torch.zeros(1, 6),
        GRIPPER_GPO: torch.zeros(1, 1, dtype=torch.uint8),
    }
    for camera in ("camera.cam1", "camera.cam2"):
        batch[rgb_key(camera)] = torch.zeros(1, 3, 480, 640, dtype=torch.uint8)
        batch[depth_key(camera)] = torch.zeros(1, 1, 480, 640, dtype=torch.uint16)
    return batch


def test_v3_registration_and_schema_serialization(tmp_path) -> None:
    config = make_policy_config("acmt_dp_v3")
    assert isinstance(config, ACMTDPV3Config)
    assert get_policy_class("acmt_dp_v3") is ACMTDPV3Policy
    assert config.type == "acmt_dp_v3"
    assert config.checkpoint_schema_version == 3
    assert config.checkpoint_schema == "v3_temporal_center480"
    assert config.wrist_camera_keys == ("camera.cam1", "camera.cam2")
    assert config.diffusion_train_steps == 100
    assert config.diffusion_inference_steps == 100
    config.save_pretrained(tmp_path)
    restored = PreTrainedConfig.from_pretrained(tmp_path)
    assert isinstance(restored, ACMTDPV3Config)
    assert restored.type == "acmt_dp_v3"


def test_v3_rejects_other_schema() -> None:
    with pytest.raises(ValueError, match="schema v3"):
        ACMTDPV3Config(checkpoint_schema_version=4, device="cpu")
    with pytest.raises(ValueError, match="v3_temporal_center480"):
        ACMTDPV3Config(checkpoint_schema="acmt_dp.native_dp_v4", device="cpu")


def test_v3_processor_uses_only_wrist_cameras_and_center480() -> None:
    step = ACMTDPV3Center480ProcessorStep(camera_keys=("camera.cam1", "camera.cam2"))
    observation = {
        rgb_key("camera.cam1"): torch.full((480, 640, 3), 11, dtype=torch.uint8),
        rgb_key("camera.cam2"): torch.full((480, 640, 3), 22, dtype=torch.uint8),
        depth_key("camera.cam1"): torch.full((480, 640, 1), 111, dtype=torch.uint16),
        depth_key("camera.cam2"): torch.full((480, 640, 1), 222, dtype=torch.uint16),
        "unrelated": torch.tensor([7]),
    }
    processed = step.observation(observation)
    for camera, value in (("camera.cam1", 11), ("camera.cam2", 22)):
        assert processed[rgb_key(camera)].shape == (1, 3, 128, 128)
        assert processed[rgb_key(camera)].dtype.is_floating_point
        torch.testing.assert_close(processed[rgb_key(camera)], torch.full((1, 3, 128, 128), value / 255.0))
        assert processed[depth_key(camera)].shape == (1, 1, 128, 128)
    assert torch.equal(processed["unrelated"], observation["unrelated"])


def test_v3_observe_fills_four_frame_history_and_zero_tactile() -> None:
    policy = ACMTDPV3Policy(_config())
    window = policy.observe(_batch())
    assert window["rgb"].shape == (1, 4, 2, 3, 128, 128)
    assert window["depth"].shape == (1, 4, 2, 1, 128, 128)
    assert window["lowdim"].shape == (1, 4, 28)
    assert window["tactile"].shape == (1, 4, 2, 35, 20, 3)
    assert torch.count_nonzero(window["tactile"]) == 0
    assert torch.equal(window["lowdim"][:, 0], window["lowdim"][:, -1])


def test_v3_postprocessor_maps_policy_gripper_to_wire_range() -> None:
    _, postprocessor = make_acmt_dp_v3_pre_post_processors(_config())
    action = torch.zeros(2, 8)
    action[0, 7] = 0.0
    action[1, 7] = 1.0
    processed = postprocessor.process_action(action)
    assert processed[0, 7].item() == pytest.approx(1.0)
    assert processed[1, 7].item() == pytest.approx(3.0 / 255.0)


def test_v3_loader_rejects_v4_artifact_before_model_construction(tmp_path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "type": "acmt_dp_v3",
                "checkpoint_schema": "acmt_dp.native_dp_v4",
                "checkpoint_schema_version": 4,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not v3"):
        ACMTDPV3Policy.from_pretrained(tmp_path)


def test_v3_factory_uses_rolling_engine() -> None:
    from lerobot.rollout.inference import ACMTDPInferenceEngine, SyncInferenceConfig, create_inference_engine

    policy = SimpleNamespace(
        name="acmt_dp_v3",
        robot_type="fr3",
        config=SimpleNamespace(
            control_hz=30.0,
            action_execution_horizon=8,
            tactile_history=4,
            pred_horizon=16,
            action_dim=8,
            checkpoint_schema_version=3,
            tactile_source="none",
        ),
    )
    engine = create_inference_engine(
        SyncInferenceConfig(),
        policy=policy,
        preprocessor=SimpleNamespace(),
        postprocessor=SimpleNamespace(),
        robot_wrapper=SimpleNamespace(robot_type="fr3"),
        hw_features={},
        dataset_features={},
        ordered_action_keys=[],
        task="peg",
        fps=30.0,
        device="cpu",
    )
    try:
        assert isinstance(engine, ACMTDPInferenceEngine)
    finally:
        engine.stop()
