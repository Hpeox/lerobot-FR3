from __future__ import annotations

import hashlib

import torch
import pytest
from torch import nn

from lerobot.policies.acmt_act.configuration_acmt_act import (
    ACMTACTConfig,
    DQ,
    GRIPPER_GPO,
    O_T_EE,
    depth_key,
    rgb_key,
)
from lerobot.policies.acmt_act.modeling_acmt_act import ACMTACTPolicy
from lerobot.policies.acmt_act.processor_acmt_act import (
    ACMTACTObservationProcessorStep,
    GEN_DEPTH,
    GEN_LOWDIM,
    GEN_POSE,
    GEN_RGB,
    make_acmt_act_pre_post_processors,
)
from lerobot.policies.acmt_act.tactigen_v5_runtime import (
    TactiGenV5Runtime,
    gello_opening_from_fr3_pos,
)
from lerobot.utils.constants import OBS_STATE


def test_v5_gripper_width_uses_accepted_fr3_normalized_gpo() -> None:
    values = torch.tensor([3.0 / 255.0, 1.0])
    torch.testing.assert_close(gello_opening_from_fr3_pos(values), torch.tensor([1.0, 0.0]))


def test_v5_observation_keeps_raw_physical_wrists_and_does_not_require_ft300() -> None:
    config = ACMTACTConfig(
        device="cpu",
        pretrained_backbone_weights=None,
        tactile_source="substitution",
        generator_backend="tactigen_v5",
        generator_checkpoint="/tmp/v5-best.pt",
        generator_bootstrap_checkpoint="/tmp/v5-bootstrap.pt",
        generator_checkpoint_sha256="0" * 64,
        source_camera_keys=("camera.cam2", "camera.cam1", "camera.cam4", "camera.cam3"),
        generator_source_camera_keys=("camera.cam1", "camera.cam2"),
    )
    preprocessor, _ = make_acmt_act_pre_post_processors(config)
    step = next(item for item in preprocessor.steps if isinstance(item, ACMTACTObservationProcessorStep))
    raw = {
        OBS_STATE: torch.zeros(1, 8),
        DQ: torch.zeros(1, 7),
        O_T_EE: torch.eye(4).unsqueeze(0),
        GRIPPER_GPO: torch.full((1, 1), 3.0),
    }
    for index in range(1, 5):
        camera = f"camera.cam{index}"
        raw[rgb_key(camera)] = torch.full((1, 3, 480, 640), index / 10)
        raw[depth_key(camera)] = torch.full((1, 1, 480, 640), float(index))
    result = step.observation(raw)
    torch.testing.assert_close(result[GEN_RGB][0, :, 0, 0, 0], torch.tensor([0.1, 0.2]))
    torch.testing.assert_close(result[GEN_DEPTH][0, :, 0, 0, 0], torch.tensor([1.0, 2.0]))
    assert result[GEN_RGB].shape == (1, 2, 3, 480, 640)
    assert result[GEN_LOWDIM].shape == (1, 15)
    mean = torch.tensor(config.image_mean).view(1, 3, 1, 1)
    std = torch.tensor(config.image_std).view(1, 3, 1, 1)
    policy_cameras = [
        ((result[rgb_key(f"camera.cam{index}")] * std + mean)[0, 0, 0, 0]).item()
        for index in range(1, 5)
    ]
    assert policy_cameras == pytest.approx([0.2, 0.1, 0.4, 0.3])


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"layout": "unsupported"}, "Unsupported TactiGen-V5 checkpoint layout"),
        ({"layout": "physics_contact_t1_fullgrid_v2_ablation", "ablation_variant": "learned_correction"}, "wo_learned_correction"),
        ({"t_obs": 3}, "input/output protocol"),
        ({"uses_ft300_input": True}, "input/output protocol"),
        ({"uses_learned_correction": True}, "Learned correction"),
    ],
)
def test_v5_runtime_rejects_wrong_checkpoint_contract(tmp_path, overrides, message) -> None:
    config = {
        "layout": "tactigen_v5_physics_contact_t1",
        "t_obs": 4,
        "t_pred": 1,
        "contact_grid": [35, 20],
        "force_grid": [35, 20],
        "force_order": ["fx", "fy", "fz"],
        "uses_ft300_input": False,
        "uses_learned_correction": False,
    }
    config.update(overrides)
    checkpoint = tmp_path / "best.pt"
    bootstrap = tmp_path / "bootstrap.pt"
    torch.save({"model_config": config}, checkpoint)
    bootstrap.write_bytes(b"bootstrap")
    with pytest.raises(ValueError, match=message):
        TactiGenV5Runtime(str(checkpoint), str(bootstrap), "cpu")


def test_v5_runtime_rejects_wrong_checkpoint_digest(tmp_path) -> None:
    checkpoint = tmp_path / "best.pt"
    bootstrap = tmp_path / "bootstrap.pt"
    checkpoint.write_bytes(b"checkpoint")
    bootstrap.write_bytes(b"bootstrap")
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() != "0" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        TactiGenV5Runtime(str(checkpoint), str(bootstrap), "cpu", "0" * 64)


def test_v5_gear_filters_only_official_residual_keys(tmp_path, monkeypatch) -> None:
    from tactigen.physics_contact import model as v5_model

    class EmptyModel(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()

    monkeypatch.setattr(v5_model, "TactiGenV5Model", EmptyModel)
    checkpoint = tmp_path / "best.pt"
    bootstrap = tmp_path / "bootstrap.pt"
    bootstrap.write_bytes(b"bootstrap")
    config = {
        "layout": "physics_contact_t1_fullgrid_v2_ablation",
        "ablation_variant": "wo_learned_correction",
        "t_obs": 4,
        "t_pred": 1,
        "contact_grid": [35, 20],
        "force_grid": [35, 20],
        "force_order": ["fx", "fy", "fz"],
        "uses_ft300_input": False,
    }
    torch.save(
        {
            "model_config": config,
            "model_state_dict": {"force_residual_head.weight": torch.ones(1), "force_residual_scale": torch.ones(1)},
        },
        checkpoint,
    )
    TactiGenV5Runtime(str(checkpoint), str(bootstrap), "cpu")
    torch.save(
        {
            "model_config": config,
            "model_state_dict": {"other_head.weight": torch.ones(1)},
        },
        checkpoint,
    )
    with pytest.raises(RuntimeError, match="Unexpected key"):
        TactiGenV5Runtime(str(checkpoint), str(bootstrap), "cpu")


def test_v5_rejects_nonphysical_wrist_mapping() -> None:
    with pytest.raises(ValueError, match="physical wrist cameras"):
        ACMTACTConfig(
            device="cpu",
            pretrained_backbone_weights=None,
            tactile_source="substitution",
            generator_backend="tactigen_v5",
            generator_checkpoint="/tmp/v5-best.pt",
            generator_bootstrap_checkpoint="/tmp/v5-bootstrap.pt",
            generator_checkpoint_sha256="0" * 64,
            generator_source_camera_keys=("camera.cam3", "camera.cam4"),
        )


def test_v5_adapter_preserves_wrist_roi_depth_pose_and_action_history() -> None:
    class RecordingModel:
        def inference(self, batch):
            self.batch = batch
            return {"force": torch.zeros(1, 1, 2, 35, 20, 3)}

    runtime = object.__new__(TactiGenV5Runtime)
    runtime.model = RecordingModel()
    rgb = torch.zeros(1, 4, 2, 3, 480, 640)
    rgb[:, :, 0, :, 176:304, 256:384] = 0.25
    rgb[:, :, 1, :, 176:304, 256:384] = 0.75
    depth = torch.zeros(1, 4, 2, 1, 480, 640)
    depth[:, :, 0] = 1000
    depth[:, :, 1] = 2000
    lowdim = torch.zeros(1, 4, 15)
    lowdim[:, :, 7:14] = 0.5
    pose = torch.eye(4).reshape(1, 1, 4, 4).expand(1, 4, 4, 4).clone()
    pose[:, :, 0, 3] = 0.4
    history = torch.full((1, 4, 8), 0.2)
    action = torch.ones(1, 8)
    force = runtime.predict_next(
        {"rgb": rgb, "depth": depth, "lowdim": lowdim}, pose, action, history
    )
    assert force.shape == (1, 2, 35, 20, 3)
    observed = runtime.model.batch
    assert observed["realsense.cam1_color"].shape == (1, 4, 3, 128, 128)
    torch.testing.assert_close(observed["realsense.cam1_color"].mean(), torch.tensor(0.25))
    torch.testing.assert_close(observed["realsense.cam2_color"].mean(), torch.tensor(0.75))
    torch.testing.assert_close(observed["realsense.cam1_depth"].mean(), torch.tensor(1000.0))
    torch.testing.assert_close(observed["realsense.cam2_depth"].mean(), torch.tensor(2000.0))
    torch.testing.assert_close(observed["robot.O_T_EE"][0, 0], torch.tensor([0.4, 0, 0, 0, 0, 0, 1]))
    torch.testing.assert_close(observed["gello.history_q"], history[:, :, :7])
    torch.testing.assert_close(observed["gello.future_q"], action[:, None, :7])


def test_v5_command_history_advances_only_after_successful_generation_and_resets() -> None:
    class RecordingRuntime:
        def __init__(self):
            self.calls = []
            self.fail = False

        def reset(self):
            pass

        def predict_next(self, observation, pose, action, command_history):
            self.calls.append((action.clone(), command_history.clone()))
            if self.fail:
                raise RuntimeError("generator failed")
            return torch.zeros(1, 2, 35, 20, 3)

    config = ACMTACTConfig(
        device="cpu",
        pretrained_backbone_weights=None,
        tactile_source="substitution",
        generator_backend="tactigen_v5",
        generator_checkpoint="/tmp/v5-best.pt",
        generator_bootstrap_checkpoint="/tmp/v5-bootstrap.pt",
        generator_checkpoint_sha256="0" * 64,
        generator_source_camera_keys=("camera.cam1", "camera.cam2"),
    )
    policy = object.__new__(ACMTACTPolicy)
    nn.Module.__init__(policy)
    policy.config = config
    fake = RecordingRuntime()
    policy._generator_runtime = fake
    policy.reset()
    lowdim = torch.zeros(1, 15)
    lowdim[:, :7] = 0.5
    lowdim[:, 14] = 3.0 / 255.0
    batch = {
        OBS_STATE: torch.zeros(1, 8),
        GEN_RGB: torch.zeros(1, 2, 3, 480, 640),
        GEN_DEPTH: torch.zeros(1, 2, 1, 480, 640),
        GEN_LOWDIM: lowdim,
        GEN_POSE: torch.eye(4).reshape(1, 4, 4),
        **{key: torch.zeros(1, 3, 32, 32) for key in config.image_features},
    }
    policy.observe(batch)
    accepted = torch.tensor([[1.0] * 7 + [3.0 / 255.0]])
    policy.notify_action_executed(accepted)
    torch.testing.assert_close(fake.calls[0][1][0, :, :7], torch.full((4, 7), 0.5))
    torch.testing.assert_close(fake.calls[0][1][0, :, 7], torch.ones(4))
    torch.testing.assert_close(fake.calls[0][0][0, 7], torch.tensor(1.0))
    assert len(policy._gen_command_history) == 4
    torch.testing.assert_close(policy._gen_command_history[-1][0, :7], torch.ones(7))
    fake.fail = True
    with pytest.raises(RuntimeError, match="generator failed"):
        policy.notify_action_executed(torch.tensor([[2.0] * 7 + [1.0]]))
    torch.testing.assert_close(policy._gen_command_history[-1][0, :7], torch.ones(7))
    policy.reset()
    assert not policy._gen_command_history
    batch[GEN_LOWDIM][:, :7] = 2.0
    policy.observe(batch)
    torch.testing.assert_close(policy._gen_command_history[-1][0, :7], torch.full((7,), 2.0))
