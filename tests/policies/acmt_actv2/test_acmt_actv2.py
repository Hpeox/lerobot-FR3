from __future__ import annotations

import json
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn

from lerobot.datasets.acmt_act_memmap import ARRAY_SPECS, ACMTACTMemmapDataset, MEMMAP_VERSION
from lerobot.policies.acmt_act.configuration_acmt_act import XENSE0, XENSE1, rgb_key
from lerobot.policies.acmt_act.processor_acmt_act import ACMTACTObservationProcessorStep
from lerobot.policies.acmt_actv2.configuration_acmt_actv2 import ACMTACTV2Config
from lerobot.policies.acmt_actv2.modeling_acmt_actv2 import ACMTACTV2Policy
from lerobot.policies.factory import get_policy_class, make_policy_config
from lerobot.utils.constants import OBS_STATE


class FakeDINO(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return image.new_zeros((image.shape[0], 768, 384))

    def train(self, mode: bool = True):
        super().train(False)
        return self


def _config(mode: str = "none", **kwargs) -> ACMTACTV2Config:
    return ACMTACTV2Config(
        device="cpu",
        tactile_source=mode,
        generator_checkpoint="/tmp/acmt-act-test-generator.pt" if mode == "substitution" else None,
        **kwargs,
    )


def test_native_factory_contract() -> None:
    config = make_policy_config("acmt_actv2", device="cpu")
    assert isinstance(config, ACMTACTV2Config)
    assert config.type == "acmt_actv2"
    assert config.checkpoint_schema == "acmt_actv2.dinov2_spatial.v1"
    assert config.checkpoint_schema_version == 3
    assert config.training_contract == "native_absolute_physical_gripper_v1"
    assert (config.n_obs_steps, config.chunk_size, config.pred_horizon) == (1, 100, 100)
    assert (config.n_action_steps, config.action_execution_horizon) == (1, 1)
    assert config.temporal_ensemble_coeff == 0.01
    assert get_policy_class("acmt_actv2") is ACMTACTV2Policy
    assert not any("goal" in key for key in config.input_features)
    assert not any("depth" in key for key in config.input_features)


def test_processor_dinov2_pad_and_center_crop() -> None:
    config = _config()
    processor = ACMTACTObservationProcessorStep(
        camera_keys=config.camera_keys,
        camera_names=config.camera_names,
        crop_params=config.crop_params,
        dinov2_spatial=True,
    )
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[..., 0] = np.arange(480, dtype=np.uint8)[:, None]
    image[..., 1] = np.arange(640, dtype=np.uint8)[None, :]
    raw = {OBS_STATE: np.zeros(8, dtype=np.float32)}
    for key in config.image_features:
        raw[key] = image.copy()
    result = processor.observation(raw)
    top = result[rgb_key("camera.cam1")]
    assert top.shape == (1, 3, 336, 448)
    mean = torch.tensor(config.image_mean).view(3, 1, 1)
    std = torch.tensor(config.image_std).view(3, 1, 1)
    restored = top[0] * std + mean
    # The first eight rows are mean-color padding, and row 8 is crop y=80,
    # x=30 followed by horizontal x=66:514.
    torch.testing.assert_close(restored[:, 0, 0], mean[:, 0, 0])
    torch.testing.assert_close(restored[:, 8, 0], torch.tensor([80, 30 + 66, 0]) / 255)
    # The synthetic green ramp is uint8, so the source value wraps at 255.
    torch.testing.assert_close(restored[:, 8, 447], torch.tensor([80, (30 + 66 + 447) % 256, 0]) / 255)


def test_policy_has_one_shared_dino_and_native_action_head() -> None:
    config = _config(
        dim_model=32,
        dinov2_projection_dim=32,
        n_heads=4,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_vae_encoder_layers=1,
        dropout=0.0,
    )
    with patch("lerobot.policies.acmt_actv2.modeling_acmt_actv2.DINOv2SpatialBackbone", FakeDINO):
        policy = ACMTACTV2Policy(config)
    assert policy.model.backbone is policy.model.backbone
    assert len([module for module in policy.model.modules() if module is policy.model.backbone]) == 1
    assert all(not parameter.requires_grad for parameter in policy.model.backbone.parameters())
    images = [torch.zeros(2, 3, 336, 448) for _ in range(4)]
    with torch.no_grad():
        tokens, positions = policy.model._dino_tokens(images)
    assert tokens.shape == (3072, 2, 32)
    assert positions.shape == (3072, 1, 32)
    assert policy.model.action_head.out_features == 8
    assert not hasattr(policy.model, "goal_head")
    assert not hasattr(policy.model, "gripper_head")

    # Keep the transformer test small while checking the complete native ACT
    # action/loss interface.  The production DINO path above still verifies
    # the required 3072-token shape.
    policy.model._dino_tokens = lambda _: (
        torch.zeros(1, 2, 32),
        torch.zeros(1, 1, 32),
    )
    batch = {
        OBS_STATE: torch.zeros(2, 8),
        XENSE0: torch.zeros(2, 3, 35, 20),
        XENSE1: torch.zeros(2, 3, 35, 20),
        "action": torch.zeros(2, 100, 8),
        "action_is_pad": torch.zeros(2, 100, dtype=torch.bool),
    }
    for index, key in enumerate(config.image_features):
        batch[key] = torch.zeros(2, 3, 336, 448)
    policy.train()
    loss, metrics = policy(batch)
    assert torch.isfinite(loss)
    assert "kld_loss" in metrics
    policy.eval()
    action = policy.predict_action_chunk({key: value for key, value in batch.items() if key not in {"action", "action_is_pad"}})
    assert action.shape == (2, 100, 8)


def test_none_and_real_parameter_structure_is_identical() -> None:
    kwargs = dict(
        dim_model=32,
        dinov2_projection_dim=32,
        n_heads=4,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_vae_encoder_layers=1,
        dropout=0.0,
    )
    with patch("lerobot.policies.acmt_actv2.modeling_acmt_actv2.DINOv2SpatialBackbone", FakeDINO):
        torch.manual_seed(42)
        none = ACMTACTV2Policy(_config("none", **kwargs))
        torch.manual_seed(42)
        real = ACMTACTV2Policy(_config("real", **kwargs))
    assert list(none.state_dict()) == list(real.state_dict())
    assert all(torch.equal(none.state_dict()[key], real.state_dict()[key]) for key in none.state_dict())


def test_temporal_ensemble_queries_every_tick() -> None:
    with patch("lerobot.policies.acmt_actv2.modeling_acmt_actv2.DINOv2SpatialBackbone", FakeDINO):
        policy = ACMTACTV2Policy(_config(dim_model=32, dinov2_projection_dim=32, n_heads=4, dim_feedforward=64, n_encoder_layers=1, n_vae_encoder_layers=1))
    calls = []

    def fake_predict(_batch):
        calls.append(1)
        return torch.arange(800, dtype=torch.float32).reshape(1, 100, 8)

    policy.predict_action_chunk = fake_predict
    dummy = {OBS_STATE: torch.zeros(1, 8)}
    assert policy.select_action(dummy).shape == (1, 8)
    assert policy.select_action(dummy).shape == (1, 8)
    assert len(calls) == 2
    assert not hasattr(policy, "_action_queue")


def test_memmap_uses_configured_100_step_chunk(tmp_path) -> None:
    n = 3
    np.save(tmp_path / "rgb.npy", np.zeros((n, 4, 320, 580, 3), np.uint8))
    np.save(tmp_path / "state.npy", np.zeros((n, 8), np.float32))
    np.save(tmp_path / "tactile.npy", np.zeros((n, 2, 35, 20, 3), np.float32))
    action = np.zeros((n, 8), np.float32)
    action[:, 7] = 1.0
    np.save(tmp_path / "action.npy", action)
    np.save(tmp_path / "sample_valid.npy", np.ones(n, bool))
    np.save(tmp_path / "episode_ends.npy", np.array([n], np.int64))
    (tmp_path / "episode_names.json").write_text(json.dumps(["demo.h5"]), encoding="utf-8")
    (tmp_path / "splits.json").write_text(json.dumps({"splits": {"train": ["demo.h5"], "val": [], "test": []}}), encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps({"memmap_version": MEMMAP_VERSION, "complete": True, "episode_count": 1, "arrays": ARRAY_SPECS}), encoding="utf-8")
    dataset = ACMTACTMemmapDataset(tmp_path, split="train", chunk_size=100, native_absolute_actions=True)
    item = dataset[1]
    assert item["action"].shape == (100, 8)
    assert item["action_is_pad"].shape == (100,)
    assert item["action_is_pad"][:2].sum() == 0
    assert item["action_is_pad"][2:].all()
    assert item["action"][0, 7] == 0.0


@pytest.mark.parametrize("field", ["checkpoint_schema", "checkpoint_schema_version", "training_contract"])
def test_config_rejects_old_checkpoint_contract(field) -> None:
    values = {
        "checkpoint_schema": "acmt_actv2.dformerv2_spatial.v1",
        "checkpoint_schema_version": 2,
        "training_contract": "residual_joint_physical_gripper_visual_goal_v1",
    }
    with pytest.raises(ValueError):
        _config(**{field: values[field]})
