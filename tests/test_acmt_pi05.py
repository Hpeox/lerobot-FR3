from __future__ import annotations

import torch
import pytest

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.acmt_pi05.configuration_acmt_pi05 import ACMTPi05Config
from lerobot.policies.acmt_pi05.processor_acmt_pi05 import (
    ACMTPi05GripperGPOProcessorStep,
    ACMTPi05ObservationProcessorStep,
)
from lerobot.policies.acmt_pi05.modeling_acmt_pi05 import ACMTPi05Pytorch, ACMTPi05TactileEncoder


def test_acmt_pi05_feature_contract_excludes_raw_force_fields() -> None:
    config = ACMTPi05Config(
        input_features={
            "observation.images.camera.cam1.rgb": PolicyFeature(FeatureType.VISUAL, (3, 320, 580)),
            "observation.state": PolicyFeature(FeatureType.STATE, (8,)),
            "observation.xense.sensor0.force_field": PolicyFeature(FeatureType.STATE, (35, 20, 3)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (8,))},
    )
    config.validate_features()
    assert "observation.xense.sensor0.force_field" not in config.input_features
    assert config.output_features["action"].shape == (8,)


def test_acmt_pi05_tactile_encoder_contract_and_gradients() -> None:
    encoder = ACMTPi05TactileEncoder(
        force_mean=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        force_std=[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
    )
    tactile = torch.randn(2, 2, 3, 35, 20)
    output = encoder(tactile)
    assert output.shape == (2, 160)
    output.square().mean().backward()
    assert any(parameter.grad is not None for parameter in encoder.parameters())


def test_acmt_pi05_runtime_camera_mapping_and_training_crops() -> None:
    step = ACMTPi05ObservationProcessorStep(
        source_camera_keys=("camera.cam4", "camera.cam3", "camera.cam1", "camera.cam2")
    )
    # Keep sentinel values in [0, 1] so the step does not perform uint8 scaling.
    obs = {}
    for camera in range(1, 5):
        image = torch.empty(480, 640, 3)
        y = torch.arange(480, dtype=torch.float32).view(480, 1, 1)
        x = torch.arange(640, dtype=torch.float32).view(1, 640, 1)
        image[:] = camera / 10.0 + y / 10000.0 + x / 1000000.0
        obs[f"observation.images.camera.cam{camera}.rgb"] = image

    result = step.observation(obs)
    expected_sources = (4, 3, 1, 2)
    expected_origins = ((80, 30), (140, 60), (80, 30), (80, 30))
    for target, (source, (y, x)) in enumerate(zip(expected_sources, expected_origins, strict=True), 1):
        value = result[f"observation.images.camera.cam{target}.rgb"]
        assert value.shape == (1, 3, 320, 580)
        assert value[0, 0, 0, 0].item() == pytest.approx(source / 10.0 + y / 10000.0 + x / 1000000.0)


def test_acmt_pi05_gripper_mapping_matches_training_semantics() -> None:
    step = ACMTPi05GripperGPOProcessorStep()
    action = torch.zeros(2, 8)
    action[1, 7] = 1.0
    result = step.action(action)
    assert result[:, 7].tolist() == pytest.approx([3.0 / 255.0, 1.0])
    assert torch.equal(result[:, :7], action[:, :7])


def test_acmt_pi05_fp16_projection_dtype_keeps_tactile_encoder_fp32(monkeypatch) -> None:
    """Direct FP16 inference must not rely on an outer autocast context."""

    import lerobot.policies.acmt_pi05.modeling_acmt_pi05 as modeling

    class DummyVLM(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    monkeypatch.setattr(modeling, "PaliGemmaWithExpertModel", DummyVLM)
    config = ACMTPi05Config(dtype="float16", device="cpu")
    model = ACMTPi05Pytorch(config)

    assert model.action_in_proj.weight.dtype == torch.float16
    assert model.action_out_proj.weight.dtype == torch.float16
    assert model.tactile_token_proj.weight.dtype == torch.float16
    assert model.time_mlp_in.weight.dtype == torch.float16
    assert model.time_mlp_out.weight.dtype == torch.float16
    assert model.tactile_encoder.spatial[0].weight.dtype == torch.float32
