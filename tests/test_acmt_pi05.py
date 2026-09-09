from __future__ import annotations

import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.acmt_pi05.configuration_acmt_pi05 import ACMTPi05Config
from lerobot.policies.acmt_pi05.modeling_acmt_pi05 import ACMTPi05TactileEncoder


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
