from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from lerobot.datasets.acmt_act_memmap import ARRAY_SPECS, ACMTACTMemmapDataset, MEMMAP_VERSION
from lerobot.policies.acmt_act.configuration_acmt_act import XENSE0, XENSE1, rgb_key
from lerobot.policies.acmt_act.processor_acmt_act import ACMTACTObservationProcessorStep
from lerobot.policies.acmt_actv2.configuration_acmt_actv2 import (
    CAMERA_KEYS,
    CAMERA_NAMES,
    ACMTACTV2Config,
    FR3_SOURCE_CAMERA_KEYS,
)
from lerobot.policies.acmt_dp.gripper_mapping import ACMTDPGripperGPOProcessorStep
from lerobot.policies.acmt_actv2.modeling_acmt_actv2 import ACMTACTV2Policy
from lerobot.policies.factory import get_policy_class, make_policy_config, make_pre_post_processors
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import OBS_STATE


def _config(mode: str = "none") -> ACMTACTV2Config:
    return ACMTACTV2Config(
        device="cpu",
        pretrained_backbone_weights=None,
        tactile_source=mode,
        generator_checkpoint="/tmp/acmt-act-test-generator.pt" if mode == "substitution" else None,
    )


def test_factory_and_three_camera_protocol() -> None:
    config = make_policy_config("acmt_actv2", device="cpu", pretrained_backbone_weights=None)
    assert isinstance(config, ACMTACTV2Config)
    assert config.type == "acmt_actv2"
    assert config.checkpoint_schema == "acmt_actv2.v1"
    assert config.camera_keys == CAMERA_KEYS
    assert config.camera_names == CAMERA_NAMES
    assert config.source_camera_keys == CAMERA_KEYS
    assert get_policy_class("acmt_actv2") is ACMTACTV2Policy


def test_three_independent_backbones_and_output_shape() -> None:
    policy = ACMTACTV2Policy(_config())
    assert len(policy.model.backbone) == 3
    assert len(policy.model.encoder_img_feat_input_proj) == 3
    assert len({id(module) for module in policy.model.backbone}) == 3
    assert len({id(module) for module in policy.model.encoder_img_feat_input_proj}) == 3
    batch = {
        OBS_STATE: torch.zeros(1, 8),
        XENSE0: torch.zeros(1, 3, 35, 20),
        XENSE1: torch.zeros(1, 3, 35, 20),
        "action": torch.zeros(1, 16, 8),
        "action_is_pad": torch.zeros(1, 16, dtype=torch.bool),
    }
    for key in policy.config.image_features:
        batch[key] = torch.zeros(1, 3, 32, 32)
    loss, _ = policy(batch)
    assert torch.isfinite(loss)
    with torch.no_grad():
        actions = policy.predict_action_chunk(
            {key: value for key, value in batch.items() if key not in {"action", "action_is_pad"}}
        )
    assert actions.shape == (1, 16, 8)


def test_processor_has_no_top_and_preserves_three_camera_crops() -> None:
    config = _config()
    processor = ACMTACTObservationProcessorStep(
        camera_keys=config.camera_keys,
        camera_names=config.camera_names,
        source_camera_keys=config.source_camera_keys,
        crop_params=config.crop_params,
    )
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[..., 0] = np.arange(480, dtype=np.uint8)[:, None]
    image[..., 1] = np.arange(640, dtype=np.uint8)[None, :]
    raw = {OBS_STATE: np.zeros(8, dtype=np.float32)}
    for key in config.image_features:
        raw[key] = image.copy()
    result = processor.observation(raw)
    assert set(config.image_features) == {
        "observation.images.camera.cam2.rgb",
        "observation.images.camera.cam3.rgb",
        "observation.images.camera.cam4.rgb",
    }
    assert "observation.images.camera.cam1.rgb" not in result
    side = result["observation.images.camera.cam2.rgb"]
    mean = torch.tensor(config.image_mean).view(3, 1, 1)
    std = torch.tensor(config.image_std).view(3, 1, 1)
    actual = side[0, :, 0, 0] * std[:, 0, 0] + mean[:, 0, 0]
    torch.testing.assert_close(actual, torch.tensor([140, 60, 0]) / 255)
    assert side.shape == (1, 3, 320, 580)


def test_fr3_deployment_mapping_reads_raw_camera_ids_without_cyclic_overwrite() -> None:
    config = ACMTACTV2Config(
        device="cpu",
        pretrained_backbone_weights=None,
        source_camera_keys=FR3_SOURCE_CAMERA_KEYS,
    )
    processor = ACMTACTObservationProcessorStep(
        camera_keys=config.camera_keys,
        camera_names=config.camera_names,
        source_camera_keys=config.source_camera_keys,
        crop_params=config.crop_params,
    )
    raw = {OBS_STATE: np.zeros(8, dtype=np.float32)}
    for camera_index in range(1, 5):
        raw[rgb_key(f"camera.cam{camera_index}")] = np.full(
            (480, 640, 3), camera_index * 16, dtype=np.uint8
        )

    result = processor.observation(raw)

    expected = {
        "camera.cam2": 48,  # side <- runtime cam3
        "camera.cam3": 16,  # wrist_left <- runtime cam1
        "camera.cam4": 32,  # wrist_right <- runtime cam2
    }
    mean = torch.tensor(config.image_mean).view(3, 1, 1)
    std = torch.tensor(config.image_std).view(3, 1, 1)
    for target_camera, source_value in expected.items():
        processed = result[rgb_key(target_camera)]
        restored = processed[0, :, 0, 0] * std[:, 0, 0] + mean[:, 0, 0]
        torch.testing.assert_close(
            restored,
            torch.full((3,), source_value / 255.0),
            atol=1e-6,
            rtol=1e-6,
        )
        assert processed.shape == (1, 3, 320, 580)
    assert processor.get_config()["source_camera_keys"] == list(FR3_SOURCE_CAMERA_KEYS)


def test_acmt_actv2_accepts_only_identity_or_fixed_fr3_source_mapping() -> None:
    assert (
        ACMTACTV2Config(device="cpu", pretrained_backbone_weights=None).source_camera_keys
        == CAMERA_KEYS
    )
    assert (
        ACMTACTV2Config(
            device="cpu",
            pretrained_backbone_weights=None,
            source_camera_keys=FR3_SOURCE_CAMERA_KEYS,
        ).source_camera_keys
        == FR3_SOURCE_CAMERA_KEYS
    )
    with pytest.raises(ValueError, match="training identity mapping"):
        ACMTACTV2Config(
            device="cpu",
            pretrained_backbone_weights=None,
            source_camera_keys=("camera.cam1", "camera.cam2", "camera.cam3"),
        )


def test_fr3_source_mapping_round_trips_through_processor_pipeline(tmp_path) -> None:
    config = ACMTACTV2Config(
        device="cpu",
        pretrained_backbone_weights=None,
        source_camera_keys=FR3_SOURCE_CAMERA_KEYS,
    )
    preprocessor, _ = make_pre_post_processors(config)
    preprocessor.save_pretrained(tmp_path)

    restored = PolicyProcessorPipeline.from_pretrained(
        tmp_path,
        config_filename="policy_preprocessor.json",
    )
    step = next(item for item in restored.steps if isinstance(item, ACMTACTObservationProcessorStep))
    assert step.source_camera_keys == FR3_SOURCE_CAMERA_KEYS


def test_acmt_actv2_gripper_adapter_preserves_joints_and_maps_gpo_endpoints() -> None:
    action = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.0],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0],
        ]
    )
    processed = ACMTDPGripperGPOProcessorStep().action(action)

    torch.testing.assert_close(processed[:, :7], action[:, :7])
    torch.testing.assert_close(processed[:, 7], torch.tensor([1.0, 3.0 / 255.0]))


def test_memmap_camera_view_excludes_top(tmp_path) -> None:
    n = 2
    np.save(tmp_path / "rgb.npy", np.zeros((n, 4, 320, 580, 3), np.uint8))
    np.save(tmp_path / "state.npy", np.zeros((n, 8), np.float32))
    np.save(tmp_path / "tactile.npy", np.zeros((n, 2, 35, 20, 3), np.float32))
    np.save(tmp_path / "action.npy", np.zeros((n, 8), np.float32))
    np.save(tmp_path / "sample_valid.npy", np.ones(n, bool))
    np.save(tmp_path / "episode_ends.npy", np.array([n], np.int64))
    (tmp_path / "episode_names.json").write_text(json.dumps(["demo.h5"]), encoding="utf-8")
    (tmp_path / "splits.json").write_text(
        json.dumps({"splits": {"train": ["demo.h5"], "val": [], "test": []}}), encoding="utf-8"
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {"memmap_version": MEMMAP_VERSION, "complete": True, "episode_count": 1, "arrays": ARRAY_SPECS}
        ),
        encoding="utf-8",
    )
    dataset = ACMTACTMemmapDataset(tmp_path, split="train", camera_indices=(1, 2, 3))
    item = dataset[0]
    assert dataset.meta.camera_keys == [
        "observation.images.camera.cam2.rgb",
        "observation.images.camera.cam3.rgb",
        "observation.images.camera.cam4.rgb",
    ]
    assert "observation.images.camera.cam1.rgb" not in item
