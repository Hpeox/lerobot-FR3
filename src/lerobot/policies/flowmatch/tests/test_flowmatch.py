"""Synthetic native-policy contract tests. Never connects any robot or reads HDF5."""

import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
import torch

from lerobot.configs import PreTrainedConfig
from lerobot.policies.factory import (
    get_policy_class,
    make_policy,
    make_policy_config,
    make_pre_post_processors,
)
from lerobot.policies.flowmatch._core.config import ModelConfig
from lerobot.policies.flowmatch._core.models import SplitPolicy
from lerobot.policies.flowmatch.checkpoint import bundle_id, read_components, read_whole
from lerobot.policies.flowmatch.configuration_flowmatch import CAMERAS, FlowMatchConfig
from lerobot.policies.flowmatch.export import export_policy
from lerobot.policies.flowmatch.mapping import ActionMapping
from lerobot.policies.flowmatch.modeling_flowmatch import FlowMatchPolicy
from lerobot.policies.flowmatch.preprocessing import preprocess_rgb
from lerobot.policies.flowmatch.runtime import ActionWorker, VisionWorker


@pytest.fixture
def saved(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(7)
    cfg = ModelConfig(
        image_size=32,
        patch_size=8,
        vision_width=32,
        vision_depth=2,
        vision_heads=4,
        vision_pretrained=False,
        action_width=32,
        action_depth=2,
        action_heads=4,
        flow_steps=2,
    )
    model = SplitPolicy(cfg, initialize_pretrained=False).eval()
    torch.nn.init.normal_(model.action.output[-1].weight, std=0.1)
    stats = {name: {"mean": [0.0] * 7 + [0.5], "std": [1.0] * 7 + [0.3]} for name in ("state", "action")}
    payload = {
        "format_version": 2,
        "trained": True,
        "step": 100000,
        "config": asdict(cfg),
        "model": model.state_dict(),
        "stats": stats,
        "tasks": ["demo"],
        "source_inventory": {"test": True},
    }
    torch.save(payload, tmp_path / "whole.pt")
    identity = bundle_id(payload["model"])
    for kind in ("vision", "action"):
        meta = {key: value for key, value in payload.items() if key != "model"}
        torch.save(
            {**meta, "component": kind, "bundle_id": identity, "weights": getattr(model, kind).state_dict()},
            tmp_path / (kind + ".pt"),
        )
    return tmp_path, payload


def batch():
    return {
        "observation.state": torch.tensor([[0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, 0.2]]),
        "task_id": torch.tensor([0]),
        **{
            f"observation.images.{name}": torch.full((1, 3, 480, 640), 32 + 48 * i, dtype=torch.uint8)
            for i, name in enumerate(CAMERAS)
        },
    }


def mapping_config():
    return {
        "camera_map": dict(
            zip(CAMERAS, [f"observation.images.camera.cam{i}.rgb" for i in (2, 1, 4, 3)], strict=True)
        ),
        "camera_mapping_source": "SYNTHETIC TEST ONLY",
        "action_mapping": {
            "joint_order": [6, 5, 4, 3, 2, 1, 0],
            "scale": [1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "offset_rad": [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "gpo_at_raw_zero": 255.0,
            "gpo_at_raw_one": 3.0,
            "verification_source": "SYNTHETIC TEST ONLY",
        },
        "default_task_id": 0,
    }


def test_factory_and_unloaded_random_model_rejected():
    assert get_policy_class("flowmatch") is FlowMatchPolicy
    cfg = make_policy_config("flowmatch", device="cpu")
    assert isinstance(cfg, FlowMatchConfig)
    # Tiny instance with no loaded checkpoint cannot accidentally drive a robot.
    cfg.model.update(
        vision_width=32,
        vision_depth=2,
        vision_heads=4,
        action_width=32,
        action_depth=2,
        action_heads=4,
        vision_pretrained=False,
    )
    with pytest.raises(ValueError, match="Load trained"):
        FlowMatchPolicy(cfg).select_action({})


def test_whole_components_and_workers_equivalent(saved):
    folder, _ = saved
    whole = FlowMatchPolicy.from_checkpoint(folder / "whole.pt", device="cpu")
    split = FlowMatchPolicy.from_components(folder / "vision.pt", folder / "action.pt", device="cpu")
    data = batch()
    expected = whole.predict_raw_action_chunk(data, seed=101)
    torch.testing.assert_close(expected, split.predict_raw_action_chunk(data, seed=101))
    vision = VisionWorker(folder / "vision.pt", clock_domain="test")
    expert = ActionWorker(folder / "action.pt", clock_domain="test")
    rgb = {name: data["observation.images." + name] for name in CAMERAS}
    packet = vision.encode(rgb, [0], episode_id="one", frame_id=2, captured_at=10.0)
    actual = expert.generate_raw(
        data["observation.state"], packet, episode_id="one", frame_id=2, now=10.1, seed=101
    )
    torch.testing.assert_close(expected, actual)


def test_native_mapping_required_and_absolute_targets_preserved(saved):
    folder, _ = saved
    policy = FlowMatchPolicy.from_checkpoint(folder / "whole.pt", device="cpu")
    with pytest.raises(ValueError, match="camera mapping"):
        policy.select_action(batch())
    spec = mapping_config()
    policy = FlowMatchPolicy.from_checkpoint(folder / "whole.pt", device="cpu", **spec)
    original = batch()
    data = {"observation.state": original["observation.state"], "task_id": torch.tensor([0])}
    for name in CAMERAS:
        data[spec["camera_map"][name]] = original["observation.images." + name]
    raw = policy.predict_raw_action_chunk(data, seed=101)
    expected = ActionMapping.from_dict(spec["action_mapping"]).convert(raw)
    torch.testing.assert_close(policy.predict_action_chunk(data, seed=101), expected)
    assert ((expected[..., 7] >= 3 / 255) & (expected[..., 7] <= 1)).all()
    for index in range(4):
        torch.testing.assert_close(policy.select_action(data, seed=101), expected[:, index])
    assert not policy._queue
    policy.select_action(data, seed=101)
    policy.reset()
    assert not policy._queue


def test_export_factory_processors_and_strict_roundtrip(saved, tmp_path):
    folder, _ = saved
    policy = FlowMatchPolicy.from_checkpoint(folder / "whole.pt", device="cpu")
    output = tmp_path / "export"
    export_policy(policy, output)
    loaded = FlowMatchPolicy.from_pretrained(output, local_files_only=True)
    expected = policy.predict_raw_action_chunk(batch(), seed=101)
    torch.testing.assert_close(expected, loaded.predict_raw_action_chunk(batch(), seed=101))
    pre, post = make_pre_post_processors(loaded.config, pretrained_path=str(output))
    processor_input = batch()
    processor_input.pop("task_id")
    processor_input["task"] = ["demo"]  # Native pipelines carry task text, not arbitrary top-level keys.
    prepared = pre(processor_input)
    torch.testing.assert_close(expected, loaded.predict_raw_action_chunk(prepared, seed=101))
    torch.testing.assert_close(post(expected[:, 0]), expected[:, 0])
    assert (
        json.loads((output / "flowmatch_export.json").read_text())["deployment_mapping_configured"] is False
    )
    with pytest.raises(FileExistsError):
        export_policy(policy, output)
    with pytest.raises(ValueError, match="strict"):
        FlowMatchPolicy.from_pretrained(output, strict=False)


def test_wrong_step_and_mixed_tensors_rejected(saved):
    folder, payload = saved
    torch.save({**payload, "step": 15000}, folder / "wrong.pt")
    with pytest.raises(ValueError, match="step100000"):
        read_whole(folder / "wrong.pt")
    action = torch.load(folder / "action.pt", weights_only=True)
    action["weights"]["input.weight"][0, 0] += 1
    torch.save(action, folder / "changed.pt")
    with pytest.raises(ValueError, match="declared bundle"):
        read_components(folder / "vision.pt", folder / "changed.pt")


def test_workers_reject_stale_wrong_episode_clock_and_metadata(saved):
    folder, _ = saved
    vision, expert = VisionWorker(folder / "vision.pt"), ActionWorker(folder / "action.pt")
    data = batch()
    packet = vision.encode(
        {name: data["observation.images." + name] for name in CAMERAS},
        [0],
        episode_id="one",
        frame_id=2,
        captured_at=10.0,
    )
    for changed, expected in [
        (replace(packet, bundle_id="other"), "bundle"),
        (replace(packet, metadata_id="other"), "metadata"),
        (replace(packet, clock_domain="other_host"), "Clock"),
        (replace(packet, frame_id=1), "episode/frame"),
        (replace(packet, episode_id="previous"), "episode/frame"),
        (replace(packet, captured_at=9.0), "Stale"),
        (replace(packet, captured_at=11.0), "Stale"),
    ]:
        with pytest.raises(ValueError, match=expected):
            expert.generate_raw(data["observation.state"], changed, episode_id="one", frame_id=2, now=10.1)


def test_rgb_float_quantization_and_no_double_normalization():
    image = torch.randint(0, 256, (1, 3, 480, 640), dtype=torch.uint8)
    torch.testing.assert_close(preprocess_rgb(image, 32), preprocess_rgb(image.float() / 255, 32))
    with pytest.raises(ValueError, match="before SigLIP"):
        preprocess_rgb(torch.full_like(image, -1, dtype=torch.float32), 32)


def test_unknown_task_and_nonfinite_state_rejected(saved):
    folder, _ = saved
    policy = FlowMatchPolicy.from_checkpoint(folder / "whole.pt", device="cpu")
    data = batch()
    data["task_id"] = torch.tensor([1])
    with pytest.raises(ValueError, match="Task IDs"):
        policy.predict_raw_action_chunk(data)
    data["task_id"] = torch.tensor([0])
    data["observation.state"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite measured"):
        policy.predict_raw_action_chunk(data)
    with pytest.raises(NotImplementedError):
        policy(data)


def test_native_factory_uses_fr3_feature_names_without_dataset_renormalization(saved, tmp_path):
    folder, _ = saved
    policy = FlowMatchPolicy.from_checkpoint(folder / "whole.pt", device="cpu", **mapping_config())
    output = tmp_path / "native_factory"
    export_policy(policy, output)
    cfg = PreTrainedConfig.from_pretrained(output, local_files_only=True)
    assert isinstance(cfg, FlowMatchConfig)
    cfg.pretrained_path = output
    features = {
        "observation.state": {"dtype": "float32", "shape": (8,), "names": cfg.action_feature_names},
        "action": {"dtype": "float32", "shape": (8,), "names": cfg.action_feature_names},
        **{
            key: {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"]}
            for key in cfg.camera_map.values()
        },
        "observation.images.camera.cam1.depth": {
            "dtype": "video",
            "shape": (480, 640, 1),
            "names": ["height", "width", "channels"],
        },
    }
    loaded = make_policy(cfg, ds_meta=SimpleNamespace(features=features, stats={"deliberately_unused": True}))
    original = batch()
    data = {"observation.state": original["observation.state"]}
    for name in CAMERAS:
        data[cfg.image_key(name)] = original["observation.images." + name]
    # Explicit default task ID supports native pipelines without arbitrary task_id fields.
    torch.testing.assert_close(policy.select_action(data, seed=101), loaded.select_action(data, seed=101))


def test_mapping_has_no_residual_addition_and_clips_only_raw_gripper():
    mapping = ActionMapping.from_dict(mapping_config()["action_mapping"])
    raw = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.2]])
    torch.testing.assert_close(
        mapping.convert(raw), torch.tensor([[6.1, -5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 3 / 255]])
    )
