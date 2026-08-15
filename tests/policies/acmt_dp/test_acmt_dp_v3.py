from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from lerobot.policies.acmt_dp.modeling_components import TemporalTactileGridEncoder
from lerobot.policies.acmt_dp.visual_preprocess import center_crop_480_resize_128


def test_center480_matches_upstream_reference() -> None:
    upstream_path = Path("/cym/TactiGen/ACMT-DP/src/acmt_dp/visual_preprocess.py")
    if not upstream_path.is_file():
        pytest.skip("upstream ACMT-DP source tree is unavailable")
    spec = importlib.util.spec_from_file_location(
        "acmt_upstream_visual_preprocess",
        upstream_path,
    )
    assert spec is not None and spec.loader is not None
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    rgb = torch.arange(2 * 3 * 480 * 640, dtype=torch.uint8).reshape(2, 3, 480, 640)
    depth = torch.randint(0, 5000, (2, 1, 480, 640), dtype=torch.uint16)
    ours_rgb, ours_depth = center_crop_480_resize_128(rgb, depth)
    ref_rgb, ref_depth = upstream.center_crop_480_resize_128(rgb, depth)
    torch.testing.assert_close(ours_rgb, ref_rgb)
    torch.testing.assert_close(ours_depth, ref_depth)


def test_temporal_tactile_encoder_requires_four_frames() -> None:
    encoder = TemporalTactileGridEncoder((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    value = torch.zeros(2, 4, 2, 35, 20, 3)
    assert encoder(value).shape == (2, 160)
    try:
        encoder(torch.zeros(2, 2, 35, 20, 3))
    except ValueError as exc:
        assert "[B,history,2,35,20,3]" in str(exc)
    else:
        raise AssertionError("single-frame tactile input must be rejected")


def test_16_predict_8_execute_queue_replaces_only_future() -> None:
    pytest.importorskip("datasets")
    from lerobot.rollout.inference.acmt_dp import ActionPlan, TimedActionQueue

    queue = TimedActionQueue()
    first = ActionPlan(0, torch.zeros(16, 8), start_time=100.0)
    queue.install_plan(first)
    assert len(queue) == 16
    for _ in range(8):
        assert queue.pop() is not None
    assert len(queue) == 8
    replacement = torch.ones(16, 8)
    removed = queue.replace_future(replacement, start_time=100.0, plan_id=1)
    assert removed == 8
    assert len(queue) == 16
    assert torch.all(queue.snapshot()[0] == 1)
