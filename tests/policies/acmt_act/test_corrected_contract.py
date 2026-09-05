from __future__ import annotations

import json

import h5py
import numpy as np
import torch

from lerobot.datasets.acmt_act_memmap import (
    ACMTACTMemmapDataset,
    build_acmt_act_policy_stats,
    build_acmt_act_targets,
    convert_h5_to_memmap,
)
from lerobot.policies.acmt_act.configuration_acmt_act import GOAL_VALID, GOAL_XYZ
from lerobot.utils.constants import OBS_STATE


def _episode(path, length: int = 24) -> None:
    with h5py.File(path, "w") as handle:
        rgb = np.zeros((length, 480, 640, 3), dtype=np.uint8)
        wrist = np.stack([rgb, rgb], axis=1)
        handle.create_dataset("observations/rgb/top", data=rgb)
        handle.create_dataset("observations/rgb/side", data=rgb)
        handle.create_dataset("observations/rgb/wrist", data=wrist)
        handle.create_dataset("observations/robot_state/q", data=np.zeros((length, 7), np.float32))
        gpo = np.array([3] * 10 + [20] * (length - 10), dtype=np.uint8)
        handle.create_dataset("observations/gripper/gPO", data=gpo)
        handle.create_dataset("observations/tactile/force", data=np.zeros((length, 2, 35, 20, 3), np.float32))
        handle.create_dataset("observations/robot_state/O_T_EE", data=np.tile(np.eye(4, dtype=np.float32), (length, 1, 1)))
        handle["observations/robot_state/O_T_EE"][:, 0, 3] = 0.55
        handle["observations/robot_state/O_T_EE"][:, 1, 3] = 0.22
        handle["observations/robot_state/O_T_EE"][:, 2, 3] = 0.31
        handle.create_dataset("actions/gello_q", data=np.ones((length, 7), np.float32) * 0.1)
        handle.create_dataset("actions/gello_gripper_cmd", data=np.ones(length, np.float32))


def test_target_sidecar_and_physical_action_contract(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _episode(source / "demo.h5")
    split = tmp_path / "splits.json"
    split.write_text(json.dumps({"train": ["demo.h5"], "val": [], "test": []}), encoding="utf-8")
    root = tmp_path / "memmap"
    convert_h5_to_memmap(source, split, root, chunk_frames=8, progress=False)
    build_acmt_act_targets(source, root, split_file=split)
    stats_path = build_acmt_act_policy_stats(root)
    assert stats_path.is_file()
    sample = ACMTACTMemmapDataset(root, split="train")[0]
    # Source command is 1=open; policy action is 0=open.
    assert sample["action"][0, 7].item() == 0.0
    assert bool(sample[GOAL_VALID])
    torch.testing.assert_close(sample[GOAL_XYZ], torch.tensor([0.55, 0.22, 0.31]))
    torch.testing.assert_close(sample[OBS_STATE][7], torch.tensor(3 / 255.0))
