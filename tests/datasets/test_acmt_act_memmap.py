from __future__ import annotations

import json
from unittest.mock import patch

import h5py
import numpy as np
import torch

from lerobot.datasets.acmt_act_memmap import (
    ACMTACTMemmapDataset,
    ACMTActMemmapStore,
    convert_h5_to_memmap,
)
from lerobot.policies.acmt_act.configuration_acmt_act import rgb_key


def _write_episode(path, length: int, offset: int) -> None:
    with h5py.File(path, "w") as handle:
        rgb = np.zeros((length, 480, 640, 3), dtype=np.uint8)
        rgb[..., 0] = np.arange(480, dtype=np.uint8)[None, :, None]
        rgb[..., 1] = (np.arange(640, dtype=np.uint8)[None, None, :] + offset) % 255
        wrist = np.stack([rgb, np.flip(rgb, axis=2)], axis=1)
        handle.create_dataset("observations/rgb/top", data=rgb)
        handle.create_dataset("observations/rgb/side", data=rgb)
        handle.create_dataset("observations/rgb/wrist", data=wrist)
        handle.create_dataset("observations/robot_state/q", data=np.zeros((length, 7), np.float32))
        handle.create_dataset("observations/gripper/gPO", data=np.full(length, 128, np.float32))
        handle.create_dataset("observations/tactile/force", data=np.zeros((length, 2, 35, 20, 3), np.float32))
        handle.create_dataset("actions/gello_q", data=np.arange(length * 7, dtype=np.float32).reshape(length, 7))
        handle.create_dataset("actions/gello_gripper_cmd", data=np.arange(length, dtype=np.float32))


def test_conversion_crop_resume_and_causal_action_padding(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    names = ["demo_00.h5", "demo_01.h5"]
    _write_episode(source / names[0], 2, 0)
    _write_episode(source / names[1], 3, 10)
    split_file = tmp_path / "splits.json"
    split_file.write_text(json.dumps({"train": [names[0]], "val": [], "test": [names[1]]}))
    output = tmp_path / "memmap"
    convert_h5_to_memmap(source, split_file, output, chunk_frames=1, progress=False)

    store = ACMTActMemmapStore(output)
    assert store.rgb.shape == (5, 4, 320, 580, 3)
    # top crop starts at (y=80,x=30); side starts at (y=140,x=60).
    np.testing.assert_array_equal(store.rgb[0, 0, 0, 0], np.array([80, 30, 0], dtype=np.uint8))
    np.testing.assert_array_equal(store.rgb[0, 1, 0, 0], np.array([140, 60, 0], dtype=np.uint8))
    assert store.episode_ends.tolist() == [2, 5]

    train = ACMTACTMemmapDataset(output, split="train")
    # Once converted, the training dataset must not touch the source H5 files.
    with patch("h5py.File", side_effect=AssertionError("training backend opened H5")):
        sample = train[1]
    assert sample[rgb_key("camera.cam1")].shape == (320, 580, 3)
    assert sample["action"].shape == (16, 8)
    assert not sample["action_is_pad"][0]
    assert sample["action_is_pad"][1:].all()
    torch.testing.assert_close(sample["action"][1], sample["action"][0])
