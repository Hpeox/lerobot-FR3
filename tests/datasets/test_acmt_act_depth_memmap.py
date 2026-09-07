from __future__ import annotations

import json

import h5py
import numpy as np

from lerobot.datasets.acmt_act_depth_memmap import (
    ACMTActDepthMemmapStore,
    convert_h5_to_depth_memmap,
)


def test_depth_sidecar_crop_and_resume_contract(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    frames = 3
    for episode in range(2):
        path = source / f"demo_{episode:02d}.h5"
        with h5py.File(path, "w") as handle:
            top = np.zeros((frames, 480, 640), dtype=np.uint16) + episode
            side = np.zeros_like(top) + 10 + episode
            wrist = np.zeros((frames, 2, 480, 640), dtype=np.uint16) + 20 + episode
            top[:, 80, 30] = np.arange(frames, dtype=np.uint16) + 100
            handle.create_dataset("observations/depth/top", data=top)
            handle.create_dataset("observations/depth/side", data=side)
            handle.create_dataset("observations/depth/wrist", data=wrist)

    rgb = tmp_path / "rgb"
    rgb.mkdir()
    (rgb / "manifest.json").write_text(
        json.dumps(
            {
                "complete": True,
                "memmap_version": "acmt_act_memmap_v1",
                "camera_order": ["top", "side", "wrist_left", "wrist_right"],
                "crop_params": {
                    "top": [80, 30, 320, 580],
                    "side": [140, 60, 320, 580],
                    "wrist_left": [80, 30, 320, 580],
                    "wrist_right": [80, 30, 320, 580],
                },
                "source_inventory": [
                    {"name": "demo_00.h5", "frames": frames},
                    {"name": "demo_01.h5", "frames": frames},
                ],
            }
        )
    )
    (rgb / "episode_names.json").write_text(json.dumps(["demo_00.h5", "demo_01.h5"]))
    (rgb / "splits.json").write_text(
        json.dumps({"splits": {"train": ["demo_00.h5"], "val": [], "test": ["demo_01.h5"]}})
    )

    output = tmp_path / "depth"
    manifest = convert_h5_to_depth_memmap(source, rgb, output, progress=False)
    assert manifest.is_file()
    store = ACMTActDepthMemmapStore(output, expected_frames=6, expected_episodes=2)
    assert store.depth.shape == (6, 4, 320, 580)
    assert store.depth[0, 0, 0, 0] == 100
    assert store.depth[0, 1, 0, 0] == 10
    assert store.depth[0, 2, 0, 0] == 20
    assert store.depth[0, 3, 0, 0] == 20
