from __future__ import annotations

import json

import h5py
import numpy as np

from lerobot.datasets.acmt_act_memmap import ACMTACTMemmapDataset, ACMTActMemmapStore, convert_h5_to_memmap
from lerobot.scripts.acmt_act_make_split import make_all_train_split


def _write_episode(path, length: int = 40) -> None:
    with h5py.File(path, "w") as handle:
        rgb = np.zeros((length, 480, 640, 3), dtype=np.uint8)
        wrist = np.stack([rgb, rgb], axis=1)
        handle.create_dataset("observations/rgb/top", data=rgb)
        handle.create_dataset("observations/rgb/side", data=rgb)
        handle.create_dataset("observations/rgb/wrist", data=wrist)
        handle.create_dataset("observations/robot_state/q", data=np.zeros((length, 7), np.float32))
        handle.create_dataset("observations/gripper/gPO", data=np.zeros(length, np.float32))
        handle.create_dataset("observations/tactile/force", data=np.zeros((length, 2, 35, 20, 3), np.float32))
        handle.create_dataset("actions/gello_q", data=np.zeros((length, 7), np.float32))
        handle.create_dataset("actions/gello_gripper_cmd", data=np.zeros(length, np.float32))


def test_build_report_invalid_rows_remove_complete_action_windows(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    episode = source / "demo.h5"
    _write_episode(episode)
    (source / "demo.build_report.json").write_text(
        json.dumps(
            {
                "warnings": [
                    {
                        "stream": "sample_valid",
                        "reason": "global_sample_invalid",
                        "hdf5_row": 10,
                    },
                    {"stream": "zmq_source_1", "reason": "stream_invalid", "hdf5_row": 11},
                ]
            }
        ),
        encoding="utf-8",
    )
    split = tmp_path / "splits.json"
    make_all_train_split(source, split)
    output = tmp_path / "memmap"
    convert_h5_to_memmap(source, split, output, chunk_frames=8, progress=False, validity_source="build_report")

    store = ACMTActMemmapStore(output)
    assert not bool(store.sample_valid[10])
    assert int(store.manifest["invalid_frame_count"]) == 1
    dataset = ACMTACTMemmapDataset(output, split="train", chunk_size=16)
    # Anchors 0..10 would include invalid row 10 in A[t:t+15]; 11..39 remain.
    assert len(dataset) == 29
    assert int(dataset._anchor_indices[0][0]) == 11
    assert all(
        bool(store.sample_valid[int(anchor) : min(int(anchor) + 16, 40)].all())
        for anchor in dataset._anchor_indices[0]
    )
