"""Training-only ACMT-ACT memmap conversion and dataset backend.

The policy never depends on this module at deployment time.  Conversion reads
the original ACMT H5 files once, while :class:`ACMTACTMemmapDataset` opens only
read-only NumPy memmaps during training.  The format intentionally keeps one
action per source frame; action chunks and padding are assembled causally in
``__getitem__`` so the episode boundary cannot leak into a label.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import h5py
import numpy as np
import torch
from numpy.lib.format import open_memmap
from torch.utils.data import Dataset
from tqdm.auto import tqdm


MEMMAP_VERSION = "acmt_act_memmap_v1"
TARGETS_VERSION = "acmt_act_targets_v1"
GOAL_XYZ = "observation.acmt_act.goal_xyz"
GOAL_VALID = "observation.acmt_act.goal_valid"
CAMERA_NAMES = ("top", "side", "wrist_left", "wrist_right")
CROP_PARAMS = {
    "top": (80, 30, 320, 580),
    "side": (140, 60, 320, 580),
    "wrist_left": (80, 30, 320, 580),
    "wrist_right": (80, 30, 320, 580),
}
RGB_SHAPE = (320, 580, 3)
REQUIRED_KEYS = (
    "observations/rgb/top",
    "observations/rgb/side",
    "observations/rgb/wrist",
    "observations/robot_state/q",
    "observations/gripper/gPO",
    "observations/tactile/force",
    "actions/gello_q",
    "actions/gello_gripper_cmd",
)
ARRAY_SPECS = {
    "rgb.npy": {"dtype": "uint8", "tail_shape": [4, 320, 580, 3]},
    "state.npy": {"dtype": "float32", "tail_shape": [8]},
    "tactile.npy": {"dtype": "float32", "tail_shape": [2, 35, 20, 3]},
    "action.npy": {"dtype": "float32", "tail_shape": [8]},
    "sample_valid.npy": {"dtype": "bool", "tail_shape": []},
    "episode_ends.npy": {"dtype": "int64", "tail_shape": []},
}


def _targets_manifest_path(root: Path) -> Path:
    return root / "acmt_act_targets_manifest.json"


def _targets_npz_path(root: Path) -> Path:
    return root / "acmt_act_targets.npz"


def _first_grasp_rise(gpo: np.ndarray) -> int | None:
    """Return the first physical closing transition, never a later re-grasp."""

    values = np.asarray(gpo, dtype=np.int16).reshape(-1)
    indices = np.flatnonzero((values[:-1] == 3) & (values[1:] > 3))
    if indices.size:
        return int(indices[0])
    # A few old recordings use a slightly different quantization.  Accept the
    # monotone equivalent, but keep the first transition only.
    indices = np.flatnonzero((values[:-1] <= 3) & (values[1:] > 3))
    return int(indices[0]) if indices.size else None


def build_acmt_act_targets(
    data_dir: str | os.PathLike[str],
    memmap_dir: str | os.PathLike[str],
    *,
    split_file: str | os.PathLike[str] | None = None,
    force: bool = False,
) -> Path:
    """Build the small action/goal sidecar used by the corrected ACMT-ACT.

    RGB and force arrays remain in the existing Memmap.  This command reads
    the source H5 once to add the fields absent from that training-only
    format, and writes atomically so an interrupted run cannot be consumed.
    """

    data_root = Path(data_dir).resolve()
    root = Path(memmap_dir).resolve()
    names_payload = _read_json(root / "episode_names.json")
    if not isinstance(names_payload, list) or not names_payload:
        raise ValueError(f"missing episode_names.json in {root}")
    names = [Path(str(name)).name for name in names_payload]
    if split_file is not None:
        splits = _normalise_splits(Path(split_file), names)
    else:
        raw_splits = _read_json(root / "splits.json")
        splits = raw_splits.get("splits", raw_splits) if isinstance(raw_splits, dict) else {}
    source_inventory = _source_inventory(data_root, names)
    source_hash = _hash_json(source_inventory)
    manifest_path = _targets_manifest_path(root)
    final_npz = _targets_npz_path(root)
    expected = {
        "targets_version": TARGETS_VERSION,
        "source_hash": source_hash,
        "names": names,
        "splits_hash": _hash_json(splits),
    }
    if not force and manifest_path.is_file() and final_npz.is_file():
        previous = _read_json(manifest_path)
        if isinstance(previous, dict) and all(previous.get(key) == value for key, value in expected.items()):
            return final_npz

    store = ACMTActMemmapStore(root)
    if len(store.episode_ends) != len(names):
        raise ValueError("targets and Memmap episode counts do not match")
    total = len(store.rgb)
    goal = np.zeros((total, 3), dtype=np.float32)
    valid = np.zeros((total,), dtype=np.bool_)
    phase = np.zeros((total,), dtype=np.int8)
    grasp_frame = np.full((len(names),), -1, dtype=np.int64)
    tmp_npz = final_npz.with_suffix(".npz.partial")
    try:
        for episode_index, name in enumerate(names):
            start, end = store.bounds(episode_index)
            with h5py.File(data_root / name, "r") as handle:
                if "observations/robot_state/O_T_EE" not in handle:
                    raise KeyError(f"{name} is missing observations/robot_state/O_T_EE required for goal labels")
                gpo = np.asarray(handle["observations/gripper/gPO"], dtype=np.uint8)
                ee = np.asarray(handle["observations/robot_state/O_T_EE"], dtype=np.float32)
            if len(gpo) != end - start or ee.shape != (end - start, 4, 4):
                raise ValueError(f"{name}: source and Memmap lengths/shapes do not match")
            onset = _first_grasp_rise(gpo)
            if onset is None:
                continue
            grasp_frame[episode_index] = onset
            grasp_goal = ee[onset, :3, 3]
            goal[start:end] = grasp_goal
            valid[start:end] = True
            local = np.arange(end - start)
            phase[start:end] = np.where(
                local < onset - 8,
                0,
                np.where(local <= onset + 8, 1, 2),
            ).astype(np.int8)
        with open(tmp_npz, "wb") as stream:
            np.savez(stream, goal_xyz=goal, goal_valid=valid, phase=phase, grasp_frame=grasp_frame)
        os.replace(tmp_npz, final_npz)
        manifest = dict(expected)
        manifest.update({"complete": True, "frames": total, "episode_count": len(names)})
        _atomic_json(manifest_path, manifest)
    finally:
        tmp_npz.unlink(missing_ok=True)
    return final_npz


def build_acmt_act_policy_stats(
    memmap_dir: str | os.PathLike[str],
    *,
    split: str = "train",
    force: bool = False,
) -> Path:
    """Compute statistics for residual joint targets and physical gripper labels."""

    root = Path(memmap_dir).resolve()
    output = root / "acmt_act_policy_stats.json"
    if output.is_file() and not force:
        return output
    store = ACMTActMemmapStore(root)
    payload = _read_json(root / "splits.json") or {}
    splits = payload.get("splits", payload)
    names = _read_json(root / "episode_names.json") or []
    name_to_index = {str(name): i for i, name in enumerate(names)}
    selected = [name_to_index[Path(str(name)).name] for name in splits.get(split, [])]
    if not selected:
        raise ValueError(f"no episodes in split {split!r}")

    count = 0
    total = np.zeros(8, np.float64)
    total_sq = np.zeros(8, np.float64)
    minimum = np.full(8, np.inf, np.float64)
    maximum = np.full(8, -np.inf, np.float64)
    goals: list[np.ndarray] = []
    for episode_index in selected:
        start, end = store.bounds(episode_index)
        q = np.asarray(store.state[start:end, :7], dtype=np.float32)
        raw_action = np.asarray(store.action[start:end], dtype=np.float32)
        length = end - start
        anchors = np.arange(length, dtype=np.int64)
        for horizon in range(16):
            target = np.minimum(anchors + horizon, length - 1)
            mask = anchors + horizon < length
            values = np.empty((length, 8), dtype=np.float32)
            values[:, :7] = raw_action[target, :7] - q
            values[:, 7] = 1.0 - raw_action[target, 7]
            values = values[mask]
            if values.size:
                count += int(values.shape[0])
                total += values.sum(0, dtype=np.float64)
                total_sq += np.square(values, dtype=np.float64).sum(0)
                minimum = np.minimum(minimum, values.min(0))
                maximum = np.maximum(maximum, values.max(0))
        if store.targets is not None and bool(store.targets["goal_valid"][start]):
            goals.append(np.asarray(store.targets["goal_xyz"][start], dtype=np.float32))
    if count == 0:
        raise ValueError("no valid action targets available for policy statistics")
    mean = total / count
    std = np.sqrt(np.maximum(total_sq / count - np.square(mean), 1e-12))
    stats: dict[str, Any] = {
        "action": {
            "count": count,
            "mean": mean.tolist(),
            "std": std.tolist(),
            "min": minimum.tolist(),
            "max": maximum.tolist(),
        }
    }
    if goals:
        goal_values = np.stack(goals)
        stats["goal"] = {
            "count": int(len(goals)),
            "mean": goal_values.mean(0).tolist(),
            "std": np.maximum(goal_values.std(0), 1e-4).tolist(),
            "min": goal_values.min(0).tolist(),
            "max": goal_values.max(0).tolist(),
        }
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, output)
    return output


def _atomic_json(path: Path, payload: Any) -> None:
    # Include the process id so an accidentally duplicated/resumed converter
    # cannot race on one shared ``.tmp`` pathname and remove the other writer's
    # temporary file before ``os.replace``.
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _read_json(path: Path) -> Any | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _hash_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _normalise_splits(split_file: Path, names: list[str]) -> dict[str, list[str]]:
    payload = _read_json(split_file)
    if not isinstance(payload, dict):
        raise ValueError(f"split file must contain a JSON object: {split_file}")
    values = payload.get("splits", payload)
    if not isinstance(values, dict):
        raise ValueError(f"split file has no splits mapping: {split_file}")
    known = set(names)
    result: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        result[split] = [Path(str(value)).name for value in values.get(split, [])]
        unknown = sorted(set(result[split]) - known)
        if unknown:
            raise ValueError(f"{split_file} references files outside data_dir: {unknown[:3]}")
    split_sets = {key: set(value) for key, value in result.items()}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_sets[left] & split_sets[right]
        if overlap:
            raise ValueError(f"{left}/{right} split overlap: {sorted(overlap)[:3]}")
    covered = set().union(*split_sets.values())
    if covered != known:
        missing = sorted(known - covered)
        extra = sorted(covered - known)
        raise ValueError(f"split file does not cover H5 inventory (missing={missing[:3]}, extra={extra[:3]})")
    return result


def _source_inventory(data_dir: Path, names: Iterable[str]) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for name in names:
        path = data_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as handle:
            missing = [key for key in REQUIRED_KEYS if key not in handle]
            if missing:
                raise KeyError(f"{path} missing keys: {missing}")
            shapes = {
                "top": list(handle["observations/rgb/top"].shape),
                "side": list(handle["observations/rgb/side"].shape),
                "wrist": list(handle["observations/rgb/wrist"].shape),
                "q": list(handle["observations/robot_state/q"].shape),
                "gPO": list(handle["observations/gripper/gPO"].shape),
                "force": list(handle["observations/tactile/force"].shape),
                "action_q": list(handle["actions/gello_q"].shape),
                "action_gripper": list(handle["actions/gello_gripper_cmd"].shape),
            }
            length = int(shapes["top"][0])
            if tuple(shapes["top"][1:]) != (480, 640, 3) or tuple(shapes["side"][1:]) != (480, 640, 3):
                raise ValueError(f"{path}: top/side RGB must be [T,480,640,3]")
            if tuple(shapes["wrist"][1:]) != (2, 480, 640, 3):
                raise ValueError(f"{path}: wrist RGB must be [T,2,480,640,3]")
            if tuple(shapes["q"][1:]) != (7,) or tuple(shapes["force"][1:]) != (2, 35, 20, 3):
                raise ValueError(f"{path}: state/tactile shapes are incompatible")
            if tuple(shapes["action_q"][1:]) != (7,) or tuple(shapes["action_gripper"][1:]) != ():
                raise ValueError(f"{path}: actions must be [T,7] and [T]")
            if any(int(shape[0]) != length for shape in shapes.values()):
                raise ValueError(f"{path}: all arrays must have the same frame count")
            if "sample_valid" in handle and tuple(handle["sample_valid"].shape) != (length,):
                raise ValueError(f"{path}: sample_valid must be [T]")
        stat = path.stat()
        inventory.append(
            {
                "name": name,
                "frames": length,
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "shapes": shapes,
            }
        )
    return inventory


def _manifest(data_dir: Path, split_file: Path, inventory: list[dict[str, Any]], splits: dict[str, list[str]]) -> dict[str, Any]:
    return {
        "memmap_version": MEMMAP_VERSION,
        "data_dir": str(data_dir),
        "split_file": str(split_file),
        "source_inventory": inventory,
        "source_inventory_sha256": _hash_json(inventory),
        "splits": splits,
        "split_sha256": _hash_json(splits),
        "camera_order": list(CAMERA_NAMES),
        "crop_params": {key: list(value) for key, value in CROP_PARAMS.items()},
        "preprocess": {
            "input_shape": [480, 640, 3],
            "resize": None,
            "crop": "camera-specific integer crop",
            "output_shape": list(RGB_SHAPE),
            "dtype": "uint8",
        },
        "arrays": ARRAY_SPECS,
        "complete": False,
    }


def _manifest_matches(existing: dict[str, Any], expected: dict[str, Any]) -> bool:
    keys = (
        "memmap_version",
        "data_dir",
        "source_inventory_sha256",
        "split_sha256",
        "camera_order",
        "crop_params",
        "preprocess",
        "arrays",
    )
    return all(existing.get(key) == expected.get(key) for key in keys)


def _read_rgb_chunk(handle: h5py.File, start: int, stop: int) -> np.ndarray:
    top = np.asarray(handle["observations/rgb/top"][start:stop])
    side = np.asarray(handle["observations/rgb/side"][start:stop])
    wrist = np.asarray(handle["observations/rgb/wrist"][start:stop])
    crops = []
    for name, image in zip(CAMERA_NAMES, (top, side, wrist[:, 0], wrist[:, 1]), strict=True):
        y, x, height, width = CROP_PARAMS[name]
        if image.shape[1:] != (480, 640, 3):
            raise ValueError(f"{name} image has unexpected shape {image.shape}")
        crops.append(image[:, y : y + height, x : x + width, :])
    return np.stack(crops, axis=1).astype(np.uint8, copy=False)


def _create_partial_arrays(root: Path, total_frames: int, episode_count: int) -> dict[str, np.memmap]:
    return {
        "rgb.npy": open_memmap(root / "rgb.npy.partial", mode="w+", dtype=np.uint8, shape=(total_frames, 4, *RGB_SHAPE)),
        "state.npy": open_memmap(root / "state.npy.partial", mode="w+", dtype=np.float32, shape=(total_frames, 8)),
        "tactile.npy": open_memmap(root / "tactile.npy.partial", mode="w+", dtype=np.float32, shape=(total_frames, 2, 35, 20, 3)),
        "action.npy": open_memmap(root / "action.npy.partial", mode="w+", dtype=np.float32, shape=(total_frames, 8)),
        "sample_valid.npy": open_memmap(root / "sample_valid.npy.partial", mode="w+", dtype=bool, shape=(total_frames,)),
        "episode_ends.npy": open_memmap(root / "episode_ends.npy.partial", mode="w+", dtype=np.int64, shape=(episode_count,)),
    }


def _required_array_bytes(total_frames: int, episode_count: int) -> int:
    """Return the final on-disk size of the published NumPy arrays."""

    total = 0
    for name, spec in ARRAY_SPECS.items():
        shape = (total_frames, *spec["tail_shape"]) if spec["tail_shape"] else (total_frames,)
        if name == "episode_ends.npy":
            shape = (episode_count,)
        total += int(np.prod(shape, dtype=np.int64)) * np.dtype(spec["dtype"]).itemsize
    return total


def _open_partial_arrays(root: Path) -> dict[str, np.memmap]:
    arrays: dict[str, np.memmap] = {}
    for name in ARRAY_SPECS:
        path = root / (name + ".partial")
        if not path.is_file():
            raise FileNotFoundError(f"incomplete conversion is missing {path}")
        arrays[name] = np.load(path, mmap_mode="r+")
    return arrays


def _flush(arrays: dict[str, np.memmap]) -> None:
    for array in arrays.values():
        array.flush()


def _write_stats(root: Path, arrays: dict[str, np.memmap], inventory: list[dict[str, Any]], splits: dict[str, list[str]], names: list[str]) -> None:
    train_indices = [names.index(name) for name in splits["train"]]
    starts = np.cumsum([0, *[int(item["frames"]) for item in inventory[:-1]]])
    frame_indices = np.concatenate(
        [np.arange(starts[index], starts[index] + int(inventory[index]["frames"])) for index in train_indices]
    )

    def stats(value: np.ndarray) -> dict[str, list[float] | int]:
        flat = np.asarray(value, dtype=np.float64).reshape(-1, value.shape[-1] if value.ndim > 1 else 1)
        return {
            "mean": flat.mean(0).astype(np.float32).tolist(),
            "std": np.maximum(flat.std(0), 1e-6).astype(np.float32).tolist(),
            "min": flat.min(0).astype(np.float32).tolist(),
            "max": flat.max(0).astype(np.float32).tolist(),
            "count": int(flat.shape[0]),
        }

    state = np.asarray(arrays["state.npy"][frame_indices], dtype=np.float32)
    action = np.asarray(arrays["action.npy"][frame_indices], dtype=np.float32)
    force = np.asarray(arrays["tactile.npy"][frame_indices], dtype=np.float32)
    force_flat = force.transpose(0, 1, 2, 3, 4).reshape(-1, 3)
    payload = {
        "observation.state": stats(state),
        "action": stats(action),
        "observation.xense.sensor0.force_field": stats(force[:, 0].reshape(-1, 3)),
        "observation.xense.sensor1.force_field": stats(force[:, 1].reshape(-1, 3)),
        "force_mean": force_flat.mean(0).astype(np.float32).tolist(),
        "force_std": np.maximum(force_flat.std(0), 1e-6).astype(np.float32).tolist(),
    }
    _atomic_json(root / "stats.json", payload)


def convert_h5_to_memmap(
    data_dir: str | os.PathLike[str],
    split_file: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    chunk_frames: int = 32,
    resume: bool = False,
    progress: bool = True,
    device: str | torch.device = "cpu",
) -> Path:
    """Convert all H5 episodes to the exact ACMT-ACT cropped memmap format."""

    del device  # Cropping is a lossless CPU copy; no GPU work is needed.
    if chunk_frames <= 0:
        raise ValueError("chunk_frames must be positive")
    data_root = Path(data_dir).resolve()
    split_path = Path(split_file).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Only one converter may mutate a destination at a time.  ``resume`` is
    # intentionally safe after interruption, but two simultaneous resumes
    # would otherwise race on the same memmap pages and manifest.
    lock_handle = (output / ".conversion.lock").open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_handle.close()
        raise RuntimeError(f"another ACMT-ACT conversion is already running: {output}") from exc
    names = sorted(path.name for path in data_root.glob("*.h5"))
    if not names:
        raise FileNotFoundError(f"no H5 files found in {data_root}")
    splits = _normalise_splits(split_path, names)
    inventory = _source_inventory(data_root, names)
    expected = _manifest(data_root, split_path, inventory, splits)
    final_manifest = output / "manifest.json"
    existing = _read_json(final_manifest)
    if existing is not None:
        if not existing.get("complete") or not _manifest_matches(existing, expected):
            raise ValueError(f"existing ACMT-ACT memmap manifest is stale or incomplete: {final_manifest}")
        # A complete manifest is only reusable when every published array is
        # still present.  This catches partial manual cleanup before a later
        # training run silently accepts a broken cache.
        missing = [name for name in ARRAY_SPECS if not (output / name).is_file()]
        if missing:
            raise FileNotFoundError(f"complete ACMT-ACT memmap is missing arrays: {missing}")
        return final_manifest

    progress_path = output / "conversion_state.json"
    state = _read_json(progress_path)
    # Fail before allocating a multi-hundred-GB sparse memmap when the target
    # filesystem cannot hold its eventual fully-written size.  A 1% cushion
    # leaves room for the manifest/statistics and avoids ending with a corrupt
    # partial conversion; callers can point ``--output-dir`` at another mount.
    if state is None:
        required_bytes = _required_array_bytes(sum(int(item["frames"]) for item in inventory), len(names))
        free_bytes = shutil.disk_usage(output).free
        if free_bytes < int(required_bytes * 1.01):
            raise OSError(
                f"insufficient free space for ACMT-ACT memmap: need about {required_bytes / 2**30:.1f} GiB "
                f"(+1%), have {free_bytes / 2**30:.1f} GiB at {output}"
            )
    total_frames = sum(int(item["frames"]) for item in inventory)
    if state is not None:
        if not resume:
            raise FileExistsError(f"partial conversion exists at {output}; pass --resume")
        if not _manifest_matches(state.get("manifest", {}), expected):
            raise ValueError("partial ACMT-ACT conversion does not match current source inventory")
        arrays = _open_partial_arrays(output)
        completed_names = list(state.get("completed_names", []))
        completed = len(completed_names)
        if names[:completed] != completed_names:
            raise ValueError("partial conversion episode order does not match current H5 inventory")
    else:
        if any((output / (name + ".partial")).exists() for name in ARRAY_SPECS):
            raise FileExistsError(f"partial array files exist at {output}; pass --resume")
        arrays = _create_partial_arrays(output, total_frames, len(names))
        completed_names = []
        completed = 0
        _atomic_json(progress_path, {"manifest": expected, "completed_names": completed_names})

    offsets = np.cumsum([0, *[int(item["frames"]) for item in inventory[:-1]]])
    initial = int(offsets[completed]) if completed < len(offsets) else total_frames
    bar = tqdm(
        total=total_frames,
        initial=initial,
        desc=f"acmt-act-memmap/{data_root.name}",
        unit="frame",
        dynamic_ncols=True,
        disable=not progress,
    )
    started = time.perf_counter()
    try:
        for episode_index in range(completed, len(names)):
            name = names[episode_index]
            length = int(inventory[episode_index]["frames"])
            destination_start = int(offsets[episode_index])
            bar.set_postfix(demo=name, episodes=f"{episode_index}/{len(names)}")
            with h5py.File(data_root / name, "r") as handle:
                for start in range(0, length, chunk_frames):
                    stop = min(length, start + chunk_frames)
                    rgb = _read_rgb_chunk(handle, start, stop)
                    q = np.nan_to_num(np.asarray(handle["observations/robot_state/q"][start:stop], dtype=np.float32))
                    gpo = np.nan_to_num(np.asarray(handle["observations/gripper/gPO"][start:stop], dtype=np.float32)).reshape(-1, 1) / 255.0
                    tactile = np.nan_to_num(np.asarray(handle["observations/tactile/force"][start:stop], dtype=np.float32))
                    action_q = np.nan_to_num(np.asarray(handle["actions/gello_q"][start:stop], dtype=np.float32))
                    action_g = np.nan_to_num(np.asarray(handle["actions/gello_gripper_cmd"][start:stop], dtype=np.float32)).reshape(-1, 1)
                    valid = np.asarray(handle["sample_valid"][start:stop], dtype=bool) if "sample_valid" in handle else np.ones(stop - start, dtype=bool)
                    destination = slice(destination_start + start, destination_start + stop)
                    arrays["rgb.npy"][destination] = rgb
                    arrays["state.npy"][destination] = np.concatenate([q, gpo], axis=-1)
                    arrays["tactile.npy"][destination] = tactile
                    arrays["action.npy"][destination] = np.concatenate([action_q, action_g], axis=-1)
                    arrays["sample_valid.npy"][destination] = valid
                    bar.update(stop - start)
            arrays["episode_ends.npy"][episode_index] = destination_start + length
            _flush(arrays)
            completed_names.append(name)
            _atomic_json(progress_path, {"manifest": expected, "completed_names": completed_names})
    finally:
        bar.close()

    if len(completed_names) != len(names):
        raise RuntimeError("conversion stopped before all episodes completed")
    _flush(arrays)
    for name in ARRAY_SPECS:
        os.replace(output / (name + ".partial"), output / name)
    _atomic_json(output / "episode_names.json", names)
    _atomic_json(output / "splits.json", {"splits": splits})
    _write_stats(output, arrays, inventory, splits, names)
    final = dict(expected)
    final.update({"complete": True, "total_frames": total_frames, "episode_count": len(names), "created_s": time.time() - started})
    _atomic_json(final_manifest, final)
    progress_path.unlink(missing_ok=True)
    return final_manifest


class ACMTActMemmapStore:
    """Strict read-only view over one completed conversion."""

    def __init__(self, root: str | os.PathLike[str]):
        self.root = Path(root).resolve()
        manifest = _read_json(self.root / "manifest.json")
        if not isinstance(manifest, dict) or manifest.get("memmap_version") != MEMMAP_VERSION or not manifest.get("complete"):
            raise ValueError(f"invalid or incomplete ACMT-ACT memmap: {self.root}")
        self.manifest = manifest
        self.rgb = self._load("rgb.npy", np.uint8)
        self.state = self._load("state.npy", np.float32)
        self.tactile = self._load("tactile.npy", np.float32)
        self.action = self._load("action.npy", np.float32)
        self.sample_valid = self._load("sample_valid.npy", np.bool_)
        self.episode_ends = self._load("episode_ends.npy", np.int64)
        targets_path = _targets_npz_path(self.root)
        targets_manifest = _read_json(_targets_manifest_path(self.root))
        if targets_path.is_file() and isinstance(targets_manifest, dict) and targets_manifest.get("complete"):
            # Do not retain NumPy's NpzFile object here.  It owns one shared
            # ZipFile handle; with DataLoader workers forked from the parent,
            # concurrent indexed reads can corrupt the shared cursor and
            # raise ``BadZipFile: Overlapped entries``.  The sidecar is tiny
            # compared with the RGB Memmap, so materialize it once and let
            # workers inherit ordinary read-only arrays.
            with np.load(targets_path, allow_pickle=False) as loaded:
                required = {"goal_xyz", "goal_valid", "phase", "grasp_frame"}
                if set(loaded.files) != required:
                    raise ValueError(f"invalid ACMT-ACT targets sidecar keys in {targets_path}")
                copied = {name: np.array(loaded[name], copy=True) for name in required}
            if copied["goal_xyz"].shape != (len(self.rgb), 3) or copied["goal_valid"].shape != (len(self.rgb),):
                raise ValueError(f"invalid ACMT-ACT goal sidecar shape in {targets_path}")
            if copied["phase"].shape != (len(self.rgb),) or copied["grasp_frame"].ndim != 1:
                raise ValueError(f"invalid ACMT-ACT phase sidecar shape in {targets_path}")
            self.targets = copied
        else:
            self.targets = None
        expected = manifest["arrays"]
        for name, value in (("rgb.npy", self.rgb), ("state.npy", self.state), ("tactile.npy", self.tactile), ("action.npy", self.action)):
            if tuple(value.shape[1:]) != tuple(expected[name]["tail_shape"]):
                raise ValueError(f"invalid {name} shape {value.shape}")
        if len(self.sample_valid) != len(self.rgb) or len(self.episode_ends) != int(manifest["episode_count"]):
            raise ValueError("ACMT-ACT memmap array lengths are inconsistent")
        if tuple(self.rgb.shape[1:]) != (4, 320, 580, 3):
            raise ValueError(f"ACMT-ACT memmap RGB shape must be [N,4,320,580,3], got {self.rgb.shape}")

    def _load(self, name: str, dtype: np.dtype[Any]) -> np.memmap:
        path = self.root / name
        value = np.load(path, mmap_mode="r")
        if value.dtype != dtype:
            raise ValueError(f"{path} dtype is {value.dtype}, expected {dtype}")
        return value

    def bounds(self, episode_index: int) -> tuple[int, int]:
        end = int(self.episode_ends[episode_index])
        start = 0 if episode_index == 0 else int(self.episode_ends[episode_index - 1])
        return start, end


class ACMTActMemmapMetadata:
    """Small metadata facade consumed by LeRobot's policy/training factory."""

    def __init__(
        self,
        store: ACMTActMemmapStore,
        selected_indices: list[int],
        repo_id: str,
        camera_indices: tuple[int, ...] | None = None,
    ):
        from lerobot.policies.acmt_act.configuration_acmt_act import XENSE0, XENSE1, rgb_key

        self.repo_id = repo_id
        self.root = store.root
        self.fps = 30
        self.robot_type = "fr3"
        self.camera_indices = tuple(range(4) if camera_indices is None else camera_indices)
        if not self.camera_indices or any(index < 0 or index >= 4 for index in self.camera_indices):
            raise ValueError(f"camera_indices must select distinct entries from the four-way memmap, got {self.camera_indices}")
        if len(set(self.camera_indices)) != len(self.camera_indices):
            raise ValueError(f"camera_indices must be distinct, got {self.camera_indices}")
        self.camera_keys = [rgb_key(f"camera.cam{index + 1}") for index in self.camera_indices]
        self.depth_keys: list[str] = []
        self.features = {
            **{key: {"dtype": "image", "shape": [480, 640, 3], "names": ["height", "width", "channel"]} for key in self.camera_keys},
            "observation.state": {"dtype": "float32", "shape": [8], "names": [f"state_{i}" for i in range(8)]},
            XENSE0: {"dtype": "float32", "shape": [35, 20, 3], "names": ["height", "width", "channel"]},
            XENSE1: {"dtype": "float32", "shape": [35, 20, 3], "names": ["height", "width", "channel"]},
            "action": {"dtype": "float32", "shape": [8], "names": [*(f"joint_{i}" for i in range(7)), "gripper"]},
        }
        raw_stats = _read_json(store.root / "stats.json") or {}
        corrected_stats = _read_json(store.root / "acmt_act_policy_stats.json") or {}
        if isinstance(corrected_stats, dict) and isinstance(corrected_stats.get("action"), dict):
            raw_stats = dict(raw_stats)
            raw_stats["action"] = corrected_stats["action"]
            if isinstance(corrected_stats.get("goal"), dict):
                raw_stats[GOAL_XYZ] = corrected_stats["goal"]
        self.stats = {
            key: {name: torch.as_tensor(value, dtype=torch.float32) if name != "count" else value for name, value in values.items()}
            for key, values in raw_stats.items()
            if isinstance(values, dict)
        }
        starts = np.cumsum([0, *[store.bounds(i)[1] - store.bounds(i)[0] for i in range(len(store.episode_ends) - 1)]])
        local_from, local_to, local_tasks = [], [], []
        total = 0
        for local_index, global_index in enumerate(selected_indices):
            start, end = store.bounds(global_index)
            length = end - start
            local_from.append(total)
            total += length
            local_to.append(total)
            local_tasks.append([repo_id])
        self.episodes = {
            "dataset_from_index": local_from,
            "dataset_to_index": local_to,
            "episode_index": list(range(len(selected_indices))),
            "tasks": local_tasks,
        }
        self._selected_indices = selected_indices
        self.total_frames = total
        self.total_episodes = len(selected_indices)
        self.total_tasks = 1
        self.tasks = SimpleNamespace(index=[repo_id])
        self.info = SimpleNamespace(features=self.features, total_frames=total, total_episodes=len(selected_indices), fps=self.fps)

    @property
    def has_language_columns(self) -> bool:
        return False


class ACMTACTMemmapDataset(Dataset):
    """Map-style training dataset with fixed episode split and causal chunks."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        split: str,
        repo_id: str = "local/acmt-act",
        episodes: list[int] | None = None,
        camera_indices: tuple[int, ...] | None = None,
    ):
        self.store = ACMTActMemmapStore(root)
        split_payload = _read_json(self.store.root / "splits.json")
        names = _read_json(self.store.root / "episode_names.json")
        if not isinstance(split_payload, dict) or not isinstance(names, list):
            raise ValueError("ACMT-ACT memmap is missing episode_names.json or splits.json")
        splits = split_payload.get("splits", split_payload)
        selected_names = [Path(str(name)).name for name in splits.get(split, [])]
        name_to_index = {str(name): index for index, name in enumerate(names)}
        selected_indices = [name_to_index[name] for name in selected_names]
        if episodes is not None:
            selected_indices = [selected_indices[index] for index in episodes]
        if not selected_indices:
            raise ValueError(f"ACMT-ACT memmap split {split!r} is empty")
        self._selected_indices = selected_indices
        self.camera_indices = tuple(range(4) if camera_indices is None else camera_indices)
        self.meta = ACMTActMemmapMetadata(self.store, selected_indices, repo_id, self.camera_indices)
        self.episodes = list(range(len(selected_indices)))
        self.num_frames = self.meta.total_frames
        self.num_episodes = self.meta.total_episodes
        self._local_ends = np.asarray(self.meta.episodes["dataset_to_index"], dtype=np.int64)
        self._local_starts = np.asarray(self.meta.episodes["dataset_from_index"], dtype=np.int64)

    def __len__(self) -> int:
        return self.num_frames

    @property
    def absolute_to_relative_idx(self) -> None:
        return None

    @property
    def hf_dataset(self):
        raise AttributeError("ACMT-ACT memmap training has no HuggingFace dataset")

    def _locate(self, local_index: int) -> tuple[int, int, int, int]:
        if local_index < 0 or local_index >= self.num_frames:
            raise IndexError(local_index)
        local_ep = int(np.searchsorted(self._local_ends, local_index, side="right"))
        local_start = int(self._local_starts[local_ep])
        global_ep = self._selected_indices[local_ep]
        global_start, global_end = self.store.bounds(global_ep)
        offset = local_index - local_start
        return local_ep, global_ep, global_start + offset, global_end

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        from lerobot.policies.acmt_act.configuration_acmt_act import XENSE0, XENSE1, rgb_key

        _, global_ep, global_index, global_end = self._locate(int(index))
        global_start, _ = self.store.bounds(global_ep)
        # The public action target is A[t:t+15].  At an episode tail, repeat
        # the last recorded action and mark those positions as padding.
        offsets = np.minimum(np.arange(16, dtype=np.int64) + global_index, global_end - 1)
        pad = (np.arange(16, dtype=np.int64) + global_index) >= global_end
        rgb = self.store.rgb[global_index]
        result: dict[str, torch.Tensor] = {
            # Memmaps are read-only by design.  Materialize each sample before
            # creating a Tensor so callers cannot trigger undefined writes on a
            # non-writable NumPy view (and worker warnings stay silent).
            "observation.state": torch.from_numpy(np.array(self.store.state[global_index], dtype=np.float32, copy=True)),
            XENSE0: torch.from_numpy(np.array(self.store.tactile[global_index, 0], dtype=np.float32, copy=True)),
            XENSE1: torch.from_numpy(np.array(self.store.tactile[global_index, 1], dtype=np.float32, copy=True)),
            "action": torch.from_numpy(np.array(self.store.action[offsets], dtype=np.float32, copy=True)),
            "action_is_pad": torch.from_numpy(pad.astype(bool)),
            # The observation processor recognizes this shape as already
            # cropped.  It remains a private training marker and is ignored by
            # policy feature selection.
            "_acmt_act.precropped": torch.tensor(True),
        }
        # The source H5 stores the Gello wire command as 1=open/0=closed;
        # expose the physical policy convention 0=open/1=closed.
        result["action"][:, 7] = 1.0 - result["action"][:, 7]
        if self.store.targets is not None:
            result[GOAL_XYZ] = torch.from_numpy(
                np.array(self.store.targets["goal_xyz"][global_index], dtype=np.float32, copy=True)
            )
            result[GOAL_VALID] = torch.tensor(bool(self.store.targets["goal_valid"][global_index]))
            result["_acmt_act.phase"] = torch.tensor(int(self.store.targets["phase"][global_index]), dtype=torch.int64)
        for camera_index in self.camera_indices:
            result[rgb_key(f"camera.cam{camera_index + 1}")] = torch.from_numpy(
                np.array(rgb[camera_index], copy=True)
            )
        return result


__all__ = [
    "ACMTACTMemmapDataset",
    "ACMTActMemmapMetadata",
    "ACMTActMemmapStore",
    "ARRAY_SPECS",
    "build_acmt_act_policy_stats",
    "build_acmt_act_targets",
    "CAMERA_NAMES",
    "CROP_PARAMS",
    "MEMMAP_VERSION",
    "convert_h5_to_memmap",
]
