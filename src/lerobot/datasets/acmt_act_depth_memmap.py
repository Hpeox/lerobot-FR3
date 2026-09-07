"""Training-only aligned depth sidecar for ACMT-ACTv2.

The RGB Memmap is intentionally left untouched.  This sidecar contains only
the four cropped uint16 depth streams, with the same frame and episode order,
so deployment can use live RGB-D without ever opening these files.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy.lib.format import open_memmap
from tqdm.auto import tqdm


DEPTH_MEMMAP_VERSION = "acmt_act_depth_memmap_v1"
DEPTH_SHAPE = (4, 320, 580)
DEPTH_CROP_PARAMS = {
    "top": (80, 30, 320, 580),
    "side": (140, 60, 320, 580),
    "wrist_left": (80, 30, 320, 580),
    "wrist_right": (80, 30, 320, 580),
}
DEPTH_ARRAY_SPECS = {
    "depth.npy": {"dtype": "uint16", "tail_shape": list(DEPTH_SHAPE)},
    "episode_ends.npy": {"dtype": "int64", "tail_shape": []},
}
DEPTH_KEYS = (
    "observations/depth/top",
    "observations/depth/side",
    "observations/depth/wrist",
)


def _read_json(path: Path) -> Any | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _hash_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _normalise_splits(split_file: Path, names: list[str]) -> dict[str, list[str]]:
    payload = _read_json(split_file)
    if not isinstance(payload, dict):
        raise ValueError(f"split file must contain an object: {split_file}")
    values = payload.get("splits", payload)
    if not isinstance(values, dict):
        raise ValueError(f"split file has no splits mapping: {split_file}")
    known = set(names)
    result = {split: [Path(str(value)).name for value in values.get(split, [])] for split in ("train", "val", "test")}
    for split, selected in result.items():
        unknown = sorted(set(selected) - known)
        if unknown:
            raise ValueError(f"{split_file} references unknown episodes: {unknown[:3]}")
    sets = {key: set(value) for key, value in result.items()}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if sets[left] & sets[right]:
            raise ValueError(f"{left}/{right} split overlap")
    if set().union(*sets.values()) != known:
        raise ValueError("split file must cover every H5 episode exactly once")
    return result


def _inventory(data_dir: Path, names: list[str]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for name in names:
        path = data_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as handle:
            missing = [key for key in DEPTH_KEYS if key not in handle]
            if missing:
                raise KeyError(f"{path} missing depth keys: {missing}")
            shapes = {key: list(handle[key].shape) for key in DEPTH_KEYS}
            if tuple(shapes[DEPTH_KEYS[0]][1:]) != (480, 640):
                raise ValueError(f"{path}: top depth must be [T,480,640]")
            if tuple(shapes[DEPTH_KEYS[1]][1:]) != (480, 640):
                raise ValueError(f"{path}: side depth must be [T,480,640]")
            if tuple(shapes[DEPTH_KEYS[2]][1:]) != (2, 480, 640):
                raise ValueError(f"{path}: wrist depth must be [T,2,480,640]")
            length = int(shapes[DEPTH_KEYS[0]][0])
            if any(int(shape[0]) != length for shape in shapes.values()):
                raise ValueError(f"{path}: depth arrays have different frame counts")
        stat = path.stat()
        result.append({"name": name, "frames": length, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "shapes": shapes})
    return result


def _expected_manifest(
    data_dir: Path,
    split_file: Path,
    rgb_manifest: dict[str, Any],
    inventory: list[dict[str, Any]],
    splits: dict[str, list[str]],
) -> dict[str, Any]:
    return {
        "depth_memmap_version": DEPTH_MEMMAP_VERSION,
        "data_dir": str(data_dir),
        "split_file": str(split_file),
        "source_inventory": inventory,
        "source_inventory_sha256": _hash_json(inventory),
        "rgb_manifest_sha256": _hash_json(rgb_manifest),
        "camera_order": ["top", "side", "wrist_left", "wrist_right"],
        "crop_params": {key: list(value) for key, value in DEPTH_CROP_PARAMS.items()},
        "preprocess": {"input_shape": [480, 640], "resize": None, "output_shape": list(DEPTH_SHAPE), "dtype": "uint16", "unit": "millimeter"},
        "arrays": DEPTH_ARRAY_SPECS,
        "splits": splits,
        "split_sha256": _hash_json(splits),
        "complete": False,
    }


def _manifest_matches(previous: dict[str, Any], expected: dict[str, Any]) -> bool:
    keys = ("depth_memmap_version", "data_dir", "source_inventory_sha256", "rgb_manifest_sha256", "camera_order", "crop_params", "preprocess", "arrays", "split_sha256")
    return all(previous.get(key) == expected.get(key) for key in keys)


def _read_depth_chunk(handle: h5py.File, start: int, stop: int) -> np.ndarray:
    top = np.asarray(handle[DEPTH_KEYS[0]][start:stop])
    side = np.asarray(handle[DEPTH_KEYS[1]][start:stop])
    wrist = np.asarray(handle[DEPTH_KEYS[2]][start:stop])
    crops = []
    for name, image in zip(("top", "side", "wrist_left", "wrist_right"), (top, side, wrist[:, 0], wrist[:, 1]), strict=True):
        y, x, height, width = DEPTH_CROP_PARAMS[name]
        if image.shape[1:] != (480, 640):
            raise ValueError(f"{name} depth has unexpected shape {image.shape}")
        crops.append(image[:, y : y + height, x : x + width])
    return np.stack(crops, axis=1).astype(np.uint16, copy=False)


def _open_partial(root: Path) -> dict[str, np.memmap]:
    arrays: dict[str, np.memmap] = {}
    for name in DEPTH_ARRAY_SPECS:
        path = root / (name + ".partial")
        if not path.is_file():
            raise FileNotFoundError(f"incomplete depth conversion is missing {path}")
        arrays[name] = np.load(path, mmap_mode="r+")
    return arrays


def convert_h5_to_depth_memmap(
    data_dir: str | os.PathLike[str],
    rgb_memmap_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    split_file: str | os.PathLike[str] | None = None,
    chunk_frames: int = 32,
    resume: bool = False,
    progress: bool = True,
) -> Path:
    """Create or resume the aligned four-camera cropped depth sidecar."""

    if chunk_frames <= 0:
        raise ValueError("chunk_frames must be positive")
    data_root = Path(data_dir).resolve()
    rgb_root = Path(rgb_memmap_dir).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".conversion.lock").open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock.close()
        raise RuntimeError(f"another depth conversion is running: {output}") from exc

    rgb_manifest = _read_json(rgb_root / "manifest.json")
    names_payload = _read_json(rgb_root / "episode_names.json")
    if not isinstance(rgb_manifest, dict) or not rgb_manifest.get("complete") or not isinstance(names_payload, list):
        raise ValueError(f"RGB Memmap is incomplete or missing episode_names.json: {rgb_root}")
    names = [Path(str(name)).name for name in names_payload]
    rgb_inventory = rgb_manifest.get("source_inventory")
    if not isinstance(rgb_inventory, list) or len(rgb_inventory) != len(names):
        raise ValueError("RGB Memmap manifest has no complete per-episode source inventory")
    if rgb_manifest.get("camera_order") != ["top", "side", "wrist_left", "wrist_right"]:
        raise ValueError("RGB Memmap camera order is incompatible with the depth sidecar")
    if rgb_manifest.get("crop_params") != {key: list(value) for key, value in DEPTH_CROP_PARAMS.items()}:
        raise ValueError("RGB Memmap crop parameters do not match the depth sidecar")
    split_path = Path(split_file).resolve() if split_file else rgb_root / "splits.json"
    splits = _normalise_splits(split_path, names)
    inventory = _inventory(data_root, names)
    for rgb_item, depth_item in zip(rgb_inventory, inventory, strict=True):
        if Path(str(rgb_item.get("name"))).name != depth_item["name"]:
            raise ValueError("RGB and depth episode order does not match")
        if int(rgb_item.get("frames", -1)) != int(depth_item["frames"]):
            raise ValueError(
                f"RGB/depth frame count mismatch for {depth_item['name']}: "
                f"{rgb_item.get('frames')} vs {depth_item['frames']}"
            )
    expected = _expected_manifest(data_root, split_path, rgb_manifest, inventory, splits)
    final_manifest = output / "manifest.json"
    existing = _read_json(final_manifest)
    if existing is not None:
        if not existing.get("complete") or not _manifest_matches(existing, expected):
            raise ValueError(f"existing depth sidecar manifest is stale: {final_manifest}")
        if not (output / "depth.npy").is_file() or not (output / "episode_ends.npy").is_file():
            raise FileNotFoundError("complete depth manifest has missing arrays")
        return final_manifest

    total_frames = sum(int(item["frames"]) for item in inventory)
    required_bytes = total_frames * int(np.prod(DEPTH_SHAPE)) * np.dtype(np.uint16).itemsize + len(names) * 8
    state_path = output / "conversion_state.json"
    state = _read_json(state_path)
    if state is None and shutil.disk_usage(output).free < int(required_bytes * 1.01):
        raise OSError(f"insufficient space for depth sidecar: need {required_bytes / 2**30:.1f} GiB plus 1%")
    if state is None:
        if any((output / (name + ".partial")).exists() for name in DEPTH_ARRAY_SPECS):
            raise FileExistsError(f"partial depth files exist at {output}; use --resume")
        arrays = {
            "depth.npy": open_memmap(output / "depth.npy.partial", mode="w+", dtype=np.uint16, shape=(total_frames, *DEPTH_SHAPE)),
            "episode_ends.npy": open_memmap(output / "episode_ends.npy.partial", mode="w+", dtype=np.int64, shape=(len(names),)),
        }
        completed_names: list[str] = []
        _atomic_json(state_path, {"manifest": expected, "completed_names": completed_names})
    else:
        if not resume:
            raise FileExistsError(f"partial depth conversion exists at {output}; pass --resume")
        if not _manifest_matches(state.get("manifest", {}), expected):
            raise ValueError("partial depth conversion does not match current source/RGB Memmap")
        arrays = _open_partial(output)
        completed_names = list(state.get("completed_names", []))
        if names[: len(completed_names)] != completed_names:
            raise ValueError("partial depth conversion episode order changed")

    offsets = np.cumsum([0, *[int(item["frames"]) for item in inventory[:-1]]])
    completed = len(completed_names)
    initial = int(offsets[completed]) if completed < len(offsets) else total_frames
    bar = tqdm(total=total_frames, initial=initial, desc=f"acmt-act-depth/{data_root.name}", unit="frame", dynamic_ncols=True, disable=not progress)
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
                    arrays["depth.npy"][destination_start + start : destination_start + stop] = _read_depth_chunk(handle, start, stop)
                    bar.update(stop - start)
            arrays["episode_ends.npy"][episode_index] = destination_start + length
            for array in arrays.values():
                array.flush()
            completed_names.append(name)
            _atomic_json(state_path, {"manifest": expected, "completed_names": completed_names})
    finally:
        bar.close()

    if len(completed_names) != len(names):
        raise RuntimeError("depth conversion stopped before all episodes completed")
    for array in arrays.values():
        array.flush()
    for name in DEPTH_ARRAY_SPECS:
        os.replace(output / (name + ".partial"), output / name)
    _atomic_json(output / "episode_names.json", names)
    _atomic_json(output / "splits.json", {"splits": splits})
    final = dict(expected)
    final.update({"complete": True, "total_frames": total_frames, "episode_count": len(names), "created_s": time.perf_counter() - started})
    _atomic_json(final_manifest, final)
    state_path.unlink(missing_ok=True)
    return final_manifest


class ACMTActDepthMemmapStore:
    """Read-only aligned depth sidecar."""

    def __init__(self, root: str | os.PathLike[str], *, expected_frames: int | None = None, expected_episodes: int | None = None):
        self.root = Path(root).resolve()
        manifest = _read_json(self.root / "manifest.json")
        if not isinstance(manifest, dict) or manifest.get("depth_memmap_version") != DEPTH_MEMMAP_VERSION or not manifest.get("complete"):
            raise ValueError(f"invalid or incomplete ACMT-ACT depth sidecar: {self.root}")
        self.manifest = manifest
        self.depth = np.load(self.root / "depth.npy", mmap_mode="r")
        self.episode_ends = np.load(self.root / "episode_ends.npy", mmap_mode="r")
        if self.depth.dtype != np.uint16 or tuple(self.depth.shape[1:]) != DEPTH_SHAPE:
            raise ValueError(f"depth.npy must be [N,4,320,580] uint16, got {self.depth.shape}/{self.depth.dtype}")
        if self.episode_ends.dtype != np.int64 or self.episode_ends.ndim != 1:
            raise ValueError("episode_ends.npy must be int64 [E]")
        if expected_frames is not None and len(self.depth) != expected_frames:
            raise ValueError("depth and RGB frame counts do not match")
        if expected_episodes is not None and len(self.episode_ends) != expected_episodes:
            raise ValueError("depth and RGB episode counts do not match")

    def bounds(self, episode_index: int) -> tuple[int, int]:
        end = int(self.episode_ends[episode_index])
        start = 0 if episode_index == 0 else int(self.episode_ends[episode_index - 1])
        return start, end


__all__ = [
    "ACMTActDepthMemmapStore",
    "DEPTH_ARRAY_SPECS",
    "DEPTH_CROP_PARAMS",
    "DEPTH_MEMMAP_VERSION",
    "DEPTH_SHAPE",
    "convert_h5_to_depth_memmap",
]
