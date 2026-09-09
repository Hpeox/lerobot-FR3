"""Prepare the small language/statistics sidecars used by ACMT-PI05.

The RGB, state and tactile arrays are intentionally not copied.  This command
only reads the original H5 episode attributes and the existing read-only
ACMT-ACT Memmap, then atomically publishes JSON files next to that Memmap.
Training never opens H5 files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np
from tqdm.auto import tqdm

from lerobot.datasets.acmt_act_memmap import ACMTActMemmapStore


SCHEMA = "acmt_pi05.episode_instructions.v1"
STATS_SCHEMA = "acmt_pi05.stats.v1"


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _json_hash(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _split_names(root: Path, names: list[str]) -> dict[str, list[str]]:
    payload = json.loads((root / "splits.json").read_text(encoding="utf-8"))
    splits = payload.get("splits", payload)
    if not isinstance(splits, dict):
        raise ValueError(f"invalid splits.json: {root / 'splits.json'}")
    known = set(names)
    result = {key: [Path(str(name)).name for name in splits.get(key, [])] for key in ("train", "val", "test")}
    for key, values in result.items():
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"split {key} contains names absent from episode_names.json: {unknown[:3]}")
    if len(set().union(*[set(values) for values in result.values()])) != len(names):
        raise ValueError("train/val/test do not cover every Memmap episode exactly once")
    if sum(len(values) for values in result.values()) != len(names):
        raise ValueError("train/val/test contain duplicate episodes")
    return result


def _stats(values: np.ndarray) -> dict[str, object]:
    values = np.asarray(values, dtype=np.float64)
    flat = values.reshape(-1, values.shape[-1])
    return {
        "count": int(flat.shape[0]),
        "mean": flat.mean(0).astype(np.float32).tolist(),
        "std": np.maximum(flat.std(0), 1e-6).astype(np.float32).tolist(),
        "min": flat.min(0).astype(np.float32).tolist(),
        "q01": np.quantile(flat, 0.01, axis=0).astype(np.float32).tolist(),
        "q10": np.quantile(flat, 0.10, axis=0).astype(np.float32).tolist(),
        "q50": np.quantile(flat, 0.50, axis=0).astype(np.float32).tolist(),
        "q90": np.quantile(flat, 0.90, axis=0).astype(np.float32).tolist(),
        "q99": np.quantile(flat, 0.99, axis=0).astype(np.float32).tolist(),
        "max": flat.max(0).astype(np.float32).tolist(),
    }


def prepare_sidecars(
    data_dir: str | os.PathLike[str],
    memmap_dir: str | os.PathLike[str],
    *,
    force: bool = False,
    progress: bool = True,
) -> tuple[Path, Path]:
    """Extract instructions and train-only PI05 normalization statistics."""

    source = Path(data_dir).resolve()
    root = Path(memmap_dir).resolve()
    store = ACMTActMemmapStore(root)
    names = [Path(str(name)).name for name in store.episode_names]
    splits = _split_names(root, names)
    inventory = []
    for name in names:
        path = source / name
        if not path.is_file():
            raise FileNotFoundError(path)
        stat = path.stat()
        inventory.append({"name": name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    source_hash = _json_hash(inventory)

    instruction_path = root / "episode_instructions.json"
    expected_instruction_header = {
        "schema": SCHEMA,
        "source_dir": str(source),
        "source_inventory_sha256": source_hash,
        "episode_count": len(names),
    }
    existing = None
    if instruction_path.is_file() and not force:
        try:
            existing = json.loads(instruction_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = None
    if not isinstance(existing, dict) or any(existing.get(key) != value for key, value in expected_instruction_header.items()):
        records: dict[str, dict[str, str]] = {}
        iterator = tqdm(names, desc="acmt-pi05/language", unit="demo", disable=not progress)
        for name in iterator:
            with h5py.File(source / name, "r") as handle:
                instruction = handle.attrs.get("language_instruction")
                task_name = handle.attrs.get("task_name", "")
            if isinstance(instruction, bytes):
                instruction = instruction.decode("utf-8", errors="replace")
            if instruction is None or not str(instruction).strip():
                raise ValueError(f"{name} has no non-empty language_instruction root attribute")
            if isinstance(task_name, bytes):
                task_name = task_name.decode("utf-8", errors="replace")
            records[name] = {
                "language_instruction": str(instruction),
                "task_name": str(task_name),
                "source_h5": str(source / name),
            }
        payload = dict(expected_instruction_header)
        payload["episodes"] = records
        _atomic_json(instruction_path, payload)

    # PI05 relative-action statistics are computed only from train episodes.
    # The action array stores Gello wire gripper values (1=open/0=closed).
    # Expose the physical convention (0=open/1=closed) before statistics.
    stats_path = root / "acmt_pi05_stats.json"
    if stats_path.is_file() and not force:
        try:
            current = json.loads(stats_path.read_text(encoding="utf-8"))
            if (
                isinstance(current, dict)
                and current.get("schema") == STATS_SCHEMA
                and current.get("source_memmap") == str(root)
                and current.get("source_inventory_sha256") == source_hash
                and current.get("split") == "train"
            ):
                return instruction_path, stats_path
        except (OSError, json.JSONDecodeError):
            pass

    name_to_index = {name: index for index, name in enumerate(names)}
    train_indices = [name_to_index[name] for name in splits["train"]]
    states: list[np.ndarray] = []
    absolute_actions: list[np.ndarray] = []
    relative_actions: list[np.ndarray] = []
    tactile0: list[np.ndarray] = []
    tactile1: list[np.ndarray] = []
    windows = 0
    iterator = tqdm(train_indices, desc="acmt-pi05/stats", unit="demo", disable=not progress)
    for episode_index in iterator:
        start, end = store.bounds(episode_index)
        state = np.asarray(store.state[start:end], dtype=np.float32)
        action = np.asarray(store.action[start:end], dtype=np.float32).copy()
        action[:, 7] = 1.0 - action[:, 7]
        force = np.asarray(store.tactile[start:end], dtype=np.float32)
        states.append(state)
        absolute_actions.append(action)
        tactile0.append(force[:, 0].reshape(-1, 3))
        tactile1.append(force[:, 1].reshape(-1, 3))
        length = end - start
        # A PI05 target is a fixed 50-step window.  At an episode tail the
        # last recorded command is repeated, exactly as the dataset returns it.
        for offset in range(length):
            indices = np.minimum(np.arange(offset, offset + 50), length - 1)
            target = action[indices].copy()
            target[:, :7] -= state[offset, :7]
            relative_actions.append(target)
            windows += 1

    state_values = np.concatenate(states, axis=0)
    abs_values = np.concatenate(absolute_actions, axis=0)
    rel_values = np.concatenate(relative_actions, axis=0)
    payload = {
        "schema": STATS_SCHEMA,
        "source_memmap": str(root),
        "source_inventory_sha256": source_hash,
        "split": "train",
        "camera_order": ["top", "side", "wrist_left", "wrist_right"],
        "counts": {
            "train_episodes": len(train_indices),
            "train_frames": int(state_values.shape[0]),
            "relative_windows": int(windows * 50),
        },
        "state": _stats(state_values),
        "action_absolute_physical": _stats(abs_values),
        "action_relative": _stats(rel_values),
        "tactile_sensor0": _stats(np.concatenate(tactile0, axis=0)),
        "tactile_sensor1": _stats(np.concatenate(tactile1, axis=0)),
    }
    _atomic_json(stats_path, payload)
    return instruction_path, stats_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--memmap-dir", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    started = time.perf_counter()
    instruction_path, stats_path = prepare_sidecars(
        args.data_dir, args.memmap_dir, force=args.force, progress=args.progress
    )
    print(
        json.dumps(
            {
                "episode_instructions": str(instruction_path),
                "acmt_pi05_stats": str(stats_path),
                "elapsed_s": round(time.perf_counter() - started, 3),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
