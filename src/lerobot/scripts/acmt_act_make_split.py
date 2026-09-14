"""Create the deterministic all-train split for a directory of ACMT H5 demos."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _atomic_write(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def make_all_train_split(data_dir: str | os.PathLike[str], output: str | os.PathLike[str], *, seed: int = 42) -> Path:
    root = Path(data_dir).resolve()
    destination = Path(output).resolve()
    names = sorted(path.name for path in root.glob("*.h5"))
    if not names:
        raise FileNotFoundError(f"no .h5 demos found in {root}")
    payload = {
        "schema": "acmt_act.split.v1",
        "seed": int(seed),
        "source_dir": str(root),
        "splits": {"train": names, "val": [], "test": []},
    }
    if destination.is_file():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        existing_splits = existing.get("splits", existing) if isinstance(existing, dict) else None
        if existing_splits != payload["splits"]:
            raise ValueError(f"existing split file does not match current H5 inventory: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(destination, payload)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    path = make_all_train_split(args.data_dir, args.output, seed=args.seed)
    names = json.loads(path.read_text(encoding="utf-8"))["splits"]["train"]
    print(f"all-train split ready: {path} ({len(names)} demos; val=0; test=0)")


if __name__ == "__main__":
    main()
