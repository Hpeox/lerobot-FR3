"""Build corrected ACMT-ACT goal labels and residual-action statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lerobot.datasets.acmt_act_memmap import build_acmt_act_policy_stats, build_acmt_act_targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--memmap-dir", required=True)
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    targets = build_acmt_act_targets(
        args.data_dir,
        args.memmap_dir,
        split_file=args.split_file,
        force=args.force,
    )
    stats = build_acmt_act_policy_stats(args.memmap_dir, force=args.force)
    print(json.dumps({"targets": str(Path(targets)), "policy_stats": str(Path(stats))}, indent=2))


if __name__ == "__main__":
    main()
