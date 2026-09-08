"""Download DINOv2-S/14 once for offline ACMT-ACTv2 training.

The resulting file is a training input only; policy checkpoints embed the
weights and deployment does not read this file.  This command is intentionally
separate from model construction so a rollout process never downloads weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from lerobot.policies.acmt_actv2.configuration_acmt_actv2 import DINOV2_MODEL_NAME


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", default=DINOV2_MODEL_NAME)
    parser.add_argument("--image-size", nargs=2, type=int, default=(336, 448), metavar=("H", "W"))
    args = parser.parse_args()
    try:
        import timm
    except ImportError as exc:  # pragma: no cover - installation-specific
        raise SystemExit("Install the optional ACMT-ACT extra before downloading DINOv2.") from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = timm.create_model(
        args.model_name,
        pretrained=True,
        num_classes=0,
        img_size=tuple(args.image_size),
        dynamic_img_size=True,
    )
    payload = {"state_dict": model.state_dict(), "model_name": args.model_name}
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    torch.save(payload, temporary)
    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
    temporary.replace(args.output)
    manifest = {
        "model_name": args.model_name,
        "image_size": list(args.image_size),
        "sha256": digest,
        "path": str(args.output.resolve()),
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
