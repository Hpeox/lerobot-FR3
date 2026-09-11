#!/usr/bin/env python

"""Load an ACMT-PI05 checkpoint with the staged Selective INT8 adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lerobot.configs import PreTrainedConfig

from .configuration_acmt_pi05 import ACMTPi05Config
from .modeling_acmt_pi05 import ACMTPi05Policy


STAGE_TO_MODULES = {
    "v1": ("language",),
    "v2": ("language", "action"),
    "v3": ("language", "action", "vision"),
}


def load_acmt_pi05_int8(
    checkpoint: str | Path,
    *,
    stage: str = "v1",
    device: str = "cuda",
) -> ACMTPi05Policy:
    """Strict-load ``checkpoint`` and apply the requested CPU-first INT8 stage."""

    if stage not in STAGE_TO_MODULES:
        raise ValueError(f"stage must be one of {tuple(STAGE_TO_MODULES)}, got {stage!r}")
    if not str(device).startswith("cuda"):
        raise ValueError("ACMT-PI05 Selective INT8 requires a CUDA device")

    config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    if not isinstance(config, ACMTPi05Config):
        raise TypeError(f"expected ACMTPi05 config, got {type(config).__name__}")
    config.device = device
    config.dtype = "float16"
    config.quantization_backend = "bitsandbytes_int8"
    config.quantization_stages = STAGE_TO_MODULES[stage]

    return ACMTPi05Policy.from_pretrained(
        checkpoint,
        config=config,
        local_files_only=True,
        strict=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--stage", choices=tuple(STAGE_TO_MODULES), default="v1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    policy = load_acmt_pi05_int8(args.checkpoint, stage=args.stage, device=args.device)
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "stage": args.stage,
        "device": args.device,
        "dtype": policy.config.dtype,
        "quantization": getattr(policy, "quantization_report", None),
    }
    print(json.dumps(report, indent=2))
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()


__all__ = ["STAGE_TO_MODULES", "load_acmt_pi05_int8"]
