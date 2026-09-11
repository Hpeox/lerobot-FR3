#!/usr/bin/env python

"""Offline CUDA benchmark for an ACMT-PI05 Selective INT8 policy."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch

from .load_acmt_pi05_int8 import load_acmt_pi05_int8


def _dummy_batch(device: str) -> dict[str, torch.Tensor]:
    return {
        **{
            f"observation.images.camera.cam{i}.rgb": torch.zeros(
                1, 3, 320, 580, dtype=torch.float32, device=device
            )
            for i in range(1, 5)
        },
        "observation.state": torch.zeros(1, 8, dtype=torch.float32, device=device),
        "observation.language.tokens": torch.ones(1, 200, dtype=torch.long, device=device),
        "observation.language.attention_mask": torch.ones(1, 200, dtype=torch.bool, device=device),
    }


def benchmark(checkpoint: str | Path, *, stage: str = "v1", warmup: int = 5, runs: int = 50) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the ACMT-PI05 INT8 benchmark")
    start = time.perf_counter()
    policy = load_acmt_pi05_int8(checkpoint, stage=stage, device="cuda")
    load_time_s = time.perf_counter() - start
    policy.eval()
    batch = _dummy_batch("cuda")

    with torch.no_grad():
        for _ in range(warmup):
            policy.predict_action_chunk(batch)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        times: list[float] = []
        output = None
        for _ in range(runs):
            torch.cuda.synchronize()
            tick = time.perf_counter()
            output = policy.predict_action_chunk(batch)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - tick)

    assert output is not None
    ordered = sorted(times)
    report = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "stage": stage,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "load_time_s": load_time_s,
        "warmup": warmup,
        "runs": runs,
        "mean_s": statistics.mean(times),
        "p50_s": ordered[len(ordered) // 2],
        "p95_s": ordered[min(len(ordered) - 1, max(0, math.ceil(len(ordered) * 0.95) - 1))],
        "max_s": max(times),
        "output_shape": list(output.shape),
        "output_dtype": str(output.dtype),
        "output_finite": bool(torch.isfinite(output).all()),
        "output_min": float(output.min()),
        "output_max": float(output.max()),
        "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
        "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
        "peak_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "quantization": getattr(policy, "quantization_report", None),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--stage", choices=("v1", "v2", "v3"), default="v1")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    report = benchmark(args.checkpoint, stage=args.stage, warmup=args.warmup, runs=args.runs)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()


__all__ = ["benchmark"]
