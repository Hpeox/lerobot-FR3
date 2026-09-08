"""Benchmark the native ACMT-ACTv2 inference stages.

The benchmark uses one synthetic, correctly shaped FR3 observation and never
opens a dataset or sends a robot command.  It is intended for measuring the
same preprocessor, shared DINOv2, ACT decoder, temporal ensemble and
postprocessor used by rollout.  A policy checkpoint is required so the
benchmark includes the real embedded DINO weights.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.configs import PreTrainedConfig
from lerobot.policies.acmt_act.configuration_acmt_act import (
    DQ,
    FT300,
    GRIPPER_GPO,
    O_T_EE,
    TAU_J,
    XENSE0,
    XENSE1,
    depth_key,
    rgb_key,
)
from lerobot.policies.acmt_actv2.configuration_acmt_actv2 import ACMTACTV2Config
from lerobot.policies.acmt_actv2.modeling_acmt_actv2 import ACMTACTV2Policy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import OBS_STATE


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _percentiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
        "mean_ms": float(array.mean()),
        "max_ms": float(array.max()),
        "theoretical_hz_p99": float(1000.0 / max(np.percentile(array, 99), 1e-9)),
    }


def _raw_observation(config: ACMTACTV2Config, device: torch.device) -> dict[str, Any]:
    """Make a B=1 raw observation in the same ABI as FR3 rollout."""

    image = np.zeros((1, 480, 640, 3), dtype=np.uint8)
    image[..., 0] = 96
    image[..., 1] = 128
    image[..., 2] = 160
    raw: dict[str, Any] = {OBS_STATE: np.zeros((1, 8), dtype=np.float32)}
    for source_key in config.source_camera_keys:
        raw[rgb_key(source_key)] = image.copy()
    raw[XENSE0] = np.zeros((1, 3, 35, 20), dtype=np.float32)
    raw[XENSE1] = np.zeros((1, 3, 35, 20), dtype=np.float32)

    if config.tactile_source == "substitution":
        for source_key in config.source_camera_keys:
            raw[depth_key(source_key)] = np.zeros((1, 480, 640), dtype=np.uint16)
        raw[DQ] = np.zeros((1, 7), dtype=np.float32)
        raw[TAU_J] = np.zeros((1, 7), dtype=np.float32)
        raw[FT300] = np.zeros((1, 6), dtype=np.float32)
        raw[O_T_EE] = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0)
        raw[GRIPPER_GPO] = np.zeros((1, 1), dtype=np.float32)
    return raw


def _time_call(fn, device: torch.device):
    _sync(device)
    start = time.perf_counter()
    value = fn()
    _sync(device)
    return value, (time.perf_counter() - start) * 1000.0


def run(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = Path(args.policy).expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"policy checkpoint directory not found: {checkpoint}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device=cuda was requested but CUDA is unavailable")

    config = PreTrainedConfig.from_pretrained(
        checkpoint,
        cli_overrides=[f"--device={device}", "--dinov2_pretrained=false", "--require_dinov2_checkpoint=false"],
    )
    if not isinstance(config, ACMTACTV2Config):
        raise TypeError(f"expected an ACMT-ACTv2 checkpoint, got {type(config).__name__}")
    policy = ACMTACTV2Policy.from_pretrained(checkpoint, config=config)
    policy.to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(checkpoint),
    )
    raw = _raw_observation(policy.config, device)

    # Warm up every stage, including the actual model's CUDA kernels.
    with torch.inference_mode():
        for _ in range(args.warmup):
            processed = preprocessor(_raw_observation(policy.config, device))
            window = policy.observe(processed)
            model_batch = policy._model_batch(window)
            policy.model(model_batch)
            policy.temporal_ensembler.reset()

    timings: dict[str, list[float]] = {name: [] for name in ("preprocess", "dino", "act", "ensemble", "postprocess", "total")}
    substitution_timings: list[float] = []
    model = policy.model
    original_dino_tokens = model._dino_tokens

    with torch.inference_mode():
        for _ in range(args.iterations):
            total_start = time.perf_counter()
            processed, elapsed = _time_call(
                lambda: preprocessor(_raw_observation(policy.config, device)), device
            )
            timings["preprocess"].append(elapsed)
            window = policy.observe(processed)
            model_batch = policy._model_batch(window)

            _, elapsed = _time_call(lambda: model._dino_tokens(model_batch["observation.images"]), device)
            timings["dino"].append(elapsed)
            visual_tokens, visual_pos = model._dino_tokens(model_batch["observation.images"])
            model._dino_tokens = lambda _images, vt=visual_tokens, vp=visual_pos: (vt, vp)
            try:
                actions, _ = _time_call(lambda: model(model_batch), device)
            finally:
                model._dino_tokens = original_dino_tokens
            timings["act"].append(elapsed)
            action, elapsed = _time_call(lambda: policy.temporal_ensembler.update(actions), device)
            timings["ensemble"].append(elapsed)
            post_action, elapsed = _time_call(lambda: postprocessor(action[:, :1]), device)
            timings["postprocess"].append(elapsed)

            if policy.config.tactile_source == "substitution":
                _, elapsed = _time_call(
                    lambda: policy.notify_action_executed(post_action, processed), device
                )
                substitution_timings.append(elapsed)
            _sync(device)
            timings["total"].append((time.perf_counter() - total_start) * 1000.0)

    report: dict[str, Any] = {
        "policy": str(checkpoint),
        "schema": [policy.config.checkpoint_schema, policy.config.checkpoint_schema_version],
        "device": str(device),
        "iterations": args.iterations,
        "warmup": args.warmup,
        "chunk_size": policy.config.chunk_size,
        "n_action_steps": policy.config.n_action_steps,
        "temporal_ensemble_coeff": policy.config.temporal_ensemble_coeff,
        "stages": {name: _percentiles(values) for name, values in timings.items()},
    }
    if substitution_timings:
        report["substitution_acmt_generation"] = _percentiles(substitution_timings)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, help="Saved ACMT-ACTv2 pretrained_model directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report path")
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0:
        parser.error("--iterations must be positive and --warmup must be non-negative")
    report = run(args)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".partial")
        temporary.write_text(rendered + "\n", encoding="utf-8")
        temporary.replace(args.output)


if __name__ == "__main__":
    main()
