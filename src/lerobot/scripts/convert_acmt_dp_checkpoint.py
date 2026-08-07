#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert the six legacy ACMT-DP checkpoints to native LeRobot directories.

The converter reads tensors and metadata only. It deliberately does not import the
legacy ACMT-DP, ACMTv4, or DFormer Python packages.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from lerobot.policies.acmt_dp.configuration_acmt_dp import ACMTDPConfig
from lerobot.policies.acmt_dp.modeling_acmt_dp import ACMTDPPolicy
from lerobot.policies.acmt_dp.processor_acmt_dp import make_acmt_dp_pre_post_processors
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

DEFAULT_POLICY_CHECKPOINTS = {
    "peg": Path("/data2/TactiGen/ACMT-DP-peg-runs/outputs"),
    "gear": Path("/data2/TactiGen/ACMT-DP-gear-runs/outputs"),
}
DEFAULT_GENERATOR_CHECKPOINTS = {
    "peg": Path("/cym/TactiGen/ACMTv4/checkpoints/action_cmt_drifting_fz_xy_v2_seed42_e20/best.pt"),
    "gear": Path(
        "/cym/TactiGen/ACMTv4/checkpoints/action_cmt_drifting_fz_xy_v2_gear_big2small_seed42_e20/best.pt"
    ),
}
MODES = ("none", "real", "generated")
TASKS = ("peg", "gear")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint {path} must contain a dictionary")
    return checkpoint


def _ema_policy_state(checkpoint: Mapping[str, Any]) -> tuple[dict[str, Tensor], float]:
    raw = checkpoint.get("model_state_dict")
    ema = checkpoint.get("ema_state_dict")
    if not isinstance(raw, Mapping) or not isinstance(ema, Mapping):
        raise KeyError("Legacy policy checkpoint requires model_state_dict and ema_state_dict")
    shadow = ema.get("shadow")
    if not isinstance(shadow, Mapping):
        raise KeyError("Legacy policy checkpoint requires ema_state_dict.shadow")
    state = dict(raw)
    unknown = sorted(set(shadow) - set(state))
    if unknown:
        raise KeyError(f"EMA contains keys absent from the full model state: {unknown[:10]}")
    for name, value in shadow.items():
        if not isinstance(value, Tensor) or not torch.is_floating_point(value):
            raise TypeError(f"EMA shadow {name!r} is not a floating-point tensor")
        state[name] = value
    return state, float(ema["decay"])


def _generator_config(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    config = checkpoint.get("model_config")
    if not isinstance(config, Mapping):
        raise KeyError("ACMT generator checkpoint requires model_config")
    result = dict(config)
    loss_weights = checkpoint.get("loss_weights")
    if isinstance(loss_weights, Mapping) and "contact_pos_weight" in loss_weights:
        result["contact_pos_weight"] = float(loss_weights["contact_pos_weight"])
    # These legacy source locations are constructor-only hints and must never enter a
    # runtime checkpoint. The native DFormer implementation is bundled with LeRobot.
    result.pop("dformerv2_repo_path", None)
    result.pop("dformerv2_checkpoint", None)
    return result


def _make_config(
    task: str,
    mode: str,
    checkpoint: Mapping[str, Any],
    generator_config: dict[str, Any] | None,
) -> ACMTDPConfig:
    legacy_config = checkpoint.get("config")
    statistics = checkpoint.get("statistics")
    if not isinstance(legacy_config, Mapping) or not isinstance(statistics, Mapping):
        raise KeyError("Legacy policy checkpoint requires config and statistics")
    source_mode = legacy_config.get("tactile_source")
    if source_mode != mode:
        raise ValueError(f"Checkpoint tactile_source={source_mode!r}, requested mode={mode!r}")
    if int(legacy_config.get("obs_horizon", -1)) != 4:
        raise ValueError("Legacy checkpoint obs_horizon is not 4")
    if int(legacy_config.get("pred_horizon", -1)) != 16:
        raise ValueError("Legacy checkpoint pred_horizon is not 16")
    if int(legacy_config.get("action_dim", -1)) != 8:
        raise ValueError("Legacy checkpoint action_dim is not 8")
    return ACMTDPConfig(
        tactile_source=mode,
        checkpoint_tactile_source=mode,
        task_variant=task,
        checkpoint_task_variant=task,
        diffusion_train_steps=int(legacy_config["diffusion_train_steps"]),
        diffusion_inference_steps=int(legacy_config["diffusion_inference_steps"]),
        unet_dims=tuple(legacy_config["unet_dims"]),
        visual_encoder_name="dformerv2",
        generator_model_config=generator_config,
        lowdim_mean=tuple(statistics["lowdim_mean"]),
        lowdim_std=tuple(statistics["lowdim_std"]),
        action_min=tuple(statistics["action_min"]),
        action_max=tuple(statistics["action_max"]),
        force_mean=tuple(statistics["force_mean"]),
        force_std=tuple(statistics["force_std"]),
        device="cpu",
        push_to_hub=False,
    )


def _strict_target_state(
    policy: ACMTDPPolicy,
    policy_state: Mapping[str, Tensor],
    generator_state: Mapping[str, Tensor] | None,
) -> dict[str, Tensor]:
    target = dict(policy_state)
    if generator_state is not None:
        target.update({f"tactile_generator.{name}": value for name, value in generator_state.items()})
    expected = policy.state_dict()
    missing = sorted(set(expected) - set(target))
    unexpected = sorted(set(target) - set(expected))
    shape_mismatch = sorted(
        name for name in set(expected) & set(target) if expected[name].shape != target[name].shape
    )
    if missing or unexpected or shape_mismatch:
        raise RuntimeError(
            "Strict ACMT-DP state mapping failed: "
            f"missing={missing[:20]}, unexpected={unexpected[:20]}, "
            f"shape_mismatch={shape_mismatch[:20]}"
        )
    policy.load_state_dict(target, strict=True)
    # Use the model-owned tensors after strict loading. Some tensors in a legacy
    # torch.save archive are storage views; safetensors correctly refuses those
    # because persisting the backing storage would be ambiguous.
    return dict(policy.state_dict())


def convert_one(
    *,
    task: str,
    mode: str,
    source_checkpoint: Path,
    output_dir: Path,
    generator_checkpoint: Path | None,
) -> Path:
    source_checkpoint = source_checkpoint.resolve()
    source = _load_checkpoint(source_checkpoint)
    policy_state, ema_decay = _ema_policy_state(source)

    generator = None
    generator_state = None
    generator_config = None
    if mode == "generated":
        if generator_checkpoint is None:
            raise ValueError("generated conversion requires a generator checkpoint")
        generator_checkpoint = generator_checkpoint.resolve()
        generator = _load_checkpoint(generator_checkpoint)
        generator_config = _generator_config(generator)
        generator_state = generator.get("model_state_dict")
        if not isinstance(generator_state, Mapping):
            raise KeyError("ACMT generator checkpoint requires model_state_dict")

    config = _make_config(task, mode, source, generator_config)
    policy = ACMTDPPolicy(config)
    target_state = _strict_target_state(policy, policy_state, generator_state)

    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        policy.save_pretrained(temporary, state_dict=target_state)
        preprocessor, postprocessor = make_acmt_dp_pre_post_processors(config)
        preprocessor.save_pretrained(temporary, config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json")
        postprocessor.save_pretrained(temporary, config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json")
        manifest = {
            "schema_version": 1,
            "policy_type": "acmt_dp",
            "task_variant": task,
            "tactile_source": mode,
            "seed": 42,
            "source_checkpoint": str(source_checkpoint),
            "source_checkpoint_sha256": _sha256(source_checkpoint),
            "source_global_step": int(source.get("global_step", -1)),
            "source_best_val_loss": float(source.get("best_val_loss", float("nan"))),
            "ema": {"selected": True, "decay": ema_decay, "base": "model_state_dict"},
            "generator_checkpoint": str(generator_checkpoint) if generator_checkpoint else None,
            "generator_checkpoint_sha256": (
                _sha256(generator_checkpoint) if generator_checkpoint is not None else None
            ),
            "generator_global_step": int(generator.get("global_step", -1)) if generator else None,
            "generator_epoch": int(generator.get("epoch", -1)) if generator else None,
            "runtime_external_python_dependencies": [],
        }
        (temporary / "conversion_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.rename(output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=(*TASKS, "all"), default="all")
    parser.add_argument("--mode", choices=(*MODES, "all"), default="all")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/acmt_dp"),
        help="Output root containing task/mode/seed42/pretrained_model",
    )
    parser.add_argument(
        "--policy-source-root",
        type=Path,
        help="Override the selected task's legacy outputs root (single task only)",
    )
    parser.add_argument(
        "--generator-checkpoint",
        type=Path,
        help="Override the selected task's full generator checkpoint (single task only)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    tasks = TASKS if args.task == "all" else (args.task,)
    modes = MODES if args.mode == "all" else (args.mode,)
    if len(tasks) != 1 and (args.policy_source_root or args.generator_checkpoint):
        raise ValueError("Source overrides require a single --task")
    for task in tasks:
        policy_root = args.policy_source_root or DEFAULT_POLICY_CHECKPOINTS[task]
        generator_checkpoint = args.generator_checkpoint or DEFAULT_GENERATOR_CHECKPOINTS[task]
        for mode in modes:
            source = policy_root / mode / "seed42" / "best.pt"
            output = args.output_root / task / mode / "seed42" / "pretrained_model"
            print(f"Converting {task}/{mode}: {source} -> {output}", flush=True)
            converted = convert_one(
                task=task,
                mode=mode,
                source_checkpoint=source,
                output_dir=output,
                generator_checkpoint=generator_checkpoint if mode == "generated" else None,
            )
            print(f"Saved {converted}", flush=True)
            gc.collect()


if __name__ == "__main__":
    main()
