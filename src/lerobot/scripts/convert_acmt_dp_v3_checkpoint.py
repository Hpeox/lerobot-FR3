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
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from lerobot.policies.acmt_dp.configuration_acmt_dp_v3 import ACMTDPV3Config
from lerobot.policies.acmt_dp.modeling_acmt_dp_v3 import ACMTDPV3Policy
from lerobot.policies.acmt_dp.processor_acmt_dp_v3 import make_acmt_dp_v3_pre_post_processors
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

DEFAULT_POLICY_CHECKPOINTS = {
    "peg": Path("/data2/TactiGen/ACMT-DP-peg-runs"),
    "gear": Path("/cym/TactiGen/ACMT-DP/outputs/gear_big2small"),
}
DEFAULT_GENERATOR_CHECKPOINTS = {
    "peg": Path("/cym/TactiGen/ACMTv4/checkpoints/action_cmt_drifting_fz_xy_v2_seed42_e20/best.pt"),
    "gear": Path(
        "/cym/TactiGen/ACMTv4/checkpoints/action_cmt_drifting_fz_xy_v2_gear_big2small_seed42_e20/best.pt"
    ),
}
MODES = ("none", "real", "tactigen")
TASKS = ("peg", "gear")
V3_RUN_NAME = "v3_center480_cached_visual"


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
    required_temporal = {
        "tactile_encoder.side_attention.in_proj_weight",
        "tactile_encoder.temporal.weight_ih_l0",
        "tactile_encoder.temporal_norm.weight",
    }
    missing_temporal = sorted(required_temporal - set(state))
    if missing_temporal:
        raise ValueError(
            "This is an ACMT-DP v1 checkpoint without the v3 temporal tactile encoder; "
            f"retrain/reconvert from v3 best.pt (missing {missing_temporal})"
        )
    if any(name.startswith("tactile_encoder.attention.") for name in state):
        raise ValueError("Legacy single-frame ACMT-DP checkpoint rejected; use v3 best.pt")
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
    device: str = "cpu",
) -> ACMTDPV3Config:
    legacy_config = checkpoint.get("config")
    statistics = checkpoint.get("statistics")
    if not isinstance(legacy_config, Mapping) or not isinstance(statistics, Mapping):
        raise KeyError("Legacy policy checkpoint requires config and statistics")
    source_mode = legacy_config.get("tactile_source")
    expected_source_mode = "real" if mode == "tactigen" else mode
    if source_mode == "generated":
        raise ValueError(
            "Legacy generated policy checkpoints are deprecated; use the real "
            "policy checkpoint as the tactigen policy base"
        )
    if source_mode != expected_source_mode:
        raise ValueError(
            f"Checkpoint tactile_source={source_mode!r}, requested mode={mode!r}; "
            f"expected policy source={expected_source_mode!r}"
        )
    if int(legacy_config.get("obs_horizon", -1)) != 4:
        raise ValueError("Legacy checkpoint obs_horizon is not 4")
    if int(legacy_config.get("pred_horizon", -1)) != 16:
        raise ValueError("Legacy checkpoint pred_horizon is not 16")
    if int(legacy_config.get("action_dim", -1)) != 8:
        raise ValueError("Legacy checkpoint action_dim is not 8")
    if int(legacy_config.get("tactile_history", -1)) != 4:
        raise ValueError("ACMT-DP checkpoint tactile_history must be 4; use a v3 checkpoint")
    if legacy_config.get("visual_preprocess") != "center480":
        raise ValueError("ACMT-DP checkpoint visual_preprocess must be center480; use a v3 checkpoint")
    if int(legacy_config.get("action_execution_horizon", -1)) != 8:
        raise ValueError("ACMT-DP checkpoint action_execution_horizon must be 8")
    if float(legacy_config.get("control_hz", -1)) != 30.0:
        raise ValueError("ACMT-DP checkpoint control_hz must be 30")
    return ACMTDPV3Config(
        tactile_source=mode,
        checkpoint_tactile_source=expected_source_mode,
        task_variant=task,
        checkpoint_task_variant=task,
        diffusion_train_steps=int(legacy_config["diffusion_train_steps"]),
        diffusion_inference_steps=int(legacy_config["diffusion_inference_steps"]),
        unet_dims=tuple(legacy_config["unet_dims"]),
        tactile_history=4,
        action_execution_horizon=8,
        control_hz=30.0,
        visual_preprocess="center480",
        checkpoint_schema_version=3,
        visual_encoder_name="dformerv2",
        generator_model_config=generator_config,
        lowdim_mean=tuple(statistics["lowdim_mean"]),
        lowdim_std=tuple(statistics["lowdim_std"]),
        action_min=tuple(statistics["action_min"]),
        action_max=tuple(statistics["action_max"]),
        force_mean=tuple(statistics["force_mean"]),
        force_std=tuple(statistics["force_std"]),
        device=device,
        push_to_hub=False,
    )


def _strict_target_state(
    policy: ACMTDPV3Policy,
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
    device: str = "cpu",
) -> Path:
    source_checkpoint = source_checkpoint.resolve()
    source = _load_checkpoint(source_checkpoint)
    policy_state, ema_decay = _ema_policy_state(source)

    generator = None
    generator_state = None
    generator_config = None
    if mode == "tactigen":
        if generator_checkpoint is None:
            raise ValueError("tactigen conversion requires a TactiGen checkpoint")
        generator_checkpoint = generator_checkpoint.resolve()
        generator = _load_checkpoint(generator_checkpoint)
        generator_config = _generator_config(generator)
        generator_state = generator.get("model_state_dict")
        if not isinstance(generator_state, Mapping):
            raise KeyError("ACMT generator checkpoint requires model_state_dict")

    config = _make_config(task, mode, source, generator_config, device=device)
    policy = ACMTDPV3Policy(config)
    target_state = _strict_target_state(policy, policy_state, generator_state)

    output_dir = output_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    backup: Path | None = None
    try:
        policy.save_pretrained(temporary, state_dict=target_state)
        # PreTrainedConfig.from_pretrained discovers the registered subclass
        # through this discriminator. draccus does not serialize @property
        # ``type`` by itself.
        config_path = temporary / "config.json"
        serialized_config = json.loads(config_path.read_text(encoding="utf-8"))
        serialized_config["type"] = "acmt_dp_v3"
        # ``PreTrainedConfig`` may fall back to CPU when conversion runs on a
        # host without CUDA. Preserve the requested runtime device in the
        # portable artifact; loading will perform the normal availability
        # check on the deployment host.
        serialized_config["device"] = device
        config_path.write_text(json.dumps(serialized_config, indent=4) + "\n", encoding="utf-8")
        preprocessor, postprocessor = make_acmt_dp_v3_pre_post_processors(config)
        preprocessor.save_pretrained(temporary, config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json")
        postprocessor.save_pretrained(temporary, config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json")
        manifest = {
            "schema_version": 3,
            "checkpoint_schema": "v3_temporal_center480",
            "policy_type": "acmt_dp_v3",
            "task_variant": task,
            "tactile_source": mode,
            "policy_checkpoint_tactile_source": "real" if mode == "tactigen" else mode,
            "device": device,
            "protocol": "single_frozen_inference" if mode == "tactigen" else "direct_policy",
            "online_protocol": "16_predict_8_execute_at_30hz",
            "tactile_history": 4,
            "visual_preprocess": "center480",
            "diffusion_train_steps": config.diffusion_train_steps,
            "diffusion_inference_steps": config.diffusion_inference_steps,
            "action_execution_horizon": 8,
            "control_hz": 30.0,
            "seed": 42,
            # Keep provenance portable: the runtime artifact never needs the
            # machine-specific legacy source path (which may contain /cym).
            "source_checkpoint": source_checkpoint.name,
            "source_policy_tactile_source": source.get("config", {}).get("tactile_source"),
            "source_checkpoint_sha256": _sha256(source_checkpoint),
            "source_global_step": int(source.get("global_step", -1)),
            "source_best_val_loss": float(source.get("best_val_loss", float("nan"))),
            "ema": {"selected": True, "decay": ema_decay, "base": "model_state_dict"},
            "generator_checkpoint": generator_checkpoint.name if generator_checkpoint else None,
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
        if output_dir.exists():
            backup = output_dir.parent / f".{output_dir.name}.backup-{uuid.uuid4().hex}"
            output_dir.rename(backup)
        temporary.rename(output_dir)
        if backup is not None:
            shutil.rmtree(backup)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        if backup is not None and backup.exists() and not output_dir.exists():
            backup.rename(output_dir)
        raise
    return output_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=(*TASKS, "all"), default="all")
    parser.add_argument("--mode", choices=(*MODES, "all"), default="all")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/acmt_dp_v3"),
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
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Device serialized in the generated artifact (default: cpu)",
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
            source_mode = "real" if mode == "tactigen" else mode
            source = policy_root / source_mode / V3_RUN_NAME / "seed42" / "best.pt"
            output = args.output_root / task / mode / "seed42" / "pretrained_model"
            print(f"Converting {task}/{mode}: {source} -> {output}", flush=True)
            converted = convert_one(
                task=task,
                mode=mode,
                source_checkpoint=source,
                output_dir=output,
                generator_checkpoint=generator_checkpoint if mode == "tactigen" else None,
                device=args.device,
            )
            print(f"Saved {converted}", flush=True)
            gc.collect()


if __name__ == "__main__":
    main()
