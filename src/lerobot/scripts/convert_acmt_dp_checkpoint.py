#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert Native-DP v4 scratch checkpoints to LeRobot format.

The converter intentionally accepts only ``acmt_dp.native_dp_v4`` scratch
checkpoints.  Training-only paths and caches are ignored; runtime artifacts
contain no dependency on the ACMT-DP source tree.
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

from lerobot.policies.acmt_dp.configuration_acmt_dp import ACMTDPConfig
from lerobot.policies.acmt_dp.modeling_acmt_dp import ACMTDPPolicy
from lerobot.policies.acmt_dp.processor_acmt_dp import make_acmt_dp_pre_post_processors
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

MODES = ("none", "real", "tactigen")
TASKS = ("peg", "gear")
UPSTREAM_COMMIT = "770a30f6941bf0d9d096fdc2025bd486b7248b23"
DEFAULT_POLICY_ROOTS = {
    "peg": Path("/data2/cym/16mm_peg_in_hole/native_dp_v4"),
    "gear": Path("/data2/cym/gear_big2small/native_dp_v4"),
}
DEFAULT_GENERATOR_CHECKPOINTS = {
    "peg": Path("/cym/TactiGen/ACMTv4/checkpoints/action_cmt_drifting_fz_xy_v2_seed42_e20/best.pt"),
    "gear": Path(
        "/cym/TactiGen/ACMTv4/checkpoints/action_cmt_drifting_fz_xy_v2_gear_big2small_seed42_e20/best.pt"
    ),
}


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


def _validate_v4_scratch(checkpoint: Mapping[str, Any], path: Path) -> None:
    if checkpoint.get("schema") != "acmt_dp.native_dp_v4":
        raise ValueError(
            f"{path} is not a Native-DP v4 checkpoint (schema={checkpoint.get('schema')!r}); "
            "v3/legacy checkpoints must be reconverted from v4"
        )
    if checkpoint.get("stage") != "scratch":
        raise ValueError(
            f"{path} has stage={checkpoint.get('stage')!r}; only scratch checkpoints are supported "
            "(frozen/finetune artifacts are rejected)"
        )
    config = checkpoint.get("config")
    if not isinstance(config, Mapping):
        raise KeyError("v4 checkpoint requires config")
    for field, expected in (
        ("obs_horizon", 4),
        ("pred_horizon", 16),
        ("action_execution_horizon", 8),
        ("state_dim", 8),
        ("action_dim", 8),
        ("tactile_dim", 160),
        ("feature_dim", 512),
        ("unet_kernel_size", 5),
        ("diffusion_step_embed_dim", 128),
    ):
        if int(config.get(field, -1)) != expected:
            raise ValueError(f"v4 checkpoint config {field} must be {expected}")
    if float(config.get("control_hz", -1)) != 30.0:
        raise ValueError("v4 checkpoint config control_hz must be 30")
    if tuple(config.get("camera_names", ())) != ("top", "side", "wrist_left", "wrist_right"):
        raise ValueError("v4 checkpoint camera_names must be top, side, wrist_left, wrist_right")
    if config.get("vision_mode") != "scratch":
        raise ValueError("only vision_mode='scratch' checkpoints can be converted")
    if config.get("vision_weights") not in (None, "none", "NONE"):
        raise ValueError("scratch v4 checkpoint must not contain vision_weights")
    if not isinstance(checkpoint.get("statistics"), Mapping):
        raise KeyError("v4 checkpoint requires statistics")
    if not isinstance(checkpoint.get("model_state_dict"), Mapping):
        raise KeyError("v4 checkpoint requires model_state_dict")
    if not isinstance(checkpoint.get("ema_state_dict"), Mapping):
        raise KeyError("v4 checkpoint requires ema_state_dict")


def _ema_policy_state(checkpoint: Mapping[str, Any]) -> tuple[dict[str, Tensor], float]:
    raw = checkpoint["model_state_dict"]
    ema = checkpoint["ema_state_dict"]
    shadow = ema.get("shadow")
    if not isinstance(raw, Mapping) or not isinstance(shadow, Mapping):
        raise KeyError("v4 checkpoint requires model_state_dict and ema_state_dict.shadow")
    state = dict(raw)
    unknown = sorted(set(shadow) - set(state))
    if unknown:
        raise KeyError(f"EMA contains keys absent from model_state_dict: {unknown[:10]}")
    for name, value in shadow.items():
        if not isinstance(value, Tensor) or not torch.is_floating_point(value):
            raise TypeError(f"EMA shadow {name!r} must be a floating-point tensor")
        state[name] = value
    required = {
        "visual_encoder.obs_encoder.key_model_map.rgb.conv1.weight",
        "tactile_encoder.spatial.0.weight",
        "normalizer.params_dict.state.offset",
    }
    missing = sorted(required - set(state))
    if missing:
        raise ValueError(f"v4 checkpoint is missing required state keys: {missing}")
    if any("temporal" in name or "side_attention" in name for name in state):
        raise ValueError("v3 temporal tactile state is incompatible with Native-DP v4")
    return state, float(ema.get("decay", 0.9999))


def _generator_config(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    config = checkpoint.get("model_config")
    if not isinstance(config, Mapping):
        raise KeyError("TactiGen checkpoint requires model_config")
    result = dict(config)
    loss_weights = checkpoint.get("loss_weights")
    if isinstance(loss_weights, Mapping) and "contact_pos_weight" in loss_weights:
        result["contact_pos_weight"] = float(loss_weights["contact_pos_weight"])
    result.pop("dformerv2_repo_path", None)
    result.pop("dformerv2_checkpoint", None)
    return result


def _make_config(
    task: str, mode: str, checkpoint: Mapping[str, Any], generator_config: dict[str, Any] | None
) -> ACMTDPConfig:
    source_config = checkpoint["config"]
    statistics = checkpoint["statistics"]
    source_mode = source_config.get("tactile_source")
    expected_source = "real" if mode == "tactigen" else mode
    if source_mode != expected_source:
        raise ValueError(
            f"checkpoint tactile_source={source_mode!r}, requested mode={mode!r}; "
            f"expected source={expected_source!r}"
        )
    return ACMTDPConfig(
        tactile_source=mode,
        checkpoint_tactile_source=expected_source,
        task_variant=task,
        checkpoint_task_variant=task,
        checkpoint_schema_version=4,
        checkpoint_schema="acmt_dp.native_dp_v4",
        vision_mode="scratch",
        visual_preprocess="resize256_center224_imagenet",
        diffusion_train_steps=int(source_config["diffusion_train_steps"]),
        diffusion_inference_steps=int(source_config.get("diffusion_inference_steps", 8)),
        unet_dims=tuple(source_config["unet_dims"]),
        unet_kernel_size=int(source_config["unet_kernel_size"]),
        diffusion_step_embed_dim=int(source_config["diffusion_step_embed_dim"]),
        cond_predict_scale=bool(source_config["cond_predict_scale"]),
        state_mean=tuple(statistics["state_mean"]),
        state_std=tuple(statistics["state_std"]),
        action_min=tuple(statistics["action_min"]),
        action_max=tuple(statistics["action_max"]),
        force_mean=tuple(statistics["force_mean"]),
        force_std=tuple(statistics["force_std"]),
        generator_model_config=generator_config,
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
    mismatch = sorted(
        name
        for name in set(expected) & set(target)
        if tuple(expected[name].shape) != tuple(target[name].shape)
    )
    if missing or unexpected or mismatch:
        raise RuntimeError(
            "Strict Native-DP v4 state mapping failed: "
            f"missing={missing[:20]}, unexpected={unexpected[:20]}, shape_mismatch={mismatch[:20]}"
        )
    policy.load_state_dict(target, strict=True)
    return dict(policy.state_dict())


def convert_one(
    *,
    task: str,
    mode: str,
    source_checkpoint: Path,
    output_dir: Path,
    generator_checkpoint: Path | None = None,
) -> Path:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    source_checkpoint = source_checkpoint.resolve()
    source = _load_checkpoint(source_checkpoint)
    _validate_v4_scratch(source, source_checkpoint)
    policy_state, ema_decay = _ema_policy_state(source)

    generator_config = None
    generator_state = None
    generator = None
    if mode == "tactigen":
        if generator_checkpoint is None:
            raise ValueError("tactigen conversion requires the task-matched TactiGen checkpoint")
        generator_checkpoint = generator_checkpoint.resolve()
        generator = _load_checkpoint(generator_checkpoint)
        generator_config = _generator_config(generator)
        generator_state = generator.get("model_state_dict")
        if not isinstance(generator_state, Mapping):
            raise KeyError("TactiGen checkpoint requires model_state_dict")

    config = _make_config(task, mode, source, generator_config)
    policy = ACMTDPPolicy(config)
    target_state = _strict_target_state(policy, policy_state, generator_state)

    output_dir = output_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    backup: Path | None = None
    try:
        policy.save_pretrained(temporary, state_dict=target_state)
        config_path = temporary / "config.json"
        serialized = json.loads(config_path.read_text(encoding="utf-8"))
        serialized["type"] = "acmt_dp"
        serialized["checkpoint_schema"] = "acmt_dp.native_dp_v4"
        serialized["checkpoint_schema_version"] = 4
        config_path.write_text(json.dumps(serialized, indent=4) + "\n", encoding="utf-8")
        preprocessor, postprocessor = make_acmt_dp_pre_post_processors(config)
        preprocessor.save_pretrained(temporary, config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json")
        postprocessor.save_pretrained(temporary, config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json")
        source_sha = _sha256(source_checkpoint)
        generator_sha = _sha256(generator_checkpoint) if generator_checkpoint else None
        manifest = {
            "schema_version": 4,
            "checkpoint_schema": "acmt_dp.native_dp_v4",
            "upstream_commit": UPSTREAM_COMMIT,
            "policy_type": "acmt_dp",
            "task_variant": task,
            "tactile_source": mode,
            "policy_checkpoint_tactile_source": "real" if mode == "tactigen" else mode,
            "policy_weight_source": "real" if mode == "tactigen" else mode,
            "protocol": "single_frozen_inference" if mode == "tactigen" else "direct_policy",
            "online_protocol": "16_predict_8_execute_at_30hz",
            "tactile_history": 4,
            "tactile_encoder": "frame_spatial_cnn_shared_no_gru",
            "camera_order": ["top", "side", "wrist_left", "wrist_right"],
            "camera_keys": list(config.camera_keys),
            "visual_preprocess": "resize256_center224_imagenet",
            "state_dim": 8,
            "feature_dim": 512,
            "tactile_dim": 160,
            "action_execution_horizon": 8,
            "control_hz": 30.0,
            "vision_mode": "scratch",
            "seed": int(source.get("config", {}).get("seed", 42)),
            "source_checkpoint": source_checkpoint.name,
            "source_checkpoint_sha256": source_sha,
            "policy_checkpoint_sha256": source_sha,
            "real_policy_checkpoint_sha256": source_sha if mode == "tactigen" else None,
            "source_global_step": int(source.get("global_step", -1)),
            "source_best_val_loss": float(source.get("best_val_loss", float("nan"))),
            "ema": {"selected": True, "decay": ema_decay, "base": "model_state_dict"},
            "generator_checkpoint": generator_checkpoint.name if generator_checkpoint else None,
            "generator_checkpoint_sha256": generator_sha,
            "tactigen_checkpoint_sha256": generator_sha,
            "generator_global_step": int(generator.get("global_step", -1)) if generator else None,
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


def _resolve_policy_source(root: Path, task: str, mode: str) -> Path:
    source_mode = "real" if mode == "tactigen" else mode
    if root.is_file():
        return root
    candidates = (
        root / source_mode / "scratch_progress_fixed" / "seed42" / "best.pt",
        root / source_mode / "scratch" / "seed42" / "best.pt",
        root / source_mode / "seed42" / "best.pt",
        root / task / source_mode / "scratch_progress_fixed" / "seed42" / "best.pt",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No v4 scratch checkpoint found for {task}/{mode} below {root}; pass --policy-checkpoint explicitly"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=(*TASKS, "all"), default="all")
    parser.add_argument("--mode", choices=(*MODES, "all"), default="all")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/acmt_dp"))
    parser.add_argument("--policy-source-root", type=Path)
    parser.add_argument(
        "--policy-checkpoint", type=Path, help="Explicit v4 scratch best.pt (single task/mode)"
    )
    parser.add_argument("--generator-checkpoint", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    tasks = TASKS if args.task == "all" else (args.task,)
    modes = MODES if args.mode == "all" else (args.mode,)
    if args.policy_checkpoint and (len(tasks) != 1 or len(modes) != 1):
        raise ValueError("--policy-checkpoint requires one --task and one --mode")
    if args.generator_checkpoint and len(tasks) != 1:
        raise ValueError("--generator-checkpoint requires one --task")
    for task in tasks:
        root = args.policy_source_root or DEFAULT_POLICY_ROOTS[task]
        generator = args.generator_checkpoint or DEFAULT_GENERATOR_CHECKPOINTS[task]
        for mode in modes:
            source = args.policy_checkpoint or _resolve_policy_source(root, task, mode)
            output = args.output_root / task / mode / "seed42" / "pretrained_model"
            print(f"Converting {task}/{mode}: {source} -> {output}", flush=True)
            convert_one(
                task=task,
                mode=mode,
                source_checkpoint=source,
                output_dir=output,
                generator_checkpoint=generator if mode == "tactigen" else None,
            )
            gc.collect()


if __name__ == "__main__":
    main()
