#!/usr/bin/env python
"""Convert native-DP v5 checkpoints to an inference-only LeRobot artifact."""

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

from lerobot.policies.acmt_dp.configuration_acmt_dp_v5 import ACMTDPV5Config
from lerobot.policies.acmt_dp.modeling_acmt_dp_v5 import ACMTDPV5Policy
from lerobot.policies.acmt_dp.processor_acmt_dp_v5 import make_acmt_dp_v5_pre_post_processors
from lerobot.policies.acmt_dp.processor_acmt_dp import ACMT_DP_DEFAULT_SOURCE_CAMERA_KEYS
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

MODES = ("none", "real", "tactigen")
TASKS = ("peg", "gear")
UPSTREAM_COMMIT = "f6306c91d59ddac077be026da60e5b1ebeaa2533"
DEFAULT_POLICY_ROOTS = {
    "peg": Path("/data2/cym/16mm_peg_in_hole/native_dp_v5"),
    "gear": Path("/data2/cym/gear_insert_big2small/native_dp_v5"),
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
    value = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(value, dict):
        raise TypeError(f"checkpoint {path} must contain a dictionary")
    return value


def _validate_v5_scratch(checkpoint: Mapping[str, Any], path: Path) -> None:
    if checkpoint.get("schema") != "acmt_dp.native_dp_v5_hybrid":
        raise ValueError(f"{path} is not a Native-DP v5 checkpoint; v3/v4 artifacts are rejected")
    config = checkpoint.get("config")
    if not isinstance(config, Mapping):
        raise KeyError("v5 checkpoint requires config")
    for field, expected in (
        ("obs_horizon", 4),
        ("n_obs_steps", 4),
        ("pad_before", 3),
        ("internal_horizon", 19),
        ("pred_horizon", 16),
        ("public_pred_horizon", 16),
        ("n_action_steps", 8),
        ("action_execution_horizon", 8),
        ("state_dim", 8),
        ("action_dim", 8),
        ("tactile_dim", 160),
        ("visual_feature_dim", 64),
        ("resize_height", 240),
        ("resize_width", 320),
        ("crop_height", 216),
        ("crop_width", 288),
        ("spatial_num_keypoints", 32),
        ("unet_kernel_size", 5),
        ("diffusion_step_embed_dim", 128),
        ("diffusion_train_steps", 100),
        ("diffusion_inference_steps", 8),
    ):
        if int(config.get(field, -1)) != expected:
            raise ValueError(f"v5 checkpoint config {field} must be {expected}")
    if tuple(config.get("camera_names", ())) != ("top", "side", "wrist_left", "wrist_right"):
        raise ValueError("v5 checkpoint camera order is invalid")
    if config.get("use_group_norm") is not True:
        raise ValueError("v5 checkpoint must use GroupNorm visual encoders")
    if config.get("random_crop") is not True:
        raise ValueError("v5 checkpoint must be trained with random_crop=true")
    if config.get("cond_predict_scale") is not True:
        raise ValueError("v5 checkpoint must use cond_predict_scale=true")
    for key in ("statistics", "model_state_dict", "ema_state_dict"):
        if not isinstance(checkpoint.get(key), Mapping):
            raise KeyError(f"v5 checkpoint requires {key}")
    if checkpoint.get("statistics", {}).get("state_min") is None:
        raise KeyError("v5 checkpoint statistics require state_min/state_max")


def _ema_policy_state(checkpoint: Mapping[str, Any]) -> tuple[dict[str, Tensor], float]:
    raw, ema = checkpoint["model_state_dict"], checkpoint["ema_state_dict"]
    shadow = ema.get("shadow")
    if not isinstance(raw, Mapping) or not isinstance(shadow, Mapping):
        raise KeyError("v5 checkpoint requires ema_state_dict.shadow")
    state = dict(raw)
    for name, value in shadow.items():
        if name not in state:
            raise KeyError(f"EMA key {name!r} is absent from model_state_dict")
        if not isinstance(value, Tensor) or not torch.is_floating_point(value):
            raise TypeError(f"EMA shadow {name!r} must be floating point")
        state[name] = value
    required = {
        "visual_encoder.camera_encoders.0.backbone.0.weight",
        "visual_encoder.camera_encoders.0.spatial.keypoint_conv.weight",
        "tactile_encoder.spatial.0.weight",
        "normalizer.params_dict.state.offset",
        "noise_predictor.final_conv.1.weight",
    }
    missing = sorted(required - set(state))
    if missing:
        raise ValueError(f"v5 checkpoint is missing required keys: {missing}")
    return state, float(ema.get("decay", 0.9999))


def _generator_config(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    config = checkpoint.get("model_config")
    if not isinstance(config, Mapping):
        raise KeyError("TactiGen checkpoint requires model_config")
    result = dict(config)
    result.pop("dformerv2_repo_path", None)
    result.pop("dformerv2_checkpoint", None)
    return result


def _make_config(
    task: str, mode: str, source: Mapping[str, Any], generator_config: dict[str, Any] | None
) -> ACMTDPV5Config:
    source_config, statistics = source["config"], source["statistics"]
    expected_source = "real" if mode == "tactigen" else mode
    if source_config.get("tactile_source") != expected_source:
        raise ValueError(
            f"checkpoint tactile_source={source_config.get('tactile_source')!r} does not match mode={mode!r}"
        )
    return ACMTDPV5Config(
        tactile_source=mode,
        checkpoint_tactile_source=expected_source,
        task_variant=task,
        checkpoint_task_variant=task,
        checkpoint_schema_version=5,
        checkpoint_schema="acmt_dp.native_dp_v5_hybrid",
        visual_preprocess="resize240_center216_range",
        obs_horizon=4,
        n_obs_steps=4,
        pad_before=3,
        internal_horizon=19,
        pred_horizon=16,
        public_pred_horizon=16,
        action_horizon=16,
        n_action_steps=8,
        action_execution_horizon=8,
        feature_dim=64,
        tactile_dim=160,
        resize_height=240,
        resize_width=320,
        random_crop=True,
        use_group_norm=True,
        crop_height=216,
        crop_width=288,
        spatial_num_keypoints=32,
        diffusion_train_steps=int(source_config["diffusion_train_steps"]),
        diffusion_inference_steps=int(source_config.get("diffusion_inference_steps", 8)),
        unet_dims=tuple(source_config["unet_dims"]),
        unet_kernel_size=int(source_config["unet_kernel_size"]),
        diffusion_step_embed_dim=int(source_config["diffusion_step_embed_dim"]),
        cond_predict_scale=bool(source_config["cond_predict_scale"]),
        state_min=tuple(statistics["state_min"]),
        state_max=tuple(statistics["state_max"]),
        action_min=tuple(statistics["action_min"]),
        action_max=tuple(statistics["action_max"]),
        force_mean=tuple(statistics["force_mean"]),
        force_std=tuple(statistics["force_std"]),
        generator_model_config=generator_config,
        device="cpu",
        push_to_hub=False,
    )


def _strict_target_state(
    policy: ACMTDPV5Policy, policy_state: Mapping[str, Tensor], generator_state: Mapping[str, Tensor] | None
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
            f"strict v5 state mapping failed: missing={missing[:10]}, unexpected={unexpected[:10]}, shape_mismatch={mismatch[:10]}"
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
    source_checkpoint = source_checkpoint.resolve()
    source = _load_checkpoint(source_checkpoint)
    _validate_v5_scratch(source, source_checkpoint)
    policy_state, ema_decay = _ema_policy_state(source)
    generator_config = generator_state = generator = None
    if mode == "tactigen":
        if generator_checkpoint is None:
            raise ValueError("tactigen conversion requires a task-matched ACMT checkpoint")
        generator_checkpoint = generator_checkpoint.resolve()
        generator = _load_checkpoint(generator_checkpoint)
        generator_config = _generator_config(generator)
        generator_state = generator.get("model_state_dict")
        if not isinstance(generator_state, Mapping):
            raise KeyError("TactiGen checkpoint requires model_state_dict")
    config = _make_config(task, mode, source, generator_config)
    policy = ACMTDPV5Policy(config)
    target_state = _strict_target_state(policy, policy_state, generator_state)
    output_dir = output_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    backup = None
    try:
        policy.save_pretrained(temporary, state_dict=target_state)
        config_path = temporary / "config.json"
        serialized = json.loads(config_path.read_text(encoding="utf-8"))
        serialized.update(
            {
                "type": "acmt_dp_v5",
                "checkpoint_schema": "acmt_dp.native_dp_v5_hybrid",
                "checkpoint_schema_version": 5,
            }
        )
        config_path.write_text(json.dumps(serialized, indent=4) + "\n", encoding="utf-8")
        preprocessor, postprocessor = make_acmt_dp_v5_pre_post_processors(config)
        preprocessor.save_pretrained(temporary, config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json")
        postprocessor.save_pretrained(temporary, config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json")
        source_sha, generator_sha = (
            _sha256(source_checkpoint),
            _sha256(generator_checkpoint) if generator_checkpoint else None,
        )
        manifest = {
            "schema_version": 5,
            "checkpoint_schema": "acmt_dp.native_dp_v5_hybrid",
            "upstream_commit": UPSTREAM_COMMIT,
            "policy_type": "acmt_dp_v5",
            "task_variant": task,
            "tactile_source": mode,
            "policy_checkpoint_tactile_source": "real" if mode == "tactigen" else mode,
            "protocol": "synchronous_select_action",
            "online_protocol": "synchronous_select_action",
            "action_alignment": "raw_internal_19_slice_3_19_public_16",
            "diffusion_train_steps": int(config.diffusion_train_steps),
            "diffusion_inference_steps": int(config.diffusion_inference_steps),
            "diffusion": {
                "train_steps": int(config.diffusion_train_steps),
                "inference_steps": int(config.diffusion_inference_steps),
                "ddim_eta": 0.0,
                "initial_noise": "fixed_per_episode",
            },
            "runtime": {"reserve_horizon": 0, "overlap_blending": False},
            "tactile_history": 4,
            "camera_order": ["top", "side", "wrist_left", "wrist_right"],
            "camera_keys": list(config.camera_keys),
            "source_camera_keys": list(ACMT_DP_DEFAULT_SOURCE_CAMERA_KEYS),
            "gripper_mapping": "policy_[0,1]_to_gpo_[255,3]",
            "visual_preprocess": "resize240_center216_range",
            "feature_dim": 64,
            "tactile_dim": 160,
            "action_execution_horizon": 8,
            "control_hz": 30.0,
            "source_checkpoint": str(source_checkpoint),
            "source_checkpoint_sha256": source_sha,
            "policy_checkpoint_sha256": source_sha,
            "source_global_step": int(source.get("global_step", -1)),
            "source_best_val_loss": float(source.get("best_val_loss", float("nan"))),
            "ema": {"selected": True, "decay": ema_decay, "base": "model_state_dict"},
            "generator_checkpoint": generator_checkpoint.name if generator_checkpoint else None,
            "generator_checkpoint_sha256": generator_sha,
            "generator_global_step": int(generator.get("global_step", -1)) if generator else None,
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


def _resolve_source(root: Path, task: str, mode: str) -> Path:
    source_mode = "real" if mode == "tactigen" else mode
    if root.is_file():
        return root
    candidates = (
        root / source_mode / "scratch" / "seed42" / "best.pt",
        root / source_mode / "seed42" / "best.pt",
        root / task / source_mode / "scratch" / "seed42" / "best.pt",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no v5 checkpoint found for {task}/{mode} below {root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=(*TASKS, "all"), default="all")
    parser.add_argument("--mode", choices=(*MODES, "all"), default="all")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/acmt_dp_v5"))
    parser.add_argument("--policy-source-root", type=Path)
    parser.add_argument("--policy-checkpoint", type=Path)
    parser.add_argument("--generator-checkpoint", type=Path)
    args = parser.parse_args()
    tasks = TASKS if args.task == "all" else (args.task,)
    modes = MODES if args.mode == "all" else (args.mode,)
    if args.policy_checkpoint and (len(tasks) != 1 or len(modes) != 1):
        raise ValueError("--policy-checkpoint requires one task and one mode")
    for task in tasks:
        root = args.policy_source_root or DEFAULT_POLICY_ROOTS[task]
        generator = args.generator_checkpoint or DEFAULT_GENERATOR_CHECKPOINTS[task]
        for mode in modes:
            source = args.policy_checkpoint or _resolve_source(root, task, mode)
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
