"""Run a small ACMT-PI05 load/forward/backward preflight.

This deliberately loads the requested PI05 base checkpoint before touching the
training loop.  A missing base model is an error; the caller must not silently
fall back to random PaliGemma weights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lerobot.datasets.acmt_act_memmap import ACMTACTMemmapDataset
from lerobot.policies.acmt_pi05.configuration_acmt_pi05 import ACMTPi05Config
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.collate import lerobot_collate_fn


def _dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memmap-dir", required=True)
    parser.add_argument("--tactile-source", choices=("none", "real"), required=True)
    parser.add_argument("--base-model", default="lerobot/pi05_base")
    parser.add_argument("--tokenizer-name", default="google/paligemma-3b-pt-224")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--gradient-accumulation-steps", type=int, required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    if args.batch_size < 1 or args.gradient_accumulation_steps < 1 or args.steps < 1:
        raise ValueError("batch-size, gradient-accumulation-steps and steps must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")

    # Ask Transformers/HF for local files only.  This gives a short actionable
    # error instead of several minutes of tokenizer/model network retries.
    from transformers.utils import cached_file

    try:
        base_path = Path(args.base_model)
        if base_path.is_dir():
            required = [base_path / "config.json", base_path / "model.safetensors"]
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise FileNotFoundError(", ".join(missing))
        else:
            cached_file(args.base_model, "config.json", local_files_only=True)
            cached_file(args.base_model, "model.safetensors", local_files_only=True)
    except Exception as exc:
        raise FileNotFoundError(
            f"PI05 base model {args.base_model!r} is not available locally. "
            "Download/cache lerobot/pi05_base first, or pass --base-model=/path/to/pi05_base."
        ) from exc

    root = Path(args.memmap_dir)
    for name in ("manifest.json", "splits.json", "episode_instructions.json", "acmt_pi05_stats.json"):
        if not (root / name).is_file():
            raise FileNotFoundError(root / name)

    dataset = ACMTACTMemmapDataset(
        root,
        split="train",
        repo_id="local/acmt-pi05-peg",
        chunk_size=50,
        policy_kind="acmt_pi05",
    )
    config = ACMTPi05Config(
        pretrained_path=args.base_model,
        tactile_source=args.tactile_source,
        tactile_stats_path=str(root / "acmt_pi05_stats.json"),
        tokenizer_name=args.tokenizer_name,
        dtype=args.dtype,
        device=str(device),
        use_relative_actions=True,
        train_expert_only=True,
        gradient_checkpointing=True,
        input_features={},
        output_features={},
    )
    policy = make_policy(config, ds_meta=dataset.meta)
    preprocessor, _ = make_pre_post_processors(config, dataset_stats=dataset.meta.stats)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        collate_fn=lerobot_collate_fn,
    )
    iterator = iter(loader)
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=config.optimizer_lr)
    policy.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        for micro in range(args.gradient_accumulation_steps):
            batch = next(iterator)
            for cam_key in dataset.meta.camera_keys:
                if batch[cam_key].dtype == torch.uint8:
                    batch[cam_key] = batch[cam_key].float() / 255.0
            batch = preprocessor(batch)
            with torch.autocast(device_type=device.type, dtype=_dtype(args.dtype), enabled=device.type == "cuda" and args.dtype != "float32"):
                loss, _ = policy(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step={step}, micro={micro}: {loss.item()}")
            (loss / args.gradient_accumulation_steps).backward()
        optimizer.step()
    learnable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    if not learnable or not any(parameter.grad is not None for parameter in learnable):
        raise RuntimeError("ACMT-PI05 preflight produced no trainable gradients")
    peak = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
    print(
        json.dumps(
            {
                "status": "PASS",
                "base_model": args.base_model,
                "tactile_source": args.tactile_source,
                "batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
                "steps": args.steps,
                "peak_memory_gib": round(peak, 3),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
