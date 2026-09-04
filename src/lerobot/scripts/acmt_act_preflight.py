"""Run the mandatory physical-batch preflight for ACMT-ACT variants."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lerobot.datasets.acmt_act_memmap import ACMTACTMemmapDataset
from lerobot.policies.acmt_act.configuration_acmt_act import ACMTACTConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memmap-dir", required=True)
    parser.add_argument("--tactile-source", choices=("none", "real"), required=True)
    parser.add_argument("--task", choices=("peg", "gear"), required=True)
    parser.add_argument("--policy-type", choices=("acmt_act", "acmt_actv2"), default="acmt_act")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    if args.batch_size != 16:
        raise ValueError("ACMT-ACT preflight requires physical batch_size=16")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("preflight requested CUDA but no CUDA device is available")

    camera_indices = (1, 2, 3) if args.policy_type == "acmt_actv2" else None
    dataset = ACMTACTMemmapDataset(
        args.memmap_dir,
        split="train",
        repo_id=f"local/acmt-act-{args.task}",
        camera_indices=camera_indices,
    )
    config_cls = ACMTACTConfig
    if args.policy_type == "acmt_actv2":
        from lerobot.policies.acmt_actv2.configuration_acmt_actv2 import ACMTACTV2Config

        config_cls = ACMTACTV2Config
    config = config_cls(
        device=str(device),
        tactile_source=args.tactile_source,
        task_variant=args.task,
        vision_backbone="resnet50",
        pretrained_backbone_weights="ResNet50_Weights.IMAGENET1K_V2",
    )
    policy = make_policy(config, ds_meta=dataset.meta)
    preprocessor, _ = make_pre_post_processors(config, dataset_stats=dataset.meta.stats)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    iterator = iter(loader)
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=config.optimizer_lr, weight_decay=config.optimizer_weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    policy.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(args.steps):
        batch = next(iterator)
        for key in dataset.meta.camera_keys:
            batch[key] = batch[key].float() / 255.0
        batch = preprocessor(batch)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            loss, _ = policy(batch)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at preflight step {step}: {loss.item()}")
    for index, backbone in enumerate(policy.model.backbone):
        if not any(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in backbone.parameters() if parameter.requires_grad):
            raise RuntimeError(f"camera backbone {index} produced no finite gradient")
    for index, projection in enumerate(policy.model.encoder_img_feat_input_proj):
        if not any(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in projection.parameters()):
            raise RuntimeError(f"camera projection {index} produced no finite gradient")
    peak = torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0
    print(f"PREFLIGHT PASS task={args.task} tactile_source={args.tactile_source} physical_batch_size=16 steps={args.steps} peak_memory_gib={peak:.2f}")


if __name__ == "__main__":
    main()
