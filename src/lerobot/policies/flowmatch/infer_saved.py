"""Offline whole / independent-worker inference from a saved observation NPZ."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .configuration_flowmatch import CAMERAS
from .modeling_flowmatch import FlowMatchPolicy
from .runtime import ActionWorker, VisionWorker


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--policy-path", type=Path)
    parser.add_argument("--vision-checkpoint", type=Path)
    parser.add_argument("--action-checkpoint", type=Path)
    parser.add_argument("--observation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--vision-device")
    parser.add_argument("--action-device")
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()
    whole = bool(args.checkpoint or args.policy_path)
    if bool(args.checkpoint) and bool(args.policy_path):
        parser.error("Choose raw checkpoint OR native policy folder")
    if whole == bool(args.vision_checkpoint or args.action_checkpoint):
        parser.error("Choose whole mode OR both independent component files")
    if not whole and not (args.vision_checkpoint and args.action_checkpoint):
        parser.error("Independent inference requires both vision and action")
    if args.output.exists() or args.output.suffix != ".npz":
        parser.error("Output must be a NEW .npz file")
    torch.set_num_threads(min(8, torch.get_num_threads()))
    with np.load(args.observation, allow_pickle=False) as packet:
        rgb = {name: np.array(packet["rgb_" + name]) for name in CAMERAS}
        state, task = np.array(packet["state"]), np.asarray(packet["task_id"]).item()
    if type(task) is not int:
        raise ValueError("task_id must be an integer")
    if whole:
        policy = (
            FlowMatchPolicy.from_checkpoint(args.checkpoint, device=args.device)
            if args.checkpoint
            else FlowMatchPolicy.from_pretrained(args.policy_path, local_files_only=True)
        )
        if args.policy_path:
            policy.config.device = args.device
        batch = {policy.config.image_key(name): torch.from_numpy(frame) for name, frame in rgb.items()}
        batch.update({"observation.state": torch.from_numpy(state), "task_id": torch.tensor([task])})
        action = policy.predict_raw_action_chunk(batch, seed=args.seed)[0].cpu().numpy()
        identity, steps = policy.config.source_bundle_id, policy.model.cfg.n_action_steps
        mode = policy.config.execution_mode
    else:
        # A saved observation has no live freshness deadline. Explicit frozen time
        # isolates offline numerical inference; do not copy this clock into a robot loop.
        vision = VisionWorker(
            args.vision_checkpoint, args.vision_device or args.device, clock_domain="saved_observation"
        )
        expert = ActionWorker(
            args.action_checkpoint, args.action_device or args.device, clock_domain="saved_observation"
        )
        context = vision.encode(rgb, [task], episode_id="saved", frame_id=0, captured_at=0.0)
        action = (
            expert.generate_raw(state, context, episode_id="saved", frame_id=0, now=0.0, seed=args.seed)[0]
            .cpu()
            .numpy()
        )
        identity, steps = context.bundle_id, expert.cfg.n_action_steps
        mode = "independent_components"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        np.savez_compressed(
            stream,
            action_chunk=action,
            execution_segment=action[:steps],
            task_id=task,
            seed=args.seed,
            checkpoint_step=100000,
            source_bundle_id=identity,
        )
    print(
        json.dumps(
            {
                "mode": mode,
                "output": str(args.output.resolve()),
                "shape": list(action.shape),
                "action_space": "raw GELLO targets; not FR3 driver commands",
                "robot_connected": False,
            }
        )
    )


if __name__ == "__main__":
    main()
