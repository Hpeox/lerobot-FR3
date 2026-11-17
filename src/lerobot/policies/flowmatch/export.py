"""Convert the existing 100k weights into a native LeRobot 0.6 policy folder.

No training, network access, dataset reads, or hardware connection.
"""

import argparse
import json
from pathlib import Path

from .modeling_flowmatch import FlowMatchPolicy
from .processor_flowmatch import make_flowmatch_pre_post_processors


def deployment_overrides(path):
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    allowed = {"camera_map", "camera_mapping_source", "action_mapping", "default_task_id"}
    return {key: value[key] for key in allowed if key in value}


def export_policy(policy, output):
    policy._require_loaded()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    policy.save_pretrained(output, push_to_hub=False)
    pre, post = make_flowmatch_pre_post_processors(policy.config)
    pre.save_pretrained(output)
    post.save_pretrained(output)
    try:
        policy.config.require_deployment_mapping()
        ready = True
    except ValueError:
        ready = False
    report = {
        "source_step": 100000,
        "source_bundle_id": policy.config.source_bundle_id,
        "execution_mode": policy.config.execution_mode,
        "deployment_mapping_configured": ready,
        "raw_action_space": "GELLO target q[7] in rad + raw continuous gripper command",
        "native_action_space": "mapped FR3 q[7] in rad + target gPO/255",
        "model_training_performed": False,
        "hardware_connection_performed": False,
        "note": "Mapping completeness is checked; its physical correctness is not automatically certified.",
    }
    (output / "flowmatch_export.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--vision-checkpoint", type=Path)
    parser.add_argument("--action-checkpoint", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execution-mode", choices=["whole", "split"])
    parser.add_argument("--deployment-config", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--vision-device")
    parser.add_argument("--action-device")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be a new directory; existing exports are never replaced")
    if args.checkpoint and (args.vision_checkpoint or args.action_checkpoint):
        parser.error("Choose whole checkpoint OR a matched pair of components")
    if not args.checkpoint and not (args.vision_checkpoint and args.action_checkpoint):
        parser.error("Provide a whole checkpoint or BOTH component files")
    overrides = {
        "device": args.device,
        "vision_device": args.vision_device,
        "action_device": args.action_device,
    }
    if args.execution_mode:
        overrides["execution_mode"] = args.execution_mode
    if args.deployment_config:
        overrides.update(deployment_overrides(args.deployment_config))
    if args.checkpoint:
        policy = FlowMatchPolicy.from_checkpoint(args.checkpoint, **overrides)
    else:
        policy = FlowMatchPolicy.from_components(args.vision_checkpoint, args.action_checkpoint, **overrides)
    if args.deployment_config:
        policy.config.require_deployment_mapping()  # Reject unfilled templates before writing files.
    report = export_policy(policy, args.output)
    print(json.dumps({"output": str(args.output.resolve()), **report}))


if __name__ == "__main__":
    main()
