"""CPU numerical acceptance against a legacy-training-version synthetic reference.

Only loads the published weights and synthetic arrays. No data, training, or robot I/O.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from lerobot.policies.flowmatch.configuration_flowmatch import CAMERAS
from lerobot.policies.flowmatch.modeling_flowmatch import FlowMatchPolicy
from lerobot.policies.flowmatch.preprocessing import preprocess_rgb
from lerobot.policies.flowmatch.runtime import ActionWorker, VisionWorker


def synthetic_rgb():
    y, x = np.indices((480, 640))
    return {
        name: np.stack(
            ((x + y * i) % 256, (2 * x + 3 * y + 17 * i) % 256, (7 * x + 11 * y + 53 * i) % 256), -1
        ).astype("uint8")
        for i, name in enumerate(CAMERAS)
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--weights", type=Path, required=True, help="policy_100k.pt and components/vision.pt,action.pt"
    )
    parser.add_argument(
        "--policy-path", type=Path, help="Also verify a converted native LeRobot policy folder"
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new report path")
    torch.set_num_threads(8)
    folder = Path(__file__).parent / "assets"
    meta = json.loads((folder / "golden_100k.json").read_text(encoding="utf-8"))
    with np.load(folder / "golden_100k.npz", allow_pickle=False) as saved:
        reference = {name: np.array(saved[name]) for name in saved.files}
    rgb = synthetic_rgb()
    batch = {"observation.state": torch.from_numpy(reference["state"])[None], "task_id": torch.tensor([0])}
    raw_path = args.weights / "policy_100k.pt"
    if raw_path.exists():
        policy = FlowMatchPolicy.from_checkpoint(raw_path, device="cpu")
        source = "original_checkpoint"
    elif args.policy_path:
        policy = FlowMatchPolicy.from_pretrained(args.policy_path, local_files_only=True)
        policy.config.device = "cpu"
        policy.config.vision_device = policy.config.action_device = "cpu"
        source = "native_export"
    else:
        parser.error("Provide policy_100k.pt in weights OR --policy-path for a native-only delivery")
    batch.update(
        {
            policy.config.image_key(name): torch.from_numpy(image).permute(2, 0, 1)[None]
            for name, image in rgb.items()
        }
    )
    if policy.config.source_bundle_id != meta["source_bundle_id"]:
        raise ValueError("Weights are not the published 100k reference bundle")
    pixels = torch.stack(
        [preprocess_rgb(torch.from_numpy(rgb[name]).permute(2, 0, 1)[None], 224) for name in CAMERAS], 1
    )
    pixel_hash = hashlib.sha256(pixels.numpy().tobytes()).hexdigest()
    if pixel_hash != meta["pixel_values_sha256"]:
        raise ValueError("Preprocessing differs from training; check official PIL backend and Pillow version")
    with torch.inference_mode():
        features = policy.model.vision.encode(pixels, torch.tensor([0]))
    actual = policy.predict_raw_action_chunk(batch, seed=101)
    torch.testing.assert_close(features, torch.from_numpy(reference["features"]), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual[0], torch.from_numpy(reference["action_chunk"]), atol=1e-5, rtol=1e-5)
    results = {
        "status": "PASS",
        "step": 100000,
        "whole_weight_source": source,
        "source_bundle_id": policy.config.source_bundle_id,
        "pixel_values_sha256_matches": True,
        "legacy_visual_features_max_abs": float(
            (features - torch.from_numpy(reference["features"])).abs().max()
        ),
        "legacy_raw_action_max_abs": float(
            (actual[0] - torch.from_numpy(reference["action_chunk"])).abs().max()
        ),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "synthetic_reference": True,
        "device": "cpu",
        "raw_hdf5_read": False,
        "weights_updated": False,
        "robot_connected": False,
    }
    del policy
    gc.collect()
    if args.policy_path:
        restored = FlowMatchPolicy.from_pretrained(args.policy_path, local_files_only=True)
        restored.config.device = "cpu"
        if restored.config.execution_mode == "split":
            restored.config.vision_device = restored.config.action_device = "cpu"
        native_batch = {"observation.state": batch["observation.state"], "task_id": torch.tensor([0])}
        native_batch.update(
            {
                restored.config.image_key(name): torch.from_numpy(rgb[name]).permute(2, 0, 1)[None]
                for name in CAMERAS
            }
        )
        native = restored.predict_raw_action_chunk(native_batch, seed=101)
        torch.testing.assert_close(actual, native, atol=1e-6, rtol=1e-6)
        results["native_export_max_abs"] = float((actual - native).abs().max())
        del restored
        gc.collect()
    vision = VisionWorker(args.weights / "components/vision.pt", clock_domain="synthetic_reference")
    action = ActionWorker(args.weights / "components/action.pt", clock_domain="synthetic_reference")
    packet = vision.encode(rgb, [0], episode_id="golden", frame_id=0, captured_at=0.0)
    split = action.generate_raw(
        reference["state"], packet, episode_id="golden", frame_id=0, now=0.0, seed=101
    )
    torch.testing.assert_close(actual, split, atol=1e-6, rtol=1e-6)
    results["independent_workers_max_abs"] = float((actual - split).abs().max())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2)
        stream.write("\n")
    print(json.dumps(results))


if __name__ == "__main__":
    main()
