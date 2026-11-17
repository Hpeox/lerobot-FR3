"""Read-only checkpoint validation; no pretrained downloads or optimizer loading."""

import hashlib
import json
from pathlib import Path

import torch

from ._core.config import ModelConfig
from .preprocessing import Normalizer


def bundle_id(weights):
    identity = hashlib.sha256()
    for name, value in sorted(weights.items()):
        identity.update(name.encode())
        identity.update(value.detach().cpu().contiguous().numpy().tobytes())
    return identity.hexdigest()


def metadata_id(payload):
    value = {name: payload[name] for name in ("config", "stats", "tasks", "source_inventory", "step")}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def validate_metadata(payload):
    if (
        payload.get("format_version") != 2
        or payload.get("trained") is not True
        or payload.get("step") != 100000
    ):
        raise ValueError("Expected trained format-v2 step100000 metadata")
    config = ModelConfig(**payload["config"])
    if (
        len(payload["tasks"]) != config.num_tasks
        or not payload["tasks"]
        or len(set(payload["tasks"])) != len(payload["tasks"])
    ):
        raise ValueError("Checkpoint task vocabulary does not match the model")
    Normalizer(payload["stats"])


def read_whole(path):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    payload.pop("optimizer", None)  # In-memory only. The source file is never saved.
    validate_metadata(payload)
    if "model" not in payload or "component" in payload:
        raise ValueError("Expected a whole checkpoint, not a split component")
    payload["bundle_id"] = bundle_id(payload["model"])
    return payload


def read_component(path, kind):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    validate_metadata(payload)
    if payload.get("component") != kind or not payload.get("bundle_id") or "weights" not in payload:
        raise ValueError(f"Expected a trained {kind} component with bundle identity")
    return payload


def read_components(vision_path, action_path):
    vision, action = read_component(vision_path, "vision"), read_component(action_path, "action")
    for name in ("config", "stats", "tasks", "source_inventory", "step", "bundle_id"):
        if vision[name] != action[name]:
            raise ValueError("Split component metadata mismatch: " + name)
    weights = {
        **{"vision." + key: value for key, value in vision["weights"].items()},
        **{"action." + key: value for key, value in action["weights"].items()},
    }
    if bundle_id(weights) != vision["bundle_id"]:
        raise ValueError("Split component tensors do not match their declared bundle")
    return {
        **{key: value for key, value in vision.items() if key not in ("weights", "component")},
        "model": weights,
    }
