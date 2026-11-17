"""Independent vision/action workers. No networking, robot drivers, or commands.

Separate processes may transport VisualPacket tensors themselves. Different hosts
need an agreed clock domain; their default monotonic clocks are not comparable.
"""

import hashlib
import math
import socket
import time
import uuid
from dataclasses import dataclass

import torch

from ._core.config import ModelConfig
from ._core.models import ActionFlow
from ._core.siglip import SiglipVisionEncoder
from .checkpoint import metadata_id, read_component
from .configuration_flowmatch import CAMERAS
from .modeling_flowmatch import as_bchw, sample_noise
from .preprocessing import Normalizer, preprocess_rgb


def local_clock_domain():
    """Host-scoped default; a generic label cannot equate two hosts' clocks."""
    identity = f"{socket.gethostname()}|{uuid.getnode()}".encode()
    return "local_monotonic:" + hashlib.sha256(identity).hexdigest()[:20]


@dataclass(frozen=True)
class VisualPacket:
    features: torch.Tensor
    task_ids: torch.Tensor
    bundle_id: str
    metadata_id: str
    episode_id: str
    frame_id: int
    captured_at: float
    clock_domain: str


class VisionWorker:
    def __init__(self, path, device="cpu", clock_domain=None):
        self.meta = read_component(path, "vision")
        self.cfg = ModelConfig(**self.meta["config"])
        self.device, self.clock_domain = device, clock_domain or local_clock_domain()
        self.module = SiglipVisionEncoder(self.cfg, initialize_pretrained=False)
        self.module.load_state_dict(self.meta["weights"], strict=True)
        self.meta.pop("weights")
        self.module.to(device).eval().requires_grad_(False)

    @torch.inference_mode()
    def encode(self, rgb, task_ids, *, episode_id, frame_id, captured_at=None):
        if not episode_id or type(frame_id) is not int or frame_id < 0:
            raise ValueError("Use a nonempty episode_id and nonnegative integer frame_id")
        stamp = time.monotonic() if captured_at is None else float(captured_at)
        if not math.isfinite(stamp):
            raise ValueError("Capture time must be finite")
        tasks = torch.as_tensor(task_ids)
        if tasks.dtype == torch.bool or tasks.is_floating_point():
            raise ValueError("Task IDs must be integers")
        tasks = tasks.long().reshape(-1)
        if (tasks < 0).any() or (tasks >= len(self.meta["tasks"])).any():
            raise ValueError("Task ID outside checkpoint vocabulary")
        pixels = torch.stack(
            [preprocess_rgb(as_bchw(rgb[name], 480, 640).cpu(), self.cfg.image_size) for name in CAMERAS], 1
        ).to(self.device)
        if len(tasks) != pixels.shape[0]:
            raise ValueError("Task/image batch mismatch")
        features = self.module.encode(pixels, tasks.to(self.device))
        return VisualPacket(
            features.detach().cpu().contiguous(),
            tasks.cpu(),
            self.meta["bundle_id"],
            metadata_id(self.meta),
            str(episode_id),
            frame_id,
            stamp,
            self.clock_domain,
        )


class ActionWorker:
    def __init__(self, path, device="cpu", *, max_visual_age=0.2, clock_domain=None):
        self.meta = read_component(path, "action")
        self.cfg = ModelConfig(**self.meta["config"])
        self.device, self.clock_domain = device, clock_domain or local_clock_domain()
        self.max_visual_age = float(max_visual_age)
        if not math.isfinite(self.max_visual_age) or self.max_visual_age <= 0:
            raise ValueError("max_visual_age must be finite and positive")
        self.normalizer = Normalizer(self.meta["stats"])
        self.module = ActionFlow(self.cfg)
        self.module.load_state_dict(self.meta["weights"], strict=True)
        self.meta.pop("weights")
        self.module.to(device).eval().requires_grad_(False)

    @torch.inference_mode()
    def generate_raw(
        self, raw_state, packet: VisualPacket, *, episode_id, frame_id, now=None, noise=None, seed=None
    ):
        if packet.bundle_id != self.meta["bundle_id"]:
            raise ValueError("Vision/action bundle mismatch")
        if packet.metadata_id != metadata_id(self.meta):
            raise ValueError("Vision/action normalization, config, or task metadata mismatch")
        if packet.clock_domain != self.clock_domain:
            raise ValueError("Clock domains differ; receiver must establish valid capture times")
        if packet.episode_id != episode_id or packet.frame_id != frame_id:
            raise ValueError("Visual packet episode/frame does not match the requested observation")
        age = (time.monotonic() if now is None else float(now)) - packet.captured_at
        if not math.isfinite(age) or age < 0 or age > self.max_visual_age:
            raise ValueError("Stale visual packet or unsynchronized clock")
        state = torch.as_tensor(raw_state, dtype=torch.float32, device=self.device)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if state.ndim != 2 or state.shape[1] != 8 or not torch.isfinite(state).all():
            raise ValueError("Expected finite measured state [B,8]")
        if ((state[:, 7] < 0) | (state[:, 7] > 1)).any():
            raise ValueError("State gripper must be measured gPO/255")
        features = packet.features.to(self.device)
        if features.shape != (len(state), 16, self.cfg.vision_width) or not torch.isfinite(features).all():
            raise ValueError("Unexpected visual feature shape/values")
        tasks = packet.task_ids.to(self.device)
        if tasks.shape != (len(state),) or (tasks < 0).any() or (tasks >= len(self.meta["tasks"])).any():
            raise ValueError("Invalid packet task IDs")
        state = self.normalizer.normalize(state, "state")
        action = self.module.sample(state, features, tasks, noise=sample_noise(state, self.cfg, noise, seed))
        action = self.normalizer.denormalize(action, "action")
        if not torch.isfinite(action).all():
            raise FloatingPointError("Nonfinite action")
        return action

    def generate(self, raw_state, packet, *, action_mapping, **kwargs):
        from .mapping import ActionMapping

        return ActionMapping.from_dict(action_mapping).convert(self.generate_raw(raw_state, packet, **kwargs))
