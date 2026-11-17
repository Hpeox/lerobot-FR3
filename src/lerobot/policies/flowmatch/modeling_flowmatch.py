"""Native inference policy: q(rad)+measured gPO/255 -> 16 targets, execute 4.

Raw prediction is explicitly separate from mapped FR3 actions. No robot I/O.
"""

from collections import deque

import torch

from lerobot.policies.pretrained import PreTrainedPolicy

from ._core.config import ModelConfig
from ._core.models import SplitPolicy
from .checkpoint import bundle_id, read_components, read_whole
from .configuration_flowmatch import CAMERAS, FlowMatchConfig
from .preprocessing import Normalizer, preprocess_rgb


def as_bchw(value, height, width):
    value = torch.as_tensor(value)
    if value.ndim == 3:
        value = value.unsqueeze(0)
    if value.ndim == 4 and value.shape[-1] == 3 and value.shape[1] != 3:
        value = value.movedim(-1, 1)
    if value.ndim != 4 or tuple(value.shape[1:]) != (3, height, width):
        raise ValueError(f"Expected RGB BCHW/BHWC at {height}x{width}, got {tuple(value.shape)}")
    return value.contiguous()


def sample_noise(state, cfg, noise=None, seed=None):
    shape = (state.shape[0], cfg.horizon, cfg.action_dim)
    if noise is not None and seed is not None:
        raise ValueError("Provide noise or seed, not both")
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        noise = torch.randn(shape, generator=generator)
    if noise is not None:
        noise = torch.as_tensor(noise, dtype=state.dtype, device=state.device)
        if tuple(noise.shape) != shape or not torch.isfinite(noise).all():
            raise ValueError("Noise shape/values do not match the action block")
    return noise


class FlowMatchPolicy(PreTrainedPolicy):
    config_class = FlowMatchConfig
    name = "flowmatch"

    def __init__(self, config: FlowMatchConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.model = SplitPolicy(ModelConfig(**config.model), initialize_pretrained=False)
        self.normalizer = Normalizer(config.stats) if config.stats else None
        self._weights_loaded = False
        self.last_raw_gripper_out_fraction = None
        self.requires_grad_(False)
        self.eval()
        self.reset()

    @classmethod
    def _from_payload(cls, payload, **overrides):
        config = FlowMatchConfig(
            model=payload["config"],
            stats=payload["stats"],
            tasks=payload["tasks"],
            trained=True,
            source_step=payload["step"],
            source_bundle_id=payload["bundle_id"],
            push_to_hub=False,
            **overrides,
        )
        policy = cls(config)
        policy.model.load_state_dict(payload["model"], strict=True)
        policy._weights_loaded = True
        policy._place_modules()
        return policy

    @classmethod
    def from_checkpoint(cls, checkpoint, **config_overrides):
        return cls._from_payload(read_whole(checkpoint), **config_overrides)

    @classmethod
    def from_components(cls, vision_path, action_path, **config_overrides):
        config_overrides.setdefault("execution_mode", "split")
        return cls._from_payload(read_components(vision_path, action_path), **config_overrides)

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, **kwargs):
        if kwargs.pop("strict", True) is not True:
            raise ValueError("FlowMatch requires strict weight loading")
        policy = super().from_pretrained(pretrained_name_or_path, strict=True, **kwargs)
        if not policy.config.trained or policy.normalizer is None or not policy.config.tasks:
            raise ValueError("Export is missing trained metadata, normalization, or task vocabulary")
        if bundle_id(policy.model.state_dict()) != policy.config.source_bundle_id:
            raise ValueError("Exported tensors differ from the checkpoint bundle identity")
        policy._weights_loaded = True
        policy.requires_grad_(False)
        policy._place_modules()
        return policy

    def _place_modules(self):
        whole = self.config.execution_mode == "whole"
        vision_device = self.config.device if whole else (self.config.vision_device or self.config.device)
        action_device = self.config.device if whole else (self.config.action_device or self.config.device)
        self.model.vision.to(vision_device)
        self.model.action.to(action_device)
        return torch.device(vision_device), torch.device(action_device)

    def _require_loaded(self):
        if not self._weights_loaded or self.normalizer is None:
            raise ValueError(
                "Load trained weights with from_checkpoint/from_components/from_pretrained first"
            )

    def _task_ids(self, batch, batch_size):
        if "task_id" in batch:
            value = torch.as_tensor(batch["task_id"])
            if value.dtype == torch.bool or value.is_floating_point():
                raise ValueError("task_id must be an integer checkpoint vocabulary index")
            task = value.long().reshape(-1)
        elif self.config.default_task_id is not None:
            task = torch.full((batch_size,), self.config.default_task_id, dtype=torch.long)
        else:
            text = batch.get("task")
            text = [text] * batch_size if isinstance(text, str) else text
            if text is None:
                raise ValueError("Provide checkpoint task text, task_id, or an explicit default_task_id")
            try:
                task = torch.tensor([self.config.tasks.index(item) for item in text])
            except ValueError as error:
                raise ValueError(
                    "Task text is outside the checkpoint vocabulary; this is not a language model"
                ) from error
        if task.shape != (batch_size,) or (task < 0).any() or (task >= len(self.config.tasks)).any():
            raise ValueError("Task IDs/batch size do not match the checkpoint")
        return task

    @torch.inference_mode()
    def predict_raw_action_chunk(self, batch, *, noise=None, seed=None):
        """Return unmodified GELLO-space [B,16,8] for offline analysis only."""
        self._require_loaded()
        self.eval()
        vision_device, action_device = self._place_modules()
        state = torch.as_tensor(batch["observation.state"], dtype=torch.float32, device=action_device)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if state.ndim != 2 or state.shape[1] != 8 or not torch.isfinite(state).all():
            raise ValueError("State must be finite measured FR3 q[7] + gPO/255, shape [B,8]")
        if ((state[:, 7] < 0) | (state[:, 7] > 1)).any():
            raise ValueError("Measured gripper state must be gPO/255 in [0,1]")
        tasks = self._task_ids(batch, state.shape[0])
        cfg = self.model.cfg
        views = []
        for name in CAMERAS:
            image = as_bchw(
                batch[self.config.image_key(name)], self.config.raw_image_height, self.config.raw_image_width
            )
            if image.shape[0] != state.shape[0]:
                raise ValueError("Camera/state batch size mismatch")
            views.append(preprocess_rgb(image.cpu(), cfg.image_size))
        pixels = torch.stack(views, 1).to(vision_device)
        features = self.model.vision.encode(pixels, tasks.to(vision_device)).to(action_device)
        state = self.normalizer.normalize(state, "state")
        raw = self.model.action.sample(
            state, features, tasks.to(action_device), noise=sample_noise(state, cfg, noise, seed)
        )
        raw = self.normalizer.denormalize(raw, "action")
        if not torch.isfinite(raw).all():
            raise FloatingPointError("Nonfinite generated action")
        self.last_raw_gripper_out_fraction = float(((raw[..., 7] < 0) | (raw[..., 7] > 1)).float().mean())
        return raw

    @torch.inference_mode()
    def predict_action_chunk(self, batch, **kwargs):
        """Native FR3 target space. Requires verified camera and action mappings."""
        mapping = self.config.require_deployment_mapping()
        return mapping.convert(self.predict_raw_action_chunk(batch, **kwargs))

    @torch.inference_mode()
    def select_action(self, batch, **kwargs):
        self._require_loaded()
        self.config.require_deployment_mapping()
        if not self._queue:
            chunk = self.predict_action_chunk(batch, **kwargs)
            self._queue.extend(chunk[:, : self.model.cfg.n_action_steps].transpose(0, 1))
        return self._queue.popleft()

    def reset(self):
        self._queue = deque()

    def forward(self, batch):
        raise NotImplementedError(
            "Inference-only integration; native FR3 labels are not raw GELLO training labels"
        )

    def get_optim_params(self):
        raise NotImplementedError("Use the original training project")
