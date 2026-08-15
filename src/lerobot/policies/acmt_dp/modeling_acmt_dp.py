# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from lerobot.utils.constants import OBS_STATE
from lerobot.utils.import_utils import require_package

from ..pretrained import PreTrainedPolicy
from .configuration_acmt_dp import (
    DQ,
    FT300,
    GRIPPER_GPO,
    O_T_EE,
    TAU_J,
    XENSE0,
    XENSE1,
    ACMTDPConfig,
    depth_key,
    rgb_key,
)
from .modeling_components import (
    DFormerv2DualRGBDEncoder,
    TemporalTactileGridEncoder,
    TinyDualRGBDEncoder,
)
from .modeling_tactile_generator import TactiGenForceFieldModel
from .modeling_unet import ConditionalUnet1D
from .visual_preprocess import prepare_for_frozen_encoder


class ActionNormalizer(nn.Module):
    def __init__(self, minimum: tuple[float, ...], maximum: tuple[float, ...]) -> None:
        super().__init__()
        self.register_buffer("minimum", torch.as_tensor(minimum, dtype=torch.float32))
        self.register_buffer("maximum", torch.as_tensor(maximum, dtype=torch.float32))

    def normalize(self, action: Tensor) -> Tensor:
        scale = (self.maximum - self.minimum).clamp_min(1e-6)
        return ((action - self.minimum) / scale) * 2.0 - 1.0

    def unnormalize(self, action: Tensor) -> Tensor:
        scale = (self.maximum - self.minimum).clamp_min(1e-6)
        value = (action + 1.0) * 0.5 * scale + self.minimum
        value[..., 7].clamp_(0.0, 1.0)
        return value


class VisualAttentionPool(nn.Module):
    def __init__(self, dim: int = 160, queries: int = 4) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.empty(1, queries, dim))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.attention = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, memory: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        query = self.queries.expand(memory.shape[0], -1, -1)
        pooled, _ = self.attention(query, memory, memory, key_padding_mask=padding_mask, need_weights=False)
        return self.norm(pooled + query).flatten(1)


class LowdimHistoryEncoder(nn.Module):
    def __init__(self, mean: tuple[float, ...], std: tuple[float, ...], dim: int = 160) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))
        self.input = nn.Sequential(nn.Linear(28, dim), nn.GELU(), nn.LayerNorm(dim))
        self.gru = nn.GRU(dim, dim, batch_first=True)

    def forward(self, value: Tensor) -> Tensor:
        value = (value.float() - self.mean) / self.std.clamp_min(1e-6)
        _, hidden = self.gru(self.input(value))
        return hidden[-1]


def matrix_to_pose_xyzw(matrices: Tensor) -> Tensor:
    """Match the legacy ACMT-DP 4x4 -> xyz+xyzw conversion exactly."""

    return matrix_to_pose_xyzw_fast(matrices)


def matrix_to_pose_xyzw_fast(matrices: Tensor) -> Tensor:
    """Vectorized GPU-friendly 4x4 -> xyz+xyzw conversion."""
    if matrices.ndim < 3 or tuple(matrices.shape[-2:]) != (4, 4):
        raise ValueError(f"expected [...,4,4] transforms, got {tuple(matrices.shape)}")
    flat = matrices.reshape(-1, 4, 4)
    rotation = flat[:, :3, :3]
    diagonal = torch.diagonal(rotation, dim1=-2, dim2=-1)
    trace = diagonal.sum(-1)
    eps = torch.finfo(flat.dtype).eps if flat.dtype.is_floating_point else 1e-7
    s = (torch.sqrt((trace + 1.0).clamp_min(eps)) * 2.0).clamp_min(eps)
    positive = trace > 0
    q_positive = torch.stack(
        [
            (rotation[:, 2, 1] - rotation[:, 1, 2]) / s,
            (rotation[:, 0, 2] - rotation[:, 2, 0]) / s,
            (rotation[:, 1, 0] - rotation[:, 0, 1]) / s,
            0.25 * s,
        ],
        dim=-1,
    )
    s0 = (
        torch.sqrt((1.0 + diagonal[:, 0] - diagonal[:, 1] - diagonal[:, 2]).clamp_min(eps)) * 2.0
    ).clamp_min(eps)
    q0 = torch.stack(
        [
            0.25 * s0,
            (rotation[:, 0, 1] + rotation[:, 1, 0]) / s0,
            (rotation[:, 0, 2] + rotation[:, 2, 0]) / s0,
            (rotation[:, 2, 1] - rotation[:, 1, 2]) / s0,
        ],
        dim=-1,
    )
    s1 = (
        torch.sqrt((1.0 + diagonal[:, 1] - diagonal[:, 0] - diagonal[:, 2]).clamp_min(eps)) * 2.0
    ).clamp_min(eps)
    q1 = torch.stack(
        [
            (rotation[:, 0, 1] + rotation[:, 1, 0]) / s1,
            0.25 * s1,
            (rotation[:, 1, 2] + rotation[:, 2, 1]) / s1,
            (rotation[:, 0, 2] - rotation[:, 2, 0]) / s1,
        ],
        dim=-1,
    )
    s2 = (
        torch.sqrt((1.0 + diagonal[:, 2] - diagonal[:, 0] - diagonal[:, 1]).clamp_min(eps)) * 2.0
    ).clamp_min(eps)
    q2 = torch.stack(
        [
            (rotation[:, 0, 2] + rotation[:, 2, 0]) / s2,
            (rotation[:, 1, 2] + rotation[:, 2, 1]) / s2,
            0.25 * s2,
            (rotation[:, 1, 0] - rotation[:, 0, 1]) / s2,
        ],
        dim=-1,
    )
    axis = diagonal.argmax(-1)
    fallback = torch.where(axis[:, None] == 0, q0, torch.where(axis[:, None] == 1, q1, q2))
    quat = torch.where(positive[:, None], q_positive, fallback)
    quat = quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp_min(eps)
    return torch.cat([flat[:, :3, 3], quat], dim=-1).reshape(*matrices.shape[:-2], 7)


class ACMTDPPolicy(PreTrainedPolicy):
    config_class = ACMTDPConfig
    name = "acmt_dp"

    def __init__(self, config: ACMTDPConfig, **_: Any) -> None:
        require_package("diffusers", extra="acmt-dp")
        super().__init__(config)
        config.validate_features()
        self.config = config
        encoder_class = (
            DFormerv2DualRGBDEncoder if config.visual_encoder_name == "dformerv2" else TinyDualRGBDEncoder
        )
        self.visual_encoder = encoder_class(visual_dim=160, t_obs=4, num_heads=4, view_dropout=0.0)
        for parameter in self.visual_encoder.parameters():
            parameter.requires_grad_(False)
        self.visual_encoder.eval()
        self.visual_pool = VisualAttentionPool()
        self.lowdim_encoder = LowdimHistoryEncoder(config.lowdim_mean, config.lowdim_std)
        self.tactile_encoder = TemporalTactileGridEncoder(
            config.force_mean,
            config.force_std,
            history=config.tactile_history,
        )
        self.action_normalizer = ActionNormalizer(config.action_min, config.action_max)
        self.noise_predictor = ConditionalUnet1D(
            input_dim=8,
            global_cond_dim=960,
            diffusion_step_embed_dim=256,
            down_dims=config.unet_dims,
            kernel_size=3,
            n_groups=8,
        )
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=config.diffusion_train_steps,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )
        self.tactile_generator: TactiGenForceFieldModel | None = None
        if config.tactile_source == "tactigen":
            generator_config = dict(config.generator_model_config or {})
            self.tactile_generator = TactiGenForceFieldModel(**generator_config)
            for parameter in self.tactile_generator.parameters():
                parameter.requires_grad_(False)
            self.tactile_generator.eval()
        self.reset()

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, *args, **kwargs):
        """Reject local v1/generated artifacts before generic config parsing."""
        local_path = Path(pretrained_name_or_path)
        config_path = local_path / "config.json"
        if config_path.is_file() and kwargs.get("config") is None:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            if raw.get("tactile_source") == "generated" or raw.get("wrist_roi") is not None:
                raise ValueError(
                    "ACMT-DP v1/generated checkpoint rejected; reconvert from a v3 best.pt "
                    "checkpoint using tactile_source=none|real|tactigen"
                )
            if raw.get("checkpoint_schema_version") != 3:
                raise ValueError(
                    "ACMT-DP checkpoint schema is not v3; reconvert from a v3 best.pt checkpoint"
                )
        manifest_path = local_path / "conversion_manifest.json"
        requested_config = kwargs.get("config")
        if manifest_path.is_file() and requested_config is not None:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            checkpoint_mode = manifest.get("tactile_source")
            requested_mode = getattr(requested_config, "tactile_source", checkpoint_mode)
            if checkpoint_mode in {"none", "real", "tactigen"} and requested_mode != checkpoint_mode:
                raise ValueError(
                    "ACMT-DP checkpoint/runtime tactile mode mismatch: "
                    f"checkpoint={checkpoint_mode!r}, requested={requested_mode!r}; "
                    "use the matching none|real|tactigen checkpoint"
                )
        return super().from_pretrained(pretrained_name_or_path, *args, **kwargs)

    @classmethod
    def _load_as_safetensor(
        cls,
        model: ACMTDPPolicy,
        model_file: str,
        map_location: str,
        strict: bool,
    ) -> ACMTDPPolicy:
        from safetensors import safe_open

        with safe_open(model_file, framework="pt", device="cpu") as archive:
            keys = set(archive.keys())
        required = {
            "tactile_encoder.side_attention.in_proj_weight",
            "tactile_encoder.temporal.weight_ih_l0",
            "tactile_encoder.temporal_norm.weight",
        }
        missing = sorted(required - keys)
        if missing:
            raise ValueError(
                "ACMT-DP v1 checkpoint rejected: missing v3 temporal tactile weights "
                f"{missing}; reconvert from a v3 best.pt checkpoint"
            )
        return super()._load_as_safetensor(model, model_file, map_location, strict)

    def train(self, mode: bool = True) -> ACMTDPPolicy:
        super().train(mode)
        self.visual_encoder.eval()
        if self.tactile_generator is not None:
            self.tactile_generator.eval()
        return self

    def get_optim_params(self) -> dict:
        raise NotImplementedError("Migrated ACMT-DP checkpoints are inference-only")

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        del batch
        raise NotImplementedError("Migrated ACMT-DP checkpoints are inference-only")

    def reset(self) -> None:
        if self.tactile_generator is not None and hasattr(self.tactile_generator, "reset"):
            self.tactile_generator.reset()
        self._history: dict[str, deque[Tensor]] = {
            "rgb": deque(maxlen=4),
            "depth": deque(maxlen=4),
            "lowdim": deque(maxlen=4),
            "pose": deque(maxlen=4),
        }
        self._tactile_history: deque[Tensor] = deque(maxlen=4)
        self._latest_window: dict[str, Tensor] | None = None
        self._previous_action_chunk: Tensor | None = None
        self._observed_batch_size: int | None = None

    @staticmethod
    def _require_shape(key: str, value: Tensor, tail: tuple[int, ...]) -> Tensor:
        if value.ndim != len(tail) + 1 or tuple(value.shape[1:]) != tail:
            raise ValueError(
                f"{key} must have shape [B,{','.join(map(str, tail))}], got {tuple(value.shape)}"
            )
        return value

    @staticmethod
    def _force_side(key: str, value: Tensor) -> Tensor:
        if value.ndim != 4:
            raise ValueError(f"{key} must have four dimensions, got {tuple(value.shape)}")
        if tuple(value.shape[1:]) == (3, 35, 20):
            return value.permute(0, 2, 3, 1).contiguous()
        if tuple(value.shape[1:]) == (35, 20, 3):
            return value
        raise ValueError(f"{key} must be [B,3,35,20] or [B,35,20,3], got {tuple(value.shape)}")

    def _extract_current(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        required = [OBS_STATE, DQ, TAU_J, FT300, GRIPPER_GPO]
        for camera in self.config.wrist_camera_keys:
            required.extend((rgb_key(camera), depth_key(camera)))
        if self.config.tactile_source == "real":
            required.extend((XENSE0, XENSE1))
        if self.config.tactile_source == "tactigen":
            required.append(O_T_EE)
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(f"ACMT-DP observation is missing: {missing}")

        state = self._require_shape(OBS_STATE, batch[OBS_STATE], (8,)).float()
        dq = self._require_shape(DQ, batch[DQ], (7,)).float()
        tau = self._require_shape(TAU_J, batch[TAU_J], (7,)).float()
        wrench = self._require_shape(FT300, batch[FT300], (6,)).float()
        gpo = batch[GRIPPER_GPO]
        if gpo.ndim == 1:
            gpo = gpo[:, None]
        self._require_shape(GRIPPER_GPO, gpo, (1,))
        lowdim = torch.cat([state[:, :7], dq, tau, wrench, gpo.float() / 255.0], dim=-1)

        rgbs, depths = [], []
        for camera in self.config.wrist_camera_keys:
            rgb, depth = prepare_for_frozen_encoder(batch[rgb_key(camera)], batch[depth_key(camera)])
            rgb = self._require_shape(rgb_key(camera), rgb, (3, 128, 128)).float()
            if rgb.numel() and rgb.detach().max() > 2.0:
                rgb = rgb / 255.0
            rgbs.append(rgb)
            depths.append(self._require_shape(depth_key(camera), depth, (1, 128, 128)))
        current: dict[str, Tensor] = {
            "rgb": torch.stack(rgbs, dim=1),
            "depth": torch.stack(depths, dim=1),
            "lowdim": lowdim,
        }
        if self.config.tactile_source == "real":
            current["tactile"] = torch.stack(
                [self._force_side(XENSE0, batch[XENSE0]), self._force_side(XENSE1, batch[XENSE1])],
                dim=1,
            )
        if self.config.tactile_source == "tactigen":
            current["pose"] = self._require_shape(O_T_EE, batch[O_T_EE], (4, 4)).float()
        return current

    def _append_history(self, current: dict[str, Tensor]) -> dict[str, Tensor]:
        batch_size = current["lowdim"].shape[0]
        if self._observed_batch_size is not None and self._observed_batch_size != batch_size:
            raise ValueError("ACMT-DP stateful online inference requires a fixed batch size")
        self._observed_batch_size = batch_size
        for key in ("rgb", "depth", "lowdim"):
            if not self._history[key]:
                self._history[key].extend(current[key] for _ in range(4))
            else:
                self._history[key].append(current[key])
        if self.config.tactile_source == "tactigen":
            pose = current["pose"]
            if not self._history["pose"]:
                self._history["pose"].extend(pose for _ in range(4))
            else:
                self._history["pose"].append(pose)
        if self.config.tactile_source == "real":
            tactile = current["tactile"]
            if not self._tactile_history:
                self._tactile_history.extend(tactile for _ in range(4))
            else:
                self._tactile_history.append(tactile)
        elif not self._tactile_history:
            self._tactile_history.extend(
                torch.zeros(batch_size, 2, 35, 20, 3, device=current["lowdim"].device) for _ in range(4)
            )
        return self._window()

    def _window(self) -> dict[str, Tensor]:
        if not self._history["lowdim"] or not self._tactile_history:
            raise RuntimeError("ACMT-DP observation history is empty")
        window = {key: torch.stack(list(values), dim=1) for key, values in self._history.items() if values}
        window["tactile"] = torch.stack(list(self._tactile_history), dim=1)
        return window

    @torch.no_grad()
    def observe(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Append one control-cycle observation without running diffusion."""
        current = self._extract_current(dict(batch))
        self._latest_window = self._append_history(current)
        return self._latest_window

    @torch.no_grad()
    def _generate_tactile(self, previous: dict[str, Tensor], executed_action: Tensor) -> Tensor:
        if self.tactile_generator is None:
            raise RuntimeError("tactigen mode has no embedded TactiGen generator")
        if executed_action.ndim == 1:
            executed_action = executed_action.unsqueeze(0)
        if tuple(executed_action.shape[1:]) != (8,):
            raise ValueError(f"executed_action must have shape [B,8], got {tuple(executed_action.shape)}")
        if hasattr(self.tactile_generator, "predict_next"):
            generated = self.tactile_generator.predict_next(
                {"rgb": previous["rgb"], "depth": previous["depth"], "lowdim": previous["lowdim"]},
                previous["pose"],
                executed_action,
            )
        else:
            generator_batch = {
                "realsense.cam1_color": previous["rgb"][:, :, 0],
                "realsense.cam1_depth": previous["depth"][:, :, 0],
                "realsense.cam2_color": previous["rgb"][:, :, 1],
                "realsense.cam2_depth": previous["depth"][:, :, 1],
                "robot.q": previous["lowdim"][..., :7],
                "robot.O_T_EE": matrix_to_pose_xyzw_fast(previous["pose"]),
                "ft300.wrench": previous["lowdim"][..., 21:27],
                "gripper.gripper_gPO": previous["lowdim"][..., 27:28],
                "gello.future_q": executed_action[:, None, :7],
                "gello.future_gripper_width": executed_action[:, None, 7:8],
            }
            outputs = self.tactile_generator.inference(generator_batch)
            generated = torch.cat([outputs["force_xy"], outputs["force_z"]], dim=-1)[:, 0]
        expected = (executed_action.shape[0], 2, 35, 20, 3)
        if tuple(generated.shape) != expected:
            raise ValueError(f"TactiGen must return {expected}, got {tuple(generated.shape)}")
        return generated

    @torch.no_grad()
    def notify_action_executed(self, action: Tensor, observation: dict[str, Tensor] | None = None) -> None:
        """Feed the successfully sent action into the TactiGen causal chain."""
        if self.config.tactile_source != "tactigen":
            return
        previous_window = (
            observation if isinstance(observation, dict) and "rgb" in observation else self._latest_window
        )
        if previous_window is None:
            raise RuntimeError("notify_action_executed called before an observation was planned")
        generated = self._generate_tactile(previous_window, action)
        self._tactile_history.append(generated)
        self._previous_action_chunk = action.detach().clone()
        self._latest_window = self._window()

    def encode_observation(self, observation: dict[str, Tensor]) -> Tensor:
        depth1 = observation["depth"][:, :, 0].to(dtype=torch.int32)
        depth2 = observation["depth"][:, :, 1].to(dtype=torch.int32)
        with torch.no_grad():
            memory, mask = self.visual_encoder(
                observation["rgb"][:, :, 0],
                depth1,
                observation["rgb"][:, :, 1],
                depth2,
                ablation="dual",
            )
        visual = self.visual_pool(memory, mask)
        lowdim = self.lowdim_encoder(observation["lowdim"])
        tactile_input = observation["tactile"]
        if self.config.tactile_source == "none":
            tactile_input = torch.zeros_like(tactile_input)
        tactile = self.tactile_encoder(tactile_input)
        return torch.cat([visual, lowdim, tactile], dim=-1)

    @torch.no_grad()
    def _plan(self, observation: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        condition = self.encode_observation(observation)
        expected = (condition.shape[0], self.config.action_horizon, self.config.action_dim)
        if noise is not None and tuple(noise.shape) != expected:
            raise ValueError(f"noise must have shape {expected}, got {tuple(noise.shape)}")
        sample = noise.to(condition) if noise is not None else torch.randn(expected, device=condition.device)
        self.noise_scheduler.set_timesteps(self.config.diffusion_inference_steps, device=condition.device)
        for timestep in self.noise_scheduler.timesteps:
            predicted = self.noise_predictor(sample, timestep, global_cond=condition)
            sample = self.noise_scheduler.step(predicted, timestep, sample).prev_sample
        return self.action_normalizer.unnormalize(sample)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        window = self.observe(batch)
        action = self._plan(window, noise=noise)
        self._previous_action_chunk = action.detach().clone()
        self._latest_window = {key: value.detach().clone() for key, value in window.items()}
        return action

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        return self.predict_action_chunk(batch, noise=noise)[:, 0]

    def causal_state_dict(self) -> dict[str, Any]:
        """Small diagnostic snapshot intentionally excluded from model serialization."""

        return {
            "history_length": len(self._history["lowdim"]),
            "tactile_history_length": len(self._tactile_history),
            "has_previous_plan": self._previous_action_chunk is not None,
        }
