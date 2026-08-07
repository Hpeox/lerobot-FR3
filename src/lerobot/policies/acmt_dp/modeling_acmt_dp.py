# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
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
from .modeling_components import DFormerv2DualRGBDEncoder, TinyDualRGBDEncoder
from .modeling_tactile_generator import TactiGenForceFieldModel
from .modeling_unet import ConditionalUnet1D


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


class TactileGridEncoder(nn.Module):
    def __init__(self, mean: tuple[float, ...], std: tuple[float, ...], dim: int = 160) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))
        self.side_embedding = nn.Parameter(torch.zeros(1, 2, dim))
        nn.init.trunc_normal_(self.side_embedding, std=0.02)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 5, padding=2),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, dim, 3, padding=1),
            nn.GroupNorm(8, dim),
            nn.GELU(),
        )
        self.attention = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, force: Tensor) -> Tensor:
        batch = force.shape[0]
        normalized = (force.float() - self.mean) / self.std.clamp_min(1e-6)
        sides = normalized.permute(0, 1, 4, 2, 3).reshape(-1, 3, 35, 20)
        pooled = F.adaptive_avg_pool2d(self.conv(sides), 1).flatten(1)
        tokens = pooled.reshape(batch, 2, -1) + self.side_embedding
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        return self.norm(tokens + attended).mean(1)


def matrix_to_pose_xyzw(matrices: Tensor) -> Tensor:
    """Match the legacy ACMT-DP 4x4 -> xyz+xyzw conversion exactly."""

    original_device = matrices.device
    mats = matrices.detach().cpu().numpy()
    flat = mats.reshape(-1, 4, 4)
    quaternions = np.zeros((len(flat), 4), np.float32)
    for index, matrix in enumerate(flat):
        rotation = matrix[:3, :3]
        trace = float(np.trace(rotation))
        if trace > 0:
            scale = np.sqrt(trace + 1.0) * 2
            qw = 0.25 * scale
            qx = (rotation[2, 1] - rotation[1, 2]) / scale
            qy = (rotation[0, 2] - rotation[2, 0]) / scale
            qz = (rotation[1, 0] - rotation[0, 1]) / scale
        else:
            axis = int(np.argmax(np.diag(rotation)))
            if axis == 0:
                scale = np.sqrt(max(1e-12, 1 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])) * 2
                qw = (rotation[2, 1] - rotation[1, 2]) / scale
                qx = 0.25 * scale
                qy = (rotation[0, 1] + rotation[1, 0]) / scale
                qz = (rotation[0, 2] + rotation[2, 0]) / scale
            elif axis == 1:
                scale = np.sqrt(max(1e-12, 1 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])) * 2
                qw = (rotation[0, 2] - rotation[2, 0]) / scale
                qx = (rotation[0, 1] + rotation[1, 0]) / scale
                qy = 0.25 * scale
                qz = (rotation[1, 2] + rotation[2, 1]) / scale
            else:
                scale = np.sqrt(max(1e-12, 1 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])) * 2
                qw = (rotation[1, 0] - rotation[0, 1]) / scale
                qx = (rotation[0, 2] + rotation[2, 0]) / scale
                qy = (rotation[1, 2] + rotation[2, 1]) / scale
                qz = 0.25 * scale
        quaternions[index] = (qx, qy, qz, qw)
    quaternions /= np.maximum(np.linalg.norm(quaternions, axis=-1, keepdims=True), 1e-6)
    xyz = flat[:, :3, 3].astype(np.float32)
    pose = np.concatenate([xyz, quaternions], axis=-1).reshape(*mats.shape[:-2], 7)
    return torch.from_numpy(pose).to(original_device)


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
        self.tactile_encoder = TactileGridEncoder(config.force_mean, config.force_std)
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
        if config.tactile_source == "generated":
            generator_config = dict(config.generator_model_config or {})
            self.tactile_generator = TactiGenForceFieldModel(**generator_config)
            for parameter in self.tactile_generator.parameters():
                parameter.requires_grad_(False)
            self.tactile_generator.eval()
        self.reset()

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
        self._history: dict[str, deque[Tensor]] = {
            "rgb": deque(maxlen=4),
            "depth": deque(maxlen=4),
            "lowdim": deque(maxlen=4),
            "pose": deque(maxlen=4),
        }
        self._previous_window: dict[str, Tensor] | None = None
        self._previous_action_chunk: Tensor | None = None

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
        if self.config.tactile_source == "generated":
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
            rgbs.append(self._require_shape(rgb_key(camera), batch[rgb_key(camera)], (3, 128, 128)).float())
            depths.append(
                self._require_shape(depth_key(camera), batch[depth_key(camera)], (1, 128, 128)).float()
            )
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
        if self.config.tactile_source == "generated":
            current["pose"] = self._require_shape(O_T_EE, batch[O_T_EE], (4, 4)).float()
        return current

    def _append_history(self, current: dict[str, Tensor]) -> dict[str, Tensor]:
        for key in ("rgb", "depth", "lowdim"):
            if not self._history[key]:
                self._history[key].extend(current[key] for _ in range(4))
            else:
                self._history[key].append(current[key])
        if self.config.tactile_source == "generated":
            pose = current["pose"]
            if not self._history["pose"]:
                self._history["pose"].extend(pose for _ in range(4))
            else:
                self._history["pose"].append(pose)
        return {key: torch.stack(list(values), dim=1) for key, values in self._history.items() if values}

    @torch.no_grad()
    def _generate_tactile(self, previous: dict[str, Tensor], action_chunk: Tensor) -> Tensor:
        if self.tactile_generator is None:
            raise RuntimeError("generated mode has no embedded ACMT generator")
        first = action_chunk[:, :1]
        generator_batch = {
            "realsense.cam1_color": previous["rgb"][:, :, 0],
            "realsense.cam1_depth": previous["depth"][:, :, 0],
            "realsense.cam2_color": previous["rgb"][:, :, 1],
            "realsense.cam2_depth": previous["depth"][:, :, 1],
            "robot.q": previous["lowdim"][..., :7],
            "robot.O_T_EE": matrix_to_pose_xyzw(previous["pose"]),
            "ft300.wrench": previous["lowdim"][..., 21:27],
            "gripper.gripper_gPO": previous["lowdim"][..., 27:28],
            "gello.future_q": first[..., :7],
            "gello.future_gripper_width": first[..., 7:8],
        }
        outputs = self.tactile_generator.inference(generator_batch)
        return torch.cat([outputs["force_xy"], outputs["force_z"]], dim=-1)[:, 0]

    def encode_observation(self, observation: dict[str, Tensor]) -> Tensor:
        with torch.no_grad():
            memory, mask = self.visual_encoder(
                observation["rgb"][:, :, 0],
                observation["depth"][:, :, 0],
                observation["rgb"][:, :, 1],
                observation["depth"][:, :, 1],
                ablation="dual",
            )
        visual = self.visual_pool(memory, mask)
        lowdim = self.lowdim_encoder(observation["lowdim"])
        if self.config.tactile_source == "none":
            tactile = torch.zeros(lowdim.shape[0], 160, device=lowdim.device, dtype=lowdim.dtype)
            tactile = tactile + sum(parameter.sum() * 0.0 for parameter in self.tactile_encoder.parameters())
        else:
            tactile = self.tactile_encoder(observation["tactile"])
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
        current = self._extract_current(dict(batch))
        window = self._append_history(current)
        if self.config.tactile_source == "none":
            window["tactile"] = torch.zeros(
                window["lowdim"].shape[0], 2, 35, 20, 3, device=window["lowdim"].device
            )
        elif self.config.tactile_source == "real":
            window["tactile"] = current["tactile"]
        elif self._previous_window is None or self._previous_action_chunk is None:
            window["tactile"] = torch.zeros(
                window["lowdim"].shape[0], 2, 35, 20, 3, device=window["lowdim"].device
            )
        else:
            window["tactile"] = self._generate_tactile(self._previous_window, self._previous_action_chunk)
        action = self._plan(window, noise=noise)
        if self.config.tactile_source == "generated":
            self._previous_window = {
                key: value.detach().clone() for key, value in window.items() if key != "tactile"
            }
            self._previous_action_chunk = action.detach().clone()
        return action

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        return self.predict_action_chunk(batch, noise=noise)[:, 0]

    def causal_state_dict(self) -> dict[str, Any]:
        """Small diagnostic snapshot intentionally excluded from model serialization."""

        return {
            "history_length": len(self._history["lowdim"]),
            "has_previous_plan": self._previous_action_chunk is not None,
        }
