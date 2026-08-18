# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inference-only Native-DP v4 ACMT-DP policy."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

import torch
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
from .modeling_native_v4 import FrameTactileEncoder, NativeLinearNormalizer, NativeVisionEncoder
from .modeling_tactile_generator import TactiGenForceFieldModel
from .modeling_unet import ConditionalUnet1D
from .visual_preprocess import prepare_for_frozen_encoder


def _force_side(key: str, value: Tensor) -> Tensor:
    if value.ndim != 4:
        raise ValueError(f"{key} must have four dimensions, got {tuple(value.shape)}")
    if tuple(value.shape[1:]) == (3, 35, 20):
        return value.permute(0, 2, 3, 1).contiguous()
    if tuple(value.shape[1:]) == (35, 20, 3):
        return value.contiguous()
    raise ValueError(f"{key} must be [B,3,35,20] or [B,35,20,3], got {tuple(value.shape)}")


class ACMTDPPolicy(PreTrainedPolicy):
    """Native-DP v4 policy with none/real/tactigen causal tactile modes."""

    config_class = ACMTDPConfig
    name = "acmt_dp"

    def __init__(self, config: ACMTDPConfig, **_: Any) -> None:
        require_package("diffusers", extra="acmt-dp")
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.visual_encoder = NativeVisionEncoder(weights=None, frozen=False)
        self.tactile_encoder = FrameTactileEncoder(config.force_mean, config.force_std, config.tactile_dim)
        self.normalizer = NativeLinearNormalizer(
            config.state_mean,
            config.state_std,
            config.action_min,
            config.action_max,
        )
        self.noise_predictor = ConditionalUnet1D(
            input_dim=config.action_dim,
            global_cond_dim=config.global_cond_dim,
            diffusion_step_embed_dim=config.diffusion_step_embed_dim,
            down_dims=config.unet_dims,
            kernel_size=config.unet_kernel_size,
            n_groups=8,
            cond_predict_scale=config.cond_predict_scale,
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
        # This migrated artifact has no training contract; deterministic
        # inference (especially ResNet BatchNorm) is the safe default even
        # when callers construct the policy directly rather than via
        # ``from_pretrained``.
        self.eval()

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, *args, **kwargs):
        """Reject v1-v3 artifacts before generic LeRobot config parsing."""

        local_path = Path(pretrained_name_or_path)
        config_path = local_path / "config.json"
        if config_path.is_file() and kwargs.get("config") is None:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            if raw.get("tactile_source") == "generated" or raw.get("wrist_roi") is not None:
                raise ValueError(
                    "ACMT-DP legacy/generated checkpoint rejected; reconvert a schema="
                    "acmt_dp.native_dp_v4 scratch checkpoint"
                )
            if (
                raw.get("checkpoint_schema_version") != 4
                or raw.get("checkpoint_schema") != "acmt_dp.native_dp_v4"
            ):
                raise ValueError(
                    "ACMT-DP checkpoint is not Native-DP v4; v3 center480/DFormer artifacts "
                    "must be reconverted from a v4 scratch best.pt"
                )
            if raw.get("vision_mode", "scratch") != "scratch":
                raise ValueError("ACMT-DP v4 LeRobot loading only accepts scratch checkpoints")
        manifest_path = local_path / "conversion_manifest.json"
        requested_config = kwargs.get("config")
        if manifest_path.is_file() and requested_config is not None:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("checkpoint_schema") not in (None, "acmt_dp.native_dp_v4"):
                raise ValueError("ACMT-DP conversion manifest is not v4; reconvert the checkpoint")
            checkpoint_mode = manifest.get("tactile_source")
            requested_mode = getattr(requested_config, "tactile_source", checkpoint_mode)
            if checkpoint_mode in {"none", "real", "tactigen"} and requested_mode != checkpoint_mode:
                raise ValueError(
                    "ACMT-DP checkpoint/runtime tactile mode mismatch: "
                    f"checkpoint={checkpoint_mode!r}, requested={requested_mode!r}"
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
            "visual_encoder.obs_encoder.key_model_map.rgb.conv1.weight",
            "tactile_encoder.spatial.0.weight",
            "normalizer.params_dict.state.offset",
        }
        missing = sorted(required - keys)
        if missing:
            raise ValueError(
                "ACMT-DP checkpoint is missing Native-DP v4 weights "
                f"{missing}; v3 temporal/center480 checkpoints must be reconverted"
            )
        if any(
            key.startswith("tactile_encoder.") and ("temporal" in key or "side_attention" in key)
            for key in keys
        ):
            raise ValueError("ACMT-DP v3 temporal tactile weights are incompatible with Native-DP v4")
        return super()._load_as_safetensor(model, model_file, map_location, strict)

    def train(self, mode: bool = True) -> ACMTDPPolicy:
        super().train(mode)
        if self.tactile_generator is not None:
            self.tactile_generator.eval()
        return self

    def get_optim_params(self) -> dict:
        raise NotImplementedError("Native-DP v4 migrated checkpoints are inference-only")

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        del batch
        raise NotImplementedError("Native-DP v4 migrated checkpoints are inference-only")

    def reset(self) -> None:
        if self.tactile_generator is not None and hasattr(self.tactile_generator, "reset"):
            self.tactile_generator.reset()
        self._history: dict[str, deque[Tensor]] = {
            "rgb": deque(maxlen=4),
            "state": deque(maxlen=4),
            "gen_rgb": deque(maxlen=4),
            "gen_depth": deque(maxlen=4),
            "gen_lowdim": deque(maxlen=4),
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
    def _rgb(value: Tensor, key: str) -> Tensor:
        if value.ndim == 3:
            if value.shape[0] == 3:
                value = value.unsqueeze(0)
            elif value.shape[-1] == 3:
                value = value.unsqueeze(0).movedim(-1, 1)
        elif value.ndim == 4 and value.shape[-1] == 3 and value.shape[1] != 3:
            value = value.movedim(-1, 1)
        if value.ndim != 4 or value.shape[1] != 3:
            raise ValueError(f"{key} must be BCHW/BHWC RGB, got {tuple(value.shape)}")
        if tuple(value.shape[-2:]) not in {(480, 640), (224, 224)}:
            raise ValueError(f"{key} must be 480x640 or 224x224, got {tuple(value.shape[-2:])}")
        return value.contiguous()

    @staticmethod
    def _depth(value: Tensor, key: str) -> Tensor:
        if value.ndim == 3:
            if value.shape[0] == 1:
                value = value.unsqueeze(0)
            elif value.shape[-1] == 1:
                value = value.unsqueeze(0).movedim(-1, 1)
        elif value.ndim == 4 and value.shape[-1] == 1 and value.shape[1] != 1:
            value = value.movedim(-1, 1)
        if value.ndim != 4 or value.shape[1] != 1:
            raise ValueError(f"{key} must be BCHW/BHWC depth, got {tuple(value.shape)}")
        if tuple(value.shape[-2:]) not in {(480, 640), (128, 128)}:
            raise ValueError(f"{key} must be 480x640 or 128x128, got {tuple(value.shape[-2:])}")
        return value.contiguous()

    def _extract_current(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        required = [OBS_STATE, GRIPPER_GPO]
        if self.config.tactile_source == "real":
            required.extend((XENSE0, XENSE1))
        if self.config.tactile_source == "tactigen":
            required.extend((DQ, TAU_J, FT300, O_T_EE))
        missing = [key for key in required if key not in batch]
        missing.extend(rgb_key(camera) for camera in self.config.camera_keys if rgb_key(camera) not in batch)
        if self.config.tactile_source == "tactigen":
            missing.extend(
                depth_key(camera)
                for camera in self.config.wrist_camera_keys
                if depth_key(camera) not in batch
            )
        if missing:
            raise KeyError(f"ACMT-DP v4 observation is missing: {sorted(set(missing))}")

        state_raw = batch[OBS_STATE]
        if state_raw.ndim != 2 or state_raw.shape[1] not in {7, 8}:
            raise ValueError(f"{OBS_STATE} must be [B,7] or [B,8], got {tuple(state_raw.shape)}")
        q = state_raw[:, :7].float()
        gpo = batch[GRIPPER_GPO]
        if gpo.ndim == 1:
            gpo = gpo[:, None]
        gpo = self._require_shape(GRIPPER_GPO, gpo, (1,)).float()
        policy_state = torch.cat([q, gpo / 255.0], dim=-1)
        rgbs = [self._rgb(batch[rgb_key(camera)], rgb_key(camera)) for camera in self.config.camera_keys]
        current: dict[str, Tensor] = {"rgb": torch.stack(rgbs, dim=1), "state": policy_state}

        if self.config.tactile_source == "real":
            current["tactile"] = torch.stack(
                [_force_side(XENSE0, batch[XENSE0]), _force_side(XENSE1, batch[XENSE1])], dim=1
            )
        if self.config.tactile_source == "tactigen":
            gen_rgbs, gen_depths = [], []
            for camera in self.config.wrist_camera_keys:
                rgb = rgbs[self.config.camera_keys.index(camera)]
                depth = self._depth(batch[depth_key(camera)], depth_key(camera))
                if tuple(rgb.shape[-2:]) == (224, 224) and tuple(depth.shape[-2:]) == (128, 128):
                    # Synthetic/offline callers may provide already-cropped
                    # policy RGB beside generator-sized depth.  The deployed
                    # hardware path uses raw 480x640 and takes the canonical
                    # center480 branch below.
                    rgb = F.interpolate(rgb.float(), size=(128, 128), mode="bilinear", align_corners=False)
                else:
                    rgb, depth = prepare_for_frozen_encoder(rgb, depth)
                gen_rgbs.append(rgb)
                gen_depths.append(depth)
            dq = self._require_shape(DQ, batch[DQ], (7,)).float()
            tau = self._require_shape(TAU_J, batch[TAU_J], (7,)).float()
            wrench = self._require_shape(FT300, batch[FT300], (6,)).float()
            current.update(
                {
                    "gen_rgb": torch.stack(gen_rgbs, dim=1),
                    "gen_depth": torch.stack(gen_depths, dim=1),
                    "gen_lowdim": torch.cat([q, dq, tau, wrench, gpo / 255.0], dim=-1),
                    "pose": self._require_shape(O_T_EE, batch[O_T_EE], (4, 4)).float(),
                }
            )
        return current

    def _append_history(self, current: dict[str, Tensor]) -> dict[str, Tensor]:
        batch_size = current["state"].shape[0]
        if self._observed_batch_size is not None and self._observed_batch_size != batch_size:
            raise ValueError("ACMT-DP stateful online inference requires a fixed batch size")
        self._observed_batch_size = batch_size
        keys = ("rgb", "state") + (
            ("gen_rgb", "gen_depth", "gen_lowdim", "pose") if self.config.tactile_source == "tactigen" else ()
        )
        for key in keys:
            if not self._history[key]:
                self._history[key].extend(current[key] for _ in range(4))
            else:
                self._history[key].append(current[key])
        if self.config.tactile_source == "real":
            tactile = current["tactile"]
            if not self._tactile_history:
                self._tactile_history.extend(tactile for _ in range(4))
            else:
                self._tactile_history.append(tactile)
        elif not self._tactile_history:
            self._tactile_history.extend(
                torch.zeros(batch_size, 2, 35, 20, 3, device=current["state"].device, dtype=torch.float32)
                for _ in range(4)
            )
        return self._window()

    def _window(self) -> dict[str, Tensor]:
        if not self._history["state"] or not self._tactile_history:
            raise RuntimeError("ACMT-DP v4 observation history is empty")
        window = {key: torch.stack(list(values), dim=1) for key, values in self._history.items() if values}
        window["tactile"] = torch.stack(list(self._tactile_history), dim=1)
        return window

    @torch.no_grad()
    def observe(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        current = self._extract_current(dict(batch))
        self._latest_window = self._append_history(current)
        return self._latest_window

    @torch.no_grad()
    def _generate_tactile(self, previous: dict[str, Tensor], executed_action: Tensor) -> Tensor:
        if self.tactile_generator is None:
            raise RuntimeError("tactigen mode has no embedded TactiGen generator")
        if executed_action.ndim == 3:
            if tuple(executed_action.shape[1:]) != (16, 8):
                raise ValueError(
                    f"executed action chunk must be [B,16,8] when 3-D, got {tuple(executed_action.shape)}"
                )
            executed_action = executed_action[:, 0]
        if executed_action.ndim == 1:
            executed_action = executed_action.unsqueeze(0)
        if tuple(executed_action.shape[1:]) != (8,):
            raise ValueError(f"executed_action must have shape [B,8], got {tuple(executed_action.shape)}")
        generated = self.tactile_generator.predict_next(
            {"rgb": previous["gen_rgb"], "depth": previous["gen_depth"], "lowdim": previous["gen_lowdim"]},
            previous["pose"],
            executed_action,
        )
        expected = (executed_action.shape[0], 2, 35, 20, 3)
        if tuple(generated.shape) != expected:
            raise ValueError(f"TactiGen must return {expected}, got {tuple(generated.shape)}")
        return generated

    @torch.no_grad()
    def notify_action_executed(self, action: Tensor, observation: dict[str, Tensor] | None = None) -> None:
        if self.config.tactile_source != "tactigen":
            return
        previous = (
            observation if isinstance(observation, dict) and "gen_rgb" in observation else self._latest_window
        )
        if previous is None:
            raise RuntimeError("notify_action_executed called before an observation was planned")
        generated = self._generate_tactile(previous, action)
        self._tactile_history.append(generated)
        self._previous_action_chunk = action.detach().clone()
        self._latest_window = self._window()

    def encode_observation(self, observation: dict[str, Tensor]) -> Tensor:
        rgb = observation["rgb"]
        if rgb.ndim != 6 or tuple(rgb.shape[1:3]) != (4, 4):
            raise ValueError(f"rgb must be [B,4,4,3,H,W], got {tuple(rgb.shape)}")
        visual = self.visual_encoder(rgb)
        state = observation["state"]
        if state.ndim != 3 or tuple(state.shape[1:]) != (4, 8):
            raise ValueError(f"state must be [B,4,8], got {tuple(state.shape)}")
        state = self.normalizer["state"].normalize(state.float())
        tactile = observation["tactile"]
        if tactile.ndim != 6 or tuple(tactile.shape[1:]) != (4, 2, 35, 20, 3):
            raise ValueError(f"tactile must be [B,4,2,35,20,3], got {tuple(tactile.shape)}")
        if self.config.tactile_source == "none":
            tactile = torch.zeros_like(tactile)
        tactile_features = self.tactile_encoder(tactile)
        condition = torch.cat([visual.reshape(visual.shape[0], 4, -1), state, tactile_features], dim=-1)
        if condition.shape[-1] * condition.shape[1] != self.config.global_cond_dim:
            raise RuntimeError(
                f"unexpected condition shape {tuple(condition.shape)} for {self.config.global_cond_dim}"
            )
        return condition.reshape(condition.shape[0], -1)

    @torch.no_grad()
    def _plan(self, observation: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        condition = self.encode_observation(observation)
        expected = (condition.shape[0], self.config.pred_horizon, self.config.action_dim)
        if noise is not None and tuple(noise.shape) != expected:
            raise ValueError(f"noise must have shape {expected}, got {tuple(noise.shape)}")
        sample = (
            noise.to(device=condition.device, dtype=condition.dtype).clone()
            if noise is not None
            else torch.randn(expected, device=condition.device, dtype=condition.dtype)
        )
        self.noise_scheduler.set_timesteps(self.config.diffusion_inference_steps, device=condition.device)
        for timestep in self.noise_scheduler.timesteps:
            predicted = self.noise_predictor(sample, timestep, global_cond=condition)
            sample = self.noise_scheduler.step(predicted, timestep, sample).prev_sample
        return self.normalizer["action"].unnormalize(sample)

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

    def action_execution_slice(self, action_chunk: Tensor) -> Tensor:
        """Return the eight commands sent before the next replanning boundary."""

        if action_chunk.ndim < 2 or tuple(action_chunk.shape[-2:]) != (16, 8):
            raise ValueError("action chunk must end in [16,8]")
        return action_chunk[..., : self.config.action_execution_horizon, :]

    def causal_state_dict(self) -> dict[str, Any]:
        return {
            "history_length": len(self._history["state"]),
            "tactile_history_length": len(self._tactile_history),
            "has_previous_plan": self._previous_action_chunk is not None,
        }
