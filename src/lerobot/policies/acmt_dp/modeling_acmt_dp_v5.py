"""Inference-only Native-DP v5 Real-Hybrid policy."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from torch import Tensor

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import OBS_STATE
from lerobot.utils.import_utils import require_package

from .configuration_acmt_dp_v5 import (
    DQ,
    FT300,
    GRIPPER_GPO,
    O_T_EE,
    TAU_J,
    XENSE0,
    XENSE1,
    ACMTDPV5Config,
    depth_key,
    rgb_key,
)
from .modeling_native_v5 import FrameTactileEncoder, NativeV5LinearNormalizer, NativeV5VisionEncoder
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


class ACMTDPV5Policy(PreTrainedPolicy):
    config_class = ACMTDPV5Config
    name = "acmt_dp_v5"

    def __init__(self, config: ACMTDPV5Config, **_: Any) -> None:
        require_package("diffusers", extra="acmt-dp")
        require_package("robomimic", extra="acmt-dp")
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.visual_encoder = NativeV5VisionEncoder(
            use_group_norm=config.use_group_norm,
            num_keypoints=config.spatial_num_keypoints,
        )
        self.tactile_encoder = FrameTactileEncoder(
            config.force_mean,
            config.force_std,
            config.tactile_dim,
        )
        self.normalizer = NativeV5LinearNormalizer(
            config.state_min,
            config.state_max,
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
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=config.diffusion_train_steps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type="epsilon",
        )
        self.model_horizon = ((config.internal_horizon + 7) // 8) * 8
        self.tactile_generator: TactiGenForceFieldModel | None = None
        if config.tactile_source == "tactigen":
            self.tactile_generator = TactiGenForceFieldModel(**dict(config.generator_model_config or {}))
            for parameter in self.tactile_generator.parameters():
                parameter.requires_grad_(False)
            self.tactile_generator.eval()
        self.reset()
        self.eval()

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, *args, **kwargs):
        local_path = Path(pretrained_name_or_path)
        config_path = local_path / "config.json"
        if config_path.is_file() and kwargs.get("config") is None:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            if raw.get("tactile_source") == "generated":
                raise ValueError(
                    "generated checkpoints are obsolete; reconvert as v5 tactigen from a real checkpoint"
                )
            if (
                raw.get("checkpoint_schema") != "acmt_dp.native_dp_v5_robomimic_hybrid"
                or raw.get("checkpoint_schema_version") != 5
            ):
                raise ValueError(
                    "checkpoint is not Native-DP v5 Robomimic; local-copy v5 and v3/v4 artifacts "
                    "cannot be loaded by acmt_dp_v5"
                )
            if raw.get("visual_preprocess") != "robomimic_0.2.0_resize240_center216_range":
                raise ValueError(
                    "Native-DP v5 requires Robomimic 0.2.0 resize240_center216_range preprocessing"
                )
            if raw.get("observation_encoder_impl") != "robomimic_0.2.0_official":
                raise ValueError("Native-DP v5 requires the official Robomimic 0.2.0 observation encoder")
        return super().from_pretrained(pretrained_name_or_path, *args, **kwargs)

    @classmethod
    def _load_as_safetensor(
        cls,
        model: ACMTDPV5Policy,
        model_file: str,
        map_location: str,
        strict: bool,
    ) -> ACMTDPV5Policy:
        from safetensors import safe_open

        with safe_open(model_file, framework="pt", device="cpu") as archive:
            keys = set(archive.keys())
        required = {
            "visual_encoder.encoder.obs_nets.top.backbone.nets.0.weight",
            "tactile_encoder.spatial.0.weight",
            "normalizer.params_dict.state.offset",
            "noise_predictor.final_conv.1.weight",
        }
        missing = sorted(required - keys)
        if not (
            "visual_encoder.encoder.obs_nets.top.pool.nets.weight" in keys
            or "visual_encoder.encoder.obs_nets.top.nets.1.nets.weight" in keys
        ):
            missing.append("visual_encoder.encoder.obs_nets.top.pool.nets.weight")
        if missing:
            raise ValueError(f"Native-DP v5 checkpoint is missing required weights: {missing}")
        return super()._load_as_safetensor(model, model_file, map_location, strict)

    def train(self, mode: bool = True) -> ACMTDPV5Policy:
        super().train(mode)
        if self.tactile_generator is not None:
            self.tactile_generator.eval()
        return self

    def get_optim_params(self) -> dict:
        raise NotImplementedError("Native-DP v5 migrated checkpoints are inference-only")

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        del batch
        raise NotImplementedError("Native-DP v5 migrated checkpoints are inference-only")

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
        self._fixed_initial_noise: Tensor | None = None
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
        if tuple(value.shape[-2:]) not in {(480, 640), (240, 320)}:
            raise ValueError(f"{key} must be 480x640 or 240x320 (4:3), got {tuple(value.shape[-2:])}")
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
        if tuple(value.shape[-2:]) != (480, 640):
            raise ValueError(f"{key} must be 480x640 for TactiGen, got {tuple(value.shape[-2:])}")
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
            raise KeyError(f"ACMT-DP v5 observation is missing: {sorted(set(missing))}")
        state_raw = batch[OBS_STATE]
        if state_raw.ndim != 2 or state_raw.shape[1] not in {7, 8}:
            raise ValueError(f"{OBS_STATE} must be [B,7] or [B,8], got {tuple(state_raw.shape)}")
        q = state_raw[:, :7].float()
        gpo = batch[GRIPPER_GPO]
        if gpo.ndim == 1:
            gpo = gpo[:, None]
        gpo = self._require_shape(GRIPPER_GPO, gpo, (1,)).float()
        current: dict[str, Tensor] = {
            "rgb": torch.stack(
                [self._rgb(batch[rgb_key(camera)], rgb_key(camera)) for camera in self.config.camera_keys],
                dim=1,
            ),
            "state": torch.cat([q, gpo / 255.0], dim=-1),
        }
        if self.config.tactile_source == "real":
            current["tactile"] = torch.stack(
                [_force_side(XENSE0, batch[XENSE0]), _force_side(XENSE1, batch[XENSE1])], dim=1
            )
        if self.config.tactile_source == "tactigen":
            gen_rgbs, gen_depths = [], []
            for camera in self.config.wrist_camera_keys:
                rgb = current["rgb"][:, self.config.camera_keys.index(camera)]
                depth = self._depth(batch[depth_key(camera)], depth_key(camera))
                rgb, depth = prepare_for_frozen_encoder(rgb, depth)
                gen_rgbs.append(rgb)
                gen_depths.append(depth)
            current.update(
                {
                    "gen_rgb": torch.stack(gen_rgbs, dim=1),
                    "gen_depth": torch.stack(gen_depths, dim=1),
                    "gen_lowdim": torch.cat(
                        [
                            q,
                            self._require_shape(DQ, batch[DQ], (7,)).float(),
                            self._require_shape(TAU_J, batch[TAU_J], (7,)).float(),
                            self._require_shape(FT300, batch[FT300], (6,)).float(),
                            gpo / 255.0,
                        ],
                        dim=-1,
                    ),
                    "pose": self._require_shape(O_T_EE, batch[O_T_EE], (4, 4)).float(),
                }
            )
        return current

    def _append_history(self, current: dict[str, Tensor]) -> dict[str, Tensor]:
        batch_size = current["state"].shape[0]
        if self._observed_batch_size is not None and self._observed_batch_size != batch_size:
            raise ValueError("ACMT-DP v5 stateful inference requires a fixed batch size")
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
            if not self._tactile_history:
                self._tactile_history.extend(current["tactile"] for _ in range(4))
            else:
                self._tactile_history.append(current["tactile"])
        elif not self._tactile_history:
            self._tactile_history.extend(
                torch.zeros(batch_size, 2, 35, 20, 3, device=current["state"].device, dtype=torch.float32)
                for _ in range(4)
            )
        return self._window()

    def _window(self) -> dict[str, Tensor]:
        if not self._history["state"] or not self._tactile_history:
            raise RuntimeError("ACMT-DP v5 observation history is empty")
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
            raise RuntimeError("v5 TactiGen mode has no embedded generator")
        if executed_action.ndim == 3:
            if tuple(executed_action.shape[1:]) != (16, 8):
                raise ValueError("executed action chunk must be [B,16,8]")
            executed_action = executed_action[:, 0]
        if executed_action.ndim == 1:
            executed_action = executed_action.unsqueeze(0)
        if tuple(executed_action.shape[1:]) != (8,):
            raise ValueError("executed_action must be [B,8]")
        generated = self.tactile_generator.predict_next(
            {
                "rgb": previous["gen_rgb"],
                "depth": previous["gen_depth"],
                "lowdim": previous["gen_lowdim"],
            },
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
        self._tactile_history.append(self._generate_tactile(previous, action))
        self._previous_action_chunk = action.detach().clone()
        self._latest_window = self._window()

    def encode_observation(self, observation: dict[str, Tensor]) -> Tensor:
        state = observation["state"]
        tactile = observation["tactile"]
        if tuple(state.shape[1:]) != (4, 8) or tuple(tactile.shape[1:]) != (4, 2, 35, 20, 3):
            raise ValueError("v5 observation history has invalid state/tactile shape")
        normalized_state = self.normalizer["state"].normalize(state.float())
        visual_state = self.visual_encoder(observation["rgb"], normalized_state)
        if self.config.tactile_source == "none":
            tactile = torch.zeros_like(tactile)
        features = torch.cat(
            [
                visual_state,
                self.tactile_encoder(tactile),
            ],
            dim=-1,
        )
        if features.shape[-1] * features.shape[1] != self.config.global_cond_dim:
            raise RuntimeError("v5 global condition shape mismatch")
        return features.reshape(features.shape[0], -1)

    @torch.no_grad()
    def _plan(
        self,
        observation: dict[str, Tensor],
        noise: Tensor | None = None,
        inference_steps: int | None = None,
    ) -> Tensor:
        condition = self.encode_observation(observation)
        expected = (condition.shape[0], self.config.internal_horizon, self.config.action_dim)
        if noise is None:
            if self._fixed_initial_noise is None or tuple(self._fixed_initial_noise.shape) != expected:
                self._fixed_initial_noise = torch.randn(
                    expected,
                    device=condition.device,
                    dtype=condition.dtype,
                )
            noise = self._fixed_initial_noise
        elif tuple(noise.shape) != expected:
            raise ValueError(f"noise must have shape {expected}")
        else:
            # An explicit seed/noise supplied by an evaluator becomes the
            # episode's reusable initial state as well.
            self._fixed_initial_noise = (
                noise.detach().to(device=condition.device, dtype=condition.dtype).clone()
            )
        sample = F.pad(
            noise.to(device=condition.device, dtype=condition.dtype),
            (0, 0, 0, self.model_horizon - self.config.internal_horizon),
            mode="replicate",
        ).clone()
        steps = int(inference_steps or self.config.diffusion_inference_steps)
        if steps <= 0:
            raise ValueError("inference_steps must be positive")
        self.noise_scheduler.set_timesteps(steps, device=condition.device)
        for timestep in self.noise_scheduler.timesteps:
            predicted = self.noise_predictor(sample, timestep, global_cond=condition)
            sample = self.noise_scheduler.step(predicted, timestep, sample, eta=0.0).prev_sample
        raw = self.normalizer["action"].unnormalize(sample[..., : self.config.internal_horizon, :])
        return raw[..., self.config.pad_before : self.config.pad_before + self.config.public_pred_horizon, :]

    @torch.no_grad()
    def predict_action_chunk(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
        inference_steps: int | None = None,
    ) -> Tensor:
        window = self.observe(batch)
        action = self._plan(window, noise=noise, inference_steps=inference_steps)
        self._previous_action_chunk = action.detach().clone()
        self._latest_window = {key: value.detach().clone() for key, value in window.items()}
        return action

    @torch.no_grad()
    def select_action(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
        inference_steps: int | None = None,
    ) -> Tensor:
        return self.predict_action_chunk(batch, noise=noise, inference_steps=inference_steps)[:, 0]

    def action_execution_slice(self, action_chunk: Tensor) -> Tensor:
        if action_chunk.ndim < 2 or tuple(action_chunk.shape[-2:]) != (16, 8):
            raise ValueError("v5 action chunk must end in [16,8]")
        return action_chunk[..., :8, :]

    def causal_state_dict(self) -> dict[str, Any]:
        return {
            "history_length": len(self._history["state"]),
            "tactile_history_length": len(self._tactile_history),
            "has_previous_plan": self._previous_action_chunk is not None,
        }
