"""Native ACT + DINOv2-S/14 ACMT-ACTv2 policy.

The v2 model keeps LeRobot's ACT VAE/Transformer/action-head implementation
and adds only one tactile token plus four-camera DINOv2 spatial tokens.  It
predicts normalized absolute actions ``[B,100,8]``; the policy wrapper applies
LeRobot's native online temporal ensembler and returns one action per tick.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import einops
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.configs import PreTrainedConfig
from lerobot.policies.act.modeling_act import ACT, ACTTemporalEnsembler
from lerobot.policies.acmt_act.modeling_acmt_act import (
    ACMTACTileEncoder,
    GEN_DEPTH,
    GEN_LOWDIM,
    GEN_POSE,
    GEN_RGB,
    TACTILE,
    _ACMTGeneratorRuntime,
    _force_side,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from .configuration_acmt_actv2 import ACMTACTV2Config
from .dinov2 import DINOv2SpatialBackbone
from lerobot.policies.acmt_act.configuration_acmt_act import (
    DQ,
    FT300,
    GRIPPER_GPO,
    O_T_EE,
    TAU_J,
    XENSE0,
    XENSE1,
    depth_key,
    rgb_key,
)


class ACMTACTV2Model(ACT):
    """ACT transformer with shared DINOv2 spatial tokens and one tactile token."""

    def __init__(self, config: ACMTACTV2Config) -> None:
        # ACT allocates a torchvision ResNet in its constructor.  Use a small
        # parameter-free placeholder, then replace it with the single DINOv2
        # module.  The same config object is restored before returning so its
        # serialized vision contract remains DINO-specific.
        original_backbone = config.vision_backbone
        original_weights = config.pretrained_backbone_weights
        config.vision_backbone = "resnet18"
        config.pretrained_backbone_weights = None
        super().__init__(config)
        config.vision_backbone = original_backbone
        config.pretrained_backbone_weights = original_weights

        del self.backbone
        del self.encoder_img_feat_input_proj
        self.backbone = DINOv2SpatialBackbone(
            config.dinov2_model_name,
            image_size=config.dinov2_image_size,
            patch_size=config.dinov2_patch_size,
            token_dim=config.dinov2_token_dim,
            pretrained=config.dinov2_pretrained,
            checkpoint=config.dinov2_checkpoint,
            require_checkpoint=config.require_dinov2_checkpoint,
        )
        self.encoder_img_feat_input_proj = nn.Linear(
            config.dinov2_token_dim,
            config.dinov2_projection_dim,
        )
        self.camera_embedding = nn.Embedding(len(config.camera_keys), config.dim_model)
        self.tactile_encoder = ACMTACTileEncoder(config.force_mean, config.force_std)
        self.encoder_tactile_input_proj = nn.Linear(config.tactile_feature_dim, config.dim_model)

        # ACT has latent + state 1-D positions.  Add exactly one tactile
        # position; camera patch positions remain the native ACT 2-D sinusoid.
        old_position = self.encoder_1d_feature_pos_embed
        self.encoder_1d_feature_pos_embed = nn.Embedding(3, config.dim_model)
        with torch.no_grad():
            self.encoder_1d_feature_pos_embed.weight[: old_position.num_embeddings].copy_(old_position.weight)
            nn.init.normal_(self.encoder_1d_feature_pos_embed.weight[old_position.num_embeddings :], std=0.02)

        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # DINO is intentionally frozen/eval for the complete 200k-step run.
        self.backbone.train(False)
        return self

    def _dino_tokens(self, images: list[Tensor]) -> tuple[Tensor, Tensor]:
        if len(images) != 4:
            raise ValueError(f"ACMT-ACTv2 requires four camera images, got {len(images)}")
        batch_size = images[0].shape[0]
        if any(image.ndim != 4 or image.shape[0] != batch_size for image in images):
            raise ValueError("all ACMT-ACTv2 camera images must have the same [B,C,H,W] shape")
        stacked = torch.stack(images, dim=1)
        if tuple(stacked.shape[2:]) != (3, 336, 448):
            raise ValueError(f"ACMT-ACTv2 DINO images must be [B,4,3,336,448], got {tuple(stacked.shape)}")
        flat = stacked.reshape(batch_size * 4, 3, 336, 448)
        raw = self.backbone(flat)
        expected_tokens = self.config.dinov2_token_count
        if tuple(raw.shape) != (batch_size * 4, expected_tokens, self.config.dinov2_token_dim):
            raise RuntimeError(
                "DINOv2 token shape mismatch: "
                f"expected {(batch_size * 4, expected_tokens, self.config.dinov2_token_dim)}, got {tuple(raw.shape)}"
            )
        projected = self.encoder_img_feat_input_proj(raw)
        projected = projected.reshape(batch_size, 4, expected_tokens, self.config.dim_model)
        camera_ids = torch.arange(4, device=projected.device)
        camera = self.camera_embedding(camera_ids).view(1, 4, 1, self.config.dim_model)
        projected = projected + camera

        # Reuse the exact ACT sinusoidal 2-D position implementation.  A
        # temporary feature-map view is only for position construction; the
        # transformer still receives all spatial tokens individually.
        feature_map = projected.reshape(batch_size * 4, 24, 32, self.config.dim_model).permute(0, 3, 1, 2)
        # ACT's sinusoidal position module intentionally returns a batch-
        # singleton position map.  Build one map for a camera and repeat it
        # for the four camera segments.  The camera identity itself is carried
        # by ``camera_embedding`` above, so spatial positions can be shared.
        position = self.encoder_cam_feat_pos_embed(feature_map[:1]).to(dtype=projected.dtype)
        position = position.permute(0, 2, 3, 1).reshape(1, expected_tokens, self.config.dim_model)
        position = position.repeat(1, 4, 1)
        tokens = projected.reshape(batch_size, 4 * expected_tokens, self.config.dim_model)
        return tokens.transpose(0, 1), position.transpose(0, 1)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor | None, Tensor | None]]:
        if self.config.use_vae and self.training:
            assert ACTION in batch, "actions must be provided when using the variational objective in training mode."
        images = batch.get(OBS_IMAGES)
        if not isinstance(images, list) or len(images) != 4:
            raise ValueError("ACMT-ACTv2 model batch must contain a list of four RGB images")
        batch_size = images[0].shape[0]

        if self.config.use_vae and ACTION in batch and self.training:
            cls_embed = einops.repeat(self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size)
            vae_inputs = [cls_embed]
            if self.config.robot_state_feature:
                state_embed = self.vae_encoder_robot_state_input_proj(batch[OBS_STATE]).unsqueeze(1)
                vae_inputs.append(state_embed)
            vae_inputs.append(self.vae_encoder_action_input_proj(batch[ACTION]))
            vae_encoder_input = torch.cat(vae_inputs, axis=1)
            pos_embed = self.vae_encoder_pos_enc.clone().detach()
            prefix_len = 2 if self.config.robot_state_feature else 1
            cls_joint_is_pad = torch.full(
                (batch_size, prefix_len), False, device=batch[OBS_STATE].device
            )
            key_padding_mask = torch.cat([cls_joint_is_pad, batch["action_is_pad"]], axis=1)
            cls_token_out = self.vae_encoder(
                vae_encoder_input.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[0]
            latent_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_params[:, : self.config.latent_dim]
            log_sigma_x2 = latent_params[:, self.config.latent_dim :]
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            mu = log_sigma_x2 = None
            latent_sample = torch.zeros(
                batch_size,
                self.config.latent_dim,
                dtype=batch[OBS_STATE].dtype,
                device=batch[OBS_STATE].device,
            )

        state = batch[OBS_STATE]
        tactile = batch[TACTILE]
        if tactile.ndim != 5 or tuple(tactile.shape[1:]) != (2, 3, 35, 20):
            raise ValueError(f"ACMT-ACTv2 tactile input must be [B,2,3,35,20], got {tuple(tactile.shape)}")
        tactile_features = self.tactile_encoder(tactile)
        encoder_in_tokens = [
            self.encoder_latent_input_proj(latent_sample),
            self.encoder_robot_state_input_proj(state),
            self.encoder_tactile_input_proj(tactile_features),
        ]
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
        visual_tokens, visual_pos = self._dino_tokens(images)
        encoder_in_tokens.extend(list(visual_tokens))
        encoder_in_pos_embed.extend(list(visual_pos))
        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)

        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)
        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=encoder_in_pos_embed.device,
        )
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )
        actions = self.action_head(decoder_out.transpose(0, 1))
        if tuple(actions.shape[-2:]) != (100, 8):
            raise RuntimeError(f"ACMT-ACTv2 must output [B,100,8], got {tuple(actions.shape)}")
        return actions, (mu, log_sigma_x2)


class ACMTACTV2Policy(PreTrainedPolicy):
    """LeRobot policy wrapper with native one-step temporal ensembling."""

    config_class = ACMTACTV2Config
    name = "acmt_actv2"

    def __init__(self, config: ACMTACTV2Config, dataset_stats: Mapping[str, Any] | None = None, **_: Any):
        super().__init__(config)
        config.validate_features()
        self.config = config
        if dataset_stats:
            means, stds = [], []
            for key in (XENSE0, XENSE1):
                stats = dataset_stats.get(key)
                if isinstance(stats, Mapping) and "mean" in stats and "std" in stats:
                    means.append(torch.as_tensor(stats["mean"], dtype=torch.float32).flatten()[:3])
                    stds.append(torch.as_tensor(stats["std"], dtype=torch.float32).flatten()[:3])
            if means:
                config.force_mean = tuple(torch.stack(means).mean(0).tolist())
                config.force_std = tuple(torch.stack(stds).mean(0).clamp_min(1e-6).tolist())
            action_stats = dataset_stats.get(ACTION)
            if isinstance(action_stats, Mapping) and "mean" in action_stats and "std" in action_stats:
                config.action_mean = tuple(torch.as_tensor(action_stats["mean"]).flatten()[:8].tolist())
                config.action_std = tuple(torch.as_tensor(action_stats["std"]).flatten()[:8].clamp_min(1e-6).tolist())

        self.model = ACMTACTV2Model(config)
        self._generator_runtime: _ACMTGeneratorRuntime | None = None
        if config.tactile_source == "substitution":
            self._generator_runtime = _ACMTGeneratorRuntime(
                config.generator_checkpoint, config.generator_model_config, config.device
            )
            if config.generator_checkpoint_sha256 is None:
                config.generator_checkpoint_sha256 = self._generator_runtime.sha256
            if config.generator_model_config is None:
                config.generator_model_config = dict(self._generator_runtime.model_config)
            if config.generator_checkpoint_sha256.lower() != self._generator_runtime.sha256:
                raise ValueError("ACMT generator checkpoint SHA256 does not match the configured digest")
        self.temporal_ensembler = ACTTemporalEnsembler(
            config.temporal_ensemble_coeff,
            config.chunk_size,
        )
        self.reset()

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, *, config=None, **kwargs: Any):
        config_path = Path(pretrained_name_or_path) / "config.json"
        raw: dict[str, Any] | None = None
        if config_path.is_file():
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            if raw.get("type") != "acmt_actv2":
                raise ValueError("ACMT-ACTv2 loader refuses non-acmt_actv2 checkpoints")
            if (raw.get("checkpoint_schema"), raw.get("checkpoint_schema_version")) != (
                "acmt_actv2.dinov2_spatial.v1",
                3,
            ):
                raise ValueError("checkpoint is not ACMT-ACTv2 DINOv2 spatial schema version 3")
            if raw.get("training_contract") != "native_absolute_physical_gripper_v1":
                raise ValueError("checkpoint is not the native absolute-action ACMT-ACTv2 contract")
            if config is not None:
                if raw.get("checkpoint_task_variant", raw.get("task_variant", "peg")) != config.task_variant:
                    raise ValueError("ACMT-ACTv2 checkpoint task does not match the requested task variant")
        if config is None:
            # Parse through the choice registry so enum fields (normalization
            # modes, feature types, etc.) are restored exactly as they were
            # during training.  Loading the DINO weights from the policy file
            # must never trigger a timm/Hub download.
            parsed = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path,
                cli_overrides=["--dinov2_pretrained=false", "--dinov2_checkpoint=null", "--require_dinov2_checkpoint=false"],
            )
            if not isinstance(parsed, ACMTACTV2Config):
                raise TypeError(f"expected an ACMT-ACTv2 config, got {type(parsed).__name__}")
            config = parsed
        else:
            if not isinstance(config, ACMTACTV2Config):
                raise TypeError("ACMT-ACTv2 loader requires ACMTACTV2Config")
            config.dinov2_pretrained = False
            config.dinov2_checkpoint = None
            config.require_dinov2_checkpoint = False
        return PreTrainedPolicy.from_pretrained.__func__(cls, pretrained_name_or_path, config=config, **kwargs)

    def get_optim_params(self) -> list[dict[str, Any]]:
        projection, other = [], []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad or name.startswith("model.backbone"):
                continue
            if name.startswith(("model.encoder_img_feat_input_proj", "model.camera_embedding")):
                projection.append(parameter)
            else:
                other.append(parameter)
        groups: list[dict[str, Any]] = []
        if other:
            groups.append({"params": other, "lr": self.config.optimizer_lr})
        if projection:
            groups.append({"params": projection, "lr": self.config.optimizer_lr_visual_projection})
        return groups

    def reset(self) -> None:
        if self._generator_runtime is not None:
            self._generator_runtime.reset()
        self.temporal_ensembler.reset()
        self._tactile_history: deque[Tensor] = deque(maxlen=self.config.tactile_history)
        self._gen_history: dict[str, deque[Tensor]] = {
            GEN_RGB: deque(maxlen=self.config.tactile_history),
            GEN_DEPTH: deque(maxlen=self.config.tactile_history),
            GEN_LOWDIM: deque(maxlen=self.config.tactile_history),
            GEN_POSE: deque(maxlen=self.config.tactile_history),
        }
        self._latest_window: dict[str, Tensor] | None = None
        self._generated_tactile: Tensor | None = None
        self._observed_batch_size: int | None = None

    def _current_tactile(self, batch: Mapping[str, Tensor], batch_size: int, device: torch.device) -> Tensor:
        if self.config.tactile_source == "real":
            return torch.stack([_force_side(XENSE0, batch[XENSE0]), _force_side(XENSE1, batch[XENSE1])], dim=1)
        if self.config.tactile_source == "substitution" and self._generated_tactile is not None:
            return self._generated_tactile.to(device=device, dtype=torch.float32)
        return torch.zeros(batch_size, 2, 3, 35, 20, device=device, dtype=torch.float32)

    def _model_batch(self, batch: Mapping[str, Tensor], *, include_target: bool = False) -> dict[str, Tensor]:
        if "rgb" in batch:
            rgb_window = batch["rgb"]
            images = [rgb_window[:, index] for index in range(4)]
            tactile = batch["tactile"][:, -1]
            state = batch["state"]
        else:
            images = [batch[key] for key in self.config.image_features]
            tactile = batch.get(TACTILE)
            state = batch[OBS_STATE]
            if tactile is None:
                if self.config.tactile_source == "real":
                    tactile = torch.stack([_force_side(XENSE0, batch[XENSE0]), _force_side(XENSE1, batch[XENSE1])], dim=1)
                else:
                    tactile = torch.zeros(
                        batch[OBS_STATE].shape[0], 2, 3, 35, 20,
                        dtype=torch.float32,
                        device=batch[OBS_STATE].device,
                    )
        result: dict[str, Tensor] = {OBS_STATE: state, TACTILE: tactile, OBS_IMAGES: images}
        if include_target:
            result[ACTION] = batch[ACTION]
            result["action_is_pad"] = batch["action_is_pad"]
        return result

    @torch.no_grad()
    def observe(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        required = [OBS_STATE, *self.config.image_features]
        if self.config.tactile_source == "real":
            required.extend((XENSE0, XENSE1))
        if self.config.tactile_source == "substitution":
            required.extend((GEN_RGB, GEN_DEPTH, GEN_LOWDIM, GEN_POSE))
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(f"ACMT-ACTv2 observation is missing: {sorted(set(missing))}")
        state = batch[OBS_STATE].float()
        if state.ndim != 2 or tuple(state.shape[1:]) != (8,):
            raise ValueError(f"observation.state must be [B,8], got {tuple(state.shape)}")
        batch_size = state.shape[0]
        if self._observed_batch_size is not None and self._observed_batch_size != batch_size:
            raise ValueError("ACMT-ACTv2 stateful inference requires a fixed batch size")
        self._observed_batch_size = batch_size
        rgb = torch.stack([batch[key].float() for key in self.config.image_features], dim=1)
        tactile = self._current_tactile(batch, batch_size, state.device)
        if not self._tactile_history:
            self._tactile_history.extend(tactile.clone() for _ in range(self.config.tactile_history))
        elif self.config.tactile_source == "real":
            self._tactile_history.append(tactile)
        if self.config.tactile_source == "substitution":
            for key in self._gen_history:
                value = batch[key].float()
                if not self._gen_history[key]:
                    self._gen_history[key].extend(value.clone() for _ in range(self.config.tactile_history))
                else:
                    self._gen_history[key].append(value)
        self._latest_window = {
            "rgb": rgb,
            "state": state,
            "tactile": torch.stack(list(self._tactile_history), dim=1),
        }
        if self.config.tactile_source == "substitution":
            self._latest_window.update({key: torch.stack(list(values), dim=1) for key, values in self._gen_history.items()})
        return self._latest_window

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **_: Any) -> Tensor:
        self.eval()
        window = self.observe(batch)
        actions, _ = self.model(self._model_batch(window))
        if tuple(actions.shape[-2:]) != (100, 8):
            raise RuntimeError(f"ACMT-ACTv2 must return [B,100,8], got {tuple(actions.shape)}")
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **_: Any) -> Tensor:
        self.eval()
        # Native ACT semantics: query on every control tick, combine the new
        # chunk with time-aligned chunks, and emit only the current action.
        return self.temporal_ensembler.update(self.predict_action_chunk(batch))

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        if self.config.tactile_source == "substitution":
            raise ValueError("substitution is evaluation-only; train the real policy checkpoint")
        model_batch = self._model_batch(batch, include_target=True)
        if self.config.tactile_source == "real" and (XENSE0 not in batch or XENSE1 not in batch):
            raise KeyError("real ACMT-ACTv2 training requires both Xense force fields")
        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(model_batch)
        abs_err = F.l1_loss(batch[ACTION], actions_hat, reduction="none")
        valid_mask = ~batch["action_is_pad"].unsqueeze(-1)
        num_valid = valid_mask.sum() * abs_err.shape[-1]
        l1_loss = (abs_err * valid_mask).sum() / num_valid.clamp_min(1)
        loss_dict: dict[str, float] = {"l1_loss": float(l1_loss.detach())}
        if self.config.use_vae and log_sigma_x2_hat is not None:
            mean_kld = (
                (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - log_sigma_x2_hat.exp())).sum(-1).mean()
            )
            loss_dict["kld_loss"] = float(mean_kld.detach())
            loss = l1_loss + mean_kld * self.config.kl_weight
        else:
            loss = l1_loss
        return loss, loss_dict

    @torch.no_grad()
    def notify_action_executed(self, action: Tensor, observation: dict[str, Tensor] | None = None) -> None:
        """Generate the next substitution tactile frame from the accepted action."""

        if self.config.tactile_source != "substitution":
            return
        if self._generator_runtime is None:
            raise RuntimeError("substitution mode has no external ACMT generator")
        # ``observe`` has already assembled a four-frame causal generator
        # window.  Prefer that state over the one-frame preprocessor payload
        # passed by SyncInferenceEngine; otherwise ACMT would receive only two
        # wrist images instead of the required four historical frames.
        previous = self._latest_window if self._latest_window is not None else observation
        if previous is None:
            raise RuntimeError("notify_action_executed called before observe")
        if GEN_RGB in previous:
            generator_observation = {
                "rgb": previous[GEN_RGB],
                "depth": previous[GEN_DEPTH],
                "lowdim": previous[GEN_LOWDIM],
            }
            pose = previous[GEN_POSE]
        else:
            generator_observation = {
                "rgb": torch.stack(list(self._gen_history[GEN_RGB]), dim=1),
                "depth": torch.stack(list(self._gen_history[GEN_DEPTH]), dim=1),
                "lowdim": torch.stack(list(self._gen_history[GEN_LOWDIM]), dim=1),
            }
            pose = torch.stack(list(self._gen_history[GEN_POSE]), dim=1)
        if action.ndim == 1:
            action = action.unsqueeze(0)
        if action.ndim == 3:
            action = action[:, 0]
        if action.ndim != 2 or tuple(action.shape[1:]) != (8,):
            raise ValueError(f"executed ACMT-ACTv2 action must be [B,8], got {tuple(action.shape)}")
        generator_action = action.clone()
        # ACMTv4 uses its historical wire direction; ACMT-ACT exposes physical
        # 0=open/1=closed, so flip only this private generator copy.
        generator_action[:, 7] = 1.0 - generator_action[:, 7]
        generated = self._generator_runtime.predict_next(generator_observation, pose, generator_action)
        self._generated_tactile = generated.permute(0, 1, 4, 2, 3).contiguous().float()
        if self._tactile_history:
            self._tactile_history.append(self._generated_tactile)
        if self._latest_window is not None and self._tactile_history:
            self._latest_window["tactile"] = torch.stack(list(self._tactile_history), dim=1)

    def action_execution_slice(self, action_chunk: Tensor) -> Tensor:
        if action_chunk.ndim < 2 or tuple(action_chunk.shape[-2:]) != (100, 8):
            raise ValueError("ACMT-ACTv2 action chunk must end in [100,8]")
        return action_chunk[..., :1, :]


__all__ = ["ACMTACTV2Model", "ACMTACTV2Policy", "DINOv2SpatialBackbone"]
