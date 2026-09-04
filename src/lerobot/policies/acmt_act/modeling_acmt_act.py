"""ACMT-ACT policy implementation.

This module intentionally keeps the ACT image/token path close to LeRobot's
reference implementation.  The only model-side addition is one token produced
by a shared two-wrist force-field encoder.  The ACMT generator used by the
``substitution`` comparison is a runtime helper, not an ``nn.Module`` member,
so it can never accidentally become part of the trainable policy checkpoint.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import einops
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.policies.act.modeling_act import (
    ACT,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from .configuration_acmt_act import (
    XENSE0,
    XENSE1,
    ACMTACTConfig,
)
from .processor_acmt_act import GEN_DEPTH, GEN_LOWDIM, GEN_POSE, GEN_RGB

TACTILE = "_acmt_act.tactile"
_GENERATOR_MODEL_CONFIG_KEYS = {
    "force_mean",
    "force_std",
    "t_obs",
    "t_pred",
    "visual_dim",
    "shared_dim",
    "decoder_dim",
    "num_heads",
    "conv_layers",
    "decoder_layout",
    "dformerv2_repo_path",
    "dformerv2_checkpoint",
    "visual_encoder_name",
    "view_dropout",
    "field_order",
    "xy_magnitude_scale",
    "xy_direction_min_magnitude",
    "contact_pos_weight",
}


def _force_side(key: str, value: Tensor) -> Tensor:
    """Convert one Xense field to ``[B,3,35,20]``."""

    if value.ndim == 3:
        if tuple(value.shape) == (3, 35, 20):
            value = value.unsqueeze(0)
        elif tuple(value.shape) == (35, 20, 3):
            value = value.permute(2, 0, 1).unsqueeze(0)
    elif value.ndim == 4:
        if tuple(value.shape[1:]) == (3, 35, 20):
            pass
        elif tuple(value.shape[1:]) == (35, 20, 3):
            value = value.permute(0, 3, 1, 2)
    if value.ndim != 4 or tuple(value.shape[1:]) != (3, 35, 20):
        raise ValueError(f"{key} must be [B,3,35,20] or [B,35,20,3], got {tuple(value.shape)}")
    return value.float().contiguous()


class ACMTACTileEncoder(nn.Module):
    """Small shared spatial encoder for two simultaneous force fields.

    The two wrists share convolution weights.  Their pooled 80-dimensional
    representations are concatenated, yielding the fixed 160-dimensional
    feature described by the policy ABI.
    """

    def __init__(self, force_mean: tuple[float, float, float], force_std: tuple[float, float, float]):
        super().__init__()
        self.register_buffer(
            "force_mean",
            torch.as_tensor(force_mean, dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=True,
        )
        self.register_buffer(
            "force_std",
            torch.as_tensor(force_std, dtype=torch.float32).clamp_min(1e-6).view(1, 1, 3, 1, 1),
            persistent=True,
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 80, kernel_size=3, padding=1),
            nn.GroupNorm(8, 80),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(160)

    def forward(self, tactile: Tensor) -> Tensor:
        if tactile.ndim != 5 or tuple(tactile.shape[1:]) != (2, 3, 35, 20):
            raise ValueError(
                "ACMT-ACT tactile input must be [B,2,3,35,20], "
                f"got {tuple(tactile.shape)}"
            )
        values = (tactile.float() - self.force_mean.to(tactile)) / self.force_std.to(tactile)
        batch_size = values.shape[0]
        encoded = self.spatial(values.reshape(batch_size * 2, 3, 35, 20))
        encoded = F.adaptive_avg_pool2d(encoded, output_size=1).flatten(1).reshape(batch_size, 2, 80)
        return self.norm(encoded.reshape(batch_size, 160))


class ACMTACT(ACT):
    """Reference ACT network plus a single tactile conditioning token."""

    def __init__(self, config: ACMTACTConfig):
        # Resolve the serialized torchvision enum before ACT.__init__ builds
        # camera 0. The remaining three cameras below receive this same enum,
        # so all four streams use exactly the same ImageNet initialization.
        pretrained = config.pretrained_backbone_weights
        if isinstance(pretrained, str):
            enum_name = config.vision_backbone.replace("resnet", "ResNet") + "_Weights"
            enum_cls = getattr(torchvision.models, enum_name, None)
            if enum_cls is None:
                raise ValueError(f"Unknown torchvision weights enum for {config.vision_backbone}")
            pretrained = getattr(enum_cls, pretrained.rsplit(".", 1)[-1])
            config.pretrained_backbone_weights = pretrained
        super().__init__(config)

        # ACT's base class creates one shared ResNet.  v3 deliberately
        # replaces that module with four separately-owned ResNet50 instances
        # and four separately-owned 1x1 projections.  They are initialized
        # from the same ImageNet checkpoint, but no parameter object is shared
        # between camera streams.
        if config.camera_backbone_mode != "independent":
            raise ValueError("ACMT-ACT v3 requires independent camera backbones")
        def make_backbone() -> IntermediateLayerGetter:
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                weights=pretrained,
                norm_layer=FrozenBatchNorm2d,
            )
            return IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

        # Remove the shared projection registered by ACT.__init__.  Reuse the
        # already-created first ResNet as camera 0 (rather than allocating a
        # fifth temporary network), then construct three independent peers.
        first_backbone = self.backbone
        del self.backbone
        del self.encoder_img_feat_input_proj
        self.backbone = nn.ModuleList([first_backbone, *(make_backbone() for _ in config.camera_keys[1:])])
        last_block = first_backbone.layer4[-1]
        # ResNet18/34 use BasicBlock (conv2), while larger variants use
        # Bottleneck (conv3).  v3 is ResNet50, but deriving this from the
        # actual first backbone keeps the projection robust to config loading.
        if hasattr(last_block, "conv3"):
            backbone_channels = last_block.conv3.out_channels
        else:
            backbone_channels = last_block.conv2.out_channels
        self.encoder_img_feat_input_proj = nn.ModuleList(
            [nn.Conv2d(backbone_channels, config.dim_model, kernel_size=1) for _ in config.camera_keys]
        )
        self.tactile_encoder = ACMTACTileEncoder(config.force_mean, config.force_std)
        self.encoder_tactile_input_proj = nn.Linear(config.tactile_feature_dim, config.dim_model)

        # ACT has latent (+ state) 1-D positions.  Extend this same position
        # table with the tactile token, preserving the learned ACT positions.
        old_position = self.encoder_1d_feature_pos_embed
        self.encoder_1d_feature_pos_embed = nn.Embedding(3, config.dim_model)
        with torch.no_grad():
            self.encoder_1d_feature_pos_embed.weight[: old_position.num_embeddings].copy_(
                old_position.weight
            )
            nn.init.normal_(
                self.encoder_1d_feature_pos_embed.weight[old_position.num_embeddings :], std=0.02
            )

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor | None, Tensor | None]]:
        if self.config.use_vae and self.training:
            assert ACTION in batch, "actions must be provided when using the variational objective in training mode."

        images = batch.get(OBS_IMAGES)
        batch_size = images[0].shape[0] if images else batch[OBS_STATE].shape[0]

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
                [batch_size, self.config.latent_dim],
                dtype=batch[OBS_STATE].dtype,
                device=batch[OBS_STATE].device,
            )

        tactile = batch[TACTILE]
        tactile_features = self.tactile_encoder(tactile)
        encoder_in_tokens = [
            self.encoder_latent_input_proj(latent_sample),
            self.encoder_robot_state_input_proj(batch[OBS_STATE]),
            self.encoder_tactile_input_proj(tactile_features),
        ]
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))

        if images:
            if len(images) != len(self.backbone):
                raise ValueError(f"ACMT-ACT expects {len(self.backbone)} camera images, got {len(images)}")
            for image, backbone, image_proj in zip(
                images, self.backbone, self.encoder_img_feat_input_proj, strict=True
            ):
                cam_features = backbone(image)["feature_map"]
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = image_proj(cam_features)
                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")
                encoder_in_tokens.extend(list(cam_features))
                encoder_in_pos_embed.extend(list(cam_pos_embed))

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
        return actions, (mu, log_sigma_x2)


class _ACMTGeneratorRuntime:
    """Plain-Python holder for the frozen ACMT model.

    It deliberately does not subclass ``nn.Module``.  As a result assigning an
    instance to a policy cannot register generator parameters in the policy's
    state dict or optimizer.
    """

    def __init__(self, checkpoint: str | Path, model_config: Mapping[str, Any] | None, device: str):
        from lerobot.policies.acmt_dp.modeling_tactile_generator import TactiGenForceFieldModel

        self.checkpoint = str(checkpoint)
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"ACMT generator checkpoint not found: {checkpoint_path}")
        with checkpoint_path.open("rb") as stream:
            self.sha256 = hashlib.sha256(stream.read()).hexdigest()
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise ValueError("ACMT generator checkpoint must contain a mapping")
        raw_config = payload.get("model_config")
        if model_config is not None:
            raw_config = dict(model_config)
        if not isinstance(raw_config, Mapping):
            raise ValueError("ACMT generator checkpoint has no model_config")
        self.model_config = dict(raw_config)
        filtered_config = {key: value for key, value in raw_config.items() if key in _GENERATOR_MODEL_CONFIG_KEYS}
        self.model = TactiGenForceFieldModel(**filtered_config)
        state_dict = payload.get("model_state_dict", payload.get("state_dict"))
        if not isinstance(state_dict, Mapping):
            raise ValueError("ACMT generator checkpoint has no model_state_dict")
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def reset(self) -> None:
        reset = getattr(self.model, "reset", None)
        if callable(reset):
            reset()

    @torch.no_grad()
    def predict_next(self, observation: dict[str, Tensor], pose: Tensor, action: Tensor) -> Tensor:
        output = self.model.predict_next(observation, pose, action)
        if output.ndim != 5 or tuple(output.shape[1:]) != (2, 35, 20, 3):
            raise ValueError(
                "ACMT generator must return [B,2,35,20,3], "
                f"got {tuple(output.shape)}"
            )
        return output


class ACMTACTPolicy(PreTrainedPolicy):
    """LeRobot policy exposing ACT's train/inference contract."""

    config_class = ACMTACTConfig
    name = "acmt_act"

    def __init__(self, config: ACMTACTConfig, dataset_stats: Mapping[str, Any] | None = None, **_: Any):
        super().__init__(config)
        config.validate_features()
        self.config = config
        # Use the training-set force statistics for both physical sensors when
        # available.  This keeps none/real initialization identical while
        # avoiding a hidden dependence on a particular task's raw scale.
        if dataset_stats:
            means, stds = [], []
            for key in (XENSE0, XENSE1):
                stats = dataset_stats.get(key)
                if isinstance(stats, Mapping) and "mean" in stats and "std" in stats:
                    means.append(torch.as_tensor(stats["mean"], dtype=torch.float32).flatten())
                    stds.append(torch.as_tensor(stats["std"], dtype=torch.float32).flatten())
            if means and all(value.numel() >= 3 for value in means):
                config.force_mean = tuple(torch.stack([value[:3] for value in means]).mean(0).tolist())
                config.force_std = tuple(
                    torch.stack([value[:3] for value in stds]).mean(0).clamp_min(1e-6).tolist()
                )
        self.model = ACMTACT(config)
        # Runtime-only state.  `_ACMTGeneratorRuntime` is not an nn.Module, so
        # no generator weights appear in `named_parameters()` or checkpoints.
        self._generator_runtime: _ACMTGeneratorRuntime | None = None
        if config.tactile_source == "substitution":
            self._generator_runtime = _ACMTGeneratorRuntime(
                config.generator_checkpoint, config.generator_model_config, config.device
            )
            if config.generator_checkpoint_sha256 is None:
                # Persist the exact generator identity in config.json while
                # keeping its parameters outside the policy state dict.
                config.generator_checkpoint_sha256 = self._generator_runtime.sha256
            if config.generator_model_config is None:
                config.generator_model_config = dict(self._generator_runtime.model_config)
            if config.generator_checkpoint_sha256 is not None and (
                config.generator_checkpoint_sha256.lower() != self._generator_runtime.sha256
            ):
                raise ValueError("ACMT generator checkpoint SHA256 does not match the configured digest")
        self.reset()

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, *, config=None, **kwargs):
        config_path = Path(pretrained_name_or_path) / "config.json"
        if config_path.is_file():
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            if raw.get("type") != "acmt_act":
                raise ValueError("ACMT-ACT loader refuses non-acmt_act checkpoints")
            if raw.get("checkpoint_schema") != "acmt_act.v3" or raw.get("checkpoint_schema_version") != 3:
                raise ValueError("checkpoint is not ACMT-ACT schema acmt_act.v3")
            source = (config.tactile_source if config is not None else raw.get("tactile_source", "none"))
            checkpoint_source = raw.get("checkpoint_tactile_source", source)
            checkpoint_task = raw.get("checkpoint_task_variant", raw.get("task_variant", "peg"))
            expected_source = "real" if source == "substitution" else source
            if checkpoint_source != expected_source:
                raise ValueError(
                    "ACMT-ACT checkpoint tactile source does not match the requested mode: "
                    f"checkpoint={checkpoint_source!r}, requested={source!r}"
                )
            if config is not None and checkpoint_task != config.task_variant:
                raise ValueError("ACMT-ACT checkpoint task does not match the requested task variant")
        return super().from_pretrained(pretrained_name_or_path, config=config, **kwargs)

    def get_optim_params(self) -> list[dict[str, Any]]:
        return [
            {
                "params": [
                    parameter
                    for name, parameter in self.named_parameters()
                    if not name.startswith("model.backbone") and parameter.requires_grad
                ]
            },
            {
                "params": [
                    parameter
                    for name, parameter in self.named_parameters()
                    if name.startswith("model.backbone") and parameter.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]

    def reset(self) -> None:
        if self._generator_runtime is not None:
            self._generator_runtime.reset()
        self._action_queue = deque([], maxlen=self.config.n_action_steps)
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

    def _model_batch(self, window: Mapping[str, Tensor], *, include_target: bool = False) -> dict[str, Tensor]:
        model_batch: dict[str, Tensor] = {OBS_STATE: window["state"], TACTILE: window["tactile"][:, -1]}
        # The v3 policy supplies four streams and acmt_actv2 supplies the
        # side plus two wrist streams.  Keep the batch adapter generic while
        # preserving the serialized v3 module layout.
        model_batch[OBS_IMAGES] = [
            window["rgb"][:, index] for index in range(len(self.config.image_features))
        ]
        if include_target:
            model_batch[ACTION] = window[ACTION]
            if "action_is_pad" in window:
                model_batch["action_is_pad"] = window["action_is_pad"]
        return model_batch

    @torch.no_grad()
    def observe(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Record one current observation and maintain the ACMT causal ring."""

        required = [OBS_STATE, *self.config.image_features]
        missing = [key for key in required if key not in batch]
        if self.config.tactile_source == "real":
            missing.extend(key for key in (XENSE0, XENSE1) if key not in batch)
        if self.config.tactile_source == "substitution":
            missing.extend(key for key in (GEN_RGB, GEN_DEPTH, GEN_LOWDIM, GEN_POSE) if key not in batch)
        if missing:
            raise KeyError(f"ACMT-ACT observation is missing: {sorted(set(missing))}")
        state = batch[OBS_STATE].float()
        if state.ndim != 2 or tuple(state.shape[1:]) != (8,):
            raise ValueError(f"observation.state must be [B,8], got {tuple(state.shape)}")
        batch_size = state.shape[0]
        if self._observed_batch_size is not None and self._observed_batch_size != batch_size:
            raise ValueError("ACMT-ACT stateful inference requires a fixed batch size")
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

        window: dict[str, Tensor] = {
            "rgb": rgb,
            "state": state,
            "tactile": torch.stack(list(self._tactile_history), dim=1),
        }
        if self.config.tactile_source == "substitution":
            window.update({key: torch.stack(list(values), dim=1) for key, values in self._gen_history.items()})
        self._latest_window = window
        return window

    @torch.no_grad()
    def _plan(self, window: Mapping[str, Tensor]) -> Tensor:
        if self._latest_window is None:
            raise RuntimeError("ACMT-ACT cannot plan before observe")
        actions, _ = self.model(self._model_batch(window))
        if tuple(actions.shape[-2:]) != (self.config.chunk_size, self.config.action_dim):
            raise RuntimeError(f"ACMT-ACT must return [B,16,8], got {tuple(actions.shape)}")
        return actions

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **_: Any) -> Tensor:
        self.eval()
        window = self.observe(batch)
        return self._plan(window)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **_: Any) -> Tensor:
        self.eval()
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        if self.config.tactile_source == "substitution":
            raise ValueError("substitution is evaluation-only; train the real policy checkpoint")
        model_batch = dict(batch)
        if self.config.image_features:
            model_batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        if self.config.tactile_source == "real":
            if XENSE0 not in batch or XENSE1 not in batch:
                raise KeyError("real ACMT-ACT training requires both Xense force fields")
            model_batch[TACTILE] = torch.stack(
                [_force_side(XENSE0, batch[XENSE0]), _force_side(XENSE1, batch[XENSE1])], dim=1
            )
        else:
            reference = next(iter(self.config.image_features), None)
            if reference is not None and reference in batch:
                batch_size = batch[reference].shape[0]
                device = batch[reference].device
            else:
                batch_size = batch[OBS_STATE].shape[0]
                device = batch[OBS_STATE].device
            model_batch[TACTILE] = torch.zeros(
                batch_size, 2, 3, 35, 20, dtype=torch.float32, device=device
            )
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
        """Generate the next tactile frame from the actual accepted command."""

        if self.config.tactile_source != "substitution":
            return
        if self._generator_runtime is None:
            raise RuntimeError("substitution mode has no external ACMT generator")
        previous = observation if isinstance(observation, Mapping) else self._latest_window
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
            raise ValueError(f"executed ACMT-ACT action must be [B,8], got {tuple(action.shape)}")
        generated = self._generator_runtime.predict_next(generator_observation, pose, action)
        self._generated_tactile = generated.permute(0, 1, 4, 2, 3).contiguous().float()
        self._tactile_history.append(self._generated_tactile)
        if self._latest_window is not None:
            self._latest_window["tactile"] = torch.stack(list(self._tactile_history), dim=1)

    def action_execution_slice(self, action_chunk: Tensor) -> Tensor:
        if action_chunk.ndim < 2 or tuple(action_chunk.shape[-2:]) != (16, 8):
            raise ValueError("ACMT-ACT action chunk must end in [16,8]")
        return action_chunk[..., :8, :]


__all__ = ["ACMTACT", "ACMTACTileEncoder", "ACMTACTPolicy", "TACTILE"]
