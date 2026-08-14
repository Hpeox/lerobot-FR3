"""Inference-compatible TactiGen force-field generator for tactigen mode."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from .modeling_components import (
    DFormerv2DualRGBDEncoder,
    HistoryEncoder,
    ModalityDropout,
    QueryFiLM,
    SpatialConvBlock,
    TensorBatch,
    TinyDualRGBDEncoder,
    coordinate_grid,
    require_tensor,
)
from .visual_preprocess import prepare_for_frozen_encoder

CONTACT_FZ_THRESHOLD = -0.01
DEFAULT_TEMPERATURES: tuple[float, ...] = ()
DEFAULT_POOL_SCALES: tuple[tuple[int, int], ...] = ()


def matrix_to_pose_xyzw_fast(matrices: torch.Tensor) -> torch.Tensor:
    """Vectorized transform conversion used by the causal TactiGen path."""
    if matrices.ndim < 3 or tuple(matrices.shape[-2:]) != (4, 4):
        raise ValueError(f"expected [...,4,4] transforms, got {tuple(matrices.shape)}")
    flat = matrices.reshape(-1, 4, 4)
    rotation = flat[:, :3, :3]
    diagonal = torch.diagonal(rotation, dim1=-2, dim2=-1)
    trace = diagonal.sum(-1)
    eps = torch.finfo(flat.dtype).eps if flat.dtype.is_floating_point else 1e-7
    positive = trace > 0
    s = (torch.sqrt((trace + 1.0).clamp_min(eps)) * 2.0).clamp_min(eps)
    qp = torch.stack(
        [
            (rotation[:, 2, 1] - rotation[:, 1, 2]) / s,
            (rotation[:, 0, 2] - rotation[:, 2, 0]) / s,
            (rotation[:, 1, 0] - rotation[:, 0, 1]) / s,
            0.25 * s,
        ],
        dim=-1,
    )
    s0 = (torch.sqrt((1 + diagonal[:, 0] - diagonal[:, 1] - diagonal[:, 2]).clamp_min(eps)) * 2).clamp_min(
        eps
    )
    q0 = torch.stack(
        [
            0.25 * s0,
            (rotation[:, 0, 1] + rotation[:, 1, 0]) / s0,
            (rotation[:, 0, 2] + rotation[:, 2, 0]) / s0,
            (rotation[:, 2, 1] - rotation[:, 1, 2]) / s0,
        ],
        dim=-1,
    )
    s1 = (torch.sqrt((1 + diagonal[:, 1] - diagonal[:, 0] - diagonal[:, 2]).clamp_min(eps)) * 2).clamp_min(
        eps
    )
    q1 = torch.stack(
        [
            (rotation[:, 0, 1] + rotation[:, 1, 0]) / s1,
            0.25 * s1,
            (rotation[:, 1, 2] + rotation[:, 2, 1]) / s1,
            (rotation[:, 0, 2] - rotation[:, 2, 0]) / s1,
        ],
        dim=-1,
    )
    s2 = (torch.sqrt((1 + diagonal[:, 2] - diagonal[:, 0] - diagonal[:, 1]).clamp_min(eps)) * 2).clamp_min(
        eps
    )
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
    quat = torch.where(positive[:, None], qp, fallback)
    quat = quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp_min(eps)
    return torch.cat([flat[:, :3, 3], quat], dim=-1).reshape(*matrices.shape[:-2], 7)


def contact_from_fz(force_z: torch.Tensor) -> torch.Tensor:
    return force_z < CONTACT_FZ_THRESHOLD


def physical_force(outputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([outputs["force_xy"], outputs["force_z"]], dim=-1)


def multiscale_fz_field_drifting_loss(*_: Any, **__: Any):
    raise NotImplementedError("The embedded ACMT generator is inference-only")


DRIFTING_DECODER_DIM = 192
DRIFTING_FIELD_ORDER = ("fz", "fx", "fy")
DRIFTING_LAYOUT = "drifting_fz_xy_heads_v2"
LEGACY_DRIFTING_LAYOUT = "drifting_xyz_192"
FZ_REGRESSION_WEIGHT = 1.0
FZ_SMOOTH_WEIGHT = 0.021222216792780163
FZ_CONTACT_WEIGHT = 0.00516480451221669
VISUAL_CONTACT_WEIGHT = 0.0037580437265385984
LEGACY_CONTACT_POS_WEIGHT = 2.159427071322711
XY_CONTACT_WEIGHT = 0.9
XY_NONCONTACT_WEIGHT = 0.1

# Conservative fallbacks for unit tests and direct library use. Formal
# training replaces them with deterministic 32-batch gradient calibration.
DRIFTING_WEIGHT = 0.005
XY_MAGNITUDE_WEIGHT = 0.05
XY_DIRECTION_WEIGHT = 0.05


def set_calibrated_loss_weights(
    *,
    drifting: float,
    magnitude: float,
    direction: float,
) -> None:
    """Install fixed run-level weights after deterministic calibration."""
    global DRIFTING_WEIGHT, XY_MAGNITUDE_WEIGHT, XY_DIRECTION_WEIGHT
    DRIFTING_WEIGHT = float(drifting)
    XY_MAGNITUDE_WEIGHT = float(magnitude)
    XY_DIRECTION_WEIGHT = float(direction)


def validate_contact_pos_weight(value: float) -> float:
    """Return a finite positive BCE class weight."""
    weight = float(value)
    if not math.isfinite(weight) or weight <= 0.0:
        raise ValueError("contact_pos_weight must be finite and positive")
    return weight


def checkpoint_contact_pos_weight(
    checkpoint: Mapping[str, Any],
) -> float:
    """Resolve new and historical checkpoint class weights in priority order."""
    for section_name in ("model_config", "force_stats", "loss_weights"):
        section = checkpoint.get(section_name)
        if isinstance(section, Mapping) and section.get("contact_pos_weight") is not None:
            return validate_contact_pos_weight(section["contact_pos_weight"])
    return LEGACY_CONTACT_POS_WEIGHT


def loss_weight_config(
    contact_pos_weight: float = LEGACY_CONTACT_POS_WEIGHT,
) -> dict[str, Any]:
    return {
        "fz_regression": FZ_REGRESSION_WEIGHT,
        "fz_smooth": FZ_SMOOTH_WEIGHT,
        "fz_contact": FZ_CONTACT_WEIGHT,
        "visual_contact": VISUAL_CONTACT_WEIGHT,
        "contact_pos_weight": validate_contact_pos_weight(contact_pos_weight),
        "contact_fz_threshold": CONTACT_FZ_THRESHOLD,
        "xy_contact": XY_CONTACT_WEIGHT,
        "xy_noncontact": XY_NONCONTACT_WEIGHT,
        "xy_magnitude": XY_MAGNITUDE_WEIGHT,
        "xy_direction": XY_DIRECTION_WEIGHT,
        "drifting": DRIFTING_WEIGHT,
        "drifting_axis": "fz",
        "drifting_temperatures": list(DEFAULT_TEMPERATURES),
        "drifting_pool_scales": [list(scale) for scale in DEFAULT_POOL_SCALES],
        "drifting_reference": "lambertae/drifting",
        "calibration_probes": 32,
        "auxiliary_gradient_ratio_target": 0.1,
    }


class RegularGridForceFieldDecoder(nn.Module):
    """Regular-grid 192-d CNN with separate Fz and joint-XY heads."""

    def __init__(
        self,
        shared_dim: int = 160,
        visual_dim: int = 160,
        decoder_dim: int = DRIFTING_DECODER_DIM,
        num_heads: int = 4,
        conv_layers: int = 3,
    ) -> None:
        super().__init__()
        if decoder_dim != DRIFTING_DECODER_DIM:
            raise ValueError(f"Regular-grid decoder dimension is fixed at {DRIFTING_DECODER_DIM}")
        self.decoder_dim = int(decoder_dim)
        self.grid_h, self.grid_w = 35, 20
        self.locations_per_side = self.grid_h * self.grid_w
        self.action_query = nn.Linear(shared_dim, decoder_dim)
        self.coord_embed = nn.Sequential(
            nn.Linear(2, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, decoder_dim),
        )
        self.side_embed = nn.Parameter(torch.zeros(2, decoder_dim))
        nn.init.trunc_normal_(self.side_embed, std=0.02)
        self.visual_key = nn.Linear(visual_dim, decoder_dim)
        self.visual_value = nn.Linear(visual_dim, decoder_dim)
        self.visual_attn = nn.MultiheadAttention(
            decoder_dim,
            num_heads,
            dropout=0.10,
            batch_first=True,
        )
        self.visual_norm = nn.LayerNorm(decoder_dim)
        self.film = nn.Sequential(
            nn.Linear(shared_dim, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, decoder_dim * 2),
        )
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)
        self.condition_norm = nn.LayerNorm(decoder_dim)
        self.layers = nn.ModuleList(SpatialConvBlock(decoder_dim) for _ in range(conv_layers))
        self.fz_head = nn.Sequential(
            nn.LayerNorm(decoder_dim),
            nn.Linear(decoder_dim, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, 1),
        )
        self.xy_head = nn.Sequential(
            nn.LayerNorm(decoder_dim),
            nn.Linear(decoder_dim, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, 2),
        )
        self.visual_contact_head = nn.Sequential(
            nn.LayerNorm(decoder_dim),
            nn.Linear(decoder_dim, 1),
        )
        self.contact_head = nn.Sequential(
            nn.LayerNorm(decoder_dim),
            nn.Linear(decoder_dim, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, 1),
        )

    def _position(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        coordinates = coordinate_grid(
            self.grid_h,
            self.grid_w,
            device,
            dtype,
        )
        embedded = self.coord_embed(coordinates)
        return (embedded[None] + self.side_embed[:, None, None].to(embedded)).reshape(
            2 * self.locations_per_side, self.decoder_dim
        )

    def forward(
        self,
        action_query: torch.Tensor,
        visual_memory: torch.Tensor,
        load: torch.Tensor,
        visual_padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        batch, prediction_steps, _ = action_query.shape
        position = self._position(action_query.device, action_query.dtype)
        seed = self.action_query(action_query).unsqueeze(2) + position[None, None]
        query = seed.reshape(
            batch * prediction_steps,
            -1,
            self.decoder_dim,
        )
        key = self.visual_key(visual_memory)
        value = self.visual_value(visual_memory)
        key = (
            key[:, None]
            .expand(
                -1,
                prediction_steps,
                -1,
                -1,
            )
            .reshape(batch * prediction_steps, 128, self.decoder_dim)
        )
        value = (
            value[:, None]
            .expand(
                -1,
                prediction_steps,
                -1,
                -1,
            )
            .reshape(batch * prediction_steps, 128, self.decoder_dim)
        )
        padding = None
        if visual_padding_mask is not None:
            padding = (
                visual_padding_mask[:, None]
                .expand(
                    -1,
                    prediction_steps,
                    -1,
                )
                .reshape(batch * prediction_steps, 128)
            )
        retrieved, _ = self.visual_attn(
            query,
            key,
            value,
            key_padding_mask=padding,
            need_weights=False,
        )
        visual_features = self.visual_norm(retrieved + position[None])
        gamma, beta = self.film(load).chunk(2, dim=-1)
        conditioned = self.condition_norm(
            (
                1.0
                + gamma.reshape(
                    batch * prediction_steps,
                    1,
                    -1,
                )
            )
            * visual_features
            + beta.reshape(batch * prediction_steps, 1, -1)
        )
        spatial = conditioned.reshape(
            batch * prediction_steps,
            2,
            self.grid_h,
            self.grid_w,
            self.decoder_dim,
        )
        for layer in self.layers:
            spatial = layer(spatial)

        normalized_fz = self.fz_head(spatial).reshape(
            batch,
            prediction_steps,
            2,
            self.grid_h,
            self.grid_w,
            1,
        )
        normalized_xy = self.xy_head(spatial).reshape(
            batch,
            prediction_steps,
            2,
            self.grid_h,
            self.grid_w,
            2,
        )
        visual_features = visual_features.reshape(
            batch,
            prediction_steps,
            2,
            self.grid_h,
            self.grid_w,
            self.decoder_dim,
        )
        spatial_features = spatial.reshape(
            batch,
            prediction_steps,
            2,
            self.grid_h,
            self.grid_w,
            self.decoder_dim,
        )
        visual_contact_logits = self.visual_contact_head(visual_features).squeeze(-1)
        spatial_contact_logits = self.contact_head(spatial_features).squeeze(-1)
        return (
            normalized_fz,
            normalized_xy,
            {
                "visual_features": visual_features,
                "conditioned_features": conditioned.reshape(
                    batch,
                    prediction_steps,
                    2,
                    self.grid_h,
                    self.grid_w,
                    self.decoder_dim,
                ),
                "spatial_features": spatial_features,
                "visual_contact_logits": visual_contact_logits,
                "contact_logits": spatial_contact_logits,
            },
        )


class TactiGenForceFieldModel(nn.Module):
    """Single-pass force-field model with a training-time drifting target."""

    def __init__(
        self,
        force_mean: tuple[float, float, float] = (0.0, 0.0, 0.0),
        force_std: tuple[float, float, float] = (1.0, 1.0, 1.0),
        t_obs: int = 4,
        t_pred: int = 1,
        visual_dim: int = 160,
        shared_dim: int = 160,
        decoder_dim: int = DRIFTING_DECODER_DIM,
        num_heads: int = 4,
        conv_layers: int = 3,
        decoder_layout: str = DRIFTING_LAYOUT,
        dformerv2_repo_path: str | None = None,
        dformerv2_checkpoint: str | None = None,
        visual_encoder_name: str = "dformerv2",
        view_dropout: float = 0.10,
        field_order: tuple[str, str, str] | list[str] = DRIFTING_FIELD_ORDER,
        xy_magnitude_scale: float = 1.0,
        xy_direction_min_magnitude: float = 0.0,
        contact_pos_weight: float = LEGACY_CONTACT_POS_WEIGHT,
    ) -> None:
        super().__init__()
        if t_obs != 4 or t_pred != 1:
            raise ValueError("TactiGen requires t_obs=4 and t_pred=1")
        if decoder_layout != DRIFTING_LAYOUT:
            raise ValueError(f"Unsupported drifting decoder layout: {decoder_layout}")
        if tuple(field_order) != DRIFTING_FIELD_ORDER:
            raise ValueError(f"Drifting field order must be {DRIFTING_FIELD_ORDER}")
        self.t_obs, self.t_pred = t_obs, t_pred
        self.decoder_layout = decoder_layout
        self.visual_encoder_name = visual_encoder_name
        self.contact_pos_weight = validate_contact_pos_weight(contact_pos_weight)
        encoder_class = (
            DFormerv2DualRGBDEncoder if visual_encoder_name == "dformerv2" else TinyDualRGBDEncoder
        )
        self.visual_encoder = encoder_class(
            visual_dim=visual_dim,
            t_obs=t_obs,
            num_heads=num_heads,
            repo_path=dformerv2_repo_path,
            checkpoint=dformerv2_checkpoint,
            view_dropout=view_dropout,
        )
        self.state_encoder = HistoryEncoder(14, shared_dim, 0.10)
        self.dynamics_encoder = nn.Sequential(
            nn.Linear(7, shared_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(shared_dim, shared_dim),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(8, shared_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(shared_dim, shared_dim),
        )
        self.state_dropout = ModalityDropout(0.05)
        self.dynamics_dropout = ModalityDropout(0.15)
        self.action_dropout = ModalityDropout(0.05)
        self.query_film = QueryFiLM(shared_dim)
        self.load_attn = nn.MultiheadAttention(
            shared_dim,
            num_heads,
            dropout=0.10,
            batch_first=True,
        )
        self.load_norm = nn.LayerNorm(shared_dim)
        self.drifting_decoder = RegularGridForceFieldDecoder(
            shared_dim,
            visual_dim,
            decoder_dim,
            num_heads,
            conv_layers,
        )
        self.register_buffer(
            "force_mean",
            torch.as_tensor(force_mean, dtype=torch.float32).view(
                1,
                1,
                1,
                1,
                1,
                3,
            ),
            persistent=True,
        )
        self.register_buffer(
            "force_std",
            torch.as_tensor(force_std, dtype=torch.float32).clamp_min(1e-6).view(1, 1, 1, 1, 1, 3),
            persistent=True,
        )
        self.register_buffer(
            "xy_magnitude_scale",
            torch.tensor(max(float(xy_magnitude_scale), 1e-6)),
            persistent=True,
        )
        self.register_buffer(
            "xy_direction_min_magnitude",
            torch.tensor(max(float(xy_direction_min_magnitude), 0.0)),
            persistent=True,
        )

    def reset(self) -> None:
        """Reset deployment causal state.

        The embedded generator is currently functionally stateless—the policy
        owns the four-frame tactile ring—but an explicit hook prevents future
        cached CUDA/RNN state from leaking across episodes.
        """
        return None

    def model_config(self) -> dict[str, Any]:
        return {
            "t_obs": self.t_obs,
            "t_pred": self.t_pred,
            "visual_dim": self.drifting_decoder.visual_key.in_features,
            "shared_dim": self.state_encoder.gru.hidden_size,
            "decoder_dim": self.drifting_decoder.decoder_dim,
            "num_heads": self.load_attn.num_heads,
            "conv_layers": len(self.drifting_decoder.layers),
            "decoder_layout": self.decoder_layout,
            "field_order": list(DRIFTING_FIELD_ORDER),
            "visual_encoder_name": self.visual_encoder_name,
            "view_dropout": float(getattr(self.visual_encoder, "view_dropout", 0.10)),
            "force_mean": self.force_mean.flatten().tolist(),
            "force_std": self.force_std.flatten().tolist(),
            "xy_magnitude_scale": float(self.xy_magnitude_scale),
            "xy_direction_min_magnitude": float(self.xy_direction_min_magnitude),
            "contact_pos_weight": float(
                getattr(
                    self,
                    "contact_pos_weight",
                    LEGACY_CONTACT_POS_WEIGHT,
                )
            ),
        }

    @property
    def force_field_decoder(self) -> RegularGridForceFieldDecoder:
        """Expose the canonical name without changing state_dict keys."""
        return self.drifting_decoder

    def encode(
        self,
        batch: TensorBatch,
        visual_ablation: str = "dual",
    ) -> dict[str, torch.Tensor | None]:
        visual_memory, visual_mask = self.visual_encoder(
            require_tensor(batch, "realsense.cam1_color"),
            require_tensor(batch, "realsense.cam1_depth"),
            require_tensor(batch, "realsense.cam2_color"),
            require_tensor(batch, "realsense.cam2_depth"),
            ablation=visual_ablation,
        )
        state = self.state_encoder(
            torch.cat(
                [
                    require_tensor(batch, "robot.q"),
                    require_tensor(batch, "robot.O_T_EE"),
                ],
                dim=-1,
            )
        )
        dynamics = self.dynamics_encoder(
            torch.cat(
                [
                    require_tensor(batch, "ft300.wrench"),
                    require_tensor(batch, "gripper.gripper_gPO"),
                ],
                dim=-1,
            )
        )
        action = self.action_encoder(
            torch.cat(
                [
                    require_tensor(batch, "gello.future_q"),
                    require_tensor(batch, "gello.future_gripper_width"),
                ],
                dim=-1,
            )
        )
        state = self.state_dropout(state)
        dynamics = self.dynamics_dropout(dynamics)
        action = self.action_dropout(action)
        query = self.query_film(action, state)
        load, _ = self.load_attn(
            query,
            dynamics,
            dynamics,
            need_weights=False,
        )
        return {
            "visual_memory": visual_memory,
            "visual_mask": visual_mask,
            "action_query": query,
            "load": self.load_norm(query + load),
        }

    def forward(
        self,
        batch: TensorBatch,
        visual_ablation: str = "dual",
    ) -> dict[str, Any]:
        encoded = self.encode(batch, visual_ablation)
        force_z_normalized, force_xy_normalized, decoder_aux = self.drifting_decoder(
            encoded["action_query"],
            encoded["visual_memory"],
            encoded["load"],
            encoded["visual_mask"],
        )
        force_x_normalized = force_xy_normalized[..., 0:1]
        force_y_normalized = force_xy_normalized[..., 1:2]
        force_x = force_x_normalized * self.force_std[..., 0:1] + self.force_mean[..., 0:1]
        force_y = force_y_normalized * self.force_std[..., 1:2] + self.force_mean[..., 1:2]
        force_z = force_z_normalized * self.force_std[..., 2:3] + self.force_mean[..., 2:3]
        contact_logits = decoder_aux["contact_logits"]
        visual_contact_logits = decoder_aux["visual_contact_logits"]
        spatial_features = decoder_aux["spatial_features"]
        visual_features = decoder_aux["visual_features"]
        return {
            "force_z": force_z,
            "force_xy": torch.cat([force_x, force_y], dim=-1),
            "aux": {
                "force_fzxy_normalized": torch.cat(
                    [
                        force_z_normalized,
                        force_x_normalized,
                        force_y_normalized,
                    ],
                    dim=-1,
                ),
                "force_z_normalized": force_z_normalized,
                "force_x_normalized": force_x_normalized,
                "force_y_normalized": force_y_normalized,
                "fz_contact_logits": contact_logits,
                "visual_contact_logits": visual_contact_logits,
                "drifting_visual_features": visual_features,
                "drifting_spatial_features": spatial_features,
                "fz_spatial_features": spatial_features,
                "xy_spatial_features": spatial_features,
                "visual_memory": encoded["visual_memory"],
                "force_mean": self.force_mean,
                "force_std": self.force_std,
                "xy_magnitude_scale": self.xy_magnitude_scale,
                "xy_direction_min_magnitude": (self.xy_direction_min_magnitude),
                "contact_pos_weight": force_z_normalized.new_tensor(
                    getattr(
                        self,
                        "contact_pos_weight",
                        LEGACY_CONTACT_POS_WEIGHT,
                    )
                ),
            },
        }

    @torch.no_grad()
    def inference(
        self,
        batch: TensorBatch,
        visual_ablation: str = "dual",
    ) -> dict[str, Any]:
        return self.forward(batch, visual_ablation)

    @torch.no_grad()
    def predict_next(
        self,
        previous_observation: Mapping[str, torch.Tensor],
        previous_pose_matrix: torch.Tensor,
        executed_action: torch.Tensor,
    ) -> torch.Tensor:
        """Generate one force frame from the last four observations and action."""
        if executed_action.ndim == 1:
            executed_action = executed_action.unsqueeze(0)
        if executed_action.ndim != 2 or tuple(executed_action.shape[1:]) != (8,):
            raise ValueError(f"executed_action must be [B,8], got {tuple(executed_action.shape)}")
        rgb = previous_observation["rgb"]
        depth = previous_observation["depth"]
        if tuple(rgb.shape[-3:]) == (3, 480, 640):
            rgb, depth = prepare_for_frozen_encoder(rgb, depth)
        depth = depth.to(dtype=torch.int32)
        pose = previous_pose_matrix
        if pose.ndim == 3:
            pose = pose.unsqueeze(0)
        if pose.ndim != 4 or tuple(pose.shape[-2:]) != (4, 4):
            raise ValueError(f"previous_pose_matrix must be [B,4,4] or [B,4,4,4], got {tuple(pose.shape)}")
        if tuple(rgb.shape[1:3]) != (4, 2) or tuple(depth.shape[1:3]) != (4, 2):
            raise ValueError("TactiGen expects RGB-D history [B,4,2,C,H,W]")
        batch = {
            "realsense.cam1_color": rgb[:, :, 0],
            "realsense.cam1_depth": depth[:, :, 0],
            "realsense.cam2_color": rgb[:, :, 1],
            "realsense.cam2_depth": depth[:, :, 1],
            "robot.q": previous_observation["lowdim"][..., :7],
            "robot.O_T_EE": matrix_to_pose_xyzw_fast(pose),
            "ft300.wrench": previous_observation["lowdim"][..., 21:27],
            "gripper.gripper_gPO": previous_observation["lowdim"][..., 27:28],
            "gello.future_q": executed_action[:, None, :7],
            "gello.future_gripper_width": executed_action[:, None, 7:8],
        }
        output = self.inference(batch)
        return torch.cat([output["force_xy"], output["force_z"]], dim=-1)[:, 0]


def _spatial_smooth(value: torch.Tensor) -> torch.Tensor:
    return (value[..., 1:, :, :] - value[..., :-1, :, :]).abs().mean() + (
        value[..., :, 1:, :] - value[..., :, :-1, :]
    ).abs().mean()


def _masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    return values[mask].mean() if bool(mask.any()) else values.sum() * 0.0


def _balanced_xy_regression(
    pred: torch.Tensor,
    target: torch.Tensor,
    contact: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    per_node = F.smooth_l1_loss(
        pred,
        target,
        reduction="none",
    ).mean(dim=-1)
    has_contact = bool(contact.any())
    noncontact = ~contact
    has_noncontact = bool(noncontact.any())
    contact_loss = _masked_mean(per_node, contact)
    noncontact_loss = _masked_mean(per_node, noncontact)
    if has_contact and has_noncontact:
        total = XY_CONTACT_WEIGHT * contact_loss + XY_NONCONTACT_WEIGHT * noncontact_loss
    elif has_contact:
        total = contact_loss
    else:
        total = noncontact_loss
    return total, contact_loss, noncontact_loss


def supervised_force_loss_components(
    outputs: Mapping[str, Any],
    batch: TensorBatch,
) -> dict[str, torch.Tensor]:
    target = require_tensor(batch, "xense.force").float()
    aux = outputs["aux"]
    target_normalized = (target - aux["force_mean"].to(target)) / aux["force_std"].to(target).clamp_min(1e-6)
    pred_z_norm = aux["force_z_normalized"].float()
    pred_xy_norm = torch.cat(
        (
            aux["force_x_normalized"].float(),
            aux["force_y_normalized"].float(),
        ),
        dim=-1,
    )
    target_z_norm = target_normalized[..., 2:3]
    target_xy_norm = target_normalized[..., 0:2]
    contact = contact_from_fz(target[..., 2])
    fz_regression = F.smooth_l1_loss(pred_z_norm, target_z_norm)
    fz_smooth = _spatial_smooth(pred_z_norm)
    contact_float = contact.to(pred_z_norm.dtype)
    pos_weight_value = aux.get(
        "contact_pos_weight",
        LEGACY_CONTACT_POS_WEIGHT,
    )
    pos_weight = (
        pos_weight_value.to(pred_z_norm)
        if isinstance(pos_weight_value, torch.Tensor)
        else pred_z_norm.new_tensor(pos_weight_value)
    )
    fz_contact = F.binary_cross_entropy_with_logits(
        aux["fz_contact_logits"].float(),
        contact_float,
        pos_weight=pos_weight,
    )
    visual_contact = F.binary_cross_entropy_with_logits(
        aux["visual_contact_logits"].float(),
        contact_float,
        pos_weight=pos_weight,
    )
    xy_regression, xy_contact, xy_noncontact = _balanced_xy_regression(
        pred_xy_norm,
        target_xy_norm,
        contact,
    )

    pred_xy = outputs["force_xy"].float()
    target_xy = target[..., :2]
    pred_magnitude = torch.linalg.vector_norm(pred_xy, dim=-1)
    target_magnitude = torch.linalg.vector_norm(target_xy, dim=-1)
    magnitude_scale = aux["xy_magnitude_scale"].to(target).clamp_min(1e-6)
    xy_magnitude = _masked_mean(
        F.smooth_l1_loss(
            pred_magnitude / magnitude_scale,
            target_magnitude / magnitude_scale,
            reduction="none",
        ),
        contact,
    )
    direction_mask = contact & (target_magnitude >= aux["xy_direction_min_magnitude"].to(target))
    direction_cosine = F.cosine_similarity(
        pred_xy,
        target_xy,
        dim=-1,
        eps=max(float(magnitude_scale) * 1e-3, 1e-8),
    )
    xy_direction = _masked_mean(1.0 - direction_cosine, direction_mask)

    weighted_fz_regression = FZ_REGRESSION_WEIGHT * fz_regression
    weighted_fz_smooth = FZ_SMOOTH_WEIGHT * fz_smooth
    weighted_fz_contact = FZ_CONTACT_WEIGHT * fz_contact
    weighted_visual_contact = VISUAL_CONTACT_WEIGHT * visual_contact
    weighted_xy_magnitude = XY_MAGNITUDE_WEIGHT * xy_magnitude
    weighted_xy_direction = XY_DIRECTION_WEIGHT * xy_direction
    fz_loss = weighted_fz_regression + weighted_fz_smooth + weighted_fz_contact + weighted_visual_contact
    xy_loss = xy_regression + weighted_xy_magnitude + weighted_xy_direction
    return {
        "loss": fz_loss + xy_loss,
        "supervised_loss": fz_loss + xy_loss,
        "fz_loss": fz_loss,
        "xy_loss": xy_loss,
        "fz_regression": fz_regression,
        "fz_smooth": fz_smooth,
        "fz_contact": fz_contact,
        "visual_contact": visual_contact,
        "xy_regression": xy_regression,
        "xy_contact_regression": xy_contact,
        "xy_noncontact_regression": xy_noncontact,
        "xy_magnitude": xy_magnitude,
        "xy_direction": xy_direction,
        "weighted_fz_regression": weighted_fz_regression,
        "weighted_fz_smooth": weighted_fz_smooth,
        "weighted_fz_contact": weighted_fz_contact,
        "weighted_visual_contact": weighted_visual_contact,
        "weighted_xy_magnitude": weighted_xy_magnitude,
        "weighted_xy_direction": weighted_xy_direction,
    }


def drifting_force_loss_components(
    outputs: Mapping[str, Any],
    batch: TensorBatch,
) -> dict[str, torch.Tensor]:
    components = supervised_force_loss_components(outputs, batch)
    target = require_tensor(batch, "xense.force").float()
    aux = outputs["aux"]
    target_z_normalized = (target[..., 2:3] - aux["force_mean"].to(target)[..., 2:3]) / aux["force_std"].to(
        target
    )[..., 2:3].clamp_min(1e-6)
    drifting, metrics = multiscale_fz_field_drifting_loss(
        aux["force_z_normalized"].float(),
        target_z_normalized,
    )
    weighted_drifting = DRIFTING_WEIGHT * drifting
    components["drifting"] = drifting
    components["weighted_drifting"] = weighted_drifting
    components["loss"] = components["loss"] + weighted_drifting
    components.update({f"drifting_{name}": value for name, value in metrics.items()})
    return components


def migrate_legacy_drifting_state_dict(
    model: TactiGenForceFieldModel,
    legacy_state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Split a legacy shared XYZ MLP into exact Fz and XY MLP copies."""
    migrated = model.state_dict()
    shared_prefix = "drifting_decoder.output_head."
    for name, value in legacy_state_dict.items():
        if not name.startswith(shared_prefix):
            if name in migrated and migrated[name].shape == value.shape:
                migrated[name] = value
            continue
        suffix = name.removeprefix(shared_prefix)
        if suffix in {"0.weight", "0.bias", "1.weight", "1.bias"}:
            migrated[f"drifting_decoder.fz_head.{suffix}"] = value
            migrated[f"drifting_decoder.xy_head.{suffix}"] = value
        elif suffix == "3.weight":
            migrated["drifting_decoder.fz_head.3.weight"] = value[0:1]
            migrated["drifting_decoder.xy_head.3.weight"] = value[1:3]
        elif suffix == "3.bias":
            migrated["drifting_decoder.fz_head.3.bias"] = value[0:1]
            migrated["drifting_decoder.xy_head.3.bias"] = value[1:3]
    return migrated


def drifting_force_loss(
    outputs: Mapping[str, Any],
    batch: TensorBatch,
) -> torch.Tensor:
    return drifting_force_loss_components(outputs, batch)["loss"]


# Compatibility aliases for historical imports and full-object torch pickles.
# They intentionally point at the canonical classes instead of subclassing.
DriftingForceFieldDecoder = RegularGridForceFieldDecoder
DriftingXYZActionQueryCMTForceModel = TactiGenForceFieldModel


__all__ = [
    "DRIFTING_DECODER_DIM",
    "DRIFTING_FIELD_ORDER",
    "DRIFTING_LAYOUT",
    "LEGACY_DRIFTING_LAYOUT",
    "DRIFTING_WEIGHT",
    "XY_DIRECTION_WEIGHT",
    "XY_MAGNITUDE_WEIGHT",
    "LEGACY_CONTACT_POS_WEIGHT",
    "DriftingForceFieldDecoder",
    "DriftingXYZActionQueryCMTForceModel",
    "RegularGridForceFieldDecoder",
    "SpatialConvBlock",
    "TactiGenForceFieldModel",
    "checkpoint_contact_pos_weight",
    "drifting_force_loss",
    "drifting_force_loss_components",
    "loss_weight_config",
    "migrate_legacy_drifting_state_dict",
    "physical_force",
    "set_calibrated_loss_weights",
    "supervised_force_loss_components",
    "validate_contact_pos_weight",
]
