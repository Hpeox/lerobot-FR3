"""Self-contained visual and conditioning components used by ACMT-DP."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from .modeling_dformer import DFormerv2_B, DFormerv2_L, DFormerv2_S

TensorBatch = Mapping[str, torch.Tensor]


def require_tensor(batch: TensorBatch, key: str) -> torch.Tensor:
    if key not in batch:
        raise KeyError(f"Missing model input: {key}")
    return batch[key]


def coordinate_grid(
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([xx, yy], dim=-1)


def _group_count(channels: int, preferred: int = 8) -> int:
    for groups in range(min(preferred, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class TemporalAttention(nn.Module):
    """Fuse four frames independently at every camera/spatial location."""

    def __init__(
        self,
        dim: int,
        t_obs: int = 4,
        heads: int = 4,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.t_obs = int(t_obs)
        self.time_embed = nn.Parameter(torch.zeros(1, self.t_obs, 1, dim))
        nn.init.trunc_normal_(self.time_embed, std=0.02)
        self.attn = nn.MultiheadAttention(
            dim,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 4 or tokens.shape[1] != self.t_obs:
            raise ValueError(f"Expected visual tokens [B,{self.t_obs},N,D], got {tuple(tokens.shape)}")
        bsz, t_obs, locations, dim = tokens.shape
        sequence = tokens + self.time_embed.to(tokens)
        sequence = sequence.permute(0, 2, 1, 3).reshape(
            bsz * locations,
            t_obs,
            dim,
        )
        latest = sequence[:, -1:]
        pooled, _ = self.attn(
            latest,
            sequence,
            sequence,
            need_weights=False,
        )
        return self.norm(latest + pooled).reshape(bsz, locations, dim)


class DFormerv2DualRGBDEncoder(nn.Module):
    """Frozen shared DFormerv2-S stage-2 encoder for two 128px RGB-D ROIs."""

    CHANNELS = {"small": 256, "base": 320, "large": 448}

    def __init__(
        self,
        visual_dim: int = 160,
        t_obs: int = 4,
        num_heads: int = 4,
        variant: str = "small",
        repo_path: str | None = None,
        checkpoint: str | None = None,
        depth_scale: float = 1000.0,
        view_dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if variant not in self.CHANNELS:
            raise ValueError(f"Unsupported DFormerv2 variant: {variant}")
        factory = {
            "small": DFormerv2_S,
            "base": DFormerv2_B,
            "large": DFormerv2_L,
        }[variant]
        self.backbone = factory(out_indices=(2,))
        # Converted LeRobot checkpoints contain the complete backbone.  The
        # legacy path arguments remain accepted so generator configs can be
        # reconstructed without retaining an external runtime dependency.
        del repo_path, checkpoint
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()

        channels = self.CHANNELS[variant]
        self.visual_dim = int(visual_dim)
        self.t_obs = int(t_obs)
        self.depth_scale = float(depth_scale)
        self.view_dropout = float(view_dropout)
        self.wrist1_proj = nn.Linear(channels, visual_dim)
        self.wrist2_proj = nn.Linear(channels, visual_dim)
        self.wrist1_norm = nn.LayerNorm(visual_dim)
        self.wrist2_norm = nn.LayerNorm(visual_dim)
        self.spatial_embed = nn.Parameter(torch.zeros(1, 1, 64, visual_dim))
        self.camera_embed = nn.Parameter(torch.zeros(1, 2, 1, visual_dim))
        nn.init.trunc_normal_(self.spatial_embed, std=0.02)
        nn.init.trunc_normal_(self.camera_embed, std=0.02)
        self.temporal = TemporalAttention(
            visual_dim,
            t_obs,
            num_heads,
            dropout=0.10,
        )
        self.register_buffer(
            "rgb_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "rgb_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def _rgb(self, value: torch.Tensor) -> torch.Tensor:
        value = value.float()
        if value.numel() and value.detach().max() > 2.0:
            value = value / 255.0
        return (value.clamp(0.0, 1.0) - self.rgb_mean.to(value)) / self.rgb_std.to(value)

    def _depth(self, value: torch.Tensor) -> torch.Tensor:
        meters = value.float() / self.depth_scale
        valid = (value > 0) & (value < 65535) & (meters >= 0.05) & (meters <= 2.0)
        normalized = ((meters - 0.05) / 1.95).clamp(0.0, 1.0)
        return torch.where(valid, normalized, torch.ones_like(normalized))

    def _frozen_stage2(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            x = self.backbone.patch_embed(self._rgb(rgb))
            x_e = self._depth(depth)[:, 0].unsqueeze(1)
            x_out = x
            for layer_idx in range(3):
                x_out, x = self.backbone.layers[layer_idx](x, x_e)
            x_out = self.backbone.extra_norms[1](x_out)
            return x_out.permute(0, 3, 1, 2).contiguous()

    def _view_mask(
        self,
        batch_size: int,
        device: torch.device,
        ablation: str,
    ) -> torch.Tensor | None:
        mask = torch.zeros(batch_size, 128, dtype=torch.bool, device=device)
        if ablation == "wrist1_only":
            mask[:, 64:] = True
        elif ablation == "wrist2_only":
            mask[:, :64] = True
        elif ablation not in {"dual", "zero", "shuffled", "stale"}:
            raise ValueError(f"Unsupported visual ablation: {ablation}")
        elif self.training and ablation == "dual" and self.view_dropout > 0:
            choice = torch.rand(batch_size, device=device)
            mask[choice < self.view_dropout, :64] = True
            mask[
                (choice >= self.view_dropout) & (choice < 2 * self.view_dropout),
                64:,
            ] = True
        return mask if bool(mask.any()) else None

    def forward(
        self,
        cam1_rgb: torch.Tensor,
        cam1_depth: torch.Tensor,
        cam2_rgb: torch.Tensor,
        cam2_depth: torch.Tensor,
        ablation: str = "dual",
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        expected = (self.t_obs, 3, 128, 128)
        if cam1_rgb.shape[1:] != expected:
            raise ValueError(f"Expected wrist RGB [B,{self.t_obs},3,128,128], got {tuple(cam1_rgb.shape)}")
        if ablation == "stale":
            cam1_rgb = cam1_rgb[:, :1].expand_as(cam1_rgb)
            cam1_depth = cam1_depth[:, :1].expand_as(cam1_depth)
            cam2_rgb = cam2_rgb[:, :1].expand_as(cam2_rgb)
            cam2_depth = cam2_depth[:, :1].expand_as(cam2_depth)
        bsz = cam1_rgb.shape[0]
        rgb = torch.stack([cam1_rgb, cam2_rgb], dim=1).reshape(
            bsz * 2 * self.t_obs,
            3,
            128,
            128,
        )
        depth = torch.stack([cam1_depth, cam2_depth], dim=1).reshape(
            bsz * 2 * self.t_obs,
            1,
            128,
            128,
        )
        features = self._frozen_stage2(rgb, depth)
        if features.shape[-2:] != (8, 8):
            raise ValueError(f"DFormerv2 stage-2 must be 8x8, got {tuple(features.shape[-2:])}")
        features = (
            features.flatten(2)
            .transpose(1, 2)
            .reshape(
                bsz,
                2,
                self.t_obs,
                64,
                -1,
            )
        )
        cam1 = self.wrist1_norm(self.wrist1_proj(features[:, 0]))
        cam2 = self.wrist2_norm(self.wrist2_proj(features[:, 1]))
        cameras = torch.stack([cam1, cam2], dim=1)
        cameras = cameras + self.spatial_embed[:, None] + self.camera_embed[:, :, None]
        fused = torch.stack(
            [self.temporal(cameras[:, camera]) for camera in range(2)],
            dim=1,
        )
        memory = fused.reshape(bsz, 128, self.visual_dim)
        if ablation == "zero":
            memory = torch.zeros_like(memory)
        elif ablation == "shuffled" and bsz > 1:
            memory = memory.roll(1, dims=0)
        return memory, self._view_mask(bsz, memory.device, ablation)


class TinyDualRGBDEncoder(nn.Module):
    """Fast shape-compatible encoder for tests and CPU smoke checks."""

    def __init__(
        self,
        visual_dim: int = 160,
        t_obs: int = 4,
        **_: Any,
    ) -> None:
        super().__init__()
        self.visual_dim = visual_dim
        self.t_obs = t_obs
        self.stem = nn.Conv2d(4, visual_dim, 4, 4)
        self.temporal = TemporalAttention(visual_dim, t_obs, 4, 0.0)
        self.camera_embed = nn.Parameter(torch.zeros(1, 2, 1, visual_dim))

    def forward(self, rgb1, depth1, rgb2, depth2, ablation: str = "dual"):
        values = []
        for rgb, depth in ((rgb1, depth1), (rgb2, depth2)):
            bsz, time = rgb.shape[:2]
            x = torch.cat(
                [rgb.float(), depth.float() / 1000.0],
                dim=2,
            ).reshape(bsz * time, 4, 128, 128)
            x = F.adaptive_avg_pool2d(self.stem(x), (8, 8))
            x = x.flatten(2).transpose(1, 2)
            values.append(self.temporal(x.reshape(bsz, time, 64, self.visual_dim)))
        memory = torch.stack(values, dim=1) + self.camera_embed
        return memory.reshape(rgb1.shape[0], 128, self.visual_dim), None


class HistoryEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(self.input(value))
        return self.norm(output[:, -1])


class ModalityDropout(nn.Module):
    def __init__(self, probability: float) -> None:
        super().__init__()
        self.probability = float(probability)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability <= 0:
            return value
        keep = (torch.rand(value.shape[0], device=value.device) >= self.probability).to(value.dtype)
        return value * keep.view(
            value.shape[0],
            *([1] * (value.ndim - 1)),
        )


class QueryFiLM(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.mod = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim * 2),
        )
        nn.init.zeros_(self.mod[-1].weight)
        nn.init.zeros_(self.mod[-1].bias)
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        action: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        gamma, beta = self.mod(state).chunk(2, dim=-1)
        return self.norm((1.0 + gamma[:, None]) * action + beta[:, None])


class SpatialConvBlock(nn.Module):
    """Depthwise 3x3 plus pointwise 1x1 convolution on each tactile side."""

    def __init__(self, dim: int, dropout: float = 0.075) -> None:
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(
                dim,
                dim,
                3,
                padding=1,
                groups=dim,
                bias=False,
            ),
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.GroupNorm(_group_count(dim), dim),
            nn.SiLU(),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch_time, sides, height, width, dim = value.shape
        flat = value.permute(0, 1, 4, 2, 3).reshape(
            batch_time * sides,
            dim,
            height,
            width,
        )
        local = (
            self.local(flat)
            .reshape(
                batch_time,
                sides,
                dim,
                height,
                width,
            )
            .permute(0, 1, 3, 4, 2)
        )
        value = self.norm1(value + local)
        return self.norm2(value + self.ffn(value))


__all__ = [
    "DFormerv2DualRGBDEncoder",
    "HistoryEncoder",
    "ModalityDropout",
    "QueryFiLM",
    "SpatialConvBlock",
    "TensorBatch",
    "TinyDualRGBDEncoder",
    "coordinate_grid",
    "require_tensor",
]
