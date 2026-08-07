# This file contains a minimal inference port of the MIT-licensed
# real-stanford/diffusion_policy ConditionalUnet1D implementation.

from __future__ import annotations

import math

import einops
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class Downsample1d(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.conv(value)


class Upsample1d(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.conv(value)


class Conv1dBlock(nn.Module):
    def __init__(self, inp_channels: int, out_channels: int, kernel_size: int, n_groups: int = 8) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.block(value)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        scale = math.log(10000) / (half_dim - 1)
        embedding = torch.exp(torch.arange(half_dim, device=value.device) * -scale)
        embedding = value[:, None] * embedding[None, :]
        return torch.cat((embedding.sin(), embedding.cos()), dim=-1)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        cond_predict_scale: bool = False,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )
        cond_channels = out_channels * 2 if cond_predict_scale else out_channels
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange("batch t -> batch t 1"),
        )
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        output = self.blocks[0](value)
        embedding = self.cond_encoder(condition)
        if self.cond_predict_scale:
            embedding = embedding.reshape(embedding.shape[0], 2, self.out_channels, 1)
            output = embedding[:, 0] * output + embedding[:, 1]
        else:
            output = output + embedding
        return self.blocks[1](output) + self.residual_conv(value)


class ConditionalUnet1D(nn.Module):
    """State-dict-compatible port of the pinned ACMT-DP diffusion U-Net."""

    def __init__(
        self,
        input_dim: int,
        local_cond_dim: int | None = None,
        global_cond_dim: int | None = None,
        diffusion_step_embed_dim: int = 256,
        down_dims: list[int] | tuple[int, ...] = (256, 512, 1024),
        kernel_size: int = 3,
        n_groups: int = 8,
        cond_predict_scale: bool = False,
    ) -> None:
        super().__init__()
        all_dims = [input_dim, *down_dims]
        start_dim = down_dims[0]
        step_dim = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(step_dim),
            nn.Linear(step_dim, step_dim * 4),
            nn.Mish(),
            nn.Linear(step_dim * 4, step_dim),
        )
        cond_dim = step_dim + (global_cond_dim or 0)
        in_out = list(zip(all_dims[:-1], all_dims[1:], strict=True))

        self.local_cond_encoder: nn.ModuleList | None = None
        if local_cond_dim is not None:
            dim_out = in_out[0][1]
            self.local_cond_encoder = nn.ModuleList(
                [
                    ConditionalResidualBlock1D(
                        local_cond_dim,
                        dim_out,
                        cond_dim,
                        kernel_size,
                        n_groups,
                        cond_predict_scale,
                    ),
                    ConditionalResidualBlock1D(
                        local_cond_dim,
                        dim_out,
                        cond_dim,
                        kernel_size,
                        n_groups,
                        cond_predict_scale,
                    ),
                ]
            )

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    mid_dim, mid_dim, cond_dim, kernel_size, n_groups, cond_predict_scale
                ),
                ConditionalResidualBlock1D(
                    mid_dim, mid_dim, cond_dim, kernel_size, n_groups, cond_predict_scale
                ),
            ]
        )

        self.down_modules = nn.ModuleList()
        for index, (dim_in, dim_out) in enumerate(in_out):
            last = index >= len(in_out) - 1
            self.down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_in, dim_out, cond_dim, kernel_size, n_groups, cond_predict_scale
                        ),
                        ConditionalResidualBlock1D(
                            dim_out, dim_out, cond_dim, kernel_size, n_groups, cond_predict_scale
                        ),
                        Downsample1d(dim_out) if not last else nn.Identity(),
                    ]
                )
            )

        self.up_modules = nn.ModuleList()
        for index, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            last = index >= len(in_out) - 1
            self.up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_out * 2, dim_in, cond_dim, kernel_size, n_groups, cond_predict_scale
                        ),
                        ConditionalResidualBlock1D(
                            dim_in, dim_in, cond_dim, kernel_size, n_groups, cond_predict_scale
                        ),
                        Upsample1d(dim_in) if not last else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | float | int,
        local_cond: torch.Tensor | None = None,
        global_cond: torch.Tensor | None = None,
        **_: object,
    ) -> torch.Tensor:
        sample = einops.rearrange(sample, "b h t -> b t h")
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif timesteps.ndim == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])
        global_feature = self.diffusion_step_encoder(timesteps)
        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], dim=-1)

        local_features: list[torch.Tensor] = []
        if local_cond is not None:
            if self.local_cond_encoder is None:
                raise ValueError("local_cond was supplied but this U-Net has no local conditioner")
            local_cond = einops.rearrange(local_cond, "b h t -> b t h")
            local_features = [module(local_cond, global_feature) for module in self.local_cond_encoder]

        value = sample
        skips: list[torch.Tensor] = []
        for index, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            value = resnet(value, global_feature)
            if index == 0 and local_features:
                value = value + local_features[0]
            value = resnet2(value, global_feature)
            skips.append(value)
            value = downsample(value)
        for module in self.mid_modules:
            value = module(value, global_feature)
        for index, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            value = torch.cat((value, skips.pop()), dim=1)
            value = resnet(value, global_feature)
            # Preserve the historical upstream behavior for checkpoint parity.
            if index == len(self.up_modules) and local_features:
                value = value + local_features[1]
            value = resnet2(value, global_feature)
            value = upsample(value)
        value = self.final_conv(value)
        return einops.rearrange(value, "b t h -> b h t")
