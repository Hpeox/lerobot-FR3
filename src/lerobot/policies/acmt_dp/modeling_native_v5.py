"""Inference-only components compatible with ACMT-DP native v5 checkpoints."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torchvision.models import resnet18


def replace_batchnorm_with_groupnorm(module: nn.Module) -> nn.Module:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            groups = max(1, child.num_features // 16)
            replacement = nn.GroupNorm(groups, child.num_features, eps=child.eps, affine=child.affine)
            if child.affine:
                replacement.weight.data.copy_(child.weight.data)
                replacement.bias.data.copy_(child.bias.data)
            setattr(module, name, replacement)
        else:
            replace_batchnorm_with_groupnorm(child)
    return module


class SpatialSoftmax(nn.Module):
    def __init__(self, input_shape: tuple[int, int, int], num_kp: int = 32, temperature: float = 1.0) -> None:
        super().__init__()
        channels, height, width = (int(value) for value in input_shape)
        self.input_shape = (channels, height, width)
        self.num_kp = int(num_kp)
        self.keypoint_conv = nn.Conv2d(channels, self.num_kp, kernel_size=1)
        self.register_buffer(
            "pos_x", torch.linspace(-1.0, 1.0, width).repeat(height, 1).reshape(1, height * width)
        )
        self.register_buffer(
            "pos_y",
            torch.linspace(-1.0, 1.0, height).reshape(height, 1).repeat(1, width).reshape(1, height * width),
        )
        self.register_buffer("temperature", torch.tensor(float(temperature)))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if tuple(value.shape[-3:]) != self.input_shape:
            raise ValueError(f"SpatialSoftmax expects [...,{self.input_shape}], got {tuple(value.shape)}")
        logits = self.keypoint_conv(value).reshape(value.shape[0], self.num_kp, -1)
        attention = F.softmax(logits / self.temperature.clamp_min(1e-6), dim=-1)
        x = (attention * self.pos_x.unsqueeze(1)).sum(dim=-1)
        y = (attention * self.pos_y.unsqueeze(1)).sum(dim=-1)
        return torch.cat([x, y], dim=-1)


class RobomimicResNet18Spatial(nn.Module):
    output_dim = 64

    def __init__(self, *, num_keypoints: int = 32) -> None:
        super().__init__()
        backbone = resnet18(weights=None)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        replace_batchnorm_with_groupnorm(self.backbone)
        self.spatial = SpatialSoftmax((512, 7, 9), num_kp=num_keypoints)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.backbone(images))


class NativeV5VisionEncoder(nn.Module):
    feature_dim = 64
    camera_count = 4

    def __init__(self, *, use_group_norm: bool = True, num_keypoints: int = 32) -> None:
        super().__init__()
        if not use_group_norm:
            raise ValueError("native_dp_v5 requires GroupNorm visual encoders")
        if num_keypoints != 32:
            raise ValueError("native_dp_v5 uses 32 SpatialSoftmax keypoints")
        self.camera_encoders = nn.ModuleList(
            RobomimicResNet18Spatial(num_keypoints=num_keypoints) for _ in range(4)
        )
        self.random_crop = False

    @staticmethod
    def preprocess(images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 5 or tuple(images.shape[1:3]) != (4, 3):
            raise ValueError(f"images must be [N,4,3,H,W], got {tuple(images.shape)}")
        if tuple(images.shape[-2:]) not in {(480, 640), (240, 320)}:
            raise ValueError(
                f"native_dp_v5 RGB must be raw 480x640 or resized 240x320 (4:3), got {tuple(images.shape[-2:])}"
            )
        source_integer = not torch.is_floating_point(images)
        values = images.float()
        if source_integer or (values.numel() and float(values.detach().amax()) > 1.0):
            values = values / 255.0
        flat = values.reshape(-1, 3, values.shape[-2], values.shape[-1])
        if tuple(flat.shape[-2:]) != (240, 320):
            flat = F.interpolate(flat, size=(240, 320), mode="bilinear", align_corners=False, antialias=True)
            flat = flat.mul(255.0).clamp(0.0, 255.0).round().div(255.0)
        cropped = flat[..., 12:228, 16:304]
        return cropped.mul(2.0).sub(1.0).reshape(images.shape[0], 4, 3, 216, 288)

    def encode_frames(self, images: torch.Tensor) -> torch.Tensor:
        processed = self.preprocess(images)
        return torch.stack(
            [encoder(processed[:, index]) for index, encoder in enumerate(self.camera_encoders)], dim=1
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim == 6:
            batch, history, cameras = images.shape[:3]
            if cameras != 4:
                raise ValueError("native_dp_v5 requires four camera views")
            features = self.encode_frames(images.reshape(batch * history, cameras, *images.shape[3:]))
            return features.reshape(batch, history, cameras, self.feature_dim)
        return self.encode_frames(images)


class FrameTactileEncoder(nn.Module):
    def __init__(
        self, force_mean: Iterable[float], force_std: Iterable[float], output_dim: int = 160
    ) -> None:
        super().__init__()
        if output_dim != 160:
            raise ValueError("native_dp_v5 tactile output is fixed at 160")
        self.register_buffer(
            "force_mean", torch.as_tensor(tuple(force_mean), dtype=torch.float32).reshape(1, 1, 1, 1, 1, 3)
        )
        self.register_buffer(
            "force_std",
            torch.as_tensor(tuple(force_std), dtype=torch.float32).reshape(1, 1, 1, 1, 1, 3).clamp_min(1e-6),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 80, kernel_size=3, padding=1),
            nn.GroupNorm(8, 80),
            nn.GELU(),
        )
        self.output_norm = nn.LayerNorm(160)

    def forward(self, force: torch.Tensor) -> torch.Tensor:
        if force.ndim != 6 or tuple(force.shape[1:]) != (4, 2, 35, 20, 3):
            raise ValueError(f"tactile must be [B,4,2,35,20,3], got {tuple(force.shape)}")
        batch, history = force.shape[:2]
        normalized = (force.float() - self.force_mean) / self.force_std
        images = normalized.permute(0, 1, 2, 5, 3, 4).reshape(batch * history * 2, 3, 35, 20)
        encoded = self.spatial(images)
        pooled = F.adaptive_avg_pool2d(encoded, 1).flatten(1)
        return self.output_norm(pooled.reshape(batch, history, 2, 80).reshape(batch, history, 160))


class _InputStats(nn.Module):
    def __init__(
        self, minimum: Iterable[float], maximum: Iterable[float], mean: Iterable[float], std: Iterable[float]
    ):
        super().__init__()
        self.register_buffer("max", torch.as_tensor(tuple(maximum), dtype=torch.float32))
        self.register_buffer("mean", torch.as_tensor(tuple(mean), dtype=torch.float32))
        self.register_buffer("min", torch.as_tensor(tuple(minimum), dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(tuple(std), dtype=torch.float32).clamp_min(1e-6))


class _FieldNormalizer(nn.Module):
    def __init__(self, minimum: Iterable[float], maximum: Iterable[float]):
        super().__init__()
        minimum = torch.as_tensor(tuple(minimum), dtype=torch.float32)
        maximum = torch.as_tensor(tuple(maximum), dtype=torch.float32)
        raw_span = maximum - minimum
        variable = raw_span >= 1e-4
        span = raw_span.clamp_min(1e-4)
        # Match diffusion_policy's limits normalizer, including its zero
        # center for constant dimensions.
        scale = torch.where(variable, 2.0 / span, torch.ones_like(span))
        offset = torch.where(variable, -1.0 - scale * minimum, -minimum)
        self.register_buffer("offset", offset)
        self.register_buffer("scale", scale)
        self.input_stats = _InputStats(minimum, maximum, (minimum + maximum) / 2.0, span / 2.0)

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.scale + self.offset

    def unnormalize(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.offset) / self.scale.clamp_min(1e-12)


class NativeV5LinearNormalizer(nn.Module):
    def __init__(
        self,
        state_min: Iterable[float],
        state_max: Iterable[float],
        action_min: Iterable[float],
        action_max: Iterable[float],
    ):
        super().__init__()
        self.params_dict = nn.ModuleDict(
            {
                "state": _FieldNormalizer(state_min, state_max),
                "action": _FieldNormalizer(action_min, action_max),
            }
        )

    def __getitem__(self, key: str) -> _FieldNormalizer:
        return self.params_dict[key]


__all__ = ["FrameTactileEncoder", "NativeV5LinearNormalizer", "NativeV5VisionEncoder"]
