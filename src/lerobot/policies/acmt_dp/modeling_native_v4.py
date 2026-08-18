"""Self-contained Native-DP v4 model components.

The upstream v4 policy uses the observation encoder and normalizer from
Diffusion Policy.  This module keeps the small subset needed for inference in
LeRobot, including the original module names so a v4 ``best.pt`` can be loaded
strictly without importing the training repository.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torchvision.models import resnet18


class _NativeMultiImageObsEncoder(nn.Module):
    """Minimal compatible ``MultiImageObsEncoder``.

    ``key_model_map.rgb`` is deliberately named like the vendored DP module;
    this is part of the v4 checkpoint ABI.  All four images share one ResNet.
    """

    def __init__(self) -> None:
        super().__init__()
        self._dummy_variable = nn.Parameter(torch.empty(0))
        backbone = resnet18(weights=None)
        backbone.fc = nn.Identity()
        self.key_model_map = nn.ModuleDict({"rgb": backbone})

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        images = [observations[f"rgb_{index}"] for index in range(4)]
        if any(image.ndim != 4 or tuple(image.shape[1:]) != (3, 224, 224) for image in images):
            raise ValueError("native RGB views must be [B,3,224,224]")
        batch = images[0].shape[0]
        values = torch.stack(images, dim=1).reshape(batch * 4, 3, 224, 224)
        features = self.key_model_map["rgb"](values)
        return features.reshape(batch, 4, 512)


class NativeVisionEncoder(nn.Module):
    """Shared scratch ResNet18 with the exact v4 resize/crop/quantization."""

    feature_dim = 512
    camera_count = 4

    def __init__(self, weights: str | None = None, frozen: bool = False) -> None:
        if weights not in (None, "none", "NONE"):
            raise ValueError("Native-DP v4 deployment only supports scratch ResNet18 weights")
        super().__init__()
        self.obs_encoder = _NativeMultiImageObsEncoder()
        self.weights_name = weights
        self._is_frozen = bool(frozen)
        if frozen:
            self.freeze()

    def freeze(self) -> None:
        self._is_frozen = True
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        super().train(False)

    def train(self, mode: bool = True) -> NativeVisionEncoder:
        super().train(mode)
        if self._is_frozen:
            super().train(False)
        return self

    @staticmethod
    def preprocess(images: torch.Tensor) -> torch.Tensor:
        """Return ImageNet-ready ``[N,4,3,224,224]`` tensors.

        Raw frames follow the training path exactly: resize to 256 with
        antialiased bilinear interpolation, center crop 224, clamp/round in
        uint8 space, then divide by 255.  Already-cropped 224 input is accepted
        for offline callers.
        """

        if images.ndim != 5 or images.shape[1] != 4 or images.shape[2] != 3:
            raise ValueError(f"images must be [N,4,3,H,W], got {tuple(images.shape)}")
        source_integer = not torch.is_floating_point(images)
        values = images.reshape(-1, 3, images.shape[-2], images.shape[-1]).float()
        if tuple(values.shape[-2:]) not in {(480, 640), (224, 224)}:
            raise ValueError("RGB resolution must be 480x640 or 224x224")
        if tuple(values.shape[-2:]) == (480, 640):
            if not source_integer and values.numel() and float(values.detach().amax()) <= 1.0:
                values = values * 255.0
            values = F.interpolate(
                values,
                size=(256, 256),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            values = values[..., 16:240, 16:240].clamp(0.0, 255.0).round() / 255.0
        elif source_integer or (values.numel() and float(values.detach().amax()) > 1.0):
            values = values / 255.0
        mean = values.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = values.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        values = (values.clamp(0.0, 1.0) - mean) / std
        return values.reshape(images.shape[0], 4, 3, 224, 224)

    def encode_frames(self, images: torch.Tensor) -> torch.Tensor:
        processed = self.preprocess(images)
        observations = {f"rgb_{index}": processed[:, index] for index in range(4)}
        return self.obs_encoder(observations)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim == 6:
            batch, history, cameras = images.shape[:3]
            if cameras != 4:
                raise ValueError("native-DP v4 requires four RGB cameras")
            encoded = self.encode_frames(images.reshape(batch * history, cameras, *images.shape[3:]))
            return encoded.reshape(batch, history, cameras, self.feature_dim)
        return self.encode_frames(images)


class FrameTactileEncoder(nn.Module):
    """Shared spatial force-field CNN producing one 160-D token per frame."""

    def __init__(
        self,
        force_mean: Iterable[float],
        force_std: Iterable[float],
        output_dim: int = 160,
    ) -> None:
        super().__init__()
        if output_dim != 160:
            raise ValueError("native_dp_v4 tactile output is fixed at 160")
        self.register_buffer(
            "force_mean",
            torch.as_tensor(tuple(force_mean), dtype=torch.float32).reshape(1, 1, 1, 1, 1, 3),
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
    def __init__(
        self,
        minimum: Iterable[float],
        maximum: Iterable[float],
        mean: Iterable[float],
        std: Iterable[float],
        mode: str,
    ) -> None:
        super().__init__()
        minimum = torch.as_tensor(tuple(minimum), dtype=torch.float32)
        maximum = torch.as_tensor(tuple(maximum), dtype=torch.float32)
        mean = torch.as_tensor(tuple(mean), dtype=torch.float32)
        std = torch.as_tensor(tuple(std), dtype=torch.float32).clamp_min(1e-6)
        if mode == "limits":
            span = (maximum - minimum).clamp_min(1e-4)
            scale = 2.0 / span
            offset = -1.0 - scale * minimum
        elif mode == "gaussian":
            scale = 1.0 / std
            offset = -mean * scale
        else:
            raise ValueError(f"unknown normalizer mode {mode}")
        self.register_buffer("offset", offset)
        self.register_buffer("scale", scale)
        self.input_stats = _InputStats(minimum, maximum, mean, std)

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.scale + self.offset

    def unnormalize(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.offset) / self.scale.clamp_min(1e-12)


class NativeLinearNormalizer(nn.Module):
    """Small ``LinearNormalizer`` subset with matching serialized names."""

    def __init__(
        self,
        state_mean: Iterable[float],
        state_std: Iterable[float],
        action_min: Iterable[float],
        action_max: Iterable[float],
    ) -> None:
        super().__init__()
        state_mean = tuple(state_mean)
        state_std = tuple(state_std)
        action_min = tuple(action_min)
        action_max = tuple(action_max)
        self.params_dict = nn.ModuleDict(
            {
                "state": _FieldNormalizer(
                    tuple(m - 4.0 * s for m, s in zip(state_mean, state_std, strict=True)),
                    tuple(m + 4.0 * s for m, s in zip(state_mean, state_std, strict=True)),
                    state_mean,
                    state_std,
                    "gaussian",
                ),
                "action": _FieldNormalizer(
                    action_min,
                    action_max,
                    tuple((a + b) * 0.5 for a, b in zip(action_min, action_max, strict=True)),
                    (1.0,) * len(action_min),
                    "limits",
                ),
            }
        )

    def __getitem__(self, key: str) -> _FieldNormalizer:
        return self.params_dict[key]


__all__ = ["FrameTactileEncoder", "NativeLinearNormalizer", "NativeVisionEncoder"]
