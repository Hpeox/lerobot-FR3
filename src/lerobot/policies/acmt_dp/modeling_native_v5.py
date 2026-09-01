"""Inference-only components compatible with ACMT-DP native v5 checkpoints."""

from __future__ import annotations

import importlib.metadata
from collections import OrderedDict
from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


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


class NativeV5CenterCropRandomizer(nn.Module):
    """Robomimic randomizer with the official DP eval-center behavior."""

    def __init__(self, input_shape, crop_height, crop_width, num_crops=1, pos_enc=False):
        super().__init__()
        self.input_shape = tuple(input_shape)
        self.crop_height = int(crop_height)
        self.crop_width = int(crop_width)
        self.num_crops = int(num_crops)
        self.pos_enc = bool(pos_enc)

    def output_shape_in(self, input_shape=None):
        return [self.input_shape[0] + (2 if self.pos_enc else 0), self.crop_height, self.crop_width]

    def output_shape_out(self, input_shape=None):
        return list(input_shape)

    def forward_in(self, inputs):
        if self.training:
            from robomimic.utils import obs_utils

            crops, _ = obs_utils.sample_random_image_crops(
                images=inputs,
                crop_height=self.crop_height,
                crop_width=self.crop_width,
                num_crops=self.num_crops,
                pos_enc=self.pos_enc,
            )
            return torch.cat([crops[:, index] for index in range(self.num_crops)], dim=0)
        top = (inputs.shape[-2] - self.crop_height) // 2
        left = (inputs.shape[-1] - self.crop_width) // 2
        cropped = inputs[..., top : top + self.crop_height, left : left + self.crop_width]
        if self.num_crops > 1:
            cropped = cropped.unsqueeze(1).expand(-1, self.num_crops, -1, -1, -1)
            cropped = cropped.reshape(-1, *cropped.shape[2:])
        return cropped

    def forward_out(self, inputs):
        if self.num_crops <= 1:
            return inputs
        batch = inputs.shape[0] // self.num_crops
        return inputs.reshape(batch, self.num_crops, *inputs.shape[1:]).mean(dim=1)


def _official_encoder() -> nn.Module:
    """Construct the Robomimic 0.2.0 encoder without a DP source checkout."""

    try:
        version = importlib.metadata.version("robomimic")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("Native-DP v5 requires robomimic==0.2.0") from exc
    if version != "0.2.0":
        raise RuntimeError(f"Native-DP v5 requires robomimic==0.2.0, found {version}")
    import robomimic.models.base_nets as rmbn
    from robomimic.models.obs_nets import obs_encoder_factory
    from robomimic.utils import obs_utils

    camera_names = ("top", "side", "wrist_left", "wrist_right")
    obs_utils.initialize_obs_modality_mapping_from_dict(
        {"low_dim": ["state"], "rgb": list(camera_names), "depth": [], "scan": []}
    )
    encoder_kwargs = {
        "low_dim": {
            "core_class": None,
            "core_kwargs": {},
            "obs_randomizer_class": None,
            "obs_randomizer_kwargs": {},
        },
        "rgb": {
            "core_class": "VisualCore",
            "core_kwargs": {
                "feature_dimension": 64,
                "flatten": True,
                "backbone_class": "ResNet18Conv",
                "backbone_kwargs": {"pretrained": False, "input_coord_conv": False},
                "pool_class": "SpatialSoftmax",
                "pool_kwargs": {
                    "num_kp": 32,
                    "learnable_temperature": False,
                    "temperature": 1.0,
                    "noise_std": 0.0,
                    "output_variance": False,
                },
            },
            "obs_randomizer_class": "CropRandomizer",
            "obs_randomizer_kwargs": {
                "crop_height": 216,
                "crop_width": 288,
                "num_crops": 1,
                "pos_enc": False,
            },
        },
    }
    shape_meta = OrderedDict([(name, [3, 240, 320]) for name in camera_names] + [("state", [8])])
    encoder = obs_encoder_factory(
        shape_meta,
        feature_activation=nn.ReLU,
        encoder_kwargs=encoder_kwargs,
    )
    replace_batchnorm_with_groupnorm(encoder)
    for name, randomizer in list(encoder.obs_randomizers.items()):
        if isinstance(randomizer, rmbn.CropRandomizer):
            replacement = NativeV5CenterCropRandomizer(
                input_shape=randomizer.input_shape,
                crop_height=randomizer.crop_height,
                crop_width=randomizer.crop_width,
                num_crops=randomizer.num_crops,
                pos_enc=randomizer.pos_enc,
            )
            encoder.obs_randomizers[name] = replacement
    if int(encoder.output_shape()[0]) != 264:
        raise RuntimeError("official Robomimic encoder must output 264 values")
    return encoder


def _preprocess_rgb(images: torch.Tensor) -> torch.Tensor:
    squeezed_history = images.ndim == 5
    if squeezed_history:
        images = images.unsqueeze(1)
    if images.ndim != 6 or tuple(images.shape[2:4]) != (4, 3):
        raise ValueError(f"images must be [B,T,4,3,H,W] or [B,4,3,H,W], got {tuple(images.shape)}")
    if tuple(images.shape[-2:]) not in {(480, 640), (240, 320)}:
        raise ValueError("native_dp_v5 RGB must be 480x640 or 240x320")
    values = images.float()
    if not torch.is_floating_point(images) or (values.numel() and float(values.detach().amax()) > 1.0):
        values = values / 255.0
    flat = values.reshape(-1, 3, values.shape[-2], values.shape[-1])
    if tuple(flat.shape[-2:]) == (480, 640):
        flat = F.interpolate(flat, size=(240, 320), mode="bilinear", align_corners=False, antialias=True)
        flat = flat.mul(255.0).clamp(0.0, 255.0).round().div(255.0)
    result = flat.reshape(images.shape[0], images.shape[1], 4, 3, 240, 320)
    # Keep deployment numerically identical to the training ABI:
    # [0,255] -> [0,1] -> [-1,1].
    result = result.mul(2.0).sub(1.0)
    return result[:, 0] if squeezed_history else result


class NativeV5VisionEncoder(nn.Module):
    """Official Robomimic RGB+state encoder, matching training checkpoints."""

    feature_dim = 64
    output_dim = 264

    def __init__(self, *, use_group_norm: bool = True, num_keypoints: int = 32) -> None:
        super().__init__()
        if not use_group_norm or num_keypoints != 32:
            raise ValueError("native_dp_v5 requires GroupNorm and 32 SpatialSoftmax keypoints")
        self.encoder = _official_encoder()
        self.random_crop = False

    @staticmethod
    def preprocess(images: torch.Tensor) -> torch.Tensor:
        values = _preprocess_rgb(images)
        if images.ndim == 5:
            flat = values.reshape(-1, 3, 240, 320)
            return flat[..., 12:228, 16:304].reshape(images.shape[0], 4, 3, 216, 288)
        flat = values.reshape(-1, 3, 240, 320)
        return flat[..., 12:228, 16:304].reshape(images.shape[0], images.shape[1], 4, 3, 216, 288)

    def forward(self, images: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if state.ndim != 3 or state.shape[:2] != images.shape[:2] or state.shape[-1] != 8:
            raise ValueError(f"state must be [B,T,8], got {tuple(state.shape)}")
        # _official_encoder's CropRandomizer accepts [B*T,C,H,W] values.  It
        # is intentionally fed the same [-1,1] representation as training.
        values = _preprocess_rgb(images)
        batch, history = values.shape[:2]
        frames = values.reshape(batch * history, 4, 3, 240, 320)
        state = state.float().reshape(batch * history, 8)
        obs = OrderedDict(
            (name, frames[:, index])
            for index, name in enumerate(("top", "side", "wrist_left", "wrist_right"))
        )
        obs["state"] = state
        # Deployment is always eval: the replacement randomizer therefore
        # uses the center crop.  Explicitly avoid accidental random crops if a
        # caller toggles this module to train mode for a smoke test.
        previous = [value.training for value in self.encoder.obs_randomizers.values() if value is not None]
        randomizers = [value for value in self.encoder.obs_randomizers.values() if value is not None]
        for value in randomizers:
            value.eval()
        try:
            result = self.encoder(obs)
        finally:
            for value, mode in zip(randomizers, previous, strict=True):
                value.train(mode)
        return result.reshape(batch, history, self.output_dim)


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
