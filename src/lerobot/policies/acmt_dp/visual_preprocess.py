"""Deployment preprocessing used by ACMT-DP TactiGen and Native-DP v4.

The policy was trained from raw 480x640 wrist RGB-D frames.  The frozen visual
encoder sees the center 480x480 crop resized to 128x128; RGB is antialiased
bilinear and depth is nearest-neighbour so millimetre values are preserved.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812

RAW_HEIGHT = 480
RAW_WIDTH = 640
CROP_LEFT = 80
CROP_TOP = 0
CROP_SIZE = 480
MODEL_SIZE = 128
NATIVE_RESIZE = 256
NATIVE_SIZE = 224


def _channels_first(value: torch.Tensor, channels: int, key: str) -> torch.Tensor:
    if value.ndim < 3:
        raise ValueError(f"{key} must include channel/height/width dimensions, got {tuple(value.shape)}")
    if value.shape[-3] == channels:
        return value
    if value.shape[-1] == channels:
        return value.movedim(-1, -3).contiguous()
    raise ValueError(f"{key} has no {channels}-channel axis: {tuple(value.shape)}")


def center_crop_480_resize_128(rgb: torch.Tensor, depth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rgb = _channels_first(rgb, 3, "rgb")
    depth = _channels_first(depth, 1, "depth")
    if tuple(rgb.shape[-3:]) != (3, RAW_HEIGHT, RAW_WIDTH):
        raise ValueError(f"raw RGB must end in [3,480,640], got {tuple(rgb.shape[-3:])}")
    if tuple(depth.shape[-3:]) != (1, RAW_HEIGHT, RAW_WIDTH):
        raise ValueError(f"raw depth must end in [1,480,640], got {tuple(depth.shape[-3:])}")

    # CUDA has no uint16 comparison kernels in the frozen DFormer validity
    # mask. Promote only at the CUDA boundary; values remain exact millimetres.
    if depth.device.type == "cuda" and depth.dtype == torch.uint16:
        depth = depth.to(dtype=torch.int32)
    original_depth_dtype = depth.dtype
    rgb_value = rgb.reshape(-1, 3, RAW_HEIGHT, RAW_WIDTH).float()
    depth_value = depth.reshape(-1, 1, RAW_HEIGHT, RAW_WIDTH).float()
    rgb_value = rgb_value[..., CROP_TOP : CROP_TOP + CROP_SIZE, CROP_LEFT : CROP_LEFT + CROP_SIZE]
    depth_value = depth_value[..., CROP_TOP : CROP_TOP + CROP_SIZE, CROP_LEFT : CROP_LEFT + CROP_SIZE]
    rgb_value = F.interpolate(
        rgb_value, size=(MODEL_SIZE, MODEL_SIZE), mode="bilinear", align_corners=False, antialias=True
    )
    depth_value = F.interpolate(depth_value, size=(MODEL_SIZE, MODEL_SIZE), mode="nearest")
    if original_depth_dtype.is_floating_point:
        processed_depth = depth_value.to(original_depth_dtype)
    else:
        processed_depth = depth_value.round().to(original_depth_dtype)
    leading = tuple(rgb.shape[:-3])
    return (
        rgb_value.reshape(*leading, 3, MODEL_SIZE, MODEL_SIZE),
        processed_depth.reshape(*leading, 1, MODEL_SIZE, MODEL_SIZE),
    )


def prepare_for_frozen_encoder(rgb: torch.Tensor, depth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rgb = _channels_first(rgb, 3, "rgb")
    depth = _channels_first(depth, 1, "depth")
    if tuple(rgb.shape[-3:]) == (3, RAW_HEIGHT, RAW_WIDTH):
        return center_crop_480_resize_128(rgb, depth)
    if tuple(rgb.shape[-3:]) == (3, MODEL_SIZE, MODEL_SIZE) and tuple(depth.shape[-3:]) == (
        1,
        MODEL_SIZE,
        MODEL_SIZE,
    ):
        return rgb, depth
    raise ValueError(
        "RGB-D must be raw [3,480,640] or processed [3,128,128], got "
        f"{tuple(rgb.shape[-3:])} and {tuple(depth.shape[-3:])}"
    )


def native_v4_rgb_preprocess(images: torch.Tensor) -> torch.Tensor:
    """Apply the v4 raw RGB transform and return float ``[N,4,3,224,224]``.

    The model calls this helper so offline checks can exercise the exact
    resize/crop/round path without constructing a policy.
    """

    if images.ndim != 5 or images.shape[1:3] != (4, 3):
        raise ValueError(f"images must be [N,4,3,H,W], got {tuple(images.shape)}")
    values = images.reshape(-1, 3, images.shape[-2], images.shape[-1]).float()
    if tuple(values.shape[-2:]) == (RAW_HEIGHT, RAW_WIDTH):
        if torch.is_floating_point(images) and values.numel() and float(values.detach().amax()) <= 1.0:
            values = values * 255.0
        values = F.interpolate(
            values, size=(NATIVE_RESIZE, NATIVE_RESIZE), mode="bilinear", align_corners=False, antialias=True
        )
        values = values[..., 16:240, 16:240].clamp(0.0, 255.0).round() / 255.0
    elif tuple(values.shape[-2:]) == (NATIVE_SIZE, NATIVE_SIZE):
        if not torch.is_floating_point(images) or (values.numel() and float(values.detach().amax()) > 1.0):
            values = values / 255.0
    else:
        raise ValueError("native v4 RGB must be raw 480x640 or 224x224")
    mean = values.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = values.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    return ((values.clamp(0.0, 1.0) - mean) / std).reshape(images.shape[0], 4, 3, NATIVE_SIZE, NATIVE_SIZE)


__all__ = [
    "CROP_LEFT",
    "CROP_SIZE",
    "CROP_TOP",
    "MODEL_SIZE",
    "NATIVE_RESIZE",
    "NATIVE_SIZE",
    "RAW_HEIGHT",
    "RAW_WIDTH",
    "center_crop_480_resize_128",
    "prepare_for_frozen_encoder",
    "native_v4_rgb_preprocess",
]
