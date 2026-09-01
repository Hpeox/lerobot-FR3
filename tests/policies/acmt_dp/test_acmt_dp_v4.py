from __future__ import annotations

import torch

from lerobot.policies.acmt_dp.modeling_native_v4 import FrameTactileEncoder, NativeVisionEncoder
from lerobot.policies.acmt_dp.visual_preprocess import native_v4_rgb_preprocess


def test_native_v4_rgb_preprocess_is_center_resize_and_imagenet_normalized() -> None:
    value = torch.zeros(1, 4, 3, 480, 640, dtype=torch.uint8)
    value[:, :, 0] = 255
    processed = native_v4_rgb_preprocess(value)
    assert processed.shape == (1, 4, 3, 224, 224)
    # Red channel of a zero/one image after ImageNet normalization.
    torch.testing.assert_close(processed[:, :, 0].mean(), torch.tensor((1.0 - 0.485) / 0.229))


def test_native_v4_resnet_and_tactile_shapes() -> None:
    vision = NativeVisionEncoder(weights=None)
    rgb = torch.zeros(2, 4, 3, 224, 224)
    assert vision(rgb).shape == (2, 4, 512)
    tactile = FrameTactileEncoder((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    force = torch.zeros(2, 4, 2, 35, 20, 3)
    assert tactile(force).shape == (2, 4, 160)
