# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor.pipeline import ObservationProcessorStep, ProcessorStepRegistry

from .feature_adapter import DEPTH_KEYS, RGB_KEYS, XENSE_KEYS


def _channels_first(value: np.ndarray | torch.Tensor, *, normalize_rgb: bool = False):
    is_tensor = isinstance(value, torch.Tensor)
    ndim = value.ndim
    if ndim == 3:  # HWC -> BCHW
        axes = (2, 0, 1)
        result = (
            value.permute(*axes).unsqueeze(0) if is_tensor else np.expand_dims(np.transpose(value, axes), 0)
        )
    elif ndim == 4:  # BHWC -> BCHW
        axes = (0, 3, 1, 2)
        result = value.permute(*axes) if is_tensor else np.transpose(value, axes)
    elif ndim == 5:  # BTHWC -> BTCHW
        axes = (0, 1, 4, 2, 3)
        result = value.permute(*axes) if is_tensor else np.transpose(value, axes)
    else:
        raise ValueError(f"expected HWC, BHWC, or BTHWC tensor, got shape {tuple(value.shape)}")
    if normalize_rgb:
        result = result.to(torch.float32) / 255.0 if is_tensor else result.astype(np.float32) / 255.0
    elif is_tensor:
        result = result.to(torch.float32)
    else:
        result = result.astype(np.float32, copy=False)
    return result.contiguous() if is_tensor else np.ascontiguousarray(result)


def prepare_fr3_array_for_policy(key: str, value: Any):
    """Apply only FR3's deterministic channel/dtype layout conversion."""

    raw_key = key.removeprefix("observation.").removeprefix("images.")
    if raw_key in XENSE_KEYS:
        return _channels_first(value)
    if raw_key in RGB_KEYS:
        return _channels_first(value, normalize_rgb=True)
    if raw_key in DEPTH_KEYS:
        return _channels_first(value)
    return value


@dataclass
@ProcessorStepRegistry.register(name="fr3_policy_observation_processor")
class FR3PolicyObservationProcessorStep(ObservationProcessorStep):
    """FR3-only policy layout conversion; it performs no I/O, alignment, or encoding."""

    def observation(self, observation):
        return {key: prepare_fr3_array_for_policy(key, value) for key, value in observation.items()}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        transformed = {kind: dict(bucket) for kind, bucket in features.items()}
        bucket = transformed.get(PipelineFeatureType.OBSERVATION, {})
        for key in tuple(bucket):
            raw_key = key.removeprefix("observation.").removeprefix("images.")
            if raw_key in XENSE_KEYS:
                bucket[key] = PolicyFeature(type=FeatureType.STATE, shape=(3, 35, 20))
            elif raw_key in RGB_KEYS:
                bucket[key] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640))
            elif raw_key in DEPTH_KEYS:
                bucket[key] = PolicyFeature(type=FeatureType.VISUAL, shape=(1, 480, 640))
        return transformed
