# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ACMT-DP gripper action adaptation for the existing FR3 wire contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor import PolicyAction
from lerobot.processor.pipeline import PolicyActionProcessorStep, ProcessorStepRegistry


# Robotiq reports the usable deployed gPO interval as 3..255. ACMT-DP
# training actions use opening percentage semantics: 0.0 is closed and 1.0
# is open. FR3's existing policy converter consumes normalized gPO instead.
GRIPPER_GPO_MIN = 3.0
GRIPPER_GPO_MAX = 255.0


def policy_gripper_to_fr3_pos(value: Tensor) -> Tensor:
    """Map an ACMT-DP opening action to FR3's normalized gPO direction.

    The input is clamped to the model's declared ``[0, 1]`` action interval.
    The returned value is normalized so the unchanged FR3 converter emits
    ``gPO=255`` at input 0 and ``gPO=3`` at input 1.
    """

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"gripper action must be a torch.Tensor, got {type(value).__name__}")
    clipped = value.clamp(0.0, 1.0)
    gpo = GRIPPER_GPO_MAX - (GRIPPER_GPO_MAX - GRIPPER_GPO_MIN) * clipped
    return gpo / 255.0


def fr3_pos_to_policy_gripper(value: Tensor) -> Tensor:
    """Invert :func:`policy_gripper_to_fr3_pos` for a policy-space consumer."""

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"gripper action must be a torch.Tensor, got {type(value).__name__}")
    normalized_gpo = value.clamp(GRIPPER_GPO_MIN / 255.0, GRIPPER_GPO_MAX / 255.0)
    return (GRIPPER_GPO_MAX - normalized_gpo * 255.0) / (GRIPPER_GPO_MAX - GRIPPER_GPO_MIN)


@dataclass
@ProcessorStepRegistry.register(name="acmt_dp_gripper_gpo_processor")
class ACMTDPGripperGPOProcessorStep(PolicyActionProcessorStep):
    """Adapt the ACMT-DP opening scalar to the existing normalized gPO ABI."""

    action_index: int = 7

    def __post_init__(self) -> None:
        if self.action_index < 0:
            raise ValueError("ACMT-DP gripper action_index must be non-negative")

    def action(self, action: PolicyAction) -> PolicyAction:
        if action.ndim == 0 or action.shape[-1] <= self.action_index:
            raise ValueError(
                "ACMT-DP policy action must have a trailing dimension containing the gripper at index 7"
            )
        result = action.clone()
        result[..., self.action_index] = policy_gripper_to_fr3_pos(result[..., self.action_index])
        return result

    def get_config(self) -> dict[str, Any]:
        return {"action_index": self.action_index}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


__all__ = [
    "ACMTDPGripperGPOProcessorStep",
    "GRIPPER_GPO_MIN",
    "GRIPPER_GPO_MAX",
    "fr3_pos_to_policy_gripper",
    "policy_gripper_to_fr3_pos",
]
