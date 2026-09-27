"""Inference adapters for the ACMT-ACT policies.

The original ``acmt_act`` policy retains its historical 16/8 queue.  The v2
policy uses the native ACT contract and therefore needs the ordinary
synchronous engine: ``select_action`` is called every tick and owns the
online temporal ensemble.
"""

import logging

import torch

from lerobot.utils.constants import ACTION

from .acmt_dp import ACMTDPInferenceEngine, JOINT_POSITION_KEYS
from .sync import SyncInferenceEngine

logger = logging.getLogger(__name__)

# Observed absolute joint-position extrema from the gear training data (rad).
GEAR_JOINT_LOWER = (
    -0.054157, -0.047530, -0.278071, -2.433730, -0.266043, 1.734929, -1.189159,
)
GEAR_JOINT_UPPER = (
    0.122640, 0.541469, 0.444214, -1.391000, 0.221219, 2.523725, 0.942100,
)


class ACMTACTInferenceEngine(ACMTDPInferenceEngine):
    """Run legacy ``acmt_act`` with its causal 16-predict/8-execute queue."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._gear_joint_indices: tuple[int, ...] | None = None
        config = self._policy.config
        if (
            getattr(self._policy, "name", None) != "acmt_act"
            or getattr(config, "checkpoint_schema", None) != "acmt_act.v3"
            or getattr(config, "checkpoint_schema_version", None) != 3
            or getattr(config, "task_variant", None) != "gear"
        ):
            return
        if not self._plan_postprocess or self._relative_action_step is None:
            raise ValueError("gear ACMT-ACT v3 requires absolute-action plan postprocessing")
        names = self._dataset_features.get(ACTION, {}).get("names")
        expected = (*JOINT_POSITION_KEYS, "gripper.pos")
        if not isinstance(names, (list, tuple)) or len(names) != 8 or set(names) != set(expected):
            raise ValueError("gear ACMT-ACT v3 requires exactly seven named joints and gripper.pos")
        if list(names) != list(getattr(self._relative_action_step, "action_names", None) or []):
            raise ValueError("gear ACMT-ACT v3 action names do not match relative-action processing")
        self._gear_joint_indices = tuple(names.index(name) for name in JOINT_POSITION_KEYS)

    def _postprocess_plan(self, action: torch.Tensor, anchor_state: torch.Tensor | None) -> torch.Tensor:
        absolute = super()._postprocess_plan(action, anchor_state)
        indices = self._gear_joint_indices
        if indices is None:
            return absolute
        if not torch.isfinite(absolute).all():
            raise ValueError("gear ACMT-ACT v3 plan contains non-finite absolute actions")
        lower = absolute.new_tensor(GEAR_JOINT_LOWER)
        upper = absolute.new_tensor(GEAR_JOINT_UPPER)
        joints = absolute[..., list(indices)]
        clipped = torch.minimum(torch.maximum(joints, lower), upper)
        if not (clipped != joints).any():
            return absolute
        result = absolute.clone()
        result[..., list(indices)] = clipped
        # Always clamp even a one-ULP overshoot; reserve warnings for changes
        # larger than float32 round-trip noise in residual restoration.
        material = (clipped - joints).abs() > 1e-6
        if material.any():
            affected = [
                name for index, name in enumerate(JOINT_POSITION_KEYS) if material[..., index].any()
            ]
            logger.warning(
                "gear ACMT-ACT v3 joint bounds clipped plan: values=%d joints=%s",
                int(material.sum()),
                ",".join(affected),
            )
        return result


class ACMTACTV2InferenceEngine(SyncInferenceEngine):
    """Run native ACMT-ACTv2 with one fresh 100-step plan per tick."""

    def __init__(self, *args, **kwargs):
        policy = kwargs.get("policy")
        if policy is None and args:
            policy = args[0]
        config = getattr(policy, "config", None)
        if config is None:
            raise ValueError("ACMT-ACTv2 inference requires a policy configuration")
        if (
            getattr(config, "checkpoint_schema", None),
            getattr(config, "checkpoint_schema_version", None),
        ) != ("acmt_actv2.dinov2_spatial.v1", 3):
            raise ValueError("ACMT-ACTv2 inference requires DINOv2 spatial schema version 3")
        if (
            getattr(config, "chunk_size", None),
            getattr(config, "n_action_steps", None),
            getattr(config, "action_execution_horizon", None),
        ) != (100, 1, 1):
            raise ValueError("ACMT-ACTv2 inference requires chunk_size=100 and one-step execution")
        super().__init__(*args, **kwargs)


__all__ = ["ACMTACTInferenceEngine", "ACMTACTV2InferenceEngine"]
