"""Inference adapters for the ACMT-ACT policies.

The original ``acmt_act`` policy retains its historical 16/8 queue.  The v2
policy uses the native ACT contract and therefore needs the ordinary
synchronous engine: ``select_action`` is called every tick and owns the
online temporal ensemble.
"""

from __future__ import annotations

import logging
import math
import time

import torch

from .acmt_dp import ACMTDPInferenceEngine, TimedAction
from .sync import SyncInferenceEngine

logger = logging.getLogger(__name__)

_ACMT_ACT_V3_SCHEMA = ("acmt_act.v3", 3)
_DEFAULT_MAX_JOINT_STEP_DEGREES = 10.0
_JOINT_POSITION_KEYS = tuple(f"fr3_joint{index}.pos" for index in range(1, 8))


class ACMTACTInferenceEngine(ACMTDPInferenceEngine):
    """Run legacy ``acmt_act`` with its causal 16-predict/8-execute queue."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = getattr(self._policy, "config", None)
        schema = (
            getattr(config, "checkpoint_schema", None),
            getattr(config, "checkpoint_schema_version", None),
        )
        self._joint_step_limiter_enabled = schema == _ACMT_ACT_V3_SCHEMA
        self._last_accepted_joint_target: torch.Tensor | None = None
        self._last_accepted_action: torch.Tensor | None = None
        self._active_plan_id: int | None = None
        self._last_slew_log = 0.0
        if not self._joint_step_limiter_enabled:
            self._joint_action_indices: tuple[int, ...] = ()
            self._max_joint_step_rad = 0.0
            return

        self._joint_action_indices = tuple(
            self._ordered_action_keys.index(key) for key in _JOINT_POSITION_KEYS
        )
        max_step_degrees = float(
            getattr(config, "max_joint_step_degrees", _DEFAULT_MAX_JOINT_STEP_DEGREES)
        )
        if not math.isfinite(max_step_degrees) or not (0.0 < max_step_degrees <= 10.0):
            raise ValueError("ACMT-ACT max_joint_step_degrees must be finite and in (0, 10]")
        self._max_joint_step_rad = math.radians(max_step_degrees)
        self._max_joint_step_degrees = max_step_degrees

    def reset(self) -> None:
        super().reset()
        with self._lock:
            self._last_accepted_joint_target = None
            self._last_accepted_action = None
            self._active_plan_id = None
            self._last_slew_log = 0.0

    def _build_boundary_bridge(
        self,
        timed: TimedAction,
        reference: torch.Tensor,
        target: torch.Tensor,
        old_plan_id: int,
    ) -> torch.Tensor | None:
        """Create seam actions before a new plan's first policy action."""
        joint_indices = torch.tensor(self._joint_action_indices, dtype=torch.long)
        requested = target.index_select(0, joint_indices)
        reference_joints = reference.index_select(0, joint_indices)
        delta = requested - reference_joints
        max_delta = float(delta.abs().max().item())
        segments = max(1, math.ceil(max_delta / self._max_joint_step_rad - 1e-9))
        inserted_count = segments - 1
        if inserted_count <= 0:
            return None

        gripper_index = self._ordered_action_keys.index("gripper.pos")
        bridge = []
        for step in range(1, segments):
            fraction = step / segments
            value = reference.clone()
            value.index_copy_(0, joint_indices, reference_joints + fraction * delta)
            value[gripper_index] = reference[gripper_index]
            bridge.append(value)
        bridge_tensor = torch.stack(bridge)
        remaining = torch.cat((bridge_tensor[1:], target.unsqueeze(0)), dim=0)
        remaining_indices = list(range(-remaining.shape[0] + 1, 1))
        self._queue.insert_after_popped(timed, remaining, remaining_indices)
        affected = [
            _JOINT_POSITION_KEYS[index]
            for index, value in enumerate(delta.abs().tolist())
            if value > self._max_joint_step_rad
        ]
        logger.info(
            "ACMT-ACT v3 boundary interpolation: old_plan_id=%s new_plan_id=%s "
            "max_delta_deg=%.3f limit_deg=%.3f inserted_steps=%d affected_joints=%s",
            old_plan_id,
            timed.plan_id,
            max_delta * 180.0 / math.pi,
            self._max_joint_step_degrees,
            inserted_count,
            affected,
        )
        return bridge_tensor[0]

    def _prepare_timed_action(self, timed: TimedAction) -> torch.Tensor:
        """Bridge a large jump when the first action of a new plan is due."""
        if not self._joint_step_limiter_enabled:
            return super()._prepare_timed_action(timed)
        if self._active_plan_id is None:
            self._active_plan_id = timed.plan_id
            return timed.value
        if timed.plan_id == self._active_plan_id:
            return timed.value
        previous_plan_id = self._active_plan_id
        self._active_plan_id = timed.plan_id
        if self._last_accepted_action is None:
            raise RuntimeError(
                "ACMT-ACT v3 boundary interpolation requires a previously accepted action"
            )
        reference = self._last_accepted_action.to(dtype=timed.value.dtype)
        target = timed.value.clone()
        bridge = self._build_boundary_bridge(timed, reference, target, previous_plan_id)
        if bridge is None:
            return target
        return bridge

    def _anchor_joint_target(self) -> torch.Tensor | None:
        anchor_state = self._current_action_anchor_state
        if anchor_state is None:
            return None
        values = anchor_state.detach().reshape(-1)
        if values.numel() < len(_JOINT_POSITION_KEYS):
            return None
        target = values[: len(_JOINT_POSITION_KEYS)].to(dtype=torch.float32)
        if not torch.isfinite(target).all():
            return None
        return target.clone()

    def _limit_joint_step(self, action: torch.Tensor) -> torch.Tensor:
        if not self._joint_step_limiter_enabled:
            return action
        if action.ndim != 1 or action.numel() != len(self._ordered_action_keys):
            raise ValueError(
                "ACMT-ACT v3 joint limiter expects a flat action matching ordered_action_keys"
            )
        reference = self._last_accepted_joint_target
        if reference is None:
            reference = self._anchor_joint_target()
        if reference is None:
            raise RuntimeError("ACMT-ACT v3 joint limiter requires a finite seven-joint observation anchor")

        joint_indices = torch.tensor(self._joint_action_indices, dtype=torch.long, device=action.device)
        requested = action.index_select(0, joint_indices)
        reference = reference.to(device=action.device, dtype=action.dtype)
        delta = requested - reference
        limited = reference + torch.clamp(delta, -self._max_joint_step_rad, self._max_joint_step_rad)
        result = action.clone()
        result.index_copy_(0, joint_indices, limited)

        changed = torch.abs(limited - requested) > 1e-9
        if bool(changed.any()):
            # Use the monotonic clock only for log rate limiting; this branch
            # deliberately does not affect action timing or queue ownership.
            now = time.monotonic()
            if self._last_slew_log == 0.0 or now - self._last_slew_log >= 1.0:
                clipped_indices = changed.nonzero(as_tuple=False).reshape(-1).tolist()
                clipped_names = [_JOINT_POSITION_KEYS[index] for index in clipped_indices]
                logger.warning(
                    "ACMT-ACT v3 joint step limited: max_delta_deg=%.3f limit_deg=%.3f "
                    "clipped_joints=%s",
                    float(delta.abs().max().item() * 180.0 / math.pi),
                    self._max_joint_step_degrees,
                    clipped_names,
                )
                self._last_slew_log = now
        return result

    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        action = super().get_action(obs_frame)
        if action is None:
            return None
        with self._lock:
            return self._limit_joint_step(action)

    def notify_action_executed(self, action: torch.Tensor, observation: dict | None = None) -> None:
        super().notify_action_executed(action, observation)
        if not self._joint_step_limiter_enabled:
            return
        if action.ndim == 2:
            action = action.squeeze(0)
        with self._lock:
            if action.ndim != 1 or action.numel() != len(self._ordered_action_keys):
                raise ValueError("executed ACMT-ACT v3 action must match ordered_action_keys")
            joint_indices = torch.tensor(self._joint_action_indices, dtype=torch.long, device=action.device)
            accepted = action.detach().to(dtype=torch.float32).index_select(0, joint_indices)
            if not torch.isfinite(accepted).all():
                raise ValueError("executed ACMT-ACT v3 joint action contains non-finite values")
            self._last_accepted_action = action.detach().to(dtype=torch.float32).cpu().clone()
            self._last_accepted_joint_target = accepted.cpu().clone()


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
