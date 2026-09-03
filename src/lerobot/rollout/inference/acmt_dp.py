"""Native-DP 16/8 rolling inference adapter for ACMT-DP v4/v5."""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from copy import copy
from dataclasses import dataclass
from typing import Any

import torch

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import make_robot_action, prepare_observation_for_inference
from lerobot.processor import PolicyProcessorPipeline

from .base import InferenceEngine

logger = logging.getLogger(__name__)

ACTION_DIM = 8
PREDICTION_HORIZON = 16
EXECUTION_HORIZON = 8
CONTROL_HZ = 30.0
DISPATCH_TOLERANCE_S = 0.005
JOINT_POSITION_KEYS = tuple(f"fr3_joint{index}.pos" for index in range(1, 8))


def _clone_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_tree(item) for item in value]
    return value


@dataclass(frozen=True)
class ActionPlan:
    plan_id: int
    actions: torch.Tensor
    start_time: float | None = None
    period_s: float = 1.0 / CONTROL_HZ

    def __post_init__(self) -> None:
        value = self.actions.detach().float()
        if tuple(value.shape) != (PREDICTION_HORIZON, ACTION_DIM):
            raise ValueError(f"ACMT-DP action plan must be [16,8], got {tuple(value.shape)}")
        if not torch.isfinite(value).all():
            raise ValueError("ACMT-DP action plan contains non-finite values")
        if self.start_time is None:
            object.__setattr__(self, "start_time", time.monotonic())
        if self.period_s <= 0:
            raise ValueError("ACMT-DP action plan period must be positive")
        object.__setattr__(self, "actions", value.clone())

    @property
    def execution(self) -> torch.Tensor:
        return self.actions[:EXECUTION_HORIZON]

    @property
    def reserve(self) -> torch.Tensor:
        return self.actions[EXECUTION_HORIZON:]

    def timed_actions(self) -> list[TimedAction]:
        return [
            TimedAction(self.plan_id, index, self.start_time + index * self.period_s, row)
            for index, row in enumerate(self.actions)
        ]


@dataclass(frozen=True)
class TimedAction:
    plan_id: int
    action_index: int
    target_time: float
    value: torch.Tensor

    def __post_init__(self) -> None:
        value = self.value.detach().float().reshape(-1)
        if tuple(value.shape) != (ACTION_DIM,):
            raise ValueError(f"TimedAction value must be [8], got {tuple(value.shape)}")
        if not torch.isfinite(value).all():
            raise ValueError("TimedAction contains non-finite values")
        object.__setattr__(self, "value", value.clone())


class TimedActionQueue:
    """Thread-safe action queue with atomic future-plan replacement."""

    def __init__(self, control_hz: float = CONTROL_HZ) -> None:
        if control_hz != CONTROL_HZ:
            raise ValueError("ACMT-DP requires control_hz=30")
        self.period_s = 1.0 / control_hz
        self._items: deque[TimedAction] = deque()
        self._lock = threading.RLock()
        self._expired = 0
        self._last_plan_id = -1
        self._last_target_time: float | None = None

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._expired = 0
            self._last_target_time = None

    def install(self, plan: ActionPlan) -> None:
        if abs(plan.period_s - self.period_s) > 1e-9:
            raise ValueError("ACMT-DP action plan period does not match control_hz")
        with self._lock:
            self._items.clear()
            self._items.extend(plan.timed_actions())
            self._last_plan_id = plan.plan_id
            self._last_target_time = plan.start_time + (PREDICTION_HORIZON - 1) * self.period_s

    install_plan = install

    def pop(self) -> torch.Tensor | None:
        """Pop the next item without applying a wall-clock schedule.

        This small compatibility helper is useful for offline queue tests. The
        online engine deliberately uses :meth:`pop_due` so a delayed planner
        can never replay an expired command.
        """
        with self._lock:
            return self._items.popleft().value if self._items else None

    def pop_due(self, now: float | None = None) -> TimedAction | None:
        now = time.monotonic() if now is None else float(now)
        with self._lock:
            while self._items and self._items[0].target_time < now - DISPATCH_TOLERANCE_S:
                self._items.popleft()
                self._expired += 1
            if not self._items or self._items[0].target_time > now + DISPATCH_TOLERANCE_S:
                return None
            return self._items.popleft()

    def replace_future(self, actions: torch.Tensor, start_time: float, plan_id: int) -> int:
        plan = ActionPlan(plan_id, actions, start_time=start_time, period_s=self.period_s)
        with self._lock:
            kept = [item for item in self._items if item.target_time < start_time]
            removed = len(self._items) - len(kept)
            self._items.clear()
            self._items.extend(kept)
            self._items.extend(plan.timed_actions())
            self._last_plan_id = plan_id
            self._last_target_time = plan.start_time + (PREDICTION_HORIZON - 1) * self.period_s
            return removed

    def replace_future_at_next_deadline(
        self, actions: torch.Tensor, now: float, plan_id: int
    ) -> tuple[int, float]:
        """Replace future actions at the next valid 30 Hz deadline.

        The currently due action remains owned by the old plan.  A completed
        background plan therefore cannot overwrite an action that the sender
        is about to dispatch, and its first action is never scheduled at an
        arbitrary inference-completion timestamp.
        """
        now = float(now)
        with self._lock:
            while self._items and self._items[0].target_time < now - DISPATCH_TOLERANCE_S:
                self._items.popleft()
                self._expired += 1

            if self._items:
                first = self._items[0].target_time
                if first <= now + DISPATCH_TOLERANCE_S:
                    start_time = first + self.period_s
                else:
                    start_time = first
            elif self._last_target_time is not None:
                start_time = max(self._last_target_time + self.period_s, now)
            else:
                start_time = now

            plan = ActionPlan(plan_id, actions, start_time=start_time, period_s=self.period_s)
            kept = [item for item in self._items if item.target_time < start_time]
            removed = len(self._items) - len(kept)
            self._items.clear()
            self._items.extend(kept)
            self._items.extend(plan.timed_actions())
            self._last_plan_id = plan_id
            self._last_target_time = plan.start_time + (PREDICTION_HORIZON - 1) * self.period_s
            return removed, start_time

    def snapshot(self) -> tuple[torch.Tensor, ...]:
        with self._lock:
            return tuple(item.value.clone() for item in self._items)

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def remaining(self) -> int:
        return len(self)

    @property
    def expired(self) -> int:
        with self._lock:
            return self._expired

    @property
    def last_plan_id(self) -> int:
        with self._lock:
            return self._last_plan_id


class ACMTDPInferenceEngine(InferenceEngine):
    """Synchronous CLI backend with asynchronous 16-step plan preparation.

    The control loop still selects ``--inference.type=sync``.  Diffusion runs
    on a single planner worker while the sender consumes the current plan.  A
    new plan is submitted only after the boundary action has been confirmed by
    ``send_action``; this makes the TactiGen input the actual command rather
    than an unexecuted prediction.
    """

    def __init__(
        self,
        policy: PreTrainedPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        dataset_features: dict,
        ordered_action_keys: list[str],
        task: str,
        device: str | None,
        robot_type: str,
    ) -> None:
        del robot_type
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._dataset_features = dataset_features
        self._ordered_action_keys = ordered_action_keys
        self._task = task
        self._device = torch.device(device or "cpu")
        self._queue = TimedActionQueue()
        self._planner = ThreadPoolExecutor(max_workers=1, thread_name_prefix="acmt-dp-planner")
        self._tactile_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="acmt-tactile")
        self._future: Future[ActionPlan] | None = None
        self._future_generation: int | None = None
        self._tactile_future: Future[None] | None = None
        self._next_plan_id = 0
        self._boundary_in_flight = False
        self._current_action_window: dict | None = None
        self._generation = 0
        self._last_hold_log = 0.0
        self._lock = threading.RLock()
        # History mutation and model forward have different ownership.  The
        # control thread only holds _policy_state_lock while publishing an
        # observation/snapshot; the planner holds _plan_lock only for model
        # forward.  Keeping these locks separate is what allows observation
        # capture to overlap diffusion.
        self._policy_state_lock = threading.RLock()
        self._plan_lock = threading.RLock()
        self._plan_started_at: float | None = None
        self._failure: BaseException | None = None
        self._stopped = False
        self._first_action_diagnostic_emitted = False

        config = getattr(policy, "config", None)
        if config is None or getattr(config, "control_hz", 30.0) != CONTROL_HZ:
            raise ValueError("ACMT-DP/ACMT-ACT rollout requires control_hz=30")
        if getattr(config, "action_execution_horizon", 8) != EXECUTION_HORIZON:
            raise ValueError("ACMT-DP/ACMT-ACT rollout requires action_execution_horizon=8")
        if getattr(config, "tactile_history", 4) != 4:
            raise ValueError("ACMT-DP/ACMT-ACT rollout requires tactile_history=4")
        if (
            getattr(config, "pred_horizon", 16) != PREDICTION_HORIZON
            or getattr(config, "action_dim", 8) != ACTION_DIM
        ):
            raise ValueError("ACMT-DP/ACMT-ACT rollout requires pred_horizon=16 and action_dim=8")
        policy_name = getattr(policy, "name", None)
        schema_version = getattr(config, "checkpoint_schema_version", 4)
        if policy_name == "acmt_act":
            if getattr(config, "checkpoint_schema", None) != "acmt_act.v2" or schema_version != 2:
                raise ValueError("ACMT-ACT rollout requires checkpoint schema acmt_act.v2")
        elif schema_version not in (3, 4, 5):
            raise ValueError("ACMT-DP rollout requires checkpoint_schema_version=3, 4 or 5")
        if getattr(config, "tactile_source", None) in {"tactigen", "substitution"}:
            logger.info("Causal tactile generation uses the sync 16/8 runtime; RTC is unsupported")

    def start(self) -> None:
        self._stopped = False

    def stop(self) -> None:
        with self._lock:
            self._stopped = True
            self._generation += 1
            future = self._future
            if future is not None:
                future.cancel()
            self._future = None
            self._future_generation = None
            if self._tactile_future is not None:
                self._tactile_future.cancel()
            self._tactile_future = None
            self._planner.shutdown(wait=False, cancel_futures=True)
            self._tactile_worker.shutdown(wait=False, cancel_futures=True)

    def reset(self) -> None:
        with self._lock:
            if self._stopped:
                self._planner = ThreadPoolExecutor(max_workers=1, thread_name_prefix="acmt-dp-planner")
                self._tactile_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="acmt-tactile")
                self._stopped = False
            self._generation += 1
            if self._future is not None:
                self._future.cancel()
            self._future = None
            self._future_generation = None
            if self._tactile_future is not None:
                self._tactile_future.cancel()
            self._tactile_future = None
            self._current_action_window = None
            self._last_hold_log = 0.0
            self._plan_started_at = None
            self._failure = None
            self._first_action_diagnostic_emitted = False
            self._queue.clear()
            self._boundary_in_flight = False
            self._next_plan_id = 0
            with self._plan_lock:
                with self._policy_state_lock:
                    self._policy.reset()
            self._preprocessor.reset()
            self._postprocessor.reset()

    @staticmethod
    def _finite_joint_vector(source: dict[str, Any] | None) -> tuple[float, ...] | None:
        if source is None:
            return None
        try:
            values = tuple(float(source[key]) for key in JOINT_POSITION_KEYS)
        except (KeyError, TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in values):
            return None
        return values

    @staticmethod
    def _max_abs_delta(
        left: tuple[float, ...] | None,
        right: tuple[float, ...] | None,
    ) -> float | None:
        if left is None or right is None:
            return None
        return max(abs(first - second) for first, second in zip(left, right, strict=True))

    def record_first_action_diagnostic(
        self,
        observation: dict[str, Any],
        planned_action: dict[str, Any],
        sent_action: dict[str, Any],
    ) -> None:
        """Log one accepted rollout action for diagnosing target jumps.

        This is intentionally an ACMT-DP-only observability hook.  It does not
        clamp, interpolate, or otherwise modify the command that was accepted
        by the robot transport.
        """
        with self._lock:
            if self._first_action_diagnostic_emitted:
                return
            self._first_action_diagnostic_emitted = True

            current_q = self._finite_joint_vector(observation)
            planned_q = self._finite_joint_vector(planned_action)
            sent_q = self._finite_joint_vector(sent_action)
            logger.info(
                "ACMT-DP first action diagnostic: current_q=%s planned_q=%s sent_q=%s "
                "max_abs_delta=%s max_abs_processor_delta=%s",
                current_q,
                planned_q,
                sent_q,
                "n/a" if (delta := self._max_abs_delta(current_q, sent_q)) is None else f"{delta:.6f}",
                "n/a"
                if (processor_delta := self._max_abs_delta(planned_q, sent_q)) is None
                else f"{processor_delta:.6f}",
            )

    def _prepare(self, obs_frame: dict) -> dict:
        observation = copy(obs_frame)
        observation = prepare_observation_for_inference(
            observation, self._device, self._task, getattr(self._policy, "robot_type", "fr3")
        )
        return self._preprocessor(observation)

    def _plan_now(self, window: dict, plan_id: int, generation: int | None = None) -> ActionPlan:
        with self._plan_lock, torch.inference_mode():
            if generation is not None and generation != self._generation:
                raise RuntimeError("ACMT-DP plan belongs to a reset episode")
            action = self._policy._plan(window)  # type: ignore[attr-defined]
            if generation is not None and generation != self._generation:
                raise RuntimeError("ACMT-DP plan belongs to a reset episode")
        if tuple(action.shape) != (1, PREDICTION_HORIZON, ACTION_DIM):
            raise RuntimeError(f"ACMT-DP policy must return [1,16,8], got {tuple(action.shape)}")
        return ActionPlan(plan_id, action[0].cpu())

    def _maybe_install_future(self) -> None:
        future = self._future
        if future is None or not future.done():
            return
        if self._future_generation != self._generation:
            self._future = None
            self._future_generation = None
            self._boundary_in_flight = False
            return
        try:
            plan = future.result()
        except Exception as exc:
            self._failure = exc
            logger.exception("ACMT-DP background plan failed")
            self._future = None
            self._future_generation = None
            self._boundary_in_flight = False
            return
        now = time.monotonic()
        removed, start_time = self._queue.replace_future_at_next_deadline(plan.actions, now, plan.plan_id)
        latency_ms = None
        if self._plan_started_at is not None:
            latency_ms = (now - self._plan_started_at) * 1000.0
        logger.info(
            "ACMT-DP plan ready: id=%d latency_ms=%s queue=%d replaced=%d start_in_ms=%.1f",
            plan.plan_id,
            "n/a" if latency_ms is None else f"{latency_ms:.1f}",
            len(self._queue),
            removed,
            max(0.0, (start_time - now) * 1000.0),
        )
        self._plan_started_at = None
        self._future = None
        self._future_generation = None
        self._boundary_in_flight = False

    def _submit_plan(self, window: dict | None = None) -> None:
        if self._future is not None:
            return
        plan_id = self._next_plan_id
        self._next_plan_id += 1
        generation = self._generation
        tactile_future = self._tactile_future
        submitted_window = window
        self._plan_started_at = time.monotonic()

        def wait_tactile_and_plan() -> ActionPlan:
            if tactile_future is not None:
                tactile_future.result()
            if generation != self._generation:
                raise RuntimeError("ACMT-DP plan belongs to a reset episode")
            plan_window = submitted_window
            # TactiGen mutates the causal tactile history after the accepted
            # action.  Preserve that legacy path by taking the post-generator
            # window only for that mode; real/none use the exact boundary
            # snapshot passed by the sender.
            if tactile_future is not None:
                with self._policy_state_lock:
                    plan_window = _clone_tree(self._policy._latest_window)  # type: ignore[attr-defined]
            if plan_window is None:
                raise RuntimeError("ACMT-DP planner has no observation window")
            return self._plan_now(plan_window, plan_id, generation=generation)

        self._future = self._planner.submit(wait_tactile_and_plan)
        self._future_generation = generation

    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        if obs_frame is None:
            return None
        with self._lock, torch.inference_mode():
            observation = self._prepare(obs_frame)
            with self._policy_state_lock:
                self._policy.observe(observation)  # type: ignore[attr-defined]
                if getattr(self._policy, "_observed_batch_size", 1) != 1:
                    raise ValueError("Native-DP v4 online rollout requires batch size 1")
                current_window = _clone_tree(self._policy._latest_window)  # type: ignore[attr-defined]
            self._maybe_install_future()
            if len(self._queue) == 0:
                if self._future is not None:
                    # No safe reserve remains. Never block the 30 Hz sender or
                    # replay an expired action while the replacement is still
                    # running; ``None`` makes the rollout keep/enter its safe
                    # hold state for this tick.
                    now = time.monotonic()
                    if now - self._last_hold_log >= 1.0:
                        logger.warning("ACMT-DP plan reserve exhausted; holding until replacement is ready")
                        self._last_hold_log = now
                    return None
                else:
                    if self._next_plan_id != 0:
                        now = time.monotonic()
                        if now - self._last_hold_log >= 1.0:
                            logger.error(
                                "ACMT-DP plan reserve exhausted without a replacement; stopping safely"
                            )
                            self._last_hold_log = now
                        return None
                    plan_id = self._next_plan_id
                    self._next_plan_id += 1
                    self._queue.install(
                        self._plan_now(
                            current_window,
                            plan_id,
                            generation=self._generation,
                        )
                    )
                    logger.info("ACMT-DP initial plan ready: id=%d queue=%d", plan_id, len(self._queue))
            timed = self._queue.pop_due()
            action = None if timed is None else timed.value
            if action is None:
                # A not-yet-due item is preferable to sending it early; an
                # expired sequence has already been discarded by pop_due().
                if len(self._queue) <= EXECUTION_HORIZON and not self._boundary_in_flight:
                    # If a delayed sender consumed/dropped the whole reserve
                    # before a feedback callback arrived, still request a
                    # replacement from the latest completed causal state.
                    self._boundary_in_flight = True
                    self._submit_plan(current_window)
                return None
            self._current_action_window = current_window
            if len(self._queue) <= EXECUTION_HORIZON:
                self._boundary_in_flight = True
            # ACMT-DP's postprocessor is CPU-owned.  The queue already holds
            # CPU actions, so avoid a per-tick CUDA round trip while the
            # planner is using the GPU.
            action_batch = action.unsqueeze(0)
            processed = self._postprocessor(action_batch)
            action_tensor = processed.squeeze(0)
            action_dict = make_robot_action(action_tensor, self._dataset_features)
            return torch.tensor([action_dict[key] for key in self._ordered_action_keys])

    def notify_action_executed(self, action: torch.Tensor, observation: dict | None = None) -> None:
        with self._lock:
            if action.ndim == 1:
                action = action.unsqueeze(0)
            if tuple(action.shape) != (1, ACTION_DIM):
                raise ValueError(f"executed ACMT-DP action must be [1,8], got {tuple(action.shape)}")
            current_window = getattr(self, "_current_action_window", None)
            self._current_action_window = None
            generation = self._generation
            if getattr(self._policy.config, "tactile_source", None) in {"tactigen", "substitution"}:
                self._tactile_future = self._tactile_worker.submit(
                    self._notify_tactile,
                    action.to(self._device),
                    current_window if current_window is not None else observation,
                    generation,
                )
            if self._boundary_in_flight:
                self._submit_plan(current_window)

    def _notify_tactile(self, action: torch.Tensor, window: dict | None, generation: int) -> None:
        # Hold the engine lock while taking the policy lock. Reset uses the
        # same order, so an in-flight worker cannot append a stale tactile
        # frame after an episode has been reset.
        with self._lock, self._policy_state_lock:
            if generation != self._generation or self._stopped:
                return
            self._policy.notify_action_executed(action, window)  # type: ignore[attr-defined]

    @property
    def ready(self) -> bool:
        return not self._stopped

    @property
    def failed(self) -> bool:
        return self._failure is not None


__all__ = [
    "ACMTDPInferenceEngine",
    "ActionPlan",
    "TimedAction",
    "TimedActionQueue",
]
