from __future__ import annotations

import threading
import time as wall_time
from types import SimpleNamespace

import pytest
import torch

from lerobot.rollout.inference import acmt_dp
from lerobot.rollout.inference.acmt_dp import (
    ACTION_DIM,
    CONTROL_HZ,
    EXECUTION_HORIZON,
    PREDICTION_HORIZON,
    ACMTDPInferenceEngine,
    ActionPlan,
    TimedActionQueue,
)
from lerobot.utils.constants import ACTION


class _IdentityProcessor:
    def __call__(self, value):
        return value

    def reset(self) -> None:
        return None


class _RecordingProcessor(_IdentityProcessor):
    def __init__(self) -> None:
        self.devices: list[torch.device] = []

    def __call__(self, value):
        self.devices.append(value.device)
        return value


class _FakePolicy:
    def __init__(self, mode: str, *, block_background: bool = False, fail_background: bool = False) -> None:
        self.config = SimpleNamespace(
            control_hz=CONTROL_HZ,
            action_execution_horizon=EXECUTION_HORIZON,
            tactile_history=4,
            pred_horizon=PREDICTION_HORIZON,
            action_dim=ACTION_DIM,
            checkpoint_schema_version=4,
            tactile_source=mode,
        )
        self.robot_type = "fr3"
        self._observed_batch_size: int | None = None
        self._latest_window: dict[str, torch.Tensor] | None = None
        self.plan_inputs: list[int] = []
        self._block_background = block_background
        self._fail_background = fail_background
        self._planner_started = threading.Event()
        self._release_planner = threading.Event()

    def reset(self) -> None:
        self._observed_batch_size = None
        self._latest_window = None
        self.plan_inputs.clear()

    def observe(self, observation: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        sequence = int(observation["sequence"])
        self._observed_batch_size = 1
        self._latest_window = {"sequence": torch.tensor([sequence])}
        return self._latest_window

    def _plan(self, window: dict[str, torch.Tensor]) -> torch.Tensor:
        sequence = int(window["sequence"].item())
        self.plan_inputs.append(sequence)
        if len(self.plan_inputs) > 1 and self._block_background:
            self._planner_started.set()
            self._release_planner.wait(timeout=2.0)
        if len(self.plan_inputs) > 1 and self._fail_background:
            raise RuntimeError("synthetic planner failure")
        return torch.full((1, PREDICTION_HORIZON, ACTION_DIM), float(sequence))


class _FakeClock:
    def __init__(self, value: float = 100.0) -> None:
        self.value = value

    def monotonic(self) -> float:
        return self.value


def _make_engine(policy: _FakePolicy) -> ACMTDPInferenceEngine:
    names = [f"action_{index}" for index in range(ACTION_DIM)]
    engine = ACMTDPInferenceEngine(
        policy=policy,
        preprocessor=_IdentityProcessor(),
        postprocessor=_IdentityProcessor(),
        dataset_features={ACTION: {"names": names}},
        ordered_action_keys=names,
        task="test",
        device="cpu",
        robot_type="fr3",
    )
    engine._prepare = lambda observation: observation  # type: ignore[method-assign]
    engine.reset()
    engine.start()
    return engine


def _observation(sequence: int) -> dict[str, torch.Tensor]:
    return {"sequence": torch.tensor(sequence)}


@pytest.mark.parametrize("mode", ["real", "none"])
def test_background_plan_does_not_block_observation_capture(mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _FakeClock()
    monkeypatch.setattr(acmt_dp.time, "monotonic", clock.monotonic)
    policy = _FakePolicy(mode, block_background=True)
    engine = _make_engine(policy)
    period = 1.0 / CONTROL_HZ

    try:
        for sequence in range(EXECUTION_HORIZON):
            clock.value = 100.0 + sequence * period
            action = engine.get_action(_observation(sequence))
            assert action is not None
            engine.notify_action_executed(action)

        assert policy._planner_started.wait(timeout=1.0)
        clock.value = 100.0 + EXECUTION_HORIZON * period
        result: dict[str, torch.Tensor | None] = {}

        def capture_action() -> None:
            result["action"] = engine.get_action(_observation(EXECUTION_HORIZON))

        worker = threading.Thread(target=capture_action)
        worker.start()
        worker.join(timeout=0.2)
        assert not worker.is_alive(), "observation capture remained serialized behind planner forward"
        action = result["action"]
        assert action is not None
        engine.notify_action_executed(action)

        assert policy.plan_inputs[:2] == [0, EXECUTION_HORIZON - 1]
    finally:
        policy._release_planner.set()
        engine.stop()


def test_boundary_plan_replaces_only_future_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _FakeClock()
    monkeypatch.setattr(acmt_dp.time, "monotonic", clock.monotonic)
    queue = TimedActionQueue()
    old = torch.zeros(PREDICTION_HORIZON, ACTION_DIM)
    new = torch.ones(PREDICTION_HORIZON, ACTION_DIM)
    queue.install(ActionPlan(0, old, start_time=100.0))

    removed, start_time = queue.replace_future_at_next_deadline(new, 100.0, plan_id=1)

    # The currently due action remains owned by plan 0; the other fifteen
    # future actions are replaced by plan 1.
    assert removed == PREDICTION_HORIZON - 1
    assert start_time == pytest.approx(100.0 + 1.0 / CONTROL_HZ)
    first = queue.pop_due(100.0)
    assert first is not None
    assert (first.plan_id, first.action_index) == (0, 0)
    replacement = queue.pop_due(100.0 + 1.0 / CONTROL_HZ)
    assert replacement is not None
    assert (replacement.plan_id, replacement.action_index) == (1, 0)


def test_background_planner_failure_is_exposed_as_engine_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _FakeClock()
    monkeypatch.setattr(acmt_dp.time, "monotonic", clock.monotonic)
    policy = _FakePolicy("none", fail_background=True)
    engine = _make_engine(policy)
    period = 1.0 / CONTROL_HZ

    try:
        for sequence in range(EXECUTION_HORIZON):
            clock.value = 100.0 + sequence * period
            action = engine.get_action(_observation(sequence))
            assert action is not None
            engine.notify_action_executed(action)

        deadline = wall_time.monotonic() + 1.0
        while not engine._future.done() and wall_time.monotonic() < deadline:  # type: ignore[union-attr]
            wall_time.sleep(0.005)
        assert engine._future is not None and engine._future.done()
        clock.value = 100.0 + EXECUTION_HORIZON * period
        assert engine.get_action(_observation(EXECUTION_HORIZON)) is not None
        assert engine.failed
    finally:
        engine.stop()


def test_reserve_exhaustion_holds_without_replaying_old_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _FakeClock()
    monkeypatch.setattr(acmt_dp.time, "monotonic", clock.monotonic)
    policy = _FakePolicy("none", block_background=True)
    engine = _make_engine(policy)
    period = 1.0 / CONTROL_HZ

    try:
        for sequence in range(EXECUTION_HORIZON):
            clock.value = 100.0 + sequence * period
            action = engine.get_action(_observation(sequence))
            assert action is not None
            engine.notify_action_executed(action)
        assert policy._planner_started.wait(timeout=1.0)

        clock.value = 100.0 + PREDICTION_HORIZON * period
        assert engine.get_action(_observation(PREDICTION_HORIZON)) is None
        assert not engine.failed
    finally:
        policy._release_planner.set()
        engine.stop()


def test_action_postprocessing_stays_on_cpu_while_planner_device_is_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _FakeClock()
    monkeypatch.setattr(acmt_dp.time, "monotonic", clock.monotonic)
    policy = _FakePolicy("none")
    postprocessor = _RecordingProcessor()
    engine = ACMTDPInferenceEngine(
        policy=policy,
        preprocessor=_IdentityProcessor(),
        postprocessor=postprocessor,
        dataset_features={ACTION: {"names": [f"action_{index}" for index in range(ACTION_DIM)]}},
        ordered_action_keys=[f"action_{index}" for index in range(ACTION_DIM)],
        task="test",
        device="cuda",
        robot_type="fr3",
    )
    engine._prepare = lambda observation: observation  # type: ignore[method-assign]
    engine.reset()
    engine.start()
    try:
        assert engine.get_action(_observation(0)) is not None
        assert postprocessor.devices == [torch.device("cpu")]
    finally:
        engine.stop()
