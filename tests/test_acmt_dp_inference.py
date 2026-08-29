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
from lerobot.rollout.strategies.core import send_next_action
from lerobot.utils.action_interpolator import ActionInterpolator
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
    def __init__(self, mode: str, *, block_background: bool = False, fail_background: bool = False, schema_version: int = 4) -> None:
        self.config = SimpleNamespace(
            control_hz=CONTROL_HZ,
            action_execution_horizon=EXECUTION_HORIZON,
            tactile_history=4,
            pred_horizon=PREDICTION_HORIZON,
            action_dim=ACTION_DIM,
            checkpoint_schema_version=schema_version,
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


def test_v5_policy_uses_the_same_rolling_engine_protocol() -> None:
    engine = _make_engine(_FakePolicy("none", schema_version=5))
    try:
        assert engine.ready
        assert engine.get_action(_observation(0)) is not None
        assert len(engine._queue) == PREDICTION_HORIZON - 1
    finally:
        engine.stop()


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


def _joint_values(offset: float) -> dict[str, float]:
    return {f"fr3_joint{index}.pos": offset + index - 1 for index in range(1, 8)}


def test_first_action_diagnostic_is_once_per_reset(caplog: pytest.LogCaptureFixture) -> None:
    engine = _make_engine(_FakePolicy("none"))
    try:
        with caplog.at_level("INFO", logger=acmt_dp.__name__):
            engine.record_first_action_diagnostic(
                _joint_values(0.0),
                _joint_values(1.0),
                _joint_values(1.25),
            )
            engine.record_first_action_diagnostic(
                _joint_values(10.0),
                _joint_values(11.0),
                _joint_values(11.25),
            )

        messages = [
            record.message
            for record in caplog.records
            if "ACMT-DP first action diagnostic" in record.message
        ]
        assert len(messages) == 1
        assert "current_q=(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)" in messages[0]
        assert "planned_q=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)" in messages[0]
        assert "sent_q=(1.25, 2.25, 3.25, 4.25, 5.25, 6.25, 7.25)" in messages[0]
        assert "max_abs_delta=1.250000" in messages[0]
        assert "max_abs_processor_delta=0.250000" in messages[0]

        caplog.clear()
        engine.reset()
        with caplog.at_level("INFO", logger=acmt_dp.__name__):
            engine.record_first_action_diagnostic(
                _joint_values(2.0),
                _joint_values(3.0),
                _joint_values(3.5),
            )
        assert sum("ACMT-DP first action diagnostic" in record.message for record in caplog.records) == 1
    finally:
        engine.stop()


class _DiagnosticEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[dict, dict, dict]] = []

    def get_action(self, _obs_frame: dict) -> torch.Tensor:
        return torch.arange(ACTION_DIM, dtype=torch.float32)

    def record_first_action_diagnostic(self, observation: dict, planned: dict, sent: dict) -> None:
        self.calls.append((observation, planned, sent))


class _ActionRobot:
    def __init__(self, *, fail: bool) -> None:
        self.fail = fail

    def send_action(self, action: dict) -> dict:
        if self.fail:
            raise RuntimeError("synthetic send failure")
        return dict(action)


@pytest.mark.parametrize("fail", [False, True])
def test_first_action_diagnostic_hook_runs_only_after_send(fail: bool) -> None:
    names = [f"action_{index}" for index in range(ACTION_DIM)]
    engine = _DiagnosticEngine()
    robot = _ActionRobot(fail=fail)
    ctx = SimpleNamespace(
        policy=SimpleNamespace(inference=engine),
        data=SimpleNamespace(
            dataset_features={ACTION: {"names": names}},
            ordered_action_keys=names,
        ),
        processors=SimpleNamespace(robot_action_processor=lambda pair: pair[0]),
        hardware=SimpleNamespace(robot_wrapper=robot),
    )
    interpolator = ActionInterpolator()

    if fail:
        with pytest.raises(RuntimeError, match="synthetic send failure"):
            send_next_action({}, {}, ctx, interpolator)
        assert engine.calls == []
    else:
        sent = send_next_action({}, {}, ctx, interpolator)
        assert sent == {key: float(index) for index, key in enumerate(names)}
        assert len(engine.calls) == 1
        assert engine.calls[0][1] == sent
        assert engine.calls[0][2] == sent
