"""Local unit/component tests for the Controlled rollout lifecycle."""

from __future__ import annotations

import importlib.util
import os
import socket
import sys
import types
from collections import deque
from threading import Event, Lock, Thread
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

# Keep the core lifecycle tests runnable in minimal installations. Tests that
# exercise the real dataset/video path explicitly skip when those optional
# dependencies are unavailable.
if importlib.util.find_spec("datasets") is None:
    from lerobot.utils import import_utils

    import_utils._require_package_cache["datasets"] = True
    import_utils._require_package_cache["av"] = True
    datasets_stub = types.ModuleType("lerobot.datasets")
    datasets_stub.__path__ = []
    datasets_stub.LeRobotDataset = type("LeRobotDataset", (), {})
    datasets_stub.VideoEncodingManager = type("VideoEncodingManager", (), {})
    datasets_stub.aggregate_pipeline_dataset_features = lambda *args, **kwargs: {}
    datasets_stub.create_initial_features = lambda *args, **kwargs: {}
    sys.modules["lerobot.datasets"] = datasets_stub
    dataset_utils_stub = types.ModuleType("lerobot.datasets.utils")
    dataset_utils_stub.DEFAULT_VIDEO_FILE_SIZE_IN_MB = 500
    sys.modules["lerobot.datasets.utils"] = dataset_utils_stub

from lerobot.rollout.configs import ControlledStrategyConfig
from lerobot.rollout.context import _require_controlled_capabilities
from lerobot.rollout.control_uds import (
    ControlledCommand,
    ControlledUDSDisconnected,
    ControlledUDSServer,
    make_command,
    parse_response,
)
from lerobot.rollout.strategies.controlled import ControlledFailStop, ControlledStrategy
from lerobot.rollout.strategies.factory import create_strategy
from lerobot.rollout.inference.rtc import RTCInferenceEngine
from lerobot.rollout.robot_wrapper import ThreadSafeRobot
from lerobot.robots.robot import Robot
from lerobot.utils.action_interpolator import ActionInterpolator


class FakeControl:
    def __init__(self, *, blocking=(), polling=(), drain_counts=()):
        self.blocking = deque(blocking)
        self.polling = deque(polling)
        self.drain_counts = deque(drain_counts)
        self.last_sequence = -1
        self.acks = []
        self.statuses = []
        self.events = []
        self.connected = True
        self.closed = False

    def accept(self):
        self.events.append("accept")

    def recv(self, *, blocking):
        source = self.blocking if blocking else self.polling
        self.events.append(f"recv:{'blocking' if blocking else 'poll'}")
        return source.popleft() if source else None

    def consume_sequence(self, sequence):
        if sequence <= self.last_sequence:
            return False
        self.last_sequence = sequence
        return True

    def send_ack(self, command, **payload):
        self.acks.append((command, payload))
        self.events.append(f"ack:{command.operation}:{payload['accepted']}")

    def drain(self):
        count = self.drain_counts.popleft() if self.drain_counts else 0
        self.events.append(f"drain:{count}")
        return count

    def publish_status(self, status, *, phase, code="ok", message=""):
        self.statuses.append((status, phase, code, message))
        self.events.append(f"status:{status}:{phase}")

    def close(self):
        self.closed = True


class FakeDataset:
    def __init__(self, pending=False):
        self.pending = pending
        self.saved = 0
        self.cleared = 0
        self.finalized = 0

    def has_pending_frames(self):
        return self.pending

    def save_episode(self):
        self.saved += 1
        self.pending = False

    def clear_episode_buffer(self):
        self.cleared += 1
        self.pending = False

    def finalize(self):
        self.finalized += 1


def command(sequence, operation):
    return ControlledCommand(sequence=sequence, operation=operation)


def make_context(*, dataset=None, robot=None):
    engine = MagicMock()
    engine.failed = False
    engine.ready = True
    engine.reset_for_controlled_rollout = None
    robot = robot or SimpleNamespace(
        initialize_rollout=MagicMock(),
        return_to_home=MagicMock(),
        is_connected=False,
    )
    wrapper = SimpleNamespace(
        inner=robot,
        get_observation=getattr(robot, "get_observation", MagicMock()),
        send_action=getattr(robot, "send_action", MagicMock()),
    )
    cfg = SimpleNamespace(
        interpolation_multiplier=1,
        fps=30.0,
        duration=0.0,
        use_torch_compile=False,
        dataset=None,
        task="test task",
        display_data=False,
    )
    ctx = SimpleNamespace(
        runtime=SimpleNamespace(cfg=cfg, shutdown_event=SimpleNamespace(is_set=lambda: False)),
        hardware=SimpleNamespace(robot_wrapper=wrapper, teleop=None),
        policy=SimpleNamespace(inference=engine),
        processors=SimpleNamespace(),
        data=SimpleNamespace(dataset=dataset, dataset_features={}),
    )
    return ctx, engine, robot


def make_strategy(control):
    strategy = ControlledStrategy(ControlledStrategyConfig(control_socket_path="/tmp/unused.sock"))
    strategy._control = control
    strategy._interpolator = ActionInterpolator(multiplier=1)
    return strategy


def make_real_dataset_context(dataset, features):
    frame_index = 0

    def get_observation():
        nonlocal frame_index
        value = float(frame_index)
        image = np.full((64, 96, 3), frame_index * 20, dtype=np.uint8)
        frame_index += 1
        return {"joint_0.pos": value, "joint_1.pos": value + 1.0, "cam": image}

    robot = SimpleNamespace(
        initialize_rollout=MagicMock(),
        return_to_home=MagicMock(),
        get_observation=get_observation,
        send_action=MagicMock(side_effect=lambda action: action),
        is_connected=False,
    )
    ctx, engine, robot = make_context(dataset=dataset, robot=robot)
    engine.get_action.return_value = torch.tensor([0.5, -0.5], dtype=torch.float32)
    ctx.runtime.cfg.dataset = SimpleNamespace(single_task="controlled dataset test")
    ctx.runtime.cfg.fps = 30.0
    ctx.processors = SimpleNamespace(
        robot_observation_processor=lambda observation: observation,
        robot_action_processor=lambda action_and_observation: action_and_observation[0],
    )
    ctx.data.dataset_features = features
    ctx.data.ordered_action_keys = ["joint_0.pos", "joint_1.pos"]
    return ctx, engine, robot


def test_controlled_config_factory_and_generic_api_boundaries():
    config = ControlledStrategyConfig()
    assert config.type == "controlled"
    assert config.control_socket_path == f"/run/user/{os.getuid()}/lerobot_controlled.sock"
    assert isinstance(create_strategy(config), ControlledStrategy)
    with pytest.raises(ValueError, match="control_socket_path"):
        ControlledStrategyConfig(control_socket_path="")
    assert "initialize_rollout" not in Robot.__dict__
    assert "return_to_home" not in Robot.__dict__
    assert "initialize_rollout" not in ThreadSafeRobot.__dict__
    assert "return_to_home" not in ThreadSafeRobot.__dict__


def test_control_uds_ack_status_sequence_and_blind_drain(tmp_path):
    path = tmp_path / "controlled.sock"
    server = ControlledUDSServer(str(path))
    client = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    client.connect(str(path))
    server.accept()
    try:
        client.send(make_command(10, "INITIALIZE"))
        received = server.recv(blocking=True)
        assert received == command(10, "INITIALIZE")
        assert server.consume_sequence(received.sequence)
        server.send_ack(received, accepted=True, code="accepted", phase="WAIT_INITIALIZE")
        ack = parse_response(client.recv(1025))
        assert ack["type"] == "ACK"
        assert ack["accepted"] is True

        client.send(make_command(11, "START"))
        client.send(make_command(12, "FAIL_STOP"))
        assert server.drain() == 2
        client.settimeout(0.01)
        with pytest.raises(TimeoutError):
            client.recv(1025)

        server.publish_status("INITIALIZED", phase="WAIT_START")
        status = parse_response(client.recv(1025))
        assert status["type"] == "STATUS"
        assert status["status"] == "INITIALIZED"
    finally:
        client.close()
        server.close()
    assert not path.exists()


def test_control_uds_disconnect_during_drain_is_fatal(tmp_path):
    path = tmp_path / "controlled.sock"
    server = ControlledUDSServer(str(path))
    client = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    client.connect(str(path))
    server.accept()
    client.close()
    try:
        with pytest.raises(ControlledUDSDisconnected):
            server.drain()
    finally:
        server.close()


def test_initializing_blind_drain_discards_stale_commands_with_real_uds(tmp_path):
    initialize_started = Event()
    release_initialize = Event()

    def initialize_rollout():
        initialize_started.set()
        assert release_initialize.wait(timeout=1)

    robot = SimpleNamespace(
        initialize_rollout=initialize_rollout,
        return_to_home=MagicMock(),
        is_connected=False,
    )
    ctx, engine, robot = make_context(robot=robot)
    strategy = make_strategy(None)
    strategy._engine = engine
    strategy._control = ControlledUDSServer(str(tmp_path / "controlled.sock"))
    errors = []

    def run_strategy():
        try:
            strategy.run(ctx)
        except BaseException as exc:
            errors.append(exc)

    worker = Thread(target=run_strategy)
    worker.start()
    client = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    client.settimeout(1)
    client.connect(str(tmp_path / "controlled.sock"))
    try:
        assert parse_response(client.recv(1025))["status"] == "READY"
        client.send(make_command(1, "INITIALIZE"))
        assert parse_response(client.recv(1025))["type"] == "ACK"
        assert parse_response(client.recv(1025))["status"] == "INITIALIZING"
        assert initialize_started.wait(timeout=1)

        client.send(make_command(2, "START"))
        client.send(make_command(3, "FAIL_STOP"))
        release_initialize.set()
        initialized = parse_response(client.recv(1025))
        assert initialized["status"] == "INITIALIZED"
        assert initialized["phase"] == "WAIT_START"

        client.send(make_command(4, "SHUTDOWN"))
        shutdown_ack = parse_response(client.recv(1025))
        assert shutdown_ack["type"] == "ACK"
        assert shutdown_ack["sequence"] == 4
        assert parse_response(client.recv(1025))["status"] == "SHUTTING_DOWN"
        worker.join(timeout=1)
        assert not worker.is_alive()
        assert not errors
        robot.return_to_home.assert_called_once()
    finally:
        release_initialize.set()
        client.close()
        worker.join(timeout=1)
        strategy.teardown(ctx)


def test_fail_stop_retries_use_normal_increasing_sequence_rules(tmp_path):
    server = ControlledUDSServer(str(tmp_path / "controlled.sock"))
    try:
        assert server.consume_sequence(100)
        assert server.consume_sequence(101)
        assert server.consume_sequence(102)
        assert not server.consume_sequence(102)
        assert not server.consume_sequence(99)
    finally:
        server.close()


def test_full_stop_cycle_orders_drain_before_status_and_reinitializes():
    control = FakeControl(
        blocking=[
            command(1, "INITIALIZE"),
            command(2, "START"),
            command(4, "SHUTDOWN"),
        ],
        polling=[command(3, "STOP")],
    )
    ctx, engine, robot = make_context()
    strategy = make_strategy(control)
    strategy._engine = engine

    strategy.run(ctx)

    assert robot.initialize_rollout.call_count == 1
    assert robot.return_to_home.call_count == 1
    engine.reset.assert_called_once()
    engine.resume.assert_called_once()
    assert engine.pause.call_count >= 2
    assert [status for status, _phase, _code, _message in control.statuses] == [
        "READY",
        "INITIALIZING",
        "INITIALIZED",
        "STARTED",
        "STOPPED",
        "SHUTTING_DOWN",
    ]
    for status in ("INITIALIZING", "INITIALIZED", "STARTED", "STOPPED", "SHUTTING_DOWN"):
        status_index = next(i for i, event in enumerate(control.events) if event.startswith(f"status:{status}:"))
        assert control.events[status_index - 1].startswith("drain:")
    stopped_index = next(i for i, event in enumerate(control.events) if event.startswith("status:STOPPED:"))
    assert control.events[stopped_index + 1] == "recv:blocking"


def test_invalid_phase_command_is_acknowledged_and_not_deferred():
    control = FakeControl(blocking=[command(1, "START"), command(2, "SHUTDOWN")])
    ctx, engine, robot = make_context()
    strategy = make_strategy(control)
    strategy._engine = engine

    strategy.run(ctx)

    assert control.acks[0][0].operation == "START"
    assert control.acks[0][1]["accepted"] is False
    assert control.acks[0][1]["code"] == "invalid_phase"
    robot.initialize_rollout.assert_not_called()
    robot.return_to_home.assert_called_once()


def test_actual_fail_stop_is_no_new_motion_fatal_teardown():
    control = FakeControl(blocking=[command(1, "FAIL_STOP")])
    ctx, engine, robot = make_context()
    strategy = make_strategy(control)
    strategy._engine = engine

    with pytest.raises(ControlledFailStop):
        strategy.run(ctx)

    robot.initialize_rollout.assert_not_called()
    robot.return_to_home.assert_not_called()
    assert control.statuses[-1][0] == "FAIL_STOPPING"
    assert control.acks[0][1]["accepted"] is True


def test_graceful_home_failure_is_non_success_without_second_motion():
    robot = SimpleNamespace(
        initialize_rollout=MagicMock(),
        return_to_home=MagicMock(side_effect=RuntimeError("home failed")),
        is_connected=False,
    )
    control = FakeControl(blocking=[command(1, "SHUTDOWN")])
    ctx, engine, robot = make_context(robot=robot)
    strategy = make_strategy(control)
    strategy._engine = engine

    with pytest.raises(RuntimeError, match="home failed"):
        strategy.run(ctx)

    robot.return_to_home.assert_called_once()
    assert control.statuses[-1][0] == "ERROR"
    assert control.statuses[-1][2] == "return_home_failed"
    strategy.teardown(ctx)
    robot.return_to_home.assert_called_once()


def test_dataset_cleanup_failure_does_not_override_home_failure(monkeypatch):
    dataset = FakeDataset()

    class FailingManager:
        def __init__(self, owned_dataset):
            self.dataset = owned_dataset

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            raise RuntimeError("dataset finalize failed")

    monkeypatch.setattr("lerobot.rollout.strategies.controlled.VideoEncodingManager", FailingManager)
    robot = SimpleNamespace(
        initialize_rollout=MagicMock(),
        return_to_home=MagicMock(side_effect=RuntimeError("home failed")),
        is_connected=False,
    )
    control = FakeControl(blocking=[command(1, "SHUTDOWN")])
    ctx, engine, robot = make_context(dataset=dataset, robot=robot)
    strategy = make_strategy(control)
    strategy._engine = engine

    with pytest.raises(RuntimeError, match="home failed") as captured:
        strategy.run(ctx)

    assert captured.value.__cause__ is not None
    assert "dataset finalize failed" in str(captured.value.__cause__)
    robot.return_to_home.assert_called_once()


def test_uds_disconnect_is_internal_fatal_without_motion():
    class DisconnectControl(FakeControl):
        def recv(self, *, blocking):
            raise ControlledUDSDisconnected("controller disconnected")

    control = DisconnectControl()
    ctx, engine, robot = make_context()
    strategy = make_strategy(control)
    strategy._engine = engine

    with pytest.raises(ControlledUDSDisconnected, match="controller disconnected"):
        strategy.run(ctx)

    robot.initialize_rollout.assert_not_called()
    robot.return_to_home.assert_not_called()


@pytest.mark.parametrize(("operation", "saved", "cleared"), [("STOP", 1, 0), ("ABORT", 0, 1)])
def test_optional_dataset_episode_ownership(monkeypatch, operation, saved, cleared):
    dataset = FakeDataset(pending=True)

    class Manager:
        def __init__(self, owned_dataset):
            self.dataset = owned_dataset

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.dataset.finalize()
            return False

    monkeypatch.setattr("lerobot.rollout.strategies.controlled.VideoEncodingManager", Manager)
    control = FakeControl(
        blocking=[command(1, "INITIALIZE"), command(2, "START"), command(4, "SHUTDOWN")],
        polling=[command(3, operation)],
    )
    ctx, engine, _robot = make_context(dataset=dataset)
    strategy = make_strategy(control)
    strategy._engine = engine

    strategy.run(ctx)

    assert dataset.saved == saved
    assert dataset.cleared == cleared
    assert ("STOPPED" if operation == "STOP" else "ABORTED") in {
        status for status, _phase, _code, _message in control.statuses
    }
    assert dataset.finalized == 1
    strategy.teardown(ctx)
    assert dataset.finalized == 1


def test_stale_observation_exception_propagates_and_only_pauses_inference():
    class StaleObservation(RuntimeError):
        pass

    robot = SimpleNamespace(
        initialize_rollout=MagicMock(),
        return_to_home=MagicMock(),
        get_observation=MagicMock(side_effect=StaleObservation("max age")),
        is_connected=False,
    )
    control = FakeControl()
    ctx, engine, robot = make_context(robot=robot)
    strategy = make_strategy(control)
    strategy._engine = engine
    strategy._phase = "RUNNING"

    with pytest.raises(StaleObservation, match="max age"):
        strategy._run_rollout(ctx)

    engine.pause.assert_called_once()
    robot.return_to_home.assert_not_called()


def test_missing_controlled_robot_capability_is_rejected():
    with pytest.raises(NotImplementedError, match="return_to_home"):
        _require_controlled_capabilities(SimpleNamespace(initialize_rollout=lambda: None))


def test_dataset_disabled_uses_no_video_manager(monkeypatch):
    monkeypatch.setattr(
        "lerobot.rollout.strategies.controlled.VideoEncodingManager",
        MagicMock(side_effect=AssertionError("must not be constructed")),
    )
    control = FakeControl(blocking=[command(1, "SHUTDOWN")])
    ctx, engine, _robot = make_context(dataset=None)
    strategy = make_strategy(control)
    strategy._engine = engine
    strategy.run(ctx)


def test_controlled_real_dataset_stop_encodes_and_reloads_video(tmp_path):
    pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")
    pytest.importorskip("av", reason="av is required (install lerobot[dataset])")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["joint_0.pos", "joint_1.pos"],
        },
        "observation.images.cam": {
            "dtype": "video",
            "shape": (64, 96, 3),
            "names": ["height", "width", "channels"],
        },
        "action": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["joint_0.pos", "joint_1.pos"],
        },
    }
    root = tmp_path / "controlled_dataset"
    dataset = LeRobotDataset.create(
        repo_id="local/controlled-dataset-test",
        fps=30,
        features=features,
        root=root,
        use_videos=True,
        video_backend="pyav",
    )
    control = FakeControl(
        blocking=[command(1, "INITIALIZE"), command(2, "START"), command(4, "SHUTDOWN")],
        polling=[None, None, None, None, command(3, "STOP")],
    )
    ctx, engine, robot = make_real_dataset_context(dataset, features)
    strategy = make_strategy(control)
    strategy._engine = engine

    strategy.run(ctx)

    assert dataset.meta.total_episodes == 1
    assert dataset.meta.total_frames == 4
    assert dataset._is_finalized is True
    assert not dataset.has_pending_frames()
    assert list(root.glob("videos/**/*.mp4"))
    robot.send_action.assert_called()

    reloaded = LeRobotDataset(
        repo_id="local/controlled-dataset-test",
        root=root,
        video_backend="pyav",
    )
    assert len(reloaded) == 4
    frame = reloaded[0]
    assert tuple(frame["observation.state"].shape) == (2,)
    assert tuple(frame["action"].shape) == (2,)
    assert tuple(frame["observation.images.cam"].shape) == (3, 64, 96)


def test_controlled_real_dataset_abort_discards_partial_episode(tmp_path):
    pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")
    pytest.importorskip("av", reason="av is required (install lerobot[dataset])")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["joint_0.pos", "joint_1.pos"],
        },
        "observation.images.cam": {
            "dtype": "video",
            "shape": (64, 96, 3),
            "names": ["height", "width", "channels"],
        },
        "action": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["joint_0.pos", "joint_1.pos"],
        },
    }
    root = tmp_path / "controlled_aborted_dataset"
    dataset = LeRobotDataset.create(
        repo_id="local/controlled-aborted-dataset-test",
        fps=30,
        features=features,
        root=root,
        use_videos=True,
        video_backend="pyav",
    )
    control = FakeControl(
        blocking=[command(1, "INITIALIZE"), command(2, "START"), command(4, "SHUTDOWN")],
        polling=[None, None, command(3, "ABORT")],
    )
    ctx, engine, _robot = make_real_dataset_context(dataset, features)
    strategy = make_strategy(control)
    strategy._engine = engine

    strategy.run(ctx)

    assert dataset.meta.total_episodes == 0
    assert dataset.meta.total_frames == 0
    assert dataset._is_finalized is True
    assert not dataset.has_pending_frames()
    assert not list(root.glob("videos/**/*.mp4"))
    assert not list(root.glob("images/**/*.png"))


def test_duration_completion_returns_to_wait_initialize():
    control = FakeControl(
        blocking=[command(1, "INITIALIZE"), command(2, "START"), command(3, "SHUTDOWN")]
    )
    ctx, engine, _robot = make_context()
    ctx.runtime.cfg.duration = 1e-9
    strategy = make_strategy(control)
    strategy._engine = engine

    strategy.run(ctx)

    completed = next(item for item in control.statuses if item[0] == "COMPLETED")
    assert completed[1] == "WAIT_INITIALIZE"


def test_rtc_controlled_reset_clears_old_inference_after_it_quiesces():
    engine = object.__new__(RTCInferenceEngine)
    engine._policy_active = Event()
    engine._policy_active.set()
    engine._inference_lock = Lock()
    engine._obs_lock = Lock()
    engine._obs_holder = {"obs": {"old": True}}
    engine._policy = MagicMock()
    engine._preprocessor = MagicMock()
    engine._postprocessor = MagicMock()

    class Queue:
        value = None

        def clear(self):
            self.value = None

    queue = Queue()
    engine._action_queue = queue
    inference_started = Event()
    release_inference = Event()
    reset_finished = Event()

    def old_inference():
        with engine._inference_lock:
            inference_started.set()
            release_inference.wait(timeout=1)
            queue.value = "old episode result"

    def reset_episode():
        engine.reset_for_controlled_rollout()
        reset_finished.set()

    inference_thread = Thread(target=old_inference)
    reset_thread = Thread(target=reset_episode)
    inference_thread.start()
    assert inference_started.wait(timeout=1)
    reset_thread.start()
    assert not reset_finished.wait(timeout=0.02)
    release_inference.set()
    inference_thread.join(timeout=1)
    reset_thread.join(timeout=1)

    assert reset_finished.is_set()
    assert queue.value is None
    assert engine._obs_holder["obs"] is None
