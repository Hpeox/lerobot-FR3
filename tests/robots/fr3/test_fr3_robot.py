import logging
from collections import deque
from unittest.mock import Mock

import numpy as np
import pytest

from lerobot.robots.fr3 import FR3, FR3Config
from lerobot.robots.fr3.feature_adapter import ACTION_KEYS
from lerobot.robots.fr3.protocols import COMMAND_FLAG_RESET_JOINT, COMMAND_STRUCT
from lerobot.robots.fr3.sensorhub.uds import make_packet, parse_packet
from lerobot.robots.robot import Robot


class FakeProcess:
    def __init__(self):
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0
        return 0


class FakeSocket:
    def __init__(self):
        self.frames = []

    def send(self, frame, flags=0):
        self.frames.append(frame)

    def close(self, linger=0):
        pass


class FakeUDS:
    def __init__(self, query_results=()):
        self.closed = False
        self.sent = []
        self.responses = deque()
        self.query_results = deque(query_results)
        self.default_query_result = None
        self.process = None

    def settimeout(self, timeout):
        pass

    def setblocking(self, blocking):
        pass

    def recv(self, size):
        if not self.responses:
            raise BlockingIOError
        return self.responses.popleft()

    def send(self, packet):
        self.sent.append(packet)
        decoded = parse_packet(packet)
        if decoded["type"] == "GET_ROBOT_RESETTING":
            result = self.query_results.popleft() if self.query_results else self.default_query_result
            if result == "FATAL":
                self.responses.append(make_packet("FATAL", 100, message="reset path failed"))
            elif result == "EOF":
                self.responses.append(b"")
            elif result == "EXIT":
                assert self.process is not None
                self.process.returncode = 9
            elif result is not None:
                self.responses.append(make_packet("ROBOT_RESETTING", 100, status_code=result))
        return len(packet)

    def close(self):
        self.closed = True


@pytest.fixture
def robot(tmp_path):
    instance = FR3(
        FR3Config(
            id="test",
            calibration_dir=tmp_path,
            reset_ack_timeout_s=0.03,
            reset_completion_timeout_s=0.03,
            reset_retry_interval_s=0.001,
        )
    )
    instance._connected = True
    instance._sensorhub = FakeProcess()
    instance._command_socket = FakeSocket()
    yield instance
    instance._connected = False
    instance._sensorhub = None


def action(gripper=0.5):
    return {**{f"fr3_joint{i}.pos": float(i) for i in range(1, 8)}, "gripper.pos": gripper}


def test_fr3_config_keeps_reset_timeouts_out_of_sensorhub_config(tmp_path):
    config = FR3Config(id="config", calibration_dir=tmp_path)
    assert config.realsense_shm_names == ("/realsense_cam1", "/realsense_cam2")
    sensorhub = config.sensorhub_dict()
    assert "required_sample_max_age_ms" in sensorhub
    assert "reset_ack_timeout_s" not in sensorhub
    assert "reset_completion_timeout_s" not in sensorhub
    assert "reset_retry_interval_s" not in sensorhub
    assert "rollout_home_joint_positions" not in sensorhub
    assert "rollout_init_delta_lower" not in sensorhub
    assert "rollout_init_delta_upper" not in sensorhub
    assert "alignment_failure_timeout_ms" not in sensorhub
    assert "camera_xense_stall_timeout_ms" not in sensorhub
    assert "ft_robot_gripper_stall_timeout_ms" not in sensorhub

    with pytest.raises(ValueError, match="reset_ack_timeout_s"):
        FR3Config(id="invalid", calibration_dir=tmp_path, reset_ack_timeout_s=np.nan)


@pytest.mark.parametrize(
    "names",
    [
        ("/only",),
        ("/cam2", "/cam1"),
        ("/cam5", "/cam4", "/cam3", "/cam2", "/cam1"),
        ["/cam2", "/cam1"],
    ],
)
def test_fr3_config_accepts_ordered_dynamic_camera_names(tmp_path, names):
    config = FR3Config(id="dynamic", calibration_dir=tmp_path, realsense_shm_names=names)
    assert config.realsense_shm_names == tuple(names)
    assert config.sensorhub_dict()["realsense_shm_names"] == list(names)


@pytest.mark.parametrize(
    ("names", "error"),
    [
        ((), ValueError),
        (("/cam1", "/cam1"), ValueError),
        (("cam1",), ValueError),
        (("/",), ValueError),
        (("/group/cam1",), ValueError),
        (("/cam\0one",), ValueError),
        ((1,), TypeError),
        ("/cam1", TypeError),
    ],
)
def test_fr3_config_rejects_invalid_camera_names(tmp_path, names, error):
    with pytest.raises(error):
        FR3Config(id="invalid", calibration_dir=tmp_path, realsense_shm_names=names)


def test_fr3_config_has_formal_controlled_rollout_defaults(tmp_path):
    config = FR3Config(id="config", calibration_dir=tmp_path)
    assert config.rollout_home_joint_positions == (
        0.1416057646,
        0.3408541381,
        -0.0186031274,
        -1.5938080549,
        0.0486696586,
        1.8890386820,
        0.0432172865,
    )
    assert config.rollout_init_delta_lower == (-0.01,) * 7
    assert config.rollout_init_delta_upper == (0.01,) * 7


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("rollout_home_joint_positions", (0.0,) * 6, ValueError),
        ("rollout_init_delta_lower", (0.0,) * 6 + (True,), TypeError),
        ("rollout_init_delta_upper", (0.0,) * 6 + (np.nan,), ValueError),
        ("rollout_init_delta_lower", (0.02,) * 7, ValueError),
    ],
)
def test_fr3_config_validates_controlled_rollout_vectors(tmp_path, field, value, error):
    kwargs = {field: value}
    with pytest.raises(error):
        FR3Config(id="invalid", calibration_dir=tmp_path, **kwargs)


def test_initialize_rollout_samples_bounded_target_and_logs(robot, caplog):
    robot._reset_joints = Mock()
    with caplog.at_level(logging.INFO):
        robot.initialize_rollout()
    target = tuple(robot._reset_joints.call_args.args[0])
    for actual, home in zip(target, robot.config.rollout_home_joint_positions, strict=True):
        assert home - 0.01 <= actual <= home + 0.01
    assert repr(target) in caplog.text


def test_return_to_home_uses_exact_target_and_logs(robot, caplog):
    robot._reset_joints = Mock()
    with caplog.at_level(logging.INFO):
        robot.return_to_home()
    target = tuple(robot._reset_joints.call_args.args[0])
    assert target == robot.config.rollout_home_joint_positions
    assert repr(target) in caplog.text


def test_send_action_clips_gripper_and_returns_actual_action(robot, caplog):
    with caplog.at_level(logging.WARNING):
        returned = robot.send_action(action(gripper=1.5))
    assert set(returned) == set(ACTION_KEYS)
    assert returned["gripper.pos"] == 1.0
    assert "clipped" in caplog.text
    frame = robot._command_socket.frames[-1]
    assert len(frame) == 112
    assert COMMAND_STRUCT.unpack(frame)[15] == 255


@pytest.mark.parametrize("gripper,gpo", [(0.0, 0), (0.5, 128), (1.0, 255)])
def test_send_action_gripper_rounding(robot, gripper, gpo):
    robot.send_action(action(gripper))
    assert COMMAND_STRUCT.unpack(robot._command_socket.frames[-1])[15] == gpo


@pytest.mark.parametrize("bad", [True, np.nan, np.inf])
def test_send_action_rejects_invalid_numbers(robot, bad):
    values = action()
    values["fr3_joint3.pos"] = bad
    with pytest.raises((TypeError, ValueError)):
        robot.send_action(values)


def test_send_action_requires_exact_fields(robot):
    values = action()
    values["extra"] = 0.0
    with pytest.raises(ValueError, match="extra"):
        robot.send_action(values)
    values = action()
    del values["fr3_joint1.pos"]
    with pytest.raises(ValueError, match="missing"):
        robot.send_action(values)


def test_fr3_feature_schema_distinguishes_xense_from_images(robot):
    schema = robot.observation_dataset_features(use_videos=False)
    assert schema["observation.xense.sensor0.force_field"]["dtype"] == "float32"
    assert schema["observation.xense.sensor0.force_field"]["shape"] == (35, 20, 3)
    assert schema["observation.images.camera.cam1.rgb"]["dtype"] == "image"
    assert schema["observation.fr3.O_T_EE"] == {
        "dtype": "float32",
        "shape": (4, 4),
        "names": None,
    }
    assert robot.observation_features["fr3.O_T_EE"].shape == (4, 4)


@pytest.mark.parametrize("camera_count", [1, 2, 5])
def test_fr3_features_expose_exact_configured_camera_count(tmp_path, camera_count):
    names = tuple(f"/cam{i}" for i in range(camera_count, 0, -1))
    instance = FR3(FR3Config(id="features", calibration_dir=tmp_path, realsense_shm_names=names))
    rgb_keys = tuple(f"camera.cam{i}.rgb" for i in range(1, camera_count + 1))
    depth_keys = tuple(f"camera.cam{i}.depth" for i in range(1, camera_count + 1))
    assert instance.visual_feature_keys == (*rgb_keys, *depth_keys)
    camera_features = {key for key in instance.observation_features if key.startswith("camera.")}
    assert camera_features == {*rgb_keys, *depth_keys}
    schema = instance.observation_dataset_features(use_videos=False)
    schema_camera_keys = {key for key in schema if key.startswith("observation.images.camera.")}
    assert schema_camera_keys == {
        *(f"observation.images.{key}" for key in rgb_keys),
        *(f"observation.images.{key}" for key in depth_keys),
    }


def test_connect_and_idempotent_disconnect_manage_only_sensorhub(monkeypatch, tmp_path):
    process = FakeProcess()
    uds = FakeUDS()
    uds.responses.append(make_packet("READY", 1))

    monkeypatch.setattr("lerobot.robots.fr3.fr3.subprocess.Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr("lerobot.robots.fr3.fr3.connect_uds", lambda *args, **kwargs: uds)
    monkeypatch.setattr("lerobot.robots.fr3.fr3.AlignedObservationClient", lambda *args: object())
    instance = FR3(FR3Config(id="lifecycle", calibration_dir=tmp_path))
    instance._open_command_socket = lambda: None
    instance.connect()
    assert instance.is_connected
    assert instance._sensorhub is process

    # Avoid calling close() on the deliberately minimal observation sentinel.
    instance._observation_client = None
    instance.disconnect()
    assert not instance.is_connected
    assert process.returncode == 0
    assert uds.closed
    assert parse_packet(uds.sent[-1])["type"] == "SHUTDOWN"
    assert parse_packet(uds.sent[-1])["sequence"] == 1
    assert instance._command_sequence == 0
    instance.disconnect()


def _attach_reset_uds(robot, query_results, *, default=None):
    uds = FakeUDS(query_results)
    uds.default_query_result = default
    uds.process = robot._sensorhub
    robot._uds = uds
    return uds


def test_reset_joints_retransmits_one_logical_command_and_waits_for_completion(robot):
    uds = _attach_reset_uds(robot, [0, 0, 1, 0])

    robot._reset_joints(np.arange(7, dtype=np.float64))

    assert len(robot._command_socket.frames) == 2
    assert robot._command_socket.frames[0] == robot._command_socket.frames[1]
    decoded = COMMAND_STRUCT.unpack(robot._command_socket.frames[0])
    assert decoded[4] == COMMAND_FLAG_RESET_JOINT
    assert decoded[5] == 1
    assert decoded[15] == 0
    requests = [parse_packet(packet) for packet in uds.sent]
    assert {packet["type"] for packet in requests} == {"GET_ROBOT_RESETTING"}
    assert [packet["sequence"] for packet in requests] == list(range(1, len(requests) + 1))
    assert robot._command_sequence == 1


def test_reset_and_action_share_command_sequence_but_not_uds_sequence(robot):
    uds = _attach_reset_uds(robot, [0, 1, 0])
    robot.send_action(action())
    robot._reset_joints(np.arange(7))
    command_socket = robot._command_socket
    robot.disconnect()

    commands = [COMMAND_STRUCT.unpack(frame)[5] for frame in command_socket.frames]
    assert commands == [1, 2]
    requests = [parse_packet(packet) for packet in uds.sent]
    assert [packet["sequence"] for packet in requests] == list(range(1, len(requests) + 1))
    assert requests[-1]["type"] == "SHUTDOWN"


@pytest.mark.parametrize("bad", [[0.0] * 6, [0.0] * 6 + [True], [0.0] * 6 + [np.nan]])
def test_reset_joints_validates_exact_finite_non_bool_targets(robot, bad):
    with pytest.raises((TypeError, ValueError)):
        robot._reset_joints(bad)
    assert not robot._command_socket.frames


def test_reset_joints_rejects_already_resetting_robot(robot):
    _attach_reset_uds(robot, [1])
    with pytest.raises(RuntimeError, match="already reports RESETTING=1"):
        robot._reset_joints(np.arange(7))
    assert not robot._command_socket.frames


def test_reset_joints_acknowledgement_timeout(robot):
    _attach_reset_uds(robot, [0], default=0)
    with pytest.raises(TimeoutError, match="acknowledge"):
        robot._reset_joints(np.arange(7))
    assert len(robot._command_socket.frames) > 1
    assert len(set(robot._command_socket.frames)) == 1


def test_reset_joints_completion_timeout_stops_retransmission(robot):
    _attach_reset_uds(robot, [0, 1], default=1)
    with pytest.raises(TimeoutError, match="complete"):
        robot._reset_joints(np.arange(7))
    assert len(robot._command_socket.frames) == 1


@pytest.mark.parametrize(
    ("failure", "match"),
    [("FATAL", "SensorHub fatal"), ("EOF", "UDS disconnected"), ("EXIT", "not running")],
)
def test_reset_joints_propagates_sensorhub_health_failures(robot, failure, match):
    _attach_reset_uds(robot, [0, failure])
    with pytest.raises(RuntimeError, match=match):
        robot._reset_joints(np.arange(7))


def test_reset_joints_is_not_part_of_generic_robot_interface():
    assert "_reset_joints" not in Robot.__dict__
