import logging

import numpy as np
import pytest

from lerobot.robots.fr3 import FR3, FR3Config
from lerobot.robots.fr3.feature_adapter import ACTION_KEYS
from lerobot.robots.fr3.protocols import COMMAND_STRUCT


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


class FakeUDS:
    def __init__(self):
        self.closed = False

    def settimeout(self, timeout):
        pass

    def setblocking(self, blocking):
        pass

    def recv(self, size):
        from lerobot.robots.fr3.sensorhub.uds import make_packet

        return make_packet("READY", 1)

    def send(self, packet):
        return len(packet)

    def close(self):
        self.closed = True


@pytest.fixture
def robot(tmp_path):
    instance = FR3(FR3Config(id="test", calibration_dir=tmp_path))
    instance._connected = True
    instance._sensorhub = FakeProcess()
    instance._command_socket = FakeSocket()
    yield instance
    instance._connected = False
    instance._sensorhub = None


def action(gripper=0.5):
    return {**{f"fr3_joint{i}.pos": float(i) for i in range(1, 8)}, "gripper.pos": gripper}


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


def test_connect_and_idempotent_disconnect_manage_only_sensorhub(monkeypatch, tmp_path):
    process = FakeProcess()
    uds = FakeUDS()

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
    instance.disconnect()
