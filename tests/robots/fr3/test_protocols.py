import errno
import math
import time

import numpy as np
import pytest

from lerobot.robots.fr3.protocols import (
    COMMAND_STRUCT,
    TELEMETRY_STRUCT,
    GripperTelemetry,
    RobotTelemetry,
    pack_command,
    parse_telemetry,
    policy_gripper_to_gpo,
)
from lerobot.robots.fr3.sensorhub.readers import TelemetryReader
from lerobot.robots.fr3.sensorhub.uds import make_packet, parse_packet


@pytest.mark.parametrize(
    ("value", "clipped", "gpo"), [(-1, 0.0, 0), (0, 0.0, 0), (0.5, 0.5, 128), (1, 1.0, 255), (2, 1.0, 255)]
)
def test_gripper_policy_scale(value, clipped, gpo):
    assert policy_gripper_to_gpo(value) == (clipped, gpo)


@pytest.mark.parametrize("value", [True, math.nan, math.inf, -math.inf])
def test_gripper_rejects_bool_and_nonfinite(value):
    with pytest.raises((TypeError, ValueError)):
        policy_gripper_to_gpo(value)


def test_command_abi_golden_frame():
    frame = pack_command(7, np.arange(7, dtype=np.float64), 128, realtime_ns=11, monotonic_ns=12)
    assert len(frame) == 112
    decoded = COMMAND_STRUCT.unpack(frame)
    assert decoded[:9] == (b"FRCMD1\0\0", 1, 48, 112, 0, 7, 11, 12, 0.0)
    assert decoded[9:15] == tuple(float(i) for i in range(1, 7))
    assert decoded[15] == 128


def _telemetry_frame(source: int, mask: int, floats: list[float], gpo=0, gcu=0):
    return TELEMETRY_STRUCT.pack(b"FGT1", 1, source, 0, 42, 12.5, mask, *floats, gpo, gcu)


def test_parse_robot_and_gripper_telemetry():
    values = [math.nan] * 58
    values[8:15] = range(7)
    values[15:22] = range(10, 17)
    values[22:29] = range(20, 27)
    robot = parse_telemetry(_telemetry_frame(2, 2, values))
    assert isinstance(robot, RobotTelemetry)
    np.testing.assert_array_equal(robot.q, np.arange(7, dtype=np.float32))
    np.testing.assert_array_equal(robot.dq, np.arange(10, 17, dtype=np.float32))
    np.testing.assert_array_equal(robot.tau_j, np.arange(20, 27, dtype=np.float32))

    gripper = parse_telemetry(_telemetry_frame(3, 4, [math.nan] * 58, 19, 3))
    assert isinstance(gripper, GripperTelemetry)
    assert (gripper.gpo, gripper.gcu) == (19, 3)


def test_parse_telemetry_rejects_wrong_size_and_mask():
    with pytest.raises(ValueError):
        parse_telemetry(b"short")
    with pytest.raises(ValueError):
        parse_telemetry(_telemetry_frame(2, 0, [0.0] * 58))


def test_uds_packet_has_exact_schema_and_bounded_diagnostic():
    packet = make_packet("FATAL", 4, status_code=7, message="x" * 2000, timestamp_ns=9)
    assert len(packet) <= 512
    decoded = parse_packet(packet)
    assert set(decoded) == {
        "protocol_version",
        "type",
        "sequence",
        "timestamp_ns",
        "status_code",
        "message",
    }
    assert decoded["type"] == "FATAL"
    assert decoded["sequence"] == 4


def test_telemetry_reader_keeps_robot_and_gripper_sources_independent(tmp_path):
    import zmq

    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    endpoint = f"ipc://{tmp_path / 'telemetry.sock'}"
    try:
        publisher.bind(endpoint)
    except zmq.ZMQError as exc:
        publisher.close(linger=0)
        context.term()
        if exc.errno in {errno.EPERM, errno.EACCES}:
            pytest.skip("sandbox forbids creating ZMQ IPC sockets")
        raise
    reader = TelemetryReader(endpoint)
    try:
        time.sleep(0.1)
        values = [0.0] * 58
        seen = set()
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline and seen != {"robot", "gripper"}:
            publisher.send(_telemetry_frame(2, 2, values))
            publisher.send(_telemetry_frame(3, 4, values, 9, 2))
            for _ in range(2):
                try:
                    sample = reader.read(timeout_s=0.02)
                except TimeoutError:
                    break
                seen.add("robot" if hasattr(sample, "q") else "gripper")
        assert seen == {"robot", "gripper"}
    finally:
        reader.close()
        publisher.close(linger=0)
        context.term()
