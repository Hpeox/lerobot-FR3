import errno
import json
import math
import socket
import time
from threading import Event, Thread

import numpy as np
import pytest

from lerobot.robots.fr3.protocols import (
    COMMAND_FLAG_RESET_JOINT,
    COMMAND_STRUCT,
    ROBOT_TELEMETRY_FLAG_RESETTING,
    TELEMETRY_STRUCT,
    GripperTelemetry,
    RobotTelemetry,
    pack_command,
    parse_telemetry,
    policy_gripper_to_gpo,
)
from lerobot.robots.fr3.sensorhub.readers import TelemetryReader
from lerobot.robots.fr3.sensorhub.samples import RobotSample
from lerobot.robots.fr3.sensorhub.uds import (
    MAX_PACKET_SIZE,
    PROTOCOL_VERSION,
    UDSControlServer,
    make_packet,
    parse_packet,
)


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


def test_reset_command_flag_preserves_fixed_abi_and_retransmission_bytes():
    frame = pack_command(
        8,
        np.arange(7, dtype=np.float64),
        0,
        realtime_ns=21,
        monotonic_ns=22,
        flags=COMMAND_FLAG_RESET_JOINT,
    )
    assert len(frame) == 112
    assert COMMAND_STRUCT.unpack(frame)[4] == COMMAND_FLAG_RESET_JOINT
    assert frame == pack_command(
        8,
        np.arange(7, dtype=np.float64),
        0,
        realtime_ns=21,
        monotonic_ns=22,
        flags=COMMAND_FLAG_RESET_JOINT,
    )
    with pytest.raises(ValueError, match="unsupported command flags"):
        pack_command(9, np.arange(7), 0, flags=1 << 4)


def _telemetry_frame(source: int, mask: int, floats: list[float], gpo=0, gcu=0, *, flags=0):
    return TELEMETRY_STRUCT.pack(b"FGT1", 1, source, flags, 42, 12.5, mask, *floats, gpo, gcu)


def test_parse_robot_and_gripper_telemetry():
    values = [math.nan] * 58
    values[8:15] = range(7)
    values[15:22] = range(10, 17)
    values[22:29] = range(20, 27)
    robot = parse_telemetry(_telemetry_frame(2, 2, values))
    assert isinstance(robot, RobotTelemetry)
    assert robot.resetting is False
    np.testing.assert_array_equal(robot.q, np.arange(7, dtype=np.float32))
    np.testing.assert_array_equal(robot.dq, np.arange(10, 17, dtype=np.float32))
    np.testing.assert_array_equal(robot.tau_j, np.arange(20, 27, dtype=np.float32))

    gripper = parse_telemetry(_telemetry_frame(3, 4, [math.nan] * 58, 19, 3))
    assert isinstance(gripper, GripperTelemetry)
    assert (gripper.gpo, gripper.gcu) == (19, 3)


def test_robot_resetting_flag_stays_out_of_alignment_samples():
    values = [0.0] * 58
    robot = parse_telemetry(
        _telemetry_frame(2, 2, values, flags=ROBOT_TELEMETRY_FLAG_RESETTING | (1 << 7))
    )
    assert isinstance(robot, RobotTelemetry)
    assert robot.resetting is True
    assert "resetting" not in RobotSample.__dataclass_fields__

    gripper = parse_telemetry(_telemetry_frame(3, 4, values, flags=ROBOT_TELEMETRY_FLAG_RESETTING))
    assert isinstance(gripper, GripperTelemetry)


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
    assert decoded["protocol_version"] == PROTOCOL_VERSION == 2


def test_uds_v2_resetting_messages_and_v1_rejection():
    for status_code in (0, 1, 2):
        packet = parse_packet(make_packet("ROBOT_RESETTING", 2, status_code=status_code))
        assert packet["status_code"] == status_code
    assert parse_packet(make_packet("GET_ROBOT_RESETTING", 3))["type"] == "GET_ROBOT_RESETTING"
    with pytest.raises(ValueError, match="status_code"):
        make_packet("ROBOT_RESETTING", 4, status_code=3)

    legacy = json.loads(make_packet("PING", 5))
    legacy["protocol_version"] = 1
    with pytest.raises(ValueError, match="unsupported SensorHub UDS"):
        parse_packet(json.dumps(legacy).encode())


def test_uds_reset_state_query_and_transitions_use_ordered_server_sequence(tmp_path):
    path = tmp_path / "sensorhub.sock"
    server = UDSControlServer(str(path), Event())
    client = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    server.start()
    try:
        client.settimeout(1)
        client.connect(str(path))
        client.send(make_packet("GET_ROBOT_RESETTING", 50))
        unknown = parse_packet(client.recv(MAX_PACKET_SIZE + 1))
        assert unknown["status_code"] == 2

        server.set_robot_resetting(False)
        idle = parse_packet(client.recv(MAX_PACKET_SIZE + 1))
        server.set_robot_resetting(True)
        resetting = parse_packet(client.recv(MAX_PACKET_SIZE + 1))
        server.set_robot_resetting(False)
        complete = parse_packet(client.recv(MAX_PACKET_SIZE + 1))

        statuses = [unknown, idle, resetting, complete]
        assert [packet["status_code"] for packet in statuses] == [2, 0, 1, 0]
        assert [packet["sequence"] for packet in statuses] == sorted(
            packet["sequence"] for packet in statuses
        )
        assert unknown["sequence"] != 50
    finally:
        client.close()
        server.close()


def test_uds_concurrent_publications_allocate_unique_server_sequences(tmp_path):
    path = tmp_path / "sensorhub_concurrent.sock"
    server = UDSControlServer(str(path), Event())
    client = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    server.start()
    try:
        client.settimeout(1)
        client.connect(str(path))
        client.send(make_packet("GET_ROBOT_RESETTING", 1))
        parse_packet(client.recv(MAX_PACKET_SIZE + 1))
        threads = [
            Thread(target=server.publish, args=("HEALTH",), kwargs={"message": str(i)})
            for i in range(8)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        packets = [parse_packet(client.recv(MAX_PACKET_SIZE + 1)) for _ in threads]
        sequences = [packet["sequence"] for packet in packets]
        assert sequences == sorted(sequences)
        assert len(set(sequences)) == len(sequences)
    finally:
        client.close()
        server.close()


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
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            publisher.send(_telemetry_frame(2, 2, values, flags=ROBOT_TELEMETRY_FLAG_RESETTING))
            try:
                sample = reader.read(timeout_s=0.02)
            except (LookupError, TimeoutError):
                continue
            if isinstance(sample, RobotSample):
                break
        assert reader.robot_resetting is True
    finally:
        reader.close()
        publisher.close(linger=0)
        context.term()
