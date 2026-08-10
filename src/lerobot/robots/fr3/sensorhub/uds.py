# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import socket
import time
from contextlib import suppress
from pathlib import Path
from threading import Event, RLock, Thread
from typing import Final

PROTOCOL_VERSION: Final = 2
MAX_PACKET_SIZE: Final = 512
MESSAGE_TYPES: Final = {
    "READY",
    "HEALTH",
    "FATAL",
    "PING",
    "PONG",
    "SHUTDOWN",
    "GET_ROBOT_RESETTING",
    "ROBOT_RESETTING",
}


def make_packet(
    message_type: str,
    sequence: int,
    *,
    status_code: int = 0,
    message: str = "",
    timestamp_ns: int | None = None,
) -> bytes:
    if message_type not in MESSAGE_TYPES:
        raise ValueError(f"unsupported SensorHub UDS message type: {message_type}")
    if message_type == "ROBOT_RESETTING" and status_code not in {0, 1, 2}:
        raise ValueError("ROBOT_RESETTING status_code must be 0, 1, or 2")
    # Diagnostics can originate in arbitrary libraries. Bound them so an unusually
    # long exception can never prevent delivery of a FATAL packet.
    message = message.encode("utf-8")[:256].decode("utf-8", errors="ignore")
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "type": message_type,
        "sequence": sequence,
        "timestamp_ns": time.time_ns() if timestamp_ns is None else timestamp_ns,
        "status_code": status_code,
        "message": message,
    }
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if len(encoded) > MAX_PACKET_SIZE:
        raise ValueError(f"SensorHub UDS packet exceeds {MAX_PACKET_SIZE} bytes")
    return encoded


def parse_packet(packet: bytes) -> dict[str, object]:
    if len(packet) > MAX_PACKET_SIZE:
        raise ValueError("SensorHub UDS packet is too large")
    decoded = json.loads(packet.decode("utf-8"))
    required = {"protocol_version", "type", "sequence", "timestamp_ns", "status_code", "message"}
    if set(decoded) != required:
        raise ValueError(f"SensorHub UDS fields must be exactly {sorted(required)}")
    if decoded["protocol_version"] != PROTOCOL_VERSION or decoded["type"] not in MESSAGE_TYPES:
        raise ValueError("unsupported SensorHub UDS protocol/type")
    if decoded["type"] == "ROBOT_RESETTING" and decoded["status_code"] not in {0, 1, 2}:
        raise ValueError("invalid ROBOT_RESETTING status_code")
    return decoded


class UDSControlServer:
    """Single-client SOCK_SEQPACKET control/status channel."""

    def __init__(self, path: str, shutdown_event: Event):
        self.path = Path(path)
        self.shutdown_event = shutdown_event
        self._sequence = 0
        self._send_lock = RLock()
        self._client: socket.socket | None = None
        self._last_status: bytes | None = None
        self._robot_resetting: bool | None = None
        self._closed = Event()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with suppress(FileNotFoundError):
            self.path.unlink()
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self._server.bind(str(self.path))
        self._server.listen(1)
        self._server.settimeout(0.1)
        self._thread = Thread(target=self._serve, name="UDSControlServer", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def publish(self, message_type: str, *, status_code: int = 0, message: str = "") -> None:
        with self._send_lock:
            self._publish_locked(message_type, status_code=status_code, message=message)

    def _publish_locked(self, message_type: str, *, status_code: int = 0, message: str = "") -> None:
        self._sequence += 1
        packet = make_packet(message_type, self._sequence, status_code=status_code, message=message)
        if message_type in {"READY", "HEALTH", "FATAL"}:
            self._last_status = packet
        if self._client is not None:
            try:
                self._client.send(packet)
            except OSError:
                self._close_client()

    def set_robot_resetting(self, resetting: bool | None) -> None:
        """Publish robot reset-state transitions without exposing them in aligned observations."""

        with self._send_lock:
            if resetting is self._robot_resetting:
                return
            self._robot_resetting = resetting
            self._publish_robot_resetting_locked()

    def _publish_robot_resetting_locked(self) -> None:
        status_code = 2 if self._robot_resetting is None else int(self._robot_resetting)
        message = (
            "unknown"
            if self._robot_resetting is None
            else ("resetting" if self._robot_resetting else "idle")
        )
        self._publish_locked("ROBOT_RESETTING", status_code=status_code, message=message)

    def _serve(self) -> None:
        while not self._closed.is_set():
            with self._send_lock:
                client = self._client
            if client is None:
                try:
                    client, _ = self._server.accept()
                except TimeoutError:
                    continue
                except OSError:
                    break
                client.settimeout(0.1)
                with self._send_lock:
                    self._client = client
                    if self._last_status is not None:
                        try:
                            client.send(self._last_status)
                        except OSError:
                            self._close_client()
                continue
            try:
                packet = client.recv(MAX_PACKET_SIZE + 1)
            except TimeoutError:
                continue
            except OSError:
                self._close_client()
                continue
            if not packet:
                self._close_client()
                continue
            try:
                decoded = parse_packet(packet)
                message_type = decoded["type"]
                if message_type == "PING":
                    self.publish("PONG", message="ok")
                elif message_type == "GET_ROBOT_RESETTING":
                    with self._send_lock:
                        self._publish_robot_resetting_locked()
                elif message_type == "SHUTDOWN":
                    self.publish("HEALTH", message="shutting down")
                    self.shutdown_event.set()
            except (ValueError, json.JSONDecodeError) as exc:
                self.publish("HEALTH", status_code=2, message=f"invalid control packet: {exc}")

    def _close_client(self) -> None:
        with self._send_lock:
            client, self._client = self._client, None
            if client is not None:
                with suppress(OSError):
                    client.close()

    def close(self) -> None:
        self._closed.set()
        self._close_client()
        with suppress(OSError):
            self._server.close()
        if self._thread.is_alive():
            self._thread.join(timeout=0.5)
        with suppress(FileNotFoundError):
            self.path.unlink()


def connect_uds(path: str, timeout_s: float) -> socket.socket:
    deadline = time.monotonic() + timeout_s
    last_error: OSError | None = None
    while time.monotonic() < deadline:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        try:
            client.connect(path)
            client.settimeout(max(0.01, deadline - time.monotonic()))
            return client
        except OSError as exc:
            last_error = exc
            client.close()
            time.sleep(0.02)
    raise TimeoutError(f"SensorHub UDS did not become available at {path}: {last_error}")
