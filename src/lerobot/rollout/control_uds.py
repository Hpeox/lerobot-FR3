# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-threaded UDS transport for externally controlled rollouts."""

from __future__ import annotations

import json
import socket
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Final

PROTOCOL_VERSION: Final = 1
MAX_PACKET_SIZE: Final = 1024

OPERATIONS: Final = {
    "INITIALIZE",
    "START",
    "STOP",
    "ABORT",
    "SHUTDOWN",
    "FAIL_STOP",
}


class ControlledUDSError(RuntimeError):
    """Base class for Controlled rollout transport failures."""


class ControlledUDSDisconnected(ControlledUDSError):
    """Raised when the controller disconnects or the UDS transport fails."""


class ControlledUDSProtocolError(ControlledUDSError):
    """Raised when a command packet cannot be parsed."""


@dataclass(frozen=True)
class ControlledCommand:
    """One parsed application-level command."""

    sequence: int
    operation: str


def make_command(sequence: int, operation: str) -> bytes:
    """Build a command packet, primarily for clients and component tests."""

    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        raise ValueError("Controlled command sequence must be a non-negative integer")
    if not isinstance(operation, str) or not operation:
        raise ValueError("Controlled command operation must be a non-empty string")
    return _encode(
        {
            "protocol_version": PROTOCOL_VERSION,
            "type": "COMMAND",
            "sequence": sequence,
            "operation": operation,
        }
    )


def parse_command(packet: bytes) -> ControlledCommand:
    """Parse one strict command packet without applying phase validation."""

    if len(packet) > MAX_PACKET_SIZE:
        raise ControlledUDSProtocolError("Controlled UDS packet is too large")
    try:
        decoded = json.loads(packet.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ControlledUDSProtocolError(f"invalid Controlled UDS JSON: {exc}") from exc
    required = {"protocol_version", "type", "sequence", "operation"}
    if not isinstance(decoded, dict) or set(decoded) != required:
        raise ControlledUDSProtocolError(f"Controlled command fields must be exactly {sorted(required)}")
    if decoded["protocol_version"] != PROTOCOL_VERSION or decoded["type"] != "COMMAND":
        raise ControlledUDSProtocolError("unsupported Controlled UDS protocol/type")
    sequence = decoded["sequence"]
    operation = decoded["operation"]
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        raise ControlledUDSProtocolError("Controlled command sequence must be a non-negative integer")
    if not isinstance(operation, str) or not operation:
        raise ControlledUDSProtocolError("Controlled command operation must be a non-empty string")
    return ControlledCommand(sequence=sequence, operation=operation)


def parse_response(packet: bytes) -> dict[str, object]:
    """Parse an ACK or STATUS packet for clients and component tests."""

    if len(packet) > MAX_PACKET_SIZE:
        raise ControlledUDSProtocolError("Controlled UDS response is too large")
    try:
        decoded = json.loads(packet.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ControlledUDSProtocolError(f"invalid Controlled UDS JSON: {exc}") from exc
    if not isinstance(decoded, dict) or decoded.get("protocol_version") != PROTOCOL_VERSION:
        raise ControlledUDSProtocolError("unsupported Controlled UDS response")
    if decoded.get("type") not in {"ACK", "STATUS"}:
        raise ControlledUDSProtocolError("Controlled UDS response must be ACK or STATUS")
    return decoded


def _encode(payload: dict[str, object]) -> bytes:
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if len(encoded) > MAX_PACKET_SIZE:
        raise ValueError(f"Controlled UDS packet exceeds {MAX_PACKET_SIZE} bytes")
    return encoded


class ControlledUDSServer:
    """One-client, single-threaded ``SOCK_SEQPACKET`` server.

    The strategy itself owns receive timing. In particular, :meth:`drain`
    discards transport backlog without parsing it, so old-phase input can
    never acquire a new meaning after a phase transition.
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with suppress(FileNotFoundError):
            self.path.unlink()
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self._server.bind(str(self.path))
        self._server.listen(1)
        self._client: socket.socket | None = None
        self._last_command_sequence = -1
        self._status_sequence = 0
        self._closed = False

    @property
    def connected(self) -> bool:
        return self._client is not None

    def accept(self) -> None:
        """Block until the single controller client connects."""

        if self._closed:
            raise ControlledUDSDisconnected("Controlled UDS server is closed")
        try:
            client, _ = self._server.accept()
        except OSError as exc:
            raise ControlledUDSDisconnected(f"Controlled UDS accept failed: {exc}") from exc
        self._client = client

    def recv(self, *, blocking: bool) -> ControlledCommand | None:
        """Receive and parse one command, or return ``None`` for an empty nonblocking poll."""

        client = self._require_client()
        client.setblocking(blocking)
        try:
            packet = client.recv(MAX_PACKET_SIZE + 1)
        except BlockingIOError:
            return None
        except OSError as exc:
            raise ControlledUDSDisconnected(f"Controlled UDS receive failed: {exc}") from exc
        if not packet:
            raise ControlledUDSDisconnected("Controlled UDS controller disconnected")
        return parse_command(packet)

    def drain(self) -> int:
        """Blindly discard all currently buffered packets until ``EAGAIN``.

        Packets are intentionally neither parsed nor acknowledged. EOF and
        transport failures remain fatal.
        """

        client = self._require_client()
        client.setblocking(False)
        discarded = 0
        while True:
            try:
                packet = client.recv(MAX_PACKET_SIZE + 1)
            except BlockingIOError:
                return discarded
            except OSError as exc:
                raise ControlledUDSDisconnected(f"Controlled UDS drain failed: {exc}") from exc
            if not packet:
                raise ControlledUDSDisconnected("Controlled UDS controller disconnected during drain")
            discarded += 1

    def consume_sequence(self, sequence: int) -> bool:
        """Apply normal strictly-increasing command sequence validation."""

        if sequence <= self._last_command_sequence:
            return False
        self._last_command_sequence = sequence
        return True

    def send_ack(
        self,
        command: ControlledCommand,
        *,
        accepted: bool,
        code: str,
        phase: str,
        message: str = "",
    ) -> None:
        """Send application-level acceptance or rejection for one parsed command."""

        self._send(
            {
                "protocol_version": PROTOCOL_VERSION,
                "type": "ACK",
                "sequence": command.sequence,
                "operation": command.operation,
                "accepted": accepted,
                "code": code,
                "phase": phase,
                "message": message,
            }
        )

    def publish_status(self, status: str, *, phase: str, code: str = "ok", message: str = "") -> None:
        """Publish a worker lifecycle transition or operation outcome."""

        self._status_sequence += 1
        self._send(
            {
                "protocol_version": PROTOCOL_VERSION,
                "type": "STATUS",
                "sequence": self._status_sequence,
                "status": status,
                "phase": phase,
                "code": code,
                "message": message,
                "timestamp_ns": time.time_ns(),
            }
        )

    def _send(self, payload: dict[str, object]) -> None:
        client = self._require_client()
        try:
            client.sendall(_encode(payload))
        except OSError as exc:
            raise ControlledUDSDisconnected(f"Controlled UDS send failed: {exc}") from exc

    def _require_client(self) -> socket.socket:
        if self._client is None:
            raise ControlledUDSDisconnected("Controlled UDS controller is not connected")
        return self._client

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        client, self._client = self._client, None
        if client is not None:
            with suppress(OSError):
                client.close()
        with suppress(OSError):
            self._server.close()
        with suppress(FileNotFoundError):
            self.path.unlink()

