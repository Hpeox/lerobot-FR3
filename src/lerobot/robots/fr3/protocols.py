#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import struct
import time
from dataclasses import dataclass
from typing import Final

import numpy as np

COMMAND_MAGIC: Final = b"FRCMD1\0\0"
COMMAND_ABI_VERSION: Final = 1
COMMAND_HEADER_SIZE: Final = 48
COMMAND_TOTAL_SIZE: Final = 112
COMMAND_STRUCT = struct.Struct("<8sIIIIQqq7dB7x")
COMMAND_FLAG_RESET_JOINT: Final = 1 << 0
COMMAND_KNOWN_FLAGS: Final = COMMAND_FLAG_RESET_JOINT

TELEMETRY_MAGIC: Final = b"FGT1"
TELEMETRY_VERSION: Final = 1
TELEMETRY_TOTAL_SIZE: Final = 504
TELEMETRY_STRUCT = struct.Struct("<4sBBHQdQ58dBB6x")
ROBOT_SOURCE: Final = 2
GRIPPER_SOURCE: Final = 3
ROBOT_VALID_MASK: Final = 2
GRIPPER_VALID_MASK: Final = 4
ROBOT_TELEMETRY_FLAG_RESETTING: Final = 1 << 0


@dataclass(frozen=True, slots=True)
class RobotTelemetry:
    sequence: int
    timestamp_ns: int
    q: np.ndarray
    dq: np.ndarray
    tau_j: np.ndarray
    O_T_EE: np.ndarray
    resetting: bool


@dataclass(frozen=True, slots=True)
class GripperTelemetry:
    sequence: int
    timestamp_ns: int
    gpo: int
    gcu: int


def policy_gripper_to_gpo(value: float) -> tuple[float, int]:
    """Clamp a policy gripper value and convert it to the uint8 wire scale."""

    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError("gripper.pos must be a real number, not bool")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("gripper.pos must be finite")
    clipped = min(max(value, 0.0), 1.0)
    return clipped, math.floor(clipped * 255.0 + 0.5)


def pack_command(
    sequence: int,
    joint_targets: list[float] | tuple[float, ...] | np.ndarray,
    gripper_gpo: int,
    *,
    realtime_ns: int | None = None,
    monotonic_ns: int | None = None,
    flags: int = 0,
) -> bytes:
    """Encode one command using the fixed 112-byte FR3 command ABI."""

    joints = tuple(float(value) for value in joint_targets)
    if len(joints) != 7:
        raise ValueError(f"expected 7 joint targets, got {len(joints)}")
    if any(not math.isfinite(value) for value in joints):
        raise ValueError("joint targets must be finite")
    if isinstance(gripper_gpo, bool) or not 0 <= gripper_gpo <= 255:
        raise ValueError("gripper_gpo must be an integer in [0, 255]")
    if isinstance(flags, bool) or not isinstance(flags, int):
        raise TypeError("command flags must be an integer, not bool")
    if flags < 0 or flags & ~COMMAND_KNOWN_FLAGS:
        raise ValueError(f"unsupported command flags: 0x{flags:x}")
    realtime_ns = time.time_ns() if realtime_ns is None else realtime_ns
    monotonic_ns = time.monotonic_ns() if monotonic_ns is None else monotonic_ns
    frame = COMMAND_STRUCT.pack(
        COMMAND_MAGIC,
        COMMAND_ABI_VERSION,
        COMMAND_HEADER_SIZE,
        COMMAND_TOTAL_SIZE,
        flags,
        sequence,
        realtime_ns,
        monotonic_ns,
        *joints,
        gripper_gpo,
    )
    if len(frame) != COMMAND_TOTAL_SIZE:
        raise AssertionError("command ABI size mismatch")
    return frame


def parse_telemetry(frame: bytes) -> RobotTelemetry | GripperTelemetry | None:
    """Decode one FGT1 v1 frame; GELLO frames intentionally return ``None``."""

    if len(frame) != TELEMETRY_TOTAL_SIZE:
        raise ValueError(f"FGT1 frame must be {TELEMETRY_TOTAL_SIZE} bytes, got {len(frame)}")
    unpacked = TELEMETRY_STRUCT.unpack(frame)
    magic, version, source, flags, sequence, stamp, valid_mask = unpacked[:7]
    floats = unpacked[7:65]
    gpo, gcu = unpacked[65:67]
    if magic != TELEMETRY_MAGIC or version != TELEMETRY_VERSION:
        raise ValueError("unsupported telemetry magic/version")
    if not math.isfinite(stamp):
        raise ValueError("telemetry timestamp must be finite")
    timestamp_ns = int(stamp * 1_000_000_000)
    if source == ROBOT_SOURCE:
        if not valid_mask & ROBOT_VALID_MASK:
            raise ValueError("robot telemetry does not set valid_mask bit 2")
        q = np.array(floats[8:15], dtype=np.float32)
        dq = np.array(floats[15:22], dtype=np.float32)
        tau_j = np.array(floats[22:29], dtype=np.float32)
        O_T_EE = np.asarray(floats[36:52], dtype=np.float32).reshape(4, 4, order="F")
        if not (
            np.isfinite(q).all()
            and np.isfinite(dq).all()
            and np.isfinite(tau_j).all()
            and np.isfinite(O_T_EE).all()
        ):
            raise ValueError("robot telemetry contains non-finite q/dq/tau_J/O_T_EE")
        return RobotTelemetry(
            sequence=sequence,
            timestamp_ns=timestamp_ns,
            q=q,
            dq=dq,
            tau_j=tau_j,
            O_T_EE=O_T_EE,
            resetting=bool(flags & ROBOT_TELEMETRY_FLAG_RESETTING),
        )
    if source == GRIPPER_SOURCE:
        if not valid_mask & GRIPPER_VALID_MASK:
            raise ValueError("gripper telemetry does not set valid_mask bit 4")
        return GripperTelemetry(sequence, timestamp_ns, gpo, gcu)
    return None
