# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import mmap
import os
import struct
import time
from pathlib import Path
from typing import Final

import numpy as np

from ..protocols import GripperTelemetry, RobotTelemetry, parse_telemetry
from .samples import CameraSample, FTSample, GripperSample, RobotSample, XenseSample

RS_MAGIC: Final = b"RSRGBD1\0"
RS_GLOBAL_HEADER_SIZE: Final = 536
RS_SLOT_HEADER_SIZE: Final = 40
RS_SLOT_COUNT: Final = 2
RS_SLOT_HEADER = struct.Struct("<QQqqq")

LOCAL_GLOBAL_HEADER = struct.Struct("<I4x")
XENSE_SLOT_HEADER = struct.Struct("<QQqq")
FT_SLOT_HEADER = struct.Struct("<QQq8x")

XENSE_FORCE_SHAPE: Final = (35, 20, 3)
XENSE_FORCE_BYTES: Final = 35 * 20 * 3 * 8
XENSE_REC_BYTES: Final = 700 * 400 * 3
XENSE_RESULTANT_BYTES: Final = 6 * 8
XENSE_FORCE0_OFFSET: Final = XENSE_REC_BYTES
XENSE_FORCE1_OFFSET: Final = (
    XENSE_REC_BYTES + XENSE_FORCE_BYTES + XENSE_FORCE_BYTES + XENSE_RESULTANT_BYTES + XENSE_REC_BYTES
)
XENSE_PAYLOAD_SIZE: Final = 2 * XENSE_REC_BYTES + 4 * XENSE_FORCE_BYTES + 2 * XENSE_RESULTANT_BYTES
XENSE_SLOT_STRIDE: Final = XENSE_SLOT_HEADER.size + XENSE_PAYLOAD_SIZE
XENSE_TOTAL_SIZE: Final = LOCAL_GLOBAL_HEADER.size + 2 * XENSE_SLOT_STRIDE

FT_WRENCH_SHAPE: Final = (6,)
FT_PAYLOAD_SIZE: Final = 6 * 8
FT_SLOT_STRIDE: Final = FT_SLOT_HEADER.size + FT_PAYLOAD_SIZE
FT_TOTAL_SIZE: Final = LOCAL_GLOBAL_HEADER.size + 2 * FT_SLOT_STRIDE


def _u32(buffer: mmap.mmap, offset: int) -> int:
    return struct.unpack_from("<I", buffer, offset)[0]


def _u64(buffer: mmap.mmap, offset: int) -> int:
    return struct.unpack_from("<Q", buffer, offset)[0]


class RealSenseReader:
    """Strict reader for the existing RealSense RGB/depth SHM ABI v1."""

    def __init__(self, shm_name: str):
        if (
            len(shm_name) <= 1
            or not shm_name.startswith("/")
            or "/" in shm_name[1:]
            or "\0" in shm_name
        ):
            raise ValueError("RealSense SHM name must contain one leading slash and a simple name")
        self.shm_name = shm_name
        self._fd = os.open(Path("/dev/shm") / shm_name[1:], os.O_RDONLY)
        size = os.fstat(self._fd).st_size
        self._mapping = mmap.mmap(self._fd, size, access=mmap.ACCESS_READ)
        self._last_sequence = 0
        try:
            self._validate(size)
        except Exception:
            self.close()
            raise

    def _validate(self, mapped_size: int) -> None:
        mapping = self._mapping
        if mapped_size < RS_GLOBAL_HEADER_SIZE or mapping[:8] != RS_MAGIC:
            raise ValueError(f"{self.shm_name}: invalid RealSense ABI magic/size")
        if _u32(mapping, 8) != 1 or _u32(mapping, 12) != 1:
            raise ValueError(f"{self.shm_name}: RealSense ABI is unsupported or not ready")
        if _u32(mapping, 16) != RS_GLOBAL_HEADER_SIZE or _u32(mapping, 20) != RS_SLOT_HEADER_SIZE:
            raise ValueError(f"{self.shm_name}: RealSense header sizes differ from ABI v1")
        if _u32(mapping, 32) != RS_SLOT_COUNT:
            raise ValueError(f"{self.shm_name}: RealSense slot_count must be 2")
        self.width = _u32(mapping, 36)
        self.height = _u32(mapping, 40)
        self.color_step = _u32(mapping, 48)
        self.depth_step = _u32(mapping, 52)
        self.color_size = _u64(mapping, 56)
        self.depth_size = _u64(mapping, 64)
        self.color_offset = _u64(mapping, 72)
        self.depth_offset = _u64(mapping, 80)
        self.slot_stride = _u64(mapping, 88)
        total_size = _u64(mapping, 24)
        if (self.width, self.height) != (640, 480):
            raise ValueError(f"{self.shm_name}: expected 640x480, got {self.width}x{self.height}")
        if self.color_step != 1920 or self.depth_step != 1280:
            raise ValueError(f"{self.shm_name}: unsupported RGB/depth row step")
        if self.color_size != 921_600 or self.depth_size != 614_400:
            raise ValueError(f"{self.shm_name}: unsupported RGB/depth payload size")
        if self.color_offset != 40 or self.depth_offset != 921_640 or self.slot_stride != 1_536_040:
            raise ValueError(f"{self.shm_name}: inconsistent RealSense slot offsets")
        if total_size != mapped_size or total_size != RS_GLOBAL_HEADER_SIZE + 2 * self.slot_stride:
            raise ValueError(f"{self.shm_name}: inconsistent RealSense mapping size")
        color_encoding = bytes(mapping[160:176]).split(b"\0", 1)[0]
        depth_encoding = bytes(mapping[176:192]).split(b"\0", 1)[0]
        if color_encoding != b"rgb8" or depth_encoding != b"16UC1":
            raise ValueError(f"{self.shm_name}: unsupported RealSense encoding")

    @property
    def latest_sequence(self) -> int:
        return _u64(self._mapping, 448)

    def read(self, timeout_s: float = 0.02) -> CameraSample:
        deadline = time.monotonic() + timeout_s
        while True:
            latest = self.latest_sequence
            if latest and latest != self._last_sequence:
                base = RS_GLOBAL_HEADER_SIZE + (latest % 2) * self.slot_stride
                seq1 = _u64(self._mapping, base)
                if seq1 == 2 * latest:
                    header = bytes(self._mapping[base : base + RS_SLOT_HEADER_SIZE])
                    rgb_start = base + self.color_offset
                    depth_start = base + self.depth_offset
                    rgb = (
                        np.frombuffer(self._mapping[rgb_start : rgb_start + self.color_size], dtype=np.uint8)
                        .reshape(480, 640, 3)
                        .copy()
                    )
                    depth = (
                        np.frombuffer(self._mapping[depth_start : depth_start + self.depth_size], dtype="<u2")
                        .reshape(480, 640, 1)
                        .copy()
                    )
                    seq2 = _u64(self._mapping, base)
                    copied_seq, bundle, stamp_ns, _color_receive, _depth_receive = RS_SLOT_HEADER.unpack(
                        header
                    )
                    if seq1 == seq2 == copied_seq and bundle == latest:
                        self._last_sequence = latest
                        return CameraSample(latest, stamp_ns, time.monotonic_ns(), rgb, depth)
            if time.monotonic() >= deadline:
                raise TimeoutError(f"no coherent RealSense snapshot from {self.shm_name}")
            time.sleep(0.0005)

    def close(self) -> None:
        mapping = getattr(self, "_mapping", None)
        if mapping is not None:
            mapping.close()
            self._mapping = None
        fd = getattr(self, "_fd", None)
        if fd is not None:
            os.close(fd)
            self._fd = None


class _PythonSharedMemoryReader:
    expected_size: int
    slot_stride: int
    slot_header: struct.Struct

    def __init__(self, shm_name: str):
        self.shm_name = shm_name.removeprefix("/")
        if not self.shm_name or "/" in self.shm_name:
            raise ValueError("SHM name must be a simple POSIX SHM name")
        self._fd = os.open(Path("/dev/shm") / self.shm_name, os.O_RDONLY)
        size = os.fstat(self._fd).st_size
        self._mapping = mmap.mmap(self._fd, size, access=mmap.ACCESS_READ)
        self._last_sequence = 0
        if size != self.expected_size:
            self.close()
            raise ValueError(
                f"{shm_name}: mapping size {size} does not match expected ABI size {self.expected_size}"
            )

    def _read_header_and_payload(self, timeout_s: float) -> tuple[tuple[int, ...], bytes]:
        deadline = time.monotonic() + timeout_s
        while True:
            latest_index = LOCAL_GLOBAL_HEADER.unpack_from(self._mapping, 0)[0]
            if latest_index < 2:
                base = LOCAL_GLOBAL_HEADER.size + latest_index * self.slot_stride
                seq1 = struct.unpack_from("<Q", self._mapping, base)[0]
                if seq1 and seq1 % 2 == 0:
                    header_bytes = bytes(self._mapping[base : base + self.slot_header.size])
                    payload = bytes(self._mapping[base + self.slot_header.size : base + self.slot_stride])
                    seq2 = struct.unpack_from("<Q", self._mapping, base)[0]
                    header = self.slot_header.unpack(header_bytes)
                    if seq1 == seq2 == header[0] and seq2 % 2 == 0 and header[1] != self._last_sequence:
                        self._last_sequence = header[1]
                        return header, payload
            if time.monotonic() >= deadline:
                raise TimeoutError(f"no coherent snapshot from {self.shm_name}")
            time.sleep(0.0005)

    def close(self) -> None:
        mapping = getattr(self, "_mapping", None)
        if mapping is not None:
            mapping.close()
            self._mapping = None
        fd = getattr(self, "_fd", None)
        if fd is not None:
            os.close(fd)
            self._fd = None


class XenseReader(_PythonSharedMemoryReader):
    """Reader for the fixed dual-Xense writer layout v2."""

    expected_size = XENSE_TOTAL_SIZE
    slot_stride = XENSE_SLOT_STRIDE
    slot_header = XENSE_SLOT_HEADER

    def read(self, timeout_s: float = 0.02) -> XenseSample:
        (_seq, frame_id, timestamp0_ns, timestamp1_ns), payload = self._read_header_and_payload(timeout_s)
        force0 = (
            np.frombuffer(payload, dtype="<f8", count=35 * 20 * 3, offset=XENSE_FORCE0_OFFSET)
            .reshape(XENSE_FORCE_SHAPE)
            .astype(np.float32)
        )
        force1 = (
            np.frombuffer(payload, dtype="<f8", count=35 * 20 * 3, offset=XENSE_FORCE1_OFFSET)
            .reshape(XENSE_FORCE_SHAPE)
            .astype(np.float32)
        )
        return XenseSample(
            frame_id,
            min(timestamp0_ns, timestamp1_ns),
            time.monotonic_ns(),
            force0,
            force1,
        )


class FT300SReader(_PythonSharedMemoryReader):
    """Reader for the fixed FT300S writer layout v2."""

    expected_size = FT_TOTAL_SIZE
    slot_stride = FT_SLOT_STRIDE
    slot_header = FT_SLOT_HEADER

    def read(self, timeout_s: float = 0.02) -> FTSample:
        (_seq, frame_id, timestamp_ns), payload = self._read_header_and_payload(timeout_s)
        wrench = np.frombuffer(payload, dtype="<f8", count=6).astype(np.float32)
        return FTSample(frame_id, timestamp_ns, time.monotonic_ns(), wrench)


class TelemetryReader:
    """Latest-only subscriber for independent FR3 and gripper FGT1 samples."""

    def __init__(self, endpoint: str):
        try:
            import zmq
        except ImportError as exc:  # pragma: no cover - installation-specific
            raise ImportError("FR3 requires the 'pyzmq-dep' extra") from exc

        self._zmq = zmq
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.SUBSCRIBE, b"")
        # Do not conflate this multiplexed stream: robot and gripper are independent
        # producers and conflation across sources could permanently starve one cache.
        self._socket.setsockopt(zmq.RCVHWM, 100)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(endpoint)
        self._robot_resetting: bool | None = None

    @property
    def robot_resetting(self) -> bool | None:
        return self._robot_resetting

    def read(self, timeout_s: float = 0.02) -> RobotSample | GripperSample:
        if not self._socket.poll(max(1, int(timeout_s * 1000)), self._zmq.POLLIN):
            raise TimeoutError("no FGT1 telemetry frame")
        decoded = parse_telemetry(self._socket.recv())
        if isinstance(decoded, RobotTelemetry):
            self._robot_resetting = decoded.resetting
            return RobotSample(
                decoded.sequence,
                decoded.timestamp_ns,
                time.monotonic_ns(),
                decoded.q,
                decoded.dq,
                decoded.tau_j,
                decoded.O_T_EE,
            )
        if isinstance(decoded, GripperTelemetry):
            return GripperSample(
                decoded.sequence,
                decoded.timestamp_ns,
                time.monotonic_ns(),
                decoded.gpo,
                decoded.gcu,
            )
        raise LookupError("GELLO telemetry is not an FR3 observation source")

    def close(self) -> None:
        self._socket.close(linger=0)
        self._context.term()
