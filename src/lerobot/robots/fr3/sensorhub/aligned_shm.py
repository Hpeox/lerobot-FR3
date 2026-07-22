# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import mmap
import os
import struct
import time
from contextlib import suppress
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Final

import numpy as np

from .samples import AlignedSample

MAGIC: Final = b"FR3OBS1\0"
ABI_VERSION: Final = 1
GLOBAL_HEADER_SIZE: Final = 320
SLOT_HEADER_SIZE: Final = 160
SLOT_COUNT: Final = 2
WIDTH: Final = 640
HEIGHT: Final = 480

GLOBAL_HEADER = struct.Struct("<8sIIIIQIIIIQQII248s")
SLOT_HEADER = struct.Struct("<QQqq" + "Qq" * 8)

RGB_SHAPE: Final = (480, 640, 3)
DEPTH_SHAPE: Final = (480, 640, 1)
XENSE_SHAPE: Final = (35, 20, 3)
RGB_BYTES: Final = 921_600
DEPTH_BYTES: Final = 614_400
XENSE_BYTES: Final = 8_400

RGB_OFFSETS: Final = tuple(i * RGB_BYTES for i in range(4))
DEPTH_OFFSETS: Final = tuple(4 * RGB_BYTES + i * DEPTH_BYTES for i in range(4))
XENSE0_OFFSET: Final = 4 * RGB_BYTES + 4 * DEPTH_BYTES
XENSE1_OFFSET: Final = XENSE0_OFFSET + XENSE_BYTES
FT_OFFSET: Final = XENSE1_OFFSET + XENSE_BYTES
Q_OFFSET: Final = FT_OFFSET + 6 * 4
DQ_OFFSET: Final = Q_OFFSET + 7 * 4
TAU_OFFSET: Final = DQ_OFFSET + 7 * 4
GRIPPER_POS_OFFSET: Final = TAU_OFFSET + 7 * 4
GPO_OFFSET: Final = GRIPPER_POS_OFFSET + 4
GCU_OFFSET: Final = GPO_OFFSET + 1
PAYLOAD_SIZE: Final = 6_160_920
SLOT_STRIDE: Final = SLOT_HEADER_SIZE + PAYLOAD_SIZE
TOTAL_SIZE: Final = GLOBAL_HEADER_SIZE + SLOT_COUNT * SLOT_STRIDE

assert GLOBAL_HEADER.size == GLOBAL_HEADER_SIZE
assert SLOT_HEADER.size == SLOT_HEADER_SIZE
assert GCU_OFFSET + 1 == PAYLOAD_SIZE - 6
assert SLOT_STRIDE == 6_161_080
assert TOTAL_SIZE == 12_322_480


def _fixed_message(message: str) -> bytes:
    encoded = message.encode("utf-8")
    if len(encoded) >= 248:
        encoded = encoded[:247]
    return encoded + b"\0" * (248 - len(encoded))


def _shm_path(name: str) -> Path:
    clean = name.removeprefix("/")
    if not clean or "/" in clean:
        raise ValueError("aligned observation SHM name must be a simple POSIX SHM name")
    return Path("/dev/shm") / clean


@dataclass(frozen=True, slots=True)
class SnapshotMetadata:
    sequence: int
    publish_realtime_ns: int
    publish_monotonic_ns: int


class AlignedObservationWriter:
    """Owner/writer for the fixed AlignedObservation SHM ABI v1."""

    def __init__(self, shm_name: str):
        self.shm_name = shm_name.removeprefix("/")
        try:
            stale = SharedMemory(name=self.shm_name, create=False)
        except FileNotFoundError:
            pass
        else:
            stale.close()
            stale.unlink()
        self._shm = SharedMemory(name=self.shm_name, create=True, size=TOTAL_SIZE)
        self._shm.buf[:] = b"\0" * TOTAL_SIZE
        self._write_global(ready=0, latest_sequence=0, fatal=0, status_code=0, message="")

    def _write_global(
        self,
        *,
        ready: int,
        latest_sequence: int,
        fatal: int,
        status_code: int,
        message: str,
    ) -> None:
        GLOBAL_HEADER.pack_into(
            self._shm.buf,
            0,
            MAGIC,
            ABI_VERSION,
            ready,
            GLOBAL_HEADER_SIZE,
            SLOT_HEADER_SIZE,
            TOTAL_SIZE,
            SLOT_COUNT,
            WIDTH,
            HEIGHT,
            0,
            SLOT_STRIDE,
            latest_sequence,
            fatal,
            status_code,
            _fixed_message(message),
        )

    def publish(self, sample: AlignedSample) -> None:
        sequence = sample.sequence
        slot_base = GLOBAL_HEADER_SIZE + (sequence % SLOT_COUNT) * SLOT_STRIDE
        payload_base = slot_base + SLOT_HEADER_SIZE
        source_pairs: list[int] = []
        for camera in sample.cameras:
            source_pairs.extend((camera.sequence, camera.source_timestamp_ns))
        source_pairs.extend((sample.xense.sequence, sample.xense.source_timestamp_ns))
        source_pairs.extend((sample.ft.sequence, sample.ft.source_timestamp_ns))
        source_pairs.extend((sample.robot.sequence, sample.robot.source_timestamp_ns))
        source_pairs.extend((sample.gripper.sequence, sample.gripper.source_timestamp_ns))

        writing_sequence = 2 * sequence - 1
        SLOT_HEADER.pack_into(
            self._shm.buf,
            slot_base,
            writing_sequence,
            sequence,
            sample.publish_realtime_ns,
            sample.publish_monotonic_ns,
            *source_pairs,
        )
        payload = self._shm.buf[payload_base : payload_base + PAYLOAD_SIZE]
        for offset, camera in zip(RGB_OFFSETS, sample.cameras, strict=True):
            self._copy_array(payload, offset, camera.rgb, np.uint8, RGB_SHAPE)
        for offset, camera in zip(DEPTH_OFFSETS, sample.cameras, strict=True):
            self._copy_array(payload, offset, camera.depth, np.uint16, DEPTH_SHAPE)
        self._copy_array(payload, XENSE0_OFFSET, sample.xense.force0, np.float32, XENSE_SHAPE)
        self._copy_array(payload, XENSE1_OFFSET, sample.xense.force1, np.float32, XENSE_SHAPE)
        self._copy_array(payload, FT_OFFSET, sample.ft.wrench, np.float32, (6,))
        self._copy_array(payload, Q_OFFSET, sample.robot.q, np.float32, (7,))
        self._copy_array(payload, DQ_OFFSET, sample.robot.dq, np.float32, (7,))
        self._copy_array(payload, TAU_OFFSET, sample.robot.tau_j, np.float32, (7,))
        struct.pack_into(
            "<fBB6x",
            payload,
            GRIPPER_POS_OFFSET,
            sample.gripper.gpo / 255.0,
            sample.gripper.gpo,
            sample.gripper.gcu,
        )
        del payload

        struct.pack_into("<Q", self._shm.buf, slot_base, 2 * sequence)
        struct.pack_into("<Q", self._shm.buf, 56, sequence)
        if sequence == 1:
            struct.pack_into("<I", self._shm.buf, 12, 1)

    @staticmethod
    def _copy_array(
        destination: memoryview,
        offset: int,
        value: np.ndarray,
        dtype: np.dtype,
        shape: tuple[int, ...],
    ) -> None:
        array = np.asarray(value, dtype=dtype)
        if array.shape != shape:
            raise ValueError(f"aligned payload shape {array.shape} does not match {shape}")
        raw = np.ascontiguousarray(array).view(np.uint8).reshape(-1)
        destination[offset : offset + raw.nbytes] = raw

    def set_fatal(self, message: str, status_code: int = 1) -> None:
        latest = struct.unpack_from("<Q", self._shm.buf, 56)[0]
        ready = struct.unpack_from("<I", self._shm.buf, 12)[0]
        self._write_global(
            ready=ready,
            latest_sequence=latest,
            fatal=1,
            status_code=status_code,
            message=message,
        )

    def close(self, unlink: bool = True) -> None:
        shm = getattr(self, "_shm", None)
        if shm is None:
            return
        shm.close()
        if unlink:
            with suppress(FileNotFoundError):
                shm.unlink()
        self._shm = None


class AlignedObservationClient:
    """Read coherent owned snapshots from the SensorHub output mapping."""

    def __init__(self, shm_name: str):
        self.shm_name = shm_name
        path = _shm_path(shm_name)
        self._fd = os.open(path, os.O_RDONLY)
        size = os.fstat(self._fd).st_size
        self._mapping = mmap.mmap(self._fd, size, access=mmap.ACCESS_READ)
        try:
            self._validate(size)
        except Exception:
            self.close()
            raise

    def _validate(self, size: int) -> None:
        header = GLOBAL_HEADER.unpack_from(self._mapping, 0)
        (
            magic,
            version,
            ready,
            global_size,
            slot_size,
            total_size,
            slot_count,
            width,
            height,
            reserved,
            slot_stride,
            *_rest,
        ) = header
        if magic != MAGIC or version != ABI_VERSION or ready != 1:
            raise ValueError("AlignedObservation SHM is unsupported or not ready")
        expected = (
            GLOBAL_HEADER_SIZE,
            SLOT_HEADER_SIZE,
            TOTAL_SIZE,
            SLOT_COUNT,
            WIDTH,
            HEIGHT,
            0,
            SLOT_STRIDE,
        )
        actual = (global_size, slot_size, total_size, slot_count, width, height, reserved, slot_stride)
        if actual != expected or size != TOTAL_SIZE:
            raise ValueError(f"AlignedObservation SHM metadata mismatch: {actual}")

    def fatal_message(self) -> str | None:
        fatal = struct.unpack_from("<I", self._mapping, 64)[0]
        if not fatal:
            return None
        status = struct.unpack_from("<I", self._mapping, 68)[0]
        raw = bytes(self._mapping[72:320]).split(b"\0", 1)[0]
        return f"SensorHub fatal ({status}): {raw.decode('utf-8', errors='replace')}"

    def read(self, timeout_ms: int = 20, max_age_ms: int = 100) -> tuple[dict, SnapshotMetadata]:
        deadline = time.monotonic() + timeout_ms / 1000.0
        while True:
            fatal = self.fatal_message()
            if fatal:
                raise RuntimeError(fatal)
            latest = struct.unpack_from("<Q", self._mapping, 56)[0]
            if latest:
                slot_base = GLOBAL_HEADER_SIZE + (latest % SLOT_COUNT) * SLOT_STRIDE
                seq1 = struct.unpack_from("<Q", self._mapping, slot_base)[0]
                if seq1 == 2 * latest:
                    owned = bytearray(self._mapping[slot_base : slot_base + SLOT_STRIDE])
                    seq2 = struct.unpack_from("<Q", self._mapping, slot_base)[0]
                    header = SLOT_HEADER.unpack_from(owned, 0)
                    if seq1 == seq2 == header[0] and header[1] == latest:
                        metadata = SnapshotMetadata(latest, header[2], header[3])
                        age_ns = time.monotonic_ns() - metadata.publish_monotonic_ns
                        if age_ns > max_age_ms * 1_000_000:
                            raise RuntimeError(
                                f"AlignedObservation snapshot is stale ({age_ns / 1e6:.1f} ms)"
                            )
                        return self._decode(owned), metadata
            if time.monotonic() >= deadline:
                raise TimeoutError("no coherent AlignedObservation snapshot")
            time.sleep(0.0005)

    @staticmethod
    def _decode(slot: bytearray) -> dict:
        payload = memoryview(slot)[SLOT_HEADER_SIZE:]
        observation: dict[str, object] = {}
        for i, offset in enumerate(RGB_OFFSETS, 1):
            observation[f"camera.cam{i}.rgb"] = np.frombuffer(
                payload, dtype=np.uint8, count=480 * 640 * 3, offset=offset
            ).reshape(RGB_SHAPE)
        for i, offset in enumerate(DEPTH_OFFSETS, 1):
            observation[f"camera.cam{i}.depth"] = np.frombuffer(
                payload, dtype="<u2", count=480 * 640, offset=offset
            ).reshape(DEPTH_SHAPE)
        observation["xense.sensor0.force_field"] = np.frombuffer(
            payload, dtype="<f4", count=35 * 20 * 3, offset=XENSE0_OFFSET
        ).reshape(XENSE_SHAPE)
        observation["xense.sensor1.force_field"] = np.frombuffer(
            payload, dtype="<f4", count=35 * 20 * 3, offset=XENSE1_OFFSET
        ).reshape(XENSE_SHAPE)
        observation["ft300s.wrench"] = np.frombuffer(payload, dtype="<f4", count=6, offset=FT_OFFSET)
        q = np.frombuffer(payload, dtype="<f4", count=7, offset=Q_OFFSET)
        observation["fr3.dq"] = np.frombuffer(payload, dtype="<f4", count=7, offset=DQ_OFFSET)
        observation["fr3.tau_J"] = np.frombuffer(payload, dtype="<f4", count=7, offset=TAU_OFFSET)
        for index, value in enumerate(q, 1):
            observation[f"fr3_joint{index}.pos"] = float(value)
        gripper_pos, gpo, gcu = struct.unpack_from("<fBB", payload, GRIPPER_POS_OFFSET)
        observation["gripper.pos"] = float(gripper_pos)
        observation["gripper.gPO"] = np.uint8(gpo)
        observation["gripper.gCU"] = np.uint8(gcu)
        return observation

    def close(self) -> None:
        mapping = getattr(self, "_mapping", None)
        if mapping is not None:
            mapping.close()
            self._mapping = None
        fd = getattr(self, "_fd", None)
        if fd is not None:
            os.close(fd)
            self._fd = None
