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

MAGIC: Final = b"FR3OBS2\0"
ABI_VERSION: Final = 2
GLOBAL_HEADER_SIZE: Final = 320
SLOT_COUNT: Final = 2
WIDTH: Final = 640
HEIGHT: Final = 480

GLOBAL_HEADER = struct.Struct("<8sIIIIQIIIIQQII248s")
SLOT_HEADER_FIXED_FORMAT: Final = "<QQqq"
SOURCE_PAIR_FORMAT: Final = "Qq"
FIXED_NON_CAMERA_SOURCE_COUNT: Final = 4

RGB_SHAPE: Final = (480, 640, 3)
DEPTH_SHAPE: Final = (480, 640, 1)
XENSE_SHAPE: Final = (35, 20, 3)
O_T_EE_SHAPE: Final = (4, 4)
RGB_BYTES: Final = 921_600
DEPTH_BYTES: Final = 614_400
XENSE_BYTES: Final = 8_400
FT_BYTES: Final = 6 * 4
JOINT_VECTOR_BYTES: Final = 7 * 4
O_T_EE_BYTES: Final = 4 * 4 * 4
GRIPPER_STRUCT = struct.Struct("<fBB6x")
FIXED_PAYLOAD_BYTES: Final = (
    2 * XENSE_BYTES + FT_BYTES + 3 * JOINT_VECTOR_BYTES + O_T_EE_BYTES + GRIPPER_STRUCT.size
)

assert GLOBAL_HEADER.size == GLOBAL_HEADER_SIZE
assert FIXED_PAYLOAD_BYTES == 16_984


@dataclass(frozen=True, slots=True)
class AlignedObservationLayout:
    camera_count: int
    slot_header: struct.Struct
    slot_header_size: int
    rgb_offsets: tuple[int, ...]
    depth_offsets: tuple[int, ...]
    xense0_offset: int
    xense1_offset: int
    ft_offset: int
    q_offset: int
    dq_offset: int
    tau_offset: int
    o_t_ee_offset: int
    gripper_pos_offset: int
    gpo_offset: int
    gcu_offset: int
    payload_size: int
    slot_stride: int
    total_size: int


def aligned_observation_layout(camera_count: int) -> AlignedObservationLayout:
    """Compute every FR3OBS2 size and payload offset from the ordered camera count."""

    if isinstance(camera_count, bool) or not isinstance(camera_count, int):
        raise TypeError("camera_count must be an integer, not bool")
    if camera_count <= 0:
        raise ValueError("camera_count must be positive")

    slot_header = struct.Struct(
        SLOT_HEADER_FIXED_FORMAT
        + SOURCE_PAIR_FORMAT * (camera_count + FIXED_NON_CAMERA_SOURCE_COUNT)
    )
    slot_header_size = 96 + 16 * camera_count
    if slot_header.size != slot_header_size:
        raise AssertionError("FR3OBS2 slot header size mismatch")

    rgb_offsets = tuple(index * RGB_BYTES for index in range(camera_count))
    depth_base = camera_count * RGB_BYTES
    depth_offsets = tuple(depth_base + index * DEPTH_BYTES for index in range(camera_count))
    xense0_offset = depth_base + camera_count * DEPTH_BYTES
    xense1_offset = xense0_offset + XENSE_BYTES
    ft_offset = xense1_offset + XENSE_BYTES
    q_offset = ft_offset + FT_BYTES
    dq_offset = q_offset + JOINT_VECTOR_BYTES
    tau_offset = dq_offset + JOINT_VECTOR_BYTES
    o_t_ee_offset = tau_offset + JOINT_VECTOR_BYTES
    gripper_pos_offset = o_t_ee_offset + O_T_EE_BYTES
    gpo_offset = gripper_pos_offset + 4
    gcu_offset = gpo_offset + 1
    payload_size = gripper_pos_offset + GRIPPER_STRUCT.size
    expected_payload_size = camera_count * (RGB_BYTES + DEPTH_BYTES) + FIXED_PAYLOAD_BYTES
    if payload_size != expected_payload_size:
        raise AssertionError("FR3OBS2 payload size mismatch")
    slot_stride = slot_header_size + payload_size
    total_size = GLOBAL_HEADER_SIZE + SLOT_COUNT * slot_stride
    return AlignedObservationLayout(
        camera_count=camera_count,
        slot_header=slot_header,
        slot_header_size=slot_header_size,
        rgb_offsets=rgb_offsets,
        depth_offsets=depth_offsets,
        xense0_offset=xense0_offset,
        xense1_offset=xense1_offset,
        ft_offset=ft_offset,
        q_offset=q_offset,
        dq_offset=dq_offset,
        tau_offset=tau_offset,
        o_t_ee_offset=o_t_ee_offset,
        gripper_pos_offset=gripper_pos_offset,
        gpo_offset=gpo_offset,
        gcu_offset=gcu_offset,
        payload_size=payload_size,
        slot_stride=slot_stride,
        total_size=total_size,
    )


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
    """Owner/writer for the dynamic two-slot AlignedObservation SHM ABI v2."""

    def __init__(self, shm_name: str, *, camera_count: int):
        self.shm_name = shm_name.removeprefix("/")
        self.layout = aligned_observation_layout(camera_count)
        try:
            stale = SharedMemory(name=self.shm_name, create=False)
        except FileNotFoundError:
            pass
        else:
            stale.close()
            stale.unlink()
        self._shm = SharedMemory(name=self.shm_name, create=True, size=self.layout.total_size)
        self._shm.buf[:] = b"\0" * self.layout.total_size
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
            self.layout.slot_header_size,
            self.layout.total_size,
            SLOT_COUNT,
            WIDTH,
            HEIGHT,
            self.layout.camera_count,
            self.layout.slot_stride,
            latest_sequence,
            fatal,
            status_code,
            _fixed_message(message),
        )

    def publish(self, sample: AlignedSample) -> None:
        if len(sample.cameras) != self.layout.camera_count:
            raise ValueError(
                f"aligned sample has {len(sample.cameras)} cameras, expected {self.layout.camera_count}"
            )
        sequence = sample.sequence
        slot_base = GLOBAL_HEADER_SIZE + (sequence % SLOT_COUNT) * self.layout.slot_stride
        payload_base = slot_base + self.layout.slot_header_size
        source_pairs: list[int] = []
        for camera in sample.cameras:
            source_pairs.extend((camera.sequence, camera.source_timestamp_ns))
        source_pairs.extend((sample.xense.sequence, sample.xense.source_timestamp_ns))
        source_pairs.extend((sample.ft.sequence, sample.ft.source_timestamp_ns))
        source_pairs.extend((sample.robot.sequence, sample.robot.source_timestamp_ns))
        source_pairs.extend((sample.gripper.sequence, sample.gripper.source_timestamp_ns))

        writing_sequence = 2 * sequence - 1
        self.layout.slot_header.pack_into(
            self._shm.buf,
            slot_base,
            writing_sequence,
            sequence,
            sample.publish_realtime_ns,
            sample.publish_monotonic_ns,
            *source_pairs,
        )
        payload = self._shm.buf[payload_base : payload_base + self.layout.payload_size]
        for offset, camera in zip(self.layout.rgb_offsets, sample.cameras, strict=True):
            self._copy_array(payload, offset, camera.rgb, np.uint8, RGB_SHAPE)
        for offset, camera in zip(self.layout.depth_offsets, sample.cameras, strict=True):
            self._copy_array(payload, offset, camera.depth, np.uint16, DEPTH_SHAPE)
        self._copy_array(
            payload, self.layout.xense0_offset, sample.xense.force0, np.float32, XENSE_SHAPE
        )
        self._copy_array(
            payload, self.layout.xense1_offset, sample.xense.force1, np.float32, XENSE_SHAPE
        )
        self._copy_array(payload, self.layout.ft_offset, sample.ft.wrench, np.float32, (6,))
        self._copy_array(payload, self.layout.q_offset, sample.robot.q, np.float32, (7,))
        self._copy_array(payload, self.layout.dq_offset, sample.robot.dq, np.float32, (7,))
        self._copy_array(payload, self.layout.tau_offset, sample.robot.tau_j, np.float32, (7,))
        self._copy_array(
            payload, self.layout.o_t_ee_offset, sample.robot.O_T_EE, np.float32, O_T_EE_SHAPE
        )
        GRIPPER_STRUCT.pack_into(
            payload,
            self.layout.gripper_pos_offset,
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
    """Read coherent owned snapshots from a dynamic FR3OBS2 mapping."""

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
        if size < GLOBAL_HEADER_SIZE:
            raise ValueError("AlignedObservation SHM is smaller than its fixed global header")
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
            camera_count,
            slot_stride,
            *_rest,
        ) = header
        if magic != MAGIC or version != ABI_VERSION or ready != 1:
            raise ValueError("AlignedObservation SHM is unsupported or not ready")
        layout = aligned_observation_layout(camera_count)
        expected = (
            GLOBAL_HEADER_SIZE,
            layout.slot_header_size,
            layout.total_size,
            SLOT_COUNT,
            WIDTH,
            HEIGHT,
            layout.camera_count,
            layout.slot_stride,
        )
        actual = (
            global_size,
            slot_size,
            total_size,
            slot_count,
            width,
            height,
            camera_count,
            slot_stride,
        )
        if actual != expected or size != layout.total_size:
            raise ValueError(f"AlignedObservation SHM metadata mismatch: {actual}")
        self.layout = layout
        self.camera_count = camera_count

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
                slot_base = GLOBAL_HEADER_SIZE + (latest % SLOT_COUNT) * self.layout.slot_stride
                seq1 = struct.unpack_from("<Q", self._mapping, slot_base)[0]
                if seq1 == 2 * latest:
                    owned = bytearray(
                        self._mapping[slot_base : slot_base + self.layout.slot_stride]
                    )
                    seq2 = struct.unpack_from("<Q", self._mapping, slot_base)[0]
                    header = self.layout.slot_header.unpack_from(owned, 0)
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

    def _decode(self, slot: bytearray) -> dict:
        payload = memoryview(slot)[self.layout.slot_header_size :]
        observation: dict[str, object] = {}
        for index, offset in enumerate(self.layout.rgb_offsets, 1):
            observation[f"camera.cam{index}.rgb"] = np.frombuffer(
                payload, dtype=np.uint8, count=480 * 640 * 3, offset=offset
            ).reshape(RGB_SHAPE)
        for index, offset in enumerate(self.layout.depth_offsets, 1):
            observation[f"camera.cam{index}.depth"] = np.frombuffer(
                payload, dtype="<u2", count=480 * 640, offset=offset
            ).reshape(DEPTH_SHAPE)
        observation["xense.sensor0.force_field"] = np.frombuffer(
            payload, dtype="<f4", count=35 * 20 * 3, offset=self.layout.xense0_offset
        ).reshape(XENSE_SHAPE)
        observation["xense.sensor1.force_field"] = np.frombuffer(
            payload, dtype="<f4", count=35 * 20 * 3, offset=self.layout.xense1_offset
        ).reshape(XENSE_SHAPE)
        observation["ft300s.wrench"] = np.frombuffer(
            payload, dtype="<f4", count=6, offset=self.layout.ft_offset
        )
        q = np.frombuffer(payload, dtype="<f4", count=7, offset=self.layout.q_offset)
        observation["fr3.dq"] = np.frombuffer(
            payload, dtype="<f4", count=7, offset=self.layout.dq_offset
        )
        observation["fr3.tau_J"] = np.frombuffer(
            payload, dtype="<f4", count=7, offset=self.layout.tau_offset
        )
        observation["fr3.O_T_EE"] = np.frombuffer(
            payload, dtype="<f4", count=16, offset=self.layout.o_t_ee_offset
        ).reshape(O_T_EE_SHAPE)
        for index, value in enumerate(q, 1):
            observation[f"fr3_joint{index}.pos"] = float(value)
        gripper_pos, gpo, gcu = struct.unpack_from(
            "<fBB", payload, self.layout.gripper_pos_offset
        )
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
