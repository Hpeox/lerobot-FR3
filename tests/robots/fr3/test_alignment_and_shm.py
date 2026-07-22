import mmap
import os
import struct
import time
import uuid
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import numpy as np
import pytest

from lerobot.robots.fr3.sensorhub.aligned_shm import (
    GLOBAL_HEADER_SIZE,
    SLOT_HEADER_SIZE,
    SLOT_STRIDE,
    TOTAL_SIZE,
    AlignedObservationClient,
    AlignedObservationWriter,
)
from lerobot.robots.fr3.sensorhub.cache import CausalAligner, SampleCache
from lerobot.robots.fr3.sensorhub.readers import (
    FT_SLOT_HEADER,
    FT_SLOT_STRIDE,
    FT_TOTAL_SIZE,
    LOCAL_GLOBAL_HEADER,
    RS_GLOBAL_HEADER_SIZE,
    RS_SLOT_HEADER,
    XENSE_FORCE0_OFFSET,
    XENSE_FORCE1_OFFSET,
    XENSE_SLOT_HEADER,
    XENSE_SLOT_STRIDE,
    XENSE_TOTAL_SIZE,
    FT300SReader,
    RealSenseReader,
    XenseReader,
)
from lerobot.robots.fr3.sensorhub.samples import (
    AlignedSample,
    CameraSample,
    FTSample,
    GripperSample,
    RobotSample,
    XenseSample,
)


def camera(sequence, ingest_ns, value=0):
    return CameraSample(
        sequence,
        ingest_ns,
        ingest_ns,
        np.full((480, 640, 3), value, dtype=np.uint8),
        np.full((480, 640, 1), value, dtype=np.uint16),
    )


def test_abi_golden_sizes():
    assert GLOBAL_HEADER_SIZE == 320
    assert SLOT_HEADER_SIZE == 160
    assert SLOT_STRIDE == 6_161_080
    assert TOTAL_SIZE == 12_322_480


def test_causal_alignment_and_duplicate_suppression():
    camera_caches = tuple(SampleCache[CameraSample](0.5) for _ in range(4))
    for index, cache in enumerate(camera_caches):
        cache.append(camera(index + 1, 90_000_000))
        cache.append(camera(index + 11, (100 + index * 5) * 1_000_000))
    xense_cache = SampleCache[XenseSample](0.5)
    ft_cache = SampleCache[FTSample](0.5)
    robot_cache = SampleCache[RobotSample](0.5)
    gripper_cache = SampleCache[GripperSample](0.5)
    field = np.zeros((35, 20, 3), dtype=np.float32)
    xense_cache.append(XenseSample(1, 95, 95_000_000, field, field))
    ft_cache.append(FTSample(1, 95, 95_000_000, np.zeros(6, dtype=np.float32)))
    robot_cache.append(RobotSample(1, 95, 95_000_000, *(np.zeros(7, dtype=np.float32) for _ in range(3))))
    gripper_cache.append(GripperSample(1, 95, 95_000_000, 1, 2))
    aligner = CausalAligner(
        camera_caches,
        xense_cache,
        ft_cache,
        robot_cache,
        gripper_cache,
        camera_max_skew_ms=50,
        required_sample_max_age_ms=100,
    )
    aligned = aligner.select(1, 120_000_000)
    assert aligned is not None
    assert [sample.sequence for sample in aligned.cameras] == [11, 2, 3, 4]
    assert aligner.select(2, 121_000_000) is None


def _aligned_sample(sequence=1):
    now = time.monotonic_ns()
    cameras = tuple(camera(i, now, i) for i in range(1, 5))
    field0 = np.full((35, 20, 3), 1.5, dtype=np.float32)
    field1 = np.full((35, 20, 3), 2.5, dtype=np.float32)
    return AlignedSample(
        sequence,
        time.time_ns(),
        now,
        cameras,
        XenseSample(5, now, now, field0, field1),
        FTSample(6, now, now, np.arange(6, dtype=np.float32)),
        RobotSample(
            7,
            now,
            now,
            np.arange(7, dtype=np.float32),
            np.arange(10, 17, dtype=np.float32),
            np.arange(20, 27, dtype=np.float32),
        ),
        GripperSample(8, now, now, 128, 3),
    )


def test_aligned_shm_roundtrip_and_owned_buffer():
    name = f"fr3_test_{uuid.uuid4().hex}"
    writer = AlignedObservationWriter(name)
    client = None
    try:
        writer.publish(_aligned_sample())
        client = AlignedObservationClient(name)
        first, metadata = client.read(max_age_ms=1000)
        assert metadata.sequence == 1
        assert first["camera.cam4.rgb"][0, 0, 0] == 4
        assert first["camera.cam1.depth"].dtype == np.uint16
        assert first["xense.sensor0.force_field"].shape == (35, 20, 3)
        assert first["xense.sensor0.force_field"][0, 0, 0] == pytest.approx(1.5)
        assert first["gripper.pos"] == pytest.approx(128 / 255)

        writer.publish(_aligned_sample(sequence=2))
        assert first["camera.cam4.rgb"][0, 0, 0] == 4
    finally:
        if client is not None:
            client.close()
        writer.close()


def test_aligned_shm_fatal_and_stale_detection():
    name = f"fr3_test_{uuid.uuid4().hex}"
    writer = AlignedObservationWriter(name)
    client = None
    try:
        sample = _aligned_sample()
        stale = AlignedSample(
            sample.sequence,
            sample.publish_realtime_ns,
            time.monotonic_ns() - 2_000_000_000,
            sample.cameras,
            sample.xense,
            sample.ft,
            sample.robot,
            sample.gripper,
        )
        writer.publish(stale)
        client = AlignedObservationClient(name)
        with pytest.raises(RuntimeError, match="stale"):
            client.read(max_age_ms=100)
        writer.set_fatal("required source stalled", status_code=9)
        with pytest.raises(RuntimeError, match="required source stalled"):
            client.read(max_age_ms=1000)
    finally:
        if client is not None:
            client.close()
        writer.close()


def test_realsense_reader_strict_v1_roundtrip():
    shm_name = f"/realsense_test_{uuid.uuid4().hex}"
    path = Path("/dev/shm") / shm_name[1:]
    slot_stride = 1_536_040
    total_size = RS_GLOBAL_HEADER_SIZE + 2 * slot_stride
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    os.ftruncate(fd, total_size)
    mapping = mmap.mmap(fd, total_size, access=mmap.ACCESS_WRITE)
    reader = None
    try:
        mapping[:] = b"\0" * total_size
        mapping[:8] = b"RSRGBD1\0"
        struct.pack_into("<IIIIQ", mapping, 8, 1, 1, 536, 40, total_size)
        struct.pack_into("<IIII", mapping, 32, 2, 640, 480, 0)
        struct.pack_into("<IIQQQQQ", mapping, 48, 1920, 1280, 921_600, 614_400, 40, 921_640, slot_stride)
        mapping[96:101] = b"cam1\0"
        mapping[160:165] = b"rgb8\0"
        mapping[176:182] = b"16UC1\0"
        struct.pack_into("<Q", mapping, 448, 1)
        base = RS_GLOBAL_HEADER_SIZE + slot_stride
        RS_SLOT_HEADER.pack_into(mapping, base, 2, 1, 123, 124, 125)
        mapping[base + 40 : base + 40 + 921_600] = bytes([7]) * 921_600
        mapping[base + 921_640 : base + 921_640 + 614_400] = np.full(480 * 640, 900, dtype="<u2").tobytes()
        reader = RealSenseReader(shm_name)
        sample = reader.read()
        assert sample.sequence == 1 and sample.source_timestamp_ns == 123
        assert sample.rgb.shape == (480, 640, 3) and sample.rgb[0, 0, 0] == 7
        assert sample.depth.shape == (480, 640, 1) and sample.depth[0, 0, 0] == 900
    finally:
        if reader is not None:
            reader.close()
        mapping.close()
        os.close(fd)
        path.unlink()


def test_xense_and_ft_reader_fixed_offsets():
    xense_name = f"xense_test_{uuid.uuid4().hex}"
    raw = SharedMemory(name=xense_name, create=True, size=XENSE_TOTAL_SIZE)
    try:
        raw.buf[:] = b"\0" * XENSE_TOTAL_SIZE
        LOCAL_GLOBAL_HEADER.pack_into(raw.buf, 0, 1)
        base = LOCAL_GLOBAL_HEADER.size + XENSE_SLOT_STRIDE
        XENSE_SLOT_HEADER.pack_into(raw.buf, base, 2, 9, 100, 101)
        payload = base + XENSE_SLOT_HEADER.size
        np.ndarray((35, 20, 3), dtype="<f8", buffer=raw.buf, offset=payload + XENSE_FORCE0_OFFSET).fill(1.25)
        np.ndarray((35, 20, 3), dtype="<f8", buffer=raw.buf, offset=payload + XENSE_FORCE1_OFFSET).fill(2.25)
        reader = XenseReader(xense_name)
        sample = reader.read()
        assert sample.sequence == 9
        assert sample.force0.dtype == np.float32
        assert sample.force0[0, 0, 0] == pytest.approx(1.25)
        assert sample.force1[0, 0, 0] == pytest.approx(2.25)
        with pytest.raises(TimeoutError):
            reader.read(timeout_s=0.002)
        reader.close()
    finally:
        raw.close()
        raw.unlink()

    ft_name = f"ft_test_{uuid.uuid4().hex}"
    raw = SharedMemory(name=ft_name, create=True, size=FT_TOTAL_SIZE)
    try:
        raw.buf[:] = b"\0" * FT_TOTAL_SIZE
        LOCAL_GLOBAL_HEADER.pack_into(raw.buf, 0, 1)
        base = LOCAL_GLOBAL_HEADER.size + FT_SLOT_STRIDE
        FT_SLOT_HEADER.pack_into(raw.buf, base, 2, 4, 100)
        struct.pack_into("<6d", raw.buf, base + FT_SLOT_HEADER.size, *range(6))
        reader = FT300SReader(ft_name)
        sample = reader.read()
        reader.close()
        np.testing.assert_array_equal(sample.wrench, np.arange(6, dtype=np.float32))
    finally:
        raw.close()
        raw.unlink()
