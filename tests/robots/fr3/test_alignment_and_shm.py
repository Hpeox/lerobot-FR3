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
    ABI_VERSION,
    GLOBAL_HEADER,
    GLOBAL_HEADER_SIZE,
    MAGIC,
    SLOT_COUNT,
    AlignedObservationClient,
    AlignedObservationWriter,
    aligned_observation_layout,
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


def transform():
    return np.array(
        [
            [0.0, -1.0, 0.0, 0.12],
            [1.0, 0.0, 0.0, -0.34],
            [0.0, 0.0, 1.0, 0.56],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


@pytest.mark.parametrize("camera_count", [1, 2, 4, 5])
def test_abi_golden_sizes(camera_count):
    layout = aligned_observation_layout(camera_count)
    assert GLOBAL_HEADER_SIZE == 320
    assert MAGIC == b"FR3OBS2\0"
    assert ABI_VERSION == 2
    assert layout.slot_header_size == 96 + 16 * camera_count
    assert layout.payload_size == camera_count * (921_600 + 614_400) + 16_984
    assert layout.slot_stride == layout.slot_header_size + layout.payload_size
    assert layout.total_size == GLOBAL_HEADER_SIZE + SLOT_COUNT * layout.slot_stride
    if camera_count == 4:
        assert layout.slot_header_size == 160
        assert layout.slot_stride == 6_161_144
        assert layout.total_size == 12_322_608


def _required_caches():
    xense_cache = SampleCache[XenseSample](0.5)
    ft_cache = SampleCache[FTSample](0.5)
    robot_cache = SampleCache[RobotSample](0.5)
    gripper_cache = SampleCache[GripperSample](0.5)
    field = np.zeros((35, 20, 3), dtype=np.float32)
    xense_cache.append(XenseSample(1, 95, 95_000_000, field, field))
    ft_cache.append(FTSample(1, 95, 95_000_000, np.zeros(6, dtype=np.float32)))
    robot_cache.append(
        RobotSample(
            1,
            95,
            95_000_000,
            *(np.zeros(7, dtype=np.float32) for _ in range(3)),
            np.eye(4, dtype=np.float32),
        )
    )
    gripper_cache.append(GripperSample(1, 95, 95_000_000, 1, 2))
    return xense_cache, ft_cache, robot_cache, gripper_cache


def test_causal_aligner_rejects_zero_cameras():
    with pytest.raises(ValueError, match="at least one"):
        CausalAligner(
            (),
            *_required_caches(),
            camera_max_skew_ms=50,
            required_sample_max_age_ms=100,
        )


@pytest.mark.parametrize("camera_count", [1, 2, 5])
def test_causal_alignment_order_and_duplicate_suppression(camera_count):
    camera_caches = tuple(SampleCache[CameraSample](0.5) for _ in range(camera_count))
    for index, cache in enumerate(camera_caches):
        cache.append(camera(index + 1, 90_000_000))
        cache.append(camera(index + 11, (100 + index * 5) * 1_000_000))
    xense_cache, ft_cache, robot_cache, gripper_cache = _required_caches()
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
    assert [sample.sequence for sample in aligned.cameras] == [11, *range(2, camera_count + 1)]
    np.testing.assert_array_equal(aligned.robot.O_T_EE, np.eye(4, dtype=np.float32))
    assert aligner.select(2, 121_000_000) is None


def test_causal_alignment_preserves_camera_skew_and_required_age_limits():
    camera_caches = tuple(SampleCache[CameraSample](0.5) for _ in range(2))
    camera_caches[0].append(camera(1, 100_000_000))
    camera_caches[1].append(camera(1, 40_000_000))
    camera_caches[1].append(camera(2, 110_000_000))
    aligner = CausalAligner(
        camera_caches,
        *_required_caches(),
        camera_max_skew_ms=50,
        required_sample_max_age_ms=100,
    )
    assert aligner.select(1, 120_000_000) is None

    fresh_cameras = tuple(SampleCache[CameraSample](0.5) for _ in range(2))
    for cache in fresh_cameras:
        cache.append(camera(1, 200_000_000))
    stale_required = _required_caches()
    aligner = CausalAligner(
        fresh_cameras,
        *stale_required,
        camera_max_skew_ms=50,
        required_sample_max_age_ms=100,
    )
    assert aligner.select(1, 210_000_000) is None


def _aligned_sample(sequence=1, camera_count=2):
    now = time.monotonic_ns()
    cameras = tuple(camera(i, now, i) for i in range(1, camera_count + 1))
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
            transform(),
        ),
        GripperSample(8, now, now, 128, 3),
    )


@pytest.mark.parametrize("camera_count", [1, 2, 5])
def test_aligned_shm_roundtrip_and_owned_buffer(camera_count):
    name = f"fr3_test_{uuid.uuid4().hex}"
    writer = AlignedObservationWriter(name, camera_count=camera_count)
    client = None
    try:
        sample = _aligned_sample(camera_count=camera_count)
        writer.publish(sample)
        global_header = GLOBAL_HEADER.unpack_from(writer._shm.buf, 0)
        assert global_header[:11] == (
            MAGIC,
            ABI_VERSION,
            1,
            GLOBAL_HEADER_SIZE,
            writer.layout.slot_header_size,
            writer.layout.total_size,
            SLOT_COUNT,
            640,
            480,
            camera_count,
            writer.layout.slot_stride,
        )
        client = AlignedObservationClient(name)
        first, metadata = client.read(max_age_ms=1000)
        assert metadata.sequence == 1
        assert client.camera_count == camera_count
        assert first[f"camera.cam{camera_count}.rgb"][0, 0, 0] == camera_count
        assert first["camera.cam1.depth"].dtype == np.uint16
        camera_keys = {key for key in first if key.startswith("camera.")}
        assert camera_keys == {
            *(f"camera.cam{i}.rgb" for i in range(1, camera_count + 1)),
            *(f"camera.cam{i}.depth" for i in range(1, camera_count + 1)),
        }
        assert first["xense.sensor0.force_field"].shape == (35, 20, 3)
        assert first["xense.sensor0.force_field"][0, 0, 0] == pytest.approx(1.5)
        assert first["gripper.pos"] == pytest.approx(128 / 255)
        assert first["fr3.O_T_EE"].dtype == np.float32
        np.testing.assert_array_equal(first["fr3.O_T_EE"], transform())

        header = writer.layout.slot_header.unpack_from(
            writer._shm.buf,
            GLOBAL_HEADER_SIZE + writer.layout.slot_stride,
        )
        source_sequences = header[4::2]
        assert source_sequences == (*range(1, camera_count + 1), 5, 6, 7, 8)

        writer.publish(_aligned_sample(sequence=2, camera_count=camera_count))
        assert first[f"camera.cam{camera_count}.rgb"][0, 0, 0] == camera_count
    finally:
        if client is not None:
            client.close()
        writer.close()


def test_aligned_shm_fatal_and_stale_detection():
    name = f"fr3_test_{uuid.uuid4().hex}"
    writer = AlignedObservationWriter(name, camera_count=2)
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


@pytest.mark.parametrize(
    ("offset", "format", "value"),
    [
        (20, "<I", 999),
        (24, "<Q", 999),
        (44, "<I", 0),
        (48, "<Q", 999),
    ],
)
def test_aligned_shm_rejects_inconsistent_header_metadata(offset, format, value):
    name = f"fr3_invalid_{uuid.uuid4().hex}"
    writer = AlignedObservationWriter(name, camera_count=2)
    try:
        writer.publish(_aligned_sample(camera_count=2))
        struct.pack_into(format, writer._shm.buf, offset, value)
        with pytest.raises(ValueError):
            AlignedObservationClient(name)
    finally:
        writer.close()


def test_aligned_shm_rejects_mapping_smaller_than_declared_layout():
    name = f"fr3_short_{uuid.uuid4().hex}"
    raw = SharedMemory(name=name, create=True, size=GLOBAL_HEADER_SIZE)
    layout = aligned_observation_layout(1)
    try:
        GLOBAL_HEADER.pack_into(
            raw.buf,
            0,
            MAGIC,
            ABI_VERSION,
            1,
            GLOBAL_HEADER_SIZE,
            layout.slot_header_size,
            layout.total_size,
            SLOT_COUNT,
            640,
            480,
            1,
            layout.slot_stride,
            0,
            0,
            0,
            b"\0" * 248,
        )
        with pytest.raises(ValueError, match="metadata mismatch"):
            AlignedObservationClient(name)
    finally:
        raw.close()
        raw.unlink()


def test_aligned_shm_rejects_camera_count_mismatch_and_incoherent_slot():
    name = f"fr3_mismatch_{uuid.uuid4().hex}"
    writer = AlignedObservationWriter(name, camera_count=2)
    client = None
    try:
        with pytest.raises(ValueError, match="expected 2"):
            writer.publish(_aligned_sample(camera_count=1))
        writer.publish(_aligned_sample(camera_count=2))
        client = AlignedObservationClient(name)
        slot_base = GLOBAL_HEADER_SIZE + writer.layout.slot_stride
        struct.pack_into("<Q", writer._shm.buf, slot_base, 1)
        with pytest.raises(TimeoutError):
            client.read(timeout_ms=2, max_age_ms=1000)
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
