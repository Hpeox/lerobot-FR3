import mmap
import os
import struct
import time
import uuid
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import numpy as np
import pytest

from lerobot.robots.fr3.sensorhub import aligned_shm as aligned_shm_module
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
    XENSE_FLOAT32_LAYOUT,
    XENSE_FLOAT64_LAYOUT,
    XENSE_SLOT_HEADER,
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


def camera(sequence, ingest_ns, value=0, *, source_ns=None):
    return CameraSample(
        sequence,
        ingest_ns if source_ns is None else source_ns,
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


def _aligner(camera_count=2, *, max_age_ms=100, wait_ms=25):
    camera_caches = tuple(SampleCache[CameraSample](0.5) for _ in range(camera_count))
    return camera_caches, CausalAligner(
        camera_caches,
        *_required_caches(),
        camera_bundle_span_warn_ms=20,
        camera_max_skew_ms=50,
        camera_bundle_wait_ms=wait_ms,
        required_sample_max_age_ms=max_age_ms,
    )


def _append_bootstrap(camera_caches, latest_sources=None):
    if latest_sources is None:
        latest_sources = [100_000_000] * len(camera_caches)
    for index, (cache, latest_source) in enumerate(
        zip(camera_caches, latest_sources, strict=True)
    ):
        cache.append(camera(1, 10_000_000 + index, source_ns=latest_source - 33_000_000))
        cache.append(camera(2, 20_000_000 + index, source_ns=latest_source))


def test_camera_bootstrap_uses_two_frame_bounded_search_and_newer_tie_break():
    camera_caches, aligner = _aligner(camera_count=4)
    _append_bootstrap(
        camera_caches,
        [200_000_000, 201_000_000, 199_000_000, 202_000_000],
    )

    bundle = aligner.initialize_cameras()

    assert bundle is not None
    assert bundle.mode == "bootstrap_nominal"
    assert bundle.source_span_ns == 3_000_000
    assert [sample.sequence for sample in bundle.cameras] == [2, 2, 2, 2]
    assert aligner.camera_commit_count == 1
    with pytest.raises(RuntimeError, match="already initialized"):
        aligner.initialize_cameras()


@pytest.mark.parametrize(
    ("latest_sources", "expected_mode"),
    [
        ([100_000_000, 119_000_000], "bootstrap_nominal"),
        ([100_000_000, 125_000_000], "bootstrap_degraded"),
    ],
)
def test_camera_bootstrap_nominal_and_degraded_hard_gate(latest_sources, expected_mode):
    camera_caches, aligner = _aligner()
    for index, (cache, latest_source) in enumerate(
        zip(camera_caches, latest_sources, strict=True)
    ):
        cache.append(camera(1, 10_000_000 + index, source_ns=latest_source - 100_000_000))
        cache.append(camera(2, 20_000_000 + index, source_ns=latest_source))
    bundle = aligner.initialize_cameras()
    assert bundle is not None
    assert bundle.mode == expected_mode


def test_camera_bootstrap_over_hard_gate_does_not_commit_and_can_retry():
    camera_caches, aligner = _aligner()
    camera_caches[0].append(camera(1, 1, source_ns=0))
    camera_caches[0].append(camera(2, 2, source_ns=10_000_000))
    camera_caches[1].append(camera(1, 3, source_ns=70_000_000))
    camera_caches[1].append(camera(2, 4, source_ns=80_000_000))
    assert aligner.initialize_cameras() is None
    assert not aligner.cameras_initialized
    assert aligner.camera_frontiers is None
    camera_caches[0].append(camera(3, 5, source_ns=90_000_000))
    camera_caches[1].append(camera(3, 6, source_ns=91_000_000))
    assert aligner.initialize_cameras() is not None


@pytest.mark.parametrize("arrival_order", [(0, 1, 2, 3), (2, 0, 3, 1)])
def test_normal_staggered_camera_arrivals_commit_only_one_bundle(arrival_order):
    camera_caches, aligner = _aligner(camera_count=4)
    _append_bootstrap(camera_caches)
    assert aligner.initialize_cameras() is not None
    assert aligner.select(1, 21_000_000) is not None

    publishes = []
    for arrival_index, camera_index in enumerate(arrival_order):
        cache = camera_caches[camera_index]
        cache.append(
            camera(
                3,
                30_000_000 + arrival_index * 2_000_000,
                source_ns=133_000_000 + camera_index,
            )
        )
        publishes.append(
            aligner.select(2 + arrival_index, 30_000_000 + arrival_index * 2_000_000)
        )

    assert sum(sample is not None for sample in publishes) == 1
    assert aligner.camera_commit_count == 2
    assert aligner.select(10, 50_000_000) is None
    assert "unchanged" in aligner.last_rejection_reason


def test_camera_source_time_controls_coherence_and_ingest_only_controls_wait():
    camera_caches, aligner = _aligner()
    _append_bootstrap(camera_caches)
    assert aligner.initialize_cameras() is not None
    assert aligner.select(1, 21_000_000) is not None
    camera_caches[0].append(camera(3, 30_000_000, source_ns=133_000_000))
    camera_caches[1].append(camera(3, 90_000_000, source_ns=134_000_000))
    aligned = aligner.select(2, 90_000_000)
    assert aligned is not None
    assert aligner.last_camera_bundle.source_span_ns == 1_000_000


def test_camera_degraded_candidate_waits_then_commits_after_timeout():
    camera_caches, aligner = _aligner()
    _append_bootstrap(camera_caches)
    assert aligner.initialize_cameras() is not None
    assert aligner.select(1, 21_000_000) is not None
    camera_caches[0].append(camera(3, 30_000_000, source_ns=130_000_000))
    camera_caches[1].append(camera(3, 31_000_000, source_ns=155_000_000))
    assert aligner.select(2, 40_000_000) is None
    assert "awaiting timeout" in aligner.last_rejection_reason
    aligned = aligner.select(3, 56_000_000)
    assert aligned is not None
    assert aligner.last_camera_bundle.degraded
    assert aligner.last_camera_bundle.mode == "degraded_best_effort"


def test_camera_hard_reject_and_stalled_camera_stop_progression():
    camera_caches, aligner = _aligner()
    _append_bootstrap(camera_caches)
    assert aligner.initialize_cameras() is not None
    assert aligner.select(1, 21_000_000) is not None

    camera_caches[1].append(camera(10, 30_000_000, source_ns=133_000_000))
    first = aligner.select(2, 56_000_000)
    assert first is not None
    assert aligner.last_camera_bundle.reused_camera_indices == (1,)

    camera_caches[1].append(camera(20, 60_000_000, source_ns=166_000_000))
    assert aligner.select(3, 90_000_000) is None
    assert "hard gate" in aligner.last_rejection_reason
    assert aligner.camera_commit_count == 2

    camera_caches[0].append(camera(30, 91_000_000, source_ns=167_000_000))
    recovered = aligner.select(4, 91_000_000)
    assert recovered is not None
    assert [sample.sequence for sample in recovered.cameras] == [30, 20]
    assert not aligner.last_camera_bundle.degraded


def test_bounded_fallback_carries_newer_sample_original_wait_age():
    camera_caches, aligner = _aligner()
    _append_bootstrap(camera_caches)
    assert aligner.initialize_cameras() is not None
    assert aligner.select(1, 21_000_000) is not None
    camera_caches[0].append(camera(3, 30_000_000, source_ns=133_000_000))
    camera_caches[0].append(camera(4, 40_000_000, source_ns=166_000_000))
    camera_caches[1].append(camera(3, 31_000_000, source_ns=134_000_000))

    first = aligner.select(2, 41_000_000)
    assert first is not None
    assert [sample.sequence for sample in first.cameras] == [3, 3]
    assert aligner.last_camera_bundle.mode == "fallback_search"
    assert aligner.select(3, 60_000_000) is None
    second = aligner.select(4, 65_000_000)
    assert second is not None
    assert [sample.sequence for sample in second.cameras] == [4, 3]
    assert aligner.last_camera_bundle.round_wait_ns == 25_000_000


def test_camera_progression_does_not_require_increasing_bundle_source_time():
    camera_caches, aligner = _aligner()
    _append_bootstrap(camera_caches)
    assert aligner.initialize_cameras() is not None
    assert aligner.select(1, 21_000_000) is not None
    for index, cache in enumerate(camera_caches):
        cache.append(camera(9 + index, 30_000_000 + index, source_ns=90_000_000 + index))
    assert aligner.select(2, 31_000_000) is not None
    assert aligner.last_camera_bundle.bundle_time_ns < 100_000_000


def test_latest_non_camera_values_are_used_once_per_camera_commit():
    camera_caches, aligner = _aligner()
    _append_bootstrap(camera_caches)
    assert aligner.initialize_cameras() is not None
    field = np.ones((35, 20, 3), dtype=np.float32)
    aligner.xense.append(XenseSample(7, 1, 20_500_000, field, field))
    aligner.ft.append(FTSample(8, 1, 20_500_000, np.ones(6, dtype=np.float32)))
    joints = np.ones(7, dtype=np.float32)
    aligner.robot.append(RobotSample(9, 1, 20_500_000, joints, joints, joints, transform()))
    aligner.gripper.append(GripperSample(10, 1, 20_500_000, 3, 4))
    aligned = aligner.select(1, 21_000_000)
    assert aligned is not None
    assert (aligned.xense.sequence, aligned.ft.sequence, aligned.robot.sequence, aligned.gripper.sequence) == (7, 8, 9, 10)
    aligner.xense.append(XenseSample(11, 1, 21_500_000, field, field))
    aligner.ft.append(FTSample(12, 1, 21_500_000, np.ones(6, dtype=np.float32)))
    aligner.robot.append(RobotSample(13, 1, 21_500_000, joints, joints, joints, transform()))
    aligner.gripper.append(GripperSample(14, 1, 21_500_000, 5, 6))
    assert aligner.select(2, 22_000_000) is None


def test_missing_bootstrap_assembly_does_not_roll_back_camera_frontier():
    camera_caches = tuple(SampleCache[CameraSample](0.5) for _ in range(2))
    empty_xense = SampleCache[XenseSample](0.5)
    _xense, ft, robot, gripper = _required_caches()
    aligner = CausalAligner(
        camera_caches,
        empty_xense,
        ft,
        robot,
        gripper,
        camera_max_skew_ms=50,
        required_sample_max_age_ms=100,
    )
    _append_bootstrap(camera_caches)
    bootstrap = aligner.initialize_cameras()
    assert bootstrap is not None
    frontiers = aligner.camera_frontiers
    assert aligner.select(1, 21_000_000) is None
    assert aligner.last_rejection_reason == "missing required samples: xense"
    assert aligner.camera_frontiers == frontiers
    assert aligner.camera_commit_count == 1

    for index, cache in enumerate(camera_caches):
        cache.append(camera(3, 30_000_000 + index, source_ns=133_000_000 + index))
    assert aligner.select(2, 31_000_000) is None
    assert aligner.camera_commit_count == 2

    field = np.zeros((35, 20, 3), dtype=np.float32)
    empty_xense.append(XenseSample(1, 1, 32_000_000, field, field))
    assert aligner.select(3, 33_000_000) is None
    assert aligner.camera_commit_count == 2

    for index, cache in enumerate(camera_caches):
        cache.append(camera(4, 40_000_000 + index, source_ns=166_000_000 + index))
    aligned = aligner.select(4, 41_000_000)
    assert aligned is not None
    assert aligner.camera_commit_count == 3


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


@pytest.mark.parametrize("camera_count", [1, 2, 4, 5])
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
        assert client.last_read_diagnostics.slot_copy_duration_ns is not None
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
        first_rgb = first["camera.cam1.rgb"]
        assert first_rgb.flags.writeable
        first_rgb[0, 0, 0] = 123

        header = writer.layout.slot_header.unpack_from(
            writer._shm.buf,
            GLOBAL_HEADER_SIZE + writer.layout.slot_stride,
        )
        source_sequences = header[4::2]
        assert source_sequences == (*range(1, camera_count + 1), 5, 6, 7, 8)

        writer.publish(_aligned_sample(sequence=2, camera_count=camera_count))
        assert first_rgb[0, 0, 0] == 123
        writer.publish(_aligned_sample(sequence=3, camera_count=camera_count))
        assert writer.timing_diagnostics.sequence == 3
        assert writer.timing_diagnostics.publish_interval_ns is not None
        assert writer.timing_diagnostics.same_slot_rewrite_interval_ns is not None
    finally:
        if client is not None:
            client.close()
        writer.close()


def test_aligned_shm_reader_retries_when_writer_reuses_slot_during_copy(monkeypatch):
    name = f"fr3_racing_reader_{uuid.uuid4().hex}"
    writer = AlignedObservationWriter(name, camera_count=2)
    client = None
    try:
        writer.publish(_aligned_sample(sequence=1, camera_count=2))
        client = AlignedObservationClient(name)
        original_copy = client._copy_owned_slot
        calls = 0

        def publish_during_first_copy(slot_base):
            nonlocal calls
            calls += 1
            if calls == 1:
                writer.publish(_aligned_sample(sequence=2, camera_count=2))
                writer.publish(_aligned_sample(sequence=3, camera_count=2))
            return original_copy(slot_base)

        monkeypatch.setattr(client, "_copy_owned_slot", publish_during_first_copy)
        _observation, metadata = client.read(timeout_ms=100, max_age_ms=1000)

        assert metadata.sequence == 3
        assert client.last_read_diagnostics.attempts >= 2
        assert client.last_read_diagnostics.retry_counts == {
            "sequence_changed_during_copy": 1
        }
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


def test_aligned_shm_writer_reports_publish_and_same_slot_intervals(monkeypatch):
    name = f"fr3_timing_{uuid.uuid4().hex}"
    ticks = iter((1_000_000_000, 1_033_000_000, 1_066_000_000))
    monkeypatch.setattr(
        aligned_shm_module,
        "time",
        type("FakeTime", (), {"monotonic_ns": staticmethod(lambda: next(ticks))}),
    )
    writer = AlignedObservationWriter(name, camera_count=1)
    try:
        writer.publish(_aligned_sample(sequence=1, camera_count=1))
        writer.publish(_aligned_sample(sequence=2, camera_count=1))
        writer.publish(_aligned_sample(sequence=3, camera_count=1))
        assert writer.timing_diagnostics.publish_interval_ns == 33_000_000
        assert writer.timing_diagnostics.same_slot_rewrite_interval_ns == 66_000_000
    finally:
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
        with pytest.raises(TimeoutError, match="retry_counts"):
            client.read(timeout_ms=2, max_age_ms=1000)
        assert client.last_read_diagnostics.last_sequence == 1
        assert client.last_read_diagnostics.retry_counts["slot_writing"] > 0
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


@pytest.mark.parametrize(
    ("layout", "expected_total_size"),
    [(XENSE_FLOAT64_LAYOUT, 3_494_664), (XENSE_FLOAT32_LAYOUT, 3_427_368)],
)
def test_xense_reader_supported_scalar_layouts(layout, expected_total_size):
    assert layout.total_size == expected_total_size
    xense_name = f"xense_test_{uuid.uuid4().hex}"
    raw = SharedMemory(name=xense_name, create=True, size=layout.total_size)
    try:
        raw.buf[:] = b"\0" * layout.total_size
        LOCAL_GLOBAL_HEADER.pack_into(raw.buf, 0, 1)
        base = LOCAL_GLOBAL_HEADER.size + layout.slot_stride
        XENSE_SLOT_HEADER.pack_into(raw.buf, base, 2, 9, 100, 101)
        payload = base + XENSE_SLOT_HEADER.size
        np.ndarray(
            (35, 20, 3),
            dtype=layout.scalar_dtype,
            buffer=raw.buf,
            offset=payload + layout.force0_offset,
        ).fill(1.25)
        np.ndarray(
            (35, 20, 3),
            dtype=layout.scalar_dtype,
            buffer=raw.buf,
            offset=payload + layout.force1_offset,
        ).fill(2.25)
        reader = XenseReader(xense_name)
        sample = reader.read()
        assert sample.sequence == 9
        assert sample.source_timestamp_ns == 101
        assert sample.force0.dtype == np.float32
        assert sample.force0[0, 0, 0] == pytest.approx(1.25)
        assert sample.force1[0, 0, 0] == pytest.approx(2.25)
        with pytest.raises(TimeoutError):
            reader.read(timeout_s=0.002)
        reader.close()
    finally:
        raw.close()
        raw.unlink()


def test_xense_reader_rejects_unknown_mapping_size():
    xense_name = f"xense_bad_size_{uuid.uuid4().hex}"
    unknown_size = XENSE_FLOAT32_LAYOUT.total_size + 8
    raw = SharedMemory(name=xense_name, create=True, size=unknown_size)
    try:
        with pytest.raises(ValueError, match="does not match supported Xense ABI sizes"):
            XenseReader(xense_name)
    finally:
        raw.close()
        raw.unlink()


def test_ft_reader_fixed_offsets():
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
