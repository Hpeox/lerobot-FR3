import logging
import os
import time
import uuid
from threading import Event, Thread
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot.robots.fr3.sensorhub import __main__ as sensorhub_main
from lerobot.robots.fr3.sensorhub import runtime as runtime_module
from lerobot.robots.fr3.sensorhub.aligned_shm import (
    AlignedObservationClient,
    AlignedObservationWriter,
)
from lerobot.robots.fr3.sensorhub.cache import AlignmentRejection, CausalAligner, SampleCache
from lerobot.robots.fr3.sensorhub.runtime import SensorHubConfig, SensorHubRuntime
from lerobot.robots.fr3.sensorhub.samples import (
    CameraSample,
    FTSample,
    GripperSample,
    RobotSample,
    XenseSample,
)


def _config(tmp_path, shm_name, realsense_shm_names=("/cam1", "/cam2")):
    return SensorHubConfig(
        telemetry_endpoint="inproc://unused",
        observation_shm_name=shm_name,
        sensorhub_socket_path=str(tmp_path / "sensorhub.sock"),
        realsense_shm_names=realsense_shm_names,
        xense_shm_name="xense",
        ft300s_shm_name="ft",
        required_sample_max_age_ms=10,
    )


def _append_round(runtime, sequence, *, robot_sequence=None, gripper_sequence=None):
    now_ns = time.monotonic_ns()
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    depth = np.zeros((480, 640, 1), dtype=np.uint16)
    for cache in runtime.camera_caches:
        cache.append(CameraSample(sequence, now_ns, now_ns, rgb, depth))
    field = np.zeros((35, 20, 3), dtype=np.float32)
    runtime.xense_cache.append(XenseSample(sequence, now_ns, now_ns, field, field))
    runtime.ft_cache.append(FTSample(sequence, now_ns, now_ns, np.zeros(6, dtype=np.float32)))
    if robot_sequence is not None:
        joint = np.zeros(7, dtype=np.float32)
        runtime.robot_cache.append(
            RobotSample(robot_sequence, now_ns, now_ns, joint, joint, joint, np.eye(4, dtype=np.float32))
        )
    if gripper_sequence is not None:
        runtime.gripper_cache.append(GripperSample(gripper_sequence, now_ns, now_ns, 0, 0))


def _wait_for_sequence(client, expected):
    deadline = time.monotonic() + 1
    last_sequence = None
    while time.monotonic() < deadline:
        _observation, metadata = client.read(timeout_ms=10, max_age_ms=10_000)
        last_sequence = metadata.sequence
        if last_sequence == expected:
            return
        time.sleep(0.002)
    raise AssertionError(f"aligned sequence did not reach {expected}; last={last_sequence}")


def test_telemetry_outage_stalls_then_resumes_same_runtime_and_shm(tmp_path):
    shm_name = f"fr3_runtime_{uuid.uuid4().hex}"
    runtime = SensorHubRuntime(_config(tmp_path, shm_name), os.getpid())
    runtime.writer = AlignedObservationWriter(shm_name, camera_count=len(runtime.camera_caches))
    client = None
    alignment_thread = Thread(target=runtime._alignment_loop)
    try:
        _append_round(runtime, 0, robot_sequence=9, gripper_sequence=19)
        _append_round(runtime, 1, robot_sequence=10, gripper_sequence=20)
        assert runtime.aligner.initialize_cameras() is not None
        alignment_thread.start()
        assert runtime.first_publish_event.wait(timeout=1)
        client = AlignedObservationClient(shm_name)
        _wait_for_sequence(client, 1)

        time.sleep(0.02)
        _append_round(runtime, 2)
        time.sleep(0.03)
        _observation, stalled = client.read(timeout_ms=10, max_age_ms=10_000)
        assert stalled.sequence == 1
        assert alignment_thread.is_alive()
        assert not runtime.fatal_event.is_set()

        _append_round(runtime, 3, robot_sequence=1, gripper_sequence=1)
        _wait_for_sequence(client, 2)
        assert runtime.writer is not None
        assert not runtime.fatal_event.is_set()
    finally:
        runtime.stop_event.set()
        alignment_thread.join(timeout=1)
        if client is not None:
            client.close()
        runtime.writer.close()
        runtime.control.close()


def test_alignment_exception_remains_fatal(tmp_path):
    runtime = SensorHubRuntime(_config(tmp_path, f"unused_{uuid.uuid4().hex}"), os.getpid())

    class FailingAligner:
        def select(self, realtime_ns, monotonic_ns):
            raise ValueError("broken invariant")

    runtime.aligner = FailingAligner()
    try:
        runtime._alignment_loop()
        assert runtime.fatal_event.is_set()
        assert runtime.stop_event.is_set()
        assert "broken invariant" in runtime._fatal_message
    finally:
        runtime.control.close()


def test_parent_disappearance_stops_without_fatal(tmp_path):
    runtime = SensorHubRuntime(_config(tmp_path, f"unused_{uuid.uuid4().hex}"), os.getpid())
    runtime._parent_is_alive = lambda: False
    try:
        runtime._supervise()
        assert runtime.stop_event.is_set()
        assert not runtime.fatal_event.is_set()
    finally:
        runtime.control.close()


def test_reader_timeouts_retry_and_unexpected_errors_remain_fatal(tmp_path):
    runtime = SensorHubRuntime(_config(tmp_path, f"unused_{uuid.uuid4().hex}"), os.getpid())
    calls = 0
    appended = []

    def temporarily_empty(*, timeout_s):
        nonlocal calls
        calls += 1
        if calls < 4:
            raise TimeoutError("temporarily empty")
        runtime.stop_event.set()
        return "recovered"

    try:
        runtime._start_reader_thread("FakeReader", temporarily_empty, appended.append)
        runtime._threads[-1].join(timeout=1)
        assert appended == ["recovered"]
        assert not runtime.fatal_event.is_set()
    finally:
        runtime.control.close()

    runtime = SensorHubRuntime(
        _config(tmp_path, f"unused_{uuid.uuid4().hex}"),
        os.getpid(),
    )

    def broken_reader(*, timeout_s):
        raise ValueError("bad source ABI")

    try:
        runtime._start_reader_thread("BrokenReader", broken_reader, appended.append)
        runtime._threads[-1].join(timeout=1)
        assert runtime.fatal_event.is_set()
        assert "bad source ABI" in runtime._fatal_message
    finally:
        runtime.control.close()


def test_fatal_is_published_and_logged_once_across_concurrent_failures(tmp_path, caplog):
    runtime = SensorHubRuntime(_config(tmp_path, f"unused_{uuid.uuid4().hex}"), os.getpid())
    published = []
    runtime.control.publish = lambda *args, **kwargs: published.append((args, kwargs))
    threads = [Thread(target=runtime._fatal, args=(f"failure {index}",)) for index in range(4)]
    try:
        caplog.set_level(logging.ERROR, logger=runtime_module.__name__)
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(published) == 1
        assert sum("SensorHub fatal: failure" in record.message for record in caplog.records) == 1
        assert runtime.fatal_event.is_set()
        assert runtime.stop_event.is_set()
    finally:
        runtime.control.close()


def test_alignment_pending_log_is_reason_change_or_one_second_throttled(caplog):
    runtime = object.__new__(SensorHubRuntime)
    runtime.aligner = SimpleNamespace(last_rejection=AlignmentRejection.ROBOT_MISSING)
    runtime._alignment_pending_rejection = None
    runtime._alignment_pending_log_monotonic = 0.0
    caplog.set_level(logging.WARNING, logger=runtime_module.__name__)

    runtime._log_alignment_pending()
    runtime._log_alignment_pending()
    runtime.aligner.last_rejection = AlignmentRejection.GRIPPER_MISSING
    runtime._log_alignment_pending()
    runtime._alignment_pending_log_monotonic -= 1.0
    runtime._log_alignment_pending()

    messages = [
        record.message
        for record in caplog.records
        if "SensorHub alignment pending before READY" in record.message
    ]
    assert messages == [
        "SensorHub alignment pending before READY: missing required samples: robot",
        "SensorHub alignment pending before READY: missing required samples: gripper",
        "SensorHub alignment pending before READY: missing required samples: gripper",
    ]


def test_runtime_alignment_rejection_logs_only_on_change_and_includes_camera(caplog):
    runtime = object.__new__(SensorHubRuntime)
    runtime.aligner = SimpleNamespace(last_rejection=AlignmentRejection.ROBOT_STALE)
    runtime._last_logged_alignment_rejection = None
    caplog.set_level(logging.WARNING, logger=runtime_module.__name__)

    runtime._log_alignment_rejected()
    runtime._log_alignment_rejected()
    runtime.aligner.last_rejection = AlignmentRejection.GRIPPER_MISSING
    runtime._log_alignment_rejected()
    runtime.aligner.last_rejection = AlignmentRejection.CAMERA_HARD_SKEW
    runtime._log_alignment_rejected()

    messages = [record.message for record in caplog.records if "alignment rejected" in record.message]
    assert messages == [
        "SensorHub alignment rejected: ROBOT_STALE",
        "SensorHub alignment rejected: GRIPPER_MISSING",
        "SensorHub alignment rejected: CAMERA_HARD_SKEW",
    ]


def test_attach_pending_log_is_reason_change_or_one_second_throttled(caplog):
    runtime = object.__new__(SensorHubRuntime)
    runtime._attach_pending_error = None
    runtime._attach_pending_log_monotonic = 0.0
    caplog.set_level(logging.WARNING, logger=runtime_module.__name__)

    runtime._log_attach_pending(FileNotFoundError("missing ft SHM"))
    runtime._log_attach_pending(FileNotFoundError("missing ft SHM"))
    runtime._log_attach_pending(FileNotFoundError("missing xense SHM"))
    runtime._attach_pending_log_monotonic -= 1.0
    runtime._log_attach_pending(FileNotFoundError("missing xense SHM"))

    messages = [
        record.message
        for record in caplog.records
        if "SensorHub attach pending before READY" in record.message
    ]
    assert messages == [
        "SensorHub attach pending before READY: FileNotFoundError: missing ft SHM",
        "SensorHub attach pending before READY: FileNotFoundError: missing xense SHM",
        "SensorHub attach pending before READY: FileNotFoundError: missing xense SHM",
    ]


def test_run_uses_one_startup_deadline_for_attach_and_ready(monkeypatch, tmp_path):
    runtime = object.__new__(SensorHubRuntime)
    runtime.config = _config(tmp_path, f"unused_{uuid.uuid4().hex}")
    runtime.stop_event = Event()
    runtime.fatal_event = Event()
    runtime.ready_event = Event()
    runtime._threads = []
    runtime._readers = []
    runtime.writer = None
    runtime.camera_caches = (SimpleNamespace(append=lambda sample: None),) * 2
    runtime.xense_cache = SimpleNamespace(append=lambda sample: None)
    runtime.ft_cache = SimpleNamespace(append=lambda sample: None)
    deadlines = []

    class FakeReader:
        def read(self, *, timeout_s):
            raise TimeoutError

        def close(self):
            pass

    class FakeWriter:
        def __init__(self, *args, **kwargs):
            pass

        def close(self, *, unlink):
            pass

    readers = tuple(FakeReader() for _ in range(5))
    runtime.control = SimpleNamespace(
        start=lambda: None,
        publish=lambda *args, **kwargs: None,
        close=lambda: None,
    )
    runtime._attach_readers = lambda deadline: deadlines.append(deadline) or (
        readers[:2],
        readers[2],
        readers[3],
        readers[4],
    )
    runtime._start_reader_thread = lambda *args, **kwargs: None
    runtime._start_telemetry_thread = lambda *args, **kwargs: None
    runtime._start_thread = lambda *args, **kwargs: None
    runtime._wait_until_camera_bootstrap = lambda deadline: deadlines.append(deadline)
    runtime._wait_until_ready = lambda deadline: deadlines.append(deadline)
    runtime._supervise = lambda: None
    monkeypatch.setattr(runtime_module, "AlignedObservationWriter", FakeWriter)

    assert runtime.run() == 0
    assert len(deadlines) == 3
    assert deadlines[0] == deadlines[1] == deadlines[2]


def test_alignment_pending_logs_stop_after_first_publish(caplog):
    runtime = object.__new__(SensorHubRuntime)
    runtime.stop_event = Event()
    runtime.first_publish_event = Event()
    runtime.ready_event = Event()
    runtime._alignment_pending_rejection = None
    runtime._alignment_pending_log_monotonic = 0.0
    runtime._last_logged_alignment_rejection = None
    published = []
    runtime.writer = SimpleNamespace(
        publish=published.append,
        timing_diagnostics=SimpleNamespace(publish_duration_ns=0),
    )
    runtime._fatal = lambda message: pytest.fail(message)

    class Aligner:
        last_rejection = AlignmentRejection.ROBOT_MISSING
        calls = 0

        def select(self, realtime_ns, monotonic_ns):
            self.calls += 1
            if self.calls == 1:
                return "first aligned sample"
            runtime.stop_event.set()
            return None

    runtime.aligner = Aligner()
    caplog.set_level(logging.WARNING, logger=runtime_module.__name__)

    runtime._alignment_loop()

    assert published == ["first aligned sample"]
    assert runtime.first_publish_event.is_set()
    assert not any("alignment pending before READY" in record.message for record in caplog.records)


def test_alignment_loop_paces_publications_without_catchup(monkeypatch):
    class FakeClock:
        def __init__(self):
            self.now_ns = 10_000_000_000

        def monotonic_ns(self):
            return self.now_ns

        def monotonic(self):
            return self.now_ns / 1_000_000_000

        def time_ns(self):
            return self.now_ns

        def sleep(self, seconds):
            self.now_ns += round(seconds * 1_000_000_000)

    clock = FakeClock()
    monkeypatch.setattr(runtime_module, "time", clock)

    runtime = object.__new__(SensorHubRuntime)
    runtime.stop_event = Event()
    runtime.first_publish_event = Event()
    runtime.ready_event = Event()
    runtime._alignment_pending_rejection = None
    runtime._alignment_pending_log_monotonic = 0.0
    runtime._last_logged_alignment_rejection = None
    runtime._log_new_camera_bundle = lambda: None
    runtime._fatal = lambda message: pytest.fail(message)

    published_at = []

    class Writer:
        timing_diagnostics = SimpleNamespace(publish_duration_ns=0)

        def publish(self, sample):
            published_at.append(clock.monotonic_ns())
            if len(published_at) == 2:
                clock.now_ns += 100_000_000

    runtime.writer = Writer()

    class Aligner:
        last_rejection = AlignmentRejection.CAMERA_TUPLE_UNCHANGED

        def __init__(self):
            self.samples = iter(("first", "second", "third"))
            self.select_at = []

        def select(self, realtime_ns, monotonic_ns):
            self.select_at.append(monotonic_ns)
            try:
                return next(self.samples)
            except StopIteration:
                runtime.stop_event.set()
                return None

    aligner = Aligner()
    runtime.aligner = aligner

    start_ns = clock.monotonic_ns()
    runtime._alignment_loop()

    assert published_at[0] == start_ns
    assert len(published_at) == 3
    assert all(
        later - earlier >= runtime_module.ALIGNED_PUBLISH_PERIOD_NS
        for earlier, later in zip(published_at, published_at[1:])
    )
    assert published_at[2] - published_at[1] >= 100_000_000 + runtime_module.ALIGNED_PUBLISH_PERIOD_NS
    assert aligner.select_at[:3] == published_at


def test_alignment_recovery_logs_after_successful_publish(monkeypatch, caplog):
    runtime = object.__new__(SensorHubRuntime)
    runtime.stop_event = Event()
    runtime.first_publish_event = Event()
    runtime.ready_event = Event()
    runtime.ready_event.set()
    runtime._last_logged_alignment_rejection = None
    runtime._log_new_camera_bundle = lambda: None
    runtime._fatal = lambda message: pytest.fail(message)

    class Writer:
        timing_diagnostics = SimpleNamespace(publish_duration_ns=0)

        def publish(self, sample):
            pass

    runtime.writer = Writer()

    class Aligner:
        last_rejection = AlignmentRejection.ROBOT_STALE
        calls = 0

        def select(self, realtime_ns, monotonic_ns):
            self.calls += 1
            if self.calls <= 2:
                return None
            if self.calls == 3:
                return "recovered sample"
            runtime.stop_event.set()
            return None

    runtime.aligner = Aligner()
    monkeypatch.setattr(runtime_module, "ALIGNED_PUBLISH_PERIOD_NS", 1_000_000)
    caplog.set_level(logging.INFO, logger=runtime_module.__name__)

    runtime._alignment_loop()

    messages = [record.message for record in caplog.records]
    assert messages.count("SensorHub alignment rejected: ROBOT_STALE") == 2
    assert messages.count("SensorHub alignment recovered") == 1


@pytest.mark.parametrize(
    ("duration_ns", "expects_warning"),
    [(33_333_333, True), (33_333_332, False)],
)
def test_alignment_loop_reports_slow_writer_at_one_period(duration_ns, expects_warning, caplog):
    runtime = object.__new__(SensorHubRuntime)
    runtime.stop_event = Event()
    runtime.first_publish_event = Event()
    runtime.ready_event = Event()
    runtime._last_logged_alignment_rejection = None
    runtime._log_new_camera_bundle = lambda: None
    runtime._fatal = lambda message: pytest.fail(message)

    class Writer:
        timing_diagnostics = SimpleNamespace(publish_duration_ns=duration_ns)

        def publish(self, sample):
            runtime.stop_event.set()

    runtime.writer = Writer()
    runtime.aligner = SimpleNamespace(
        select=lambda realtime_ns, monotonic_ns: "sample",
        last_rejection=None,
    )
    caplog.set_level(logging.WARNING, logger=runtime_module.__name__)

    runtime._alignment_loop()

    warnings = ["aligned writer slow" in record.message for record in caplog.records]
    assert any(warnings) is expects_warning


def test_late_cadence_gate_warns_only_once_during_retries(monkeypatch, caplog):
    class FakeClock:
        def __init__(self):
            self.now_ns = 10_000_000_000
            self.inject_delay = False

        def monotonic_ns(self):
            return self.now_ns

        def time_ns(self):
            return self.now_ns

        def sleep(self, seconds):
            self.now_ns += round(seconds * 1_000_000_000)
            if self.inject_delay:
                self.now_ns += 2 * runtime_module.ALIGNED_PUBLISH_PERIOD_NS
                self.inject_delay = False

    clock = FakeClock()
    monkeypatch.setattr(runtime_module, "time", clock)
    runtime = object.__new__(SensorHubRuntime)
    runtime.stop_event = Event()
    runtime.first_publish_event = Event()
    runtime.ready_event = Event()
    runtime.ready_event.set()
    runtime._last_logged_alignment_rejection = None
    runtime._log_new_camera_bundle = lambda: None
    runtime._fatal = lambda message: pytest.fail(message)

    class Writer:
        timing_diagnostics = SimpleNamespace(publish_duration_ns=0)

        def publish(self, sample):
            clock.inject_delay = True

    runtime.writer = Writer()

    class Aligner:
        last_rejection = AlignmentRejection.ROBOT_STALE
        calls = 0

        def select(self, realtime_ns, monotonic_ns):
            self.calls += 1
            if self.calls == 1:
                return "first"
            if self.calls == 5:
                runtime.stop_event.set()
            return None

    runtime.aligner = Aligner()
    caplog.set_level(logging.WARNING, logger=runtime_module.__name__)

    runtime._alignment_loop()

    messages = [record.message for record in caplog.records if "publisher late" in record.message]
    assert len(messages) == 1


def test_on_time_gate_then_prolonged_rejection_does_not_warn_late(monkeypatch, caplog):
    class FakeClock:
        def __init__(self):
            self.now_ns = 10_000_000_000

        def monotonic_ns(self):
            return self.now_ns

        def time_ns(self):
            return self.now_ns

        def sleep(self, seconds):
            self.now_ns += round(seconds * 1_000_000_000)

    clock = FakeClock()
    monkeypatch.setattr(runtime_module, "time", clock)
    monkeypatch.setattr(runtime_module, "ALIGNED_PUBLISH_PERIOD_NS", 1_000_000)
    runtime = object.__new__(SensorHubRuntime)
    runtime.stop_event = Event()
    runtime.first_publish_event = Event()
    runtime.ready_event = Event()
    runtime.ready_event.set()
    runtime._last_logged_alignment_rejection = None
    runtime._log_new_camera_bundle = lambda: None
    runtime._fatal = lambda message: pytest.fail(message)

    class Writer:
        timing_diagnostics = SimpleNamespace(publish_duration_ns=0)

        def publish(self, sample):
            pass

    runtime.writer = Writer()

    class Aligner:
        last_rejection = AlignmentRejection.ROBOT_STALE
        calls = 0

        def select(self, realtime_ns, monotonic_ns):
            self.calls += 1
            if self.calls == 1:
                return "first"
            if self.calls == 50:
                runtime.stop_event.set()
            return None

    runtime.aligner = Aligner()
    caplog.set_level(logging.WARNING, logger=runtime_module.__name__)

    runtime._alignment_loop()

    assert not any("publisher late" in record.message for record in caplog.records)


def test_ready_timeout_reports_each_source_progress():
    runtime = object.__new__(SensorHubRuntime)
    runtime.config = SimpleNamespace(startup_timeout_s=0.001)
    runtime.fatal_event = Event()
    runtime.first_publish_event = Event()
    runtime.stop_event = Event()
    runtime._parent_is_alive = lambda: True
    runtime.camera_caches = tuple(SampleCache[CameraSample](0.5) for _ in range(2))
    runtime.xense_cache = SampleCache[XenseSample](0.5)
    runtime.ft_cache = SampleCache[FTSample](0.5)
    runtime.robot_cache = SampleCache[RobotSample](0.5)
    runtime.gripper_cache = SampleCache[GripperSample](0.5)
    runtime.camera_caches[0].append(camera_sample := CameraSample(
        1,
        1,
        1,
        np.zeros((480, 640, 3), dtype=np.uint8),
        np.zeros((480, 640, 1), dtype=np.uint16),
    ))
    runtime.camera_caches[0].append(
        CameraSample(2, 2, 2, camera_sample.rgb, camera_sample.depth)
    )

    with pytest.raises(TimeoutError) as exc_info:
        runtime._wait_until_ready()

    message = str(exc_info.value)
    assert "first_publish=False" in message
    assert "'camera_1': 2" in message
    assert "'camera_2': 0" in message
    assert "'robot': 0" in message
    assert "insufficient_sources=['camera_2', 'xense', 'ft', 'robot', 'gripper']" in message


def _camera_bootstrap_runtime(camera_caches):
    runtime = object.__new__(SensorHubRuntime)
    runtime.camera_caches = camera_caches
    runtime.xense_cache = SampleCache[XenseSample](0.5)
    runtime.ft_cache = SampleCache[FTSample](0.5)
    runtime.robot_cache = SampleCache[RobotSample](0.5)
    runtime.gripper_cache = SampleCache[GripperSample](0.5)
    runtime.aligner = CausalAligner(
        camera_caches,
        runtime.xense_cache,
        runtime.ft_cache,
        runtime.robot_cache,
        runtime.gripper_cache,
        camera_bundle_span_warn_ms=20,
        camera_max_skew_ms=50,
        camera_bundle_wait_ms=25,
        required_sample_max_age_ms=100,
    )
    runtime.fatal_event = Event()
    runtime.stop_event = Event()
    runtime._fatal_message = ""
    runtime._parent_is_alive = lambda: True
    runtime._last_logged_camera_commit_count = 0
    runtime._alignment_pending_rejection = None
    runtime._alignment_pending_log_monotonic = 0.0
    runtime._log_new_camera_bundle = lambda: None
    return runtime


def test_sensorhub_main_uses_lerobot_logging(monkeypatch):
    calls = []
    config = object()
    monkeypatch.setattr(sensorhub_main, "init_logging", lambda: calls.append("logging"))
    monkeypatch.setattr(
        sensorhub_main,
        "SensorHubConfig",
        SimpleNamespace(from_dict=lambda values: calls.append(values) or config),
    )
    monkeypatch.setattr(
        sensorhub_main,
        "SensorHubRuntime",
        lambda parsed_config, parent_pid: SimpleNamespace(
            run=lambda: calls.append((parsed_config, parent_pid)) or 0
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["sensorhub", "--config-json", "{}", "--parent-pid", "123"],
    )

    assert sensorhub_main.main() == 0
    assert calls == ["logging", {}, (config, 123)]


def test_camera_bootstrap_is_independent_of_non_camera_readiness():
    camera_caches = tuple(SampleCache[CameraSample](0.5) for _ in range(2))
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    depth = np.zeros((480, 640, 1), dtype=np.uint16)
    for index, cache in enumerate(camera_caches):
        cache.append(CameraSample(1, 100 + index, 1 + index, rgb, depth))
        cache.append(CameraSample(2, 200 + index, 3 + index, rgb, depth))
    runtime = _camera_bootstrap_runtime(camera_caches)

    runtime._wait_until_camera_bootstrap(time.monotonic() + 0.1)

    assert runtime.aligner.cameras_initialized
    assert runtime.aligner.camera_commit_count == 1
    assert runtime.xense_cache.latest() is None


def test_camera_bootstrap_retries_hard_gate_with_same_startup_deadline():
    camera_caches = tuple(SampleCache[CameraSample](0.5) for _ in range(2))
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    depth = np.zeros((480, 640, 1), dtype=np.uint16)
    camera_caches[0].append(CameraSample(1, 0, 1, rgb, depth))
    camera_caches[0].append(CameraSample(2, 10_000_000, 2, rgb, depth))
    camera_caches[1].append(CameraSample(1, 70_000_000, 3, rgb, depth))
    camera_caches[1].append(CameraSample(2, 80_000_000, 4, rgb, depth))
    runtime = _camera_bootstrap_runtime(camera_caches)

    def append_recovery_samples():
        time.sleep(0.01)
        camera_caches[0].append(CameraSample(3, 90_000_000, 5, rgb, depth))
        camera_caches[1].append(CameraSample(3, 91_000_000, 6, rgb, depth))

    producer = Thread(target=append_recovery_samples)
    producer.start()
    runtime._wait_until_camera_bootstrap(time.monotonic() + 0.2)
    producer.join(timeout=1)

    assert runtime.aligner.cameras_initialized
    assert [sample.sequence for sample in runtime.aligner.camera_frontiers] == [3, 3]


def test_sensorhub_config_validates_camera_bundle_thresholds():
    values = {
        "telemetry_endpoint": "inproc://unused",
        "observation_shm_name": "/observation",
        "sensorhub_socket_path": "/tmp/sensorhub.sock",
        "realsense_shm_names": ["/cam1"],
        "xense_shm_name": "xense",
        "ft300s_shm_name": "ft",
    }
    config = SensorHubConfig.from_dict(values)
    assert (
        config.camera_bundle_span_warn_ms,
        config.camera_max_skew_ms,
        config.camera_bundle_wait_ms,
    ) == (20, 50, 25)
    with pytest.raises(ValueError, match="camera_bundle_wait_ms"):
        SensorHubConfig.from_dict(dict(values, camera_bundle_wait_ms=0))
    with pytest.raises(ValueError, match="must be <="):
        SensorHubConfig.from_dict(
            dict(values, camera_bundle_span_warn_ms=51, camera_max_skew_ms=50)
        )


def test_robot_resetting_updates_only_runtime_control_state(tmp_path):
    runtime = SensorHubRuntime(_config(tmp_path, f"unused_{uuid.uuid4().hex}"), os.getpid())
    transitions = []
    runtime.control.set_robot_resetting = transitions.append

    class Telemetry:
        def __init__(self):
            self.states = iter((False, True))
            self.robot_resetting = None
            self.sequence = 0

        def read(self, *, timeout_s):
            self.robot_resetting = next(self.states)
            self.sequence += 1
            if self.sequence == 2:
                runtime.stop_event.set()
            joints = np.zeros(7, dtype=np.float32)
            return RobotSample(
                self.sequence,
                time.monotonic_ns(),
                time.monotonic_ns(),
                joints,
                joints,
                joints,
                np.eye(4, dtype=np.float32),
            )

    try:
        runtime._start_telemetry_thread(Telemetry())
        runtime._threads[-1].join(timeout=1)
        assert transitions == [False, True]
        assert runtime.robot_cache.latest().sequence == 2
        assert "resetting" not in RobotSample.__dataclass_fields__
    finally:
        runtime.control.close()


def test_sensorhub_config_and_cache_count_preserve_dynamic_camera_order(tmp_path):
    names = ("/cam5", "/cam2", "/cam1", "/cam4", "/cam3")
    config = _config(tmp_path, f"unused_{uuid.uuid4().hex}", names)
    runtime = SensorHubRuntime(config, os.getpid())
    try:
        assert config.realsense_shm_names == names
        assert len(runtime.camera_caches) == 5
    finally:
        runtime.control.close()


def test_sensorhub_config_from_dict_preserves_order_and_validates_names():
    values = {
        "telemetry_endpoint": "inproc://unused",
        "observation_shm_name": "/observation",
        "sensorhub_socket_path": "/tmp/sensorhub.sock",
        "realsense_shm_names": ["/cam2", "/cam1", "/cam5"],
        "xense_shm_name": "xense",
        "ft300s_shm_name": "ft",
    }
    config = SensorHubConfig.from_dict(values)
    assert config.realsense_shm_names == ("/cam2", "/cam1", "/cam5")
    for names, error in [([], ValueError), (["/cam1", "/cam1"], ValueError), ([1], TypeError)]:
        invalid = dict(values, realsense_shm_names=names)
        with pytest.raises(error):
            SensorHubConfig.from_dict(invalid)


def test_attach_readers_uses_only_configured_names_in_order(monkeypatch, tmp_path):
    attached_names = []

    class FakeReader:
        def __init__(self, name):
            self.name = name

        def close(self):
            pass

    monkeypatch.setattr(
        runtime_module,
        "RealSenseReader",
        lambda name: attached_names.append(name) or FakeReader(name),
    )
    monkeypatch.setattr(runtime_module, "XenseReader", FakeReader)
    monkeypatch.setattr(runtime_module, "FT300SReader", FakeReader)
    monkeypatch.setattr(runtime_module, "TelemetryReader", FakeReader)

    names = ("/runtime_cam5", "/runtime_cam1")
    runtime = object.__new__(SensorHubRuntime)
    runtime.config = _config(tmp_path, "unused", names)
    runtime.stop_event = Event()
    runtime._parent_is_alive = lambda: True
    cameras, _xense, _ft, _telemetry = runtime._attach_readers()
    assert attached_names == list(names)
    assert tuple(reader.name for reader in cameras) == names
