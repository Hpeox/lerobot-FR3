# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from threading import Event, Lock, Thread

from ..config_fr3 import normalize_realsense_shm_names
from .aligned_shm import AlignedObservationWriter
from .cache import CausalAligner, SampleCache
from .readers import FT300SReader, RealSenseReader, TelemetryReader, XenseReader
from .samples import CameraSample, FTSample, GripperSample, RobotSample, XenseSample
from .uds import UDSControlServer

logger = logging.getLogger(__name__)


class _ParentExited(RuntimeError):
    """Internal non-fatal signal that the owning FR3 process disappeared."""


@dataclass(frozen=True, slots=True)
class SensorHubConfig:
    telemetry_endpoint: str
    observation_shm_name: str
    sensorhub_socket_path: str
    realsense_shm_names: tuple[str, ...]
    xense_shm_name: str
    ft300s_shm_name: str
    startup_timeout_s: float = 10.0
    cache_horizon_s: float = 0.5
    camera_max_skew_ms: int = 50
    required_sample_max_age_ms: int = 100

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "realsense_shm_names",
            normalize_realsense_shm_names(self.realsense_shm_names),
        )

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> SensorHubConfig:
        values = dict(values)
        raw_names = values.pop("realsense_shm_names")
        names = normalize_realsense_shm_names(raw_names)
        return cls(realsense_shm_names=names, **values)  # type: ignore[arg-type]


class SensorHubRuntime:
    """Managed reader/alignment runtime. It never creates or stops upstream writers."""

    def __init__(self, config: SensorHubConfig, parent_pid: int):
        self.config = config
        self.parent_pid = parent_pid
        self.stop_event = Event()
        self.fatal_event = Event()
        self.first_publish_event = Event()
        self._fatal_lock = Lock()
        self._fatal_message = ""
        self._alignment_pending_reason: str | None = None
        self._alignment_pending_log_monotonic = 0.0
        self._threads: list[Thread] = []
        self._readers: list[object] = []
        self.writer: AlignedObservationWriter | None = None
        self.control = UDSControlServer(config.sensorhub_socket_path, self.stop_event)

        self.camera_caches = tuple(
            SampleCache[CameraSample](config.cache_horizon_s) for _ in config.realsense_shm_names
        )
        self.xense_cache = SampleCache[XenseSample](config.cache_horizon_s)
        self.ft_cache = SampleCache[FTSample](config.cache_horizon_s)
        self.robot_cache = SampleCache[RobotSample](config.cache_horizon_s)
        self.gripper_cache = SampleCache[GripperSample](config.cache_horizon_s)
        self.aligner = CausalAligner(
            self.camera_caches,
            self.xense_cache,
            self.ft_cache,
            self.robot_cache,
            self.gripper_cache,
            camera_max_skew_ms=config.camera_max_skew_ms,
            required_sample_max_age_ms=config.required_sample_max_age_ms,
        )

    def run(self) -> int:
        self.control.start()
        try:
            cameras, xense, ft, telemetry = self._attach_readers()
            self._readers = [*cameras, xense, ft, telemetry]
            self.writer = AlignedObservationWriter(
                self.config.observation_shm_name,
                camera_count=len(self.config.realsense_shm_names),
            )
            for index, reader in enumerate(cameras):
                self._start_reader_thread(
                    f"RealSenseReader-{index + 1}", reader.read, self.camera_caches[index].append
                )
            self._start_reader_thread("XenseReader", xense.read, self.xense_cache.append)
            self._start_reader_thread("FT300SReader", ft.read, self.ft_cache.append)
            self._start_telemetry_thread(telemetry)
            self._start_thread("AlignmentPublisher", self._alignment_loop)

            self._wait_until_ready()
            self.control.publish("READY", message="first aligned snapshot published")
            self._supervise()
            return 1 if self.fatal_event.is_set() else 0
        except _ParentExited:
            return 0
        except Exception as exc:
            self._fatal(f"SensorHub startup/runtime failure: {exc}")
            return 1
        finally:
            self.stop_event.set()
            for thread in self._threads:
                thread.join(timeout=0.5)
            for reader in reversed(self._readers):
                try:
                    reader.close()  # type: ignore[attr-defined]
                except Exception:
                    logger.exception("failed to close SensorHub reader")
            if self.writer is not None:
                self.writer.close(unlink=True)
            self.control.close()

    def _attach_readers(self):
        deadline = time.monotonic() + self.config.startup_timeout_s
        last_error: Exception | None = None
        while time.monotonic() < deadline and not self.stop_event.is_set():
            if not self._parent_is_alive():
                self.stop_event.set()
                raise _ParentExited("parent process exited while SensorHub was attaching readers")
            attached: list[object] = []
            try:
                camera_list = []
                for name in self.config.realsense_shm_names:
                    camera = RealSenseReader(name)
                    camera_list.append(camera)
                    attached.append(camera)
                cameras = tuple(camera_list)
                xense = XenseReader(self.config.xense_shm_name)
                attached.append(xense)
                ft = FT300SReader(self.config.ft300s_shm_name)
                attached.append(ft)
                telemetry = TelemetryReader(self.config.telemetry_endpoint)
                return cameras, xense, ft, telemetry
            except (FileNotFoundError, ValueError, OSError) as exc:
                last_error = exc
                for reader in reversed(attached):
                    reader.close()  # type: ignore[attr-defined]
                time.sleep(0.05)
        if not self._parent_is_alive():
            self.stop_event.set()
            raise _ParentExited("parent process exited while SensorHub was attaching readers")
        raise TimeoutError(f"required upstream writers were not ready: {last_error}")

    def _start_thread(self, name: str, target: Callable[[], None]) -> None:
        thread = Thread(target=target, name=name, daemon=True)
        self._threads.append(thread)
        thread.start()

    def _start_reader_thread(self, name: str, read: Callable, append: Callable) -> None:
        def loop() -> None:
            while not self.stop_event.is_set():
                try:
                    append(read(timeout_s=0.02))
                except TimeoutError:
                    continue
                except Exception as exc:
                    self._fatal(f"{name} failed: {exc}")
                    return

        self._start_thread(name, loop)

    def _start_telemetry_thread(self, reader: TelemetryReader) -> None:
        def loop() -> None:
            while not self.stop_event.is_set():
                try:
                    sample = reader.read(timeout_s=0.02)
                    if isinstance(sample, RobotSample):
                        self.robot_cache.append(sample)
                        self.control.set_robot_resetting(reader.robot_resetting)
                    else:
                        self.gripper_cache.append(sample)
                except (TimeoutError, LookupError):
                    continue
                except Exception as exc:
                    self._fatal(f"FR3TelemetryReader failed: {exc}")
                    return

        self._start_thread("FR3TelemetryReader", loop)

    def _alignment_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                now_ns = time.monotonic_ns()
                sample = self.aligner.select(time.time_ns(), now_ns)
                if sample is None:
                    if not self.first_publish_event.is_set():
                        self._log_alignment_pending()
                    time.sleep(0.001)
                    continue
                assert self.writer is not None
                self.writer.publish(sample)
                self.first_publish_event.set()
            except Exception as exc:
                self._fatal(f"AlignmentPublisher failed: {exc}")
                return
            time.sleep(0.001)

    def _log_alignment_pending(self) -> None:
        """Log the startup alignment rejection on change or once per second."""
        reason = self.aligner.last_rejection_reason or "unknown alignment rejection"
        now = time.monotonic()
        if (
            reason != self._alignment_pending_reason
            or now - self._alignment_pending_log_monotonic >= 1.0
        ):
            logger.warning("SensorHub alignment pending before READY: %s", reason)
            self._alignment_pending_reason = reason
            self._alignment_pending_log_monotonic = now

    def _all_sources_advanced(self) -> bool:
        caches = (
            *self.camera_caches,
            self.xense_cache,
            self.ft_cache,
            self.robot_cache,
            self.gripper_cache,
        )
        return all(cache.sequence_count() >= 2 for cache in caches)

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self.config.startup_timeout_s
        while time.monotonic() < deadline:
            if self.fatal_event.is_set():
                raise RuntimeError(self._fatal_message)
            if not self._parent_is_alive():
                self.stop_event.set()
                raise _ParentExited("parent process exited during SensorHub startup")
            if self._all_sources_advanced() and self.first_publish_event.is_set():
                return
            time.sleep(0.005)
        counts = self._source_sequence_counts()
        insufficient = [name for name, count in counts.items() if count < 2]
        raise TimeoutError(
            "readers did not produce advancing coherent samples and an aligned snapshot: "
            f"first_publish={self.first_publish_event.is_set()} "
            f"insufficient_sources={insufficient} sequence_counts={counts}"
        )

    def _source_sequence_counts(self) -> dict[str, int]:
        """Return startup progress for every source required by READY."""
        counts = {
            f"camera_{index + 1}": cache.sequence_count()
            for index, cache in enumerate(self.camera_caches)
        }
        counts.update(
            xense=self.xense_cache.sequence_count(),
            ft=self.ft_cache.sequence_count(),
            robot=self.robot_cache.sequence_count(),
            gripper=self.gripper_cache.sequence_count(),
        )
        return counts

    def _supervise(self) -> None:
        while not self.stop_event.is_set():
            if not self._parent_is_alive():
                self.stop_event.set()
                return
            time.sleep(0.01)

    def _parent_is_alive(self) -> bool:
        return self.parent_pid == os.getppid() and os.path.exists(f"/proc/{self.parent_pid}")

    def _fatal(self, message: str) -> None:
        with self._fatal_lock:
            if self.fatal_event.is_set():
                return
            self._fatal_message = message
            self.fatal_event.set()
            logger.error("SensorHub fatal: %s", message)
            try:
                if self.writer is not None:
                    self.writer.set_fatal(message)
            except Exception:
                logger.exception("failed to record SensorHub fatal state in aligned SHM")
            try:
                self.control.publish("FATAL", status_code=1, message=message)
            except Exception:
                logger.exception("failed to publish SensorHub fatal state over UDS")
            finally:
                self.stop_event.set()
