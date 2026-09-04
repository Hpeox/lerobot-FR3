# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from threading import Event, Lock, Thread
from typing import Final

from ..config_fr3 import normalize_realsense_shm_names
from .aligned_shm import AlignedObservationWriter
from .cache import AlignmentRejection, CausalAligner, SampleCache, format_alignment_rejection
from .readers import FT300SReader, RealSenseReader, TelemetryReader, XenseReader
from .samples import CameraSample, FTSample, GripperSample, RobotSample, XenseSample
from .uds import UDSControlServer

logger = logging.getLogger(__name__)

# RealSense and the FR3 rollout contract are both 30 Hz.  Keeping this gate
# local to the alignment publisher prevents scheduler delays from turning into
# catch-up bursts that can overwrite the other SHM slot during a reader copy.
ALIGNED_PUBLISH_PERIOD_NS: Final = round(1_000_000_000 / 30)


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
    camera_bundle_span_warn_ms: int = 20
    camera_max_skew_ms: int = 50
    camera_bundle_wait_ms: int = 25
    required_sample_max_age_ms: int = 100

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "realsense_shm_names",
            normalize_realsense_shm_names(self.realsense_shm_names),
        )
        positive = {
            "startup_timeout_s": self.startup_timeout_s,
            "cache_horizon_s": self.cache_horizon_s,
            "camera_bundle_span_warn_ms": self.camera_bundle_span_warn_ms,
            "camera_max_skew_ms": self.camera_max_skew_ms,
            "camera_bundle_wait_ms": self.camera_bundle_wait_ms,
            "required_sample_max_age_ms": self.required_sample_max_age_ms,
        }
        invalid = [name for name, value in positive.items() if not math.isfinite(value) or value <= 0]
        if invalid:
            raise ValueError(f"SensorHub timeout/cache values must be positive: {invalid}")
        if self.camera_bundle_span_warn_ms > self.camera_max_skew_ms:
            raise ValueError(
                "camera_bundle_span_warn_ms must be <= camera_max_skew_ms"
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
        self.ready_event = Event()
        self._fatal_lock = Lock()
        self._fatal_message = ""
        self._attach_pending_error: str | None = None
        self._attach_pending_log_monotonic = 0.0
        self._alignment_pending_rejection: AlignmentRejection | None = None
        self._alignment_pending_log_monotonic = 0.0
        self._last_logged_alignment_rejection: AlignmentRejection | None = None
        self._last_logged_camera_commit_count = 0
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
            camera_bundle_span_warn_ms=config.camera_bundle_span_warn_ms,
            camera_max_skew_ms=config.camera_max_skew_ms,
            camera_bundle_wait_ms=config.camera_bundle_wait_ms,
            required_sample_max_age_ms=config.required_sample_max_age_ms,
        )

    def run(self) -> int:
        self.control.start()
        try:
            startup_deadline = time.monotonic() + self.config.startup_timeout_s
            cameras, xense, ft, telemetry = self._attach_readers(startup_deadline)
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
            self._wait_until_camera_bootstrap(startup_deadline)
            self._start_thread("AlignmentPublisher", self._alignment_loop)

            self._wait_until_ready(startup_deadline)
            self.control.publish("READY", message="first aligned snapshot published")
            self.ready_event.set()
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

    def _attach_readers(self, deadline: float | None = None):
        if deadline is None:
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
                self._log_attach_pending(exc)
                for reader in reversed(attached):
                    reader.close()  # type: ignore[attr-defined]
                time.sleep(0.05)
        if not self._parent_is_alive():
            self.stop_event.set()
            raise _ParentExited("parent process exited while SensorHub was attaching readers")
        raise TimeoutError(f"required upstream writers were not ready: {last_error}")

    def _log_attach_pending(self, exc: Exception) -> None:
        """Log the current upstream attach failure on change or once per second."""
        message = f"{type(exc).__name__}: {exc}"
        now = time.monotonic()
        if (
            message != self._attach_pending_error
            or now - self._attach_pending_log_monotonic >= 1.0
        ):
            logger.warning("SensorHub attach pending before READY: %s", message)
            self._attach_pending_error = message
            self._attach_pending_log_monotonic = now

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
        next_publish_not_before_ns = 0
        cadence_gate_checked = True
        while not self.stop_event.is_set():
            try:
                now_ns = time.monotonic_ns()
                if now_ns < next_publish_not_before_ns:
                    time.sleep(
                        min(
                            (next_publish_not_before_ns - now_ns) / 1_000_000_000,
                            0.001,
                        )
                    )
                    continue
                if not cadence_gate_checked:
                    lateness_ns = now_ns - next_publish_not_before_ns
                    if lateness_ns >= ALIGNED_PUBLISH_PERIOD_NS:
                        logger.warning(
                            "SensorHub alignment publisher late: lateness_ms=%.3f",
                            lateness_ns / 1_000_000,
                        )
                    cadence_gate_checked = True
                sample = self.aligner.select(time.time_ns(), now_ns)
                self._log_new_camera_bundle()
                if sample is None:
                    if self.ready_event.is_set():
                        self._log_alignment_rejected()
                    elif not self.first_publish_event.is_set():
                        self._log_alignment_pending()
                    time.sleep(0.001)
                    continue
                assert self.writer is not None
                self.writer.publish(sample)
                publish_duration_ns = self.writer.timing_diagnostics.publish_duration_ns
                if (
                    publish_duration_ns is not None
                    and publish_duration_ns >= ALIGNED_PUBLISH_PERIOD_NS
                ):
                    logger.warning(
                        "SensorHub aligned writer slow: duration_ms=%.3f",
                        publish_duration_ns / 1_000_000,
                    )
                if self._last_logged_alignment_rejection is not None:
                    logger.info("SensorHub alignment recovered")
                    self._last_logged_alignment_rejection = None
                self.first_publish_event.set()
                # Schedule from completion so a delayed iteration does not
                # immediately publish several queued camera bundles.
                next_publish_not_before_ns = (
                    time.monotonic_ns() + ALIGNED_PUBLISH_PERIOD_NS
                )
                cadence_gate_checked = False
            except Exception as exc:
                self._fatal(f"AlignmentPublisher failed: {exc}")
                return
            time.sleep(0.001)

    def _log_new_camera_bundle(self) -> None:
        commit_count = getattr(self.aligner, "camera_commit_count", 0)
        if commit_count == getattr(self, "_last_logged_camera_commit_count", 0):
            return
        bundle = getattr(self.aligner, "last_camera_bundle", None)
        if bundle is None:
            return
        logger.debug(
            "SensorHub camera bundle committed: mode=%s source_span_ms=%.3f "
            "round_wait_ms=%.3f sequences=%s resynced=%s degraded=%s reused_cameras=%s",
            bundle.mode,
            bundle.source_span_ns / 1_000_000,
            bundle.round_wait_ns / 1_000_000,
            tuple(sample.sequence for sample in bundle.cameras),
            bundle.resynced,
            bundle.degraded,
            bundle.reused_camera_indices,
        )
        self._last_logged_camera_commit_count = commit_count

    def _log_alignment_pending(self) -> None:
        """Log the startup alignment rejection on change or once per second."""
        rejection = self.aligner.last_rejection
        now = time.monotonic()
        if (
            rejection != self._alignment_pending_rejection
            or now - self._alignment_pending_log_monotonic >= 1.0
        ):
            logger.warning(
                "SensorHub alignment pending before READY: %s",
                format_alignment_rejection(rejection),
            )
            self._alignment_pending_rejection = rejection
            self._alignment_pending_log_monotonic = now

    def _log_alignment_rejected(self) -> None:
        rejection = self.aligner.last_rejection
        if rejection is None:
            raise RuntimeError("aligner returned no sample without a rejection")
        if rejection != self._last_logged_alignment_rejection:
            logger.warning("SensorHub alignment rejected: %s", rejection.value)
            self._last_logged_alignment_rejection = rejection

    def _all_sources_advanced(self) -> bool:
        caches = (
            *self.camera_caches,
            self.xense_cache,
            self.ft_cache,
            self.robot_cache,
            self.gripper_cache,
        )
        return all(cache.sequence_count() >= 2 for cache in caches)

    def _wait_until_camera_bootstrap(self, deadline: float) -> None:
        """Acquire and commit camera phase once within the existing startup deadline."""
        while time.monotonic() < deadline:
            if self.fatal_event.is_set():
                raise RuntimeError(self._fatal_message)
            if not self._parent_is_alive():
                self.stop_event.set()
                raise _ParentExited("parent process exited during SensorHub camera bootstrap")
            if all(cache.sequence_count() >= 2 for cache in self.camera_caches):
                bundle = self.aligner.initialize_cameras()
                if bundle is not None:
                    self._log_new_camera_bundle()
                    return
                self._log_alignment_pending()
            time.sleep(0.005)
        counts = self._source_sequence_counts()
        raise TimeoutError(
            "camera bootstrap did not produce a coherent tuple: "
            f"reason={self.aligner.last_rejection_reason} sequence_counts={counts}"
        )

    def _wait_until_ready(self, deadline: float | None = None) -> None:
        if deadline is None:
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
