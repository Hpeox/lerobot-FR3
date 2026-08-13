# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import deque
from threading import Lock
from typing import Protocol

from .samples import AlignedSample, CameraSample, FTSample, GripperSample, RobotSample, XenseSample


class TimestampedSample(Protocol):
    sequence: int
    ingest_monotonic_ns: int


class SampleCache[SampleT: TimestampedSample]:
    """A small thread-safe monotonic-time cache of immutable samples."""

    def __init__(self, horizon_s: float):
        self._horizon_ns = int(horizon_s * 1_000_000_000)
        self._samples: deque[SampleT] = deque()
        self._lock = Lock()

    def append(self, sample: SampleT) -> None:
        with self._lock:
            if self._samples and sample.sequence == self._samples[-1].sequence:
                return
            self._samples.append(sample)
            cutoff = sample.ingest_monotonic_ns - self._horizon_ns
            while self._samples and self._samples[0].ingest_monotonic_ns < cutoff:
                self._samples.popleft()

    def latest(self) -> SampleT | None:
        with self._lock:
            return self._samples[-1] if self._samples else None

    def latest_at_or_before(self, reference_ns: int) -> SampleT | None:
        with self._lock:
            for sample in reversed(self._samples):
                if sample.ingest_monotonic_ns <= reference_ns:
                    return sample
        return None

    def sequence_count(self) -> int:
        with self._lock:
            return len({sample.sequence for sample in self._samples})


class CausalAligner:
    """Select the newest causal sample from every required modality."""

    def __init__(
        self,
        cameras: tuple[SampleCache[CameraSample], ...],
        xense: SampleCache[XenseSample],
        ft: SampleCache[FTSample],
        robot: SampleCache[RobotSample],
        gripper: SampleCache[GripperSample],
        *,
        camera_max_skew_ms: int,
        required_sample_max_age_ms: int,
    ):
        if not cameras:
            raise ValueError("CausalAligner requires at least one camera cache")
        self.cameras = cameras
        self.xense = xense
        self.ft = ft
        self.robot = robot
        self.gripper = gripper
        self.camera_max_skew_ns = camera_max_skew_ms * 1_000_000
        self.required_sample_max_age_ns = required_sample_max_age_ms * 1_000_000
        self._last_camera_sequences: tuple[int, ...] | None = None
        self._publish_sequence = 0

    def select(self, realtime_ns: int, monotonic_ns: int) -> AlignedSample | None:
        camera_latest = [cache.latest() for cache in self.cameras]
        if any(sample is None for sample in camera_latest):
            return None
        reference_ns = min(sample.ingest_monotonic_ns for sample in camera_latest if sample is not None)
        cameras = tuple(cache.latest_at_or_before(reference_ns) for cache in self.cameras)
        if any(sample is None for sample in cameras):
            return None
        typed_cameras = tuple(cameras)  # type: ignore[arg-type]
        camera_times = [sample.ingest_monotonic_ns for sample in typed_cameras]
        if max(camera_times) - min(camera_times) > self.camera_max_skew_ns:
            return None
        camera_sequences = tuple(sample.sequence for sample in typed_cameras)
        if camera_sequences == self._last_camera_sequences:
            return None

        selected = (
            self.xense.latest_at_or_before(reference_ns),
            self.ft.latest_at_or_before(reference_ns),
            self.robot.latest_at_or_before(reference_ns),
            self.gripper.latest_at_or_before(reference_ns),
        )
        if any(sample is None for sample in selected):
            return None
        if any(
            reference_ns - sample.ingest_monotonic_ns > self.required_sample_max_age_ns for sample in selected
        ):
            return None
        xense, ft, robot, gripper = selected
        self._publish_sequence += 1
        self._last_camera_sequences = camera_sequences
        return AlignedSample(
            sequence=self._publish_sequence,
            publish_realtime_ns=realtime_ns,
            publish_monotonic_ns=monotonic_ns,
            cameras=typed_cameras,  # type: ignore[arg-type]
            xense=xense,  # type: ignore[arg-type]
            ft=ft,  # type: ignore[arg-type]
            robot=robot,  # type: ignore[arg-type]
            gripper=gripper,  # type: ignore[arg-type]
        )
