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
        self._last_rejection_reason: str | None = None

    @property
    def last_rejection_reason(self) -> str | None:
        """Return why the most recent alignment attempt produced no sample."""
        return self._last_rejection_reason

    def _reject(self, reason: str) -> None:
        self._last_rejection_reason = reason
        return None

    def select(self, realtime_ns: int, monotonic_ns: int) -> AlignedSample | None:
        camera_latest = [cache.latest() for cache in self.cameras]
        if any(sample is None for sample in camera_latest):
            missing = [
                f"camera_{index + 1}"
                for index, sample in enumerate(camera_latest)
                if sample is None
            ]
            return self._reject(f"missing latest samples: {','.join(missing)}")
        reference_ns = min(sample.ingest_monotonic_ns for sample in camera_latest if sample is not None)
        cameras = tuple(cache.latest_at_or_before(reference_ns) for cache in self.cameras)
        if any(sample is None for sample in cameras):
            missing = [
                f"camera_{index + 1}"
                for index, sample in enumerate(cameras)
                if sample is None
            ]
            return self._reject(
                f"missing causal samples at camera reference: {','.join(missing)}"
            )
        typed_cameras = tuple(cameras)  # type: ignore[arg-type]
        camera_times = [sample.ingest_monotonic_ns for sample in typed_cameras]
        camera_skew_ns = max(camera_times) - min(camera_times)
        if camera_skew_ns > self.camera_max_skew_ns:
            return self._reject(
                "camera skew exceeded: "
                f"observed_ms={camera_skew_ns / 1_000_000:.3f} "
                f"limit_ms={self.camera_max_skew_ns / 1_000_000:.3f}"
            )
        camera_sequences = tuple(sample.sequence for sample in typed_cameras)
        if camera_sequences == self._last_camera_sequences:
            return self._reject(f"camera sequences unchanged: {camera_sequences}")

        selected = (
            self.xense.latest_at_or_before(reference_ns),
            self.ft.latest_at_or_before(reference_ns),
            self.robot.latest_at_or_before(reference_ns),
            self.gripper.latest_at_or_before(reference_ns),
        )
        if any(sample is None for sample in selected):
            names = ("xense", "ft", "robot", "gripper")
            missing = [name for name, sample in zip(names, selected, strict=True) if sample is None]
            return self._reject(f"missing required samples: {','.join(missing)}")
        names = ("xense", "ft", "robot", "gripper")
        stale = [
            f"{name}(age_ms={(reference_ns - sample.ingest_monotonic_ns) / 1_000_000:.3f})"
            for name, sample in zip(names, selected, strict=True)
            if reference_ns - sample.ingest_monotonic_ns > self.required_sample_max_age_ns
        ]
        if stale:
            return self._reject(
                "required samples stale: "
                f"{','.join(stale)} limit_ms={self.required_sample_max_age_ns / 1_000_000:.3f}"
            )
        xense, ft, robot, gripper = selected
        self._publish_sequence += 1
        self._last_camera_sequences = camera_sequences
        self._last_rejection_reason = None
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
