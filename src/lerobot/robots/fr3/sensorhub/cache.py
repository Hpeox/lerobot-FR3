# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from itertools import product
from threading import Lock
from typing import Protocol

from .samples import AlignedSample, CameraSample, FTSample, GripperSample, RobotSample, XenseSample


class AlignmentRejection(str, Enum):
    CAMERA_NOT_INITIALIZED = "CAMERA_NOT_INITIALIZED"
    CAMERA_BOOTSTRAP_INSUFFICIENT_SAMPLES = "CAMERA_BOOTSTRAP_INSUFFICIENT_SAMPLES"
    CAMERA_BOOTSTRAP_HARD_SKEW = "CAMERA_BOOTSTRAP_HARD_SKEW"
    CAMERA_SEARCH_WINDOW_EMPTY = "CAMERA_SEARCH_WINDOW_EMPTY"
    CAMERA_TUPLE_UNCHANGED = "CAMERA_TUPLE_UNCHANGED"
    CAMERA_ROUND_AWAITING_ARRIVALS = "CAMERA_ROUND_AWAITING_ARRIVALS"
    CAMERA_NO_ADVANCING_TUPLE = "CAMERA_NO_ADVANCING_TUPLE"
    CAMERA_DEGRADED_AWAITING_TIMEOUT = "CAMERA_DEGRADED_AWAITING_TIMEOUT"
    CAMERA_HARD_SKEW = "CAMERA_HARD_SKEW"
    XENSE_MISSING = "XENSE_MISSING"
    XENSE_STALE = "XENSE_STALE"
    FT_MISSING = "FT_MISSING"
    FT_STALE = "FT_STALE"
    ROBOT_MISSING = "ROBOT_MISSING"
    ROBOT_STALE = "ROBOT_STALE"
    GRIPPER_MISSING = "GRIPPER_MISSING"
    GRIPPER_STALE = "GRIPPER_STALE"


_REJECTION_TEXT = {
    AlignmentRejection.CAMERA_NOT_INITIALIZED: "camera bundler is not initialized",
    AlignmentRejection.CAMERA_BOOTSTRAP_INSUFFICIENT_SAMPLES: (
        "camera bootstrap requires two advancing samples"
    ),
    AlignmentRejection.CAMERA_BOOTSTRAP_HARD_SKEW: (
        "camera bootstrap source span exceeded hard gate"
    ),
    AlignmentRejection.CAMERA_SEARCH_WINDOW_EMPTY: "camera rolling search window is empty",
    AlignmentRejection.CAMERA_TUPLE_UNCHANGED: "camera tuple unchanged",
    AlignmentRejection.CAMERA_ROUND_AWAITING_ARRIVALS: "camera round awaiting arrivals",
    AlignmentRejection.CAMERA_NO_ADVANCING_TUPLE: (
        "camera bounded search found no advancing tuple"
    ),
    AlignmentRejection.CAMERA_DEGRADED_AWAITING_TIMEOUT: (
        "camera degraded candidate awaiting timeout"
    ),
    AlignmentRejection.CAMERA_HARD_SKEW: "camera source span exceeded hard gate",
    AlignmentRejection.XENSE_MISSING: "missing required samples: xense",
    AlignmentRejection.XENSE_STALE: "required samples stale: xense",
    AlignmentRejection.FT_MISSING: "missing required samples: ft",
    AlignmentRejection.FT_STALE: "required samples stale: ft",
    AlignmentRejection.ROBOT_MISSING: "missing required samples: robot",
    AlignmentRejection.ROBOT_STALE: "required samples stale: robot",
    AlignmentRejection.GRIPPER_MISSING: "missing required samples: gripper",
    AlignmentRejection.GRIPPER_STALE: "required samples stale: gripper",
}


def format_alignment_rejection(rejection: AlignmentRejection | None) -> str:
    if rejection is None:
        return "unknown alignment rejection"
    return _REJECTION_TEXT[rejection]


_PENDING_RETRY_REJECTIONS = frozenset(
    {
        AlignmentRejection.CAMERA_TUPLE_UNCHANGED,
        AlignmentRejection.CAMERA_ROUND_AWAITING_ARRIVALS,
        AlignmentRejection.CAMERA_DEGRADED_AWAITING_TIMEOUT,
    }
)


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

    def snapshot(self) -> tuple[SampleT, ...]:
        """Return an ordered immutable view of the current rolling search window."""
        with self._lock:
            return tuple(self._samples)

    def sequence_count(self) -> int:
        with self._lock:
            return len({sample.sequence for sample in self._samples})


@dataclass(frozen=True, slots=True)
class CameraBundle:
    cameras: tuple[CameraSample, ...]
    source_span_ns: int
    bundle_time_ns: int
    mode: str
    degraded: bool
    resynced: bool
    reused_camera_indices: tuple[int, ...]
    round_wait_ns: int


def _same_sample(left: CameraSample, right: CameraSample) -> bool:
    return left is right or (
        left.sequence == right.sequence
        and left.ingest_monotonic_ns == right.ingest_monotonic_ns
    )


class CausalCameraBundler:
    """Select coherent camera tuples from already-arrived rolling cache windows."""

    def __init__(
        self,
        cameras: tuple[SampleCache[CameraSample], ...],
        *,
        camera_bundle_span_warn_ms: int,
        camera_max_skew_ms: int,
        camera_bundle_wait_ms: int,
    ):
        if not cameras:
            raise ValueError("CausalCameraBundler requires at least one camera cache")
        self.cameras = cameras
        self.camera_bundle_span_warn_ns = camera_bundle_span_warn_ms * 1_000_000
        self.camera_max_skew_ns = camera_max_skew_ms * 1_000_000
        self.camera_bundle_wait_ns = camera_bundle_wait_ms * 1_000_000
        self._frontiers: tuple[CameraSample, ...] | None = None
        self._last_bundle: CameraBundle | None = None
        self._commit_count = 0
        self._last_rejection: AlignmentRejection | None = None

    @property
    def initialized(self) -> bool:
        return self._frontiers is not None

    @property
    def frontiers(self) -> tuple[CameraSample, ...] | None:
        return self._frontiers

    @property
    def last_bundle(self) -> CameraBundle | None:
        return self._last_bundle

    @property
    def commit_count(self) -> int:
        return self._commit_count

    @property
    def last_rejection(self) -> AlignmentRejection | None:
        return self._last_rejection

    @property
    def last_rejection_reason(self) -> str | None:
        if self._last_rejection is None:
            return None
        return format_alignment_rejection(self._last_rejection)

    def _reject(self, rejection: AlignmentRejection) -> None:
        self._last_rejection = rejection
        return None

    @staticmethod
    def _score(cameras: tuple[CameraSample, ...]) -> tuple[int, int]:
        source_times = tuple(sample.source_timestamp_ns for sample in cameras)
        return max(source_times) - min(source_times), -max(source_times)

    @classmethod
    def _best_candidate(
        cls,
        candidate_lists: tuple[tuple[CameraSample, ...], ...],
        frontiers: tuple[CameraSample, ...] | None = None,
    ) -> tuple[CameraSample, ...] | None:
        best: tuple[tuple[int, int], tuple[CameraSample, ...]] | None = None
        for candidate in product(*candidate_lists):
            typed_candidate = tuple(candidate)
            if frontiers is not None and all(
                _same_sample(sample, frontier)
                for sample, frontier in zip(typed_candidate, frontiers, strict=True)
            ):
                continue
            score = cls._score(typed_candidate)
            if best is None or score < best[0]:
                best = (score, typed_candidate)
        return None if best is None else best[1]

    def initialize(self) -> CameraBundle | None:
        """Commit one bootstrap tuple, or leave all frontiers untouched for a retry."""
        if self.initialized:
            raise RuntimeError("camera bundler is already initialized")
        snapshots = tuple(cache.snapshot() for cache in self.cameras)
        if any(len({sample.sequence for sample in samples}) < 2 for samples in snapshots):
            return self._reject(AlignmentRejection.CAMERA_BOOTSTRAP_INSUFFICIENT_SAMPLES)
        candidate_lists = tuple(tuple(samples[-2:]) for samples in snapshots)
        candidate = self._best_candidate(candidate_lists)
        assert candidate is not None
        source_span_ns, negative_bundle_time_ns = self._score(candidate)
        if source_span_ns > self.camera_max_skew_ns:
            return self._reject(AlignmentRejection.CAMERA_BOOTSTRAP_HARD_SKEW)
        degraded = source_span_ns > self.camera_bundle_span_warn_ns
        bundle = CameraBundle(
            cameras=candidate,
            source_span_ns=source_span_ns,
            bundle_time_ns=-negative_bundle_time_ns,
            mode="bootstrap_degraded" if degraded else "bootstrap_nominal",
            degraded=degraded,
            resynced=False,
            reused_camera_indices=(),
            round_wait_ns=0,
        )
        return self._commit(bundle)

    @staticmethod
    def _frontier_index(
        samples: tuple[CameraSample, ...], frontier: CameraSample
    ) -> int | None:
        for index in range(len(samples) - 1, -1, -1):
            if _same_sample(samples[index], frontier):
                return index
        return None

    def _post_frontier(
        self, samples: tuple[CameraSample, ...], frontier: CameraSample
    ) -> tuple[CameraSample, ...]:
        index = self._frontier_index(samples, frontier)
        if index is None:
            return samples
        return samples[index + 1 :]

    def _candidate_window(
        self, samples: tuple[CameraSample, ...], frontier: CameraSample
    ) -> tuple[CameraSample, ...]:
        index = self._frontier_index(samples, frontier)
        eligible = samples if index is None else samples[index:]
        return eligible[-2:]

    def select(self, monotonic_ns: int) -> CameraBundle | None:
        if self._frontiers is None:
            return self._reject(AlignmentRejection.CAMERA_NOT_INITIALIZED)
        snapshots = tuple(cache.snapshot() for cache in self.cameras)
        if any(not samples for samples in snapshots):
            return self._reject(AlignmentRejection.CAMERA_SEARCH_WINDOW_EMPTY)
        post_frontier = tuple(
            self._post_frontier(samples, frontier)
            for samples, frontier in zip(snapshots, self._frontiers, strict=True)
        )
        arrived = tuple(sample for samples in post_frontier for sample in samples)
        if not arrived:
            return self._reject(AlignmentRejection.CAMERA_TUPLE_UNCHANGED)
        round_start_ingest_ns = min(sample.ingest_monotonic_ns for sample in arrived)
        round_wait_ns = max(0, monotonic_ns - round_start_ingest_ns)
        wait_expired = round_wait_ns >= self.camera_bundle_wait_ns
        all_cameras_arrived = all(post_frontier)
        if not all_cameras_arrived and not wait_expired:
            return self._reject(AlignmentRejection.CAMERA_ROUND_AWAITING_ARRIVALS)

        latest = tuple(samples[-1] for samples in snapshots)
        latest_span_ns, negative_latest_time_ns = self._score(latest)
        if latest_span_ns <= self.camera_bundle_span_warn_ns:
            return self._commit(
                self._bundle(
                    latest,
                    latest_span_ns,
                    -negative_latest_time_ns,
                    mode="latest",
                    round_wait_ns=round_wait_ns,
                    resynced=False,
                )
            )

        candidate_lists = tuple(
            self._candidate_window(samples, frontier)
            for samples, frontier in zip(snapshots, self._frontiers, strict=True)
        )
        candidate = self._best_candidate(candidate_lists, self._frontiers)
        if candidate is None:
            return self._reject(AlignmentRejection.CAMERA_NO_ADVANCING_TUPLE)
        source_span_ns, negative_bundle_time_ns = self._score(candidate)
        if source_span_ns <= self.camera_bundle_span_warn_ns:
            return self._commit(
                self._bundle(
                    candidate,
                    source_span_ns,
                    -negative_bundle_time_ns,
                    mode="fallback_search",
                    round_wait_ns=round_wait_ns,
                    resynced=True,
                )
            )
        if source_span_ns <= self.camera_max_skew_ns and wait_expired:
            return self._commit(
                self._bundle(
                    candidate,
                    source_span_ns,
                    -negative_bundle_time_ns,
                    mode="degraded_best_effort",
                    round_wait_ns=round_wait_ns,
                    resynced=not all(
                        _same_sample(sample, latest_sample)
                        for sample, latest_sample in zip(candidate, latest, strict=True)
                    ),
                )
            )
        if source_span_ns <= self.camera_max_skew_ns:
            return self._reject(AlignmentRejection.CAMERA_DEGRADED_AWAITING_TIMEOUT)
        return self._reject(AlignmentRejection.CAMERA_HARD_SKEW)

    def _bundle(
        self,
        cameras: tuple[CameraSample, ...],
        source_span_ns: int,
        bundle_time_ns: int,
        *,
        mode: str,
        round_wait_ns: int,
        resynced: bool,
    ) -> CameraBundle:
        assert self._frontiers is not None
        reused = tuple(
            index
            for index, (sample, frontier) in enumerate(
                zip(cameras, self._frontiers, strict=True), 1
            )
            if _same_sample(sample, frontier)
        )
        return CameraBundle(
            cameras=cameras,
            source_span_ns=source_span_ns,
            bundle_time_ns=bundle_time_ns,
            mode=mode,
            degraded=source_span_ns > self.camera_bundle_span_warn_ns,
            resynced=resynced,
            reused_camera_indices=reused,
            round_wait_ns=round_wait_ns,
        )

    def _commit(self, bundle: CameraBundle) -> CameraBundle:
        if self._frontiers is not None and all(
            _same_sample(sample, frontier)
            for sample, frontier in zip(bundle.cameras, self._frontiers, strict=True)
        ):
            raise AssertionError("identical camera tuple cannot be recommitted")
        self._frontiers = bundle.cameras
        self._last_bundle = bundle
        self._commit_count += 1
        self._last_rejection = None
        return bundle


class LatestObservationAssembler:
    """Assemble latest available non-camera state for one committed camera bundle."""

    def __init__(
        self,
        xense: SampleCache[XenseSample],
        ft: SampleCache[FTSample],
        robot: SampleCache[RobotSample],
        gripper: SampleCache[GripperSample],
        *,
        required_sample_max_age_ms: int,
    ):
        self.xense = xense
        self.ft = ft
        self.robot = robot
        self.gripper = gripper
        self.required_sample_max_age_ns = required_sample_max_age_ms * 1_000_000
        self._publish_sequence = 0
        self._last_rejection: AlignmentRejection | None = None

    @property
    def last_rejection(self) -> AlignmentRejection | None:
        return self._last_rejection

    @property
    def last_rejection_reason(self) -> str | None:
        if self._last_rejection is None:
            return None
        return format_alignment_rejection(self._last_rejection)

    def assemble(
        self, bundle: CameraBundle, realtime_ns: int, monotonic_ns: int
    ) -> AlignedSample | None:
        selected = (
            self.xense.latest(),
            self.ft.latest(),
            self.robot.latest(),
            self.gripper.latest(),
        )
        missing_rejections = (
            AlignmentRejection.XENSE_MISSING,
            AlignmentRejection.FT_MISSING,
            AlignmentRejection.ROBOT_MISSING,
            AlignmentRejection.GRIPPER_MISSING,
        )
        stale_rejections = (
            AlignmentRejection.XENSE_STALE,
            AlignmentRejection.FT_STALE,
            AlignmentRejection.ROBOT_STALE,
            AlignmentRejection.GRIPPER_STALE,
        )
        for sample, missing, stale in zip(
            selected, missing_rejections, stale_rejections, strict=True
        ):
            if sample is None:
                self._last_rejection = missing
                return None
            if monotonic_ns - sample.ingest_monotonic_ns > self.required_sample_max_age_ns:
                self._last_rejection = stale
                return None
        xense, ft, robot, gripper = selected
        self._publish_sequence += 1
        self._last_rejection = None
        return AlignedSample(
            sequence=self._publish_sequence,
            publish_realtime_ns=realtime_ns,
            publish_monotonic_ns=monotonic_ns,
            cameras=bundle.cameras,
            xense=xense,  # type: ignore[arg-type]
            ft=ft,  # type: ignore[arg-type]
            robot=robot,  # type: ignore[arg-type]
            gripper=gripper,  # type: ignore[arg-type]
        )


class CausalAligner:
    """Compatibility facade for camera bundling and latest-state assembly."""

    def __init__(
        self,
        cameras: tuple[SampleCache[CameraSample], ...],
        xense: SampleCache[XenseSample],
        ft: SampleCache[FTSample],
        robot: SampleCache[RobotSample],
        gripper: SampleCache[GripperSample],
        *,
        camera_bundle_span_warn_ms: int = 20,
        camera_max_skew_ms: int,
        camera_bundle_wait_ms: int = 25,
        required_sample_max_age_ms: int,
    ):
        self.cameras = cameras
        self.xense = xense
        self.ft = ft
        self.robot = robot
        self.gripper = gripper
        self._camera_bundler = CausalCameraBundler(
            cameras,
            camera_bundle_span_warn_ms=camera_bundle_span_warn_ms,
            camera_max_skew_ms=camera_max_skew_ms,
            camera_bundle_wait_ms=camera_bundle_wait_ms,
        )
        self._assembler = LatestObservationAssembler(
            xense,
            ft,
            robot,
            gripper,
            required_sample_max_age_ms=required_sample_max_age_ms,
        )
        self._pending_camera_bundle: CameraBundle | None = None
        self._last_rejection: AlignmentRejection | None = None

    @property
    def last_rejection(self) -> AlignmentRejection | None:
        return self._last_rejection

    @property
    def last_rejection_reason(self) -> str | None:
        if self._last_rejection is None:
            return None
        return format_alignment_rejection(self._last_rejection)

    @property
    def cameras_initialized(self) -> bool:
        return self._camera_bundler.initialized

    @property
    def camera_frontiers(self) -> tuple[CameraSample, ...] | None:
        return self._camera_bundler.frontiers

    @property
    def last_camera_bundle(self) -> CameraBundle | None:
        return self._camera_bundler.last_bundle

    @property
    def camera_commit_count(self) -> int:
        return self._camera_bundler.commit_count

    def initialize_cameras(self) -> CameraBundle | None:
        bundle = self._camera_bundler.initialize()
        self._last_rejection = self._camera_bundler.last_rejection
        if bundle is not None:
            self._pending_camera_bundle = bundle
        return bundle

    def select(self, realtime_ns: int, monotonic_ns: int) -> AlignedSample | None:
        bundle = self._camera_bundler.select(monotonic_ns)
        if bundle is not None:
            self._pending_camera_bundle = bundle
        elif (
            self._camera_bundler.last_rejection in _PENDING_RETRY_REJECTIONS
            and self._pending_camera_bundle is not None
        ):
            bundle = self._pending_camera_bundle
        else:
            self._last_rejection = self._camera_bundler.last_rejection
            return None
        aligned = self._assembler.assemble(bundle, realtime_ns, monotonic_ns)
        self._last_rejection = self._assembler.last_rejection
        if aligned is not None:
            self._pending_camera_bundle = None
        return aligned
