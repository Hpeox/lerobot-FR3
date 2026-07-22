# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class CameraSample:
    sequence: int
    source_timestamp_ns: int
    ingest_monotonic_ns: int
    rgb: np.ndarray
    depth: np.ndarray


@dataclass(frozen=True, slots=True)
class XenseSample:
    sequence: int
    source_timestamp_ns: int
    ingest_monotonic_ns: int
    force0: np.ndarray
    force1: np.ndarray


@dataclass(frozen=True, slots=True)
class FTSample:
    sequence: int
    source_timestamp_ns: int
    ingest_monotonic_ns: int
    wrench: np.ndarray


@dataclass(frozen=True, slots=True)
class RobotSample:
    sequence: int
    source_timestamp_ns: int
    ingest_monotonic_ns: int
    q: np.ndarray
    dq: np.ndarray
    tau_j: np.ndarray


@dataclass(frozen=True, slots=True)
class GripperSample:
    sequence: int
    source_timestamp_ns: int
    ingest_monotonic_ns: int
    gpo: int
    gcu: int


@dataclass(frozen=True, slots=True)
class AlignedSample:
    sequence: int
    publish_realtime_ns: int
    publish_monotonic_ns: int
    cameras: tuple[CameraSample, CameraSample, CameraSample, CameraSample]
    xense: XenseSample
    ft: FTSample
    robot: RobotSample
    gripper: GripperSample
