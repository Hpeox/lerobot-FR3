#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import math
import os
from numbers import Real
from dataclasses import dataclass, field

from ..config import RobotConfig


def _default_sensorhub_socket_path() -> str:
    return f"/run/user/{os.getuid()}/fr3_sensorhub.sock"


def normalize_realsense_shm_names(values: object) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple)):
        raise TypeError("realsense_shm_names must be an ordered tuple or list of SHM names")
    names = tuple(values)
    if not names:
        raise ValueError("at least one RealSense SHM name is required")
    if any(not isinstance(name, str) for name in names):
        raise TypeError("RealSense SHM names must be strings")
    invalid_names = [
        name
        for name in names
        if len(name) <= 1 or not name.startswith("/") or "/" in name[1:] or "\0" in name
    ]
    if invalid_names:
        raise ValueError(
            "RealSense SHM names must contain one leading slash and a non-empty simple name: "
            f"{invalid_names}"
        )
    if len(set(names)) != len(names):
        raise ValueError("RealSense SHM names must be unique")
    return names


@RobotConfig.register_subclass("fr3")
@dataclass(kw_only=True)
class FR3Config(RobotConfig):
    """Configuration for the FR3 Robot and its managed SensorHub process."""

    command_endpoint: str = "tcp://192.168.1.37:6001"
    telemetry_endpoint: str = "tcp://192.168.1.37:6000"
    observation_shm_name: str = "/fr3_aligned_observation"
    sensorhub_socket_path: str = field(default_factory=_default_sensorhub_socket_path)
    realsense_shm_names: tuple[str, ...] = (
        "/realsense_cam1",
        "/realsense_cam2",
    )
    xense_shm_name: str = "xense_sensor_frame"
    ft300s_shm_name: str = "ft300_sensor_frame"
    sensorhub_start_timeout_s: float = 10.0
    sensorhub_stop_timeout_s: float = 2.0
    snapshot_read_timeout_ms: int = 20
    max_snapshot_age_ms: int = 100
    cache_horizon_s: float = 0.5
    camera_max_skew_ms: int = 50
    required_sample_max_age_ms: int = 100
    reset_ack_timeout_s: float = 2.0
    reset_completion_timeout_s: float = 30.0
    reset_retry_interval_s: float = 0.1
    # Deterministic center captured in
    # zmq_franka_gello/test_artifacts/reverse_startup_drift_20260812_194933/
    # cycle_01/robot_fgt1.jsonl.
    rollout_home_joint_positions: tuple[float, ...] = (
        0.1416057646,
        0.3408541381,
        -0.0186031274,
        -1.5938080549,
        0.0486696586,
        1.8890386820,
        0.0432172865,
    )
    rollout_init_delta_lower: tuple[float, ...] = (-0.01,) * 7
    rollout_init_delta_upper: tuple[float, ...] = (0.01,) * 7

    def __post_init__(self) -> None:
        super().__post_init__()
        self.realsense_shm_names = normalize_realsense_shm_names(self.realsense_shm_names)
        positive = {
            "sensorhub_start_timeout_s": self.sensorhub_start_timeout_s,
            "sensorhub_stop_timeout_s": self.sensorhub_stop_timeout_s,
            "snapshot_read_timeout_ms": self.snapshot_read_timeout_ms,
            "max_snapshot_age_ms": self.max_snapshot_age_ms,
            "cache_horizon_s": self.cache_horizon_s,
            "camera_max_skew_ms": self.camera_max_skew_ms,
            "required_sample_max_age_ms": self.required_sample_max_age_ms,
            "reset_ack_timeout_s": self.reset_ack_timeout_s,
            "reset_completion_timeout_s": self.reset_completion_timeout_s,
            "reset_retry_interval_s": self.reset_retry_interval_s,
        }
        invalid = [name for name, value in positive.items() if not math.isfinite(value) or value <= 0]
        if invalid:
            raise ValueError(f"FR3 timeout/cache values must be positive: {invalid}")
        vectors = {
            "rollout_home_joint_positions": self.rollout_home_joint_positions,
            "rollout_init_delta_lower": self.rollout_init_delta_lower,
            "rollout_init_delta_upper": self.rollout_init_delta_upper,
        }
        for name, values in vectors.items():
            if len(values) != 7:
                raise ValueError(f"{name} must contain exactly 7 values")
            for value in values:
                if isinstance(value, bool) or not isinstance(value, Real):
                    raise TypeError(f"{name} values must be real numbers, not bool")
                if not math.isfinite(value):
                    raise ValueError(f"{name} values must be finite")
        if any(
            lower > upper
            for lower, upper in zip(self.rollout_init_delta_lower, self.rollout_init_delta_upper, strict=True)
        ):
            raise ValueError("rollout_init_delta_lower must be <= rollout_init_delta_upper per joint")

    def sensorhub_dict(self) -> dict[str, object]:
        """Return the JSON-safe subset consumed by the SensorHub CLI."""

        return {
            "telemetry_endpoint": self.telemetry_endpoint,
            "observation_shm_name": self.observation_shm_name,
            "sensorhub_socket_path": self.sensorhub_socket_path,
            "realsense_shm_names": list(self.realsense_shm_names),
            "xense_shm_name": self.xense_shm_name,
            "ft300s_shm_name": self.ft300s_shm_name,
            "startup_timeout_s": self.sensorhub_start_timeout_s,
            "cache_horizon_s": self.cache_horizon_s,
            "camera_max_skew_ms": self.camera_max_skew_ms,
            "required_sample_max_age_ms": self.required_sample_max_age_ms,
        }
