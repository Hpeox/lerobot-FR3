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
from dataclasses import dataclass, field

from ..config import RobotConfig


def _default_sensorhub_socket_path() -> str:
    return f"/run/user/{os.getuid()}/fr3_sensorhub.sock"


@RobotConfig.register_subclass("fr3")
@dataclass(kw_only=True)
class FR3Config(RobotConfig):
    """Configuration for the FR3 Robot and its managed SensorHub process."""

    command_endpoint: str = "tcp://192.168.1.37:6001"
    telemetry_endpoint: str = "tcp://192.168.1.37:6000"
    observation_shm_name: str = "/fr3_aligned_observation"
    sensorhub_socket_path: str = field(default_factory=_default_sensorhub_socket_path)
    realsense_shm_names: tuple[str, str, str, str] = (
        "/realsense_cam1",
        "/realsense_cam2",
        "/realsense_cam3",
        "/realsense_cam4",
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

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.realsense_shm_names) != 4:
            raise ValueError("FR3 requires exactly four RealSense SHM names")
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
