#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import math
import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import suppress
from typing import Any

import numpy as np

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from .config_fr3 import FR3Config
from .feature_adapter import (
    ACTION_KEYS,
    DEPTH_KEYS,
    JOINT_POSITION_KEYS,
    RGB_KEYS,
    XENSE_KEYS,
    fr3_action_dataset_features,
    fr3_observation_dataset_features,
)
from .protocols import pack_command, policy_gripper_to_gpo
from .sensorhub.aligned_shm import AlignedObservationClient
from .sensorhub.uds import MAX_PACKET_SIZE, connect_uds, make_packet, parse_packet

logger = logging.getLogger(__name__)


class FR3(Robot):
    """LeRobot FR3 adapter backed by a managed SensorHub subprocess."""

    config_class = FR3Config
    name = "fr3"

    def __init__(self, config: FR3Config):
        super().__init__(config)
        self.config = config
        self._connected = False
        self._sensorhub: subprocess.Popen | None = None
        self._uds: socket.socket | None = None
        self._observation_client: AlignedObservationClient | None = None
        self._zmq_context = None
        self._command_socket = None
        self._command_sequence = 0
        self._fatal_message: str | None = None

    @property
    def observation_features(self) -> dict[str, type | tuple[int, ...] | PolicyFeature]:
        features: dict[str, type | tuple[int, ...] | PolicyFeature] = {
            **dict.fromkeys(JOINT_POSITION_KEYS, float),
            "fr3.dq": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
            "fr3.tau_J": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
            "gripper.pos": float,
            "gripper.gPO": np.uint8,
            "gripper.gCU": np.uint8,
            "ft300s.wrench": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
            **dict.fromkeys(XENSE_KEYS, PolicyFeature(type=FeatureType.STATE, shape=(35, 20, 3))),
            **dict.fromkeys(RGB_KEYS, (480, 640, 3)),
            **dict.fromkeys(DEPTH_KEYS, (480, 640, 1)),
        }
        return features

    @property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(ACTION_KEYS, float)

    @property
    def visual_feature_keys(self) -> tuple[str, ...]:
        return (*RGB_KEYS, *DEPTH_KEYS)

    def observation_dataset_features(self, *, use_videos: bool = True) -> dict[str, dict[str, Any]]:
        return fr3_observation_dataset_features(use_videos=use_videos)

    def action_dataset_features(self) -> dict[str, dict[str, Any]]:
        return fr3_action_dataset_features()

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        if self._sensorhub is not None and self._sensorhub.poll() is None:
            raise RuntimeError("this FR3 instance already manages a live SensorHub process")
        deadline = time.monotonic() + self.config.sensorhub_start_timeout_s
        command = [
            sys.executable,
            "-m",
            "lerobot.robots.fr3.sensorhub",
            "--config-json",
            json.dumps(self.config.sensorhub_dict(), separators=(",", ":")),
            "--parent-pid",
            str(os.getpid()),
        ]
        try:
            self._sensorhub = subprocess.Popen(command, start_new_session=True)  # noqa: S603
            remaining = max(0.01, deadline - time.monotonic())
            self._uds = connect_uds(self.config.sensorhub_socket_path, remaining)
            self._wait_for_sensorhub_ready(deadline)
            self._observation_client = AlignedObservationClient(self.config.observation_shm_name)
            self._open_command_socket()
            self._connected = True
            logger.info("%s connected", self)
        except Exception:
            self._cleanup_resources(force_process=True)
            raise

    def _wait_for_sensorhub_ready(self, deadline: float) -> None:
        assert self._uds is not None
        while time.monotonic() < deadline:
            if self._sensorhub is not None and self._sensorhub.poll() is not None:
                raise RuntimeError(f"SensorHub exited during startup with code {self._sensorhub.returncode}")
            self._uds.settimeout(max(0.01, deadline - time.monotonic()))
            try:
                packet = parse_packet(self._uds.recv(MAX_PACKET_SIZE + 1))
            except TimeoutError:
                continue
            if packet["type"] == "READY":
                self._uds.setblocking(False)
                return
            if packet["type"] == "FATAL":
                raise RuntimeError(f"SensorHub failed to start: {packet['message']}")
        raise TimeoutError("SensorHub did not report READY before startup timeout")

    def _open_command_socket(self) -> None:
        try:
            import zmq
        except ImportError as exc:  # pragma: no cover - installation-specific
            raise ImportError("FR3 requires the 'pyzmq-dep' extra") from exc
        self._zmq_context = zmq.Context()
        self._command_socket = self._zmq_context.socket(zmq.PUB)
        self._command_socket.setsockopt(zmq.SNDHWM, 1)
        self._command_socket.setsockopt(zmq.CONFLATE, 1)
        self._command_socket.setsockopt(zmq.LINGER, 0)
        self._command_socket.connect(self.config.command_endpoint)

    def _check_health(self) -> None:
        if self._fatal_message:
            raise RuntimeError(self._fatal_message)
        if self._sensorhub is None or self._sensorhub.poll() is not None:
            code = None if self._sensorhub is None else self._sensorhub.returncode
            raise RuntimeError(f"managed SensorHub is not running (exit code {code})")
        if self._uds is not None:
            while True:
                try:
                    raw = self._uds.recv(MAX_PACKET_SIZE + 1)
                except BlockingIOError:
                    break
                except OSError as exc:
                    self._fatal_message = f"SensorHub UDS failed: {exc}"
                    raise RuntimeError(self._fatal_message) from exc
                if not raw:
                    break
                packet = parse_packet(raw)
                if packet["type"] == "FATAL":
                    self._fatal_message = f"SensorHub fatal: {packet['message']}"
                    raise RuntimeError(self._fatal_message)

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        self._check_health()
        assert self._observation_client is not None
        observation, _metadata = self._observation_client.read(
            timeout_ms=self.config.snapshot_read_timeout_ms,
            max_age_ms=self.config.max_snapshot_age_ms,
        )
        return observation

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        self._check_health()
        expected = set(ACTION_KEYS)
        received = set(action)
        if received != expected:
            missing = sorted(expected - received)
            extra = sorted(received - expected)
            raise ValueError(f"FR3 action fields mismatch; missing={missing}, extra={extra}")

        validated: dict[str, float] = {}
        for key in ACTION_KEYS:
            value = action[key]
            if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
                raise TypeError(f"{key} must be a real number, not bool")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(f"{key} must be finite")
            validated[key] = numeric

        clipped_gripper, gpo = policy_gripper_to_gpo(validated["gripper.pos"])
        if clipped_gripper != validated["gripper.pos"]:
            logger.warning(
                "gripper.pos was clipped from %.6g to %.6g", validated["gripper.pos"], clipped_gripper
            )
        self._command_sequence += 1
        frame = pack_command(
            self._command_sequence,
            [validated[key] for key in JOINT_POSITION_KEYS],
            gpo,
        )
        assert self._command_socket is not None
        try:
            import zmq

            self._command_socket.send(frame, flags=zmq.NOBLOCK)
        except zmq.Again as exc:
            raise RuntimeError("FR3 local command PUB queue is unavailable") from exc
        return {
            **{key: validated[key] for key in JOINT_POSITION_KEYS},
            "gripper.pos": clipped_gripper,
        }

    def disconnect(self) -> None:
        if not self._connected and self._sensorhub is None:
            return
        self._connected = False
        if self._uds is not None:
            with suppress(OSError):
                self._uds.send(
                    make_packet("SHUTDOWN", self._command_sequence + 1, message="robot disconnect")
                )
        self._cleanup_resources(force_process=False)
        logger.info("%s disconnected", self)

    def _cleanup_resources(self, *, force_process: bool) -> None:
        process = self._sensorhub
        if process is not None and process.poll() is None and not force_process:
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=self.config.sensorhub_stop_timeout_s)
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=self.config.sensorhub_stop_timeout_s)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                if process.poll() is None:
                    with suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=self.config.sensorhub_stop_timeout_s)

        if self._command_socket is not None:
            self._command_socket.close(linger=0)
        if self._zmq_context is not None:
            self._zmq_context.term()
        if self._observation_client is not None:
            self._observation_client.close()
        if self._uds is not None:
            self._uds.close()
        self._command_socket = None
        self._zmq_context = None
        self._observation_client = None
        self._uds = None
        self._sensorhub = None
        self._connected = False
