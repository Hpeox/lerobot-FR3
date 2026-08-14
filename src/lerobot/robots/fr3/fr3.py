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
from collections import deque
from collections.abc import Sequence
from contextlib import suppress
from threading import Lock
from typing import Any

import numpy as np

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from .config_fr3 import FR3Config
from .feature_adapter import (
    ACTION_KEYS,
    JOINT_POSITION_KEYS,
    XENSE_KEYS,
    fr3_action_dataset_features,
    fr3_camera_feature_keys,
    fr3_observation_dataset_features,
)
from .protocols import COMMAND_FLAG_RESET_JOINT, pack_command, policy_gripper_to_gpo
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
        self._uds_sequence = 0
        self._uds_send_lock = Lock()
        self._command_lock = Lock()
        self._resetting_events: deque[bool | None] = deque()
        self._fatal_message: str | None = None

    @property
    def observation_features(self) -> dict[str, type | tuple[int, ...] | PolicyFeature]:
        rgb_keys, depth_keys = fr3_camera_feature_keys(len(self.config.realsense_shm_names))
        features: dict[str, type | tuple[int, ...] | PolicyFeature] = {
            **dict.fromkeys(JOINT_POSITION_KEYS, float),
            "fr3.dq": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
            "fr3.tau_J": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
            "fr3.O_T_EE": PolicyFeature(type=FeatureType.STATE, shape=(4, 4)),
            "gripper.pos": float,
            "gripper.gPO": np.uint8,
            "gripper.gCU": np.uint8,
            "ft300s.wrench": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
            **dict.fromkeys(XENSE_KEYS, PolicyFeature(type=FeatureType.STATE, shape=(35, 20, 3))),
            **dict.fromkeys(rgb_keys, (480, 640, 3)),
            **dict.fromkeys(depth_keys, (480, 640, 1)),
        }
        return features

    @property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(ACTION_KEYS, float)

    @property
    def visual_feature_keys(self) -> tuple[str, ...]:
        rgb_keys, depth_keys = fr3_camera_feature_keys(len(self.config.realsense_shm_names))
        return (*rgb_keys, *depth_keys)

    def observation_dataset_features(self, *, use_videos: bool = True) -> dict[str, dict[str, Any]]:
        return fr3_observation_dataset_features(
            camera_count=len(self.config.realsense_shm_names), use_videos=use_videos
        )

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
            self._dispatch_uds_packet(packet, startup=True)
        diagnostic_deadline = time.monotonic() + min(1.0, self.config.sensorhub_stop_timeout_s)
        while time.monotonic() < diagnostic_deadline:
            self._uds.settimeout(max(0.01, diagnostic_deadline - time.monotonic()))
            try:
                packet = parse_packet(self._uds.recv(MAX_PACKET_SIZE + 1))
            except TimeoutError:
                continue
            if packet["type"] == "FATAL":
                self._dispatch_uds_packet(packet, startup=True)
            if self._sensorhub is not None and self._sensorhub.poll() is not None:
                break
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
                    self._fatal_message = "SensorHub UDS disconnected"
                    raise RuntimeError(self._fatal_message)
                try:
                    packet = parse_packet(raw)
                except (ValueError, UnicodeDecodeError) as exc:
                    self._fatal_message = f"SensorHub UDS protocol failure: {exc}"
                    raise RuntimeError(self._fatal_message) from exc
                self._dispatch_uds_packet(packet)

    def _dispatch_uds_packet(self, packet: dict[str, object], *, startup: bool = False) -> None:
        message_type = packet["type"]
        if message_type == "FATAL":
            prefix = "SensorHub failed to start" if startup else "SensorHub fatal"
            self._fatal_message = f"{prefix}: {packet['message']}"
            raise RuntimeError(self._fatal_message)
        if message_type == "ROBOT_RESETTING":
            status_code = packet["status_code"]
            if status_code not in {0, 1, 2}:
                raise RuntimeError(f"invalid ROBOT_RESETTING status_code: {status_code}")
            self._resetting_events.append(None if status_code == 2 else bool(status_code))

    def _send_uds_request(self, message_type: str, *, message: str = "") -> None:
        if self._uds is None:
            raise RuntimeError("SensorHub UDS is not connected")
        with self._uds_send_lock:
            self._uds_sequence += 1
            packet = make_packet(message_type, self._uds_sequence, message=message)
            try:
                self._uds.send(packet)
            except OSError as exc:
                self._fatal_message = f"SensorHub UDS failed: {exc}"
                raise RuntimeError(self._fatal_message) from exc

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
        with self._command_lock:
            self._check_health()
            self._command_sequence += 1
            frame = pack_command(
                self._command_sequence,
                [validated[key] for key in JOINT_POSITION_KEYS],
                gpo,
            )
            self._send_command_frame(frame)
        return {
            **{key: validated[key] for key in JOINT_POSITION_KEYS},
            "gripper.pos": clipped_gripper,
        }

    def _send_command_frame(self, frame: bytes) -> None:
        assert self._command_socket is not None
        try:
            import zmq

            self._command_socket.send(frame, flags=zmq.NOBLOCK)
        except zmq.Again as exc:
            raise RuntimeError("FR3 local command PUB queue is unavailable") from exc

    def initialize_rollout(self) -> None:
        """Randomize around the configured home pose and synchronously reset the FR3.

        ``RESET_JOINT`` also commands the gripper to OPEN by protocol design.
        """

        delta = np.random.uniform(
            self.config.rollout_init_delta_lower,
            self.config.rollout_init_delta_upper,
        )
        target = tuple(
            float(home + sampled)
            for home, sampled in zip(self.config.rollout_home_joint_positions, delta, strict=True)
        )
        logger.info("FR3 rollout initialization target q_reset=%s", target)
        self._reset_joints(target)

    def return_to_home(self) -> None:
        """Synchronously reset to the deterministic configured home pose.

        ``RESET_JOINT`` also commands the gripper to OPEN by protocol design.
        """

        target = tuple(self.config.rollout_home_joint_positions)
        logger.info("FR3 graceful shutdown target q_home=%s", target)
        self._reset_joints(target)

    @check_if_not_connected
    def _reset_joints(self, q_reset: Sequence[float]) -> None:
        """Request and synchronously wait for one FR3 joint reset trajectory."""

        try:
            joints = tuple(q_reset)
        except TypeError as exc:
            raise TypeError("q_reset must be a sequence of 7 real numbers") from exc
        if len(joints) != 7:
            raise ValueError(f"expected 7 reset joint targets, got {len(joints)}")
        validated: list[float] = []
        for value in joints:
            if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
                raise TypeError("q_reset values must be real numbers, not bool")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError("q_reset values must be finite")
            validated.append(numeric)

        with self._command_lock:
            self._check_health()
            self._resetting_events.clear()
            current = self._wait_for_reset_state(
                timeout_s=self.config.reset_ack_timeout_s,
                query_interval_s=self.config.reset_retry_interval_s,
            )
            if current:
                raise RuntimeError("cannot start reset while robot telemetry already reports RESETTING=1")

            self._command_sequence += 1
            frame = pack_command(
                self._command_sequence,
                validated,
                0,  # RESET_JOINT formally commands the gripper OPEN.
                flags=COMMAND_FLAG_RESET_JOINT,
            )
            ack_deadline = time.monotonic() + self.config.reset_ack_timeout_s
            next_retry = 0.0
            while True:
                now = time.monotonic()
                if now >= ack_deadline:
                    raise TimeoutError("robot did not acknowledge RESET_JOINT with RESETTING=1")
                if now >= next_retry:
                    self._send_command_frame(frame)
                    self._send_uds_request("GET_ROBOT_RESETTING")
                    next_retry = now + self.config.reset_retry_interval_s
                state = self._next_reset_state(ack_deadline, next_retry)
                if state is True:
                    break

            completion_deadline = time.monotonic() + self.config.reset_completion_timeout_s
            next_query = 0.0
            while True:
                now = time.monotonic()
                if now >= completion_deadline:
                    raise TimeoutError("robot did not complete RESET_JOINT with RESETTING=0")
                if now >= next_query:
                    self._send_uds_request("GET_ROBOT_RESETTING")
                    next_query = now + self.config.reset_retry_interval_s
                state = self._next_reset_state(completion_deadline, next_query)
                if state is False:
                    return

    def _wait_for_reset_state(self, *, timeout_s: float, query_interval_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        next_query = 0.0
        while True:
            now = time.monotonic()
            if now >= deadline:
                raise TimeoutError("robot RESETTING state was not available")
            if now >= next_query:
                self._send_uds_request("GET_ROBOT_RESETTING")
                next_query = now + query_interval_s
            state = self._next_reset_state(deadline, next_query)
            if state is not None:
                return state

    def _next_reset_state(self, deadline: float, next_request: float) -> bool | None:
        self._check_health()
        if self._resetting_events:
            return self._resetting_events.popleft()
        wait_s = min(deadline, next_request) - time.monotonic()
        if wait_s > 0:
            time.sleep(min(0.005, wait_s))
        return None

    def disconnect(self) -> None:
        if not self._connected and self._sensorhub is None:
            return
        self._connected = False
        if self._uds is not None:
            with suppress(OSError, RuntimeError):
                self._send_uds_request("SHUTDOWN", message="robot disconnect")
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
