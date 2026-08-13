# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persistent rollout worker driven by an external controller over UDS."""

from __future__ import annotations

import contextlib
import logging
import sys
import time

from lerobot.datasets import VideoEncodingManager
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.robot_utils import precise_sleep

from ..configs import ControlledStrategyConfig
from ..context import RolloutContext
from ..control_uds import (
    OPERATIONS,
    ControlledCommand,
    ControlledUDSProtocolError,
    ControlledUDSServer,
)
from .core import RolloutStrategy, send_next_action

logger = logging.getLogger(__name__)

WAIT_INITIALIZE = "WAIT_INITIALIZE"
INITIALIZING = "INITIALIZING"
WAIT_START = "WAIT_START"
RUNNING = "RUNNING"
SHUTTING_DOWN = "SHUTTING_DOWN"
FAIL_STOPPING = "FAIL_STOPPING"


class ControlledFailStop(RuntimeError):
    """Non-success termination requested by the external controller."""


class ControlledStrategy(RolloutStrategy):
    """Linear, single-threaded rollout lifecycle for an external controller."""

    config: ControlledStrategyConfig

    def __init__(self, config: ControlledStrategyConfig) -> None:
        super().__init__(config)
        self._control: ControlledUDSServer | None = None
        self._phase = WAIT_INITIALIZE
        self._dataset_session_finalized = False
        self._engine_stopped = False
        self._teardown_complete = False

    def setup(self, ctx: RolloutContext) -> None:
        """Bind the control socket and start the long-lived paused inference engine."""

        self._control = ControlledUDSServer(self.config.control_socket_path)
        self._init_engine(ctx)
        self._engine.pause()
        logger.info("Controlled strategy ready on %s", self.config.control_socket_path)

    def run(self, ctx: RolloutContext) -> None:
        """Run the persistent Controlled session until an explicit or fatal termination."""

        dataset = ctx.data.dataset
        manager = VideoEncodingManager(dataset) if dataset is not None else contextlib.nullcontext()
        session_error: BaseException | None = None
        try:
            try:
                with manager:
                    try:
                        self._run_session(ctx)
                    except ControlledFailStop as exc:
                        session_error = exc
                        self._clear_episode_best_effort(dataset)
                        raise
                    except BaseException as exc:
                        session_error = exc
                        self._clear_episode_best_effort(dataset)
                        self._publish_error_best_effort(exc)
                        raise
            except BaseException as exc:
                if session_error is not None and exc is not session_error:
                    logger.error("Dataset/video finalization failed while handling %s: %s", session_error, exc)
                    raise session_error.with_traceback(session_error.__traceback__) from exc
                if session_error is None:
                    self._clear_episode_best_effort(dataset)
                    self._publish_error_best_effort(exc, code="dataset_finalize_failed")
                raise
        finally:
            # VideoEncodingManager owns the single dataset finalization when present.
            self._dataset_session_finalized = dataset is not None

    def _run_session(self, ctx: RolloutContext) -> None:
        control = self._require_control()
        control.accept()
        self._phase = WAIT_INITIALIZE
        control.publish_status("READY", phase=self._phase)

        while True:
            operation = self._wait_for_command({"INITIALIZE", "SHUTDOWN", "FAIL_STOP"})
            if operation == "SHUTDOWN":
                self._graceful_shutdown(ctx)
                return
            if operation == "FAIL_STOP":
                self._fatal_stop(ctx)

            self._transition_to(INITIALIZING, "INITIALIZING")
            ctx.hardware.robot_wrapper.inner.initialize_rollout()
            self._transition_to(WAIT_START, "INITIALIZED")

            operation = self._wait_for_command({"START", "SHUTDOWN", "FAIL_STOP"})
            if operation == "SHUTDOWN":
                self._graceful_shutdown(ctx)
                return
            if operation == "FAIL_STOP":
                self._fatal_stop(ctx)

            self._prepare_episode(ctx)
            self._engine.resume()
            self._transition_to(RUNNING, "STARTED")
            outcome = self._run_rollout(ctx)

            if outcome == "SHUTDOWN":
                self._clear_episode(ctx.data.dataset)
                self._graceful_shutdown(ctx)
                return
            if outcome == "FAIL_STOP":
                self._clear_episode(ctx.data.dataset)
                self._fatal_stop(ctx)

            if outcome in {"STOPPED", "COMPLETED"}:
                self._save_episode(ctx.data.dataset)
            else:
                self._clear_episode(ctx.data.dataset)
            self._transition_to(WAIT_INITIALIZE, outcome)

    def _prepare_episode(self, ctx: RolloutContext) -> None:
        engine = self._engine
        reset_controlled = getattr(engine, "reset_for_controlled_rollout", None)
        if callable(reset_controlled):
            reset_controlled()
        else:
            engine.reset()
        self._interpolator.reset()
        self._cached_obs_processed = None

    def _run_rollout(self, ctx: RolloutContext) -> str:
        cfg = ctx.runtime.cfg
        robot = ctx.hardware.robot_wrapper
        dataset = ctx.data.dataset
        task = (cfg.dataset.single_task or cfg.task) if cfg.dataset is not None else cfg.task
        interpolator = self._interpolator
        control_interval = interpolator.get_control_interval(cfg.fps)
        start_time = time.perf_counter()

        try:
            while True:
                loop_start = time.perf_counter()
                if ctx.runtime.shutdown_event.is_set():
                    raise RuntimeError("Controlled rollout interrupted by process shutdown signal")
                if self._engine.failed:
                    raise RuntimeError("Controlled rollout inference engine failed")

                operation = self._poll_command({"STOP", "ABORT", "SHUTDOWN", "FAIL_STOP"})
                if operation is not None:
                    return {
                        "STOP": "STOPPED",
                        "ABORT": "ABORTED",
                        "SHUTDOWN": "SHUTDOWN",
                        "FAIL_STOP": "FAIL_STOP",
                    }[operation]

                if cfg.duration > 0 and (time.perf_counter() - start_time) >= cfg.duration:
                    return "COMPLETED"

                obs = robot.get_observation()
                obs_processed = self._process_observation_and_notify(ctx.processors, obs)

                if self._handle_warmup(cfg.use_torch_compile, loop_start, control_interval):
                    continue

                action_dict = send_next_action(obs_processed, obs, ctx, interpolator)
                if dataset is not None and action_dict is not None:
                    obs_frame = build_dataset_frame(ctx.data.dataset_features, obs_processed, prefix=OBS_STR)
                    action_frame = build_dataset_frame(ctx.data.dataset_features, action_dict, prefix=ACTION)
                    dataset.add_frame({**obs_frame, **action_frame, "task": task})
                self._log_telemetry(obs_processed, action_dict, ctx.runtime)

                dt = time.perf_counter() - loop_start
                if (sleep_t := control_interval - dt) > 0:
                    precise_sleep(sleep_t)
                else:
                    logger.warning(
                        "Controlled loop is running slower (%.1f Hz) than target FPS (%.1f Hz)",
                        1 / dt,
                        cfg.fps,
                    )
        finally:
            self._engine.pause()

    def _wait_for_command(self, allowed: set[str]) -> str:
        while True:
            operation = self._receive_command(allowed, blocking=True)
            if operation is not None:
                return operation

    def _poll_command(self, allowed: set[str]) -> str | None:
        return self._receive_command(allowed, blocking=False)

    def _receive_command(self, allowed: set[str], *, blocking: bool) -> str | None:
        control = self._require_control()
        try:
            command = control.recv(blocking=blocking)
        except ControlledUDSProtocolError as exc:
            logger.warning("Dropping malformed Controlled UDS command in %s: %s", self._phase, exc)
            return None
        if command is None:
            return None
        return self._validate_and_ack(command, allowed)

    def _validate_and_ack(self, command: ControlledCommand, allowed: set[str]) -> str | None:
        control = self._require_control()
        if not control.consume_sequence(command.sequence):
            control.send_ack(
                command,
                accepted=False,
                code="stale_sequence",
                phase=self._phase,
                message="sequence must be strictly increasing",
            )
            return None
        if command.operation not in OPERATIONS:
            control.send_ack(
                command,
                accepted=False,
                code="unsupported_operation",
                phase=self._phase,
                message="unsupported operation",
            )
            return None
        if command.operation not in allowed:
            control.send_ack(
                command,
                accepted=False,
                code="invalid_phase",
                phase=self._phase,
                message="operation is not valid in the current phase",
            )
            return None
        control.send_ack(command, accepted=True, code="accepted", phase=self._phase)
        return command.operation

    def _transition_to(self, phase: str, status: str) -> None:
        """Drain old transport input before committing and publishing a new phase."""

        control = self._require_control()
        discarded = control.drain()
        if discarded:
            logger.warning("Discarded %d stale Controlled UDS packet(s) leaving %s", discarded, self._phase)
        self._phase = phase
        control.publish_status(status, phase=phase)

    def _graceful_shutdown(self, ctx: RolloutContext) -> None:
        self._engine.pause()
        self._clear_episode(ctx.data.dataset)
        self._transition_to(SHUTTING_DOWN, "SHUTTING_DOWN")
        ctx.hardware.robot_wrapper.inner.return_to_home()

    def _fatal_stop(self, ctx: RolloutContext) -> None:
        self._engine.pause()
        self._clear_episode_best_effort(ctx.data.dataset)
        self._transition_to(FAIL_STOPPING, "FAIL_STOPPING")
        raise ControlledFailStop("Controlled FAIL_STOP requested")

    @staticmethod
    def _save_episode(dataset) -> None:
        if dataset is not None and dataset.has_pending_frames():
            dataset.save_episode()

    @staticmethod
    def _clear_episode(dataset) -> None:
        if dataset is not None and dataset.has_pending_frames():
            dataset.clear_episode_buffer()

    @staticmethod
    def _clear_episode_best_effort(dataset) -> None:
        if dataset is None:
            return
        with contextlib.suppress(Exception):
            if dataset.has_pending_frames():
                dataset.clear_episode_buffer()

    def _publish_error_best_effort(self, exc: BaseException, *, code: str | None = None) -> None:
        logger.error("Controlled rollout failed in phase %s: %s", self._phase, exc)
        control = self._control
        if control is None or not control.connected:
            return
        if code is None:
            code = "return_home_failed" if self._phase == SHUTTING_DOWN else "internal_error"
        with contextlib.suppress(Exception):
            control.publish_status("ERROR", phase=self._phase, code=code, message=str(exc))

    def _require_control(self) -> ControlledUDSServer:
        if self._control is None:
            raise RuntimeError("Controlled strategy is not set up")
        return self._control

    def teardown(self, ctx: RolloutContext) -> None:
        """Stop resources without initiating robot motion."""

        if self._teardown_complete:
            return
        self._teardown_complete = True
        active_error = sys.exc_info()[1]
        cleanup_errors: list[BaseException] = []

        if self._engine is not None and not self._engine_stopped:
            try:
                self._engine.stop()
            except BaseException as exc:
                cleanup_errors.append(exc)
            self._engine_stopped = True

        dataset = ctx.data.dataset
        if dataset is not None and not self._dataset_session_finalized:
            # ``run`` sets ownership before entering VideoEncodingManager. If
            # setup failed before run, finalize the context-created dataset here.
            try:
                dataset.finalize()
                self._dataset_session_finalized = True
            except BaseException as exc:
                cleanup_errors.append(exc)

        if self._control is not None:
            try:
                self._control.close()
            except BaseException as exc:
                cleanup_errors.append(exc)

        robot = ctx.hardware.robot_wrapper.inner
        if robot.is_connected:
            try:
                robot.disconnect()
            except BaseException as exc:
                cleanup_errors.append(exc)
        teleop = ctx.hardware.teleop
        if teleop is not None and teleop.is_connected:
            try:
                teleop.disconnect()
            except BaseException as exc:
                cleanup_errors.append(exc)

        for exc in cleanup_errors:
            logger.error("Controlled teardown cleanup failed: %s", exc)
        if cleanup_errors and active_error is None:
            raise cleanup_errors[0]
        logger.info("Controlled strategy teardown complete")
