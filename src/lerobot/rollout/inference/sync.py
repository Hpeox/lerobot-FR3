# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Synchronous inference engine: inline policy call per control tick."""

from __future__ import annotations

import logging
import math
import threading
from contextlib import nullcontext
from copy import copy
from typing import Any

import torch

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import make_robot_action, prepare_observation_for_inference
from lerobot.processor import PolicyProcessorPipeline

from .base import InferenceEngine

logger = logging.getLogger(__name__)

JOINT_POSITION_KEYS = tuple(f"fr3_joint{index}.pos" for index in range(1, 8))


# TODO(Steven): support relative-action policies.  The per-tick flow refreshes
# ``RelativeActionsProcessorStep._last_state`` every call, so cached chunk
# actions popped on later ticks get reanchored to the *current* robot state and
# absolute targets drift through the chunk.  Relative-action policies are
# rejected at context-build time today; RTC postprocesses the whole chunk and
# is unaffected.
#
# Candidate fix: drive the policy via ``predict_action_chunk`` and serve a
# local FIFO of postprocessed actions.  Eliminates drift by construction and
# saves per-tick pre/post work, but bypasses ``select_action`` — needs
# fallbacks for SAC (raises), ACT temporal ensembling (ensembler lives in
# ``select_action``), and Diffusion-family (obs-history queues populated as a
# side effect of ``select_action``).


class SyncInferenceEngine(InferenceEngine):
    """Inline synchronous inference: compute one action per call.

    ``get_action`` runs the full policy pipeline (pre/post-processor +
    ``select_action``) on the given observation frame and returns a
    CPU action tensor reordered to match the dataset action keys.
    """

    def __init__(
        self,
        policy: PreTrainedPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        dataset_features: dict,
        ordered_action_keys: list[str],
        task: str,
        device: str | None,
        robot_type: str,
    ) -> None:
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._dataset_features = dataset_features
        self._ordered_action_keys = ordered_action_keys
        self._task = task
        self._device = torch.device(device or "cpu")
        self._robot_type = robot_type
        self._diagnostic_lock = threading.RLock()
        self._first_action_diagnostic_emitted = False
        logger.info(
            "SyncInferenceEngine initialized (device=%s, action_keys=%d)",
            self._device,
            len(ordered_action_keys),
        )

    def start(self) -> None:
        """No background resources to start."""
        logger.info("SyncInferenceEngine started (inline mode — no background thread)")

    def stop(self) -> None:
        """No background resources to stop."""
        logger.info("SyncInferenceEngine stopped")

    def reset(self) -> None:
        """Reset the policy and pre/post-processors."""
        logger.info("Resetting sync inference state (policy + processors)")
        self._policy.reset()
        self._preprocessor.reset()
        self._postprocessor.reset()
        with self._diagnostic_lock:
            self._first_action_diagnostic_emitted = False

    @staticmethod
    def _finite_joint_vector(source: dict[str, Any] | None) -> tuple[float, ...] | None:
        if source is None:
            return None
        try:
            values = tuple(float(source[key]) for key in JOINT_POSITION_KEYS)
        except (KeyError, TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in values):
            return None
        return values

    @staticmethod
    def _max_abs_delta(
        left: tuple[float, ...] | None,
        right: tuple[float, ...] | None,
    ) -> float | None:
        if left is None or right is None:
            return None
        return max(abs(first - second) for first, second in zip(left, right, strict=True))

    def record_first_action_diagnostic(
        self,
        observation: dict[str, Any],
        planned_action: dict[str, Any],
        sent_action: dict[str, Any],
    ) -> None:
        """Log one accepted ACMT-DP action for diagnosing target jumps."""
        if getattr(self._policy, "name", None) not in {"acmt_dp", "acmt_dp_v5"}:
            return
        with self._diagnostic_lock:
            if self._first_action_diagnostic_emitted:
                return
            self._first_action_diagnostic_emitted = True

            current_q = self._finite_joint_vector(observation)
            planned_q = self._finite_joint_vector(planned_action)
            sent_q = self._finite_joint_vector(sent_action)
            logger.info(
                "ACMT-DP first action diagnostic: current_q=%s planned_q=%s sent_q=%s "
                "max_abs_delta=%s max_abs_processor_delta=%s",
                current_q,
                planned_q,
                sent_q,
                "n/a" if (delta := self._max_abs_delta(current_q, sent_q)) is None else f"{delta:.6f}",
                "n/a"
                if (processor_delta := self._max_abs_delta(planned_q, sent_q)) is None
                else f"{processor_delta:.6f}",
            )

    def notify_action_executed(self, action: torch.Tensor, observation: dict | None = None) -> None:
        """Forward accepted ACMT-DP actions to its causal policy state.

        The ordinary synchronous backend has no action queue of its own.  The
        Native-DP policy still exposes an optional feedback hook for the
        ``tactigen`` mode, so preserve that hook without changing other
        synchronous policies.
        """
        if getattr(self._policy, "name", None) not in {"acmt_dp", "acmt_dp_v5"}:
            return
        feedback = getattr(self._policy, "notify_action_executed", None)
        if callable(feedback):
            feedback(action.to(self._device), observation)

    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        """Run the full inference pipeline on ``obs_frame`` and return an action tensor."""
        if obs_frame is None:
            return None
        # Shallow copy is intentional: the caller (`send_next_action`) builds
        # ``obs_frame`` fresh per tick via ``build_dataset_frame``, so the
        # tensor/array values are not shared with any other reader.
        observation = copy(obs_frame)
        autocast_ctx = (
            torch.autocast(device_type=self._device.type)
            if self._device.type == "cuda" and self._policy.config.use_amp
            else nullcontext()
        )
        with torch.inference_mode(), autocast_ctx:
            observation = prepare_observation_for_inference(
                observation, self._device, self._task, self._robot_type
            )
            observation = self._preprocessor(observation)
            action = self._policy.select_action(observation)
            action = self._postprocessor(action)
        action_tensor = action.squeeze(0).cpu()

        # Reorder to match dataset action ordering so the caller can treat
        # the returned tensor uniformly across backends.
        action_dict = make_robot_action(action_tensor, self._dataset_features)
        return torch.tensor([action_dict[k] for k in self._ordered_action_keys])
