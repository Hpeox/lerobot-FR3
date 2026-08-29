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

"""Minimal tests for the rollout module's public API."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

# ---------------------------------------------------------------------------
# Import smoke tests
# ---------------------------------------------------------------------------


def test_rollout_top_level_imports():
    import lerobot.rollout

    for name in lerobot.rollout.__all__:
        assert hasattr(lerobot.rollout, name), f"Missing export: {name}"


def test_inference_submodule_imports():
    import lerobot.rollout.inference

    for name in lerobot.rollout.inference.__all__:
        assert hasattr(lerobot.rollout.inference, name), f"Missing export: {name}"


def test_strategies_submodule_imports():
    import lerobot.rollout.strategies

    for name in lerobot.rollout.strategies.__all__:
        assert hasattr(lerobot.rollout.strategies, name), f"Missing export: {name}"


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


def test_strategy_config_types():
    from lerobot.rollout import (
        BaseStrategyConfig,
        ControlledStrategyConfig,
        DAggerStrategyConfig,
        EpisodicStrategyConfig,
        HighlightStrategyConfig,
        SentryStrategyConfig,
    )

    assert BaseStrategyConfig().type == "base"
    assert ControlledStrategyConfig().type == "controlled"
    assert SentryStrategyConfig().type == "sentry"
    assert HighlightStrategyConfig().type == "highlight"
    assert DAggerStrategyConfig().type == "dagger"
    assert EpisodicStrategyConfig().type == "episodic"


def test_dagger_config_invalid_input_device():
    from lerobot.rollout import DAggerStrategyConfig

    with pytest.raises(ValueError, match="input_device must be 'keyboard' or 'pedal'"):
        DAggerStrategyConfig(input_device="joystick")


def test_dagger_config_defaults():
    from lerobot.rollout import DAggerStrategyConfig

    cfg = DAggerStrategyConfig()
    assert cfg.num_episodes is None
    assert cfg.record_autonomous is False
    assert cfg.input_device == "keyboard"


def test_inference_config_types():
    from lerobot.rollout import RTCInferenceConfig, SyncInferenceConfig

    assert SyncInferenceConfig().type == "sync"

    rtc = RTCInferenceConfig()
    assert rtc.type == "rtc"
    assert rtc.queue_threshold == 30
    assert rtc.rtc is not None


def test_sentry_config_defaults():
    from lerobot.rollout import SentryStrategyConfig

    cfg = SentryStrategyConfig()
    assert cfg.upload_every_n_episodes == 5
    assert cfg.target_video_file_size_mb is None


# ---------------------------------------------------------------------------
# RolloutRingBuffer
# ---------------------------------------------------------------------------


def test_ring_buffer_append_and_eviction():
    from lerobot.rollout.ring_buffer import RolloutRingBuffer

    buf = RolloutRingBuffer(max_seconds=0.5, max_memory_mb=100.0, fps=10.0)
    # max_frames = 5
    for i in range(8):
        buf.append({"val": i})
    assert len(buf) == 5


def test_ring_buffer_drain():
    from lerobot.rollout.ring_buffer import RolloutRingBuffer

    buf = RolloutRingBuffer(max_seconds=1.0, max_memory_mb=100.0, fps=10.0)
    for i in range(3):
        buf.append({"val": i})
    frames = buf.drain()
    assert len(frames) == 3
    assert len(buf) == 0
    assert buf.estimated_bytes == 0


def test_ring_buffer_clear():
    from lerobot.rollout.ring_buffer import RolloutRingBuffer

    buf = RolloutRingBuffer(max_seconds=1.0, max_memory_mb=100.0, fps=10.0)
    buf.append({"val": 1})
    buf.clear()
    assert len(buf) == 0
    assert buf.estimated_bytes == 0


def test_ring_buffer_tensor_bytes():
    from lerobot.rollout.ring_buffer import RolloutRingBuffer

    buf = RolloutRingBuffer(max_seconds=1.0, max_memory_mb=100.0, fps=10.0)
    t = torch.zeros(100, dtype=torch.float32)  # 400 bytes
    buf.append({"tensor": t})
    assert buf.estimated_bytes >= 400


# ---------------------------------------------------------------------------
# ThreadSafeRobot
# ---------------------------------------------------------------------------


def test_thread_safe_robot_delegates():
    from lerobot.rollout.robot_wrapper import ThreadSafeRobot
    from tests.mocks.mock_robot import MockRobot, MockRobotConfig

    robot = MockRobot(MockRobotConfig(n_motors=3))
    robot.connect()
    wrapper = ThreadSafeRobot(robot)

    obs = wrapper.get_observation()
    assert "motor_1.pos" in obs
    assert "motor_2.pos" in obs
    assert "motor_3.pos" in obs

    action = {"motor_1.pos": 0.0, "motor_2.pos": 1.0, "motor_3.pos": 2.0}
    result = wrapper.send_action(action)
    assert result == action

    robot.disconnect()


def test_thread_safe_robot_properties():
    from lerobot.rollout.robot_wrapper import ThreadSafeRobot
    from tests.mocks.mock_robot import MockRobot, MockRobotConfig

    robot = MockRobot(MockRobotConfig(n_motors=3))
    robot.connect()
    wrapper = ThreadSafeRobot(robot)

    assert wrapper.name == "mock_robot"
    assert "motor_1.pos" in wrapper.observation_features
    assert "motor_1.pos" in wrapper.action_features
    assert wrapper.is_connected is True
    assert wrapper.inner is robot

    robot.disconnect()


# ---------------------------------------------------------------------------
# Strategy factory
# ---------------------------------------------------------------------------


def test_create_strategy_dispatches():
    from lerobot.rollout import (
        BaseStrategy,
        BaseStrategyConfig,
        ControlledStrategy,
        ControlledStrategyConfig,
        DAggerStrategy,
        DAggerStrategyConfig,
        EpisodicStrategy,
        EpisodicStrategyConfig,
        SentryStrategy,
        SentryStrategyConfig,
        create_strategy,
    )

    assert isinstance(create_strategy(BaseStrategyConfig()), BaseStrategy)
    assert isinstance(create_strategy(ControlledStrategyConfig()), ControlledStrategy)
    assert isinstance(create_strategy(SentryStrategyConfig()), SentryStrategy)
    assert isinstance(create_strategy(DAggerStrategyConfig()), DAggerStrategy)
    assert isinstance(create_strategy(EpisodicStrategyConfig()), EpisodicStrategy)


def test_create_strategy_unknown_raises():
    from lerobot.rollout import create_strategy

    cfg = MagicMock()
    cfg.type = "bogus"
    with pytest.raises(ValueError, match="Unknown strategy type"):
        create_strategy(cfg)


# ---------------------------------------------------------------------------
# Inference factory
# ---------------------------------------------------------------------------


def test_create_inference_engine_sync():
    from lerobot.rollout import SyncInferenceConfig, SyncInferenceEngine, create_inference_engine

    engine = create_inference_engine(
        SyncInferenceConfig(),
        policy=MagicMock(),
        preprocessor=MagicMock(),
        postprocessor=MagicMock(),
        robot_wrapper=MagicMock(robot_type="mock"),
        hw_features={},
        dataset_features={},
        ordered_action_keys=["k"],
        task="test",
        fps=30.0,
        device="cpu",
    )
    assert isinstance(engine, SyncInferenceEngine)


def test_create_inference_engine_acmt_dp_uses_inline_sync_backend():
    from lerobot.rollout import SyncInferenceConfig, SyncInferenceEngine, create_inference_engine

    policy = SimpleNamespace(
        name="acmt_dp",
        robot_type="fr3",
        config=SimpleNamespace(use_amp=False),
    )
    engine = create_inference_engine(
        SyncInferenceConfig(),
        policy=policy,
        preprocessor=MagicMock(),
        postprocessor=MagicMock(),
        robot_wrapper=MagicMock(robot_type="fr3"),
        hw_features={},
        dataset_features={},
        ordered_action_keys=["k"],
        task="test",
        fps=30.0,
        device="cpu",
    )
    assert type(engine) is SyncInferenceEngine
    assert not hasattr(engine, "_queue")


def test_sync_engine_forwards_acmt_action_feedback():
    from lerobot.rollout import SyncInferenceEngine

    policy = MagicMock(name="policy")
    policy.name = "acmt_dp"
    policy.config.use_amp = False
    engine = SyncInferenceEngine(
        policy=policy,
        preprocessor=MagicMock(),
        postprocessor=MagicMock(),
        dataset_features={},
        ordered_action_keys=[],
        task="test",
        device="cpu",
        robot_type="fr3",
    )
    action = torch.ones(1, 8)
    observation = {"observation": 1}

    engine.notify_action_executed(action, observation)

    policy.notify_action_executed.assert_called_once_with(action, observation)


def test_sync_engine_acmt_action_diagnostic_is_once_per_reset(caplog):
    from lerobot.rollout import SyncInferenceEngine

    policy = SimpleNamespace(
        name="acmt_dp",
        config=SimpleNamespace(use_amp=False),
        reset=MagicMock(),
    )
    engine = SyncInferenceEngine(
        policy=policy,
        preprocessor=MagicMock(),
        postprocessor=MagicMock(),
        dataset_features={},
        ordered_action_keys=[],
        task="test",
        device="cpu",
        robot_type="fr3",
    )
    q = {f"fr3_joint{index}.pos": float(index) for index in range(1, 8)}

    with caplog.at_level("INFO", logger="lerobot.rollout.inference.sync"):
        engine.record_first_action_diagnostic(q, q, q)
        engine.record_first_action_diagnostic(q, q, q)

    assert sum("ACMT-DP first action diagnostic" in record.message for record in caplog.records) == 1

    engine.reset()
    with caplog.at_level("INFO", logger="lerobot.rollout.inference.sync"):
        engine.record_first_action_diagnostic(q, q, q)
    assert sum("ACMT-DP first action diagnostic" in record.message for record in caplog.records) == 2


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def test_estimate_max_episode_seconds_no_video():
    from lerobot.rollout.strategies import estimate_max_episode_seconds

    assert estimate_max_episode_seconds({}, fps=30.0) == 300.0


def test_estimate_max_episode_seconds_with_video():
    from lerobot.rollout.strategies import estimate_max_episode_seconds

    features = {"cam": {"dtype": "video", "shape": (480, 640, 3)}}
    result = estimate_max_episode_seconds(features, fps=30.0)
    assert result > 0
    # With a real camera, duration should differ from the fallback
    assert result != 300.0


def test_safe_push_to_hub():
    from lerobot.rollout.strategies import safe_push_to_hub

    ds = MagicMock()
    ds.num_episodes = 0
    assert safe_push_to_hub(ds) is False
    ds.push_to_hub.assert_not_called()

    ds.num_episodes = 5
    assert safe_push_to_hub(ds, tags=["test"]) is True
    ds.push_to_hub.assert_called_once_with(tags=["test"], private=False)


# ---------------------------------------------------------------------------
# DAgger state machine
# ---------------------------------------------------------------------------


def test_dagger_full_transition_cycle():
    from lerobot.rollout.strategies import DAggerEvents, DAggerPhase

    events = DAggerEvents()
    assert events.phase == DAggerPhase.AUTONOMOUS

    # AUTONOMOUS -> PAUSED
    events.request_transition("pause_resume")
    old, new = events.consume_transition()
    assert (old, new) == (DAggerPhase.AUTONOMOUS, DAggerPhase.PAUSED)

    # PAUSED -> CORRECTING
    events.request_transition("correction")
    old, new = events.consume_transition()
    assert (old, new) == (DAggerPhase.PAUSED, DAggerPhase.CORRECTING)

    # CORRECTING -> PAUSED
    events.request_transition("correction")
    old, new = events.consume_transition()
    assert (old, new) == (DAggerPhase.CORRECTING, DAggerPhase.PAUSED)

    # PAUSED -> AUTONOMOUS
    events.request_transition("pause_resume")
    old, new = events.consume_transition()
    assert (old, new) == (DAggerPhase.PAUSED, DAggerPhase.AUTONOMOUS)


def test_dagger_invalid_transition_ignored():
    from lerobot.rollout.strategies import DAggerEvents, DAggerPhase

    events = DAggerEvents()
    events.request_transition("correction")  # Not valid from AUTONOMOUS
    assert events.consume_transition() is None
    assert events.phase == DAggerPhase.AUTONOMOUS


def test_dagger_events_reset():
    from lerobot.rollout.strategies import DAggerEvents, DAggerPhase

    events = DAggerEvents()
    events.request_transition("pause_resume")
    events.consume_transition()  # -> PAUSED
    events.upload_requested.set()
    events.reset()
    assert events.phase == DAggerPhase.AUTONOMOUS
    assert not events.upload_requested.is_set()


# ---------------------------------------------------------------------------
# Context dataclass
# ---------------------------------------------------------------------------


def test_rollout_context_fields():
    from lerobot.rollout import RolloutContext

    field_names = {f.name for f in dataclasses.fields(RolloutContext)}
    assert field_names == {"runtime", "hardware", "policy", "processors", "data"}


def test_visual_feature_coverage_requires_every_policy_camera_but_allows_robot_superset():
    from lerobot.rollout.context import _validate_visual_feature_coverage

    cam1 = {
        "observation.images.camera.cam1.rgb",
        "observation.images.camera.cam1.depth",
    }
    cam2 = {
        "observation.images.camera.cam2.rgb",
        "observation.images.camera.cam2.depth",
    }
    _validate_visual_feature_coverage(cam1, cam1 | cam2)
    with pytest.raises(ValueError, match="camera.cam2"):
        _validate_visual_feature_coverage(cam1 | cam2, cam1)


def test_create_inference_engine_acmt_dp_v5_uses_inline_sync_backend():
    from lerobot.rollout import SyncInferenceConfig, SyncInferenceEngine, create_inference_engine

    policy = SimpleNamespace(
        name="acmt_dp_v5",
        robot_type="fr3",
        config=SimpleNamespace(use_amp=False),
    )
    engine = create_inference_engine(
        SyncInferenceConfig(),
        policy=policy,
        preprocessor=MagicMock(),
        postprocessor=MagicMock(),
        robot_wrapper=MagicMock(robot_type="fr3"),
        hw_features={},
        dataset_features={},
        ordered_action_keys=["k"],
        task="gear_insert_big2small",
        fps=30.0,
        device="cpu",
    )
    assert type(engine) is SyncInferenceEngine
    assert not hasattr(engine, "_queue")

def test_sync_engine_forwards_acmt_dp_v5_action_feedback():
    from lerobot.rollout import SyncInferenceEngine

    policy = MagicMock(name="policy")
    policy.name = "acmt_dp_v5"
    policy.config.use_amp = False
    engine = SyncInferenceEngine(
        policy=policy,
        preprocessor=MagicMock(),
        postprocessor=MagicMock(),
        dataset_features={},
        ordered_action_keys=[],
        task="gear_insert_big2small",
        device="cpu",
        robot_type="fr3",
    )
    action = torch.ones(1, 8)
    observation = {"observation": 1}

    engine.notify_action_executed(action, observation)

    policy.notify_action_executed.assert_called_once_with(action, observation)
