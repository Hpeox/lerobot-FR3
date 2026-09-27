"""Gear-only absolute joint bounds for the ACMT-ACT v3 rolling engine."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from lerobot.processor.relative_action_processor import AbsoluteActionsProcessorStep, RelativeActionsProcessorStep
from lerobot.rollout.inference.acmt_act import (
    ACMTACTInferenceEngine,
    GEAR_JOINT_LOWER,
    GEAR_JOINT_UPPER,
)
from lerobot.rollout.inference.acmt_dp import JOINT_POSITION_KEYS
from lerobot.utils.constants import ACTION


ACTION_NAMES = (*JOINT_POSITION_KEYS, "gripper.pos")


def test_gear_training_joint_extrema_match_deployment_contract() -> None:
    assert GEAR_JOINT_LOWER == (
        -0.054157, -0.047530, -0.278071, -2.433730, -0.266043, 1.734929, -1.189159,
    )
    assert GEAR_JOINT_UPPER == (
        0.122640, 0.541469, 0.444214, -1.391000, 0.221219, 2.523725, 0.942100,
    )


class _Policy:
    name = "acmt_act"

    def __init__(self, task_variant: str, names: tuple[str, ...], mode: str = "none") -> None:
        self.config = SimpleNamespace(
            checkpoint_schema="acmt_act.v3",
            checkpoint_schema_version=3,
            task_variant=task_variant,
            tactile_source=mode,
            control_hz=30.0,
            action_execution_horizon=8,
            tactile_history=4,
            pred_horizon=16,
            action_dim=8,
        )
        self.actions = torch.zeros(1, 16, 8)

    def _plan(self, _window: dict) -> torch.Tensor:
        return self.actions


class _Processor:
    def reset(self) -> None:
        pass


class _AbsoluteProcessor(_Processor):
    def __init__(self, names: tuple[str, ...]) -> None:
        self.relative_step = RelativeActionsProcessorStep(
            enabled=True, exclude_joints=["gripper"], action_names=list(names)
        )
        self.steps = [AbsoluteActionsProcessorStep(enabled=True, relative_step=self.relative_step)]
        self.calls = 0

    def __call__(self, action: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        anchor = self.relative_step.get_cached_state()
        assert anchor is not None
        result = action.clone()
        result[..., :7] += anchor[:, :7].unsqueeze(1)
        return result


def _engine(
    task_variant: str = "gear",
    names: tuple[str, ...] = ACTION_NAMES,
    mode: str = "none",
    postprocessor: _Processor | None = None,
) -> tuple[ACMTACTInferenceEngine, _Policy, _Processor]:
    policy = _Policy(task_variant, names, mode)
    postprocessor = postprocessor or _AbsoluteProcessor(names)
    engine = ACMTACTInferenceEngine(
        policy=policy,
        preprocessor=_Processor(),
        postprocessor=postprocessor,
        dataset_features={ACTION: {"names": list(names)}},
        ordered_action_keys=list(names),
        task="test",
        device="cpu",
        robot_type="fr3",
    )
    return engine, policy, postprocessor


@pytest.mark.parametrize("joint", range(7))
@pytest.mark.parametrize("position", ("below", "lower", "inside", "upper", "above"))
def test_gear_joint_bounds_after_absolute_restoration(joint: int, position: str, caplog) -> None:
    engine, _, postprocessor = _engine()
    try:
        anchor = torch.full((1, 8), 0.1)
        targets = [(low + high) / 2 for low, high in zip(GEAR_JOINT_LOWER, GEAR_JOINT_UPPER, strict=True)]
        low, high = GEAR_JOINT_LOWER[joint], GEAR_JOINT_UPPER[joint]
        targets[joint] = {
            "below": low - 0.01,
            "lower": low,
            "inside": (low + high) / 2,
            "upper": high,
            "above": high + 0.01,
        }[position]
        raw = torch.tensor([*(target - 0.1 for target in targets), 0.73]).reshape(1, 1, 8).expand(1, 16, 8)
        with caplog.at_level("WARNING", logger="lerobot.rollout.inference.acmt_act"):
            result = engine._postprocess_plan(raw, anchor)
        expected = min(max(targets[joint], low), high)
        assert result.shape == (1, 16, 8)
        assert result[0, :, joint].tolist() == pytest.approx([expected] * 16)
        for other in range(7):
            if other != joint:
                assert result[0, :, other].tolist() == pytest.approx([targets[other]] * 16)
        torch.testing.assert_close(result[..., 7], torch.full((1, 16), 0.73))
        assert postprocessor.calls == 1
        messages = [record.message for record in caplog.records if "joint bounds clipped plan" in record.message]
        assert len(messages) == (1 if position in {"below", "above"} else 0)
        if messages:
            assert f"values=16 joints={JOINT_POSITION_KEYS[joint]}" in messages[0]
    finally:
        engine.stop()


@pytest.mark.parametrize("mode", ("none", "real", "substitution"))
def test_all_gear_tactile_modes_use_same_boundaries(mode: str) -> None:
    engine, _, _ = _engine(mode=mode)
    try:
        raw = torch.full((1, 16, 8), 100.0)
        result = engine._postprocess_plan(raw, torch.zeros(1, 8))
        torch.testing.assert_close(result[0, :, :7], torch.tensor(GEAR_JOINT_UPPER).expand(16, 7))
        assert (result[..., 7] == 100.0).all()
    finally:
        engine.stop()


def test_reordered_action_names_clip_only_matching_joints() -> None:
    names = ("gripper.pos", JOINT_POSITION_KEYS[6], *JOINT_POSITION_KEYS[:6])
    engine, _, _ = _engine(names=names)
    try:
        raw = torch.zeros(1, 16, 8)
        raw[..., 0] = 0.72
        for index, name in enumerate(names[1:], start=1):
            raw[..., index] = GEAR_JOINT_UPPER[JOINT_POSITION_KEYS.index(name)] + 1.0
        result = engine._postprocess_plan(raw, torch.zeros(1, 8))
        assert (result[..., 0] == 0.72).all()
        for index, name in enumerate(names[1:], start=1):
            assert result[0, 0, index].item() == pytest.approx(GEAR_JOINT_UPPER[JOINT_POSITION_KEYS.index(name)])
    finally:
        engine.stop()


@pytest.mark.parametrize("names", (ACTION_NAMES[:-1], (*ACTION_NAMES[:-1], ACTION_NAMES[0])))
def test_gear_startup_rejects_missing_or_duplicate_action_names(names: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="exactly seven named joints"):
        _engine(names=names)


def test_gear_startup_requires_absolute_action_restoration() -> None:
    with pytest.raises(ValueError, match="absolute-action plan postprocessing"):
        _engine(postprocessor=_Processor())


def test_gear_rejects_nonfinite_absolute_plan() -> None:
    engine, _, _ = _engine()
    try:
        raw = torch.zeros(1, 16, 8)
        raw[0, 0, 0] = float("nan")
        with pytest.raises(ValueError, match="non-finite absolute actions"):
            engine._postprocess_plan(raw, torch.zeros(1, 8))
    finally:
        engine.stop()


def test_initial_and_replacement_plans_are_bounded_before_queueing() -> None:
    engine, policy, postprocessor = _engine()
    try:
        policy.actions = torch.full((1, 16, 8), 100.0)
        first = engine._plan_now({}, 0, anchor_state=torch.zeros(1, 8))
        engine._queue.install(first)
        assert len(engine._queue.snapshot()) == 16
        for action in engine._queue.snapshot():
            torch.testing.assert_close(action[:7], torch.tensor(GEAR_JOINT_UPPER))
        for _ in range(8):
            assert engine._queue.pop() is not None
        policy.actions = torch.full((1, 16, 8), -100.0)
        second = engine._plan_now({}, 1, anchor_state=torch.zeros(1, 8))
        engine._queue.replace_future(second.actions, first.start_time + 8 / 30, 1)
        assert len(engine._queue.snapshot()) == 16
        for action in engine._queue.snapshot():
            torch.testing.assert_close(action[:7], torch.tensor(GEAR_JOINT_LOWER))
        assert postprocessor.calls == 2
    finally:
        engine.stop()


def test_peg_remains_unclipped() -> None:
    engine, _, _ = _engine(task_variant="peg")
    try:
        raw = torch.full((1, 16, 8), 100.0)
        result = engine._postprocess_plan(raw, torch.zeros(1, 8))
        torch.testing.assert_close(result, raw)
    finally:
        engine.stop()
