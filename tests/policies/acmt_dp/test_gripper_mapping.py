# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

from __future__ import annotations

import torch

from lerobot.policies.acmt_dp.gripper_mapping import (
    ACMTDPGripperGPOProcessorStep,
    fr3_pos_to_policy_gripper,
    policy_gripper_to_fr3_pos,
)
from lerobot.processor import PolicyProcessorPipeline


def test_policy_gripper_mapping_matches_deployed_gpo_endpoints() -> None:
    policy_values = torch.tensor([0.0, 0.5, 1.0])

    normalized_gpo = policy_gripper_to_fr3_pos(policy_values)

    torch.testing.assert_close(normalized_gpo, torch.tensor([1.0, 129.0 / 255.0, 3.0 / 255.0]))
    assert torch.equal(policy_values, torch.tensor([0.0, 0.5, 1.0]))


def test_policy_gripper_mapping_clamps_and_is_invertible_on_deployed_range() -> None:
    policy_values = torch.tensor([-1.0, 0.0, 0.25, 1.0, 2.0])

    normalized_gpo = policy_gripper_to_fr3_pos(policy_values)
    restored = fr3_pos_to_policy_gripper(normalized_gpo)

    torch.testing.assert_close(normalized_gpo, torch.tensor([1.0, 1.0, 192.0 / 255.0, 3.0 / 255.0, 3.0 / 255.0]))
    torch.testing.assert_close(restored, torch.tensor([0.0, 0.0, 0.25, 1.0, 1.0]))


def test_policy_gripper_processor_only_changes_the_eighth_action() -> None:
    step = ACMTDPGripperGPOProcessorStep()
    action = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.0]])

    processed = step.action(action)

    torch.testing.assert_close(processed[0, :7], action[0, :7])
    assert processed[0, 7].item() == 1.0
    assert torch.equal(action[0, 7:], torch.tensor([0.0]))
    assert step.get_config() == {"action_index": 7}


def test_current_peg_artifacts_load_the_policy_gripper_processor() -> None:
    for mode in ("none", "real"):
        pipeline = PolicyProcessorPipeline.from_pretrained(
            f"outputs/acmt_dp/peg/{mode}/seed42/pretrained_model",
            config_filename="policy_postprocessor.json",
        )
        assert isinstance(pipeline.steps[-1], ACMTDPGripperGPOProcessorStep)
        processed = pipeline.process_action(torch.tensor([[0.0] * 7 + [0.0]]))
        assert processed[0, 7].item() == 1.0
