"""Inference adapters for the ACMT-ACT policies.

The original ``acmt_act`` policy retains its historical 16/8 queue.  The v2
policy uses the native ACT contract and therefore needs the ordinary
synchronous engine: ``select_action`` is called every tick and owns the
online temporal ensemble.
"""

from .acmt_dp import ACMTDPInferenceEngine
from .sync import SyncInferenceEngine


class ACMTACTInferenceEngine(ACMTDPInferenceEngine):
    """Run legacy ``acmt_act`` with its causal 16-predict/8-execute queue."""

    def _postprocess_plan(self, action, anchor_state):
        result = super()._postprocess_plan(action, anchor_state)
        config = self._policy.config
        if (
            self._plan_postprocess
            and getattr(config, "checkpoint_schema", None) == "acmt_act.v3"
            and getattr(config, "vision_backbone", None) == "resnet50"
        ):
            # Experiment: reverse normalized gPO after the existing processors,
            # retaining the deployed 3..255 endpoints and joint anchors.
            result = result.clone()
            result[..., 7] = (255.0 + 3.0) / 255.0 - result[..., 7]
        return result


class ACMTACTV2InferenceEngine(SyncInferenceEngine):
    """Run native ACMT-ACTv2 with one fresh 100-step plan per tick."""

    def __init__(self, *args, **kwargs):
        policy = kwargs.get("policy")
        if policy is None and args:
            policy = args[0]
        config = getattr(policy, "config", None)
        if config is None:
            raise ValueError("ACMT-ACTv2 inference requires a policy configuration")
        if (
            getattr(config, "checkpoint_schema", None),
            getattr(config, "checkpoint_schema_version", None),
        ) != ("acmt_actv2.dinov2_spatial.v1", 3):
            raise ValueError("ACMT-ACTv2 inference requires DINOv2 spatial schema version 3")
        if (
            getattr(config, "chunk_size", None),
            getattr(config, "n_action_steps", None),
            getattr(config, "action_execution_horizon", None),
        ) != (100, 1, 1):
            raise ValueError("ACMT-ACTv2 inference requires chunk_size=100 and one-step execution")
        super().__init__(*args, **kwargs)


__all__ = ["ACMTACTInferenceEngine", "ACMTACTV2InferenceEngine"]
