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

    pass


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
