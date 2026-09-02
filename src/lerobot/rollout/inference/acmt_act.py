"""ACMT-ACT 16/8 rollout adapter.

The queue and timestamp semantics are shared with the ACMT-DP adapter.  A
separate class name keeps policy dispatch explicit while avoiding a second,
slightly different implementation of the safety-critical queue.
"""

from .acmt_dp import ACMTDPInferenceEngine


class ACMTACTInferenceEngine(ACMTDPInferenceEngine):
    """Run an ``acmt_act`` policy with the causal 16-predict/8-execute queue."""

    pass


__all__ = ["ACMTACTInferenceEngine"]
