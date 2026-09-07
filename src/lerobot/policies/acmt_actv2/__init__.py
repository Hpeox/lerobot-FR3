"""Three-camera ACMT-ACT policy variant.

The v2 policy deliberately keeps the v3 ACT/tactile network contract while
removing the top camera from the policy observation. It is registered under a
separate policy type and checkpoint schema so four-camera checkpoints cannot be
loaded accidentally.
"""

from .configuration_acmt_actv2 import ACMTACTV2Config
from .modeling_acmt_actv2 import ACMTACTV2Policy

__all__ = ["ACMTACTV2Config", "ACMTACTV2Policy"]
