# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from lerobot.utils.action_interpolator import ActionInterpolator as ActionInterpolator

from .acmt_dp import (
    ACMTDPCenter480ProcessorStep as ACMTDPCenter480ProcessorStep,
    ACMTDPConfig as ACMTDPConfig,
    ACMTDPPolicy as ACMTDPPolicy,
    make_acmt_dp_pre_post_processors as make_acmt_dp_pre_post_processors,
)
from .act.configuration_act import ACTConfig as ACTConfig
from .diffusion.configuration_diffusion import DiffusionConfig as DiffusionConfig
from .eo1.configuration_eo1 import EO1Config as EO1Config
from .evo1.configuration_evo1 import Evo1Config as Evo1Config
from .factory import get_policy_class, make_policy, make_policy_config, make_pre_post_processors
from .fastwam.configuration_fastwam import FastWAMConfig as FastWAMConfig
from .gaussian_actor.configuration_gaussian_actor import GaussianActorConfig as GaussianActorConfig
from .groot.configuration_groot import GrootConfig as GrootConfig
from .lingbot_va.configuration_lingbot_va import LingBotVAConfig as LingBotVAConfig
from .molmoact2.configuration_molmoact2 import MolmoAct2Config as MolmoAct2Config
from .multi_task_dit.configuration_multi_task_dit import MultiTaskDiTConfig as MultiTaskDiTConfig
from .pi0.configuration_pi0 import PI0Config as PI0Config
from .pi0_fast.configuration_pi0_fast import PI0FastConfig as PI0FastConfig
from .pi05.configuration_pi05 import PI05Config as PI05Config
from .pretrained import PreTrainedPolicy as PreTrainedPolicy
from .smolvla.configuration_smolvla import SmolVLAConfig as SmolVLAConfig
from .tdmpc.configuration_tdmpc import TDMPCConfig as TDMPCConfig
from .utils import make_robot_action, prepare_observation_for_inference
from .vqbet.configuration_vqbet import VQBeTConfig as VQBeTConfig
from .wall_x.configuration_wall_x import WallXConfig as WallXConfig
from .xvla.configuration_xvla import XVLAConfig as XVLAConfig

# NOTE: Most policy modeling classes (e.g., GaussianActorPolicy) are intentionally
# NOT re-exported here. They have heavy optional dependencies and are loaded lazily
# via get_policy_class(). ACMT-DP is the lightweight inference-only exception so
# its native policy/processor contract is available from the top-level package.

__all__ = [
    # Configuration classes
    "ACMTDPConfig",
    "ACMTDPPolicy",
    "ACMTDPCenter480ProcessorStep",
    "make_acmt_dp_pre_post_processors",
    "ACTConfig",
    "DiffusionConfig",
    "EO1Config",
    "FastWAMConfig",
    "GaussianActorConfig",
    "Evo1Config",
    "GrootConfig",
    "LingBotVAConfig",
    "MolmoAct2Config",
    "MultiTaskDiTConfig",
    "PI0Config",
    "PI0FastConfig",
    "PI05Config",
    "SmolVLAConfig",
    "TDMPCConfig",
    "VQBeTConfig",
    "WallXConfig",
    "XVLAConfig",
    # Base class
    "PreTrainedPolicy",
    # RTC utilities
    "ActionInterpolator",
    # Utility functions
    "make_robot_action",
    "prepare_observation_for_inference",
    # Factory functions
    "get_policy_class",
    "make_policy",
    "make_policy_config",
    "make_pre_post_processors",
]
