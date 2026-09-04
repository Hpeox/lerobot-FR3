"""Three-camera ACMT-ACT policy with a strict v2 checkpoint ABI."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from torch import Tensor

from lerobot.policies.acmt_act.modeling_acmt_act import ACMTACTPolicy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import OBS_IMAGES

from .configuration_acmt_actv2 import ACMTACTV2Config


class ACMTACTV2Policy(ACMTACTPolicy):
    """The v3 ACT/tactile network contract with exactly three image streams."""

    config_class = ACMTACTV2Config
    name = "acmt_actv2"

    def _model_batch(self, window: Mapping[str, Tensor], *, include_target: bool = False) -> dict[str, Tensor]:
        model_batch = super()._model_batch(window, include_target=include_target)
        model_batch[OBS_IMAGES] = [
            window["rgb"][:, index] for index in range(len(self.config.image_features))
        ]
        return model_batch

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, *, config=None, **kwargs: Any):
        config_path = Path(pretrained_name_or_path) / "config.json"
        if config_path.is_file():
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            if raw.get("type") != "acmt_actv2":
                raise ValueError("ACMT-ACTv2 loader refuses non-acmt_actv2 checkpoints")
            if raw.get("checkpoint_schema") != "acmt_actv2.v1" or raw.get("checkpoint_schema_version") != 1:
                raise ValueError("checkpoint is not ACMT-ACTv2 schema acmt_actv2.v1")
            if config is not None:
                checkpoint_task = raw.get("checkpoint_task_variant", raw.get("task_variant", "peg"))
                if checkpoint_task != config.task_variant:
                    raise ValueError("ACMT-ACTv2 checkpoint task does not match the requested task variant")
        return PreTrainedPolicy.from_pretrained.__func__(cls, pretrained_name_or_path, config=config, **kwargs)


__all__ = ["ACMTACTV2Policy"]
