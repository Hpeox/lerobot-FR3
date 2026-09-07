"""Four-camera RGB-D ACMT-ACTv2 policy."""

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
from lerobot.policies.acmt_act.modeling_acmt_act import OBS_DEPTH


class ACMTACTV2Policy(ACMTACTPolicy):
    """ACMT-ACT policy with four independent DFormer RGB-D streams."""

    config_class = ACMTACTV2Config
    name = "acmt_actv2"

    def _model_batch(self, window: Mapping[str, Tensor], *, include_target: bool = False) -> dict[str, Tensor]:
        model_batch = super()._model_batch(window, include_target=include_target)
        model_batch[OBS_IMAGES] = [
            window["rgb"][:, index] for index in range(len(self.config.camera_keys))
        ]
        model_batch[OBS_DEPTH] = [
            window["depth"][:, index] for index in range(len(self.config.camera_keys))
        ]
        return model_batch

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, *, config=None, **kwargs: Any):
        config_path = Path(pretrained_name_or_path) / "config.json"
        if config_path.is_file():
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            if raw.get("type") != "acmt_actv2":
                raise ValueError("ACMT-ACTv2 loader refuses non-acmt_actv2 checkpoints")
            if raw.get("checkpoint_schema") != "acmt_actv2.dformerv2_spatial.v1" or raw.get("checkpoint_schema_version") != 2:
                raise ValueError("checkpoint is not ACMT-ACTv2 DFormer spatial schema")
            if config is not None:
                checkpoint_task = raw.get("checkpoint_task_variant", raw.get("task_variant", "peg"))
                if checkpoint_task != config.task_variant:
                    raise ValueError("ACMT-ACTv2 checkpoint task does not match the requested task variant")
        if config is None:
            # The serialized checkpoint contains the complete DFormer weights;
            # the original external pretraining file is not a deployment
            # dependency.  Ask the config parser to allow its absence while
            # retaining the hash as provenance metadata.
            config = ACMTACTV2Config.from_pretrained(
                pretrained_name_or_path,
                cli_overrides=["--require_dformer_checkpoint=false"],
                **kwargs,
            )
        else:
            config.require_dformer_checkpoint = False
        return PreTrainedPolicy.from_pretrained.__func__(cls, pretrained_name_or_path, config=config, **kwargs)


__all__ = ["ACMTACTV2Policy"]
