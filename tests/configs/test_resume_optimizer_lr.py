#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from unittest.mock import Mock

import pytest

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.act.configuration_act import ACTConfig


def make_train_config(tmp_path, *, resume: bool, learning_rate: float | None) -> TrainPipelineConfig:
    config = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="local/test"),
        policy=ACTConfig(push_to_hub=False),
        output_dir=tmp_path / "output",
        resume=resume,
        resume_optimizer_lr=learning_rate,
    )
    config._resolve_pretrained_from_cli = Mock()
    return config


def test_resume_optimizer_lr_accepts_scheduler_free_resume(tmp_path):
    config = make_train_config(tmp_path, resume=True, learning_rate=3e-6)

    config.validate()


def test_resume_optimizer_lr_requires_resume(tmp_path):
    config = make_train_config(tmp_path, resume=False, learning_rate=3e-6)

    with pytest.raises(ValueError, match="requires resume=true"):
        config.validate()


def test_resume_optimizer_lr_must_be_positive(tmp_path):
    config = make_train_config(tmp_path, resume=True, learning_rate=0.0)

    with pytest.raises(ValueError, match="must be > 0"):
        config.validate()


def test_resume_optimizer_lr_rejects_scheduler(tmp_path):
    config = make_train_config(tmp_path, resume=True, learning_rate=3e-6)
    config.scheduler = Mock()

    with pytest.raises(ValueError, match="requires scheduler=None"):
        config.validate()
