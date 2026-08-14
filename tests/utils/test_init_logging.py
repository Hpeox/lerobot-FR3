#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import re
from pathlib import Path

from lerobot.utils.utils import init_logging


def test_init_logging_preserves_source_basename(capsys):
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level
    try:
        init_logging()
        logging.getLogger(__name__).info("source basename marker")
        output = capsys.readouterr().err
    finally:
        root.handlers.clear()
        root.handlers.extend(original_handlers)
        root.setLevel(original_level)

    source_name = Path(__file__).name
    assert re.search(
        rf"INFO \d{{4}}-\d{{2}}-\d{{2}} \d{{2}}:\d{{2}}:\d{{2}} "
        rf"{re.escape(source_name)}:\d+ source basename marker",
        output,
    )
