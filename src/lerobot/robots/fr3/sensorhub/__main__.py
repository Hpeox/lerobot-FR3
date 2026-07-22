#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import logging

from .runtime import SensorHubConfig, SensorHubRuntime


def main() -> int:
    parser = argparse.ArgumentParser(description="FR3 SensorHub managed subprocess")
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--parent-pid", required=True, type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    config = SensorHubConfig.from_dict(json.loads(args.config_json))
    return SensorHubRuntime(config, args.parent_pid).run()


if __name__ == "__main__":
    raise SystemExit(main())
