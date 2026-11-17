"""Opt-in trained100k verification, never uses raw HDF5 or a robot."""

import json
import os
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(not os.environ.get("FLOWMATCH_WEIGHTS_ROOT"), reason="100k weights are not requested")
def test_published_100k_whole_native_export_and_independent_workers(tmp_path, monkeypatch):
    from lerobot.policies.flowmatch.export import export_policy
    from lerobot.policies.flowmatch.modeling_flowmatch import FlowMatchPolicy
    from lerobot.policies.flowmatch.verify_100k import main

    weights = Path(os.environ["FLOWMATCH_WEIGHTS_ROOT"])
    output = Path(os.environ.get("FLOWMATCH_EXPORT_ROOT", tmp_path / "export"))
    policy = FlowMatchPolicy.from_checkpoint(weights / "policy_100k.pt", device="cpu")
    assert policy.config.type == "flowmatch"
    assert policy.config.chunk_size == 16 and policy.config.n_action_steps == 4
    export_policy(policy, output)
    assert json.loads((output / "config.json").read_text())["type"] == "flowmatch"
    report_path = Path(os.environ.get("FLOWMATCH_REPORT_PATH", tmp_path / "verification.json"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_100k",
            "--weights",
            str(weights),
            "--policy-path",
            str(output),
            "--output",
            str(report_path),
        ],
    )
    main()
    report = json.loads(report_path.read_text())
    assert report["status"] == "PASS"
    # verify_100k.main already enforces its stated floating-point tolerances.
    assert report["independent_workers_max_abs"] <= 1e-6
