"""Exercise the unchanged native dynamic factories and the folder-local launcher."""

import sys

import pytest

from lerobot.configs import PreTrainedConfig
from lerobot.policies.flowmatch import FlowMatchConfig, cli


def test_launcher_registers_then_forwards_original_arguments(monkeypatch):
    seen = []
    argv_before = sys.argv

    def capture(module, run_name):
        assert PreTrainedConfig.get_choice_class("flowmatch") is FlowMatchConfig
        seen.append((module, run_name, list(sys.argv)))

    monkeypatch.setattr(cli.runpy, "run_module", capture)
    cli.main(["record", "--policy.path=/saved/whole", "--policy.device=cuda"])
    assert seen == [
        (
            "lerobot.scripts.lerobot_record",
            "__main__",
            ["lerobot.scripts.lerobot_record", "--policy.path=/saved/whole", "--policy.device=cuda"],
        )
    ]
    assert sys.argv is argv_before


def test_launcher_rejects_unknown_entry_without_running_anything(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("An unknown entry must not execute a script")

    monkeypatch.setattr(cli.runpy, "run_module", forbidden)
    with pytest.raises(SystemExit) as error:
        cli.main(["unknown"])
    assert error.value.code == 2
