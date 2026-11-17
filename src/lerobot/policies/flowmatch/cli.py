"""A registration bootstrap; factory.py and the original CLI scripts are unchanged."""

import argparse
import runpy
import sys

COMMANDS = {
    "record": "lerobot.scripts.lerobot_record",
    "eval": "lerobot.scripts.lerobot_eval",
    "export": "lerobot.policies.flowmatch.export",
    "infer": "lerobot.policies.flowmatch.infer_saved",
    "verify": "lerobot.policies.flowmatch.verify_100k",
    "download": "lerobot.policies.flowmatch.download_weights",
}


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Register flowmatch, then forward arguments unchanged to the selected entry point.",
        usage="python -m lerobot.policies.flowmatch {record,eval,export,infer,verify,download} [arguments]",
    )
    parser.add_argument("command", choices=COMMANDS)
    if not arguments or arguments[0] in {"-h", "--help"}:
        parser.print_help()
        return
    command = parser.parse_args(arguments[:1]).command
    from . import register

    register()
    original_argv = sys.argv
    try:
        sys.argv = [COMMANDS[command], *arguments[1:]]
        runpy.run_module(COMMANDS[command], run_name="__main__")
    finally:
        sys.argv = original_argv
