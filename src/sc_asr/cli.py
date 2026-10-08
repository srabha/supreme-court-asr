"""Command-line entry point."""

import argparse
import importlib
import json
import sys

# Modules are imported only for the command being run, so `data-prep` works without torch/transformers.
COMMANDS = {
    "data-prep": ("sc_asr.data.pipeline", "main",
                  "Long-form audio + PDF transcripts -> aligned chunks with NeMo and HF manifests"),
    "train": ("sc_asr.train", "train", "Fine-tune Whisper with LoRA"),
    "evaluate": ("sc_asr.evaluate", "evaluate", "Evaluate on a held-out manifest"),
}


def main() -> None:
    parser = argparse.ArgumentParser(prog="sc-asr")
    commands = parser.add_subparsers(dest="command", required=True)
    requested = next((a for a in sys.argv[1:] if not a.startswith("-")), None)
    module = None
    for name, (module_name, _, help_text) in COMMANDS.items():
        subparser = commands.add_parser(name, help=help_text)
        if name == requested:
            module = importlib.import_module(module_name)
            module.add_arguments(subparser)
    args = parser.parse_args()
    result = getattr(module, COMMANDS[args.command][1])(args)
    if result is not None:
        print(json.dumps(result, indent=2))
