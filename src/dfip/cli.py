"""Install, prepare, deploy, and call DFIP from any working directory."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from dfip import Client
from dfip.models import MODELS, get_model
from dfip.schema import validate_records


def modal_command(action, module, arguments=()):
    # Modal is a dependency of the same interpreter that runs the installed CLI.
    return subprocess.call(
        [sys.executable, "-m", "modal", action, "-m", module, *arguments]
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="dfip", description="Decision-focused inference on Modal"
    )
    parser.add_argument("--version", action="version", version="dfip 0.1.0")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("models", help="List supported checkpoints and GPUs")
    prepare = commands.add_parser("prepare", help="Download checkpoints on Modal")
    prepare.add_argument("--model", default="liquid", help="Model alias, HF ID, or all")
    commands.add_parser("deploy", help="Deploy private authenticated Modal methods")
    setup = commands.add_parser("setup", help="Prepare checkpoints, then deploy")
    setup.add_argument("--model", default="liquid")
    predict = commands.add_parser("predict", help="Submit a JSON list of records")
    predict.add_argument("--model", default="liquid")
    predict.add_argument("--input", default="-", help="JSON file, or - for stdin")
    predict.add_argument("--app", default=None)
    diagnostics = commands.add_parser(
        "diagnostics", help="Show remote engine diagnostics"
    )
    diagnostics.add_argument("--model", default="liquid")
    diagnostics.add_argument("--app", default=None)
    for name in ("benchmark", "validate", "smoke"):
        command = commands.add_parser(
            name,
            help={
                "benchmark": "Benchmark native and configured inference on Modal",
                "validate": "Run full model fidelity checks",
                "smoke": "Check packaged inference against native on Modal",
            }[name],
        )
        command.add_argument("--model", default="liquid")
        command.add_argument(
            "--output", default=f"{name}.json", help="Local JSON result file"
        )
        if name == "benchmark":
            command.add_argument("--quality", type=int, default=400)
            command.add_argument("--repetitions", type=int, default=100)
    args = parser.parse_args(argv)
    try:
        if args.command == "models":
            print(json.dumps([vars(model) for model in MODELS], indent=2))
            return 0
        if hasattr(args, "model") and args.model != "all":
            args.model = get_model(args.model).alias
        if args.command in ("prepare", "setup"):
            result = modal_command("run", "dfip.cloud.prepare", ["--model", args.model])
            if result or args.command == "prepare":
                return result
        if args.command in ("deploy", "setup"):
            return modal_command("deploy", "dfip.cloud.serve")
        if args.command == "predict":
            if args.input == "-":
                records = json.load(sys.stdin)
            else:
                records = json.loads(Path(args.input).read_text())
            validate_records(records)
            print(
                json.dumps(
                    Client(args.model, app_name=args.app).predict(records), indent=2
                )
            )
            return 0
        if args.command == "diagnostics":
            print(
                json.dumps(
                    Client(args.model, app_name=args.app).diagnostics(), indent=2
                )
            )
            return 0
        if args.command in ("benchmark", "validate", "smoke"):
            if args.model == "all":
                raise ValueError("Select one model for this command")
            command = [
                "--action",
                args.command,
                "--model",
                args.model,
                "--output",
                args.output,
            ]
            if args.command == "benchmark":
                if args.quality < 1 or args.quality > 400 or args.repetitions < 1:
                    raise ValueError(
                        "Quality must be 1–400 and repetitions must be positive"
                    )
                command += [
                    "--quality",
                    str(args.quality),
                    "--repetitions",
                    str(args.repetitions),
                ]
            return modal_command("run", "dfip.cloud.evaluate", command)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    return 0
