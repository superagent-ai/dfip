"""Serialize model admission jobs and retain exact source and every outcome."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from run_strengthened import wait_retired

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=("native", "shared", "shared_compiled"), required=True)
    parser.add_argument("--models", default="omni,gliner,d3,jev9,jev27")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--start-round", type=int, default=0)
    parser.add_argument("--volume", default="dfip-research-v1")
    args = parser.parse_args()
    executable = str(ROOT / ".venv/bin/modal")
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), DFIP_VOLUME=args.volume)
    # Preparation is CPU-only and may overlap. Every study GPU must retire.
    wait_retired("dfip-study5-native", executable, env)
    wait_retired("dfip-study5-shared", executable, env)
    folder = ROOT / "benchmarks/results/contracts"; folder.mkdir(parents=True, exist_ok=True)
    for model in args.models.split(","):
        # GPU time must not be spent on checkpoints whose CPU preparation has
        # not finished. Local receipts are written only after Volume.commit().
        receipt = folder / (model + "-prepared.json")
        print(json.dumps({"model": model, "state": "waiting for CPU preparation"}), flush=True)
        deadline = time.monotonic() + 7200
        while True:
            metadata = json.loads(receipt.read_text()) if receipt.exists() else {}
            if metadata and (model != "d3" or metadata.get("preparation_version", 0) >= 2):
                break
            if time.monotonic() > deadline:
                raise TimeoutError("CPU preparation receipt missing for " + model)
            time.sleep(5)
        for round_id in range(args.start_round, args.start_round + args.rounds):
            source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
            subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "src", "scripts/run_contract_study.py"], cwd=ROOT, check=True)
            archive = ROOT / "benchmarks" / f"source-v5-{source[:12]}.zip"
            if not archive.exists():
                paths = ["src", "scripts", "tests", "pyproject.toml", "benchmarks/generality-protocol.json"]
                if (ROOT / "benchmarks/shared-batching-protocol.json").exists():
                    paths.append("benchmarks/shared-batching-protocol.json")
                if (ROOT / "benchmarks/shared-compiled-followup-protocol.json").exists():
                    paths.append("benchmarks/shared-compiled-followup-protocol.json")
                subprocess.run(["git", "archive", "--format=zip", "--prefix=dfip-source/", "--output="+str(archive),
                                source, *paths], cwd=ROOT, check=True)
            prefix = f"dfip-study5-{args.action}-{model}-{round_id}"
            log = ROOT / "work" / f"contracts-{args.action}-{model}-{round_id}-{source[:12]}.log"
            started = time.time()
            print(json.dumps({"model": model, "action": args.action, "round": round_id, "state": "started", "source": source}), flush=True)
            with log.open("w") as stream:
                module = "dfip.cloud.contracts_jev" if model in ("jev9", "jev27") else "dfip.cloud.contracts"
                process = subprocess.run([executable, "run", "-m", module, "--action", args.action,
                                          "--model", model, "--round-id", str(round_id), "--commit", source],
                                         cwd=ROOT, env=dict(env, DFIP_APP=prefix), stdout=stream, stderr=subprocess.STDOUT)
            record = {"model": model, "action": args.action, "round": round_id, "source_commit": source,
                      "source_archive": str(archive.relative_to(ROOT)), "log": str(log.relative_to(ROOT)),
                      "started_unix": started, "finished_unix": time.time(), "exit_code": process.returncode}
            receipt = folder / f"{model}-{args.action}-{round_id}-{source[:12]}.json"
            record["experiment_status"] = json.loads(receipt.read_text())["status"] if receipt.exists() else "no local result"
            with (folder / "job-journal.jsonl").open("a") as stream:
                stream.write(json.dumps(record)+"\n")
            print(json.dumps(record), flush=True)
            wait_retired(prefix, executable, env)


if __name__ == "__main__":
    main()
