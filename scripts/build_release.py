"""Build reviewable release assets from a committed, tracked-only source snapshot.

Requires the dev and paper extras. This runs CPU checks and a Python package
build; it never publishes a release, uploads packages, or calls Modal.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def run(arguments, *, cwd=ROOT):
    result = subprocess.run(arguments, cwd=cwd, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(output):
    if run(["git", "status", "--porcelain", "--untracked-files=no"]):
        raise ValueError("Commit tracked source changes before building release assets")
    commit = run(["git", "rev-parse", "HEAD"])
    version = re.search(r'^version = "([^"]+)"$', (ROOT / "pyproject.toml").read_text(), re.M)[1]
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Use an empty output directory to preserve previous release assets")
    archive = output / f"dfip-{version}-source.zip"
    run(["git", "archive", "--format=zip", "--prefix=dfip/", f"--output={archive}", commit])
    with tempfile.TemporaryDirectory(prefix="dfip-release-") as temporary:
        with zipfile.ZipFile(archive) as files:
            files.extractall(temporary)
        snapshot = Path(temporary) / "dfip"
        integrity = json.loads(run(
            [sys.executable, "scripts/verify_artifacts.py", "--regenerate"], cwd=snapshot
        ))
        run([sys.executable, "-m", "build", "--outdir", str(output)], cwd=snapshot)
        shutil.copyfile(snapshot / "paper/dfip.pdf", output / "dfip-paper.pdf")
        shutil.copyfile(snapshot / "paper/generated/focused-summary.json", output / "dfip-results.json")
        shutil.copyfile(snapshot / "CHANGELOG.md", output / "release-notes.md")
    assets = {
        path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
        for path in sorted(output.iterdir()) if path.is_file()
    }
    manifest = {
        "version": version,
        "source_commit": commit,
        "repository": "https://github.com/superagent-ai/dfip",
        "status": "prepared_not_published",
        "research_preview": True,
        "arxiv_submitted": False,
        "tracked_source_only": True,
        "historical_provenance_metadata_retained": True,
        "artifact_verification": integrity,
        "assets": assets,
    }
    (output / "release-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "SHA256SUMS").write_text("".join(
        f"{sha256(path)}  {path.name}\n" for path in sorted(output.iterdir()) if path.is_file()
    ))
    return {"version": version, "source_commit": commit, "output": str(output), "assets": len(assets)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/release")
    arguments = parser.parse_args()
    try:
        print(json.dumps(build(arguments.output), indent=2))
    except (ValueError, subprocess.CalledProcessError) as error:
        if isinstance(error, subprocess.CalledProcessError):
            print(error.stderr or error.stdout, file=sys.stderr)
        parser.exit(1, f"Release build failed: {error}\n")


if __name__ == "__main__":
    main()
