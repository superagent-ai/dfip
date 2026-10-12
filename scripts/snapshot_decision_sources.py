"""Archive pinned model source and configuration without downloading weights.

Usage: python scripts/snapshot_decision_sources.py MODEL_ID REVISION OUTPUT
The snapshot is evidence for manual contract review, never a compatibility test.
"""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from urllib.parse import quote
from urllib.request import urlopen

LIMIT = 4 * 1024 * 1024


def read(url):
    with urlopen(url, timeout=60) as response:
        data = response.read(LIMIT + 1)
    if len(data) > LIMIT:
        raise ValueError(f"Source exceeds {LIMIT} bytes: {url}")
    return data


def selected(name):
    path = PurePosixPath(name)
    return (
        path.suffix == ".py"
        or path.name in ("README.md", "requirements.txt", "LICENSE")
        or (path.suffix == ".json" and any(
            term in path.name.lower() for term in ("config", "decision", "calibrat", "schema")
        ))
    )


def snapshot(model, revision, output):
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", model):
        raise ValueError("Expected a Hugging Face owner/model ID")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Use an immutable 40-character revision SHA")
    metadata_url = f"https://huggingface.co/api/models/{model}/revision/{revision}"
    metadata = json.loads(read(metadata_url))
    if metadata["sha"] != revision:
        raise ValueError("Resolved revision differs from requested revision")
    output = Path(output)
    if output.exists():
        raise ValueError("Use a new output directory to preserve previous evidence")
    output.mkdir(parents=True)
    (output / "hub-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    evidence = []
    for sibling in metadata.get("siblings", []):
        name = sibling["rfilename"]
        if not selected(name):
            continue
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid source path")
        url = f"https://huggingface.co/{model}/resolve/{revision}/{quote(name, safe='/')}"
        data = read(url)
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        evidence.append({"file": name, "url": url, "bytes": len(data),
                         "sha256": hashlib.sha256(data).hexdigest()})
    report = {"model_id": model, "revision": revision,
              "license": metadata.get("cardData", {}).get("license"),
              "weights_downloaded": False, "source_executed": False,
              "files": evidence}
    (output / "source-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("revision")
    parser.add_argument("output")
    args = parser.parse_args()
    report = snapshot(args.model, args.revision, args.output)
    print(json.dumps({"model_id": report["model_id"], "revision": report["revision"],
                      "source_files": len(report["files"]), "output": args.output}))
