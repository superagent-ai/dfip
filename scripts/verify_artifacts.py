"""Verify archived evidence and optionally regenerate focused CPU paper inputs.

The default check uses only Python's standard library, opens no credentials and
never downloads checkpoints, deploys an app or schedules a GPU. Historical
manifest formats are handled explicitly rather than rehashing changed evidence.
"""

import argparse
from collections import Counter
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class IntegrityError(ValueError):
    """A recorded artifact cannot be verified."""


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def inside(root, relative):
    """Resolve an archive-manifest path without permitting escape from root."""
    relative = PurePosixPath(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise IntegrityError(f"Unsafe manifest path: {relative}")
    path = (root / str(relative)).resolve()
    if not path.is_relative_to(root.resolve()):
        raise IntegrityError(f"Manifest path escapes repository: {relative}")
    return path


def check_bytes(data, expected, label):
    if not isinstance(expected, str) or not SHA256.fullmatch(expected):
        raise IntegrityError(f"Invalid SHA-256 in {label}")
    if sha256(data) != expected:
        raise IntegrityError(f"SHA-256 mismatch: {label}")


def check_file(root, relative, expected):
    path = inside(root, relative)
    if not path.is_file():
        raise IntegrityError(f"Missing artifact: {relative}")
    check_bytes(path.read_bytes(), expected, relative)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def result_path(root, manifest, relative):
    if relative.startswith("benchmarks/"):
        return relative
    return str((manifest.parent / relative).relative_to(root))


def verify_results(root, counts):
    for manifest in sorted((root / "benchmarks/results").rglob("sha256.json")):
        entries = read_json(manifest)
        if not isinstance(entries, dict) or not entries:
            raise IntegrityError(f"Empty or invalid result manifest: {manifest}")
        for relative, expected in entries.items():
            check_file(root, result_path(root, manifest, relative), expected)
            counts["result_files"] += 1
        counts["result_manifests"] += 1


def zip_members(archive):
    """Read safe, unique ZIP members without extracting any files."""
    members = {}
    for entry in archive.infolist():
        name = PurePosixPath(entry.filename)
        if name.is_absolute() or ".." in name.parts or "\\" in entry.filename:
            raise IntegrityError(f"Unsafe ZIP member: {entry.filename}")
        if entry.is_dir():
            continue
        if entry.filename in members:
            raise IntegrityError(f"Duplicate ZIP member: {entry.filename}")
        members[entry.filename] = archive.read(entry)
    return members


def verify_zip(root, relative, expected=None, file_hashes=None):
    path = inside(root, relative)
    if not path.is_file():
        raise IntegrityError(f"Missing source archive: {relative}")
    if expected is not None:
        check_bytes(path.read_bytes(), expected, relative)
    with zipfile.ZipFile(path) as archive:
        members = zip_members(archive)
    for name, digest in (file_hashes or {}).items():
        if name not in members:
            raise IntegrityError(f"Missing ZIP member: {relative}:{name}")
        check_bytes(members[name], digest, f"{relative}:{name}")
    return members


def verify_git_snapshot(root, members, commit):
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise IntegrityError(f"Invalid immutable source commit: {commit}")
    names = {}
    for archived_name, data in members.items():
        if not archived_name.startswith("dfip-source/"):
            raise IntegrityError(f"Unexpected source ZIP prefix: {archived_name}")
        names[archived_name.removeprefix("dfip-source/")] = data
    result = subprocess.run(
        ["git", "archive", "--format=zip", commit, "--", *sorted(names)],
        cwd=root, capture_output=True, check=False,
    )
    if result.returncode:
        raise IntegrityError(f"Cannot read Git source {commit}; use a full clone for --git-history")
    with zipfile.ZipFile(io.BytesIO(result.stdout)) as archive:
        historical = zip_members(archive)
    if names != historical:
        raise IntegrityError(f"Source ZIP differs from recorded Git commit {commit}")


def verify_sources(root, counts, notices, git_history):
    manifest = root / "benchmarks/source-v1-manifest.json"
    record = read_json(manifest)
    members = verify_zip(root, "benchmarks/source-v1.zip", file_hashes=record["sha256"])
    counts["source_archives"] += 1
    counts["source_members_with_recorded_hashes"] += len(record["sha256"])
    notices.append(
        "v1 is a reconstructed snapshot: its manifest checks "
        f"{len(record['sha256'])}/{len(members)} files and records no whole-ZIP hash."
    )
    for version in (3, 4):
        for name, record in read_json(root / f"benchmarks/source-v{version}-manifest.json").items():
            members = verify_zip(root, f"benchmarks/{name}", record["archive_sha256"], record["files_sha256"])
            counts["source_archives"] += 1
            counts["source_members_with_recorded_hashes"] += len(record["files_sha256"])
            if git_history:
                verify_git_snapshot(root, members, record["source_commit"])
                counts["source_archives_matched_to_git"] += 1
    manifest = read_json(root / "benchmarks/source-v5-manifest.json")
    shared_commits = {read_json(path)["source_commit"]
                      for path in (root / "benchmarks/results/contracts").glob("*-shared*.json")}
    policy_count = 0
    checked_commits = set()
    for record in manifest["archives"]:
        members = verify_zip(root, record["archive"], record["sha256"])
        if len(members) != record["files_verified_against_git"]:
            raise IntegrityError(f"Recorded source member count differs: {record['archive']}")
        counts["source_archives"] += 1
        if record["source_commit"] in shared_commits:
            if "dfip-source/src/dfip/shared.py" not in members:
                raise IntegrityError(f"Shared study snapshot lacks shared policy: {record['archive']}")
            check_bytes(members["dfip-source/src/dfip/shared.py"],
                        manifest["shared_policy_sha256"], record["archive"] + ":shared.py")
            policy_count += 1
            checked_commits.add(record["source_commit"])
        if git_history:
            verify_git_snapshot(root, members, record["source_commit"])
            counts["source_archives_matched_to_git"] += 1
    if not policy_count or checked_commits != shared_commits:
        raise IntegrityError("Shared study source commits are not fully covered by frozen policy archives")
    counts["frozen_shared_policy_copies"] = policy_count


def verify_logs(root, counts):
    for manifest in sorted((root / "benchmarks/logs").rglob("manifest.json")):
        for name, record in read_json(manifest)["logs"].items():
            relative = str((manifest.parent / name).relative_to(root))
            check_file(root, relative, record["gzip_sha256"])
            try:
                decoded = gzip.decompress(inside(root, relative).read_bytes())
            except (OSError, EOFError) as error:
                raise IntegrityError(f"Invalid compressed log: {relative}") from error
            check_bytes(decoded, record["decoded_sha256"], relative + " (decoded)")
            counts["compressed_logs"] += 1


def verify_research_sources(root, counts):
    """Check pinned upstream sources used in the structural contract audit."""
    folder = root / "benchmarks/research-audits"
    manifest = read_json(folder / "model-contracts.json")
    relative = str((folder / manifest["source_archive"]).relative_to(root))
    members = verify_zip(root, relative, manifest["source_archive_sha256"])
    model_ids = [model["model_id"] for model in manifest["models"]]
    if not model_ids or len(model_ids) != len(set(model_ids)):
        raise IntegrityError("Research source audit has empty or duplicate model entries")
    verified = set()
    for model in manifest["models"]:
        prefix = model["model_id"].replace("/", "__") + "/"
        filenames = [record["file"] for record in model["files"]]
        if not filenames or len(filenames) != len(set(filenames)):
            raise IntegrityError(f"Research source audit has empty or duplicate files: {model['model_id']}")
        for record in model["files"]:
            name = prefix + record["file"]
            if name not in members:
                raise IntegrityError(f"Missing research source member: {name}")
            check_bytes(members[name], record["sha256"], relative + ":" + name)
            if len(members[name]) != record["bytes"]:
                raise IntegrityError(f"Research source byte count differs: {name}")
            verified.add(name)
        source_manifest = prefix + "source-manifest.json"
        if source_manifest not in members:
            raise IntegrityError(f"Missing embedded research source manifest: {source_manifest}")
        embedded = json.loads(members[source_manifest])
        if any(embedded[key] != model[key] for key in ("model_id", "revision", "files")):
            raise IntegrityError(f"Embedded research source manifest differs: {source_manifest}")
    if len(verified) != sum(len(model["files"]) for model in manifest["models"]):
        raise IntegrityError("Research source member paths overlap between models")
    counts["research_source_archives"] = 1
    counts["research_source_archive_members"] = len(members)
    counts["research_source_files_with_recorded_hashes"] = len(verified)


def verify_focused(root, counts):
    summary = read_json(root / "paper/generated/focused-summary.json")
    evidence = summary["evidence"]
    names = [record["file"] for record in evidence]
    if len(names) != len(set(names)) or not names:
        raise IntegrityError("Focused evidence is empty or contains duplicate records")
    for record in evidence:
        if record["model"] not in summary["checkpoints"]:
            raise IntegrityError(f"Unselected checkpoint in focused evidence: {record['file']}")
        check_file(root, record["file"], record["sha256"])
    for relative, digest in summary["analysis_sources"].items():
        check_file(root, relative, digest)
    counts["focused_result_files"] = len(evidence)
    counts["focused_analysis_sources"] = len(summary["analysis_sources"])
    build = read_json(root / "paper/review-build.json")
    check_file(root, "paper/dfip.pdf", build["paper_sha256"])
    counts["review_pdf"] = 1
    if "bibliography_sha256" in build:
        check_file(root, "paper/main.bbl", build["bibliography_sha256"])
        counts["paper_bibliography"] = 1
    return summary


def recorded_environments(raw):
    """Keep top-level and per-mode setup environments distinct, including absences."""
    found = []
    candidates = [("environment", raw.get("environment"))]
    candidates.extend((f"modes/{mode}/setup/environment", report.get("setup", {}).get("environment"))
                      for mode, report in raw.get("modes", {}).items())
    for location, environment in candidates:
        if isinstance(environment, dict):
            found.append({"location": location, **{
                key: environment.get(key) for key in ("python", "cuda", "packages")}})
    return found


def normalized_package(name):
    return re.sub(r"[-_.]+", "-", name.lower())


def check_constraints(path, observed):
    """Reject pins without matching version evidence, including unknown transitives."""
    count = 0
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s]+)", line)
        if not match or observed.get(normalized_package(match[1])) != match[2]:
            raise IntegrityError(f"Constraint lacks matching observed version: {path.name}: {line}")
        count += 1
    if not count:
        raise IntegrityError(f"Empty observed constraints: {path.name}")
    return count


def verify_reproduction(root, summary, counts):
    index = read_json(root / "repro/source-index.json")
    expected = {record["file"]: record for record in summary["evidence"]}
    records = index["records"]
    if len(records) != len(expected) or {r["file"] for r in records} != set(expected):
        raise IntegrityError("Reproduction index does not cover the selected evidence exactly")
    gliner = []
    fast = None
    for record in records:
        evidence = expected[record["file"]]
        if record["sha256"] != evidence["sha256"]:
            raise IntegrityError("Reproduction raw hash differs: " + record["file"])
        if any(record[key] != evidence[key] for key in ("model", "protocol", "status")):
            raise IntegrityError("Reproduction record identity differs: " + record["file"])
        raw = read_json(inside(root, record["file"]))
        source = raw["source_commit"]
        full = record["resolved_source_commit"]
        if (record["recorded_source_commit"] != source or not re.fullmatch(r"[0-9a-f]{40}", full)
                or not full.startswith(source)):
            raise IntegrityError("Reproduction source identity differs: " + record["file"])
        if record["recorded_environments"] != recorded_environments(raw):
            raise IntegrityError("Reproduction environment differs: " + record["file"])
        if evidence["model"] == "gliner":
            gliner.append(raw["environment"]["packages"])
        if raw["protocol"] == "dfip-paper-v3-optimized-native" and raw["status"] == "complete":
            fast = raw["environment"]["packages"]
    check_file(root, index["supplementary_fast_inventory"], index["supplementary_fast_inventory_sha256"])
    inventory = read_json(inside(root, index["supplementary_fast_inventory"]))
    observed_fast = {normalized_package(name): version for name, version in inventory["fast_image"]["packages"]}
    observed_fast["causal-conv1d"] = fast["causal-conv1d"]
    observed_gliner = {normalized_package(name): version for name, version in gliner[0].items()
                       if version is not None and all(packages.get(name) == version for packages in gliner)}
    counts["observed_gpu_dependency_pins"] = (
        check_constraints(root / "repro/fast-observed.constraints.txt", observed_fast)
        + check_constraints(root / "repro/gliner-observed.constraints.txt", observed_gliner))
    counts["reproduction_source_records"] = len(records)


def regenerate_focused(root, summary, counts, notices):
    """Run analysis in an isolated copy, keeping the checked-in archive intact."""
    with tempfile.TemporaryDirectory(prefix="dfip-analysis-") as directory:
        scratch = Path(directory).resolve()
        for relative in set(summary["analysis_sources"]) | {r["file"] for r in summary["evidence"]}:
            target = inside(scratch, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(inside(root, relative), target)
        shutil.copytree(root / "src", scratch / "src", dirs_exist_ok=True)
        (scratch / "paper/generated").mkdir(parents=True)
        environment = os.environ.copy()
        environment.update(PYTHONPATH=str(scratch / "src"), MPLCONFIGDIR=str(scratch / "mplconfig"),
                           PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run([sys.executable, str(scratch / "scripts/analyze_focused.py")],
                                cwd=scratch, env=environment, capture_output=True, text=True, check=False)
        if result.returncode:
            raise IntegrityError("Focused CPU analysis failed: " + result.stderr[-2000:])
        expected = sorted((root / "paper/generated").glob("focused*.tex"))
        expected.append(root / "paper/generated/focused-summary.json")
        for path in expected:
            regenerated = scratch / path.relative_to(root)
            if not regenerated.is_file() or regenerated.read_bytes() != path.read_bytes():
                raise IntegrityError(f"Regenerated focused input differs: {path.name}")
        figure = scratch / "paper/generated/focused_workloads.png"
        if not figure.is_file() or not figure.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
            raise IntegrityError("Focused workload figure was not regenerated")
        counts["focused_text_artifacts_regenerated"] = len(expected)
        counts["focused_figures_regenerated"] = 1
        notices.append("Focused summary and TeX inputs matched byte for byte; PNG rendering is regenerated, "
                       "but is not compared byte for byte across font and plotting-library versions.")


def verify(root, git_history=False, regenerate=False):
    root = root.resolve()
    counts = Counter()
    notices = []
    verify_results(root, counts)
    verify_sources(root, counts, notices, git_history)
    verify_logs(root, counts)
    verify_research_sources(root, counts)
    licenses = root / "licenses/manifest.json"
    if licenses.is_file():
        for record in read_json(licenses)["files"]:
            check_file(root, record["file"], record["sha256"])
            counts["third_party_notices"] += 1
    summary = verify_focused(root, counts)
    verify_reproduction(root, summary, counts)
    if regenerate:
        regenerate_focused(root, summary, counts, notices)
    return {"status": "verified", "checks": dict(sorted(counts.items())), "limitations": notices}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository root")
    parser.add_argument("--git-history", action="store_true",
                        help="Also match v3-v5 ZIP contents to source commits; requires full Git history")
    parser.add_argument("--regenerate", action="store_true",
                        help="Regenerate focused summary, TeX inputs and figure on CPU; needs paper dependencies")
    args = parser.parse_args()
    try:
        report = verify(args.root, args.git_history, args.regenerate)
    except (IntegrityError, KeyError, ValueError, OSError, zipfile.BadZipFile) as error:
        parser.exit(1, f"Artifact verification failed: {error}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
