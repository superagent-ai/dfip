"""Integrity failures must stop review builds rather than bless changed evidence."""

from collections import Counter
import gzip
import importlib.util
import json
from pathlib import Path
import warnings
import zipfile

import pytest


spec = importlib.util.spec_from_file_location(
    "artifact_integrity", Path(__file__).parents[1] / "scripts/verify_artifacts.py"
)
integrity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(integrity)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


@pytest.mark.parametrize("pin", ["torch==2.15.0", "sentencepiece==0.2.0"])
def test_gpu_constraints_reject_changed_or_unrecorded_versions(tmp_path, pin):
    constraints = tmp_path / "observed.constraints.txt"
    constraints.write_text(pin + "\n")
    with pytest.raises(integrity.IntegrityError, match="lacks matching observed version"):
        integrity.check_constraints(constraints, {"torch": "2.14.0"})


def test_reproduction_environments_retain_different_mode_setups_and_missing_fields():
    raw = {"modes": {
        "native": {"setup": {"environment": {"packages": {"torch": "2.14.0"}}}},
        "failed": {},
        "replacement": {"setup": {"environment": {"packages": {"torch": "2.13.0"}}}},
    }}
    found = integrity.recorded_environments(raw)
    assert [(r["location"], r["packages"]["torch"]) for r in found] == [
        ("modes/native/setup/environment", "2.14.0"),
        ("modes/replacement/setup/environment", "2.13.0"),
    ]
    assert all(r["python"] is None and r["cuda"] is None for r in found)


def test_results_reject_changed_raw_bytes_with_root_and_local_manifest_paths(tmp_path):
    folder = tmp_path / "benchmarks/results/phase"
    raw = folder / "nested/round.json"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b'{"timing": 10}')
    for name in ("nested/round.json", "benchmarks/results/phase/nested/round.json"):
        write_json(folder / "sha256.json", {name: integrity.sha256(raw.read_bytes())})
        counts = Counter()
        integrity.verify_results(tmp_path, counts)
        assert counts["result_files"] == 1
    raw.write_bytes(b'{"timing": 9}')
    with pytest.raises(integrity.IntegrityError, match="mismatch.*round.json"):
        integrity.verify_results(tmp_path, Counter())


@pytest.mark.parametrize("name", ["../secret", "/private/secret", "nested/../../secret"])
def test_manifest_cannot_verify_files_outside_repository(tmp_path, name):
    with pytest.raises(integrity.IntegrityError, match="Unsafe"):
        integrity.check_file(tmp_path, name, "0" * 64)


def test_zip_rejects_changed_member_even_when_outer_digest_is_updated(tmp_path):
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("source/code.py", b"changed")
    with pytest.raises(integrity.IntegrityError, match="mismatch.*code.py"):
        integrity.verify_zip(tmp_path, "source.zip", integrity.sha256(archive.read_bytes()),
                             {"source/code.py": integrity.sha256(b"original")})


@pytest.mark.parametrize("names", [["../secret"], ["source/code.py", "source/code.py"]])
def test_zip_ambiguous_or_unsafe_members_are_rejected(tmp_path, names):
    archive = tmp_path / "source.zip"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(archive, "w") as output:
            for name in names:
                output.writestr(name, b"data")
    with pytest.raises(integrity.IntegrityError, match="Unsafe|Duplicate"):
        integrity.verify_zip(tmp_path, "source.zip")


def test_decoded_log_hash_is_verified_independently_of_compressed_bytes(tmp_path):
    folder = tmp_path / "benchmarks/logs/study"
    folder.mkdir(parents=True)
    compressed = gzip.compress(b"changed log", mtime=0)
    (folder / "round.log.gz").write_bytes(compressed)
    write_json(folder / "manifest.json", {"logs": {"round.log.gz": {
        "gzip_sha256": integrity.sha256(compressed),
        "decoded_sha256": integrity.sha256(b"original log"),
    }}})
    with pytest.raises(integrity.IntegrityError, match="mismatch.*decoded"):
        integrity.verify_logs(tmp_path, Counter())


@pytest.mark.parametrize("update_outer_hash", [False, True])
def test_research_source_audit_rejects_tampering_inside_zip(tmp_path, update_outer_hash):
    folder = tmp_path / "benchmarks/research-audits"
    folder.mkdir(parents=True)
    model = {"model_id": "org/model", "revision": "a" * 40,
             "files": [{"file": "code.py", "bytes": 8, "sha256": integrity.sha256(b"original")}]}
    archive = folder / "sources.zip"

    def write_source(content):
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr("org__model/code.py", content)
            output.writestr("org__model/source-manifest.json", json.dumps(model))

    write_source(b"original")
    manifest = {"source_archive": "sources.zip", "source_archive_sha256": integrity.sha256(archive.read_bytes()),
                "models": [model]}
    write_json(folder / "model-contracts.json", manifest)
    counts = Counter()
    integrity.verify_research_sources(tmp_path, counts)
    assert counts["research_source_files_with_recorded_hashes"] == 1
    write_source(b"tampered")
    if update_outer_hash:
        manifest["source_archive_sha256"] = integrity.sha256(archive.read_bytes())
        write_json(folder / "model-contracts.json", manifest)
    with pytest.raises(integrity.IntegrityError, match="SHA-256 mismatch"):
        integrity.verify_research_sources(tmp_path, Counter())


def test_research_source_audit_rejects_inconsistent_byte_count(tmp_path):
    folder = tmp_path / "benchmarks/research-audits"
    folder.mkdir(parents=True)
    model = {"model_id": "org/model", "revision": "a" * 40,
             "files": [{"file": "code.py", "bytes": 7, "sha256": integrity.sha256(b"original")}]}
    archive = folder / "sources.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("org__model/code.py", b"original")
        output.writestr("org__model/source-manifest.json", json.dumps(model))
    write_json(folder / "model-contracts.json", {
        "source_archive": "sources.zip", "source_archive_sha256": integrity.sha256(archive.read_bytes()),
        "models": [model],
    })
    with pytest.raises(integrity.IntegrityError, match="byte count differs"):
        integrity.verify_research_sources(tmp_path, Counter())


def test_focused_summary_rejects_checkpoint_outside_declared_scope(tmp_path):
    write_json(tmp_path / "paper/generated/focused-summary.json", {
        "checkpoints": {"gliner": "GLiNER"},
        "evidence": [{"file": "benchmarks/results/round.json", "model": "other"}],
    })
    with pytest.raises(integrity.IntegrityError, match="Unselected checkpoint"):
        integrity.verify_focused(tmp_path, Counter())


def test_focused_summary_rejects_changed_analyzer_source(tmp_path):
    raw = tmp_path / "benchmarks/results/round.json"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"raw")
    analyzer = tmp_path / "scripts/analysis.py"
    analyzer.parent.mkdir(parents=True)
    analyzer.write_bytes(b"changed analysis")
    write_json(tmp_path / "paper/generated/focused-summary.json", {
        "checkpoints": {"gliner": "GLiNER"},
        "evidence": [{"file": "benchmarks/results/round.json", "model": "gliner",
                      "sha256": integrity.sha256(b"raw")}],
        "analysis_sources": {"scripts/analysis.py": integrity.sha256(b"frozen analysis")},
    })
    with pytest.raises(integrity.IntegrityError, match="mismatch.*analysis.py"):
        integrity.verify_focused(tmp_path, Counter())
