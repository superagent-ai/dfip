"""A source upload must be complete, minimal, and contained in its directory."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "arxiv_bundle", Path(__file__).parents[1] / "scripts/build_arxiv.py")
bundle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bundle)


def test_upload_closure_keeps_only_referenced_assets_and_bibliography(tmp_path):
    (tmp_path / "main.tex").write_text(r"\input{table}\includegraphics[width=1cm]{figure.png}\bibliography{references}")
    (tmp_path / "table.tex").write_text("table")
    (tmp_path / "main.bbl").write_text("bibliography")
    (tmp_path / "references.bib").write_text("references")
    (tmp_path / "figure.png").write_bytes(b"figure")
    for name in ["main.pdf", "draft.tex", "unused.png", "main.log"]:
        (tmp_path / name).write_text("do not upload")
    assert set(bundle.paper_inputs(tmp_path)) == {
        "main.tex", "table.tex", "figure.png", "main.bbl", "references.bib"}


def test_upload_closure_rejects_a_missing_referenced_asset(tmp_path):
    (tmp_path / "main.tex").write_text(r"\input{missing}")
    (tmp_path / "main.bbl").write_text("bibliography")
    with pytest.raises(ValueError, match="Missing paper input: missing.tex"):
        bundle.paper_inputs(tmp_path)


def test_upload_closure_rejects_paths_outside_the_submission(tmp_path):
    (tmp_path / "main.tex").write_text(r"\input{../private}")
    (tmp_path / "main.bbl").write_text("bibliography")
    with pytest.raises(ValueError, match="escapes source directory"):
        bundle.paper_inputs(tmp_path)
