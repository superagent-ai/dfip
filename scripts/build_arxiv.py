"""Prepare a minimal arXiv source ZIP and separate metadata; never upload it."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def paper_inputs(source):
    """Follow literal inputs so unused results, scratch files and PDFs stay out."""
    source = source.resolve()
    inputs = {}
    pending = ["main.tex", "main.bbl"]
    while pending:
        relative = pending.pop()
        path = (source / relative).resolve()
        if not path.is_relative_to(source):
            raise ValueError("Paper input escapes source directory: " + relative)
        name = path.relative_to(source).as_posix()
        if name in inputs:
            continue
        if not path.is_file():
            raise ValueError("Missing paper input: " + name)
        inputs[name] = path.read_bytes()
        if path.suffix != ".tex":
            continue
        text = path.read_text()
        for command, target in re.findall(r"\\(input|include|includegraphics)(?:\[[^\]]*\])?\{([^}]+)\}", text):
            if "\\" in target or "#" in target:
                raise ValueError("Nonliteral paper input requires explicit handling: " + target)
            pending.append(target if command == "includegraphics" or Path(target).suffix else target + ".tex")
        for names in re.findall(r"\\bibliography\{([^}]+)\}", text):
            pending.extend(name.strip() + ".bib" for name in names.split(","))
    return dict(sorted(inputs.items()))


def metadata(tex, pages, abstract_override=None):
    title = re.search(r"\\title\{([^}]+)\}", tex)[1].replace(r"\protect\\", " ")
    author = re.search(r"\\author\{([^}]+)\}", tex)[1].split(r"\\")
    names = [name.strip() for name in author[0].split(r"\qquad")]
    affiliation = " ".join(author[1:]).strip()
    abstract = abstract_override if abstract_override is not None else re.search(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)[1]
    abstract = abstract.replace(r"$\times$", "x").replace(r"\%", "%").replace("~", " ")
    if "\\" in title or "\\" in abstract or "\\" in affiliation:
        raise ValueError("Submission metadata needs manual handling of remaining TeX macros")
    abstract = " ".join(abstract.split())
    if len(abstract) > 1920:
        raise ValueError("Portal abstract exceeds arXiv's 1920-character limit; provide paper/arxiv-abstract.txt")
    if not all(value.isascii() for value in [title, abstract, affiliation, *names]):
        raise ValueError("arXiv metadata must use ASCII or supported TeX equivalents")
    figures = len(re.findall(r"\\begin\{figure\*?\}", tex))
    figure_word = "figure" if figures == 1 else "figures"
    return {
        "status": "prepared_not_submitted",
        "title": " ".join(title.split()),
        "authors": [{"name": name, "affiliation": affiliation} for name in names],
        "abstract": abstract,
        "suggested_primary_category": "cs.PF",
        "possible_cross_list": "cs.LG",
        "category_selection": "author must confirm during submission",
        "tex_processor": "PDFLaTeX",
        "top_level_tex": "main.tex",
        "comments": f"{pages} pages, {figures} {figure_word}. Code and reproducible results: https://github.com/superagent-ai/dfip",
        "license": None,
        "license_selection": "authors must choose in arXiv; repository MIT license does not select the article license",
        "journal_reference": None,
        "doi": None,
        "arxiv_identifier": None,
    }


def build(output):
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip():
        raise ValueError("Commit tracked source changes before packaging")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    tracked = set(subprocess.check_output(["git", "ls-files", "paper"], cwd=ROOT, text=True).splitlines())
    files = paper_inputs(ROOT / "paper")
    if any("paper/" + name not in tracked for name in files):
        raise ValueError("Every paper input, including main.bbl, must be tracked")
    build_record = json.loads((ROOT / "paper/review-build.json").read_text())
    pdf = ROOT / "paper/dfip.pdf"
    digest = lambda data: hashlib.sha256(data).hexdigest()
    if digest(pdf.read_bytes()) != build_record["paper_sha256"]:
        raise ValueError("Paper PDF differs from its build record")
    if digest(files["main.bbl"]) != build_record.get("bibliography_sha256"):
        raise ValueError("Generated bibliography differs from its build record")
    abstract_file = ROOT / "paper/arxiv-abstract.txt"
    if abstract_file.exists() and "paper/arxiv-abstract.txt" not in tracked:
        raise ValueError("The portal abstract must be tracked")
    submission = metadata(files["main.tex"].decode(), build_record["pages"],
                          abstract_file.read_text() if abstract_file.exists() else None)
    submission["abstract_source"] = "paper/arxiv-abstract.txt" if abstract_file.exists() else "paper/main.tex"
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Use an empty output directory to preserve previous assets")
    upload = output / "dfip-arxiv-source.zip"
    with zipfile.ZipFile(upload, "w") as archive:
        for name, payload in files.items():
            entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, payload)
    (output / "submission-metadata.json").write_text(json.dumps(submission, indent=2) + "\n")
    (output / "submission-abstract.txt").write_text(submission["abstract"] + "\n")
    shutil.copy2(pdf, output / "dfip-paper.pdf")
    manifest = {
        "source_commit": commit,
        "compiled_source_commit": build_record["compiled_source_commit"],
        "status": "prepared_not_submitted",
        "upload_file": upload.name,
        "source_files": {name: digest(payload) for name, payload in files.items()},
        "assets": {p.name: {"sha256": digest(p.read_bytes()), "bytes": p.stat().st_size}
                   for p in sorted(output.iterdir()) if p.is_file()},
        "validation": "source closure and PDF hash verified; compilation of extracted ZIP must be checked separately",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"output": str(output), "source_commit": commit, "upload_members": len(files)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/arxiv")
    args = parser.parse_args()
    try:
        print(json.dumps(build(args.output), indent=2))
    except (ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"arXiv package failed: {error}\n")


if __name__ == "__main__":
    main()
