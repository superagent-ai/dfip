"""Typeset the multi-file LaTeX review draft on a CPU-only Modal worker."""

import json
from pathlib import Path
import modal

app = modal.App("dfip-paper-typeset")
image = modal.Image.debian_slim(python_version="3.12").apt_install(
    "latexmk", "texlive-latex-extra", "texlive-fonts-recommended", "texlive-science"
)


@app.function(image=image, cpu=2, timeout=1200, max_containers=1)
def typeset(files):
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        for name, content in files.items():
            path = root / name
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("Invalid paper path")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        process = subprocess.run(
            ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
            cwd=root, capture_output=True, text=True, timeout=240,
        )
        log = (root / "main.log").read_text() if (root / "main.log").exists() else process.stdout
        if process.returncode:
            raise RuntimeError((process.stdout + process.stderr)[-16000:])
        return {"pdf": (root / "main.pdf").read_bytes(), "log": log}


@app.local_entrypoint()
def main(source: str = "paper", output: str = "paper/dfip.pdf"):
    root = Path(source)
    files = {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and path.suffix in (".tex", ".bib", ".png", ".sty", ".cls")
    }
    result = typeset.remote(files)
    dest = Path(output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(result["pdf"])
    Path("work/paper-typeset.log").write_text(result["log"])
    print(json.dumps({"saved": str(dest), "bytes": dest.stat().st_size}))
