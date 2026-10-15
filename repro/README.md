# Reproduction profiles

CPU analysis and new GPU measurements have different reproducibility boundaries.
The archived results are the evidence; these profiles do not represent a new
performance experiment or retroactively lock the original containers.

## Locked CPU analysis

Use Python 3.12 and `analysis-py312.lock`, which pins the complete resolved
dependency set and distribution hashes for the CPU analyzer. It includes Modal
1.5.3, NumPy 2.4.6 and Matplotlib 3.11.2. No GPU libraries or checkpoint downloads
are required. CI installs this file with hash checking before regenerating the
focused summary and all ten TeX inputs, including the four-model overview.

```sh
python3.12 -m venv .venv-analysis
source .venv-analysis/bin/activate
python -m pip install --require-hashes -r repro/analysis-py312.lock
python scripts/verify_artifacts.py --git-history --regenerate
```

`analysis.in` records the primary pins used to resolve the lock. Updating the
lock requires rerunning the numerical regeneration check; an updated environment
is not evidence that the historical GPU timing measurements reproduce.
The older root `uv.lock` predates the paper extras and is not the paper analysis
lock. The package's broad optional dependency ranges are for development.

## Observed GPU dependency constraints

`fast-observed.constraints.txt` pins the performance-sensitive packages recorded
in the v3/v4 fast-image supplementary inventory, plus causal-conv1d from the
per-run environments. `gliner-observed.constraints.txt` pins only the six
packages recorded in all three selected GLiNER environments. Package names are
normalized for pip; the raw JSON preserves the original names and versions.

These are pip **constraints**, not installable complete requirements or container
locks. They constrain packages when those packages are installed. When rebuilding
an archived image, make the applicable file available inside the image and pass
`-c /path/to/constraints.txt` to each pip installation step. For the fast image,
retain CUDA 13.0, Python 3.12, the original build order and compiler settings,
install Torch before the extensions, and retain `--no-build-isolation` for
causal-conv1d. Do not install the GPU profiles into the lightweight local client.

Apply the fast profile only to its recorded image family. GLiNER's installation
can resolve additional dependencies, so the fast image's supplementary inventory
is not a full inventory of the GLiNER image. Unrecorded GLiNER transitive versions,
including sentencepiece and protobuf, remain unknown. We do not infer them from
another model's environment. The legacy Liquid ablation lacks causal-conv1d and
uses a different Python environment; it is not the v3/v4 fast profile.

The profiles contain no wheel hashes for original GPU binaries. Base-image tags,
system libraries, drivers, unrecorded transitive dependencies, build products and
persistent compiler caches remain partly uncontrolled. A reconstructed image is
a new experimental condition. Record its full resolved package inventory,
source commit, hardware and outcomes, and retain failures with new round IDs.

## Exact execution source

`source-index.json` maps every selected raw record to its recorded source commit,
resolved full Git commit, SHA-256, protocol, model, status and recorded package
environment. It is derived directly from the focused evidence and raw JSON.
Per-run source commits are distinct from the later manuscript/analysis revision.
The legacy v1 source snapshot remains reconstructed, with the limited provenance
reported by the artifact verifier.

See [the main reproduction guide](../docs/reproducibility.md) for the frozen
analysis revision, phase commands, source archives and one-GPU-at-a-time policy.
