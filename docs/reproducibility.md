# Reproducing the DFIP evidence

The public research preview contains the client and experimental paths, the paper source and PDF, raw measurements, failed attempts, and historical source snapshots. Model weights and downloaded corpora are not redistributed. The focused paper covers PPLX-Decider v1.1 27B, Liquid d1-3B, Clef-Flash, and GLiNER2.5-Decide; older experiments remain in the archive for provenance.

## Check the archive without a GPU

From a clone or unpacked repository archive, run:

```sh
python scripts/verify_artifacts.py
```

This uses the Python standard library and needs no Modal or Hugging Face credentials. It checks all recorded result SHA-256 manifests, source ZIP manifests, compressed and decoded log hashes, the paper PDF hash, and the selected evidence and analyzer hashes in `paper/generated/focused-summary.json`. Nested result paths and the historical manifest formats are handled explicitly. A missing or changed artifact exits with an error; the verifier does not replace its expected hash.

A full Git clone can also check archived source contents against their recorded commits:

```sh
python scripts/verify_artifacts.py --git-history
```

The original v1 source ZIP is a reconstructed snapshot, whose manifest records hashes for 29 of its 42 files and no whole-ZIP hash. It cannot establish the same provenance as the later exact snapshots. The verifier reports this limitation. The v3–v5 source archives have whole-ZIP hashes; their contents can also be matched to Git history. Frozen shared-policy checks apply to commits used in the shared studies, not an earlier native-only snapshot that contains an earlier scheduler implementation.

Logs are supplementary local captures. Their manifests distinguish hashes of original local output from the ANSI-stripped bytes actually archived. Only the compressed and decoded archived bytes can be checked from this release. Some final remote output was not delivered to the local log; a valid checksum does not make a partial log complete.

## Recompute the paper inputs on CPU

Create a Python 3.12 environment and install the client and plotting dependencies:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[paper]'
python scripts/verify_artifacts.py --regenerate
```

The last command runs the focused analyzer in an isolated temporary copy. It compares the summary and nine generated TeX inputs byte for byte with the checked-in files. It also regenerates the workload figure. PNG pixels are not compared byte for byte because fonts and plotting libraries can render differently across platforms. The numerical data and the text inputs must match; the tracked archive is never overwritten by this verification.

To deliberately regenerate editable paper inputs in the checkout, run:

```sh
PYTHONPATH=src python scripts/analyze_focused.py
```

The analyzer independently recomputes paired worker statistics, resident medians, request accounting and available answer audits. It records the 30 selected raw result paths and seven analysis-source hashes. It retains failed or excluded attempts and keeps different protocols separate. Matching these calculations confirms the archived analysis, not a new hardware replication.

Compile the manuscript locally with a LaTeX installation:

```sh
cd paper
latexmk -pdf main.tex
```

Alternatively, the repository's `dfip.cloud.paper` entry point uses a Modal CPU worker. This allocates no GPU but does require a Modal account and may incur CPU charges. See [paper instructions](../paper/README.md).

## Rerun measurements on Modal

New GPU measurements require a Modal account, permission to use the pinned models, accepted model licenses or access terms, and the study-specific preparation steps. GPU runs incur charges. Use a new research Volume and keep **one study GPU application running at a time**. Use a fresh checkout or a separate worktree so new records cannot overwrite this release's evidence.

The experiment notes identify the immutable source snapshots, protocols, preparation commands, workload dimensions and driver options:

| Phase | Instructions | Archived evidence |
| --- | --- | --- |
| Optimized native and released-compiler comparisons; external serving | [Strengthened study](../benchmarks/strengthened-study.md) | `benchmarks/results/strengthened/` |
| Post hoc bounded graph-admission pilot | [Adaptive study](../benchmarks/adaptive-study.md) | `benchmarks/results/adaptive/` |
| GLiNER native conformance, shared scheduling and released compiler | [Shared batching study](../benchmarks/shared-batching-study.md) and [compiler protocol](../benchmarks/shared-compiled-followup-protocol.json) | `benchmarks/results/contracts/` |
| Historical Liquid component ablation | [Original archive](../benchmarks/README.md) and `source-v1-manifest.json` | `benchmarks/results/liquid-round-*.json` |

For an exact source snapshot, use its recorded commit in a separate Git worktree or unpack its source ZIP into a separate directory. Read that snapshot's protocol before starting a job. The v5 driver expects a local `.venv/bin/modal`, completed preparation receipts, and a `work/` directory. Create `work/` before running historical drivers. Preparation receipts point to the study's remote Volume; copied receipts are not proof that a new Volume is prepared. Commit source changes before new experiments and use new round identifiers. Preserve failures and excluded rounds, source ZIPs, environment records and request accounting alongside successful results.

The retained package inventories describe the observed execution environments; they are not complete dependency lockfiles. Checkpoint and compiler caches persisted on remote Volumes. Fresh runs can therefore differ in transitive dependencies, GPU scheduling, cold setup and compiler behavior even with the same source. Resident timings exclude HTTP and cold start; external Decider/Clef serving and short resident GLiNER traces have different timing boundaries. Compare drops and errors alongside successful-request latency.

## What continuous integration verifies

CPU CI runs package tests on Python 3.10 and 3.12, checks the wheel and source distribution notices, and imports an installed wheel from outside the checkout without GPU libraries. A separate Python 3.12 job checks the research archive against full Git history and recomputes focused summary and table inputs. CI uses no cloud credentials, schedules no GPU and does not reproduce the published performance measurements.
