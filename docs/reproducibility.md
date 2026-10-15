# Reproducing the DFIP evidence

The public research preview contains the client and experimental paths, the paper source and PDF, raw measurements, failed attempts, and historical source snapshots. Model weights and downloaded corpora are not redistributed. The focused paper covers PPLX-Decider v1.1 27B, Liquid d1-3B, Clef-Flash, and GLiNER2.5-Decide; older experiments remain in the archive for provenance.

## Check the archive without a GPU

From a clone or unpacked repository archive, run:

```sh
python scripts/verify_artifacts.py
```

This uses the Python standard library and needs no Modal or Hugging Face credentials. It checks all recorded result SHA-256 manifests, source ZIP manifests, compressed and decoded log hashes, the paper PDF hash, and the selected evidence and analyzer hashes in `paper/generated/focused-summary.json`. It also verifies the pinned upstream source audit ZIP against `benchmarks/research-audits/model-contracts.json`, checks its 58 listed source-file hashes and byte counts, and matches its embedded source manifests. The ZIP's 68 members include metadata covered by the whole-archive hash. Nested result paths and the historical manifest formats are handled explicitly. A missing or changed artifact exits with an error; the verifier does not replace its expected hash.

A full Git clone can also check archived source contents against their recorded commits:

```sh
python scripts/verify_artifacts.py --git-history
```

The original v1 source ZIP is a reconstructed snapshot, whose manifest records hashes for 29 of its 42 files and no whole-ZIP hash. It cannot establish the same provenance as the later exact snapshots. The verifier reports this limitation. The v3–v5 source archives have whole-ZIP hashes; their contents can also be matched to Git history. Frozen shared-policy checks apply to commits used in the shared studies, not an earlier native-only snapshot that contains an earlier scheduler implementation.

Logs are supplementary local captures. Their manifests distinguish hashes of original local output from the ANSI-stripped bytes actually archived. Only the compressed and decoded archived bytes can be checked from this release. Some final remote output was not delivered to the local log; a valid checksum does not make a partial log complete.

## Recompute the paper inputs on CPU

The frozen analysis/evidence revision is
[`f2eef1746e5e39b92827759344d4ef47410fc464`](https://github.com/superagent-ai/dfip/tree/f2eef1746e5e39b92827759344d4ef47410fc464).
It includes the raw archive, analyzer, overview-table generator, per-run source
index and dependency profiles. It is distinct from the original GPU execution
commits and the later manuscript build recorded in `paper/review-build.json`.
Use a full clone and the hash-locked Python 3.12 analysis environment:

```sh
git clone https://github.com/superagent-ai/dfip.git
cd dfip
git checkout f2eef1746e5e39b92827759344d4ef47410fc464
python3.12 -m venv .venv-analysis
. .venv-analysis/bin/activate
python -m pip install --require-hashes -r repro/analysis-py312.lock
python scripts/verify_artifacts.py --git-history --regenerate
```

The last command runs the focused analyzer in an isolated temporary copy. It compares the summary and ten generated TeX inputs, including the four-model overview, byte for byte with the checked-in files. It also regenerates the workload figure. PNG pixels are not compared byte for byte because fonts and plotting libraries can render differently across platforms. The numerical data and the text inputs must match; the tracked archive is never overwritten by this verification. The CPU lock has been validated in a clean Python 3.12 environment; it is not the original GPU environment.

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

The [per-run source index](../repro/source-index.json) covers all 30 selected
records, including failures and exclusions. Its nine execution revisions are:

| Execution phase | Exact source revision |
| --- | --- |
| Decider strengthened resident comparison | [`731e3f2`](https://github.com/superagent-ai/dfip/tree/731e3f26aca477d4726010345dae724bc13d3c4b) |
| Clef strengthened resident comparison, including failed preflight | [`b776896`](https://github.com/superagent-ai/dfip/tree/b7768962a93c43ef18230163d0079ece17708357) |
| Liquid compiler comparison; Decider/Clef literal audits and external serving | [`bef713c`](https://github.com/superagent-ai/dfip/tree/bef713cac4a57a3f1092abc5d7cb35d63a3399c5) |
| Initial bounded-admission pilot, including excluded Clef attempt | [`8c9fc08`](https://github.com/superagent-ai/dfip/tree/8c9fc08a9b61fd6876ab14acf79e9421048050b0) |
| Corrected Clef bounded-admission pilot | [`bc1bfa1`](https://github.com/superagent-ai/dfip/tree/bc1bfa16e272f2d04a858c364c99110213b8bc44) |
| GLiNER native conformance/profiling | [`01b6f87`](https://github.com/superagent-ai/dfip/tree/01b6f8723a972cac2cdf5b2b0298fde594c427cb) |
| GLiNER eager shared scheduling | [`7356ac7`](https://github.com/superagent-ai/dfip/tree/7356ac7f169ede4d3df767709bd1b4fecd814193) |
| GLiNER compiled follow-up | [`4adadc3`](https://github.com/superagent-ai/dfip/tree/4adadc36c6cc2e0cd95dad0b6c67a07462649886) |
| Legacy Liquid ablation; reconstructed source snapshot | [`822f6c3`](https://github.com/superagent-ai/dfip/tree/822f6c3c84d66990084f006562773df85b2545a4) |

For an exact source snapshot, use its recorded commit in a separate Git worktree or unpack its source ZIP into a separate directory. Read that snapshot's protocol before starting a job. The v5 driver expects a local `.venv/bin/modal`, completed preparation receipts, and a `work/` directory. Create `work/` before running historical drivers. Preparation receipts point to the study's remote Volume; copied receipts are not proof that a new Volume is prepared. Commit source changes before new experiments and use new round identifiers. Preserve failures and excluded rounds, source ZIPs, environment records and request accounting alongside successful results.

The [reproduction profiles](../repro/README.md) provide pip constraints for 19
observed fast-image package versions and six recorded GLiNER package versions.
The verifier checks those pins against the archived evidence and checks the
source index against the raw records, including per-mode setup environments.
The profiles pin recorded performance-sensitive libraries; they do not infer
unrecorded transitive versions or supply hashes for the original GPU binaries.
Apply the appropriate constraints inside the reconstructed image, with the
recorded CUDA/Python family, compiler settings and extension build order.
The profiles are not applied to historical snapshots automatically. Any source
or image change needs a new recorded execution revision and new measurements.

Historical inventories are not complete dependency lockfiles. In particular,
GLiNER's sentencepiece/protobuf versions were not recorded and are not borrowed
from another image. Checkpoint and compiler caches persisted on remote Volumes.
Fresh runs can therefore differ in unrecorded transitive dependencies, system
libraries, drivers, GPU scheduling, cold setup and compiler behavior even with
the same source. Resident timings exclude HTTP and cold start; external
Decider/Clef serving and short resident GLiNER traces have different timing
boundaries. Compare drops and errors alongside successful-request latency.

## What continuous integration verifies

CPU CI runs package tests on Python 3.10 and 3.12, checks the wheel and source distribution notices, and imports an installed wheel from outside the checkout without GPU libraries. A separate Python 3.12 job installs `repro/analysis-py312.lock` with hash checking, checks the research archive against full Git history and recomputes focused summary and table inputs. CI uses no cloud credentials, schedules no GPU and does not reproduce the published performance measurements.
