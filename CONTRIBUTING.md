# Contributing

DFIP is a research preview. Contributions should improve the client,
reproducibility, or measured inference behavior without overstating the results.
Open an issue before a substantial new adapter or change to an experimental
protocol so its scope and native reference can be reviewed.

## Local development

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
ruff check src tests
python -m compileall -q src
dfip --help
```

These checks require no GPU. CI runs the CPU tests and package build on Python
3.10 and 3.12. GPU measurements must be run explicitly on Modal and incur usage.
Use your own research app and Volume, run one GPU application at a time, and keep
existing deployments separate.

The lightweight client lives in `src/dfip/client.py`, GPU execution in
`src/dfip/runtime/`, and Modal definitions in `src/dfip/cloud/`. Cloud images
include the installed package source. Top-level imports must not load GPU
libraries or weights.

## Model and optimizer changes

1. Pin an immutable checkpoint revision and record the upstream license.
2. Implement the released native behavior before adding an optimization.
   Preserve prompt formatting, normalization, calibration, pooling, aliases,
   typed outputs, and exposed confidence fields.
3. Test supported schemas, changing input values, candidate sets, batching, and
   native fallbacks. A shared interface does not imply a shared output contract.
4. Compare against the strongest applicable released native path. Keep compiler
   settings, dtype, optional kernels, and timing scope explicit.
5. Record decision changes, probability and score drift, confidence-threshold
   crossings, latency, cold setup, memory, drops, and errors. A stable winning
   label alone is not evidence of application equivalence.

Client deployment defaults remain native. Moving an optimization into those
defaults requires reviewed fidelity and serving evidence, including varied
traffic. A microkernel speedup or a native fallback is not sufficient evidence
of end-to-end acceleration. GLiNER currently has a research adapter only; adding
it to the client requires a separate API and deployment review.

## Reproducible research

The [reproducibility guide](docs/reproducibility.md) and
[benchmark archive](benchmarks/README.md) document protocols and reproduction
commands. Regenerate the current paper tables with:

```sh
python -m pip install -e '.[paper]'
PYTHONPATH=src python scripts/analyze_focused.py
```

Commit experimental source before running a new study. Preserve raw paired
samples, source and checkpoint revisions, environment details, workload inputs,
and failed or excluded attempts. Explain exclusions; never silently replace a
failed round. Keep distinct phases and statistical units separate. Archived
records should not be rewritten to match a new implementation.

Pull requests should describe the behavior changed, the evidence supporting it,
and relevant checks. Include enough information to reproduce a result, with a
clear statement of whether transport, admission, and cold starts are timed. The
public typed-decision corpus does not replace application-specific evaluation.

Never commit model weights, downloaded datasets, credentials, private request
data, or logs containing them. Use synthetic or public examples. Model and
dataset terms apply independently of DFIP's [MIT license](LICENSE).
