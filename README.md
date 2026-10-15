# DFIP: Decision-Focused Inference Path

**A research preview for System One model inference on Modal.** DFIP provides a
small Python client, pinned model adapters, and reproducible experiments for
System One models (also known as decision models). The client runs locally;
weights and GPU execution stay in your Modal account.

All client deployments use the released native inference paths. CUDA graph
replay, Triton candidate projection, and shared batching are separate research
paths. The experiments show useful gains on some resident workloads, but do not
establish a general acceleration method or production readiness.

The research applies established execution techniques to four pinned models.
Its contribution is the comparison against stronger native baselines, the
audit of exposed answers, and the separate serving evaluation. The findings
show where resident acceleration survives a stronger baseline, where compilation
closes the gap, and where changing traffic or confidence drift limits the gain.

## Install and run

Requires Python 3.10+, a Modal account with GPU access, and access to the selected
Hugging Face checkpoint. Install from this repository:

```sh
git clone https://github.com/superagent-ai/dfip.git
cd dfip
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
modal setup

# Choose names that do not replace an existing deployment or reuse its cache.
export DFIP_APP=dfip-preview
export DFIP_VOLUME=dfip-preview-models
dfip setup --model liquid
dfip predict --model liquid --input examples/request.json
```

`setup` downloads only the selected pinned checkpoint into a Modal Volume, then
deploys authenticated Modal methods. The quick start uses an L40S. Preparing and
calling models incurs Modal usage; containers scale to zero after 60 seconds of
inactivity, and the first call includes model loading. No public HTTP endpoint
is created. Keep the environment variables set for subsequent commands.

For a gated checkpoint, accept its Hugging Face terms and create a Modal secret
named `dfip-hf` containing `HF_TOKEN`. Set `DFIP_HF_SECRET=dfip-hf` before running
`dfip prepare --model <alias>`. Credentials belong in that secret, not in request
files. Preparation injects the secret into the download worker.

## Python client

```python
from dfip import Client

client = Client("liquid")
result = client.predict([
    {
        "state": "I was charged twice. Please refund the duplicate payment.",
        "questions": {
            "team": {
                "type": "choice",
                "instructions": "Which team should handle this request?",
                "criteria": {
                    "billing": "Payments and refunds",
                    "technical": "Software bugs",
                },
            },
        },
    },
])
print(result[0]["answers"]["team"]["choice"])
```

`await client.predict_async(records)` supports asynchronous calls.
`client.diagnostics()` reports the remote execution policy. Text requests support
`choice`, `noul` (yes/no), and `score` questions. Outputs retain each released
model's `answers` format; extra fields differ between models. See the
[request example](examples/request.json) for the input format.

## Client models

| Alias | Pinned checkpoint | GPU | Default inference |
| --- | --- | --- | --- |
| `julia` | [SupersonicLabs/Julia-1](https://huggingface.co/SupersonicLabs/Julia-1) | L40S | Native FastEngine |
| `liquid` | [LiquidAI/d1-3B](https://huggingface.co/LiquidAI/d1-3B) | L40S | Released native engine |
| `clef` | [Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash) | L40S | Released native API |
| `decider` | [perplexity-ai/pplx-decider-v1.1-27b](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b) | B200 | Released native API |
| `security` | [superagent-ai/security-one-27b](https://huggingface.co/superagent-ai/security-one-27b) | B200 | Native SGLang 0.5.19 |

`dfip models` prints immutable checkpoint revisions. Aliases and the listed
Hugging Face IDs are accepted. Prepare each model before calling it; downloading
all models is unnecessary for the quick start. Arbitrary Hugging Face models
are not supported automatically. **GLiNER2.5-Decide is a research adapter and is
not available through this client registry.**

## Research results

The [paper and analysis](paper/README.md) study PPLX-Decider, Liquid d1-3B,
Clef-Flash, and GLiNER2.5-Decide. The manuscript is a review draft and has not
been submitted to arXiv.

| Comparison | Finding | Scope |
| --- | --- | --- |
| PPLX-Decider replay vs strengthened native | 2.47× [1.88, 3.26] | Short Boolean resident calls, five completed container rounds |
| Clef-Flash replay vs strengthened native | 1.45× [1.33, 1.71] | Same resident workload, four completed rounds; one native-preflight failure retained |
| Liquid vs released compiled native | 1.01× [0.97, 1.09] | Same resident workload, three rounds; effectively tied |
| GLiNER shared batching vs released compiled native batching | 1.40× | Diverse resident trace, one worker; exploratory follow-up |

The bracketed ranges are paired 95% bootstrap confidence intervals where shown.
These phases are not pooled. Resident timings exclude client transport and cold
starts. Changing external traffic reverses the original Decider/Clef replay
advantage through recapture, queues, drops, and a retained CUDA failure. Clef
also changes probabilities and some confidence-threshold outcomes. GLiNER's
compiled shared scheduler improves resident completion time, but has worse p95
latency than compiled native batching in its short arrival replay. These limits
are part of the evidence, not resolved serving guarantees.

Regenerate the four-model tables from the archived results without a GPU, using
the hash-locked Python 3.12 analysis environment:

```sh
python3.12 -m venv .venv-analysis
source .venv-analysis/bin/activate
python -m pip install --require-hashes -r repro/analysis-py312.lock
python scripts/verify_artifacts.py --git-history --regenerate
PYTHONPATH=src python scripts/analyze_focused.py
```

[Reproducibility instructions](docs/reproducibility.md) and the
[benchmark archive](benchmarks/README.md) link the protocols, raw records, failed
runs, source snapshots, and Modal reproduction commands. Run new GPU experiments
one GPU application at a time, in separate research apps and Volumes. Earlier
studies remain in the archive for provenance and do not contribute to the
focused paper's claims. The opt-in adaptive serving pilot is documented
separately in [adaptive serving](docs/adaptive-serving.md).

The [reproduction profiles](repro/README.md) distinguish the locked CPU analysis
from observed GPU dependency constraints. The [per-run source index](repro/source-index.json)
records exact commits and environments for all 30 selected raw records; the
historical GPU environments were not fully locked.

## Configuration and checks

| Variable | Default | Purpose |
| --- | --- | --- |
| `DFIP_APP` | `dfip-inference` | Modal deployment name |
| `DFIP_VOLUME` | `dfip-models-v1` | Checkpoint and cache Volume |
| `DFIP_HF_SECRET` | unset | Optional download secret containing `HF_TOKEN` |

Set these consistently for preparation, deployment, evaluation, and inference.
`Client("liquid", app_name="my-dfip", environment_name="staging")` selects a
deployment explicitly. Modal's environment selection applies to deployment.

After checkpoint preparation, the following checks run on Modal and incur usage:

```sh
dfip smoke --model liquid --output liquid-smoke.json
dfip validate --model liquid --output liquid-validation.json
dfip benchmark --model liquid --quality 400 --repetitions 100 --output liquid-benchmark.json
```

`smoke` and `benchmark` support all five client models; `validate` supports the
four in-process adapters. These commands check the packaged default path.
Experimental comparisons use the separate research entry points.

For local development:

```sh
python -m pip install -e '.[dev]'
pytest
ruff check src tests
```

CI checks the CPU client and packaging on Python 3.10 and 3.12. GPU studies run
explicitly on Modal. See [contributing](CONTRIBUTING.md), [support](SUPPORT.md),
[security policy](SECURITY.md), and [release notes](CHANGELOG.md). Citation
metadata is in [CITATION.cff](CITATION.cff).

DFIP source is [MIT licensed](LICENSE). Model weights and downloaded datasets
are not bundled. The research archive includes pinned upstream source snapshots
under their own terms; see [third-party notices](THIRD_PARTY_NOTICES.md). Model
licenses apply independently, including Liquid's commercial-use conditions.
Pinned upstream custom Python code executes remotely where required by a model.
