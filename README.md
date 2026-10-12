# DFIP

**Decision-focused inference on Modal.** Run open decision models through a common
Python client with conservative model-specific defaults. Python/PyTorch handles
model execution. The research paths evaluate Triton candidate projection and
CUDA graph replay. Serving defaults use the released native paths.

The [review paper](paper/README.md) focuses on PPLX-Decider, Liquid d1-3B,
Clef-Flash and GLiNER2.5-Decide. It evaluates graph replay, candidate projection
and schema-aware batching against stronger native baselines. Some resident
workloads improve; numerical drift and changing serving traffic limit those
gains. The prototype does not establish a general acceleration method or
production readiness. Earlier experiments remain archived for provenance.

Model weights, datasets, and GPU dependencies stay on Modal. The local package
contains only the client and deployment source. DFIP currently supports text
requests with `choice`, `noul` (yes/no), and `score` questions.

## Quick start

Requires Python 3.10+, a Modal account with GPU access, and access to your chosen
Hugging Face checkpoint. The quick start uses Liquid on an L40S; the 27B models use a B200.

```sh
git clone https://github.com/superagent-ai/dfip.git
cd dfip
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
modal setup
dfip setup --model liquid
dfip predict --model liquid --input examples/request.json
```

`setup` downloads the pinned model into your Modal volume and deploys the private
`dfip-inference` app in your account. Only the selected model is downloaded. Other
classes become usable after preparing their checkpoint. No API keys belong in
request files. Predictions use your authenticated Modal SDK credentials.

For a gated model, first accept its Hugging Face terms and create a Modal secret
named `dfip-hf` with an `HF_TOKEN` key in the Modal dashboard. Then:

```sh
export DFIP_HF_SECRET=dfip-hf
dfip prepare --model security
dfip predict --model security --input examples/request.json
```

There is no public HTTP endpoint in this release. Containers scale to zero after
60 seconds of inactivity. A first request includes model loading and engine setup;
Security-One also starts its SGLang server. Modal bills preparation, execution,
and the idle window according to your account's rates.

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
`client.diagnostics()` reports the chosen optimization and graph activity.
Outputs retain the released model's `answers` format, including probabilities
and typed values; model-specific extra fields may differ. Score criteria are an
ordered list of descriptions; `noul` criteria optionally describe `false` and
`true`. See [the request example](examples/request.json).

## Models

| Alias | Hugging Face checkpoint | Default GPU | Serving path |
| --- | --- | --- | --- |
| `julia` | [SupersonicLabs/Julia-1](https://huggingface.co/SupersonicLabs/Julia-1) | L40S | Native FastEngine; replay under investigation |
| `liquid` | [LiquidAI/d1-3B](https://huggingface.co/LiquidAI/d1-3B) | L40S | Released native engine; acceleration experimental |
| `clef` | [Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash) | L40S | Released native API; replay experimental |
| `decider` | [perplexity-ai/pplx-decider-v1.1-27b](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b) | B200 | Released native API; replay experimental |
| `security` | [superagent-ai/security-one-27b](https://huggingface.co/superagent-ai/security-one-27b) | B200 | Native SGLang 0.5.19 |

`dfip models` lists immutable checkpoint revisions. `dfip prepare --model all`
downloads all five checkpoints remotely. HF IDs also work wherever aliases do.
The [decision-model tag](https://huggingface.co/models?other=decision-model) is a
source of candidates for future adapters; arbitrary tagged models are not
compatible automatically. Model weights and downloaded upstream code retain
their own licenses. Pinned upstream custom Python code executes in the remote
container when required by the released model.

## Configuration

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `DFIP_APP` | `dfip-inference` | Modal deployment name |
| `DFIP_VOLUME` | `dfip-models-v1` | Persistent checkpoint and cache volume |
| `DFIP_HF_SECRET` | unset | Optional Modal secret containing `HF_TOKEN` |

Set these consistently for prepare, deploy, evaluation, and inference. For
example, use `DFIP_APP=my-dfip DFIP_VOLUME=my-dfip-models dfip setup --model julia`.
Use `Client("julia", app_name="my-dfip", environment_name="staging")` to select a
deployment explicitly. Modal's own environment selection applies to deployment.

## Benchmarks and verification

The review branch defaults all five models to released native inference, with
Security-One using native SGLang. GPU failures, probability drift, and sustained
serving losses keep custom acceleration in the research paths. These source defaults
have not been deployed to an existing application. Experimental paths remain
accessible through the research entry points.

The separate strengthened study installs optional native Qwen kernels and tests the released Liquid compiler setting. Decider retains a short Boolean speedup of 2.47× (five completed container rounds); Clef gives 1.45× conditional on four completed rounds, with a fifth failing in native preflight. Liquid is effectively tied with compiled native on that workload. These are resident measurements, not production readiness evidence. [Study protocol and results](benchmarks/strengthened-study.md) include direct answer checks and external serving measurements.

The sustained tests reverse the Decider/Clef replay advantage: changing shapes cause frequent recapture, large queues, and drops, with one Decider CUDA failure. The separate bounded-admission pilot reduces recapture work but does not establish uniform serving gains or Clef confidence equivalence.

GLiNER2.5-Decide is a research adapter, separate from the public client registry. Its shared scheduler is 1.57× faster than eager native batching and 1.40× faster than released compiled native batching on the diverse resident trace, with one worker per substrate and passing exposed-label/confidence audits. The short varied-traffic replay does not show an incremental tail-latency gain: compiled shared p95 is 72.1 ms versus 43.2 ms for compiled native batching. The compiler follow-up reuses fixtures after the initial result and is exploratory. See the [focused paper results](paper/generated/focused-summary.json); these phases are not pooled with the earlier replay study.

Repeated resident short Boolean calls in the strengthened image:

| Model | Completed/planned rounds | Native p50 | DFIP p50 | Paired speedup, 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Clef, L40S | 4/5 | 51.69 ms | 35.68 ms | 1.45× [1.33, 1.71] |
| Decider, B200 | 5/5 | 74.54 ms | 30.14 ms | 2.47× [1.88, 3.26] |
| Liquid, L40S, compiled native | 3/3 | 10.74 ms | 10.66 ms | 1.01× [0.97, 1.09] |

Point estimates are geometric means of paired container medians, with 50 warm
samples per workload and mode. Clef's interval is conditional on retained rounds.
They exclude external network latency, cold starts, admission, and idle billing.
The compiler and graph paths can change probabilities even when choices agree;
Clef's direct audit also finds returned-score drift. The original exploratory
[report](docs/dfip-report.html), [summary CSV](docs/dfip-benchmarks.csv), and
[raw results](docs/dfip-benchmarks.json) are historical development evidence and
are not pooled into these estimates. See the [paper](paper/README.md) for failures,
workload scope, literal answer checks, and serving results. Regenerate its four-model
tables with `PYTHONPATH=src python scripts/analyze_focused.py`.

```sh
# Quick GPU/kernel check, downloaded weights remain on Modal.
dfip smoke --model liquid --output liquid-smoke.json

# Full 400-case fidelity + 128 varied single-question checks.
dfip validate --model liquid --output liquid-validation.json

# Benchmark the selected policy; experimental acceleration is in the research entry points.
dfip benchmark --model liquid --quality 400 --repetitions 100 --output liquid-benchmark.json
```

All five models support `smoke` and `benchmark`. Security's benchmark measures
its selected native SGLang path; its head ablation is a separate experiment:

```sh
modal run -m dfip.cloud.sglang_benchmark --quality 400 --repetitions 100 --output security-head-ablation
```

`validate` covers the four in-process adapters. Benchmark and validation result
JSON is returned locally; model weights and the dataset stay in the remote volume.

## Experimental varied-traffic support

Clef and Decider also have a separate opt-in adaptive app. It admits frequent shapes into at most four pinned graphs and uses native inference for the rest, bounding recapture work under changing traffic. The pilot has uneven gains, and Clef still changes confidence-threshold routing. It is exploratory and does not establish that the original CUDA failures are fixed. See [API usage](docs/adaptive-serving.md) and the separate [pilot protocol and outcomes](benchmarks/adaptive-study.md).

## Operational limits

Graph caches store computation, never predictions. Replay calls within one engine are
serialized because CUDA graph input buffers are mutable. Security-One dispatches
concurrently to its native scheduler. A shape becomes eligible
after three matching calls; the graph cache holds four shapes and falls back to
native inference for unsupported captures. Liquid updates the answer-token union
and invalidates graphs when new candidate tokens appear.

Liquid/Clef/Decider graph capture is limited to sequences of at most 512 tokens.
Clef/Decider additionally require a single-row forward. Decider multi-question
and batched requests use native execution because padded hybrid graphs failed
stress validation. Julia preserves its strict 8,192-token encoding limit.
Security-One supports at most 8,192 tokens and 255 options, without truncation.
Other adapters retain the released APIs' context constraints.

Experimental RMSNorm replacements, Julia marker-only projection, and Security
head patches are excluded from serving defaults. Some changed probabilities or
decisions; see the report before using them. Custom projection retains the
original model weights, so it reduces readout computation rather than model size.

## Development

```sh
python -m pip install -e '.[dev]'
pytest
ruff check src tests
```

CI checks packaging, local client behavior, and imports without GPU libraries on
Python 3.10 and 3.12. GPU tests run explicitly on Modal and incur cloud usage.
See [CONTRIBUTING.md](CONTRIBUTING.md) for adding a model safely. DFIP source is
MIT licensed; checkpoint licenses and dataset terms apply independently.
