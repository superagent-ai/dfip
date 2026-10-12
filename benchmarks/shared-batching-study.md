# Shared decision-contract batching pilot

This v5 exploratory phase extends the study to five pinned checkpoints in four additional families: Liquid d1-omni, GLiNER2.5-Decide, d3-flash, and JEV at 9B and 27B. [The specification](shared-batching-protocol.json) is separate from the larger, still-proposed [generality design](generality-protocol.json). The short resident replays do not fulfill that design's independent 30-minute serving trials.

## Method

The shared optimizer lives in `src/dfip/shared.py`. A native adapter supplies a batch compatibility key, a length estimate, and released single and batch calls. One policy applies across adapters without model-name branches. It preserves ordered schemas, restores request order, bounds estimated padded tokens to 8,192, bounds length ratio to four, and never batches more than eight requests. The oldest pending request is included, and compatible requests of similar length are preferred.

Admission tests sizes two, four and eight separately. Every calibration comparison must preserve literal fields, structure, and sampled threshold classifications at 0.1, 0.25, 0.5, 0.75 and 0.9. Absolute numerical tolerance is 0.001, with 0.01 for a returned ordinal score. Per-request token usage is retained. A size also needs a median gain of at least 1.10 over serial native calls in three exploratory measurements. Partial queues cannot silently use a rejected or untested batch size. Rejected schemas and sizes use native single-request inference.

Admission is finite empirical evidence. It does not establish equivalence for every future request. The held-out replay compares all exposed answer fields against native references and retains failures after admission. GLiNER exposes the winning label and confidence, so its audit cannot establish full-distribution equivalence. JEV exposes option probabilities and its selected bucket, rather than an invented confidence or expected score. D3's confidence is entropy-derived. Task accuracy and calibration quality are separate questions; these synthetic fixtures have no ground-truth accuracy claim.

## Evaluation

Native admission runs 29 changing records, repeated calls, 20 warm timing samples in eight cells, single native batch probes at sizes two/four/eight, and operator profiles. The batch-probe ratios are diagnostic single measurements, not independent speedup estimates. Optional native CUDA kernels are installed in the tensor-model image; D3's released ROCm-specific fast path is unavailable on NVIDIA and is recorded as such. JEV uses native vLLM 0.31.0, BF16, decision LoRA, processed log probabilities, prefix caching, and native calibration. Its initial profiles observe the engine boundary; a separate worker CUDA trace is collected after the shared experiment's timing.

Shared evaluation uses 96 calibration records, then two 96-record resident traces with different state templates: recurring categorical schemas and mixed decision kinds/widths/lengths with some reordered schemas. Five warm paired timings use seeded randomized mode order in one worker per checkpoint. Both serial native and native batching are comparators. Native batching already groups compatible schemas at a ceiling of eight; a gain over serial alone must not be presented as a novel DFIP gain.

Two 256-request real-time resident replays then use scheduled arrivals. The recurring schema trace offers 25 requests/s. The varied trace alternates 32-request groups at 80 and 20 requests/s. Native batching and DFIP share a two-millisecond accumulation window; queues hold at most 128 requests with a two-second queue deadline. Latency starts at scheduled arrival and includes shared length estimation. Every request is accounted for as completed or dropped, and actual execution batches are archived. There is no HTTP transport in this phase. References warm the fixtures before replay; prefix-cache history and repeated payloads limit interpretation. Arrival windows last roughly 8--11 seconds, rather than a production soak test.

## Reproduction

Run from the repository root with Modal configured. Preparation uses CPU workers, with pinned checkpoints stored on the research Volume. GPU execution is serialized across applications, and the driver waits for worker retirement between jobs. Production applications are not changed.

```sh
PYTHONPATH=src modal run -m dfip.cloud.contracts_prepare
python scripts/run_contract_study.py --action native
python scripts/run_contract_study.py --action shared
PYTHONPATH=src python scripts/analyze_contracts.py
```

The driver expects a local `.venv/bin/modal`; create that environment from the project's declared dependencies. CPU preparation receipts are written only after the remote Volume commit. The driver waits for these before GPU allocation. Commit source changes before executing jobs: each result records its exact commit and a commit-named source ZIP, while `results/contracts/job-journal.jsonl` preserves every invocation. Use `--start-round` to retain corrected or repeated attempts under a different round rather than overwrite evidence.

The initial D3 preparation omitted non-weight files required by its package verifier, and two JEV jobs were launched before preparation completed. Those load failures remain in the namespace. The corrected D3 experiment completed on GPU but initially failed local result deserialization because a Torch version string subclass required a local Torch installation. The committed JSON was recovered from the Volume without rerunning that GPU job. A later corrected adapter preserves D3 text token accounting and is evaluated separately. These infrastructure outcomes are distinct from runtime numerical fidelity and speed.

`paper/generated/contracts-summary.json` and the generated LaTeX tables contain the numerical outcomes. The raw JSON retains calibration/test fixtures, outputs, timing samples, batches, worker identities, package versions, and failure traces. No v5 estimate is pooled with the earlier v3 or v4 experiments. Independent long-duration trials and a materially better result than native batching remain requirements for a general acceleration claim.
