# Research toward a general decision inference method

DFIP's objective is a reusable inference method that reduces latency and cost across open decision models while preserving their decision contracts. The existing paper is an initial characterization of a prototype. Its resident speedups and serving failures motivate this next phase; they do not establish the intended general method.

## What general means

An adapter may describe a checkpoint's released prompt encoding, attention semantics, decision head, calibration, answer conversion and supported backend. The shared optimizer must choose execution from these declared capabilities and measured workload properties. A new checkpoint must not require another model-name branch, an ad hoc kernel patch, or performance thresholds tuned on its evaluation results.

A common API and universal native fallback establish compatibility, not general acceleration. Report the fraction of checkpoints and offered requests actually accelerated, alongside latency, throughput, fidelity and cost. Failed, unsupported and fallback-only cases remain in the selected-cohort denominator. Generality is bounded by declared contracts and tested hardware; it cannot mean a speedup on every possible model.

## Initial source audit

The audit pins five additional checkpoints and reviews source without executing it or downloading weights. Immutable revisions, source hashes and evidence locations are in `research-audits/model-contracts.json`. These candidates are not yet supported by the client or GPU-validated.

| Candidate | Structural finding | Research consequence |
| --- | --- | --- |
| AutoTrust JEV 9B and 27B | Released serving uses a decision LoRA module, answer-token IDs, bias and per-kind calibration. | Preserve the full decision recipe and compare against the recommended vLLM path. Treat two sizes as one family. Open AutoTrust JEV students are distinct from hosted TypeSafe Jev. |
| vLLM Semantic Router d3 flash | The released runtime projects the last hidden state through a small answer-code readout and exposes its own optimized path. | Full-vocabulary elimination is already present. Audit backend activation on NVIDIA; do not compare an optimized DFIP path only against an artificially weakened reference. |
| Liquid d1 omni 600M | A dedicated Transformer decision head scores option markers; no full-vocabulary output head is used in the inspected decision forward. | Test scheduling, batching and selective execution rather than assuming candidate projection supplies a gain. |
| GLiNER2.5 Decide | The checkpoint uses a DeBERTa encoder and schema extraction architecture. The GLiNER2 paper describes label-specific classification heads and distinct single-label and multi-label normalization. | Provide a genuinely different family and head contract. Pin and inspect the actual library before asserting this checkpoint's numerical readout semantics. |

These findings reject full-vocabulary pruning as the sole general mechanism. They do not rule out a shared optimizer with several transformations selected by capability.

## Method to investigate

Separate released encoding and answer conversion from tensor execution. Describe the needed output positions, candidate mapping, head operator, normalization, calibration, padding semantics and state ownership in a decision contract. Every transformation must declare the conditions under which it preserves that contract.

Investigate token-budget batching, verified length buckets, compiler-supported partial capture and selective head operations. Padding and batching are hypotheses to validate: recurrent attention metadata, positions, masks and marker locations can change behavior. Stable buffers require explicit lifetimes and exclusive ownership or backend-managed scheduling.

Choose native, compiled, batched or captured execution using a shared cost model. Charge encoding, padding, copies, compilation, capture, queueing and memory use to the selected path. Capture only when expected reuse can repay setup cost within a bounded horizon. Evaluate changing popularity and novel shapes; a cache that becomes permanently ineffective is not a solution to varied traffic. A fatal CUDA error invalidates a worker rather than falling back within a damaged context.

PyGraph already studies compiler transformations and cost-guided graph selection. PyTorch documents partial capture and fixed-address constraints. DFIP must explain and measure an additional contribution specific to decision contracts rather than claim those existing mechanisms as novel.

## Evaluation and research gates

1. Establish native contract conformance and component profiles before custom optimization. Preserve literal choice, score, confidence, ordering, truncation and unsupported-input behavior. Set numerical tolerances and application thresholds before evaluating an optimized path. A label match alone does not pass fidelity.
2. Implement the shared optimizer on development families. Freeze its code, adapter specification, profiling budget, policy and evaluation protocol before measuring reserved families. Structural source review of a reserved model is allowed; latency-driven policy tuning on it is not.
3. Evaluate transfer on reserved DeBERTa/GLiNER and dedicated-head Liquid omni families without model-specific optimizer changes. Document integration effort and rejected transformations. Keep development and evaluation traces separate.
4. Run resident comparisons and serving load sweeps with warm, startup-inclusive, bursty, diverse-shape and popularity-shift workloads. Randomize order across fresh workers and retain every attempt. Pilot runs estimate feasible load; they cannot be pooled into the later confirmatory serving results.
5. Run three independent 30-minute serving rounds for each candidate configuration that passes the preceding gates, with one study GPU at a time. Record offered arrivals, completions, errors, drops, queueing, p50/p95/p99, capture and compilation work, memory and matched output fields. Long tests remain bounded evidence rather than a production certificate.
6. Report per-model and per-contract outcomes against a strong supported native configuration on the same hardware. Count fallback-only cases explicitly. Separate rate-based resource estimates from actual billed cost and include startup and idle costs.

The machine-readable design in `generality-protocol.json` remains a draft until adapters, native baselines, tolerances, traces, hardware and policy are implemented and frozen. The next gate is native conformance and profiling for the five new candidates. No new performance results are implied by the source audit.

## Paper direction

The intended paper should present a decision contract, its supported transformations, a workload-aware execution policy and evidence of transfer to reserved families. The current measurements remain the motivation and initial evidence. If the shared method cannot accelerate diverse contracts under serving traffic, retain that result and narrow the eventual claim; do not turn native fallback into a successful acceleration result.

## Sources

- [PyGraph, version 3](https://arxiv.org/html/2503.19779v3), Sections 2.3 and 4.3: capture constraints and cost-guided graph selection.
- [GLiNER2](https://arxiv.org/html/2507.18546v1), Appendix A: schema composition, label-position readout and normalization contracts.
- [PyTorch CUDA semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html), CUDA graph constraints and partial-network capture.
- Candidate model sources are linked at immutable revisions in the audit manifest.
