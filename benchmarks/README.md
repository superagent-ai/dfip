# DFIP experiment archive

The current [review paper](../paper/README.md) focuses on **PPLX-Decider v1.1 27B, Liquid d1-3B, Clef-Flash and GLiNER2.5-Decide**. It evaluates acceleration over stronger native baselines and retains failures, probability drift and varied-traffic losses for these models. It does not certify production readiness or establish a universal kernel.

## Regenerate the focused paper

```sh
python -m pip install -e '.[paper]'
PYTHONPATH=src python scripts/analyze_focused.py
PYTHONPATH=src modal run -m dfip.cloud.paper
```

The analyzer selects the four-model evidence from the original records and writes `paper/generated/focused-summary.json` and manuscript tables. It recomputes retained paired-container statistics and resident medians, checks request accounting, and records the selected raw paths and SHA-256 hashes. The CPU typesetter allocates no GPU.

## Evidence used in the manuscript

- [Strengthened study](strengthened-study.md): optional native Qwen kernels, repeated Decider/Clef resident comparisons, released Liquid compiler comparisons, literal answer-field audits and minute-scale external traffic. Records are in `results/strengthened/`.
- [Bounded-admission pilot](adaptive-study.md): a separate post hoc Decider/Clef mitigation with native fallback. It preserves the excluded Clef worker-replacement attempt and its corrected round. Records are in `results/adaptive/`.
- [Shared batching pilot](shared-batching-study.md): native GLiNER conformance/profiling, comparisons with released native batching, and short scheduled-arrival resident replays. The [released compiler follow-up](shared-compiled-followup-protocol.json) is post hoc and reuses the same fixtures. Records are in `results/contracts/`.

These phases have different workloads, statistical units and limitations. They are not pooled into a single speedup. A native fallback is not an acceleration result. Winning-label agreement does not establish confidence-threshold equivalence.

## Reproduction on Modal

Use a separate research app and Volume, prepare the pinned checkpoints, and run **one GPU application at a time**. The exact entry points, source revisions and command options for each phase are in its linked notes. Retain failed and excluded rounds rather than replacing them silently. Commit source changes before new experiments so records identify a reviewable source snapshot. Source ZIPs, job journals, environment manifests and compressed logs accompany the records.

Model weights, downloaded datasets and GPU dependencies remain on Modal. Resident timings exclude external transport and cold start. The external Decider/Clef tests include scheduled-arrival delay, dispatch, network and queueing. GLiNER's short resident replays include neither HTTP transport nor a long-duration soak. Successful-request latency must be read alongside drops and request errors.

## Historical evidence outside the focused manuscript

The repository preserves earlier model studies, source audits and raw failures for provenance. Those models and their results are excluded from the current manuscript's inputs and claims. Existing deployed applications and public client compatibility are not changed by the paper's narrower scope.

The original `protocol.json`, `source-v1.zip`, `results/` and `failures/` describe the early fallback-image study. Separate follow-up records retain lifetime diagnostics and a single optimized-convolution trial. The broader `generality-protocol.json` and [general method design](general-method-study.md) are proposed longer-term research, not completed independent confirmation. Original all-model analyzers remain available to reconstruct their historical tables; `analyze_focused.py` is the current manuscript entry point.
