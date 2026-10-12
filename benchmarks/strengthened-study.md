# Optimized-native and sustained serving study

This experiment is separate from the original v1 and the one-container optional-extension follow-up. `strengthened-protocol.json` records the planned design before the serving runs. Source snapshots are in `source-v3-optimized.zip` (first Decider comparison, commit `731e3f2`) and `source-v3-serving.zip` (Clef comparison, commit `b776896`). The later direct-readout audits and serving instrumentation have their own commit-named source archive, linked from the job journal. A snapshot contains source code only; checkpoints and corpora remain on Modal.

The Qwen comparisons preserve the released SDPA/noncausal attention contract and use an image with `causal-conv1d==1.7.0` and `fla-core==0.5.2`. Installing these extensions is an optimized native baseline, not a claim to match every specialized serving engine. Serving setup records backend availability and a native CUDA-kernel profile outside timing. Liquid's compiler fidelity check contains 128 single categorical requests and does not validate every Boolean/ordinal timing schema. Its separate comparison uses the released runner's `torch.compile(mode="reduce-overhead")` setting; compiler admission cost and failures are retained.

Serving uses authenticated Modal RPC from the author workstation. Each phase schedules fixed arrivals at 5, 15 or 30 requests/s for 60 seconds. The latency clock begins at the scheduled arrival, so delayed client dispatch is included. There is a 128-request in-flight limit; arrivals beyond it count as dropped, rather than disappearing. Transformer GPU execution is serialized inside a 64-input worker, matching the mutable-buffer protection used by the current wrapper. Security-One separately compares serialized dispatch with concurrent dispatch to its native SGLang scheduler. Engine settings remain fixed (32 maximum running requests, radix cache disabled, native SGLang CUDA graphs enabled); this isolates wrapper dispatch and is not a tuned backend ceiling. Transformer dynamic batching and a tuned maximum native-serving ceiling are not established. Requests are completely drained between phases and modes.

Pre/post fidelity checks and returned distributions are archived. Analysis also matches successfully returned request IDs between modes. Missing requests are counted separately; probability comparisons cannot establish equivalence for dropped requests. Clef chooses its categorical answer before rounding probability slots, and rounds its score separately. A separate full 3,024-decision audit therefore records literal returned choice, score, and confidence fields; later compiler/serving records preserve them too. Earlier distribution-only comparisons concern reconstructed argmax and probability-derived expected scores, not hidden unrounded internals. Reported tail latencies concern successful requests and must be read alongside error/drop counts. A minute-scale soak is not an hours-long reliability certification. The changing twelve-length request family stresses a four-entry graph cache; it does not measure serving of a single recurring shape. Checkpoint and compiler caches persist on the remote volume, so setup is not an empty-cache cold-start benchmark. Full same-image package inventories are supplementary CPU-only records.

## Reproduction

Prepare the pinned model checkpoints in a Modal volume following the repository README. Do not download weights to the local client. Then run the entire study serially:

```sh
PYTHONPATH=src python scripts/run_strengthened.py --volume YOUR_MODAL_VOLUME --modal /path/to/modal
```

The driver waits for previous study workers to retire before starting the next application. Existing production applications are never stopped or reconfigured. Each job preserves its source revision, exit status and timing in `results/strengthened/job-journal.jsonl`; raw result files retain failures. A failed experimental path is not silently retried until it succeeds.

The supplementary full image package inventories can be captured without a GPU:

```sh
PYTHONPATH=src modal run scripts/capture_study_environment.py --commit SOURCE_COMMIT
```

The inventory records the capture script hash separately from the image-definition revision. It is evidence of the observed environment, not a fully resolved dependency lock or another hardware replication.

To regenerate tables after the jobs finish:

```sh
PYTHONPATH=src python scripts/analyze_strengthened.py
cd paper
latexmk -pdf main.tex
```

Read `paper/generated/strengthened-summary.json` for full per-workload results, serving accounting and matched stream fidelity. The analysis rejects inconsistent medians, altered summaries, duplicate decision IDs, and any serving round that replaces its worker container. Confidence intervals for optimized-native comparisons resample container rounds; serving tail summaries aggregate container-level quantiles rather than treating thousands of requests as independent hardware trials.
