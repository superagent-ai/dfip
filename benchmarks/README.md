# DFIP review-paper experiments

The paper is a review draft. Main results compare experimental paths with released native baselines and include minute-scale external traffic tests. They do not certify production reliability or compare every optimized serving backend.

`protocol.json` specifies five planned single-use container rounds, four admission calls, and 50 warm samples per workload/mode. Complete raw JSON records are in `results/`; GPU failure excerpts and attrition are in `failures/`. Do not silently replace failed rounds or combine the separate optional-extension and lifetime-fix experiments with the original protocol.

The original implementation is preserved in `source-v1.zip`. It combines base commit `822f6c3` with the two Security-One research files that had not been committed when those jobs started. Its manifest records reconstructed source hashes; the source revision field in raw JSON is the base commit, not a claim that those two files existed in that commit. Cloud scheduler limits were reduced from two to one during execution. This changes worker scheduling, not the computation of an individual round.

## Regenerate paper results without GPUs

```sh
python -m pip install -e '.[paper]'
python scripts/analyze_paper.py
cd paper
latexmk -pdf main.tex
```

Analysis rejects invalid samples and inconsistent reported medians. The checksum manifest covers raw round files. The geometric mean of each round's median is the point estimate; paired round ratios define the bootstrap. Liquid/Clef intervals are conditional on three completed rounds of five planned and do not correct for attrition.

## Re-run on Modal

Create your own Modal workspace, authenticate the CLI, and prepare the checkpoints using the repository's setup instructions. Weights and corpora stay on the remote volume. Set `DFIP_VOLUME` to your volume name. Run one application at a time, then let it finish before starting the next:

```sh
PYTHONPATH=src modal run -m dfip.cloud.research --model liquid --rounds 5 --repetitions 50 --commit YOUR_SOURCE_COMMIT
PYTHONPATH=src modal run -m dfip.cloud.research --model clef --rounds 5 --repetitions 50 --commit YOUR_SOURCE_COMMIT
PYTHONPATH=src modal run -m dfip.cloud.research --model decider --rounds 5 --repetitions 50 --commit YOUR_SOURCE_COMMIT
PYTHONPATH=src modal run -m dfip.cloud.research --model julia --rounds 5 --repetitions 50 --commit YOUR_SOURCE_COMMIT
PYTHONPATH=src modal run -m dfip.cloud.security_research --rounds 5 --repetitions 50 --commit YOUR_SOURCE_COMMIT
```

Current research workers have `max_containers=1`. The original archive uses the pre-alert cap of two; change it to one for low-concurrency replay. Its graph lifetime behavior matches the first results, while the current branch contains an explicit synchronization change. Preserve this source difference when reporting a reproduction.

The optional native convolution builder is CPU-only for `--action probe`. A subsequent GPU benchmark must use a separate output folder, such as `--output benchmarks/results/fast-conv`. Its initial failure and any resulting experiment are reported separately.

## Interpretation

- Resident API timing excludes remote client network and startup. Security-One includes in-container HTTP and native scheduling.
- BANKING77 produces correlated views of 256 utterances; ordinal labels are diagnostic, not validated annotations.
- Probability drift and sampled threshold crossings are distinct from winning-label agreement.
- GPU-seconds cost is a rate-times-latency estimate, not invoice cost. Startup, admission, idle time, CPU, memory, storage, and networking remain relevant.
- The initial Qwen reference image lacks the optional native `causal-conv1d` extension. Do not claim a gain over an optimized-kernel ceiling from that baseline.

## Separate follow-up outcomes

The synthetic 60-call graph eviction/clear check passed. Julia still failed on the derived fidelity workload with synchronized retirement. A single Clef container with `causal-conv1d==1.7.0` completed: short Boolean native/graph medians were 60.31/37.10 ms, but two probability slots crossed sampled thresholds and maximum new-workload probability drift reached 0.0129. This n=1 result is not pooled with the v1 confidence intervals. `scripts/run_followup.py` reproduces these diagnostics serially.

The review branch defaults all five models to native inference pending broader validation; Security-One uses native SGLang. Sustained changing-request tests expose replay losses and one Decider CUDA failure. Existing deployed apps were not changed.

## Strengthened optimized-native and serving study

See [strengthened-study.md](strengthened-study.md) and [strengthened-protocol.json](strengthened-protocol.json) for the separate repeated optional-kernel comparisons, released Liquid compiler baseline, and external open-loop sustained serving protocol. Results live in `results/strengthened/`; never pool them with v1 or the first exploratory follow-up.
