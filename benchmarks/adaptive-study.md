# Bounded admission pilot (v4)

This post hoc experiment responds to v3's varied-shape recapture losses. It is separate from the frozen native-baseline and serving study. The admission thresholds were recorded in `adaptive-protocol.json` before the first pilot. They were not selected by optimizing these results.

One new container per model was planned for Decider/B200 and Clef/L40S, with the optimized-native image used in v3. Native and adaptive run in randomized mode order within a worker. The original diverse trace precedes a synthetic mixture: 60% recurring short request family with changing ticket content and 40% original diversity. The short family fixes the state-repetition count and two-candidate schema; tokenized ticket numbers can still change exact sequence length, so it is not guaranteed to share one graph key. Both profiles use 5/15/30 fixed arrivals/s for 60 seconds each. The 128-client-in-flight limit, 64 worker inputs, serialized GPU access, scheduled-arrival latency and full phase drain match v3. Admission persists across rates and profiles in a mode and resets between modes. Report each profile separately: the hot-mixture results include prior diverse-traffic cache history.

The adaptive resident set uses a 128-call recent window, at least 32 observations, eight hits and 12.5% frequency, with four lifetime capture attempts and no eviction. Native setup and fidelity checks precede enabling admission. Final fidelity checks freeze admission but can replay existing captures. Final capture/replay counters include those checks and each capture's immediate first replay; they are not pure traffic-hit counts. No previous prediction is cached.

## Validated outcomes

Decider round 0 and corrected Clef round 1 complete all twelve phases without request errors, with four captures each. Decider records 474 replays and Clef 400, including final checks and immediate capture replays. Most requests still use native fallback. Neither pilot establishes uniform performance gains or production reliability.

At 5 requests/s, both modes have no loss. Decider's adaptive/native p95 is 0.48/0.32 s on diverse traffic and 0.35/0.48 s on mixed traffic. Clef's is 0.42/0.36 s and 0.35/0.34 s. At 15 requests/s, both Clef modes complete 14.93 requests/s within the window with no loss. At 30 requests/s, diverse Decider improves from 7.53 to 9.62 completions/s, but still loses 60.8% of arrivals. Clef's adaptive/native completions are 15.88/15.95 on diverse traffic and 16.55/16.25 on mixed traffic, with about 38–40% drops.

Decider preserves literal choices and probabilities across 2,057 matched successful categorical pairs. Clef preserves choices across 3,844 matched pairs, but maximum probability/returned-confidence drift is 0.0065 and fourteen mixed-trace pairs cross the returned-confidence threshold 0.9. This policy does not solve numerical confidence sensitivity. No ordinal scores are compared in these categorical pilots. Full per-rate accounting, unmatched counts and actual field comparisons are in the generated summary. There are no confidence intervals because each model has one validated container.

## Retained lifecycle failure

Clef round 0 returned client status `complete`, but its worker identity changed during adaptive traffic and the replacement finished in native mode with no admitted graphs. The whole comparative trial is excluded. Raw results and logs remain intact. The cause of replacement is unknown. `adaptive-correction.json`, recorded before the separate round 1, adds actual worker-ID/mode checks to every response and the final audit, and raises the idle window from 10 to 300 seconds as a precaution. This is not proof that idle scaling caused the replacement. The corrected run has a distinct source archive and round ID.

Only validated trials contribute comparative serving summaries. Matched fidelity uses successful IDs returned in both modes, reports unmatched requests, and cannot establish equivalence for dropped requests. A single validated container per model gives no confidence interval or long-run reliability claim. Synthetic popularity, fixed profile order, overload, mode history and cloud variability limit generalization.

## Reproduce

Use a fresh checkout of the source commit recorded in the relevant raw result or the matching source ZIP, to avoid overwriting archived results. Prepare checkpoints in your own remote volume. Keep weights and corpora remote. The first pilot source is `8c9fc08a9b61fd6876ab14acf79e9421048050b0`; the corrected Clef source is `bc1bfa16e272f2d04a858c364c99110213b8bc44`.

```sh
# Original two-model pilot, sequential workers and one study GPU at a time:
PYTHONPATH=src python scripts/run_adaptive_pilot.py --volume YOUR_MODAL_VOLUME

# Corrected Clef trial from its source revision, after prior study workers retire:
DFIP_APP=my-dfip-study4-clef DFIP_VOLUME=YOUR_MODAL_VOLUME PYTHONPATH=src \
  modal run -m dfip.cloud.adaptive --model clef --round-id 1 --commit SOURCE_COMMIT

# From the review branch, validate archived outcomes and generate separate tables:
PYTHONPATH=src python scripts/analyze_adaptive.py
```

The command's exit code alone does not establish success: inspect the raw status, per-response worker IDs/modes, final audit, capture budget, request accounting and analysis exclusions. Source manifests verify every archived source file against its recorded Git commit. Job journals record execution dates, source archives and captured CLI logs. Compressed CLI logs and their checksums retain the worker-replacement attempt; CLI delivery may omit final remote log lines. `paper/generated/adaptive-summary.json` is the complete validated summary, including excluded attempts. The deployment API in [adaptive-serving.md](../docs/adaptive-serving.md) is an opt-in integration of the policy, not a production certification from the pilot.
