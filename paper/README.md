# DFIP review draft

**Authors:** Ismail Pelaseyed and Alan Zabihi, Superagent Technologies Inc.

`dfip.pdf` is the compiled review draft. `main.tex` and `references.bib` are the editable LaTeX source. The paper has not been submitted to arXiv. Read [the benchmark notes](../benchmarks/README.md) before interpreting the results.

The original expanded protocol retained three complete Liquid rounds, three Clef rounds, and five PPLX-Decider rounds. Julia's application failed without a complete round. Security-One completed five rounds using separately restarted SGLang servers with bracketing native measurements. A single-container optimized-convolution Clef follow-up is separate; it shows two sampled probability-threshold crossings. Actual retained counts appear in the generated tables; GPU failure excerpts and source versions accompany the raw data.

The separate strengthened study adds five optimized-native Decider rounds, four complete Clef rounds plus one native-preflight failure, three released-compiler Liquid rounds, and direct answer-field audits. Its [protocol and reproduction notes](../benchmarks/strengthened-study.md) explain the external serving tests and separate result namespace. The compiler comparison does not show a general DFIP latency advantage over compiled native Liquid.

The nine sustained-serving attempts retain eight complete rounds and one Decider replay-path CUDA failure. Varied traffic reverses Decider/Clef resident replay gains; Clef also crosses literal confidence thresholds. Security-One concurrent native dispatch has no losses at 30 requests/s across all three rounds, with nearly 30 within-window completions/s and p95 0.65 seconds. All five review defaults use native inference. No existing deployment has been changed.

A separate v4 post hoc pilot tests bounded graph admission with native fallback. Its [protocol and lifecycle correction](../benchmarks/adaptive-study.md) retain the excluded Clef worker-replacement attempt. Each validated worker uses four captures, with uneven performance gains. Clef still changes returned confidence and crosses 0.9 on fourteen matched pairs. The pilot is not pooled with v3, and one validated container per model cannot establish production reliability. The opt-in API and native Security-One concurrency are source changes for review; existing deployments are unchanged.

Tables and figures are regenerated from JSON by `python scripts/analyze_paper.py` and `PYTHONPATH=src python scripts/analyze_strengthened.py`, and `PYTHONPATH=src python scripts/analyze_adaptive.py`. Build from this directory with `latexmk -pdf main.tex`, or from the repository root with:

```sh
PYTHONPATH=src modal run -m dfip.cloud.paper
```

The cloud build uses a CPU worker and does not allocate a GPU. Optional local plotting/rendering dependencies are `pip install -e '.[paper]'`.

Review priorities are the strength of the native baseline, failure attrition, probability drift, and the scope of claims. A public frozen artifact and review by both authors must precede an arXiv submission. Reading notes list the research papers and sections that informed the design.
