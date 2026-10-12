# DFIP review draft

**Authors:** Ismail Pelaseyed and Alan Zabihi, Superagent Technologies Inc.

`dfip.pdf` is the compiled review draft. `main.tex` and `references.bib` are the editable LaTeX source. The paper has not been submitted to arXiv. Read [the benchmark notes](../benchmarks/README.md) before interpreting the results.

The original expanded protocol retained three complete Liquid rounds, three Clef rounds, and five PPLX-Decider rounds. Julia's application failed without a complete round. Security-One completed five rounds using separately restarted SGLang servers with bracketing native measurements. A single-container optimized-convolution Clef follow-up is separate; it shows two sampled probability-threshold crossings. Actual retained counts appear in the generated tables; GPU failure excerpts and source versions accompany the raw data.

Tables and the plot are regenerated from JSON by `python scripts/analyze_paper.py`. Build from this directory with `latexmk -pdf main.tex`, or from the repository root with:

```sh
PYTHONPATH=src modal run -m dfip.cloud.paper
```

The cloud build uses a CPU worker and does not allocate a GPU. Optional local plotting/rendering dependencies are `pip install -e '.[paper]'`.

Review priorities are the strength of the native baseline, failure attrition, probability drift, and the scope of claims. A public frozen artifact, author approval, and any stronger serving or optimized-baseline evidence must precede an arXiv submission. Reading notes list the research papers and sections that informed the design.
