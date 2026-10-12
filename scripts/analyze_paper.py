"""Regenerate every numerical paper table and plot from retained raw samples."""

import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"julia": "Julia-1", "liquid": "d1-3B", "clef": "Clef-Flash",
          "decider": "PPLX-Decider", "security": "Security-One"}
SCENARIOS = ["short_binary", "short_choice", "short_3_questions", "medium_context",
             "long_context", "2_candidates", "8_candidates", "16_candidates", "batch_8"]


def geometric_mean(values):
    return math.exp(statistics.mean(math.log(x) for x in values))


def interval(ratios, seed=20261012):
    """Paired cluster bootstrap: each draw is an entire container round."""
    if not ratios or any(x <= 0 or not math.isfinite(x) for x in ratios):
        raise ValueError("Speed ratios must be finite and positive")
    rng = random.Random(seed)
    draws = sorted(geometric_mean(rng.choices(ratios, k=len(ratios))) for _ in range(10000))
    return geometric_mean(ratios), draws[249], draws[9749]


def timings(result, mode):
    values = {}
    for row in result["modes"][mode]["timings"]:
        raw = row["samples_ms"]
        if len(raw) != 50 or any(x <= 0 or not math.isfinite(x) for x in raw):
            raise ValueError("Unexpected timing sample count or value")
        median = statistics.median(raw)
        if not math.isclose(median, row["p50_ms"], rel_tol=1e-10):
            raise ValueError("Reported median differs from raw samples")
        values[row["scenario"]] = median
    return values


def collect():
    groups = {}
    for path in sorted((ROOT / "benchmarks/results").glob("*-round-*.json")):
        # Optional extension builds are a separate exploratory baseline.
        if "-fast-" in path.name:
            continue
        result = json.loads(path.read_text())
        model = path.name.split("-round-")[0]
        groups.setdefault(model, []).append(result)
    return groups


def analysis(groups):
    rows = []
    for model, results in groups.items():
        selected = "triton" if model == "security" else "production"
        for scenario in SCENARIOS:
            base, optimized, ratios = [], [], []
            for result in results:
                opt = timings(result, selected)[scenario]
                if model == "security":
                    native = geometric_mean([timings(result, m)[scenario]
                                             for m in ("native_before", "native_after")])
                else:
                    native = timings(result, "native")[scenario]
                base.append(native)
                optimized.append(opt)
                ratios.append(native / opt)
            estimate, lower, upper = interval(ratios)
            rate = 0.001736 if model in ("decider", "security") else 0.000542
            decisions = next(t["decisions_per_call"] for t in results[0]["modes"][selected]["timings"]
                             if t["scenario"] == scenario)
            rows.append({"model": model, "scenario": scenario, "rounds": len(results),
                         "native_p50_ms_geomean": geometric_mean(base),
                         "optimized_p50_ms_geomean": geometric_mean(optimized),
                         "speedup": estimate, "bootstrap_lower": lower, "bootstrap_upper": upper,
                         "gpu_seconds_cost_usd_per_1000_decisions":
                         geometric_mean(optimized) * rate / decisions,
                         "policy": selected, "round_speedups": ratios})
    return rows


def main():
    groups = collect()
    rows = analysis(groups)
    generated = ROOT / "paper/generated"
    generated.mkdir(parents=True, exist_ok=True)
    summary = {"statistics": "geometric mean of per-container medians; paired 95% percentile bootstrap",
               "attrition": "Liquid/Clef v1 contain three completed rounds of five planned; conditional intervals",
               "prices_checked": "2026-10-12", "rows": rows}
    (generated / "summary.json").write_text(json.dumps(summary, indent=2))
    with (ROOT / "benchmarks/results/summary.csv").open("w", newline="") as stream:
        fields = [k for k in rows[0] if k != "round_speedups"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    table = []
    for model in LABELS:
        found = next((r for r in rows if r["model"] == model and r["scenario"] == "short_binary"), None)
        if found:
            r = found
            table.append(f"{LABELS[model]} & {r['rounds']} & {r['native_p50_ms_geomean']:.2f} & "
                         f"{r['optimized_p50_ms_geomean']:.2f} & {r['speedup']:.2f} "
                         f"[{r['bootstrap_lower']:.2f}, {r['bootstrap_upper']:.2f}] \\\\")
        else:
            table.append(f"{LABELS[model]} & 0 & -- & -- & GPU error \\\\")
    (generated / "latency_rows.tex").write_text("\n".join(table) + "\n")
    table = []
    for model, results in groups.items():
        result = next((r for r in results if r["round"] == 0), None)
        if not result:
            continue
        mode = "triton" if model == "security" else "production"
        report = result["modes"][mode]
        a, b = report["public_fidelity"], report["new_workload_fidelity"]
        crossings = sum(a["threshold_crossings"].values()) + sum(b["threshold_crossings"].values())
        table.append(f"{LABELS[model]} & {a['decisions'] + b['decisions']} & "
                     f"{a['argmax_flips'] + b['argmax_flips']} & {crossings} & "
                     f"{max(a['max_probability_drift'], b['max_probability_drift']):.3g} \\\\")
    (generated / "fidelity_rows.tex").write_text("\n".join(table) + "\n")
    ablations = []
    liquid = groups.get("liquid", [])
    for mode in ("native", "graphs_only", "candidate_cublas", "candidate_triton", "production_cublas", "production"):
        if liquid:
            values = [timings(r, mode)["short_binary"] for r in liquid]
            ratio = [timings(r, "native")["short_binary"] / v for r, v in zip(liquid, values)]
            estimate, lo, hi = interval(ratio)
            label = mode.replace("_", "\\_")
            ablations.append(f"\\texttt{{{label}}} & {geometric_mean(values):.2f} & "
                             f"{estimate:.2f} [{lo:.2f}, {hi:.2f}] \\\\")
    (generated / "ablation_rows.tex").write_text("\n".join(ablations) + "\n")
    manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted((ROOT / "benchmarks/results").glob("*-round-*.json"))}
    (ROOT / "benchmarks/results/sha256.json").write_text(json.dumps(manifest, indent=2))
    headers = {
        "latency_rows.tex": ("lrrrr", "Checkpoint & Complete rounds & Native median (ms) & Tested median (ms) & Paired speedup [95\\% CI]"),
        "fidelity_rows.tex": ("lrrrr", "Checkpoint & Decisions & Flips & Crossings & Max $|\\Delta p|$"),
        "ablation_rows.tex": ("lrr", "d1-3B mode & Median (ms) & Speedup [95\\% CI]"),
    }
    for name, (columns, header) in headers.items():
        path = generated / name
        body = path.read_text()
        path.write_text("\\begin{tabular}{" + columns + "}\n\\toprule\n" + header + " " + chr(92)*2 + "\n\\midrule\n" + body + "\\bottomrule\n\\end{tabular}\n")
    plots(rows, groups)
    print(json.dumps({"raw_rounds": {m: len(r) for m, r in groups.items()}, "summary_rows": len(rows)}))


def plots(rows, groups):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    models = [m for m in LABELS if m in groups]
    values = np.array([[next(r["speedup"] for r in rows if r["model"] == m and r["scenario"] == s)
                        for s in SCENARIOS] for m in models])
    fig, ax = plt.subplots(figsize=(7.4, 2.9), layout="constrained")
    ax.imshow(np.log2(values), cmap="RdYlGn", vmin=-1, vmax=2)
    for i in range(len(models)):
        for j in range(len(SCENARIOS)):
            ax.text(j, i, f"{values[i,j]:.2f}x", ha="center", va="center", fontsize=8)
    ax.set_yticks(range(len(models)), [LABELS[m] for m in models])
    ax.set_xticks(range(len(SCENARIOS)), ["Binary", "Choice", "3 questions", "Medium", "Long", "2 options", "8 options", "16 options", "Batch 8"], rotation=32, ha="right")
    ax.set_title("Paired speedup of container medians (native / tested policy)", fontsize=10)
    fig.savefig(ROOT / "paper/generated/workloads.png", dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    main()
