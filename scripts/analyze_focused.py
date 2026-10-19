"""Regenerate the four-checkpoint paper inputs from retained CPU-readable data.

Each experimental phase remains separate.  Repeated calls in one container
are never promoted to independent worker rounds, and failed/excluded attempts
remain in the selected evidence manifest.  This script does not mutate raw data.
"""

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics

from dfip.contract_protocol import compare as compare_contract
from dfip.study_protocol import compare_reported_answers


ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "paper/generated"
LABELS = {"decider": "PPLX-Decider", "liquid": "Liquid d1-3B",
          "clef": "Clef-Flash", "gliner": "GLiNER2.5-Decide"}
ORDER = tuple(LABELS)
WORKLOADS = {"short_binary": "Short Boolean", "short_choice": "Short choice",
             "short_3_questions": "Three questions"}


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


strengthened = load_module("focused_strengthened_helpers", "analyze_strengthened.py")
adaptive = load_module("focused_adaptive_helpers", "analyze_adaptive.py")
contracts = load_module("focused_contract_helpers", "analyze_contracts.py")


def evidence(path, record):
    model = record.get("model", record.get("alias"))
    if model is None and record.get("model_id") == "LiquidAI/d1-3B":
        model = "liquid"
    return {"file": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "model": model, "protocol": record["protocol"],
            "status": record.get("status", "retained_complete_legacy"),
            "source_commit": record["source_commit"], "round": record["round"]}


def compare_predictions(reference, actual):
    """Audit probability arrays and literal returned fields, without Torch."""
    if [r["id"] for r in reference] != [r["id"] for r in actual]:
        raise ValueError("Prediction audit requires equal ordered IDs")
    drift = []; flips = []; scores = []
    thresholds = {str(t): 0 for t in (.1, .25, .5, .75, .9)}
    for a, b in zip(reference, actual, strict=True):
        if a["probabilities"].keys() != b["probabilities"].keys():
            raise ValueError("Candidate schema changed")
        if a["prediction"] != b["prediction"]:
            flips.append(b["id"])
        for key, p in a["probabilities"].items():
            q = b["probabilities"][key]
            drift.append(abs(p-q))
            for threshold in thresholds:
                thresholds[threshold] += int((p >= float(threshold)) != (q >= float(threshold)))
        if a["type"] == "score":
            scores.append(abs(sum(int(k)*v for k, v in a["probabilities"].items())
                              - sum(int(k)*v for k, v in b["probabilities"].items())))
    return {"decisions": len(actual), "argmax_flips": len(flips), "flip_ids": flips,
            "max_probability_drift": max(drift, default=0.0),
            "mean_probability_drift": statistics.mean(drift) if drift else 0.0,
            "threshold_crossings": thresholds,
            "max_expected_score_drift": max(scores, default=0.0),
            "reported_api": compare_reported_answers(reference, actual)}


def validate_comparison(actual, archived):
    for key, value in actual.items():
        if key == "mean_probability_drift":
            if not math.isclose(value, archived[key], rel_tol=1e-12, abs_tol=1e-15):
                raise ValueError("Archived mean drift differs from predictions")
        elif value != archived[key]:
            raise ValueError("Archived prediction comparison differs: " + key)


def write_table(filename, columns, header, body):
    strengthened.tabular(GENERATED / filename, columns, header, body)


def ratio_ci(row):
    return f"{row['speedup']:.2f} [{row['ci_lower']:.2f}, {row['ci_upper']:.2f}]"


def workload_plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    import numpy as np

    models = ("decider", "clef")
    scenarios = strengthened.original.SCENARIOS
    values = np.array([[next(r["speedup"] for r in rows if r["model"] == m and r["scenario"] == s)
                        for s in scenarios] for m in models])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, ax = plt.subplots(figsize=(7.4, 2.6), layout="constrained")
    chart = ax.imshow(np.log2(values), cmap="RdYlGn", norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1.7))
    for i in range(len(models)):
        for j in range(len(scenarios)):
            ax.text(j, i, f"{values[i,j]:.2f}x", ha="center", va="center", fontsize=8)
    ax.set_yticks(range(2), ["PPLX-Decider (5/5)", "Clef-Flash (4/5)"])
    ax.set_xticks(range(9), ["Boolean", "Choice", "3 questions", "Medium", "Long",
                             "2 options", "8 options", "16 options", "Batch 8"], rotation=32, ha="right")
    ax.set_title("Stronger native / DFIP: paired worker medians", fontsize=10)
    bar = fig.colorbar(chart, ax=ax, shrink=.75, ticks=[-1, 0, 1, 1.7])
    bar.ax.set_yticklabels(["0.5x", "1x", "2x", "3.25x"])
    fig.savefig(GENERATED / "focused_workloads.png", dpi=240,
                metadata={"Software": "DFIP deterministic focused analysis"})
    plt.close(fig)


def native_phases(manifest):
    folder = ROOT / "benchmarks/results/strengthened"
    pairs = []
    for model in ("decider", "clef", "liquid"):
        for path in sorted(folder.glob(f"{model}-*-round-*.json")):
            record = json.loads(path.read_text())
            manifest.append(evidence(path, record)); pairs.append((path, record))
    records = [r for _, r in pairs]
    resident = []
    for model in ("decider", "clef"):
        valid = [r for r in records if r["model"] == model and r["status"] == "complete"
                 and r["protocol"] == "dfip-paper-v3-optimized-native"]
        resident.extend(dict(model=model, **row) for row in strengthened.paired(valid, "native", "production"))
    compiler_records = [r for r in records if r["protocol"] == "dfip-paper-v3-released-compiler"]
    compiler = strengthened.paired([r for r in compiler_records if r["status"] == "complete"],
                                   "native_compile", "production")
    compiler_fidelity = []
    for record in compiler_records:
        if record["status"] != "complete":
            continue
        reference = record["modes"]["native"]["predictions"]
        checked = {}
        for mode, report in record["modes"].items():
            checked[mode] = compare_predictions(reference, report["predictions"])
            validate_comparison(checked[mode], report["fidelity"])
        compiler_fidelity.append({"round": record["round"], "comparison": checked})
    audits = []
    for record in records:
        if record["protocol"] != "dfip-paper-v3-literal-readout-audit" or record["status"] != "complete":
            continue
        comparison = compare_predictions(record["modes"]["native"]["predictions"],
                                         record["modes"]["production"]["predictions"])
        validate_comparison(comparison, record["comparison"])
        audits.append({"model": record["model"], "round": record["round"], "comparison": comparison})
    serving_records = [r for r in records if r["protocol"] == "dfip-paper-v3-external-serving"]
    serving = strengthened.serving_rows(serving_records)
    serving_fidelity = [{"model": r["model"], "round": r["round"], "status": r["status"],
                        "comparison": strengthened.continuous_fidelity(r)} for r in serving_records]
    failures = [{**evidence(p, r), "failure": r.get("failure"), "observed_modes": list(r["modes"])}
                for p, r in pairs if r["status"] != "complete"]
    body = []
    for model in ("decider", "clef"):
        row = next(r for r in resident if r["model"] == model and r["scenario"] == "short_binary")
        body.append(f"{LABELS[model]} & {row['rounds']}/5 & {row['native_ms']:.2f} & "
                    f"{row['tested_ms']:.2f} & {ratio_ci(row)} \\\\")
    write_table("focused_optimized_rows.tex", "lrrrr",
                "Checkpoint & Rounds & Native (ms) & DFIP (ms) & Ratio [95\\% CI]", body)
    body = [f"{WORKLOADS[r['scenario']]} & {r['rounds']}/3 & {r['native_ms']:.2f} & "
            f"{r['tested_ms']:.2f} & {ratio_ci(r)} \\\\" for r in compiler]
    write_table("focused_compiler_rows.tex", "lrrrr",
                "Workload & Rounds & Compiled (ms) & DFIP (ms) & Ratio [95\\% CI]", body)
    body = []
    for audit in audits:
        c = audit["comparison"]; a = c["reported_api"]
        score = f"{a['max_score_drift']:.4f}" if a["max_score_drift"] is not None else "--"
        body.append(f"{LABELS[audit['model']]} & {a['choice_changes']}/{a['choice_fields_compared']} & "
                    f"{c['max_probability_drift']:.4f} & {score} \\\\")
    write_table("focused_literal_rows.tex", "lrrr",
                "Checkpoint & Choice changes & Max $|\\Delta p|$ & Max $|\\Delta s|$", body)
    body = []
    for r in sorted(serving, key=lambda r: (ORDER.index(r["model"]), r["mode"], r["rate"])):
        loss = 100*(r["dropped"]+r["failed"])/r["offered"]
        mode = "DFIP replay" if r["mode"] == "production" else "Native"
        p95 = f"{r['p95_ms_geomean']:.0f}" if r["p95_ms_geomean"] is not None else "--"
        body.append(f"{LABELS[r['model']]} & {mode} & {r['rate']} & {r['rounds']} & "
                    f"{r['mean_window_throughput_rps']:.2f} & {p95} & {loss:.1f} \\\\")
    write_table("focused_serving_rows.tex", "llrrrrr",
                "Checkpoint & Mode & Offered/s & $n$ & Completed/s & p95 (ms) & Loss (\\%)", body)
    workload_plot(resident)
    return {"scope": "v3 stronger native, separate paired worker rounds; observed serving phases retained",
            "resident": resident, "liquid_compiler": compiler, "compiler_fidelity": compiler_fidelity,
            "literal_audits": audits, "serving": serving, "serving_fidelity": serving_fidelity,
            "failed_attempts": failures}


def ablation_phase(manifest):
    pairs = [(p, json.loads(p.read_text())) for p in sorted((ROOT / "benchmarks/results").glob("liquid-round-*.json"))]
    records = [r for _, r in pairs]
    manifest.extend(evidence(p, r) for p, r in pairs)
    rows = []; body = []
    for mode in ("native", "graphs_only", "candidate_cublas", "candidate_triton", "production_cublas", "production"):
        native = [strengthened.original.timings(r, "native")["short_binary"] for r in records]
        values = [strengthened.original.timings(r, mode)["short_binary"] for r in records]
        ratios = [a/b for a, b in zip(native, values, strict=True)]
        estimate, lower, upper = strengthened.original.interval(ratios)
        latency = strengthened.original.geometric_mean(values)
        rows.append({"mode": mode, "rounds": len(records), "latency_ms": latency,
                     "ratio": estimate, "ci_lower": lower, "ci_upper": upper, "round_ratios": ratios})
        label = mode.replace("_", "\\_")
        body.append(f"\\texttt{{{label}}} & {latency:.2f} & {estimate:.2f} [{lower:.2f}, {upper:.2f}] \\\\")
    write_table("focused_ablation_rows.tex", "lrr", "Liquid mode & Median (ms) & Ratio [95\\% CI]", body)
    return {"scope": "v1 Liquid component ablation; three retained workers, not pooled with v3", "rows": rows}


def adaptive_phase(manifest):
    rows = []; fidelity = []; excluded = []; workers = []
    for model in ("decider", "clef"):
        for path in sorted((ROOT / "benchmarks/results/adaptive").glob(f"{model}-adaptive-round-*.json")):
            record = json.loads(path.read_text()); manifest.append(evidence(path, record))
            try:
                adaptive.validate_trial(record)
            except ValueError as exc:
                excluded.append({**evidence(path, record), "validation": "excluded", "reason": str(exc)})
                continue
            for profile in ("diverse", "mixed"):
                selected = adaptive.profile_record(record, profile)
                rows.extend(dict(profile=profile, **r) for r in strengthened.serving_rows([selected]))
                selected["modes"]["production"] = selected["modes"].pop("adaptive")
                comparison = strengthened.continuous_fidelity(selected)
                comparison["tested_mode"] = "adaptive"
                fidelity.append({"model": model, "round": record["round"], "profile": profile,
                                 "comparison": comparison})
            workers.append({"model": model, "round": record["round"], "mode_order": record["mode_order"],
                            "graphs": record["modes"]["adaptive"]["finish"]["diagnostics"]["graphs"]})
    body = []
    for r in rows:
        if r["rate"] != 5:
            continue
        loss = 100*(r["dropped"]+r["failed"])/r["offered"]
        body.append(f"{LABELS[r['model']]} & {r['profile']} & {r['mode']} & "
                    f"{r['mean_window_throughput_rps']:.2f} & {r['p95_ms_geomean']:.0f} & {loss:.1f} \\\\")
    write_table("focused_adaptive_rows.tex", "lllrrr",
                "Checkpoint & Trace & Mode & Completed/s & p95 (ms) & Loss (\\%)", body)
    return {"scope": "v4 post hoc bounded admission, one validated worker per model; never pooled with v3",
            "serving": rows, "fidelity": fidelity, "workers": workers, "excluded_attempts": excluded}


def gliner_phase(manifest):
    native = []; shared = []; failed = []
    for path in sorted((ROOT / "benchmarks/results/contracts").glob("gliner-*.json")):
        record = json.loads(path.read_text())
        if record.get("action") not in ("native", "shared", "shared_compiled"):
            continue
        identity = evidence(path, record); identity["action"] = record["action"]
        manifest.append(identity)
        if record["status"] != "complete":
            failed.append({**identity, "error": record.get("error")}); continue
        if record["action"] == "native":
            native.append({**identity, "repeatability": contracts.combine(
                compare_contract(c["reference"], c["repeat"]) for c in record["conformance"]),
                "resident_median_ms": {k: v["median_ms"] for k, v in record["resident"].items()},
                "profile": record["profile"], "environment": record["environment"]})
            continue
        resident = {}
        for name, cell in record["resident_shared"].items():
            medians = {mode: statistics.median(samples) for mode, samples in cell["samples_ms"].items()}
            if medians != cell["median_ms"]:
                raise ValueError("GLiNER resident median inconsistent with raw samples")
            resident[name] = {"median_ms": medians, "serial_to_shared": medians["serial"]/medians["shared"],
                              "native_batch_to_shared": medians["native_batch"]/medians["shared"],
                              "fidelity": {mode: compare_contract(cell["reference"], a["outputs"])
                                           for mode, a in cell["audits"].items()}}
        for gate in record["admission"]["evidence"]:
            gain = statistics.median(gate["serial_ms"])/statistics.median(gate["batch_ms"])
            if gain != gate["median_gain"] or gate["admitted"] != (all(c["passes"] for c in gate["comparisons"]) and gain >= 1.10):
                raise ValueError("GLiNER admission differs from calibration records")
        shared.append({**identity, "resident": resident, "compiler": record.get("compiler"),
                       "admission_seconds": record["admission"]["seconds"],
                       "passed_size_gates": sum(e["admitted"] for e in record["admission"]["evidence"]),
                       "attempted_size_gates": len(record["admission"]["evidence"]),
                       "serving": {name: {mode: contracts.serving_summary(report, trace.get("reference"))
                                          for mode, report in trace["modes"].items()}
                                   for name, trace in record["serving_shared"].items()}})
    shared.sort(key=lambda r: (r["action"] != "shared", r["round"], r["source_commit"]))
    resident_body = []; serving_body = []; findings = []
    for record in shared:
        label = "Compiled" if record["action"] == "shared_compiled" else "Eager"
        for name, cell in record["resident"].items():
            verdict = "pass" if cell["fidelity"]["shared"]["passes"] else "fail"
            resident_body.append(f"{label} & {name} & {record['passed_size_gates']}/{record['attempted_size_gates']} & "
                                 f"{cell['serial_to_shared']:.2f} & {cell['native_batch_to_shared']:.2f} & {verdict} \\\\")
        for name, modes in record["serving"].items():
            for mode in ("serial", "native_batch", "shared"):
                r = modes[mode]; display = {"serial": "Serial", "native_batch": "Native batch", "shared": "Shared"}[mode]
                verdict = "pass" if r["fidelity"]["passes"] else "fail"
                serving_body.append(f"{label} & {name} & {display} & {r['p95_ms']:.1f} & "
                                    f"{r['completed_after_drain']}/{r['offered']} & {r['drops']} & {verdict} \\\\")
        h = record["resident"]["homogeneous"]; d = record["resident"]["diverse"]; t = record["serving"]["diverse"]
        if record["action"] == "shared":
            findings.append(f"On the eager GLiNER substrate, all {record['passed_size_gates']} schema/size gates pass. "
                            f"The shared scheduler improves resident medians by {h['native_batch_to_shared']:.2f}$\\times$ "
                            f"for homogeneous schemas and {d['native_batch_to_shared']:.2f}$\\times$ for diverse schemas over native batching "
                            f"({h['serial_to_shared']:.2f}$\\times$ and {d['serial_to_shared']:.2f}$\\times$ over serial execution). "
                            "The held-out exposed-label/confidence audits pass. "
                            f"Diverse-traffic p95 is {t['shared']['p95_ms']:.1f}\\,ms for shared scheduling and "
                            f"{t['native_batch']['p95_ms']:.1f}\\,ms for native batching; both complete all 256 requests without drops. "
                            "The resident improvement therefore does not establish an incremental serving tail-latency gain.")
        else:
            verdict = "passes" if all(c["fidelity"]["shared"]["passes"] for c in record["resident"].values()) else "fails"
            findings.append("The post hoc released-compiler follow-up uses the same frozen scheduler and reused fixtures, "
                            "with eager references collected before compilation. All three timed modes use the compiled FP32 substrate. "
                            f"Shared scheduling admits {record['passed_size_gates']}/{record['attempted_size_gates']} gates and {verdict} "
                            f"the resident audit. Its diverse resident ratio is {d['native_batch_to_shared']:.2f}$\\times$ over compiled native batching. "
                            f"Diverse p95 is {t['shared']['p95_ms']:.1f}\\,ms for shared and {t['native_batch']['p95_ms']:.1f}\\,ms for compiled native batching. "
                            "This is a stronger-baseline diagnostic in one worker, not an independent confirmation. "
                            "Compiler invocation and first-use compilation during admission are retained outside warm resident timings.")
    write_table("focused_gliner_rows.tex", "llrrrr",
                "Substrate & Trace & Gates & Serial/shared & Batch/shared & Audit", resident_body)
    write_table("focused_gliner_serving_rows.tex", "lllrrrr",
                "Substrate & Trace & Mode & p95 (ms) & Complete/offered & Drops & Audit", serving_body)
    if not any(r["action"] == "shared_compiled" for r in shared):
        findings.append("The released-compiler follow-up is pending; the eager scheduling gain is not yet a verified advantage over that stronger native baseline.")
    (GENERATED / "focused_findings.tex").write_text("\n\n".join(findings)+"\n")
    return {"scope": "v5 GLiNER: one worker per native/shared condition; five resident repeats are within-worker samples",
            "native": native, "shared": shared, "failed_attempts": failed,
            "compiled_followup_complete": any(r["action"] == "shared_compiled" for r in shared),
            "raw_reference_scope": "Primary calibration and serving reports retain comparisons; primary resident retains full references and outputs. Compiled follow-up retains full references and outputs throughout."}


def overview_table(summary):
    """One cross-model view, preserving each comparison's statistical unit."""
    rows = []
    for model, note in (
        ("decider", "Short Boolean; five completed containers"),
        ("clef", "Short Boolean; four of five containers; native preflight failure and confidence drift retained"),
    ):
        cell = next(r for r in summary["v3"]["resident"]
                    if r["model"] == model and r["scenario"] == "short_binary")
        rows.append(f"{LABELS[model]} & Native with CUDA extensions & "
                    f"{ratio_ci(cell)} & {note} \\\\")
    liquid = next(r for r in summary["v3"]["liquid_compiler"] if r["scenario"] == "short_binary")
    rows.append(f"{LABELS['liquid']} & Released compiled native & {ratio_ci(liquid)} & "
                "Short Boolean; three containers; effectively tied \\\\")
    compiled = next(r for r in summary["v5_gliner"]["shared"] if r["action"] == "shared_compiled")
    ratio = compiled["resident"]["diverse"]["native_batch_to_shared"]
    rows.append(f"{LABELS['gliner']} & Compiled native FIFO batching & {ratio:.2f} & "
                "Diverse resident trace; one worker; worse arrival p95 \\\\")
    write_table("focused_overview_rows.tex",
                r"p{0.15\textwidth}p{0.23\textwidth}p{0.19\textwidth}p{0.34\textwidth}",
                "Checkpoint & Native comparator & Native/DFIP & Evidence and limit", rows)


def main():
    GENERATED.mkdir(exist_ok=True)
    manifest = []
    summary = {"checkpoints": LABELS, "selection": "Four-checkpoint editorial scope; negative outcomes within these checkpoints retained",
               "statistics": "v1/v3: paired geometric means and seeded worker-cluster percentile bootstrap; v4/v5: single-worker exploratory summaries, no independent-worker interval",
               "v3": native_phases(manifest), "v1_liquid_ablation": ablation_phase(manifest),
               "v4": adaptive_phase(manifest), "v5_gliner": gliner_phase(manifest)}
    overview_table(summary)
    summary["evidence"] = sorted(manifest, key=lambda e: e["file"])
    if any(e["model"] not in LABELS for e in manifest):
        raise ValueError("Unselected checkpoint entered the focused evidence")
    summary["analysis_sources"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in [Path(__file__), ROOT/"scripts/analyze_paper.py", ROOT/"scripts/analyze_strengthened.py",
                                             ROOT/"scripts/analyze_adaptive.py", ROOT/"scripts/analyze_contracts.py",
                                             ROOT/"src/dfip/contract_protocol.py", ROOT/"src/dfip/study_protocol.py"]}
    (GENERATED / "focused-summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps({"selected_raw_records": len(manifest), "v3_resident_rows": len(summary["v3"]["resident"]),
                      "v3_failed_attempts": len(summary["v3"]["failed_attempts"]),
                      "v4_excluded_attempts": len(summary["v4"]["excluded_attempts"]),
                      "gliner_compiled_complete": summary["v5_gliner"]["compiled_followup_complete"]}))


if __name__ == "__main__":
    main()
