"""Paired full-model benchmarks, with raw samples and held-out fidelity checks."""

import collections
import hashlib
import json
import platform
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch

from dfip.runtime.adapters import load_adapter

DATA_REVISION = "c76749ec58bd8c3d2ea706b31c333a9059c38f90"
DATA_SHA256 = "4f294f218ea1da27f3efef936359389c62ea4d3973a41457732990f1d31b647c"


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def dataset():
    import pyarrow.parquet as pq

    path = Path("/research/datasets/typed-decisions.parquet")
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        urllib.request.urlretrieve(
            f"https://huggingface.co/datasets/LocalLLaMA/typed-decisions/resolve/{DATA_REVISION}/all/test-00000-of-00001.parquet",
            path,
        )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == DATA_SHA256
    cases = pq.read_table(path).to_pylist()
    return [
        {
            "id": c["id"],
            "state": json.loads(c["state"]),
            "questions": json.loads(c["questions"]),
            "gold": json.loads(c["gold"]),
        }
        for c in cases
    ]


def scenarios():
    choice = {
        "type": "choice",
        "instructions": "Which team should handle this request?",
        "criteria": {
            "billing": "Payments, invoices and refunds",
            "technical": "Software bugs and errors",
            "security": "Threats or unauthorized access",
        },
    }
    noul = {
        "type": "noul",
        "instructions": "Does the customer request a refund?",
        "criteria": {
            "false": "No refund requested",
            "true": "Refund explicitly requested",
        },
    }
    score = {
        "type": "score",
        "instructions": "How urgent is this?",
        "criteria": ["Can wait", "Needs attention today", "Blocking production now"],
    }
    short = (
        "I was charged twice for the same order. Please refund the duplicate charge."
    )
    log = "2026-10-12 INFO checkout completed normally; no customer impact. "
    cases = []
    for title, state, qs, batch in [
        ("short_binary", short, {"refund": noul}, 1),
        ("short_choice", short, {"team": choice}, 1),
        (
            "short_3_questions",
            short,
            {"team": choice, "refund": noul, "urgency": score},
            1,
        ),
        ("medium_context", log * 80 + "\n" + short, {"team": choice}, 1),
        ("long_context", log * 300 + "\n" + short, {"team": choice}, 1),
        (
            "16_candidates",
            short,
            {
                "team": dict(
                    choice,
                    criteria={
                        f"team_{i}": (
                            "Payments and refunds"
                            if i == 0
                            else f"Group {i} unrelated specialist work"
                        )
                        for i in range(16)
                    },
                )
            },
            1,
        ),
        ("batch_8", short, {"team": choice}, 8),
    ]:
        records = [
            {"state": state + f" Request identifier {i}.", "questions": qs}
            for i in range(batch)
        ]
        cases.append(
            {
                "name": title,
                "records": records,
                "decisions": batch * len(qs),
                "requests": batch,
            }
        )
    return cases


def distributions(records, outputs):
    result = []
    for r, out in zip(records, outputs, strict=True):
        answers = out["answers"]
        for key, q in r["questions"].items():
            a = answers[key]
            kind = q["type"]
            if kind == "noul":
                p = (
                    {"false": 1 - float(a["noul"]), "true": float(a["noul"])}
                    if "noul" in a
                    else a["probabilities"]
                )
            else:
                p = a["probabilities"]
            values = {str(k): float(v) for k, v in p.items()}
            assert all(np.isfinite(v) and 0 <= v <= 1 for v in values.values()), values
            # Clef's published API rounds each probability to four decimals.
            assert abs(sum(values.values()) - 1) < 1e-3, values
            result.append(
                {
                    "id": str(r.get("id", "workload")) + ":" + key,
                    "type": kind,
                    "probabilities": values,
                    "prediction": max(values, key=values.get),
                    "gold": str(r["gold"][key]["label"]) if "gold" in r else None,
                }
            )
    return result


def summarize_quality(rows, reference=None):
    totals = collections.defaultdict(lambda: {"correct": 0, "count": 0})
    brier = []
    confidence = []
    correct = []
    for row in rows:
        ok = int(row["prediction"] == row["gold"])
        p = row["probabilities"]
        totals[row["type"]]["correct"] += ok
        totals[row["type"]]["count"] += 1
        correct.append(ok)
        confidence.append(max(p.values()))
        brier.append(sum((v - float(k == row["gold"])) ** 2 for k, v in p.items()))
    ece = 0.0
    for i in range(15):
        indices = [
            j
            for j, c in enumerate(confidence)
            if i / 15 <= c <= (i + 1) / 15 and (i == 0 or c > i / 15)
        ]
        if indices:
            ece += (
                len(indices)
                / len(rows)
                * abs(
                    np.mean([confidence[j] for j in indices])
                    - np.mean([correct[j] for j in indices])
                )
            )
    report = {
        "decisions": len(rows),
        "accuracy": float(np.mean(correct)),
        "brier": float(np.mean(brier)),
        "ece_15_equal_width": float(ece),
        "by_type": dict(totals),
    }
    if reference:
        assert [r["id"] for r in rows] == [r["id"] for r in reference]
        drift = []
        flips = []
        for a, b in zip(rows, reference):
            assert a["probabilities"].keys() == b["probabilities"].keys()
            drift.extend(
                abs(a["probabilities"][k] - b["probabilities"][k])
                for k in a["probabilities"]
            )
            if a["prediction"] != b["prediction"]:
                flips.append(a["id"])
        report.update(
            {
                "argmax_flips": len(flips),
                "flip_ids": flips,
                "max_probability_drift": max(drift),
                "mean_probability_drift": float(np.mean(drift)),
                "accuracy_change": report["accuracy"]
                - float(np.mean([r["prediction"] == r["gold"] for r in reference])),
            }
        )
    return report


def time_workload(adapter, scenario, repetitions=30):
    records = scenario["records"]
    first = time.perf_counter()
    adapter.predict(records)
    sync()
    first = time.perf_counter() - first
    for _ in range(4):
        adapter.predict(records)
    sync()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(repetitions):
        sync()
        t = time.perf_counter()
        adapter.predict(records)
        sync()
        samples.append((time.perf_counter() - t) * 1000)
    p50 = float(np.median(samples))
    mean = float(np.mean(samples))
    return {
        "scenario": scenario["name"],
        "decisions_per_call": scenario["decisions"],
        "requests_per_call": scenario["requests"],
        "first_call_seconds": first,
        "p50_ms": p50,
        "p95_ms": float(np.percentile(samples, 95)),
        "mean_ms": mean,
        "decisions_per_second": scenario["decisions"] * 1000 / mean,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30
        if torch.cuda.is_available()
        else None,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30
        if torch.cuda.is_available()
        else None,
        "samples_ms": samples,
    }


def run(
    model_id,
    output_name="initial",
    quality_limit=400,
    repetitions=30,
    modes=("native", "dfip"),
):
    import transformers

    if not modes or modes[0] != "native" or modes.count("native") != 1:
        raise ValueError(
            "A paired benchmark must start with exactly one native baseline"
        )
    if "native_compile" in modes and modes != ("native", "native_compile"):
        raise ValueError(
            "Run the compiler baseline separately as native,native_compile"
        )
    if "production" in modes and modes != ("native", "production"):
        raise ValueError("Run the deployment benchmark separately as native,production")
    from dfip.runtime.optimization import optimize

    torch.set_num_threads(4)
    torch.manual_seed(20261012)
    t = time.perf_counter()
    adapter = load_adapter(model_id)
    load_seconds = time.perf_counter() - t
    records = dataset()[:quality_limit]
    workloads = scenarios()
    result = {
        "model_id": model_id,
        "revision": json.loads(
            (
                Path("/research/prepared") / (model_id.replace("/", "__") + ".json")
            ).read_text()
        )["revision"],
        "hardware": torch.cuda.get_device_name()
        if torch.cuda.is_available()
        else "Modal CPU (4 requested cores, 4 PyTorch threads)",
        "torch": str(torch.__version__),
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "cuda": torch.version.cuda,
        "load_seconds": load_seconds,
        "dataset_revision": DATA_REVISION,
        "dataset_sha256": DATA_SHA256,
        "quality_cases": len(records),
        "modes": {},
        "timing_scope": "resident in-container decision API, includes encoding and readout; synchronized GPU completion",
        "timing_repetitions": repetitions,
    }
    folder = Path("/research/results") / output_name
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (model_id.replace("/", "__") + ".json")
    reference = None
    workload_references = {}
    for mode in modes:
        print(
            json.dumps({"model": model_id, "mode": mode, "stage": "starting"}),
            flush=True,
        )
        optimization = (
            optimize(
                adapter, mode, records + [r for w in workloads for r in w["records"]]
            )
            if mode != "native"
            else {}
        )
        rows = []
        start = time.perf_counter()
        for i, r in enumerate(records):
            rows.extend(distributions([r], adapter.predict([r])))
            if (i + 1) % 50 == 0:
                print(
                    json.dumps(
                        {"model": model_id, "mode": mode, "quality_done": i + 1}
                    ),
                    flush=True,
                )
        quality = summarize_quality(rows, reference)
        if reference is None:
            reference = rows
        quality["elapsed_seconds"] = time.perf_counter() - start
        if hasattr(adapter, "enable_graphs"):
            adapter.enable_graphs(True)
            check = []
            for r in records[:8]:
                check.extend(distributions([r], adapter.predict([r])))
            quality["graph_fidelity"] = summarize_quality(
                check, reference[: len(check)]
            )
        (
            folder / (model_id.replace("/", "__") + "-" + mode + "-predictions.json")
        ).write_text(json.dumps(rows))
        timings = []
        for workload in workloads:
            row = time_workload(adapter, workload, repetitions)
            timings.append(row)
            print(
                json.dumps(
                    {
                        "model": model_id,
                        "mode": mode,
                        "scenario": row["scenario"],
                        "p50_ms": row["p50_ms"],
                    }
                ),
                flush=True,
            )
        workload_rows = []
        workload_native = []
        for workload in workloads:
            actual = distributions(
                workload["records"], adapter.predict(workload["records"])
            )
            if workload["name"] not in workload_references:
                workload_references[workload["name"]] = actual
            workload_rows.extend(actual)
            workload_native.extend(workload_references[workload["name"]])
        quality["workload_fidelity"] = summarize_quality(workload_rows, workload_native)
        result["modes"][mode] = {
            "quality": quality,
            "timings": timings,
            "optimization": optimization,
            "graph_stats": adapter.graph_stats()
            if hasattr(adapter, "graph_stats")
            else {},
        }
        path.write_text(json.dumps(result, indent=2))
    return result
