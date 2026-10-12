"""Frozen paper protocol: randomized ablations, independent rounds, and fidelity."""

import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch

from dfip.models import get_model
from dfip.runtime.adapters import load_adapter
from dfip.runtime.benchmark import (
    dataset,
    distributions,
    scenarios,
    summarize_quality,
    sync,
)
from dfip.runtime.optimization import optimize

BANKING_REVISION = "8eed2b348ec07d40ffbe91dcc4718e2f0f977714"
BANKING_SHA256 = "535fc96c4c2b4c2dbdeb0d4b31f24a4859620cf663d6c19c1bd1f97450c410be"
PROTOCOL = "dfip-paper-v2-synchronized-retirement"


def manifest():
    packages = {}
    for name in (
        "torch",
        "transformers",
        "triton",
        "fla-core",
        "causal-conv1d",
        "numpy",
        "sglang",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "packages": packages,
        "gpu": gpu,
        "cuda": torch.version.cuda,
        "python": __import__("sys").version,
        "container_id": os.environ.get("MODAL_TASK_ID"),
        "cpu_threads": torch.get_num_threads(),
    }


def banking_records(count=256):
    """New diagnostic workloads, not a claim of unseen checkpoint training data."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        "PolyAI/banking77",
        "data/test-00000-of-00001.parquet",
        repo_type="dataset",
        revision=BANKING_REVISION,
    )
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == BANKING_SHA256
    table = pq.read_table(path)
    metadata = json.loads(table.schema.metadata[b"huggingface"])
    names = metadata["info"]["features"]["label"]["names"]
    rows = table.to_pylist()
    rng = random.Random(20261012)
    indices = rng.sample(range(len(rows)), count)
    result = []
    for j, index in enumerate(indices):
        row = rows[index]
        label = int(row["label"])
        width = (2, 4, 8, 16)[j % 4]
        candidates = rng.sample(
            [i for i in range(len(names)) if i != label], width - 1
        ) + [label]
        rng.shuffle(candidates)
        target = (
            label
            if j % 2 == 0
            else rng.choice([i for i in range(len(names)) if i != label])
        )
        description = lambda i: names[i].replace("_", " ")
        questions = {
            "intent": {
                "type": "choice",
                "instructions": "Which banking intent best describes the customer utterance?",
                "criteria": {names[i]: description(i) for i in candidates},
            },
            "matches": {
                "type": "noul",
                "instructions": "Does this customer utterance express the intent: "
                + description(target)
                + "?",
                "criteria": {
                    "false": "The utterance expresses another intent",
                    "true": "The utterance expresses this intent",
                },
            },
            "strength": {
                "type": "score",
                "instructions": "How directly does the utterance express the intent: "
                + description(target)
                + "?",
                "criteria": [
                    "Unrelated intent",
                    "Related topic but another intent",
                    "Direct expression of the intent",
                ],
            },
        }
        result.append(
            {
                "id": f"banking77-test-{index}",
                "state": row["text"],
                "questions": questions,
                "gold": {
                    "intent": {"label": names[label]},
                    "matches": {"label": "true" if target == label else "false"},
                    "strength": {"label": "2" if target == label else "0"},
                },
                "source_index": index,
            }
        )
    return result


def new_records():
    # Single-question forms exercise graph eligibility, three-question forms batching.
    records = []
    for row in banking_records():
        one = copy.deepcopy(row)
        one["id"] += "-choice"
        one["questions"] = {"intent": one["questions"]["intent"]}
        one["gold"] = {"intent": one["gold"]["intent"]}
        records.append(one)
        records.append(row)
    return records


def workloads():
    result = scenarios()
    template = copy.deepcopy(result[1])
    for width in (2, 8):
        row = copy.deepcopy(template)
        row["name"] = f"{width}_candidates"
        row["records"][0]["questions"]["team"]["criteria"] = {
            f"team_{i}": "Billing and refunds"
            if i == 0
            else f"Unrelated department {i}"
            for i in range(width)
        }
        result.append(row)
    return result


def compare(reference, actual):
    assert len(reference) == len(actual) and [r["id"] for r in reference] == [
        r["id"] for r in actual
    ]
    drift = []
    flips = []
    thresholds = {str(t): 0 for t in (0.1, 0.25, 0.5, 0.75, 0.9)}
    scores = []
    for expected, observed in zip(reference, actual, strict=True):
        assert expected["probabilities"].keys() == observed["probabilities"].keys()
        if expected["prediction"] != observed["prediction"]:
            flips.append(observed["id"])
        for key, p in expected["probabilities"].items():
            q = observed["probabilities"][key]
            drift.append(abs(p - q))
            for threshold in thresholds:
                thresholds[threshold] += int(
                    (p >= float(threshold)) != (q >= float(threshold))
                )
        if expected["type"] == "score":
            scores.append(
                abs(
                    sum(int(k) * v for k, v in expected["probabilities"].items())
                    - sum(int(k) * v for k, v in observed["probabilities"].items())
                )
            )
    return {
        "decisions": len(actual),
        "argmax_flips": len(flips),
        "flip_ids": flips,
        "max_probability_drift": max(drift, default=0.0),
        "mean_probability_drift": float(np.mean(drift)) if drift else 0.0,
        "threshold_crossings": thresholds,
        "max_expected_score_drift": max(scores, default=0.0),
    }


class AblationEngine:
    """Switch policies without mixing capture buffers or changing model weights."""

    def __init__(self, model_id):
        self.adapter = load_adapter(model_id)
        self.native_predict = self.adapter.predict
        self.original_head = getattr(self.adapter.model, "lm_head", None)
        self.original_mask = (
            self.adapter.model.encoder._update_attention_mask
            if model_id == "SupersonicLabs/Julia-1"
            else None
        )
        self.report = optimize(self.adapter, "production", [])
        self.guarded_predict = self.adapter.predict
        self.candidate_head = getattr(self.adapter.model, "lm_head", None)
        self.graph_mask = (
            self.adapter.model.encoder._update_attention_mask
            if self.original_mask
            else None
        )
        self.mode = None

    def configure(self, mode):
        self.mode = mode
        for graph in self.adapter.graphs:
            graph.clear()
            graph.failures.clear()
            graph.shape_hits.clear()
            graph.captures = 0
            graph.replays = 0
        if self.original_mask:
            self.adapter.model.encoder._update_attention_mask = (
                self.original_mask if mode == "native" else self.graph_mask
            )
        uses_head = (
            mode
            in (
                "production",
                "candidate_triton",
                "candidate_cublas",
                "production_cublas",
            )
            and self.original_head is not None
            and hasattr(self.candidate_head, "ensure_tokens")
        )
        if self.original_head is not None:
            self.adapter.model.lm_head = (
                self.candidate_head if uses_head else self.original_head
            )
        if uses_head:
            self.candidate_head.use_triton = (
                mode != "candidate_cublas" and mode != "production_cublas"
            )
        self.predict_fn = self.guarded_predict if uses_head else self.native_predict
        self.adapter.enable_graphs(
            mode in ("production", "graphs_only", "production_cublas")
        )

    def predict(self, records):
        return self.predict_fn(records)

    def diagnostics(self):
        return {
            "mode": self.mode,
            "graphs": self.adapter.graph_stats(),
            "selected_candidate_tokens": len(self.candidate_head.ids)
            if hasattr(self.candidate_head, "ids")
            else None,
        }


def measure(engine, workload, repetitions):
    print(json.dumps({"benchmark_start": workload["name"], "mode": engine.mode}), flush=True)
    first = []
    for _ in range(4):
        started = time.perf_counter()
        engine.predict(workload["records"])
        sync()
        first.append((time.perf_counter() - started) * 1000)
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(repetitions):
        started = time.perf_counter()
        engine.predict(workload["records"])
        sync()
        samples.append((time.perf_counter() - started) * 1000)
    return {
        "scenario": workload["name"],
        "decisions_per_call": workload["decisions"],
        "samples_ms": samples,
        "admission_calls_ms": first,
        "p50_ms": float(np.median(samples)),
        "p95_ms": float(np.percentile(samples, 95)),
        "mean_ms": float(np.mean(samples)),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "diagnostics": engine.diagnostics(),
    }


def round_run(
    model_name, round_id, repetitions=50, fidelity=False, source_commit="unknown",
    progress=None,
):
    model = get_model(model_name)
    torch.set_num_threads(8 if model.gpu == "B200" else 4)
    torch.manual_seed(20261012 + round_id)
    started = time.perf_counter()
    engine = AblationEngine(model.model_id)
    sync()
    load_seconds = time.perf_counter() - started
    cases = workloads()
    modes = ["native", "production"]
    if model.alias == "liquid":
        modes += [
            "graphs_only",
            "candidate_cublas",
            "candidate_triton",
            "production_cublas",
        ]
    rng = random.Random(20261012 + round_id)
    rng.shuffle(modes)
    result = {
        "protocol": PROTOCOL,
        "source_commit": source_commit,
        "model_id": model.model_id,
        "revision": model.revision,
        "round": round_id,
        "environment": manifest(),
        "load_seconds": load_seconds,
        "mode_order": modes,
        "timing_scope": "resident full API; synchronized GPU; excludes client network and model loading",
        "modes": {},
    }
    if progress:
        progress(result)
    engine.configure("native")
    references = {
        w["name"]: distributions(w["records"], engine.predict(w["records"]))
        for w in cases
    }
    public = dataset() if fidelity else []
    extra = new_records() if fidelity else []
    public_reference = []
    extra_reference = []
    for record in public:
        public_reference.extend(distributions([record], engine.predict([record])))
    for record in extra:
        extra_reference.extend(distributions([record], engine.predict([record])))
    if fidelity:
        result["diagnostic_manifest"] = {
            "banking_revision": BANKING_REVISION,
            "banking_sha256": BANKING_SHA256,
            "records": len(extra),
            "records_sha256": hashlib.sha256(
                json.dumps(extra, sort_keys=True).encode()
            ).hexdigest(),
            "notes": "New workload for DFIP development. Checkpoint training overlap unknown. Derived score labels are diagnostic, not a validated annotation.",
        }
    for mode in modes:
        engine.configure(mode)
        order = cases.copy()
        rng.shuffle(order)
        timings = [measure(engine, w, repetitions) for w in order]
        actual = []
        reference = []
        for workload in cases:
            actual.extend(
                distributions(workload["records"], engine.predict(workload["records"]))
            )
            reference.extend(references[workload["name"]])
        report = {"timings": timings, "workload_fidelity": compare(reference, actual)}
        if fidelity:
            public_actual = []
            extra_actual = []
            for i, record in enumerate(public):
                public_actual.extend(distributions([record], engine.predict([record])))
                if (i + 1) % 100 == 0:
                    print(
                        json.dumps(
                            {"model": model.alias, "mode": mode, "public_done": i + 1}
                        ),
                        flush=True,
                    )
            for record in extra:
                extra_actual.extend(distributions([record], engine.predict([record])))
            report["public_fidelity"] = compare(public_reference, public_actual)
            report["new_workload_fidelity"] = compare(extra_reference, extra_actual)
            report["public_quality"] = summarize_quality(
                public_actual, public_reference
            )
            # Choice and Boolean labels derive from the dataset; score remains fidelity only.
            labelled = [r for r in extra_actual if r["type"] != "score"]
            labelled_ref = [r for r in extra_reference if r["type"] != "score"]
            report["new_workload_diagnostic_quality"] = summarize_quality(
                labelled, labelled_ref
            )
            report["new_predictions"] = extra_actual
        result["modes"][mode] = report
        if progress:
            progress(result)
        print(
            json.dumps(
                {
                    "model": model.alias,
                    "round": round_id,
                    "mode": mode,
                    "short_binary_p50": next(
                        t["p50_ms"] for t in timings if t["scenario"] == "short_binary"
                    ),
                }
            ),
            flush=True,
        )
    # Cold admission vs one-off shapes: same model, empty shape caches; no result cache.
    engine.configure("production")
    stream = []
    for i in range(120):
        record = copy.deepcopy(cases[1]["records"][0])
        record["state"] = (
            ("Unrelated audit context. " * (i % 40 + 1))
            + (
                "Please refund the duplicate charge."
                if i % 2 == 0
                else "My application crashed and I cannot sign in."
            )
            + f" Reference {i}."
        )
        stream.append(record)
    engine.configure("native")
    reference = [distributions([r], engine.predict([r])) for r in stream]
    engine.configure("production")
    times = []
    observed = []
    for r in stream:
        started = time.perf_counter()
        output = engine.predict([r])
        sync()
        times.append((time.perf_counter() - started) * 1000)
        observed.extend(distributions([r], output))
    result["changing_shapes"] = {
        "samples_ms": times,
        "p50_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "fidelity": compare([x for row in reference for x in row], observed),
        "diagnostics": engine.diagnostics(),
    }
    return result
