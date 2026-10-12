"""Bracketing native SGLang around randomized answer-head experiments."""

import json
import random
import time

import numpy as np

from dfip.models import get_model
from dfip.runtime.benchmark import dataset, distributions
from dfip.runtime.research import PROTOCOL, compare, manifest, new_records, workloads
from dfip.runtime.sglang_runtime import SGLangDecisionEngine


def run(round_id, repetitions=50, fidelity=False, source_commit="unknown"):
    order = ["triton", "cublas"]
    random.Random(20261012 + round_id).shuffle(order)
    order = ["native_before"] + order + ["native_after"]
    model = get_model("security")
    cases = workloads()
    public = dataset() if fidelity else []
    extra = new_records() if fidelity else []
    result = {
        "protocol": PROTOCOL,
        "source_commit": source_commit,
        "model_id": model.model_id,
        "revision": model.revision,
        "round": round_id,
        "environment": manifest(),
        "mode_order": order,
        "modes": {},
        "timing_scope": "resident in-container SGLang HTTP API; includes encoding, scheduling, readout; excludes server start and client network",
    }
    references = {}
    public_reference = []
    extra_reference = []
    for mode in order:
        engine = SGLangDecisionEngine(
            head_mode=mode if mode in ("triton", "cublas") else "native"
        )
        try:
            if (
                mode in ("triton", "cublas")
                and '"dfip_decision_head": true' not in engine.log_path.read_text()
            ):
                raise RuntimeError("Candidate-head patch did not activate")
            timings = []
            reference = []
            actual = []
            scenario_order = cases.copy()
            random.Random(20261012 + round_id + len(result["modes"])).shuffle(
                scenario_order
            )
            for workload in scenario_order:
                admission = []
                for _ in range(4):
                    started = time.perf_counter()
                    output = engine.predict(workload["records"])
                    admission.append((time.perf_counter() - started) * 1000)
                samples = []
                for _ in range(repetitions):
                    started = time.perf_counter()
                    output = engine.predict(workload["records"])
                    samples.append((time.perf_counter() - started) * 1000)
                row = distributions(workload["records"], output)
                if workload["name"] not in references:
                    references[workload["name"]] = row
                reference.extend(references[workload["name"]])
                actual.extend(row)
                timings.append(
                    {
                        "scenario": workload["name"],
                        "decisions_per_call": workload["decisions"],
                        "samples_ms": samples,
                        "admission_calls_ms": admission,
                        "p50_ms": float(np.median(samples)),
                        "p95_ms": float(np.percentile(samples, 95)),
                        "mean_ms": float(np.mean(samples)),
                    }
                )
            report = {
                "timings": timings,
                "workload_fidelity": compare(reference, actual),
                "diagnostics": engine.diagnostics(),
            }
            if fidelity:
                public_actual = []
                extra_actual = []
                for i, record in enumerate(public):
                    public_actual.extend(
                        distributions([record], engine.predict([record]))
                    )
                    if (i + 1) % 100 == 0:
                        print(
                            json.dumps({"security_mode": mode, "public_done": i + 1}),
                            flush=True,
                        )
                for record in extra:
                    extra_actual.extend(
                        distributions([record], engine.predict([record]))
                    )
                if not public_reference:
                    public_reference = public_actual
                if not extra_reference:
                    extra_reference = extra_actual
                report["public_fidelity"] = compare(public_reference, public_actual)
                report["new_workload_fidelity"] = compare(extra_reference, extra_actual)
                report["new_predictions"] = extra_actual
            result["modes"][mode] = report
            print(
                json.dumps(
                    {
                        "model": "security",
                        "round": round_id,
                        "mode": mode,
                        "short_binary_p50": next(
                            t["p50_ms"]
                            for t in timings
                            if t["scenario"] == "short_binary"
                        ),
                    }
                ),
                flush=True,
            )
        finally:
            engine.close()
    return result
