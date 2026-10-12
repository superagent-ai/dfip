"""GPU verification and paired benchmarks for the packaged deployment paths."""

import modal
from dfip.config import settings
from dfip.models import get_model
from dfip.cloud.images import image, julia_image, sglang_image, volume

app = modal.App(settings().app_name + "-evaluate")


def evaluate(model_name, action, quality, repetitions):
    import torch
    from dfip.runtime.benchmark import distributions, run, scenarios
    from dfip.runtime.validation import deployment_checks, kernel_checks

    model = get_model(model_name)
    if model.alias == "security":
        from dfip.runtime.sglang_runtime import SGLangDecisionEngine

        engine = SGLangDecisionEngine()
        try:
            if action == "validate":
                raise ValueError(
                    "Use the separately documented SGLang ablation to validate Security-One"
                )
            if action == "smoke":
                records = scenarios()[1]["records"]
                return {
                    "model": model.alias,
                    "result": engine.predict(records),
                    "diagnostics": engine.diagnostics(),
                }
            import time
            import numpy as np
            from dfip.runtime.benchmark import dataset, summarize_quality

            rows = []
            for record in dataset()[:quality]:
                rows.extend(distributions([record], engine.predict([record])))
            timings = []
            for workload in scenarios():
                for _ in range(4):
                    engine.predict(workload["records"])
                samples = []
                for _ in range(repetitions):
                    started = time.perf_counter()
                    engine.predict(workload["records"])
                    samples.append((time.perf_counter() - started) * 1000)
                timings.append(
                    {
                        "scenario": workload["name"],
                        "samples_ms": samples,
                        "p50_ms": float(np.median(samples)),
                        "p95_ms": float(np.percentile(samples, 95)),
                    }
                )
            return {
                "model_id": model.model_id,
                "revision": model.revision,
                "mode": "native_sglang",
                "quality": summarize_quality(rows),
                "timings": timings,
                "diagnostics": engine.diagnostics(),
                "timing_scope": "Resident in-container HTTP API; excludes cold start and client network",
            }
        finally:
            engine.close()
    if action == "benchmark":
        return run(
            model.model_id,
            "packaged-benchmark",
            quality,
            repetitions,
            ("native", "production"),
        )
    if action == "validate":
        return deployment_checks(model.model_id)
    from dfip.runtime.adapters import load_adapter
    from dfip.runtime.engine import DecisionEngine

    adapter = load_adapter(model.model_id)
    records = scenarios()[1]["records"]
    expected = distributions(records, adapter.predict(records))
    engine = DecisionEngine(model.model_id, adapter=adapter)
    results = []
    for _ in range(5):
        results.append(engine.predict(records))
    actual = distributions(records, results[-1])
    drift = max(
        abs(a["probabilities"][key] - b["probabilities"][key])
        for a, b in zip(actual, expected, strict=True)
        for key in a["probabilities"]
    )
    flips = sum(
        a["prediction"] != b["prediction"]
        for a, b in zip(actual, expected, strict=True)
    )
    tolerance = 0.001 + 1e-9 if model.alias == "clef" else 1e-6
    assert flips == 0 and drift <= tolerance, (flips, drift)
    result = {
        "model": model.alias,
        "gpu": torch.cuda.get_device_name(),
        "argmax_flips": flips,
        "max_probability_drift": drift,
        "result": results[-1],
        "diagnostics": engine.diagnostics(),
    }
    if model.alias == "liquid":
        result["kernels"] = kernel_checks()
    return result


def wrapped(model_name, action, quality, repetitions):
    volume.reload()
    try:
        return evaluate(model_name, action, quality, repetitions)
    finally:
        volume.commit()


@app.function(
    image=julia_image,
    gpu="L40S",
    cpu=4,
    memory=32768,
    volumes={"/research": volume},
    timeout=7200,
)
def julia(model, action, quality, repetitions):
    return wrapped(model, action, quality, repetitions)


@app.function(
    image=image,
    gpu="L40S",
    cpu=4,
    memory=32768,
    volumes={"/research": volume},
    timeout=7200,
)
def small(model, action, quality, repetitions):
    return wrapped(model, action, quality, repetitions)


@app.function(
    image=image,
    gpu="B200",
    cpu=8,
    memory=98304,
    volumes={"/research": volume},
    timeout=7200,
)
def decider(model, action, quality, repetitions):
    return wrapped(model, action, quality, repetitions)


@app.function(
    image=sglang_image,
    gpu="B200",
    cpu=8,
    memory=98304,
    volumes={"/research": volume},
    timeout=7200,
)
def security(model, action, quality, repetitions):
    return wrapped(model, action, quality, repetitions)


@app.local_entrypoint()
def main(
    model: str = "liquid",
    action: str = "smoke",
    output: str = "smoke.json",
    quality: int = 400,
    repetitions: int = 100,
):
    import json
    from pathlib import Path

    if action not in ("smoke", "benchmark", "validate"):
        raise ValueError("Action must be smoke, benchmark, or validate")
    spec = get_model(model)
    fn = (
        julia
        if spec.alias == "julia"
        else security
        if spec.alias == "security"
        else decider
        if spec.alias == "decider"
        else small
    )
    result = fn.remote(spec.alias, action, quality, repetitions)
    Path(output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
