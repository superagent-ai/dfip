"""GPU correctness checks for kernels, graph replay, and dynamic answer coverage."""

import json
import torch
from dfip.runtime.kernels import candidate_projection, rms_norm
from dfip.runtime.optimizations import CandidateHead, GraphRunner


def kernel_checks():
    torch.manual_seed(20261012)
    results = []
    for m, n, k in [
        (1, 2, 127),
        (1, 42, 2048),
        (3, 65, 768),
        (8, 255, 5120),
        (17, 31, 257),
    ]:
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        actual = candidate_projection(x, w)
        expected = torch.nn.functional.linear(x, w)
        golden = torch.nn.functional.linear(x.float(), w.float()).to(x.dtype)
        torch.testing.assert_close(actual, golden, rtol=0.01, atol=0.02)
        results.append(
            {
                "shape": [m, n, k],
                "triton_vs_fp32_reference_max_error": float(
                    (actual - golden).abs().max()
                ),
                "native_bf16_vs_fp32_reference_max_error": float(
                    (expected - golden).abs().max()
                ),
                "triton_vs_native_bf16_max_error": float(
                    (actual - expected).abs().max()
                ),
            }
        )
    x = torch.randn(3, 2048, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(2048, device="cuda", dtype=torch.bfloat16)
    norm = []
    for offset, rounding in [(0.0, True), (1.0, False)]:
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        if rounding:
            y = y.to(x.dtype).float()
        expected = (y * (w.float() + offset)).to(x.dtype)
        actual = rms_norm(x, w, 1e-6, offset=offset, round_before_weight=rounding)
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.04)
        norm.append(
            {
                "offset": offset,
                "round_before_weight": rounding,
                "max_absolute_error": float((actual - expected).abs().max()),
            }
        )
    graph = GraphRunner(lambda value: (value.square() + 3 * value).sum(-1))
    for i in range(12):
        value = torch.randn(8, 127, device="cuda")
        expected = (value.square() + 3 * value).sum(-1)
        actual = graph(value)
        torch.testing.assert_close(actual, expected)
    assert graph.captures == 1 and graph.replays == 12
    admitted = GraphRunner(lambda value: (value.square() + 3 * value).sum(-1))
    admitted.capture_min_hits = 3
    for i in range(12):
        value = torch.randn(8, 127, device="cuda")
        torch.testing.assert_close(
            admitted(value), (value.square() + 3 * value).sum(-1)
        )
        if i < 2:
            assert admitted.captures == 0
    assert admitted.captures == 1 and admitted.replays == 10
    churn = GraphRunner(lambda value: (value.square() + 3 * value).sum(-1))
    for i in range(60):
        value = torch.randn(3, (63, 127, 257, 129, 65, 193)[i % 6], device="cuda")
        torch.testing.assert_close(churn(value), (value.square() + 3 * value).sum(-1))
        assert len(churn.cache) <= 4
        if i == 29:
            churn.clear()
    original = torch.nn.Linear(
        127, 300, bias=False, device="cuda", dtype=torch.bfloat16
    )
    head = CandidateHead(original, [1, 7], True)
    assert head.ensure_tokens([1, 7, 99]) and not head.ensure_tokens([99])
    expected = original(value[:, :127].to(torch.bfloat16))
    torch.testing.assert_close(
        head(value.to(torch.bfloat16))[:, head.ids],
        expected[:, head.ids],
        rtol=0.02,
        atol=0.02,
    )
    return {
        "candidate_projection": results,
        "norm_experiments": norm,
        "graph_changed_values": {"captures": graph.captures, "replays": graph.replays},
        "dynamic_head_coverage": "passed",
        "graph_eviction_and_clear": {"calls": 60, "cache_bound": 4, "status": "passed"},
        "delayed_graph_admission": {
            "captures": admitted.captures,
            "replays": admitted.replays,
        },
    }


def model_checks(model_id):
    from dfip.runtime.adapters import load_adapter
    from dfip.runtime.benchmark import distributions, scenarios
    from dfip.runtime.optimization import optimize

    adapter = load_adapter(model_id)
    records = []
    for i in range(12):
        r = {
            "id": f"replay-{i}",
            "state": (
                "Please refund my duplicate charge."
                if i % 2
                else "I cannot log in; please fix the error."
            )
            + f" Ticket {i}.",
            "questions": scenarios()[1]["records"][0]["questions"],
        }
        records.append(r)
    # Add a schema that expands the initial candidate union.
    if model_id == "LiquidAI/d1-3B":
        records.append(
            {
                "id": "expanded-coverage",
                "state": "Please refund a duplicate charge.",
                "questions": {
                    "team": {
                        "type": "choice",
                        "instructions": "Which team?",
                        "criteria": {
                            f"group_{i}": ("Billing refunds" if i == 0 else f"Team {i}")
                            for i in range(64)
                        },
                    }
                },
            }
        )
    native = [distributions([r], adapter.predict([r])) for r in records]
    optimize(
        adapter,
        "dfip" if model_id == "LiquidAI/d1-3B" else "native_graphs",
        records[:1],
    )
    adapter.enable_graphs(True)
    differences = []
    for r, reference in zip(records, native):
        actual = distributions([r], adapter.predict([r]))
        dif = max(
            abs(a["probabilities"][k] - b["probabilities"][k])
            for a, b in zip(actual, reference)
            for k in a["probabilities"]
        )
        flips = sum(
            a["prediction"] != b["prediction"] for a, b in zip(actual, reference)
        )
        assert dif <= 1e-6 and flips == 0, (model_id, r["id"], dif, flips)
        differences.append(dif)
    return {
        "model_id": model_id,
        "records": len(records),
        "argmax_flips": 0,
        "max_probability_drift": max(differences),
        "graph_stats": adapter.graph_stats(),
    }


def deployment_checks(model_id):
    from dfip.runtime.adapters import load_adapter
    from dfip.runtime.engine import DecisionEngine
    from dfip.runtime.benchmark import dataset, distributions, summarize_quality

    adapter = load_adapter(model_id)
    records = dataset()
    native = []
    for r in records:
        native.extend(distributions([r], adapter.predict([r])))
    from dfip.runtime.benchmark import scenarios

    single = []
    for i in range(128):
        state = (
            "The user requests a duplicate payment refund. "
            if i % 2
            else "The user reports a software error, without any payment issue. "
        ) * (i % 7 + 1) + f" Ticket {i}."
        single.append(
            {
                "id": f"single-{i}",
                "state": state,
                "questions": scenarios()[1]["records"][0]["questions"],
            }
        )
    single_native = []
    for r in single:
        single_native.extend(distributions([r], adapter.predict([r])))
    engine = DecisionEngine(model_id, adapter=adapter)
    actual = []
    for i, r in enumerate(records):
        actual.extend(distributions([r], engine.predict([r])))
        if (i + 1) % 50 == 0:
            print(
                json.dumps({"deployment_validation": model_id, "done": i + 1}),
                flush=True,
            )
    quality = summarize_quality(actual, native)
    assert quality["argmax_flips"] == 0 and quality["max_probability_drift"] <= 1e-6, (
        quality
    )
    if model_id == "perplexity-ai/pplx-decider-v1.1-27b":
        assert sum(g.captures for g in adapter.graphs) == 0, (
            "Multi-question graph policy was not enforced"
        )
    single_actual = []
    for r in single:
        single_actual.extend(distributions([r], engine.predict([r])))
    single_quality = summarize_quality(single_actual, single_native)
    tolerance = 1e-3 + 1e-9 if model_id == "Cloudflare/clef-flash" else 1e-6
    assert (
        single_quality["argmax_flips"] == 0
        and single_quality["max_probability_drift"] <= tolerance
    ), single_quality
    single_quality["probability_tolerance"] = tolerance
    return {
        "model_id": model_id,
        "quality": quality,
        "single_question_stress": single_quality,
        "diagnostics": engine.diagnostics(),
    }
