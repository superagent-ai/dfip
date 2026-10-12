"""CPU-only diagnostics and explicit fidelity accounting for research adapters."""
import math


def record(index, kind="choice", width=4, repeats=1):
    labels = ["billing", "technical", "returns", "sales", "fraud", "account", "shipping", "other",
              "legal", "privacy", "marketing", "finance", "support", "operations", "product", "security"][:width]
    if kind == "choice":
        criteria = {k: k.replace("_", " ") for k in labels}
        instructions = "Which team should handle this customer request?"
    elif kind == "noul":
        criteria = {"false": "No", "true": "Yes"}
        instructions = "Is the customer asking for a refund?"
    else:
        criteria = [str(i) for i in range(6)]
        instructions = "Rate the urgency from 0 (routine) to 5 (immediate action)."
    state = f"Ticket {index}. " + ("I was charged twice for a damaged order. Please refund the duplicate payment. " * repeats)
    return {"id": f"contract-{index}", "state": state,
            "questions": {"decision": {"type": kind, "instructions": instructions, "criteria": criteria}}}


def conformance_records():
    rows = [record(i, kind, width, repeats) for i, (kind, width, repeats) in enumerate(
        (k, w, r) for k in ("choice", "noul", "score") for w in ((2, 4, 8, 16) if k == "choice" else (2,)) for r in (1, 7)
    )]
    for i in range(12, 28):
        rows.append(record(i, "choice", (2, 4, 8, 16)[i % 4], (1, 3, 7, 15)[i % 4]))
    combined = record(28)
    combined["questions"] = {k: record(28, k)["questions"]["decision"] for k in ("choice", "noul", "score")}
    rows.append(combined)
    return rows


def compare(reference, actual, numeric_tolerance=0.001, score_tolerance=0.01):
    report = {"numeric_fields": 0, "literal_changes": 0, "structure_changes": 0,
              "max_numeric_drift": 0.0, "max_score_drift": 0.0, "threshold_crossings": 0,
              "probability_threshold_crossings": 0,
              "out_of_tolerance": 0}

    def walk(a, b, path=()):
        if isinstance(a, dict):
            if not isinstance(b, dict) or list(a) != list(b):
                report["structure_changes"] += 1
                return
            for k in a:
                if k not in ("elapsed_seconds", "num_model_requests", "model"):
                    walk(a[k], b[k], path + (k,))
        elif isinstance(a, (tuple, list)):
            if not isinstance(b, (tuple, list)) or len(a) != len(b):
                report["structure_changes"] += 1
                return
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, path + (i,))
        elif isinstance(a, (int, float)) and not isinstance(a, bool):
            if not isinstance(b, (int, float)) or not math.isfinite(a) or not math.isfinite(b):
                report["structure_changes"] += 1
                return
            drift = abs(a - b); score = path and path[-1] == "score"
            report["numeric_fields"] += 1
            field = "max_score_drift" if score else "max_numeric_drift"
            report[field] = max(report[field], drift)
            report["out_of_tolerance"] += drift > (score_tolerance if score else numeric_tolerance)
            if path and path[-1] in ("confidence", "noul"):
                report["threshold_crossings"] += sum((a >= t) != (b >= t) for t in (0.1, 0.25, 0.5, 0.75, 0.9))
            if "probabilities" in path:
                report["probability_threshold_crossings"] += sum((a >= t) != (b >= t) for t in (0.1, 0.25, 0.5, 0.75, 0.9))
        elif a != b:
            report["literal_changes"] += 1

    walk(reference, actual)
    report["passes"] = not any(report[k] for k in ("literal_changes", "structure_changes", "threshold_crossings", "probability_threshold_crossings", "out_of_tolerance"))
    return report


def validate_output(output):
    """Validate exposed fields; absence of a distribution is not a distribution audit."""
    if not isinstance(output, dict) or not isinstance(output.get("answers"), dict) or not output["answers"]:
        raise ValueError("Missing named decision answers")
    for answer in output["answers"].values():
        if not isinstance(answer, dict) or "error" in answer:
            raise ValueError("Missing or failed decision")
        for field in ("confidence", "noul"):
            if field in answer and not (math.isfinite(answer[field]) and 0 <= answer[field] <= 1):
                raise ValueError("Invalid probability or confidence")
        probabilities = answer.get("probabilities")
        if probabilities is not None:
            values = list(probabilities.values()) if isinstance(probabilities, dict) else list(probabilities)
            if not values or any(not math.isfinite(v) or v < 0 or v > 1 for v in values) or abs(sum(values)-1) > 0.0001:
                raise ValueError("Invalid categorical distribution")
            options = probabilities if isinstance(probabilities, dict) else answer.get("options", [])
            if "choice" in answer and answer["choice"] not in options:
                raise ValueError("Choice is not an option")
        if "score" in answer and not math.isfinite(answer["score"]):
            raise ValueError("Invalid score")
    return True
