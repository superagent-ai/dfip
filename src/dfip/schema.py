"""Shared request validation without any GPU dependencies."""

import json


def validate_records(records):
    if not isinstance(records, list) or not records:
        raise ValueError("Provide a nonempty JSON list of records")
    for record in records:
        if not isinstance(record, dict) or "state" not in record:
            raise ValueError("Each record needs a state field")
        if record.get("images") or record.get("videos"):
            raise ValueError("DFIP 0.1 supports text decisions only")
        questions = record.get("questions")
        if not isinstance(questions, dict) or not questions:
            raise ValueError("Each record needs a nonempty questions mapping")
        for name, question in questions.items():
            if not isinstance(name, str) or not isinstance(question, dict):
                raise ValueError(
                    "Question names must be strings and questions must be objects"
                )
            kind = question.get("type")
            criteria = question.get("criteria")
            if kind not in ("choice", "noul", "score"):
                raise ValueError("Question type must be choice, noul, or score")
            if not isinstance(question.get("instructions", ""), str):
                raise ValueError("Question instructions must be text")
            if kind == "choice":
                if not isinstance(criteria, dict) or not criteria:
                    raise ValueError(
                        "Choice criteria must be a nonempty label-to-description mapping"
                    )
                if any(
                    not isinstance(k, str) or not isinstance(v, str)
                    for k, v in criteria.items()
                ):
                    raise ValueError("Choice labels and descriptions must be strings")
            if kind == "score" and (
                not isinstance(criteria, list)
                or not criteria
                or any(not isinstance(v, str) for v in criteria)
            ):
                raise ValueError(
                    "Score criteria must be a nonempty ordered list of descriptions"
                )
            if kind == "noul" and criteria is not None:
                if (
                    not isinstance(criteria, dict)
                    or set(criteria) != {"false", "true"}
                    or any(not isinstance(v, str) for v in criteria.values())
                ):
                    raise ValueError("Noul criteria must describe false and true")
    try:
        json.dumps(records, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Records must contain JSON-serializable values, without NaN or infinity"
        ) from exc
    return records
