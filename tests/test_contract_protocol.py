import pytest

from dfip.contract_protocol import compare, conformance_records, validate_output


def test_matching_choices_do_not_hide_confidence_routing_changes():
    a = {"answers": {"q": {"choice": "a", "confidence": 0.901}}}
    b = {"answers": {"q": {"choice": "a", "confidence": 0.899}}}
    report = compare(a, b, numeric_tolerance=0.01)
    assert report["literal_changes"] == 0
    assert report["threshold_crossings"] == 1
    assert not report["passes"]


def test_native_entropy_confidence_need_not_equal_winning_probability():
    assert validate_output({"answers": {"q": {"choice": "a", "probabilities": {"a": 0.7, "b": 0.3}, "confidence": 0.1187}}})


def test_native_list_distributions_retain_their_option_order():
    assert validate_output({"answers": {"q": {"choice": "a", "options": ["a", "b"], "probabilities": [0.7, 0.3]}}})
    with pytest.raises(ValueError):
        validate_output({"answers": {"q": {"options": ["a", "b"], "probabilities": [0.7, 0.7]}}})


def test_missing_numeric_fields_cannot_pass_as_exact_outputs():
    report = compare({"confidence": 0.8}, {})
    assert report["numeric_fields"] == 0 and not report["passes"]


def test_conformance_covers_typed_outputs_and_unique_ids():
    rows = conformance_records()
    assert len({r["id"] for r in rows}) == len(rows)
    assert {q["type"] for r in rows for q in r["questions"].values()} == {"choice", "noul", "score"}
    assert {len(q["criteria"]) for r in rows for q in r["questions"].values() if q["type"] == "choice"} == {2,4,8,16}
