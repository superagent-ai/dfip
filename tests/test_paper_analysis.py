import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location(
    "paper_analysis", Path(__file__).parents[1] / "scripts/analyze_paper.py"
)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_paired_constant_speedup_has_no_artificial_sampling_variance():
    assert analysis.interval([2.0] * 5) == pytest.approx((2, 2, 2))


def test_bracketed_native_reference_is_geometric():
    def mode(milliseconds):
        return {"timings": [{"scenario": scenario, "samples_ms": [milliseconds] * 50,
                            "p50_ms": milliseconds, "decisions_per_call": 1}
                           for scenario in analysis.SCENARIOS]}
    result = {"modes": {"native_before": mode(10), "native_after": mode(40), "triton": mode(10)}}
    rows = analysis.analysis({"security": [result]})
    assert rows[0]["native_p50_ms_geomean"] == pytest.approx(20)
    assert rows[0]["speedup"] == pytest.approx(2)
    assert rows[0]["rounds"] == 1


def test_stale_hand_entered_median_cannot_enter_paper_table():
    result = {"modes": {"native": {"timings": [{"scenario": "short_binary",
                                               "samples_ms": [10] * 50, "p50_ms": 9}]}}}
    with pytest.raises(ValueError, match="median"):
        analysis.timings(result, "native")


@pytest.mark.parametrize("ratios", [[], [0], [-1], [float("inf")], [float("nan")]])
def test_invalid_speed_ratios_are_rejected(ratios):
    with pytest.raises(ValueError):
        analysis.interval(ratios)
