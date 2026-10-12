import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('adaptive_analysis', Path(__file__).parents[1]/'scripts/analyze_adaptive.py')
analysis = importlib.util.module_from_spec(spec); spec.loader.exec_module(analysis)


def test_profiles_cannot_be_pooled_or_drop_a_retained_partial_mode():
    diverse = {'profile': 'diverse', 'samples': []}
    mixed = {'profile': 'mixed', 'samples': []}
    r = {'model': 'decider', 'modes': {'native': {'phases': [diverse, mixed]},
                                      'adaptive': {'phases': [diverse]}}}
    selected = analysis.profile_record(r, 'mixed')
    assert selected['modes']['native']['phases'] == [mixed]
    assert selected['modes']['adaptive']['phases'] == []
    assert r['modes']['native']['phases'] == [diverse, mixed]


def test_raw_capture_budget_violation_rejected_before_paper_generation():
    import pytest
    report = {'finish': {'diagnostics': {'graphs': [
        {'captures': 5, 'resident_entries': 4, 'admission': {'capture_attempts': 5, 'max_capture_attempts': 4}}
    ]}}}
    with pytest.raises(ValueError, match='invariant'):
        analysis.validate_budget(report)


def trial():
    graph = {'captures': 0, 'resident_entries': 0,
             'admission': {'capture_attempts': 0, 'max_capture_attempts': 4}}
    modes = {}
    for mode in ('native', 'adaptive'):
        modes[mode] = {'setup': {'environment': {'container_id': 'worker'}},
                       'phases': [], 'finish': {'mode': mode, 'container_id': 'worker',
                                                'diagnostics': {'graphs': [graph]}}}
    return {'model': 'clef', 'status': 'complete', 'modes': modes}


def test_complete_client_status_does_not_hide_lost_final_mode():
    import pytest
    record = trial()
    record['modes']['adaptive']['finish']['mode'] = 'native'
    with pytest.raises(ValueError, match='final audit'):
        analysis.validate_trial(record)


def test_final_container_identity_checked_even_with_no_traffic_samples():
    import pytest
    record = trial()
    record['modes']['adaptive']['finish']['container_id'] = 'replacement'
    with pytest.raises(ValueError, match='final audit'):
        analysis.validate_trial(record)
