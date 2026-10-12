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
    from dfip.study_protocol import summarize_phase
    import copy
    phases = []
    for profile in ('diverse', 'mixed'):
        for rate in (5, 15, 30):
            rows = [{'status': 'dropped', 'dispatch_lag_ms': 0, 'finished_seconds': 0}
                    for _ in range(rate*60)]
            phases.append({'profile': profile, 'samples': rows,
                           'summary': summarize_phase(rows, 60, rate)})
    modes = {}
    for mode in ('native', 'adaptive'):
        modes[mode] = {'setup': {'environment': {'container_id': 'worker'}},
                       'phases': copy.deepcopy(phases), 'finish': {'mode': mode, 'container_id': 'worker',
                                                'diagnostics': {'graphs': [graph]}}}
    return {'model': 'clef', 'status': 'complete', 'modes': modes}


def test_complete_client_status_does_not_hide_lost_final_mode():
    import pytest
    record = trial()
    record['modes']['adaptive']['finish']['mode'] = 'native'
    with pytest.raises(ValueError, match='final audit'):
        analysis.validate_trial(record)


def test_final_container_identity_checked_even_with_no_successful_samples():
    import pytest
    record = trial()
    record['modes']['adaptive']['finish']['container_id'] = 'replacement'
    with pytest.raises(ValueError, match='final audit'):
        analysis.validate_trial(record)


def test_raw_budget_metadata_cannot_expand_the_prespecified_limit():
    import pytest
    report = {'finish': {'diagnostics': {'graphs': [
        {'captures': 5, 'resident_entries': 5, 'admission': {'capture_attempts': 5, 'max_capture_attempts': 5}}
    ]}}}
    with pytest.raises(ValueError, match='invariant'):
        analysis.validate_budget(report)


def test_response_mode_checked_even_when_worker_identity_is_unchanged():
    import pytest
    from dfip.study_protocol import summarize_phase
    record = trial()
    phase = record['modes']['adaptive']['phases'][0]
    phase['samples'][0] = {'status': 'ok', 'latency_ms': 100, 'dispatch_lag_ms': 0,
                           'finished_seconds': 1, 'service_ms': 50, 'queue_ms': 0,
                           'container_id': 'worker', 'worker_mode': 'native'}
    phase['summary'] = summarize_phase(phase['samples'], 60, 5)
    with pytest.raises(ValueError, match='configured mode'):
        analysis.validate_trial(record)


def test_complete_trial_cannot_silently_omit_a_planned_phase():
    import pytest
    record = trial()
    record['modes']['adaptive']['phases'].pop()
    with pytest.raises(ValueError, match='six planned traffic phases'):
        analysis.validate_trial(record)
